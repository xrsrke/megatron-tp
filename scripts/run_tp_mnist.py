from copy import deepcopy

import torch
import torch.distributed as dist
from torch.optim import SGD
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
from torchvision import datasets, transforms
import megatron.layers as layers
from megatron.logger import Logger
import megatron.initialize as mpu
from megatron.pipegoose_utils import spawn, write_bin, read_bin
from megatron.testing import dist_init, set_random_seed
from megatron.utils import divide_and_check_no_remainder
import wandb

class NN(nn.Module):
    def __init__(self, input_size, output_size):
        super(NN, self).__init__()
        self.debug_single_mlp = nn.Linear(input_size, output_size)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.debug_single_mlp(x)
        return x

class MNISTloader:
    def __init__(
        self,
        batch_size: int = 64,
        data_dir: str = "./data/",
        num_workers: int = 0,
        pin_memory: bool = False,
        shuffle: bool = False,
        train_val_split: float = 0.1,
    ):
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.shuffle = shuffle
        self.train_val_split = train_val_split

        self.setup()

    def setup(self):
        transform = transforms.Compose(
            [
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ]
        )

        self.train_dataset = datasets.MNIST(
            self.data_dir, train=True, download=True, transform=transform
        )
        val_split = int(len(self.train_dataset) * self.train_val_split)
        train_split = len(self.train_dataset) - val_split

        self.train_dataset, self.val_dataset = random_split(
            self.train_dataset, [train_split, val_split]
        )
        self.test_dataset = datasets.MNIST(
            self.data_dir, train=False, download=True, transform=transform
        )

        print(
            "Image Shape:    {}".format(self.train_dataset[0][0].numpy().shape),
            end="\n\n",
        )
        print("Training Set:   {} samples".format(len(self.train_dataset)))
        print("Validation Set: {} samples".format(len(self.val_dataset)))
        print("Test Set:       {} samples".format(len(self.test_dataset)))

    def load(self):
        train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=self.shuffle,
        )

        val_loader = DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=self.shuffle,
        )

        test_loader = DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=self.shuffle,
        )

        return train_loader, val_loader, test_loader

def tensor_parallel_pipegoose(model):

    def get_partition(data: torch.Tensor, dim: int) -> torch.Tensor:
        rank = mpu.get_model_parallel_rank()
        world_size = mpu.get_model_parallel_world_size()
        chunks = torch.chunk(data, world_size, dim=0)
        return chunks[rank].contiguous()

    in_features = model.debug_single_mlp.in_features
    out_features = model.debug_single_mlp.out_features
    linear_layer = layers.ColumnParallelLinear(in_features, out_features, keep_master_weight_for_test=True)
    
    # Split weight and bias
    linear_layer.weight.data = get_partition(model.debug_single_mlp.weight.data, dim=0)
    linear_layer.bias.data = get_partition(model.debug_single_mlp.bias.data, dim=0)

    # Assign to model
    model.debug_single_mlp = linear_layer
    
    return model

def tensor_parallel_megatron(model):

    def get_partition(data: torch.Tensor, dim: int) -> torch.Tensor:
        world_size = mpu.get_model_parallel_world_size()
        per_partition_size = divide_and_check_no_remainder(out_features, world_size)
        stride = 1

        per_partition_per_stride_size = divide_and_check_no_remainder(per_partition_size, stride)
        weight_list = torch.split(data, per_partition_per_stride_size, dim=dim)
        rank = mpu.get_model_parallel_rank()
        my_weight_list = weight_list[rank::world_size]
        return torch.cat(my_weight_list, dim=dim)

    in_features = model.debug_single_mlp.in_features
    out_features = model.debug_single_mlp.out_features
    linear_layer = layers.ColumnParallelLinear(in_features, out_features, keep_master_weight_for_test=False)
    
    # Split weight and bias
    linear_layer.weight.data = get_partition(model.debug_single_mlp.weight.data, dim=0)
    linear_layer.bias.data = get_partition(model.debug_single_mlp.bias.data, dim=0)

    # Assign to model
    model.debug_single_mlp = linear_layer
    
    return model

def save_grad(state, epoch, param_name):
    # TODO: maybe register_hook backward: https://pytorch.org/docs/stable/notes/autograd.html#backward-hooks-execution
    state[f"epoch_{epoch}"] = {}
    def hook(grad):
        state[f"epoch_{epoch}"][param_name] = grad
    return hook


def run_column_parallel(rank, world_size, port, model_parallel_size):
    NUM_EPOCHS = 60
    LR = 2e-1
    SEED = 12345

    dist_init(rank, model_parallel_size, port)

    mpu.initialize_model_parallel(model_parallel_size)
    if torch.distributed.get_rank() == 0:
        print("> testing ColumnParallelLinear with model parallel size: {}".format(model_parallel_size))
    model_parallel_size = mpu.get_model_parallel_world_size()

    Logger()(f"rank = {mpu.get_model_parallel_rank()}")

    set_random_seed(SEED)

    # Load batch of data
    debug_batch = torch.load("debug_batch.pt")
    debug_target = torch.load("debug_target.pt")

    model = NN(input_size=32 * 32, output_size=10)
    model.load_state_dict(torch.load("model.pt"))
    ref_model = deepcopy(model)

    dist.barrier()

    model = tensor_parallel_pipegoose(model)
    # model = tensor_parallel_megatron(model)
    optim = SGD(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    torch.cuda.empty_cache()
    model.to("cuda")
    device = next(model.parameters()).device

    Logger()(f"rank={rank}, model is moved to device: {device}")

    ref_model.to(device)
    Logger()(f"rank={rank}, ref_model is moved to device: {device}")
    ref_optim = SGD(ref_model.parameters(), lr=LR)
    ref_criterion = nn.CrossEntropyLoss()

    model.train()
    ref_model.train()
    dist.barrier()


    if rank == 0:
        state_ref_0 = {}
        state_0 = {}
    elif rank == 1:
        state_ref_1 = {}
        state_1 = {}

    dist.barrier()


    if rank == 0:
        def get_time_name():
            import datetime

            today = datetime.datetime.now()
            return today.strftime("%d/%m/%Y_%H:%M:%S")

        wandb.init(
            project="pipegoose",
            name=f"{get_time_name()}.test_tp_mnist_converegence",
            config={
                "tensor_parallel_size": world_size,
                "model": "NN",
                "dataset": "MNIST",
                "epochs": NUM_EPOCHS,
                "learning_rate": LR,
                "seed": SEED,
            },
        )

    for epoch in range(NUM_EPOCHS):
    
        inputs, labels = debug_batch.to(device), debug_target.to(device)

        ref_outputs = ref_model(inputs)
        ref_loss = ref_criterion(ref_outputs, labels)
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)


        if rank == 0:
            ref_model.debug_single_mlp.weight.register_hook(save_grad(state_ref_0, epoch, "weight"))
            ref_model.debug_single_mlp.bias.register_hook(save_grad(state_ref_0, epoch, "bias"))
            model.debug_single_mlp.weight.register_hook(save_grad(state_0, epoch, "weight"))
            model.debug_single_mlp.bias.register_hook(save_grad(state_0, epoch, "bias"))
        elif rank == 1:
            ref_model.debug_single_mlp.weight.register_hook(save_grad(state_ref_1, epoch, "weight"))
            ref_model.debug_single_mlp.bias.register_hook(save_grad(state_ref_1, epoch, "bias"))
            model.debug_single_mlp.weight.register_hook(save_grad(state_1, epoch, "weight"))
            model.debug_single_mlp.bias.register_hook(save_grad(state_1, epoch, "bias"))

        dist.barrier()

        ref_optim.zero_grad()
        ref_loss.backward()
        ref_optim.step()

        optim.zero_grad()
        loss.backward()
        optim.step()
        
        if rank == 0:
            print(f"epoch={epoch}, rank={rank}, train_loss={loss}, ref_train_loss={ref_loss}")

            wandb.log(
                {
                    "train_loss": loss,
                    "ref_train_loss": ref_loss,
                    "epoch": epoch,
                }
            )


        dist.barrier()
        # clear hook
        ref_model.debug_single_mlp.weight._backward_hooks.clear()
        ref_model.debug_single_mlp.bias._backward_hooks.clear()
        model.debug_single_mlp.weight._backward_hooks.clear()
        model.debug_single_mlp.bias._backward_hooks.clear()

        dist.barrier()

    dist.barrier()


    if rank == 0:
        torch.save(state_ref_0, "state_ref_0.pt")
        torch.save(state_0, "state_0.pt")
    elif rank == 1:
        torch.save(state_ref_1, "state_ref_1.pt")
        torch.save(state_1, "state_1.pt")

    dist.barrier()                

    wandb.finish()
    model.cpu()

if __name__ == "__main__":
    TENSOR_PARALLEL_SIZE = 2

    spawn(
        run_column_parallel,
        world_size=TENSOR_PARALLEL_SIZE,
        model_parallel_size=TENSOR_PARALLEL_SIZE
    )