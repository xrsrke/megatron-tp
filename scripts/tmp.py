import torch
import torch.nn as nn
import megatron.layers as layers
import megatron.initialize as mpu
from copy import deepcopy
from megatron.pipegoose_utils import spawn
from megatron.testing import dist_init, set_random_seed
from megatron.logger import Logger
from megatron.utils import divide_and_check_no_remainder

class NN(nn.Module):
    def __init__(self, input_size, output_size):
        super(NN, self).__init__()
        self.debug_single_mlp = nn.Linear(input_size, output_size)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.debug_single_mlp(x)
        return x

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
    linear_layer = layers.ColumnParallelLinear(in_features, out_features, keep_master_weight_for_test=True)
    
    # Split weight and bias
    linear_layer.weight.data = get_partition(model.debug_single_mlp.weight.data, dim=0)
    linear_layer.bias.data = get_partition(model.debug_single_mlp.bias.data, dim=0)

    # Assign to model
    model.debug_single_mlp = linear_layer
    
    return model


def func(rank, world_size, port, model_parallel_size):
    dist_init(rank, model_parallel_size, port)

    mpu.initialize_model_parallel(model_parallel_size)
    if torch.distributed.get_rank() == 0:
        print("> testing ColumnParallelLinear with model parallel size: {}".format(model_parallel_size))
    model_parallel_size = mpu.get_model_parallel_world_size()

    Logger()(f"rank = {mpu.get_model_parallel_rank()}")

    seed = 12345
    set_random_seed(seed)

    model = NN(input_size=32 * 32, output_size=10)
    # model_tp = tensor_parallel_pipegoose(deepcopy(model))
    model_tp = tensor_parallel_megatron(deepcopy(model))


if __name__ == "__main__":
    TENSOR_PARALLEL_SIZE = 2

    spawn(
        func,
        world_size=TENSOR_PARALLEL_SIZE,
        model_parallel_size=TENSOR_PARALLEL_SIZE
    )
