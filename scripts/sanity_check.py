import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from megatron.pipegoose_utils import read_bin

class NN(nn.Module):
    def __init__(self, input_size, output_size):
        super(NN, self).__init__()
        self.debug_single_mlp = nn.Linear(input_size, output_size)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.debug_single_mlp(x)
        return x

torch.set_printoptions(precision=15)

# Load batch of data
debug_batch = torch.load("debug_batch.pt")
debug_target = torch.load("debug_target.pt")

model = NN(input_size=32 * 32, output_size=10)
ref_model = deepcopy(model)
criterion = nn.CrossEntropyLoss()

torch.cuda.empty_cache()
model.to("cuda")
device = next(model.parameters()).device
ref_model.to(device)

inputs, labels = debug_batch.to(device), debug_target.to(device)

EPOCH = 18

# Model reference
ref_model.debug_single_mlp.weight.data = torch.tensor(read_bin(f"state/epoch{EPOCH}/ref_weight.bin")).to(device, torch.float32)
ref_model.debug_single_mlp.bias.data = torch.tensor(read_bin(f"state/epoch{EPOCH}/ref_bias.bin")).to(device, torch.float32)

ref_outputs = ref_model(inputs)
ref_loss = criterion(ref_outputs, labels)

ref_loss_loaded = torch.tensor(read_bin(f"state/epoch{EPOCH}/ref_loss.bin")).to(device, torch.float32)

try:
    torch.testing.assert_close(ref_loss, ref_loss_loaded[0], atol=1e-15, rtol=1e-15, msg=lambda msg: f"\n{msg}",)
except AssertionError as e:
    print("====== REFERENCE =======")
    print(e)
    print(f"ref_loss: {ref_loss}")
    print(f"ref_loss_loaded: {ref_loss_loaded[0]}")
    pass

# Model parallel
input_parallel = torch.tensor(read_bin(f"state/epoch{EPOCH}/input_parallel_flatten.bin")).to(device, torch.float32)

try:
    torch.testing.assert_close(torch.flatten(inputs, 1), input_parallel, atol=1e-15, rtol=1e-15, msg=lambda msg: f"\n{msg}")
except AssertionError as e:
    print("====== INPUT PARALELL =======")
    print(e)
    pass

weight_0 = torch.tensor(read_bin(f"state/epoch{EPOCH}/weight_0.bin")).to(device, torch.float32)
weight_1 = torch.tensor(read_bin(f"state/epoch{EPOCH}/weight_1.bin")).to(device, torch.float32)
bias_0 = torch.tensor(read_bin(f"state/epoch{EPOCH}/bias_0.bin")).to(device, torch.float32)
bias_1 = torch.tensor(read_bin(f"state/epoch{EPOCH}/bias_1.bin")).to(device, torch.float32)

output_0 = F.linear(input_parallel, weight_0, bias_0)
output_1 = F.linear(input_parallel, weight_1, bias_1)

# output_parallel = torch.cat([output_0, output_1], dim=1)
output_parallel = torch.zeros_like(ref_outputs)
output_parallel[:, :5] = output_0
output_parallel[:, 5:] = output_1

output_cat = torch.tensor(read_bin(f"state/epoch{EPOCH}/output_cat.bin")).to(device, torch.float32)

try:
    torch.testing.assert_close(output_parallel, ref_outputs, msg=lambda msg: f"\n{msg}", atol=1e-15, rtol=1e-15)
except AssertionError as e:
    print("====== OUTPUT PARALELL =======")
    print(e)
    pass

try:
    torch.testing.assert_close(output_parallel, output_cat, msg=lambda msg: f"\n{msg}", atol=1e-15, rtol=1e-15)
except AssertionError as e:
    print("====== OUTPUT PARALELL vs OUTPUT CAT =======")
    print(e)
    pass


loss = criterion(output_parallel, labels)
loss_loaded = torch.tensor(read_bin(f"state/epoch{EPOCH}/loss.bin")).to(device, torch.float32)

try:
    torch.testing.assert_close(loss, loss_loaded[0], atol=1e-15, rtol=1e-15, msg=lambda msg: f"\n{msg}")
except AssertionError as e:
    print("====== LOSS =======")
    print(e)
    print(f"loss: {loss}")
    print(f"loss_loaded: {loss_loaded[0]}")
    pass

try:
    torch.testing.assert_close(loss, ref_loss_loaded[0], atol=1e-15, rtol=1e-15, msg=lambda msg: f"\n{msg}")
except AssertionError as e:
    print("====== LOSS vs MULTI-GPU loss  =======")
    print(e)
    print(f"loss: {loss}")
    print(f"ref_loss_loaded: {ref_loss_loaded[0]}")
    pass