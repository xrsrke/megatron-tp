import os

import pytest
import torch
# import torch.nn.init as init
from torch.nn.parameter import Parameter

from megatron.testing import dist_init, set_random_seed, spawn_for_all_world_sizes
import megatron.initialize as mpu
import megatron.layers as layers
from megatron.pipegoose_utils import spawn


class IdentityLayer2D(torch.nn.Module):
    def __init__(self, m, n):
        super(IdentityLayer2D, self).__init__()
        self.weight = Parameter(torch.Tensor(m, n))
        torch.nn.init.xavier_normal_(self.weight)

    def forward(self):
        return self.weight


# def run_test_column_parallel_linear(rank, model_parallel_size, filename, filename_rpc):
def run_test_column_parallel_linear(rank, world_size, port, model_parallel_size):
    # NOTE: neuralink removed filename and filename_rpc
    # dist_init(rank, model_parallel_size, filename, filename_rpc)
    dist_init(rank, model_parallel_size, port)

    mpu.initialize_model_parallel(model_parallel_size)
    if torch.distributed.get_rank() == 0:
        print("> testing ColumnParallelLinear with model parallel size: {}".format(model_parallel_size))
    model_parallel_size = mpu.get_model_parallel_world_size()

    seed = 12345
    set_random_seed(seed)
    input_size_coeff = 13
    input_size = input_size_coeff * model_parallel_size
    output_size_coeff = 17
    output_size = output_size_coeff * model_parallel_size
    batch_size = 7

    # Network
    # identity_layer = IdentityLayer2D(batch_size, input_size).cuda()
    identity_layer = IdentityLayer2D(batch_size, input_size)
    # linear_layer = layers.ColumnParallelLinear(input_size, output_size, keep_master_weight_for_test=True).cuda()
    linear_layer = layers.ColumnParallelLinear(input_size, output_size, keep_master_weight_for_test=True)
    # loss_weight = torch.randn([batch_size, output_size]).cuda()
    loss_weight = torch.randn([batch_size, output_size])
    # Forward
    input_ = identity_layer()
    output = linear_layer(input_)
    loss = torch.mul(output, loss_weight).sum()
    # Backward
    loss.backward()

    # Values.
    dLdY = loss_weight
    X = identity_layer.weight
    # A = linear_layer.master_weight.cuda()
    A = linear_layer.master_weight
    dLdA = torch.matmul(dLdY.t(), X)
    # dLdb = torch.matmul(torch.ones(batch_size, 1).cuda().t(), dLdY).view(-1)
    dLdb = torch.matmul(torch.ones(batch_size, 1).t(), dLdY).view(-1)
    dLdX = torch.matmul(dLdY, A)

    rank = mpu.get_model_parallel_rank()
    my_dLdA = torch.split(dLdA, output_size_coeff, dim=0)[rank].contiguous().clone()
    error = my_dLdA.sub(linear_layer.weight.grad).abs().max()
    torch.distributed.barrier()
    print("   error in dLdA on global rank {}: {}".format(torch.distributed.get_rank(), error))
    assert error < 1.0e-6, error

    my_dLdb = torch.split(dLdb, output_size_coeff, dim=0)[rank].contiguous().clone()
    error = my_dLdb.sub(linear_layer.bias.grad).abs().max()
    torch.distributed.barrier()
    print("   error in dLdb on global rank {}: {}".format(torch.distributed.get_rank(), error))
    assert error < 1.0e-6, error

    error = dLdX.sub(identity_layer.weight.grad).abs().max()
    torch.distributed.barrier()
    print("   error in dLdX on global rank {}: {}".format(torch.distributed.get_rank(), error))
    assert error < 1.0e-6, error

    # Reset groups
    mpu.destroy_model_parallel()

    torch.distributed.barrier()
    if torch.distributed.get_rank() == 0:
        print(" >> passed the test :-)")


if __name__ == "__main__":
    # spawn_for_all_world_sizes(run_test_column_parallel_linear, deterministic=True)

    TENSOR_PARALLEL_SIZE = 2

    spawn(
        run_test_column_parallel_linear,
        world_size=TENSOR_PARALLEL_SIZE,
        model_parallel_size=TENSOR_PARALLEL_SIZE
    )
