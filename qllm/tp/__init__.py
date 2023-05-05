# import dist
import torch.distributed as dist
import torch 

from . import utils

def partition_a_into_b_bins(a, b):
    remainders = a % b
    ideal_allocation = a // b
    allocation = []
    for i in range(b):
        allocation.append(ideal_allocation) 
    for i in range(remainders):
        allocation[i] += 1 
    return allocation

def ensure_divisibility(numerator, denominator):
    """Ensure that numerator is divisible by the denominator."""
    assert numerator % denominator == 0, "{} is not divisible by {}".format(
        numerator, denominator
    )


def divide(numerator, denominator):
    """Ensure that numerator is divisible by the denominator and return
    the division value."""
    ensure_divisibility(numerator, denominator)
    return numerator // denominator

'''
    for col-wise
'''
def _broad_cast(tensor, group):
    dist.broadcast(tensor, src=0, group=group)
    return tensor

def _all_gather(tensor, index, split_k, group):
    gather_tensor_list = [torch.empty_like(tensor) for _ in range(split_k)]
    # all gather the result
    dist.all_gather(gather_tensor_list, tensor, group=group)
    output = torch.cat(gather_tensor_list, dim=-1).contiguous()
    return output

def _all_gather_last_dim(tensor, index, split_k, group):
    gather_tensor_list = [torch.empty_like(tensor) for _ in range(split_k)]
    # all gather the result
    dist.all_gather(gather_tensor_list, tensor, group=group)
    output = torch.cat(gather_tensor_list, dim=-1).contiguous()
    return output

'''
    for row-wise
'''
def _scatter_last_dim(tensor, index, split_k, group):
    tensor_size = tensor.size()
    last_dim = tensor.dim() - 1
    last_dim_size = divide(tensor.size()[last_dim], split_k)
    if index == 0:
        tensor_list = torch.split(tensor, last_dim_size, dim=last_dim)
        tensor_list = [chunk.contiguous() for chunk in tensor_list]
    else:
        tensor_list = None
    sliced_tensor_size = (*tensor_size[:-1], last_dim_size)
    dtype = tensor.dtype
    sliced_x = torch.empty(sliced_tensor_size, dtype=dtype, device=tensor.device)
    dist.scatter(sliced_x, tensor_list, src=0, group=group)
    return sliced_x

def _scatter_first_dim(tensor, index, split_k, group):
    tensor_size = tensor.size()
    first_dim = 0
    first_dim_size = divide(tensor.size()[first_dim], split_k)
    if index == 0:
        tensor_list = torch.split(tensor, first_dim_size, dim=first_dim)
        tensor_list = [chunk.contiguous() for chunk in tensor_list]
    else:
        tensor_list = None
    sliced_tensor_size = (first_dim_size, *tensor_size[1:])
    dtype = tensor.dtype
    sliced_x = torch.empty(sliced_tensor_size, dtype=dtype, device=tensor.device)
    dist.scatter(sliced_x, tensor_list, src=0, group=group)
    return sliced_x

def _all_reduce_sum(tensor, group):
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
    return tensor

def _all_reduce_mean(tensor, split_k, group):
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
    tensor /= split_k
    return tensor