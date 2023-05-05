import os 
import qllm
import torch.distributed as dist
import torch
def register_tp_group(group_ranks=[]):
    # check whether local world size in env
    if "LOCAL_WORLD_SIZE" not in os.environ:
        return None
    local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
    if local_world_size == 1:
        return None
    # by default, use all local devices
    if len(group_ranks) == 0:
        group_ranks = list(range(0, local_world_size))
    if not dist.is_initialized():
        # Initialize the distributed environment
        dist.init_process_group(backend='nccl')
    comm_group = dist.new_group(group_ranks, backend='nccl')
    qllm._globals.__TENSOR__MODEL_PARALLEL__GROUP__ = comm_group
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    if rank not in group_ranks:
        return None
    index = group_ranks.index(rank)
    small_world_size = len(group_ranks)
    return rank, index, small_world_size

def all_barrier():
    if not dist.is_initialized():
        return None
    dist.barrier()