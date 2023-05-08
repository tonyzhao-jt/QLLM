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
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if rank not in group_ranks:
        return None
    index = group_ranks.index(rank)
    small_world_size = len(group_ranks)
    # setup group and index.
    qllm._globals.__TENSOR__MODEL_PARALLEL__GROUP__ = comm_group
    qllm._globals.__TP__GLOBAL__RANK__ = rank
    qllm._globals.__TP__LOCAL_INDEX__ = index
    qllm._globals.__TP__SMALL_WORLD_SIZE__ = small_world_size
    
    dist.barrier(comm_group)
    return rank, index, small_world_size

def register_tp_group_and_update_strategy(group_ranks=[], sharding_strategy={}):
    res = register_tp_group(group_ranks)
    if res is None:
        pass
    else:
        # upate the sharding result
        rank, index, k = res
        if rank not in sharding_strategy:
            pass 
        else:
            rank_cors_shards = sharding_strategy[rank]
            for layer_idx, layer_spec in rank_cors_shards.items():
                layer_spec['tp_config'] =  {"k": k, "index": index} # update the sharding strategy
    return res

def all_barrier():
    if not dist.is_initialized():
        return None
    dist.barrier()

def disable_broadcast():
    qllm._globals.__TP__BROADCAST__GROUP__ = False

def get_tp_group():
    return qllm._globals.__TENSOR__MODEL_PARALLEL__GROUP__

def get_tp_configs():
    config = {
        "group": qllm._globals.__TENSOR__MODEL_PARALLEL__GROUP__,
        "global_rank": qllm._globals.__TP__GLOBAL__RANK__,
        "tp_index": qllm._globals.__TP__LOCAL_INDEX__,
        "split_k": qllm._globals.__TP__SMALL_WORLD_SIZE__,
        'broadcast_group': qllm._globals.__TP__BROADCAST__GROUP__
    }
    return config

def empty_tp_configs():
    qllm._globals.__TENSOR__MODEL_PARALLEL__GROUP__ = None
    qllm._globals.__TP__GLOBAL__RANK__ = None
    qllm._globals.__TP__LOCAL_INDEX__ = None
    qllm._globals.__TP__SMALL_WORLD_SIZE__ = None
    qllm._globals.__TP__BROADCAST__GROUP__ = True

def load_tp_configs(config):
    qllm._globals.__TENSOR__MODEL_PARALLEL__GROUP__ = config['group']
    qllm._globals.__TP__GLOBAL__RANK__ = config['global_rank']
    qllm._globals.__TP__LOCAL_INDEX__ = config['tp_index']
    qllm._globals.__TP__SMALL_WORLD_SIZE__ = config['split_k']
    qllm._globals.__TP__BROADCAST__GROUP__ = config['broadcast_group']