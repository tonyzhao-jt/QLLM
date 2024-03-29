# now support to pass tp into the decoder layer.
# make judgement when only ATTEN exists, add FFN split at the top
# now passes a tp config into the decoder layer, too.

# new config {tp_group:{ranks_in_group: [], index: <index>}}
# when tp_group is None, follow default pipeline parallelism
# when index=0, the stage is number is recorded, else the stage will be launched by comm.

# check the availability in running qpipe with tensor-parallel
import torch
import torch.distributed as dist
import os 
import qllm.tp as tp
import qllm.tp.utils as qllm_tp_utils
import qllm
import qllm.nn as qllm_nn
from qllm.models.BLOOM import BloomAttention, BloomForCausalLMSeq
from qllm.models import bloom
from time import perf_counter
from transformers import AutoTokenizer

dist.init_process_group(backend='gloo', init_method='env://')
# Define the group membership
world_size = dist.get_world_size()
rank = dist.get_rank()
local_rank = int(os.environ["LOCAL_RANK"])
local_device = torch.device('cuda', local_rank)
group_ranks = list(range(0, 4))
group_ranks = [2, 3]

slice_k = len(group_ranks)
# set devivce
torch.cuda.set_device(local_rank)

# Create a new process group with the NCCL backend
qllm_tp_utils.register_tp_group(group_ranks)
comm_group = qllm_tp_utils.get_tp_group()
# PS: test ok.

from lptorch import CalibHelper, quantize_one_linear_module, AdaQTPConfig

dist.barrier()
comm_group_size = dist.get_world_size(group=comm_group)
hidden_size = 1536
tensor_size = (3, 128, hidden_size)
dtype = torch.float16
sample_x = torch.rand(tensor_size)
# linear = torch.nn.Linear(hidden_size, hidden_size) # normal kv
linear = torch.nn.Linear(hidden_size, 3 * hidden_size) # fused qkv
# caliber
caliber = CalibHelper(linear)
caliber.register_forward_hooks()
y = linear(sample_x)
caliber.remove_forward_hooks()

# test linear result
if rank in group_ranks:
    index = group_ranks.index(rank)
    if index == 0:
        print("original result shape", y.shape)

if rank == 0:
    print("---- TEST SIMPLE TP CONFIG ----")
if rank in group_ranks:
    index = group_ranks.index(rank)

    col_li = qllm_nn.Linear1D.from_linear(linear, TP_TYPE='COLUMN')
    row_li = qllm_nn.Linear1D.from_linear(linear)
    classify = qllm_nn.Classifier1D.from_classi(linear, broadcast=True)

    # quantize linear
    print("rank", rank, slice_k, index)

    qllm._globals.HIDDEN_STATE_TENSOR_SHAPE = sample_x.shape
    if index == 0:
        result = linear.cuda().half()(sample_x.cuda().half())
        sample_x = sample_x.cuda().half()
    else:
        # still ok
        # sample_x = torch.empty_like(sample_x).cuda()
        # print(qllm._globals.HIDDEN_STATE_TENSOR_SHAPE)
        sample_x = torch.empty(qllm._globals.HIDDEN_STATE_TENSOR_SHAPE).cuda().half()
    

    
    '''
        Classify
    '''
    classify = classify.cuda()
    output = classify(sample_x)
    if index == 0:
        print("Classify output shape", output.shape)
        print(torch.max(torch.abs(output - result)))

    '''
    for col-wise
    '''
    tp._broad_cast(sample_x, rank, index, comm_group)
    col_li = col_li.cuda()
    res = col_li(sample_x)
    output = tp._all_gather(res, index, slice_k, comm_group)
    if index == 0:
        print("COL output shape", output.shape)
        print(torch.max(torch.abs(output - result)))

    '''
    for row-wise
    '''
    sliced_x = tp._scatter_last_dim(sample_x, rank, index, slice_k, comm_group)

    row_li = row_li.cuda()
    res = row_li(sliced_x)
    # print(sliced_x, row_li.inner_layer.weight, index)
    # do all_reduce
    # print(res, "result", index)
    # res = tp._all_reduce_sum(res, comm_group) all reduce is added in the forward function
    dist.barrier(group=comm_group)
    if index == 0:
        print("ROW output shape", res.shape)
        print(torch.max(torch.abs(res - result)))

# also test a sample embedding result
test_emb = torch.nn.Embedding(200, 200)
test_emb = test_emb.to(local_device)
test_emb_input = torch.tensor([1, 2, 4,101,134]).to(local_device)
if rank == group_ranks[0]:
    test_emb_out = test_emb(test_emb_input)
# broadcast weight
rank_0_weight = test_emb.weight.detach().clone()
if rank in group_ranks:
    index = group_ranks.index(rank)
    tp._broad_cast(rank_0_weight, rank, index, comm_group)
    test_emb.weight.data.copy_(rank_0_weight) # change weight data
    local_embed = qllm_nn.Embedding1D.from_embed(test_emb)

    local_embed_out = local_embed(test_emb_input)
    if rank == group_ranks[0]:
        print(torch.max(torch.abs(local_embed_out - test_emb_out)))

exit()
# block test.
dist.barrier()
if rank == 0:
    print("---- TEST BLOOM ----")

model_size = '560m'
config = bloom.model_cards[model_size]
tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom")
# sample text
max_length = 512
# sample text
batched_ids = tokenizer.batch_encode_plus(["Hi, where is my dog. ", "Just test performance. How about you. ", \
                                            "The quick brown fox jumps over the lazy dog. It's a beautiful day outside, the sun is shining and the birds are chirping. I feel like going for a"], \
                                                padding='max_length', max_length=max_length, return_tensors="pt")
weight_loaded_model = BloomForCausalLMSeq._from_config(config, torch_dtype=torch.float16)
weight_loaded_model = weight_loaded_model.cuda()
batched_ids = {k: v.cuda() for k, v in batched_ids.items()}
request_token = weight_loaded_model.preprocess(**batched_ids, use_cache=True, request_id=1)
bloom_attn = weight_loaded_model.transformer.h[0].self_attention 
bloom_mlp = weight_loaded_model.transformer.h[0].mlp

hidden_size = bloom_attn.hidden_size
hidden_states, causal_mask, head_mask, alibi, use_cache, request_id = request_token
cnt_times = 10
caliber.set_model(weight_loaded_model)
caliber.set_fake()
caliber.load_fake_calib_data('./fake_calib_bloom_1b1.pkl')

# test attn
if rank == 0:
    torch.cuda.synchronize()
    bloom_attn_cuda = bloom_attn.cuda()
    # warmup
    for _ in range(cnt_times):
        fake_out = bloom_attn_cuda(hidden_states, residual=hidden_states, alibi=alibi, attention_mask=causal_mask)
        torch.cuda.synchronize()
    start = perf_counter()
    for _ in range(cnt_times):
        fake_out = bloom_attn_cuda(hidden_states, residual=hidden_states, alibi=alibi, attention_mask=causal_mask)
        torch.cuda.synchronize()
    end = perf_counter()
    print("bloom attn time, single", end - start)

dist.barrier()
if rank in group_ranks:
    # start tp the attention
    index = group_ranks.index(rank)
    bit = 16
    # bloom_attn.register_tp(8, slice_k, index, caliber, comm_group)
    bloom_attn.register_tp(bit, caliber)
    bloom_attn_cuda = bloom_attn.cuda()
    dist.barrier(group=comm_group)
    # warmup
    for _ in range(cnt_times):
        out = bloom_attn_cuda(hidden_states, residual=hidden_states, alibi=alibi, attention_mask=causal_mask)
        torch.cuda.synchronize()
    start = perf_counter()
    for _ in range(cnt_times):
        out = bloom_attn_cuda(hidden_states, residual=hidden_states, alibi=alibi, attention_mask=causal_mask)
        torch.cuda.synchronize()
    end = perf_counter()
    if index == 0:
        print(f"bloom attn time, tp, bit {bit}", end - start)

# test ffn
if rank == 0:
    torch.cuda.synchronize()
    bloom_mlp_cuda = bloom_mlp.cuda()
    # warmup
    for _ in range(cnt_times):
        fake_out = bloom_mlp_cuda(hidden_states, hidden_states)
        torch.cuda.synchronize()
    start = perf_counter()
    for _ in range(cnt_times):
        fake_out = bloom_mlp_cuda(hidden_states, hidden_states)
        torch.cuda.synchronize()
    end = perf_counter()
    print("bloom mlp time, single", end - start)

dist.barrier()
if rank in group_ranks:
    # start tp the mlp
    index = group_ranks.index(rank)
    bit = 16
    # bloom_mlp.register_tp(8, slice_k, index, caliber, comm_group)
    bloom_mlp.register_tp(bit, caliber)
    bloom_mlp_cuda = bloom_mlp.cuda()
    dist.barrier(group=comm_group)
    # warmup
    for _ in range(cnt_times):
        out = bloom_mlp_cuda(hidden_states, hidden_states)
        torch.cuda.synchronize()
    start = perf_counter()
    for _ in range(cnt_times):
        out = bloom_mlp_cuda(hidden_states, hidden_states)
        torch.cuda.synchronize()
    end = perf_counter()
    if index == 0:
        print(f"bloom mlp time, tp, bit {bit}", end - start)

