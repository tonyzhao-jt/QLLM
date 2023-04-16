import torch, transformers
from accelerate import init_empty_weights, dispatch_model
import torch.nn as nn
import os 
import psutil
from qllm.models.OPT import OPTForCausalLMSeq, OPTDecoderSeq
from qllm.models import opt
cur_mem = 0
# get cpu mem
virtual_memory = psutil.virtual_memory()
cur_mem = virtual_memory.used
print("cpu mem", cur_mem)
size = '13b'
config = opt.model_cards[size]
# weight_loaded_model = OPTForCausalLMSeq._from_config(config, torch_dtype=torch.float16)
# decoder = weight_loaded_model.model.decoder
# with init_empty_weights():
#     weight_loaded_model = OPTForCausalLMSeq._from_config(config, torch_dtype=torch.float16)
#     decoder = weight_loaded_model.model.decoder
os.environ['SET_DECODERS_META'] = "1"
weight_loaded_model = OPTForCausalLMSeq._from_config(config, torch_dtype=torch.float16)
decoder = weight_loaded_model.model.decoder

for i in decoder.layers:
    for name, param in i.named_parameters():
        if 'weight' in name:
            print(param.is_meta, end='')
            break

# get cpu mem
memory_used = psutil.virtual_memory().used
print("used memory: ", (memory_used - cur_mem) / 1024 /1024) 

decoder._shard_decoders({
    0: {'shard': [0, 1], 'bits': [16, 16]},
    1: {'shard': [0, 1], 'bits': [16, 16]},
    2: {'shard': [0, 1], 'bits': [16, 16]},
    3: {'shard': [0, 1], 'bits': [16, 16]},
})

for i in decoder.layers:
    if i is not None:
        for name, param in i.named_parameters():
            if 'weight' in name:
                print(param.is_meta, end='')
                break
    else:
        print("None", end='')

# get cpu mem
memory_used = psutil.virtual_memory().used
print("used memory: ", (memory_used - cur_mem) / 1024 /1024) 
cur_mem = memory_used

exit()
print("New model")
import os
os.environ['SET_DECODERS_META'] = "1"
new_loaded_model = OPTForCausalLMSeq._from_config(config, torch_dtype=torch.float16)
for i in new_loaded_model.model.decoder.layers:
    for name, param in i.named_parameters():
        if 'weight' in name:
            print(param.is_meta, end='')
            break
new_loaded_model.model.decoder._shard_decoders({
    0: {'shard': [0, 1], 'bits': [16, 16]},
    1: {'shard': [0, 1], 'bits': [16, 16]},
    2: {'shard': [0, 1], 'bits': [16, 16]},
})

for i in new_loaded_model.model.decoder.layers:
    if i is not None:
        for name, param in i.named_parameters():
            if 'weight' in name:
                print(param.is_meta, end='')
                break
    else:
        print("None", end='')
memory_used = psutil.virtual_memory().used
print("used memory: ", memory_used - cur_mem)
cur_mem = memory_used
# # linear
# virtual_memory = psutil.virtual_memory()
# memory_used = virtual_memory.used
# print("used: ", memory_used)

# # li = nn.Linear(10000, 10000)
# with init_empty_weights():
#     li = nn.Linear(10000, 10000)

# # print the cpu ram usage
# virtual_memory = psutil.virtual_memory()
# memory_used = virtual_memory.used
# print("used: ", memory_used)

# a = torch.randn(3, 3)
# li = li.to("cpu")
# li(a)