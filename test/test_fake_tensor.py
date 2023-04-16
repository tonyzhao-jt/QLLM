import torch, transformers
from accelerate import init_empty_weights, dispatch_model
import torch.nn as nn
import psutil
from qllm.models.OPT import OPTForCausalLMSeq, OPTDecoderSeq
with init_empty_weights():
    weight_loaded_model = OPTForCausalLMSeq.from_pretrained("facebook/opt-125m", torch_dtype=torch.float16)
    decoder = OPTDecoderSeq(weight_loaded_model.config)
print(weight_loaded_model.config)
print(decoder.config)

for i in decoder.layers:
    for name, param in i.named_parameters():
        if 'weight' in name:
            print(param.is_meta)
            break

decoder._shard_decoders({
    0: {'shard': [0, 1], 'bits': [16, 16]},
    1: {'shard': [0, 1], 'bits': [16, 16]},
    2: {'shard': [0, 1], 'bits': [16, 16]},
})

for i in decoder.layers:
    if i is not None:
        for name, param in i.named_parameters():
            if 'weight' in name:
                print(param.is_meta)
                break
    else:
        print("None")

print("New model")
import os
os.environ['SET_DECODERS_META'] = "1"
new_loaded_model = OPTForCausalLMSeq.from_pretrained("facebook/opt-125m", torch_dtype=torch.float16)
for i in new_loaded_model.model.decoder.layers:
    for name, param in i.named_parameters():
        if 'weight' in name:
            print(param.is_meta)
            break
new_loaded_model.model.decoder._shard_decoders({
    0: {'shard': [0, 1], 'bits': [16, 16]},
    1: {'shard': [0, 1], 'bits': [16, 16]},
    2: {'shard': [0, 1], 'bits': [16, 16]},
})
print("updated")
for i in new_loaded_model.model.decoder.layers:
    if i is not None:
        for name, param in i.named_parameters():
            if 'weight' in name:
                print(param.is_meta)
                break
    else:
        print("None")

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