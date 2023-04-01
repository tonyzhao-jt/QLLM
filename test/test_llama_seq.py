from qllm.models import llama
import torch 
import torch.nn as nn 
from time import perf_counter
def load_llama_7b():
    ava_model_sizes = llama.get_available_models()
    print(ava_model_sizes)
    llama_7b, tokenizer = llama.get_empty_model('7b')
    print("model_loaded")
    return llama_7b, tokenizer
llama_7b, tokenizer = load_llama_7b()

llama_7b.config.use_cache = False # not use cache to store the pevious computed result
# model structures
layers = llama_7b.model.layers
embed_tokens = llama_7b.model.embed_tokens
norm = llama_7b.model.norm
model_hidden_size = llama_7b.config.hidden_size
# the model dtype
dtype = next(iter(llama_7b.parameters())).dtype 

# create samples
seq_length = 2048 
nsamples = 2

dev="cuda:1"

inps = torch.ones(
        (nsamples, seq_length, model_hidden_size), dtype=dtype, device=dev
)
cache = {'i': 0, 'attention_mask': None} # manually set cache
outs = torch.zeros_like(inps)
attention_mask = cache['attention_mask']

with torch.no_grad():
    # We only test sequential execution but not others
    layer_length = len(layers)
    for idx in range(layer_length):
        layer = layers[idx]
        layer = layer.to(dev)
        outs = layer(inps, attention_mask=attention_mask)[0]
        layers[idx] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

output1 = inps
# output2
new_inputs = torch.ones(
        (nsamples, seq_length, model_hidden_size), dtype=dtype, device=dev
)

llama_7b = llama_7b.to(dev)
torch.cuda.synchronize()
start = perf_counter()
with torch.no_grad():
    output2 = nn.Sequential(*layers)(new_inputs)
torch.cuda.synchronize()
end = perf_counter()
torch.cuda.empty_cache()
print(output2 - output1)
print(f"sequential execution time (CUDA): {end-start}")

# cpu time
llama_7b = llama_7b.to('cpu')
new_inputs = new_inputs.to('cpu')
start = perf_counter()
with torch.no_grad():
    output2 = nn.Sequential(*layers)(new_inputs)
end = perf_counter()
print(f"sequential execution time (CPU): {end-start}")