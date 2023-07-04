# generation reference for the bloom and opt
from qllm.utils.argparser import model_config_argparser
from qllm.models import bare_load_pretrained_from_size
from qllm.utils import batch_encode_plus
import torch 
import time 
import logging

# test greedy
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    LogitsProcessorList,
    MinLengthLogitsProcessor,
    StoppingCriteriaList,
    MaxLengthCriteria,
)

args = model_config_argparser()
model_name = args.model_name
model_size = args.model_size
model, tokenizer, key = bare_load_pretrained_from_size(model_name, model_size)
print(f"model_name: {model_name}, model_size: {model_size}")
# prompt refer to https://github.com/huggingface/transformers-bloom-inference/blob/main/bloom-inference-scripts/bloom-accelerate-inference.py
prompts = [
    "He is working on",
    "He has a",
    "Everyone is happy and I can",
]
num_tokens = 10
max_prompt_length = 20
generate_kwargs = dict(max_new_tokens=num_tokens, do_sample=False)
eos_token = tokenizer.eos_token  
# REFERECNE implementation for batch encode plus
# hf didn't support batch encode plus
# def batch_encode_plus(tokenizer, prompts, return_tensors=None, max_length=None):
#     # hf didn't provide the left padding for batched tokens
#     # do manual padding
#     interested_keys = ['input_ids', 'attention_mask', 'token_type_ids']
#     sample_out = []
#     for prompt in prompts:
#         token_i = tokenizer(prompt, return_tensors="pt")
#         for k in token_i.keys():
#             if k in interested_keys:
#                 if k == 'input_ids':
#                     # padd left, add 1
#                     token_i[k] = torch.cat([torch.ones(max_length - token_i[k].shape[1], dtype=torch.long), token_i[k][0]])
#                 elif k == 'attention_mask':
#                     # padd left, add 0
#                     token_i[k] = torch.cat([torch.zeros(max_length - token_i[k].shape[1], dtype=torch.long), token_i[k][0]])
#                 # if the token_i dim is 1, then use view to make it 1,dim
#                 if len(token_i[k].shape) == 1:
#                     token_i[k] = token_i[k].view(1, -1)
#         sample_out.append(token_i)
#     # merge along the batch dimension
#     batched_out = {}
#     for sample in sample_out:
#         for key in sample:
#             if key not in batched_out:
#                 batched_out[key] = []
#             batched_out[key].append(sample[key])
#     for key in batched_out:
#         batched_out[key] = torch.cat(batched_out[key], dim=0)
#     return batched_out


# input_tokens = batch_encode_plus(tokenizer, prompts, return_tensors="pt", max_length=max_prompt_length)
# import pdb; pdb.set_trace()

def generate():
    """returns a list of zipped inputs, outputs and number of new tokens"""

    input_tokens = batch_encode_plus(tokenizer, prompts, return_tensors="pt", max_length=max_prompt_length)
    for t in input_tokens:
        if torch.is_tensor(input_tokens[t]):
            input_tokens[t] = input_tokens[t].to("cuda:0")

    outputs = model.generate(**input_tokens, **generate_kwargs)

    input_tokens_lengths = [x.shape[0] for x in input_tokens['input_ids']]
    output_tokens_lengths = [x.shape[0] for x in outputs]

    total_new_tokens = [o - i for i, o in zip(input_tokens_lengths, output_tokens_lengths)]
    outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    return zip(prompts, outputs, total_new_tokens)

model = model.cuda()
t_generate_start = time.time()
generated = generate()
t_generate_span = time.time() - t_generate_start
print(f"generate time: {t_generate_span}")
for prompt, output, num_new_tokens in generated:
    print(f"prompt: \n {prompt}")
    print(f"output: \n {output}")
    print(f"num_new_tokens: {num_new_tokens}")