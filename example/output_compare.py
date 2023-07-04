'''
    ensure the output is same for QLLM and original version for LLM inference
'''
# generation reference for the bloom and opt
from qllm.utils import batch_encode_plus, to_device_recursive
from qllm.utils.argparser import model_config_argparser
from qllm.models import bloom, opt, qllm_load_pretrained_from_size
from qllm.utils import batch_encode_plus
import torch 
import os 

args = model_config_argparser()
model_name = args.model_name
model_size = args.model_size
target_storage_folder = '/data/llms/converted_weights'
if model_name == 'bloom':
    qllm_model, _, key = qllm_load_pretrained_from_size(model_name, model_size)
    ref_model, tokenizer = bloom.load_pretained_model_from_net(key)
    ref_model.eval()
elif model_name == 'opt':
    # use converted weight
    path = os.path.join(target_storage_folder, f"{model_name}_{model_size}")
    if not os.path.exists(path):
        raise ValueError("Please run weight_convert.py first")
    qllm_model, _, key = qllm_load_pretrained_from_size(model_name, model_size, target_storage_folder=target_storage_folder)
    ref_model, tokenizer = opt.load_pretained_model_from_net(key)
    ref_model.eval()

num_tokens_to_generate = 10
max_length = 20
# sample input
prompts = [
    "He is working on",
    "He has a",
    "Everyone is happy and I can",
]
batched_ids = batch_encode_plus(tokenizer, prompts, return_tensors="pt", max_length=max_length)

ref_inout = {}
def debug_forward_hook(module, input, output):
    name = module.name 
    ref_inout[name] = (input, output)

# move to cuda 
ref_model = ref_model.cuda()
qllm_model = qllm_model.cuda()
batched_ids = to_device_recursive(batched_ids, torch.device('cuda'))    

# fp16 compare
if model_name == 'bloom':
    first_block_ref = ref_model.transformer.h[0]
    first_block_qllm = qllm_model.transformer.h[0]
    first_block_ref.name = 'first_block_ref'
    first_block_qllm.name = 'first_block_qllm'
elif model_name == 'opt':
    first_block_ref = ref_model.model.decoder.layers[0]
    first_block_qllm = qllm_model.model.decoder.layers[0]
    first_block_ref.name = 'first_block_ref'
    first_block_qllm.name = 'first_block_qllm'
# add forward hooks
first_block_ref.register_forward_hook(debug_forward_hook)
first_block_qllm.register_forward_hook(debug_forward_hook)
with torch.no_grad():
    ref_output = ref_model(**batched_ids)
    print("Ref model run successfully")

with torch.no_grad():
    qllm_output = qllm_model(**batched_ids)
    print("QLLM model run successfully")

first_input = ref_inout['first_block_ref'][0][0]
first_input_qllm = ref_inout['first_block_qllm'][0][0]
if not torch.allclose(first_input, first_input_qllm):
    print('first layer input is not consistent')
    import pdb; pdb.set_trace()
first_output = ref_inout['first_block_ref'][1][0]
first_output_qllm = ref_inout['first_block_qllm'][1][0]
if not torch.allclose(first_output, first_output_qllm):
    print('first layer output is not consistent')
    import pdb; pdb.set_trace()

# check final output
if not torch.allclose(ref_output.logits, qllm_output.logits):
    print('output is not consistent')
    import pdb; pdb.set_trace()
print('Output diff', torch.max(torch.abs(ref_output.logits - qllm_output.logits)))
print(model_name, model_size, "verified")
    



