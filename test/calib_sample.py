
from qllm.models import opt 
from qllm.utils import get_model_size_cuda, list_tensors_on_cuda
from qllm.utils import (
    check_model_weights_dtype,
    to_device_recursive, to_device_recursive_except, 
    to_dtype_recursive, to_dtype_recursive_except, to_dtype_except_linear_layer, to_half_for_modules
)
from qllm.models.OPT import OPTForCausalLMSeq
import lptorch
import torch
if __name__ == '__main__':
    model_size = '125m'.lower()
    opt_125M, tokenizer = opt.load_pretained_model_from_net(f'facebook/opt-{model_size}')
    # sample text
    input_ids = tokenizer.encode("Hi, where is my dog", return_tensors="pt")
    weight_loaded_model = OPTForCausalLMSeq.from_pretrained(f"facebook/opt-{model_size}", torch_dtype=torch.float32)
    
    caliber = lptorch.inner_caliber
    caliber.set_model(weight_loaded_model)
    # caliber.default_hook = caliber.torch_int_forward_hook
    caliber.register_forward_hooks()
    # get calib result
    with torch.no_grad():
        weight_loaded_model(input_ids)
    caliber.remove_forward_hooks()
    caliber.save_calib_data(f'./opt_{model_size}_calib_data.pkl')
    caliber.clear_calib_data()
    caliber.load_calib_data(f'./opt_{model_size}_calib_data.pkl')

