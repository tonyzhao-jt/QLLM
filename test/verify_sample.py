
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
    opt_125M, tokenizer = opt.load_pretained_model_from_net('facebook/opt-350m')
    # sample text
    input_ids = tokenizer.encode("Hi, where is my dog", return_tensors="pt")
    weight_loaded_model = OPTForCausalLMSeq.from_pretrained("facebook/opt-350m", torch_dtype=torch.float16)
    print("decoder layernum", weight_loaded_model.model.decoder.get_decoder_layer_num())

    def verify_bits(loaded_model, bit):
        tmp_sharding_strategy = {
            1: {
                0: {'shard': [0,1], 'bits': [bit, bit]},
            },   
        }
        model = loaded_model.shard_model(tmp_sharding_strategy, 1) # avoid the master check
        verify_result = model.model.decoder.verify_decoder_layers()
        print("Verify Result for bit {}: selfattn:{}, ffn:{}".format(bit, verify_result[0], verify_result[1]))
        del model, verify_result
    # set calib results
    caliber = lptorch.inner_caliber
    caliber.set_model(weight_loaded_model)
    caliber.load_calib_data('./opt_350M_calib_data.pkl')

    caliber.set_bs(64)

    verify_bits(weight_loaded_model, '8:tc')
    verify_bits(weight_loaded_model, '8:tc-li')
    verify_bits(weight_loaded_model, 8)
    verify_bits(weight_loaded_model, 4)
    verify_bits(weight_loaded_model, 2)






    