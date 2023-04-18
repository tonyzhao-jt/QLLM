
from qllm.models import opt 
from qllm.utils import get_model_size_cuda, list_tensors_on_cuda
from qllm.utils import (
    check_model_weights_dtype,
    to_device_recursive, to_device_recursive_except, 
    to_dtype_recursive, to_dtype_recursive_except, to_dtype_except_linear_layer, to_half_for_modules
)
from qllm.models.OPT import OPTForCausalLMSeq, OPTDecoderSeq, OPTDecoderLayerSharded
from qllm.models.OPT.opt import model_cards

import lptorch
import torch
if __name__ == '__main__':
    model_size = '125m'.lower()
    # fake input = b * 1 * hidden_size
    config = model_cards[model_size]
    batch_size = 10
    input_seq_length = 1
    h1 = config.hidden_size
    fake_input = torch.randn(batch_size, input_seq_length, h1)
    decoder_layer = OPTDecoderLayerSharded(config)

    caliber = lptorch.inner_caliber
    caliber.set_fake()  

    # caliber.set_model(decoder_layer)
    # # caliber.default_hook = caliber.torch_int_forward_hook
    # caliber.register_forward_hooks()
    # # get calib result
    # with torch.no_grad():
    #     decoder_layer(fake_input)
    # caliber.remove_forward_hooks()
    # for layer_name, calib_res in caliber.collected_calib_data.items():
    #     calib_shape = caliber.collected_input_shape[layer_name]
    #     caliber.set_fake_module_calib_data(layer_name, calib_shape, calib_res)
    
    # caliber.save_fake_calib_data(f'fake_calib_{model_size}_.pkl')
    caliber.load_fake_calib_data(f'fake_calib_{model_size}_.pkl')

    sharding_strategy = {
        0: {
            0: {'shard': [0, 1], 'bits': ['8:tc', '8:tc']},
            1: {'shard': [0, 1], 'bits': ['8:tc', '8:tc']},
            2: {'shard': [0, 1], 'bits': ['8:tc', '8:tc']},
            3: {'shard': [0, 1], 'bits': ['8:tc', '8:tc']},
            4: {'shard': [0, 1], 'bits': [16, 16]},
            5: {'shard': [0, 1], 'bits': [16, 16]},
            6: {'shard': [0], 'bits': [16]},
        },
        1: {
            6: {'shard': [1], 'bits': [8]},
            7: {'shard': [0,1], 'bits': [8, 8]},
            8: {'shard': [0,1], 'bits': [8, 8]},
        },
        2: {
            9: {'shard': [0,1], 'bits': [16, 16]},
            10: {'shard': [0,1], 'bits': [16, 16]},
            11: {'shard': [0,1], 'bits': [16, 16]},
        }
    }

    # try quantization.
    weight_loaded_model = OPTForCausalLMSeq.from_pretrained(f"facebook/opt-{model_size}", torch_dtype=torch.float32)
    model = weight_loaded_model.shard_model(sharding_strategy, 0)
    model.cuda()
    

    


