
from qllm.models import opt 
from qllm.utils import get_model_size_cuda, list_tensors_on_cuda
from qllm.utils import (
    check_model_weights_dtype,
    to_device_recursive, to_device_recursive_except, 
    to_dtype_recursive, to_dtype_recursive_except, to_dtype_except_linear_layer, to_half_for_modules
)
from qllm.models.OPT import OPTForCausalLMSeq
import torch
if __name__ == '__main__':
    opt_125M, tokenizer = opt.load_pretained_model_from_net('facebook/opt-350m')
    # sample text
    input_ids = tokenizer.encode("Hi, where is my dog", return_tensors="pt")
    weight_loaded_model = OPTForCausalLMSeq.from_pretrained("facebook/opt-350m", torch_dtype=torch.float16)
    # sharding_strategy = {
    #     0: {
    #         0: {'shard': [0, 1], 'bits': [8, 8]},
    #         1: {'shard': [0, 1], 'bits': [8, 8]},
    #         2: {'shard': [0, 1], 'bits': [8, 8]},
    #         3: {'shard': [0, 1], 'bits': [8, 8]},
    #         4: {'shard': [0, 1], 'bits': [8, 8]},
    #         5: {'shard': [0, 1], 'bits': [8, 8]},
    #         6: {'shard': [0], 'bits': [8]},
    #     },
    #     1: {
    #         6: {'shard': [1], 'bits': [8]},
    #         7: {'shard': [0,1], 'bits': [8, 8]},
    #         8: {'shard': [0,1], 'bits': [8, 8]},
    #     },
    #     2: {
    #         9: {'shard': [0,1], 'bits': [8, 8]},
    #         10: {'shard': [0,1], 'bits': [8, 8]},
    #         11: {'shard': [0,1], 'bits': [8, 8]},
    #     }
    # }
    print("decoder layernum", weight_loaded_model.model.decoder.get_decoder_layer_num())
    sharding_strategy = {
        0: {
        },
        1: {
            0: {'shard': [0, 1], 'bits': [16, 16]},
            1: {'shard': [0, 1], 'bits': [16, 16]},
            2: {'shard': [0, 1], 'bits': [16, 16]},
            3: {'shard': [0, 1], 'bits': [16, 16]},
            4: {'shard': [0, 1], 'bits': [16, 16]},
            5: {'shard': [0, 1], 'bits': [16, 16]},
            6: {'shard': [0, 1], 'bits': [16, 16]},
            7: {'shard': [0, 1], 'bits': [16, 16]},
            8: {'shard': [0], 'bits': [16]},
        },
        2: {
            8: {'shard': [1], 'bits': [16]},
            9: {'shard': [0,1], 'bits': [16, 16]},
            10: {'shard': [0,1], 'bits': [16, 16]},
            11: {'shard': [0,1], 'bits': [16, 16]},
            # 350M
            12: {'shard': [0,1], 'bits': [16, 16]},
            13: {'shard': [0,1], 'bits': [16, 16]},
            14: {'shard': [0,1], 'bits': [16, 16]},
            15: {'shard': [0,1], 'bits': [16, 16]},
            16: {'shard': [0,1], 'bits': [16, 16]},
            17: {'shard': [0,1], 'bits': [16, 16]},
            18: {'shard': [0,1], 'bits': [16, 16]},
            19: {'shard': [0,1], 'bits': [16, 16]},
            20: {'shard': [0,1], 'bits': [16, 16]},
            21: {'shard': [0,1], 'bits': [16, 16]},
            22: {'shard': [0,1], 'bits': [16, 16]}, 
            23: {'shard': [0,1], 'bits': [16, 16]},
        }
    }
    model_pre_and_post = weight_loaded_model._pure_pre_and_post()
    model = weight_loaded_model.shard_model(sharding_strategy, 0)
    model_2 = weight_loaded_model.shard_model(sharding_strategy, 1)
    model_3 = weight_loaded_model.shard_model(sharding_strategy, 2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_pre_and_post.cuda()
    model.decoder_layers_to_device(device)
    model_2.decoder_layers_to_device(device)
    model_3.decoder_layers_to_device(device)
    opt_125M = opt_125M.cuda()
    # becareful to use this one
    input_ids = input_ids.cuda()

    # eval mode
    # model.eval()
    # model_2.eval()
    # model_3.eval()
    # opt_125M.eval()

    # print model 1, 2, 3 size in MB
    print("Original Model Size:", get_model_size_cuda(opt_125M.model, 'MB'))
    # print model 1, 2, 3 size in MB
    print("Model 1 size: ", get_model_size_cuda(model.model, 'MB'))
    print("Model 2 size: ", get_model_size_cuda(model_2.model, 'MB'))
    print("Model 3 size: ", get_model_size_cuda(model_3.model, 'MB'))

    with torch.no_grad():
        res_2 = opt_125M(input_ids)

    # this part is communication in distributed serving
    with torch.no_grad():
        pre_result = model_pre_and_post.preprocess(input_ids, use_cache=True)
        intermediate_results = model.decode(pre_result)
        intermediate_results = model_2.decode(intermediate_results)
        intermediate_results = model_3.decode(intermediate_results)
        res_1 = model_pre_and_post.postprocess(intermediate_results, None)

    print(torch.max(res_1.logits - res_2.logits))