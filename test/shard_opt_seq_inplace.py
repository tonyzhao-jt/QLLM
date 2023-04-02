
from qllm.models import opt 
from qllm.utils import get_model_size, get_model_size_cuda
from qllm.models.OPT import OPTForCausalLMSeq
import torch
if __name__ == '__main__':
    opt_125M, tokenizer = opt.load_pretained_model_from_net('facebook/opt-125m')
    # sample text
    input_ids = tokenizer.encode("Hi, where is my dog", return_tensors="pt")
    model = OPTForCausalLMSeq.from_pretrained("facebook/opt-125m")
    model_2 = OPTForCausalLMSeq.from_pretrained("facebook/opt-125m")
    model_3 = OPTForCausalLMSeq.from_pretrained("facebook/opt-125m")

    sharding_strategy = {
        0: {
            0: {'shard': [0, 1], 'bits': [8, 8]},
            1: {'shard': [0, 1], 'bits': [8, 8]},
            2: {'shard': [0, 1], 'bits': [8, 8]},
            3: {'shard': [0, 1], 'bits': [8, 8]},
            4: {'shard': [0, 1], 'bits': [8, 8]},
            5: {'shard': [0, 1], 'bits': [8, 8]},
            6: {'shard': [0], 'bits': [8]},
        },
        1: {
            6: {'shard': [1], 'bits': [8]},
            7: {'shard': [0,1], 'bits': [8, 8]},
            8: {'shard': [0,1], 'bits': [8, 8]},
        },
        2: {
            9: {'shard': [0,1], 'bits': [8, 8]},
            10: {'shard': [0,1], 'bits': [8, 8]},
            11: {'shard': [0,1], 'bits': [8, 8]},
        }
    }
    model._shard_model(sharding_strategy, 0)
    model_2._shard_model(sharding_strategy, 1)
    model_3._shard_model(sharding_strategy, 2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.cuda()
    model_2.decoder_to_device(device)
    model_3.decoder_to_device(device)
    opt_125M = opt_125M.cuda()
    input_ids = input_ids.cuda()

    # print model 1, 2, 3 size in MB
    print("Model 1 size: ", get_model_size_cuda(model, 'MB'))
    print("Model 2 size: ", get_model_size_cuda(model_2, 'MB'))
    print("Model 3 size: ", get_model_size_cuda(model_3, 'MB'))

    with torch.no_grad():
        res_2 = opt_125M(input_ids)
        pre_result = model.preprocess(input_ids, use_cache=True)

        # simulate the broadcast operation
        model_2.other_decode_params = model.other_decode_params
        model_3.other_decode_params = model.other_decode_params

        intermediate_results = model.decode(pre_result)
        intermediate_results = model_2.decode(intermediate_results)
        intermediate_results = model_3.decode(intermediate_results)
        
        res_1 = model.postprocess(intermediate_results, None)

    print(res_1.logits - res_2.logits)