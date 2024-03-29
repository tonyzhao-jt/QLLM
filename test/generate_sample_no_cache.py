
from qllm.models import opt 
from qllm.utils import get_model_size_cuda, list_tensors_on_cuda
from qllm.utils import (
    check_model_weights_dtype,
    to_device_recursive, to_device_recursive_except, 
    to_dtype_recursive, to_dtype_recursive_except, to_dtype_except_linear_layer, to_half_for_modules,
    get_iter_variable_size
)
from qllm.models.OPT import OPTForCausalLMSeq
import lptorch
import torch
import copy
from transformers import LogitsProcessorList, StoppingCriteriaList

if __name__ == '__main__':
    seed=42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    opt_125M, tokenizer = opt.load_pretained_model_from_net('facebook/opt-125m')
    # sample text
    input_ids = tokenizer.encode("Hi, where is my dog", return_tensors="pt")
    weight_loaded_model = OPTForCausalLMSeq.from_pretrained("facebook/opt-125m", torch_dtype=torch.float16)
    sharding_strategy = {
        0: {
            0: {'shard': [0, 1], 'bits': [16, 16]},
            1: {'shard': [0, 1], 'bits': [16, 16]},
            2: {'shard': [0, 1], 'bits': [16, 16]},
            3: {'shard': [0, 1], 'bits': [16, 16]},
            4: {'shard': [0, 1], 'bits': [16, 16]},
            5: {'shard': [0, 1], 'bits': [16, 16]},
            6: {'shard': [0], 'bits': [16]},
        },
        1: {
            6: {'shard': [1], 'bits': [16]},
            7: {'shard': [0,1], 'bits': [16, 16]},
            8: {'shard': [0,1], 'bits': [16, 16]},
        },
        2: {
            9: {'shard': [0,1], 'bits': [16, 16]},
            10: {'shard': [0,1], 'bits': [16, 16]},
            11: {'shard': [0,1], 'bits': [16, 16]},
        }
    }
    print("decoder layernum", weight_loaded_model.model.decoder.get_decoder_layer_num())
    # sharding_strategy = {
    #     0: {
    #     },
    #     1: {
    #         0: {'shard': [0, 1], 'bits': [16, 16]},
    #         1: {'shard': [0, 1], 'bits': [16, 16]},
    #         2: {'shard': [0, 1], 'bits': [16, 16]},
    #         3: {'shard': [0, 1], 'bits': [16, 16]},
    #         4: {'shard': [0, 1], 'bits': [16, 16]},
    #         5: {'shard': [0, 1], 'bits': [16, 16]},
    #         6: {'shard': [0, 1], 'bits': [16, 16]},
    #         7: {'shard': [0, 1], 'bits': [16, 16]},
    #         8: {'shard': [0], 'bits': [16]},
    #     },
    #     2: {
    #         8: {'shard': [1], 'bits': [16]},
    #         9: {'shard': [0,1], 'bits': [16, 16]},
    #         10: {'shard': [0,1], 'bits': [16, 16]},
    #         11: {'shard': [0,1], 'bits': [16, 16]},
    #         # 350M
    #         12: {'shard': [0,1], 'bits': [16, 16]},
    #         13: {'shard': [0,1], 'bits': [16, 16]},
    #         14: {'shard': [0,1], 'bits': [16, 16]},
    #         15: {'shard': [0,1], 'bits': [16, 16]},
    #         16: {'shard': [0,1], 'bits': [16, 16]},
    #         17: {'shard': [0,1], 'bits': [16, 16]},
    #         18: {'shard': [0,1], 'bits': [16, 16]},
    #         19: {'shard': [0,1], 'bits': [16, 16]},
    #         20: {'shard': [0,1], 'bits': [16, 16]},
    #         21: {'shard': [0,1], 'bits': [16, 16]},
    #         22: {'shard': [0,1], 'bits': [16, 16]}, 
    #         23: {'shard': [0,1], 'bits': [16, 16]},
    #     }
    # }    
    
    # set calib results
    caliber = lptorch.inner_caliber
    caliber.set_model(weight_loaded_model)
    caliber.load_calib_data('./opt_350M_calib_data.pkl')

    model_pre_and_post = weight_loaded_model._pure_pre_and_post()
    model = weight_loaded_model.shard_model(sharding_strategy, 0)
    model_2 = weight_loaded_model.shard_model(sharding_strategy, 1)
    model_3 = weight_loaded_model.shard_model(sharding_strategy, 2)
    model_packs = [model, model_2, model_3]

    # init KV cache
    num_tokens_to_generate = 10
    bs, prompt_length = input_ids['input_ids'].shape

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

    
    def generate_one_token(request_token, input_ids):
        with torch.no_grad():
            intermediate_results = model.decode(request_token)
            intermediate_results = model_2.decode(intermediate_results)
            intermediate_results = model_3.decode(intermediate_results)

        request_id = intermediate_results[-1]
        # preprocessing  
        outputs = model_pre_and_post.postprocess(intermediate_results, None)
        next_token_logits = outputs.logits[:, -1, :]
        # pre-process distribution
        next_tokens_scores = logits_processor(input_ids, next_token_logits)
        next_tokens = torch.argmax(next_tokens_scores, dim=-1)
        new_input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        return new_input_ids

    # prepare the logits processor
    logits_processor = LogitsProcessorList()
    generation_config = opt_125M.generation_config
    inputs_tensor, model_input_name, model_kwargs = model._prepare_model_inputs(
        input_ids, generation_config.bos_token_id, {}
    )
    input_ids_seq_length = input_ids.shape[-1]
    # 8. prepare distribution pre_processing samplers
    logits_processor = model._get_logits_processor(
        generation_config=generation_config,
        input_ids_seq_length=input_ids_seq_length,
        encoder_input_ids=inputs_tensor,
        prefix_allowed_tokens_fn=None,
        logits_processor=logits_processor,
    )
    
    # generate input token
    request_token = model_pre_and_post.preprocess(input_ids, use_cache=False, request_id=1)

    num_tokens_to_generate = 8
    original_token = copy.deepcopy(input_ids)

    for i in range(num_tokens_to_generate):
        new_input_ids = generate_one_token(request_token, input_ids)
        request_token = model_pre_and_post.preprocess(new_input_ids, use_cache=False, request_id=1)
        # print(request_token[1])
        # print("KV Cache Size 2: ", get_iter_variable_size(model.model.decoder.kv_cache, unit='MB'))
        input_ids = new_input_ids
        # print(input_ids)

    # print model 1, 2, 3 size in MB
    print("Model 1 size: ", get_model_size_cuda(model.model, 'MB'))
    print("Model 2 size: ", get_model_size_cuda(model_2.model, 'MB'))
    print("Model 3 size: ", get_model_size_cuda(model_3.model, 'MB'))

    result_one_time = tokenizer.batch_decode(new_input_ids, skip_special_tokens=True)
    print("Onetime Run: ", result_one_time)
    


    



