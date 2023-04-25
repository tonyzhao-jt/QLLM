
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
    opt_125M, tokenizer = opt.load_pretained_model_from_net('facebook/opt-125m')
    # sample text
    max_length = 512
    # sample text
    batched_ids = tokenizer.batch_encode_plus(["Hi, where is my dog. ", "Just test performance. How about you. ", \
                                                "The quick brown fox jumps over the lazy dog. It's a beautiful day outside, the sun is shining and the birds are chirping. I feel like going for a"], \
                                                padding='max_length', max_length=max_length, return_tensors="pt")
    
    weight_loaded_model = OPTForCausalLMSeq.from_pretrained("facebook/opt-125m", torch_dtype=torch.float16)
    sharding_strategy = {
        0: {
            0: {'shard': [0, 1], 'bits': [16, 16]},
            1: {'shard': [0, 1], 'bits': [16, 16]},
            2: {'shard': [0, 1], 'bits': ['8:tc', '8:tc']},
            3: {'shard': [0, 1], 'bits': ['8:tc-li', 16]},
            4: {'shard': [0, 1], 'bits': [16, 4]},
            5: {'shard': [0, 1], 'bits': [16, 16]},
            6: {'shard': [0], 'bits': ['8:tc']},
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_pre_and_post.cuda()
    model.decoder_layers_to_device(device)
    model_2.decoder_layers_to_device(device)
    model_3.decoder_layers_to_device(device)
    opt_125M = opt_125M.cuda()

    # init KV cache
    num_tokens_to_generate = 100
    bs, prompt_length = batched_ids['input_ids'].shape
    # becareful to use this one
    batched_ids = to_device_recursive(dict(batched_ids), device)

    # print model 1, 2, 3 size in MB
    print("Original Model Size:", get_model_size_cuda(opt_125M.model, 'MB'))
    # print model 1, 2, 3 size in MB
    print("Model 1 size: ", get_model_size_cuda(model.model, 'MB'))
    print("Model 2 size: ", get_model_size_cuda(model_2.model, 'MB'))
    print("Model 3 size: ", get_model_size_cuda(model_3.model, 'MB'))

    with torch.no_grad():
        res_2 = opt_125M(**batched_ids)

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
        return new_input_ids, next_tokens

    input_ids = batched_ids['input_ids']
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
    request_token = model_pre_and_post.preprocess(**batched_ids, use_cache=True, request_id=1)
    request_token2 = model_pre_and_post.preprocess(**batched_ids, use_cache=True, request_id=2)

    # print(request_token)
    # init kv cache for all requests
    for k_model in model_packs:
        k_model.init_kv_cache(bs, prompt_length, num_tokens_to_generate, request_id=1)
        k_model.init_kv_cache(bs, prompt_length, num_tokens_to_generate, request_id=2)

    original_token = copy.deepcopy(input_ids)
    input_ids2 = copy.deepcopy(input_ids)
    for i in range(num_tokens_to_generate):
        new_input_ids, next_token = generate_one_token(request_token, input_ids)
        new_input_ids2, next_token2 = generate_one_token(request_token2, input_ids2)
        request_token = model_pre_and_post.preprocess_one_token(new_input_ids, next_token, use_cache=True, request_id=1)
        request_token2 = model_pre_and_post.preprocess_one_token(new_input_ids, next_token2, use_cache=True, request_id=2)
        # print("KV Cache Size 2: ", get_iter_variable_size(model.model.decoder.kv_cache, unit='MB'))

        input_ids = new_input_ids
        input_ids2 = new_input_ids2
        # print("token generated: ", i)


    print(original_token.shape, new_input_ids.shape, new_input_ids2.shape)
    # print model 1, 2, 3 size in MB
    print("Model 1 size: ", get_model_size_cuda(model.model, 'MB'))
    print("Model 2 size: ", get_model_size_cuda(model_2.model, 'MB'))
    print("Model 3 size: ", get_model_size_cuda(model_3.model, 'MB'))

    result_one_time = tokenizer.batch_decode(new_input_ids, skip_special_tokens=True)
    result_one_time2 = tokenizer.batch_decode(new_input_ids2, skip_special_tokens=True)
    print("Onetime Run: ", result_one_time)
    print("Onetime Run 2: ", result_one_time2)
    


    



