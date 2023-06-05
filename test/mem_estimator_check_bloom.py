
from qllm.utils import get_model_size_cuda, list_tensors_on_cuda
from qllm.utils import (
    check_model_weights_dtype,
    to_device_recursive, to_device_recursive_except, 
    to_dtype_recursive, to_dtype_recursive_except, to_dtype_except_linear_layer, to_half_for_modules,
    get_iter_variable_size,
    ModelMemEstimator
)
from qllm.models.BLOOM import BloomForCausalLMSeq
from qllm.models import bloom

import lptorch
import torch
import copy
from transformers import LogitsProcessorList, StoppingCriteriaList

if __name__ == '__main__':
    model_size = '1b7'
    model_config = f'bigscience/bloom-{model_size}'
    bloom_560m, tokenizer = bloom.load_pretained_model_from_net(model_config)
    # sample text
    max_length = 512
    # sample text
    batched_ids = tokenizer.batch_encode_plus(["Hi, where is my dog. ", "Just test performance. How about you. ", \
                                                "The quick brown fox jumps over the lazy dog. It's a beautiful day outside, the sun is shining and the birds are chirping. I feel like going for a"], \
                                                padding='max_length', max_length=max_length, return_tensors="pt")
    
    weight_loaded_model = BloomForCausalLMSeq.from_pretrained(model_config, torch_dtype=torch.float16)
    print("decoder layernum", weight_loaded_model.get_decoder_layer_num())
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
            8: {'shard': [1], 'bits': [2]},
            9: {'shard': [0,1], 'bits': [16, 8]},
            10: {'shard': [0,1], 'bits': [4, 8]},
            11: {'shard': [0,1], 'bits': [16, 16]},
            # 350M
            12: {'shard': [0,1], 'bits': ['8:tc-li', 16]},
            13: {'shard': [0,1], 'bits': [8, 16]},
            14: {'shard': [0,1], 'bits': [16, 16]},
            15: {'shard': [0,1], 'bits': [16, 16]},
            16: {'shard': [0,1], 'bits': [16, 8]},
            17: {'shard': [0,1], 'bits': [16, 16]},
            18: {'shard': [0,1], 'bits': [16, 8]},
            19: {'shard': [0,1], 'bits': [16, 16]},
            20: {'shard': [0,1], 'bits': [4, 16]},
            21: {'shard': [0,1], 'bits': [16, 2]},
            22: {'shard': [0,1], 'bits': [16, 16]}, 
            23: {'shard': [0,1], 'bits': [16, 16]},
        }
    }    

    # set calib results
    caliber = lptorch.inner_caliber
    caliber.set_model(weight_loaded_model)
    caliber.set_fake()
    caliber.load_fake_calib_data(f'./fake_calib_bloom_{model_size}.pkl')

    model_pre_and_post = weight_loaded_model._pure_pre_and_post()
    model = weight_loaded_model.shard_model(sharding_strategy, 0)
    model_2 = weight_loaded_model.shard_model(sharding_strategy, 1)
    model_3 = weight_loaded_model.shard_model(sharding_strategy, 2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_pre_and_post = model_pre_and_post.cuda()
    model.decoder_layers_to_device(device)
    model_2.decoder_layers_to_device(device)
    model_3.decoder_layers_to_device(device)
    model_packs = [model, model_2, model_3]

    # init KV cache
    num_tokens_to_generate = 100
    bs, prompt_length = batched_ids['input_ids'].shape
    # becareful to use this one
    batched_ids = to_device_recursive(dict(batched_ids), device)

    
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
    generation_config = bloom_560m.generation_config
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

        input_ids = new_input_ids
        input_ids2 = new_input_ids2

    # generated token size
    original_length = original_token.shape[1]
    generated_length = input_ids.shape[1]
    print("Generated tokens:", generated_length - original_length)
    # h1 = bloom_560m.config.hidden_size
    # h2 = bloom_560m.decoders.layers[0].fc2.weight.shape[0]
    h1 = bloom_560m.config.hidden_size
    h2 = h1 * 4
    b = input_ids.shape[0]
    s = max_length
    n = num_tokens_to_generate
    request_num = 2

    config = bloom_560m.config
    vocab_size = config.vocab_size
    max_position_embeddings = 0
    word_embed_proj_dim = h1

    print(h1, h2)
    print(b, s, n)
    model_mem_estimator = ModelMemEstimator(h1, h2, b, s, n, \
                                            vocab_size, max_position_embeddings, word_embed_proj_dim)
    
    # model pre and post size
    print("Model pre and post size: ", get_model_size_cuda(model_pre_and_post, 'MB')[0] + get_model_size_cuda(model_pre_and_post.lm_head, 'MB')[0])
    print("Est Model pre and post size: ", model_mem_estimator.calculate_prepost_mem(unit='MB')[1])
    # print model 1, 2, 3 size in MB
    print("Model 1 size: ", get_model_size_cuda(model, 'MB')[1])
    print("Model 2 size: ", get_model_size_cuda(model_2, 'MB')[1])
    print("Model 3 size: ", get_model_size_cuda(model_3, 'MB')[1])
    # estimated model size
    print("Estimated Model1", model_mem_estimator.calculate_model_occupation_of_partition(sharding_strategy[0], unit='MB')[1])
    print("Estimated Model2", model_mem_estimator.calculate_model_occupation_of_partition(sharding_strategy[1], unit='MB')[1])
    print("Estimated Model3", model_mem_estimator.calculate_model_occupation_of_partition(sharding_strategy[2], unit='MB')[1])

    # KV size
    print("Model 1 KV size: ", get_iter_variable_size(model.transformer.get_all_kv_cache_dict(), unit='MB'))
    print("Model 2 KV size: ", get_iter_variable_size(model_2.transformer.get_all_kv_cache_dict(), unit='MB'))
    print("Model 3 KV size: ", get_iter_variable_size(model_3.transformer.get_all_kv_cache_dict(), unit='MB'))

    # estimator
    print("Estimated Model1 KV:", request_num * model_mem_estimator.calculate_kv_occupation_of_partition(sharding_strategy[0], 'MB')[0])
    print("Estimated Model2 KV:", request_num * model_mem_estimator.calculate_kv_occupation_of_partition(sharding_strategy[1], 'MB')[0])
    print("Estimated Model3 KV:", request_num * model_mem_estimator.calculate_kv_occupation_of_partition(sharding_strategy[2], 'MB')[0])





