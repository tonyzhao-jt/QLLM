
from qllm.models import bloom
from qllm.models.BLOOM import BloomForCausalLMSeq
from qllm.utils import get_model_size_cuda, list_tensors_on_cuda
from qllm.utils import (
    check_model_weights_dtype,
    to_device_recursive, to_device_recursive_except, 
    create_ds_indexes
)
from qllm.models.OPT import OPTForCausalLMSeq
from qllm.scheduler import DSScheduler
import lptorch
import torch
import copy
from transformers import LogitsProcessorList, StoppingCriteriaList
if __name__ == '__main__':
    model_config = 'bigscience/bloom-560m'
    bloom_560m, tokenizer = bloom.load_pretained_model_from_net(model_config)
    # sample text
    max_length = 512
    sample_text = ["Hi, where is my dog. ", "Hi, where is my dog. ", "Just test performance. How about you. ", "Just test performance. How about you. ", \
                                                "The quick brown fox jumps over the lazy dog. It's a beautiful day outside, the sun is shining and the birds are chirping. I feel like going for a"]
    
    
    weight_loaded_model = BloomForCausalLMSeq.from_pretrained(model_config, torch_dtype=torch.float16)

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
        },
        3:{
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
    print("decoder layernum", weight_loaded_model.get_decoder_layer_num())
    
    # set calib results
    caliber = lptorch.inner_caliber
    caliber.set_model(weight_loaded_model)
    caliber.set_fake()
    caliber.load_fake_calib_data('./fake_calib_opt_350m.pkl')

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
    bloom_560m = bloom_560m.cuda()



    # create prefill bs 1 and decode bs 2
    sample_text = ["Hi, where is my dog. ", "Hi, where is my dog. ", "Just test performance. How about you. ", "Just test performance. How about you. ", \
                                                "The quick brown fox jumps over the lazy dog. It's a beautiful day outside, the sun is shining and the birds are chirping. I feel like going for a"]
    
    decoder_bss = [2, 3]
    prefill_bs = 1
    prompt_length = max_length
    ds_scheduler = DSScheduler(prefill_bs, decoder_bss)
    sample_text_dict = ds_scheduler.split_list_of_prompts(sample_text)
    prefill_bs_indexes = ds_scheduler.create_ds_indexes()
    for request_id, sample_text_list in sample_text_dict.items():
        sample_text_dict[request_id] = to_device_recursive([dict(tokenizer.batch_encode_plus(text, padding='max_length', max_length=max_length, return_tensors="pt")) for text \
                        in sample_text_list], device)

    num_tokens_to_generate = 100
    # sample reference
    batched_ids = tokenizer.batch_encode_plus(sample_text, padding='max_length', max_length=max_length, return_tensors="pt")
    batched_ids = dict(batched_ids)
    batched_ids = to_device_recursive(batched_ids, device)
    
    with torch.no_grad():
        res_2 = bloom_560m(**batched_ids)

    def generate_one_token(request_token, input_ids):
        with torch.no_grad():
            intermediate_results = model.decode(request_token)
            intermediate_results = model_2.decode(intermediate_results)
            intermediate_results = model_3.decode(intermediate_results)

        request_id = intermediate_results[-2].item()
        # preprocessing  
        outputs = model_pre_and_post.postprocess(intermediate_results, None)
        next_token_logits = outputs.logits[:, -1, :]
        # pre-process distribution
        next_tokens_scores = logits_processor(input_ids, next_token_logits)
        next_tokens = torch.argmax(next_tokens_scores, dim=-1)
        flag, concat_tokens = ds_scheduler.pass_scheduler(request_id, next_tokens)
        if flag:
            new_input_ids = torch.cat([input_ids, concat_tokens], dim=-1)
            return new_input_ids, concat_tokens, request_id
        else:
            return None, None, request_id

    input_ids = batched_ids['input_ids']
    # prepare the logits processor
    logits_processor = LogitsProcessorList()
    generation_config =bloom_560m.generation_config
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
    # watch whether 
    # generate input token
    request_token_queues = []
    input_id_dict = {}
    for request_id, prefill_bs_indexes in prefill_bs_indexes.items():
        for idx, prefill_bs_index in enumerate(prefill_bs_indexes):
            current_sub_request_batch_ids = sample_text_dict[request_id][idx]
            if request_id not in input_id_dict:
                input_id_dict[request_id] = current_sub_request_batch_ids['input_ids']
            else:
                input_id_dict[request_id] = torch.cat([input_id_dict[request_id], current_sub_request_batch_ids['input_ids']], dim=0)
            # print(current_sub_request_batch_ids['input_ids'].shape)
            request_token = model_pre_and_post.preprocess(**current_sub_request_batch_ids, use_cache=True, request_id=request_id, batch_index=prefill_bs_index)
            request_token_queues.append(request_token)

        # init kv cache for all requests
        for k_model in model_packs:
            k_model.init_kv_cache(decoder_bss[request_id], prompt_length, num_tokens_to_generate, request_id=request_id)

    new_queue = []
    # token generation
    for i in range(num_tokens_to_generate):
        while request_token_queues:
            request_token = request_token_queues.pop(0)
            input_ids = input_id_dict[request_token[-2].item()]
            new_input_ids, next_token, request_id = generate_one_token(request_token, input_ids)
            if new_input_ids is not None:
                request_token = model_pre_and_post.preprocess_one_token(new_input_ids, next_token, use_cache=True, request_id=request_id)
                input_id_dict[request_id] = new_input_ids
                new_queue.append(request_token)
        request_token_queues = [token for token in new_queue]
        new_queue = []
        print(f"generated {i}-th token")



    print(batched_ids['input_ids'].shape)
    for request, input_id in input_id_dict.items():
        print("Request: ", request)
        print(tokenizer.batch_decode(input_id, skip_special_tokens=True))

    


    



