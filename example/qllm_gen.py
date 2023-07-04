
from qllm.utils import batch_encode_plus, get_model_size_cuda,  to_device_recursive, greedy_processor
from qllm.models import bloom, opt, qllm_load_pretrained_from_size
import lptorch
import torch
import copy
import os 

from qllm.tp import utils as tp_utils
from qllm.utils.argparser import model_sample_gen_argparser

# import sharding strategies helper
from example_utils import create_uniform_sharding_strategies
if __name__ == '__main__':
    args = model_sample_gen_argparser()
    model_name = args.model_name
    model_size = args.model_size
    # sample workload
    max_gen_tokens = args.max_gen_tokens
    max_prompt_length = args.max_prompt_length
    # bitwidth and shards
    bitwidth = args.bitwidth
    num_shards = args.num_shards
    # sample prompts
    prompts = [
        "He is working on",
        "He has a",
        "Everyone is happy and I can",
    ]

    target_storage_folder = '/data/llms/converted_weights'
    # load model 
    if model_name == 'bloom':
        qllm_model, tokenizer, key = qllm_load_pretrained_from_size(model_name, model_size)
    elif model_name == 'opt':
        # use converted weight
        path = os.path.join(target_storage_folder, f"{model_name}_{model_size}")
        if not os.path.exists(path):
            raise ValueError("Please run weight_convert.py first")
        qllm_model, tokenizer, key = qllm_load_pretrained_from_size(model_name, model_size, target_storage_folder=target_storage_folder)

    batched_ids = batch_encode_plus(tokenizer, prompts, return_tensors="pt", max_length=max_prompt_length)
    decoder_layer_nums = qllm_model.get_decoder_layer_num()
    sharding_strategy = create_uniform_sharding_strategies(num_shards, decoder_layer_nums, bitwidth)
    
    # TODO: make it unnecessary later. 
    # set calibration data for the quantizer. leaves for the implementation of the smoothquant
    caliber = lptorch.inner_caliber
    caliber.set_model(qllm_model)
    caliber.set_fake()
    # check whether the calibration data is loaded
    calib_file_name = f'./fake_calib_{model_name}_{model_size}.pkl'
    if not os.path.exists(calib_file_name):
        raise ValueError(f"Please run the fake_calib_gen.py script first. {calib_file_name} not found")
    caliber.load_fake_calib_data(calib_file_name)

    # shard models
    model_pre_and_post = qllm_model._pure_pre_and_post()
    model_shard_nums = len(sharding_strategy.keys())
    model_packs = [qllm_model.shard_model(sharding_strategy, i) for i in range(model_shard_nums)]
    # check whether the packs number equals to the sharding strategy
    assert len(model_packs) == len(sharding_strategy), "model packs number should be equal to the sharding strategy"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Shard on the same gpu for reference
    # move all models to cuda
    model_pre_and_post = model_pre_and_post.cuda()
    [model.decoder_layers_to_device(device) for model in model_packs]
    # move tensor to device
    batched_ids = to_device_recursive(dict(batched_ids), device)

    # print model 1, 2, 3 size in MB
    for idx, model in enumerate(model_packs):
        print("Model {} size: ".format(idx), get_model_size_cuda(model, 'MB'))
    
    # be careful that, always shards before quantization.
    # some quantizer like llm.int8 triggers when the model is run cuda() or to(device)
    # if you first move the model to cuda, then shard it, the quantizer will not work

    def generate_one_token(request_token, input_ids):
        with torch.no_grad():
            intermediate_results = request_token
            for model in model_packs:
                intermediate_results = model.decode(intermediate_results)

        request_id = intermediate_results[-1]
        # preprocessing  
        outputs = model_pre_and_post.postprocess(intermediate_results, None)
        # 2361 - 2385
        next_token_logits = outputs.logits[:, -1, :]
        next_tokens_scores = logits_processor(input_ids, next_token_logits)
        next_tokens = torch.argmax(next_tokens_scores, dim=-1)
        next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)
        new_input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        
        return new_input_ids, next_tokens

    input_ids = batched_ids['input_ids']
    logits_processor,(unfinished_sequences, pad_token_id) = greedy_processor(qllm_model, input_ids, max_gen_tokens, max_prompt_length)
    # generate input token
    request_token = model_pre_and_post.preprocess(**batched_ids, use_cache=True, request_id=1)
    request_token2 = model_pre_and_post.preprocess(**batched_ids, use_cache=True, request_id=2)


    # init kv cache for all requests
    bs, _ = batched_ids['input_ids'].shape
    for k_model in model_packs:
        k_model.init_kv_cache(bs, max_prompt_length, max_gen_tokens, request_id=1)
        k_model.init_kv_cache(bs, max_prompt_length, max_gen_tokens, request_id=2)

    original_token = copy.deepcopy(input_ids)
    input_ids2 = copy.deepcopy(input_ids)

    for i in range(max_gen_tokens):
        new_input_ids, next_token = generate_one_token(request_token, input_ids)
        new_input_ids2, next_token2 = generate_one_token(request_token2, input_ids2)
        request_token = model_pre_and_post.preprocess_one_token(new_input_ids, next_token, attention_mask=request_token[1], use_cache=True, request_id=1)
        request_token2 = model_pre_and_post.preprocess_one_token(new_input_ids, next_token2, attention_mask=request_token2[1], use_cache=True, request_id=2)
        # print("KV Cache Size 2: ", get_iter_variable_size(model.model.decoder.kv_cache, unit='MB'))

        input_ids = new_input_ids
        input_ids2 = new_input_ids2

    result_one_time = tokenizer.batch_decode(new_input_ids, skip_special_tokens=True)
    result_one_time2 = tokenizer.batch_decode(new_input_ids2, skip_special_tokens=True)
    print("Prompts: ", prompts)
    print("Generated Token 1: ", result_one_time)
    print("Generated Token 2: ", result_one_time2)
    


    



