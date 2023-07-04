from transformers import LogitsProcessorList, StoppingCriteriaList
from transformers.generation import GenerationConfig
import torch 
def greedy_processor(model, input_ids, tokens_to_generate, max_prompt_length):
    logits_processor = LogitsProcessorList()
    generation_config = model.generation_config
    new_generation_config = GenerationConfig.from_model_config(model.config)
    pad_token_id = new_generation_config.pad_token_id if new_generation_config.pad_token_id is not None else 3 # by default to be 3 or 1
    generate_kwargs = dict(max_new_tokens=tokens_to_generate, do_sample=False)
    new_generation_config.update(**generate_kwargs)
    new_generation_config.validate()
    new_generation_config.max_length = new_generation_config.max_new_tokens + input_ids.shape[-1]

    # "bos_token_id": 1,
    # "eos_token_id": 2,
    # "max_length": 30,
    # "max_new_tokens": 10,
    # "pad_token_id": 3,

    inputs_tensor, model_input_name, model_kwargs = model._prepare_model_inputs(
        input_ids, generation_config.bos_token_id, {}
    )
    input_ids_seq_length = input_ids.shape[-1]
    # 8. prepare distribution pre_processing samplers
    logits_processor = model._get_logits_processor(
        generation_config=new_generation_config,
        input_ids_seq_length=input_ids_seq_length,
        encoder_input_ids=inputs_tensor,
        prefix_allowed_tokens_fn=None,
        logits_processor=logits_processor,
    )
    # 2332
    unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)
    return logits_processor,(unfinished_sequences, pad_token_id)