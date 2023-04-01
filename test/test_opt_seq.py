import torch 
import torch.nn as nn 
from qllm.models import opt
from qllm import SequentialGenerate
from time import perf_counter

model_size = 'facebook/opt-125m'
ava_model_sizes = opt.get_available_models()
print(ava_model_sizes)
# opt_125M, tokenizer = opt.get_empty_model(model_size)
opt_125M, tokenizer = opt.load_pretained_model_from_net(model_size)
print("model_loaded")

model = opt_125M

# from transformers import GenerationConfig
# print("Default Generation Config:", model.generation_config)

# get basic information
model.config.use_cache = False # not use cache to store the pevious computed result
# model structures
layers = model.model.decoder.layers
embed_tokens = model.model.decoder.embed_tokens
embed_positions = model.model.decoder.embed_positions
# the model dtype
dtype = next(iter(model.parameters())).dtype 

# sample text
input_ids = tokenizer.encode("Hi, where is my dog", return_tensors="pt")

# dev
device = "cuda:0"

input_ids = input_ids.to(device)
model = model.to(device)
generation_config = model.generation_config

# from transformers import LogitsProcessorList, StoppingCriteriaList

# logits_processor = LogitsProcessorList()

# inputs_tensor, model_input_name, model_kwargs = model._prepare_model_inputs(
#     input_ids, generation_config.bos_token_id, {}
# )
# input_ids_seq_length = input_ids.shape[-1]
# # 8. prepare distribution pre_processing samplers
# logits_processor = model._get_logits_processor(
#     generation_config=generation_config,
#     input_ids_seq_length=input_ids_seq_length,
#     encoder_input_ids=inputs_tensor,
#     prefix_allowed_tokens_fn=None,
#     logits_processor=logits_processor,
# )

# stopping_criteria = StoppingCriteriaList()
# # 9. prepare stopping criteria
# stopping_criteria = model._get_stopping_criteria(
#     generation_config=generation_config, stopping_criteria=stopping_criteria
# )

# # logits_processor = LogitsProcessorList([MinLengthLogitsProcessor(10, eos_token_id=model.generation_config.eos_token_id),])
# # stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(max_length=20)])

# with torch.no_grad():
#     generated_ids_naive = model.greedy_search(
#         input_ids,
#         logits_processor=logits_processor,
#         stopping_criteria=stopping_criteria,
#         pad_token_id=generation_config.pad_token_id,
#         eos_token_id=generation_config.eos_token_id,
#         output_scores=generation_config.output_scores,
#         return_dict_in_generate=generation_config.return_dict_in_generate,
#         synced_gpus=False,
#         **{},
#     )
# with torch.no_grad():
#     greedy_generate_ids = model.greedy_search(input_ids)
# result_greedy = tokenizer.batch_decode(greedy_generate_ids, skip_special_tokens=True)
# print("Onetime Run: ", result_greedy)
layers = model.model.decoder.layers
seq_gen = SequentialGenerate(model, input_ids=input_ids)
with torch.no_grad():
    generated_ids_seq = seq_gen.greedy_search(input_ids)

with torch.no_grad():
    generated_ids = model.generate(
        input_ids
    )
print(generated_ids_seq - generated_ids)
import pdb; pdb.set_trace()
result_one_time = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
print("Onetime Run: ", result_one_time)


# sample_generate_ids = model.sample(input_ids)
# result_sample = tokenizer.batch_decode(sample_generate_ids, skip_special_tokens=True)


# 
# # model_kwargs = generation_config.update(**kwargs)



prompt = "Hi, where is my dog?"
inputs = tokenizer(prompt, return_tensors="pt").to(device)

input_ids = inputs.input_ids
seq_len = input_ids.shape[-1]

# Wrap the transformer layers in an nn.Sequential module
opt_layers = nn.Sequential(*model.base_model.decoder.layers)

# Initialize the hidden state
hidden_state = model.base_model.decoder.get_input_embeddings()(input_ids)

# Execute the layers one by one
for layer in opt_layers:
    hidden_state = layer(hidden_state, attention_mask=None)[0]

# Apply the final linear layer to get logits
logits = model.lm_head(hidden_state)

# Get the predicted token ids by selecting the most probable token at each position
predicted_token_ids = torch.argmax(logits, dim=-1)

# Truncate the generated sequence to the desired length
max_length = 30
predicted_token_ids = predicted_token_ids[:, :max_length]
print(predicted_token_ids)
# Decode the predicted token ids to get the generated text
generated_text = tokenizer.batch_decode(predicted_token_ids, skip_special_tokens=True)

print(generated_text)






