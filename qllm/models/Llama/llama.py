import torch
# Wait decapoca merged
from transformers import (
    LlamaForCausalLM,
    LlamaConfig,
    AutoTokenizer,
    LlamaTokenizer
)
# from transformers import (
#     LLaMAForCausalLM as  LlamaForCausalLM,
#     LLaMAConfig as LlamaConfig,
#     AutoTokenizer
# )

# according to its original paper: https://arxiv.org/pdf/2302.13971.pdf
model_cards = {
    '7b': LlamaConfig(
        hidden_size=4096,
        num_hidden_layers=32,
        num_attention_heads=32,
    ),
    '13b': LlamaConfig(
        hidden_size=5120,
        num_hidden_layers=40,
        num_attention_heads=40,
    ),
    '33b': LlamaConfig(
        hidden_size=6656,
        num_hidden_layers=60,
        num_attention_heads=52,
    ),
    '65b': LlamaConfig(
        hidden_size=8192,
        num_hidden_layers=80,
        num_attention_heads=64,
    ),
}

def get_empty_model(model_size:str='7b'):
    assert model_size in model_cards, f"model size {model_size} not available"
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    config = model_cards[model_size]
    # model = LlamaModel(config)
    model =  LlamaForCausalLM(config)
    tokenizer = LlamaTokenizer.from_pretrained("decapoda-research/llama-7b-hf")
    return model, tokenizer

def get_available_models():
    return model_cards.keys()


AVAILABLE_MAP = {
    "decapoda-research/llama-7b-hf": "https://huggingface.co/decapoda-research/llama-7b-hf",
    "decapoda-research/llama-13b-hf": "https://huggingface.co/decapoda-research/llama-13b-hf",
    "decapoda-research/llama-30b-hf": "https://huggingface.co/decapoda-research/llama-30b-hf",
    "decapoda-research/llama-65b-hf": "https://huggingface.co/decapoda-research/llama-65b-hf",
    "decapoda-research/llama-7b-hf-int4": "https://huggingface.co/decapoda-research/llama-7b-hf-int4"
}

def load_pretained_model_from_net(repo_name):
    assert repo_name in AVAILABLE_MAP, f"model {repo_name} not available in repo"
    model =  LlamaForCausalLM.from_pretrained(repo_name)
    tokenizer = LlamaTokenizer.from_pretrained(repo_name)
    return model, tokenizer

def get_available_pretained_models():
    return AVAILABLE_MAP.keys()