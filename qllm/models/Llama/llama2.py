import torch
# Wait decapoca merged
from transformers import (
    LlamaForCausalLM as LLMCasualLM,
    LlamaConfig as LLMConfig,
    AutoTokenizer
)
# from transformers import (
#     LLaMAForCausalLM as LLMCasualLM,
#     LLaMAConfig as LLMConfig,
#     AutoTokenizer
# )

# according to its original paper: https://arxiv.org/pdf/2302.13971.pdf
model_cards = {
    '7b': LLMConfig(
        hidden_size=4096,
        num_hidden_layers=32,
        num_attention_heads=32,
    ),
    '13b': LLMConfig(
        hidden_size=5120,
        num_hidden_layers=40,
        num_attention_heads=40,
    ),
    '70b': LLMConfig(
        hidden_size=6656,
        num_hidden_layers=60,
        num_attention_heads=52,
    )
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
    model = LLMCasualLM(config)
    tokenizer = AutoTokenizer.from_pretrained("decapoda-research/llama-7b-hf")
    return model, tokenizer

def get_available_models():
    return model_cards.keys()


AVAILABLE_MAP = {
    "meta-llama/Llama-2-7b-hf": "https://huggingface.co/meta-llama/Llama-2-7b-hf",
    "meta-llama/Llama-2-13b-hf": "https://huggingface.co/meta-llama/Llama-2-13b-hf",
    "meta-llama/Llama-2-70b-hf": "https://huggingface.co/meta-llama/Llama-2-70b-hf",
}

def load_pretained_model_from_net(repo_name):
    assert repo_name in AVAILABLE_MAP, f"model {repo_name} not available in repo"
    model = LLMCasualLM.from_pretrained(repo_name)
    tokenizer = AutoTokenizer.from_pretrained(repo_name)
    return model, tokenizer

def get_available_pretained_models():
    return AVAILABLE_MAP.keys()