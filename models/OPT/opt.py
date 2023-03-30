import torch
from transformers import (
    OPTConfig,
    OPTForCausalLM
)
# according to its original paper: https://arxiv.org/pdf/2205.01068.pdf
model_cards = {
    '125M': OPTConfig(
        hidden_size=768,
        num_attention_heads=12,
        num_hidden_layers=24,
    ),
    '350M': OPTConfig(
        hidden_size=1024,
        num_hidden_layers=24,
        num_attention_heads=16,
    ),
    '1.3b': OPTConfig(
        hidden_size=2048,
        num_hidden_layers=24,
        num_attention_heads=32
    ),
    '2.7b': OPTConfig(
        hidden_size=2560,
        num_hidden_layers=32,
        num_attention_heads=32
    ),
    '6.7b': OPTConfig(
        hidden_size=4096,
        num_hidden_layers=32,
        num_attention_heads=32
    ),
    '13b': OPTConfig(
        hidden_size=5120,
        num_hidden_layers=40,
        num_attention_heads=40,
    ),
    '30b': OPTConfig(
        hidden_size=7168,
        num_hidden_layers=48,
        num_attention_heads=56,
    ),
    '66b': OPTConfig(
        hidden_size=9216,
        num_hidden_layers=64,
        num_attention_heads=72,
    ),
    '175b': OPTConfig(
        hidden_size=12288,
        num_hidden_layers=96,
        num_attention_heads=96,
    ),
}

def get_empty_model(model_size:str='125M'):
    assert model_size in model_cards, f"model size {model_size}b not available"
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    config = model_cards[model_size]
    # model = LlamaModel(config)
    model = OPTForCausalLM(config)
    return model


AVAILABLE_MAP = {
    "facebook/opt-125m": "https://huggingface.co/facebook/opt-125m/blob/main/config.json",
    "facebook/opt-350m": "https://huggingface.co/facebook/opt-350m/blob/main/config.json",
    "facebook/opt-1.3b": "https://huggingface.co/facebook/opt-1.3b/blob/main/config.json",
    "facebook/opt-2.7b": "https://huggingface.co/facebook/opt-2.7b/blob/main/config.json",
    "facebook/opt-6.7b": "https://huggingface.co/facebook/opt-6.7b/blob/main/config.json",
    "facebook/opt-13b": "https://huggingface.co/facebook/opt-13b/blob/main/config.json",
}

def load_pretained_model_from_net(repo_name):
    assert repo_name in AVAILABLE_MAP, f"model {repo_name} not available in repo"
    model = OPTForCausalLM.from_pretrained(repo_name)
    return model