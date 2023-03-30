import torch
from transformers import (
    BloomConfig,
    BloomForCausalLM
)
# according to its original paper: https://arxiv.org/pdf/2211.05100.pdf, paper 21
model_cards = {
    '560M': BloomConfig(
        num_hidden_layers=24,
        hidden_size=1024,
        num_attention_heads=16,
    ),
    '1b1': BloomConfig(
        num_hidden_layers=24,
        hidden_size=1536,
        num_attention_heads=16,
    ),
    '1b7': BloomConfig(
        num_hidden_layers=24,
        hidden_size=2048,
        num_attention_heads=16,
    ),
    '3b': BloomConfig(
        num_hidden_layers=30,
        hidden_size=2560,
        num_attention_heads=32,
    ),
    '7b1': BloomConfig(
        num_hidden_layers=30,
        hidden_size=4096,
        num_attention_heads=32,
    ),
    '176b': BloomConfig(
        num_hidden_layers=70,
        hidden_size=14336,
        num_attention_heads=112,
    )
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
    model = BloomForCausalLM(config)
    return model


AVAILABLE_MAP = {
    "bigscience/bloom": "https://huggingface.co/bigscience/bloom/resolve/main/config.json",
    "bigscience/bloom-560m": "https://huggingface.co/bigscience/bloom-560m/blob/main/config.json",
    "bigscience/bloom-1b1": "https://huggingface.co/bigscience/bloom-1b1/blob/main/config.json",
    "bigscience/bloom-1b7": "https://huggingface.co/bigscience/bloom-1b7/blob/main/config.json",
    "bigscience/bloom-3b": "https://huggingface.co/bigscience/bloom-3b/blob/main/config.json",
    "bigscience/bloom-7b1": "https://huggingface.co/bigscience/bloom-7b1/blob/main/config.json",
}

def load_pretained_model_from_net(repo_name):
    assert repo_name in AVAILABLE_MAP, f"model {repo_name} not available in repo"
    model = BloomForCausalLM.from_pretrained(repo_name)
    return model