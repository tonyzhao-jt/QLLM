import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import LlamaModel, LlamaConfig
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
    assert model_size in model_cards, f"model size {model_size}b not available"
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    config = model_cards[model_size]
    # model = LlamaModel(config)
    model = LlamaForCausalLM(config)
    return model