import torch
from transformers import (
    BloomConfig,
    BloomForCausalLM,
    AutoTokenizer
)
from transformers import BloomTokenizerFast
# according to its original paper: https://arxiv.org/pdf/2211.05100.pdf, paper 21
# for bloom, by default, f2=4f1
model_cards = {
    '560m': BloomConfig(
        n_layer=24,
        n_embed=1024,
        num_attention_heads=16,
    ),
    '1b1': BloomConfig(
        n_layer=24,
        n_embed=1536,
        num_attention_heads=16,
    ),
    '1b7': BloomConfig(
        n_layer=24,
        n_embed=2048,
        num_attention_heads=16,
    ),
    '3b': BloomConfig(
        n_layer=30,
        n_embed=2560,
        num_attention_heads=32,
    ),
    '7b1': BloomConfig(
        n_layer=30,
        n_embed=4096,
        num_attention_heads=32,
    ),
    '176b': BloomConfig(
        n_layer=70,
        n_embed=14336,
        num_attention_heads=112,
    )
}

def get_empty_model(model_size:str='125m'):
    assert model_size in model_cards, f"model size {model_size} not available"
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    config = model_cards[model_size]
    # model = LlamaModel(config)
    model = BloomForCausalLM(config)
    # tokenizer = BloomTokenizerFast.from_pretrained("bigscience/bloom")
    tokenizer = AutoTokenizer.from_pretrained("facebook/bloom")
    return model, tokenizer

def get_available_models():
    return model_cards.keys()

AVAILABLE_MAP = {
    "bigscience/bloom": "https://huggingface.co/bigscience/bloom/resolve/main/config.json",
    "bigscience/bloom-560m": "https://huggingface.co/bigscience/bloom-560m/blob/main/config.json",
    "bigscience/bloom-1b1": "https://huggingface.co/bigscience/bloom-1b1/blob/main/config.json",
    "bigscience/bloom-1b7": "https://huggingface.co/bi gscience/bloom-1b7/blob/main/config.json",
    "bigscience/bloom-3b": "https://huggingface.co/bigscience/bloom-3b/blob/main/config.json",
    "bigscience/bloom-7b1": "https://huggingface.co/bigscience/bloom-7b1/blob/main/config.json",
}

def load_pretained_model_from_net(repo_name, dtype=torch.float16, cache_dir=None):
    assert repo_name in AVAILABLE_MAP, f"model {repo_name} not available in repo"
    if cache_dir is None:
        model = BloomForCausalLM.from_pretrained(repo_name, torch_dtype=dtype, cache_dir=cache_dir)
    else:
        model = BloomForCausalLM.from_pretrained(repo_name, torch_dtype=dtype)
    tokenizer = AutoTokenizer.from_pretrained(repo_name)
    return model, tokenizer

def get_model_size_key(model_size):
    AVAILABLE_MAP_keys = list(AVAILABLE_MAP.keys())
    for key in AVAILABLE_MAP_keys:
        if str(model_size) in key:
            return key
        if model_size == '176b':
            return 'bigscience/bloom'
    return None 

def load_pretrained_from_size(model_size, dtype=torch.float16, cache_dir=None):
    AVAILABLE_MAP_keys = list(AVAILABLE_MAP.keys())
    key = get_model_size_key(model_size)
    if key is not None:
        return load_pretained_model_from_net(key, dtype=dtype, cache_dir=cache_dir), key 
    raise Exception(f"model size {model_size} not available")

def get_available_pretained_models():
    return AVAILABLE_MAP.keys()