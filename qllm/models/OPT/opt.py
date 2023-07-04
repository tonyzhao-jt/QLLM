import torch
from transformers import (
    OPTConfig,
    OPTForCausalLM,
    AutoTokenizer
)

# according to its original paper: https://arxiv.org/pdf/2205.01068.pdf
model_cards = {
    '125m': OPTConfig(
        hidden_size=768,
        ffn_dim=3072,
        word_embed_proj_dim=768,
        num_attention_heads=12,
        num_hidden_layers=12,
    ),
    '350m': OPTConfig(
        hidden_size=1024,
        ffn_dim=4096,
        word_embed_proj_dim=512,
        num_hidden_layers=24,
        num_attention_heads=16,
    ),
    '1.3b': OPTConfig(
        hidden_size=2048,
        ffn_dim=8192,
        word_embed_proj_dim=2048,
        num_hidden_layers=24,
        num_attention_heads=32
    ),
    '2.7b': OPTConfig(
        hidden_size=2560,
        ffn_dim=10240,
        word_embed_proj_dim=2560,
        num_hidden_layers=32,
        num_attention_heads=32
    ),
    '6.7b': OPTConfig(
        hidden_size=4096,
        ffn_dim=16384,
        word_embed_proj_dim=4096,
        num_hidden_layers=32,
        num_attention_heads=32
    ),
    '13b': OPTConfig(
        hidden_size=5120,
        ffn_dim=20480,
        word_embed_proj_dim=5120,
        num_hidden_layers=40,
        num_attention_heads=40,
    ),
    '30b': OPTConfig(
        hidden_size=7168,
        ffn_dim=28672,
        word_embed_proj_dim=7168,
        num_hidden_layers=48,
        num_attention_heads=56,
    ),
    '66b': OPTConfig(
        hidden_size=9216,
        ffn_dim=36864,
        word_embed_proj_dim=9216,
        num_hidden_layers=64,
        num_attention_heads=72,
    ),
    '175b': OPTConfig(
        hidden_size=12288,
        ffn_dim=49152,
        word_embed_proj_dim=12288,
        num_hidden_layers=96,
        num_attention_heads=96,
    ),
}

def get_empty_model(model_size:str='125M'):
    assert model_size in model_cards, f"model size {model_size} not available"
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    config = model_cards[model_size]
    # model = LlamaModel(config)
    model = OPTForCausalLM(config)
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")
    return model, tokenizer

def get_available_models():
    return model_cards.keys()

AVAILABLE_MAP = {
    "facebook/opt-125m": "https://huggingface.co/facebook/opt-125m/blob/main/config.json",
    "facebook/opt-350m": "https://huggingface.co/facebook/opt-350m/blob/main/config.json",
    "facebook/opt-1.3b": "https://huggingface.co/facebook/opt-1.3b/blob/main/config.json",
    "facebook/opt-2.7b": "https://huggingface.co/facebook/opt-2.7b/blob/main/config.json",
    "facebook/opt-6.7b": "https://huggingface.co/facebook/opt-6.7b/blob/main/config.json",
    "facebook/opt-13b": "https://huggingface.co/facebook/opt-13b/blob/main/config.json",
    "facebook/opt-30b": "https://huggingface.co/facebook/opt-30b/blob/main/config.json",
    "facebook/opt-66b": "https://huggingface.co/facebook/opt-66b/blob/main/config.json",
}

def load_pretained_model_from_net(repo_name, dtype=torch.float16, cache_dir=None):
    assert repo_name in AVAILABLE_MAP, f"model {repo_name} not available in repo"
    if cache_dir is not None:
        model = OPTForCausalLM.from_pretrained(repo_name, torch_dtype=dtype, cache_dir=cache_dir)
    else:
        model = OPTForCausalLM.from_pretrained(repo_name, torch_dtype=dtype)
    tokenizer = AutoTokenizer.from_pretrained(repo_name)
    return model, tokenizer

def get_model_size_key(model_size):
    AVAILABLE_MAP_keys = list(AVAILABLE_MAP.keys())
    for key in AVAILABLE_MAP_keys:
        if str(model_size) in key:
            return key
    return None 

def load_pretrained_from_size(model_size, dtype=torch.float16, cache_dir=None):
    AVAILABLE_MAP_keys = list(AVAILABLE_MAP.keys())
    key = get_model_size_key(model_size)
    if key is not None:
        return load_pretained_model_from_net(key, dtype=dtype, cache_dir=cache_dir), key
    raise Exception(f"model size {model_size} not available")

def get_available_pretained_models():
    return AVAILABLE_MAP.keys()