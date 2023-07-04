from .LLaMa import llama
from .OPT import opt, OPTForCausalLMSeq, OPTDecoderLayerSharded
from .BLOOM import bloom, BloomForCausalLMSeq, BloomBlockSharded
from transformers import (
    BloomConfig,
    OPTConfig,
    AutoTokenizer,
)
import torch


def create_model_config(model_name, model_size):
    if model_name == 'opt':
        model_cards = opt.model_cards
        assert model_size in model_cards, f"model size {model_size} is not in model cards {model_cards.keys()}"
        config = model_cards[model_size]
    elif model_name == 'bloom':
        model_cards = bloom.model_cards
        assert model_size in model_cards, f"model size {model_size} is not in model cards {model_cards.keys()}"
        config = model_cards[model_size]
    return config

def create_empty_model(model_name, model_size, torch_dtype=torch.float16):
    if model_name == 'opt':
        model_cards = opt.model_cards
        assert model_size in model_cards, f"model size {model_size} is not in model cards {model_cards.keys()}"
        config = model_cards[model_size]
        loaded_llm_cpu = OPTForCausalLMSeq._from_config(config, torch_dtype=torch_dtype)
    elif model_name == 'bloom':
        model_cards = bloom.model_cards
        assert model_size in model_cards, f"model size {model_size} is not in model cards {model_cards.keys()}"
        config = model_cards[model_size]
        loaded_llm_cpu = BloomForCausalLMSeq._from_config(config, torch_dtype=torch_dtype)
    loaded_llm_cpu.eval()
    return loaded_llm_cpu

def create_empty_decoder(model_name, model_size):
    h1, h2 = 0, 0
    if model_name == 'opt':
        model_cards = opt.model_cards
        assert model_size in model_cards, f"model size {model_size} is not in model cards {model_cards.keys()}"
        config = model_cards[model_size]
        decoder_layer = OPTDecoderLayerSharded(config)
        h1 = config.hidden_size
        h2 = config.ffn_dim

    elif model_name == 'bloom':
        model_cards = bloom.model_cards
        assert model_size in model_cards, f"model size {model_size} is not in model cards {model_cards.keys()}"
        config = model_cards[model_size]
        decoder_layer = BloomBlockSharded(config)
        h1 = config.hidden_size
        h2 = h1 * 4
    decoder_layer.eval()
    return decoder_layer, (h1, h2), config

def bare_load_pretrained_from_size(model_name, model_size, torch_dtype=torch.float16):
    if model_name == 'opt':
        (loaded_llm_cpu, tokenizer), key = opt.load_pretrained_from_size(model_size, dtype=torch_dtype)
    elif model_name == 'bloom':
        (loaded_llm_cpu, tokenizer), key = bloom.load_pretrained_from_size(model_size, dtype=torch_dtype)
    loaded_llm_cpu.eval()
    return loaded_llm_cpu, tokenizer, key

def load_qllm_weight(QLLM_LM, model_name, model_size, target_storage_folder, torch_dtype, key=None):
    import os 
    if target_storage_folder is not None:
        path = os.path.join(target_storage_folder, f"{model_name}_{model_size}")
        return QLLM_LM.from_pretrained(path, torch_dtype=torch_dtype)
    assert key, "key must be provided if target_storage_folder is None"
    return QLLM_LM.from_pretrained(key, torch_dtype=torch_dtype)

def qllm_load_pretrained_from_size(model_name, model_size, torch_dtype=torch.float16, target_storage_folder=None):
    if model_name == 'opt':
        key = opt.get_model_size_key(model_size)
        QLLM_LM = OPTForCausalLMSeq
    elif model_name == 'bloom':
        key = bloom.get_model_size_key(model_size)
        QLLM_LM = BloomForCausalLMSeq
    tokenizer = AutoTokenizer.from_pretrained(key)
    loaded_llm_cpu = load_qllm_weight(QLLM_LM, model_name, model_size, target_storage_folder, torch_dtype, key=key)
    loaded_llm_cpu.eval()
    return loaded_llm_cpu, tokenizer, key

def return_config_name(model_config):
    if isinstance(model_config, OPTConfig):
        return 'opt'
    elif isinstance(model_config, BloomConfig):
        return 'bloom'

def return_h1_h2(model_config):
    if isinstance(model_config, OPTConfig):
        return model_config.hidden_size, model_config.ffn_dim
    elif isinstance(model_config, BloomConfig):
        return model_config.hidden_size, model_config.hidden_size * 4

def get_kv_size(decoder_instance):
    if isinstance(decoder_instance, OPTDecoderLayerSharded):
        return decoder_instance.kv_size
    elif isinstance(decoder_instance, BloomBlockSharded):
        return decoder_instance.kv_size