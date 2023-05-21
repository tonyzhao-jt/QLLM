from .LLaMa import llama
from .OPT import opt, OPTForCausalLMSeq, OPTDecoderLayerSharded
from .BLOOM import bloom, BloomForCausalLMSeq, BloomBlockSharded
from transformers import (
    BloomConfig,
    OPTConfig,
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
    return decoder_layer, (h1, h2), config


def return_config_name(model_config):
    if isinstance(model_config, OPTConfig):
        return 'opt'
    elif isinstance(model_config, BloomConfig):
        return 'bloom'

def get_kv_size(decoder_instance):
    if isinstance(decoder_instance, OPTDecoderLayerSharded):
        return decoder_instance.kv_size
    elif isinstance(decoder_instance, BloomBlockSharded):
        return decoder_instance.kv_size