import os 
import numpy as np 
import torch

def load_np_weight_opt_non_layer(folder_path, model):
    # lm_weight,
    # embed_positions
    # embed_tokens
    model_weight_template = [
        'lm_head.weight',
        'model.decoder.embed_positions.weight',
        'model.decoder.embed_tokens.weight',
        'model.decoder.final_layer_norm.bias',
        'model.decoder.final_layer_norm.weight',
        # proj in and out, may not exists
    ]
    model_weight_may_not_exists = [
        "model.decoder.project_in.weight",
        "model.decoder.project_out.weight",
    ]
    # load the weight from the file
    loaded_weight = {}
    for template in model_weight_template:
        file_name = template
        abs_path = os.path.join(folder_path, file_name)
        # load the weight
        with open(abs_path, 'rb') as f:
            weight = np.load(f)
            loaded_weight[file_name] = weight
    # handle the may not exists weight
    for template in model_weight_may_not_exists:
        file_name = template
        abs_path = os.path.join(folder_path, file_name)
        # check if the file exists
        if os.path.exists(abs_path):
            with open(abs_path, 'rb') as f:
                weight = np.load(f)
                loaded_weight[file_name] = weight
    
    # load the weight to the model
    # lm_head
    model.lm_head.weight.data = torch.from_numpy(loaded_weight['lm_head.weight'])
    # embed_positions
    model.model.decoder.embed_positions.weight.data = torch.from_numpy(loaded_weight['model.decoder.embed_positions.weight'])
    # embed_tokens
    model.model.decoder.embed_tokens.weight.data = torch.from_numpy(loaded_weight['model.decoder.embed_tokens.weight'])
    # final_layer_norm
    model.model.decoder.final_layer_norm.bias.data = torch.from_numpy(loaded_weight['model.decoder.final_layer_norm.bias'])
    model.model.decoder.final_layer_norm.weight.data = torch.from_numpy(loaded_weight['model.decoder.final_layer_norm.weight'])

    # check if the proj in and out exists
    if "model.decoder.project_in.weight" in loaded_weight:
        model.model.decoder.project_in.weight.data = torch.from_numpy(loaded_weight['model.decoder.project_in.weight'])
    if "model.decoder.project_out.weight" in loaded_weight:
        model.model.decoder.project_out.weight.data = torch.from_numpy(loaded_weight['model.decoder.project_out.weight'])
    return model
    

def load_np_weight_opt_layer(folder_path, layer_idx, opt_layer):
    model_weight_template = [
        # attention layer
        'model.decoder.layers.{}.self_attn.k_proj.weight',
        'model.decoder.layers.{}.self_attn.k_proj.bias',
        'model.decoder.layers.{}.self_attn.q_proj.weight',
        'model.decoder.layers.{}.self_attn.q_proj.bias',
        'model.decoder.layers.{}.self_attn.v_proj.weight',
        'model.decoder.layers.{}.self_attn.v_proj.bias',
        # outproj
        'model.decoder.layers.{}.self_attn.out_proj.weight',
        'model.decoder.layers.{}.self_attn.out_proj.bias',
        'model.decoder.layers.{}.self_attn_layer_norm.bias',
        'model.decoder.layers.{}.self_attn_layer_norm.weight',
        # mlp layer
        'model.decoder.layers.{}.mlp.fc1.weight',
        'model.decoder.layers.{}.mlp.fc1.bias',
        'model.decoder.layers.{}.mlp.fc2.weight',
        'model.decoder.layers.{}.mlp.fc2.bias',
        'model.decoder.layers.{}.mlp.final_layer_norm.bias',
        'model.decoder.layers.{}.mlp.final_layer_norm.weight',
    ]
    # load the weight from the file
    loaded_weight = {}
    for template in model_weight_template:
        file_name = template.format(layer_idx)
        abs_path = os.path.join(folder_path, file_name)
        # load the weight
        with open(abs_path, 'rb') as f:
            weight = np.load(f)
            loaded_weight[file_name] = weight
           
    # load the weight to the opt layer
    # attention layer
    opt_layer.self_attn.k_proj.weight.data = torch.from_numpy(loaded_weight['model.decoder.layers.{}.self_attn.k_proj.weight'.format(layer_idx)])
    opt_layer.self_attn.k_proj.bias.data = torch.from_numpy(loaded_weight['model.decoder.layers.{}.self_attn.k_proj.bias'.format(layer_idx)])
    opt_layer.self_attn.q_proj.weight.data = torch.from_numpy(loaded_weight['model.decoder.layers.{}.self_attn.q_proj.weight'.format(layer_idx)])
    opt_layer.self_attn.q_proj.bias.data = torch.from_numpy(loaded_weight['model.decoder.layers.{}.self_attn.q_proj.bias'.format(layer_idx)])
    opt_layer.self_attn.v_proj.weight.data = torch.from_numpy(loaded_weight['model.decoder.layers.{}.self_attn.v_proj.weight'.format(layer_idx)])
    opt_layer.self_attn.v_proj.bias.data = torch.from_numpy(loaded_weight['model.decoder.layers.{}.self_attn.v_proj.bias'.format(layer_idx)])
    # outproj
    opt_layer.self_attn.out_proj.weight.data = torch.from_numpy(loaded_weight['model.decoder.layers.{}.self_attn.out_proj.weight'.format(layer_idx)])
    opt_layer.self_attn.out_proj.bias.data = torch.from_numpy(loaded_weight['model.decoder.layers.{}.self_attn.out_proj.bias'.format(layer_idx)])
    opt_layer.self_attn_layer_norm.weight.data = torch.from_numpy(loaded_weight['model.decoder.layers.{}.self_attn_layer_norm.weight'.format(layer_idx)])
    opt_layer.self_attn_layer_norm.bias.data = torch.from_numpy(loaded_weight['model.decoder.layers.{}.self_attn_layer_norm.bias'.format(layer_idx)])
    # mlp layer
    opt_layer.mlp.fc1.weight.data = torch.from_numpy(loaded_weight['model.decoder.layers.{}.mlp.fc1.weight'.format(layer_idx)])
    opt_layer.mlp.fc1.bias.data = torch.from_numpy(loaded_weight['model.decoder.layers.{}.mlp.fc1.bias'.format(layer_idx)])
    opt_layer.mlp.fc2.weight.data = torch.from_numpy(loaded_weight['model.decoder.layers.{}.mlp.fc2.weight'.format(layer_idx)])
    opt_layer.mlp.fc2.bias.data = torch.from_numpy(loaded_weight['model.decoder.layers.{}.mlp.fc2.bias'.format(layer_idx)])
    opt_layer.mlp.final_layer_norm.weight.data = torch.from_numpy(loaded_weight['model.decoder.layers.{}.mlp.final_layer_norm.weight'.format(layer_idx)])
    opt_layer.mlp.final_layer_norm.bias.data = torch.from_numpy(loaded_weight['model.decoder.layers.{}.mlp.final_layer_norm.bias'.format(layer_idx)])
    return opt_layer




def load_np_weight_bloom_non_layer(folder_path, model):
    # lm_weight,
    # embed_positions
    # embed_tokens
    model_weight_template = [
        'lm_head.weight',
        'transformer.ln_f.bias',
        'transformer.ln_f.weight',
        'transformer.word_embeddings.weight',
        'transformer.word_embeddings_layernorm.bias',
        'transformer.word_embeddings_layernorm.weight'
    ]
    model_weight_may_not_exists = [
    ]
    # load the weight from the file
    loaded_weight = {}
    for template in model_weight_template:
        file_name = template
        abs_path = os.path.join(folder_path, file_name)
        # load the weight
        with open(abs_path, 'rb') as f:
            weight = np.load(f)
            loaded_weight[file_name] = weight
    # handle the may not exists weight
    for template in model_weight_may_not_exists:
        file_name = template
        abs_path = os.path.join(folder_path, file_name)
        # check if the file exists
        if os.path.exists(abs_path):
            with open(abs_path, 'rb') as f:
                weight = np.load(f)
                loaded_weight[file_name] = weight
    
    # load the weight to the model
    # lm_head
    model.lm_head.weight.data = torch.from_numpy(loaded_weight['lm_head.weight'])
    # transformer
    model.transformer.ln_f.bias.data = torch.from_numpy(loaded_weight['transformer.ln_f.bias'])
    model.transformer.ln_f.weight.data = torch.from_numpy(loaded_weight['transformer.ln_f.weight'])
    model.transformer.word_embeddings.weight.data = torch.from_numpy(loaded_weight['transformer.word_embeddings.weight'])
    model.transformer.word_embeddings_layernorm.bias.data = torch.from_numpy(loaded_weight['transformer.word_embeddings_layernorm.bias'])
    model.transformer.word_embeddings_layernorm.weight.data = torch.from_numpy(loaded_weight['transformer.word_embeddings_layernorm.weight'])
    # check if the proj in and out exists
    # pass
    return model
    

def load_np_weight_bloom_layer(folder_path, layer_idx, bloom_layer):
    model_weight_template = [
        # attention layer
        'transformer.h.{}.input_layernorm.bias',
        'transformer.h.{}.input_layernorm.weight',
        'transformer.h.{}.mlp.dense_4h_to_h.bias',
        'transformer.h.{}.mlp.dense_4h_to_h.weight',
        'transformer.h.{}.mlp.dense_h_to_4h.bias',
        'transformer.h.{}.mlp.dense_h_to_4h.weight',
        'transformer.h.{}.post_attention_layernorm.bias',
        'transformer.h.{}.post_attention_layernorm.weight',
        'transformer.h.{}.self_attention.dense.bias',
        'transformer.h.{}.self_attention.dense.weight',
        'transformer.h.{}.self_attention.query_key_value.bias',
        'transformer.h.{}.self_attention.query_key_value.weight',
    ]
    # load the weight from the file
    loaded_weight = {}
    for template in model_weight_template:
        file_name = template.format(layer_idx)
        abs_path = os.path.join(folder_path, file_name)
        # load the weight
        with open(abs_path, 'rb') as f:
            weight = np.load(f)
            loaded_weight[file_name] = weight
           
    # load the weight to the model
    # attention layer
    bloom_layer.input_layernorm.bias.data = torch.from_numpy(loaded_weight['transformer.h.{}.input_layernorm.bias'.format(layer_idx)])
    bloom_layer.input_layernorm.weight.data = torch.from_numpy(loaded_weight['transformer.h.{}.input_layernorm.weight'.format(layer_idx)])
    bloom_layer.mlp.dense_4h_to_h.bias.data = torch.from_numpy(loaded_weight['transformer.h.{}.mlp.dense_4h_to_h.bias'.format(layer_idx)])
    bloom_layer.mlp.dense_4h_to_h.weight.data = torch.from_numpy(loaded_weight['transformer.h.{}.mlp.dense_4h_to_h.weight'.format(layer_idx)])
    bloom_layer.mlp.dense_h_to_4h.bias.data = torch.from_numpy(loaded_weight['transformer.h.{}.mlp.dense_h_to_4h.bias'.format(layer_idx)])
    bloom_layer.mlp.dense_h_to_4h.weight.data = torch.from_numpy(loaded_weight['transformer.h.{}.mlp.dense_h_to_4h.weight'.format(layer_idx)])
    bloom_layer.post_attention_layernorm.bias.data = torch.from_numpy(loaded_weight['transformer.h.{}.post_attention_layernorm.bias'.format(layer_idx)])
    bloom_layer.post_attention_layernorm.weight.data = torch.from_numpy(loaded_weight['transformer.h.{}.post_attention_layernorm.weight'.format(layer_idx)])
    bloom_layer.self_attention.dense.bias.data = torch.from_numpy(loaded_weight['transformer.h.{}.self_attention.dense.bias'.format(layer_idx)])
    bloom_layer.self_attention.dense.weight.data = torch.from_numpy(loaded_weight['transformer.h.{}.self_attention.dense.weight'.format(layer_idx)])
    bloom_layer.self_attention.query_key_value.bias.data = torch.from_numpy(loaded_weight['transformer.h.{}.self_attention.query_key_value.bias'.format(layer_idx)])
    bloom_layer.self_attention.query_key_value.weight.data = torch.from_numpy(loaded_weight['transformer.h.{}.self_attention.query_key_value.weight'.format(layer_idx)])
    # check if the proj in and out exists
    # pass
    return bloom_layer

