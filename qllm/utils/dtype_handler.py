import torch
import torch.nn as nn
from lptorch import AdaQLinear
# recusively to dtype
def to_dtype_recursive(obj, dtype):
    if isinstance(obj, torch.Tensor):
        return obj.to(dtype)
    elif isinstance(obj, list) or isinstance(obj, tuple):
        new_obj = [to_dtype_recursive(item, dtype) for item in obj]
        return type(obj)(new_obj)
    elif isinstance(obj, dict):
        new_obj = {k: to_dtype_recursive(v, dtype) for k, v in obj.items()}
        return new_obj
    else:
        return obj

def to_dtype_recursive_except(obj, dtype, except_list):
    if isinstance(obj, torch.Tensor):
        return obj.to(dtype)
    elif isinstance(obj, list) or isinstance(obj, tuple):
        new_obj = [to_dtype_recursive_except(item, dtype, except_list) for item in obj]
        return type(obj)(new_obj)
    elif isinstance(obj, dict):
        new_obj = {k: to_dtype_recursive_except(v, dtype, except_list) for k, v in obj.items()}
        return new_obj
    elif isinstance(obj, nn.Module):
        if obj not in except_list:
            return obj.to(dtype)
        else:
            return obj
    else:
        return obj

def to_dtype_except_linear_layer(model, dtype):
    for name, child in model.named_children():
        if not isinstance(child, (nn.Linear, AdaQLinear)):
            new_mod = child.to(dtype)
            setattr(model, name, new_mod)
        else:
            continue

def to_half_for_modules(model, modules=(nn.LayerNorm, nn.Embedding, nn.LayerNorm)):
    def inner_half(module):
        for name, child in module.named_children():
            if isinstance(child, modules):
                new_mod = child.to(torch.float16)
                setattr(module, name, new_mod)
            else:
                inner_half(child)
    inner_half(model)