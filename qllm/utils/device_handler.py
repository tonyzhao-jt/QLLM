import torch
import torch.nn as nn 
def to_device_recursive(obj, device):
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, list) or isinstance(obj, tuple):
        new_obj = [to_device_recursive(item, device) for item in obj]
        return type(obj)(new_obj)
    elif isinstance(obj, dict):
        new_dict = {}
        for k, v in obj.items():
            new_dict[k] = to_device_recursive(v, device)
        return new_dict
    else:
        return obj

# to device except for some module/layers
def to_device_recursive_except(obj, device, except_list):
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, list) or isinstance(obj, tuple):
        new_obj = [to_device_recursive_except(item, device, except_list) for item in obj]
        return type(obj)(new_obj)
    elif isinstance(obj, dict):
        new_obj = {k: to_device_recursive_except(v, device, except_list) for k, v in obj.items()}
        return new_obj
    elif isinstance(obj, nn.Module):
        if obj not in except_list:
            return obj.to(device)
        else:
            return obj
    else:
        return obj