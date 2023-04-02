import torch
from functools import reduce
from operator import mul

def get_model_size(model, unit='bytes'):
    params = list(model.parameters())
    total_params = sum(reduce(mul, p.size(), 1) for p in params)
    total_buffers = sum(reduce(mul, b.size(), 1) for b in model.buffers())
    
    if unit == 'bytes':
        size = total_params + total_buffers
        unit_str = 'bytes'
    elif unit == 'KB':
        size = (total_params + total_buffers) / 1024
        unit_str = 'KB'
    elif unit == 'MB':
        size = (total_params + total_buffers) / (1024 * 1024)
        unit_str = 'MB'
    elif unit == 'GB':
        size = (total_params + total_buffers) / (1024 * 1024 * 1024)
        unit_str = 'GB'
    else:
        raise ValueError(f"Invalid unit: {unit}")
        
    return f"Total size of the model: {size:.2f} {unit_str}"

# only calculate the tensors on cuda
def get_model_size_cuda(model, unit='bytes'):
    params = list(model.parameters())
    total_params = sum(reduce(mul, p.size(), 1) for p in params if p.is_cuda)
    total_buffers = sum(reduce(mul, b.size(), 1) for b in model.buffers() if b.is_cuda)
    
    if unit == 'bytes':
        size = total_params + total_buffers
        unit_str = 'bytes'
    elif unit == 'KB':
        size = (total_params + total_buffers) / 1024
        unit_str = 'KB'
    elif unit == 'MB':
        size = (total_params + total_buffers) / (1024 * 1024)
        unit_str = 'MB'
    elif unit == 'GB':
        size = (total_params + total_buffers) / (1024 * 1024 * 1024)
        unit_str = 'GB'
    else:
        raise ValueError(f"Invalid unit: {unit}")
        
    return f"Total size of the model: {size:.2f} {unit_str}"


# list all tensors on cuda
def list_tensors_on_cuda(model):
    for name, param in model.named_parameters():
        if param.is_cuda:
            print(name, param.size())