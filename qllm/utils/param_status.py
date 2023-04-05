import torch

import torch


def get_tensor_memory_size(tensor: torch.Tensor) -> int:
    return tensor.element_size() * tensor.nelement()


def get_iter_variable_size(obj, unit='B'):
    total_size = 0

    if isinstance(obj, (tuple, list)):
        for item in obj:
            total_size += get_iter_variable_size(item, unit)
    elif isinstance(obj, dict):
        for _, item in obj.items():
            total_size += get_iter_variable_size(item, unit)
    elif isinstance(obj, torch.Tensor):
        total_size += get_tensor_memory_size(obj)

    if unit == 'B':
        return total_size
    elif unit == 'KB':
        return total_size / 1024
    elif unit == 'MB':
        return total_size / (1024 * 1024)
    elif unit == 'GB':
        return total_size / (1024 * 1024 * 1024)
    else:
        raise ValueError(f"Invalid unit '{unit}', choose from 'B', 'KB', or 'MB'.")

