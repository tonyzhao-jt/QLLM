import torch
def get_iter_variable_size(obj, unit='B'):
    def get_iter_variable_inner(obj):
        total_size = 0
        if isinstance(obj, (tuple, list)):
            for item in obj:
                total_size += get_iter_variable_inner(item)
        elif isinstance(obj, dict):
            for _, item in obj.items():
                total_size += get_iter_variable_inner(item)
        elif isinstance(obj, torch.Tensor):
            tensor = obj
            total_size += tensor.element_size() * tensor.nelement()
        return total_size
    total_size = get_iter_variable_inner(obj)
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

