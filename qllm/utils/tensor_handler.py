import torch 
import math
def object_to_tensor(obj):
    if isinstance(obj, torch.Tensor):
        return obj
    elif isinstance(obj, int):
        return torch.tensor(obj, dtype=torch.long)
    elif isinstance(obj, float):
        return torch.tensor(obj, dtype=torch.float)
    elif isinstance(obj, bool):
        return torch.tensor(obj, dtype=torch.bool)
    elif obj == None:
        return torch.tensor(math.nan)
    elif isinstance(obj, list):
        return [object_to_tensor(o) for o in obj]
    elif isinstance(obj, tuple):
        return tuple(object_to_tensor(o) for o in obj)
    elif isinstance(obj, dict):
        return {k: object_to_tensor(v) for k, v in obj.items()}

def return_none_if_nan(tensor):
    if isinstance(tensor, torch.Tensor):
        if torch.all(torch.isnan(tensor)):
            return None
        else:
            return tensor
    elif isinstance(tensor, list):
        return [return_none_if_nan(t) for t in tensor]
    elif isinstance(tensor, tuple):
        return tuple(return_none_if_nan(t) for t in tensor)
    elif isinstance(tensor, dict):
        return {k: return_none_if_nan(v) for k, v in tensor.items()}

