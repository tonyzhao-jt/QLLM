
from qllm.models import opt 
from qllm.utils import get_model_size_cuda, list_tensors_on_cuda
from qllm.utils import (
    check_model_weights_dtype,
    to_device_recursive, to_device_recursive_except, 
    to_dtype_recursive, to_dtype_recursive_except, to_dtype_except_linear_layer, to_half_for_modules,
    get_iter_variable_size
)
from qllm.models.OPT import OPTForCausalLMSeq
import lptorch
import torch
import copy

from time import perf_counter

if __name__ == '__main__':
    opt_125M, tokenizer = opt.load_pretained_model_from_net('facebook/opt-1.3b', dtype=torch.float32)
    batched_ids = tokenizer.batch_encode_plus(16 * [ "The quick brown fox jumps over the lazy dog. It's a beautiful day outside, the sun is shining and the birds are chirping. I feel like going for a"], \
                                              max_length=512, padding=True, return_tensors="pt")

    cnt = 5
    opt_125M.generate(**batched_ids)
    start = perf_counter()
    for i in range(cnt):
        opt_125M.generate(**batched_ids, use_cache=True, min_new_tokens=49, max_new_tokens=50)
    end = perf_counter()
    print("opt_125M.generate", (end - start) / cnt)

    # cuda case
    opt_125M.cuda()
    batched_ids = to_device_recursive(dict(batched_ids), torch.device("cuda:0"))
    opt_125M.generate(**batched_ids)
    start = perf_counter()
    for i in range(cnt):
        opt_125M.generate(**batched_ids, use_cache=True, min_new_tokens=49, max_new_tokens=50)
        torch.cuda.synchronize()
    end = perf_counter()
    print("opt_125M.generate cuda", (end - start) / cnt)
    

    