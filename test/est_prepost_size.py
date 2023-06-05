from qllm.models import create_empty_model
import torch 
import os 
from qpipe.partitioner.helper import create_mem_estimator
from qllm.models.BLOOM.bloom import model_cards
# from qllm.models.OPT.opt import model_cards
from qllm.utils import get_model_size_cuda
# set env
os.environ['SET_DECODERS_META'] = "1"
os.environ['PERF_MODE'] = "1"
b = 32
s = 512
n = 80
# model_name= 'opt'
# model_size = "30b"
model_name= 'bloom'
model_size = "176b"
config = model_cards[model_size]
mem_estimator = create_mem_estimator(b, s, n, config)
print("mem estimator loaded")
print(mem_estimator.calculate_prepost_mem(unit='MB')[0])
loaded_llm_cpu = create_empty_model(model_name, model_size)
print("model loaded")
model_pre_and_post = loaded_llm_cpu._pure_pre_and_post()
print("cpu model loaded")
model_pre_and_post = model_pre_and_post.cuda()
print(get_model_size_cuda(model_pre_and_post.transformer, 'MB'), get_model_size_cuda(model_pre_and_post.lm_head, 'MB'))
# print(get_model_size_cuda(model_pre_and_post.model.decoder, 'MB'), get_model_size_cuda(model_pre_and_post.lm_head, 'MB'))
print("Max memory: ", torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024, "GB")
import pdb; pdb.set_trace()
