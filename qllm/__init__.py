from .qllm_model import QLLM
from . import dataloader
from . import models
from . import utils

from .custom_generate import SequentialGenerate

available_bits = [2, 4, 8, '8:tc', '8:tc-li', 16]
def get_available_bits():
    return available_bits