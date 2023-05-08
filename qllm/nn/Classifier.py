import torch 
import torch.nn as nn 
import torch.nn.functional as F
import qllm.tp.utils as qllm_tp_utils
import qllm.tp as tp
from lptorch import quantize_linear_module_with_bit, quantize_one_linear_module, ForwardTokenizer, AdaQTPConfig

# The last linear layer in model.
# The previous output could be paralleled or not, typically, parallel
class Classifier1D(nn.Module):
    def __init__(self, new_li,
                 broadcast: bool = False,
                 *args, **kwargs):
        super().__init__()
        self.li = new_li
        self.broadcast = broadcast

    @torch.no_grad()
    def from_classi(li: nn.Linear, broadcast=False, kernel_bit=16, caliber=None):
        tp_config = qllm_tp_utils.get_tp_configs()
        global_rank = tp_config['global_rank']
        tp_index = tp_config['tp_index']
        split_k = tp_config['split_k']
        group = tp_config['group']

        if group is None:
            return li 
        # use column linear partition.
        tp_config = AdaQTPConfig(split_k=split_k, global_rank=global_rank, tp_index=tp_index, split_type='COLUMN', comm_group=group)
        new_li = quantize_one_linear_module(li, kernel_bit=kernel_bit, caliber=caliber, tp_config=tp_config)
        return Classifier1D(new_li, broadcast)
    
    # TODO: because i am lazy. i don't want to implement columnwise cond here.
    @torch.no_grad()
    def sole_forward(self, input_: torch.Tensor) -> torch.Tensor:
        if self.broadcast:
            # broadcast in tp
            tp_config = qllm_tp_utils.get_tp_configs()
            global_rank = tp_config['global_rank']; tp_index = tp_config['tp_index']; group = tp_config['group']
            tp._broad_cast(input_, global_rank, tp_index, group)
        return self.li(input_)
    
    @torch.no_grad()
    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        output_parallel = self.sole_forward(input_)
        # reduce input.
        tp_config = qllm_tp_utils.get_tp_configs()
        split_k = tp_config['split_k']; tp_index = tp_config['tp_index']; group = tp_config['group']
        output = tp._all_gather(output_parallel, tp_index, split_k, group)
        return output

