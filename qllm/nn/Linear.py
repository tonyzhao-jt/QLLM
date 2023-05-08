import torch 
import torch.nn as nn 
import torch.nn.functional as F
import qllm.tp.utils as qllm_tp_utils
import qllm.tp as tp
from lptorch import quantize_linear_module_with_bit, quantize_one_linear_module, ForwardTokenizer, AdaQTPConfig

class Linear1D(nn.Module):
    def __init__(self, new_li,
                 tp_type: str = 'ROW',
                 broadcast: bool = False,
                 *args, **kwargs):
        super().__init__()
        self.li = new_li
        self.tp_type = tp_type
        self.broadcast = broadcast

    @torch.no_grad()
    def from_linear(li: nn.Linear, TP_TYPE="ROW", broadcast=False, kernel_bit=16, caliber=None):
        tp_config = qllm_tp_utils.get_tp_configs()
        global_rank = tp_config['global_rank']
        tp_index = tp_config['tp_index']
        split_k = tp_config['split_k']
        group = tp_config['group']

        if group is None:
            return li 
        
        if TP_TYPE == 'ROW':
            tp_config= AdaQTPConfig(split_k=split_k, global_rank=global_rank, tp_index=tp_index, split_type='ROW', comm_group=group)
        else:
            tp_config = AdaQTPConfig(split_k=split_k, global_rank=global_rank, tp_index=tp_index, split_type='COLUMN', comm_group=group)

        new_li = quantize_one_linear_module(li, kernel_bit=kernel_bit, caliber=caliber, tp_config=tp_config)
        return Linear1D(new_li, TP_TYPE, broadcast)
    
    # TODO: because i am lazy. i don't want to implement columnwise cond here.
    @torch.no_grad()
    def sole_forward(self, input_: torch.Tensor) -> torch.Tensor:
        if self.tp_type == 'COLUMN' and self.broadcast:
            # broadcast in tp
            tp_config = qllm_tp_utils.get_tp_configs()
            global_rank = tp_config['global_rank']; tp_index = tp_config['tp_index']; group = tp_config['group']
            tp._broad_cast(input_, global_rank, tp_index, group)
        return self.li(input_)
    
    @torch.no_grad()
    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        output_parallel = self.sole_forward(input_)
        # reduce input.
        group = qllm_tp_utils.get_tp_group()
        if self.tp_type == 'ROW':
            output_parallel = tp._all_reduce_sum(output_parallel, group)
        return output_parallel

