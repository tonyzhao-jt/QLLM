import torch 
import torch.nn as nn 
import torch.nn.functional as F
import qllm.tp.utils as qllm_tp_utils
import qllm.tp as tp


class Embedding1D(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, padding_idx=None, *args, **kwargs):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        device = kwargs.get('device', torch.device('cpu'))
        dtype = kwargs.get('dtype', torch.float32)

        self.vocab_start_index = kwargs.get('vocab_start_index', 0)
        self.vocab_end_index = kwargs.get('vocab_end_index', self.num_embeddings)

        self.weight = nn.Parameter(torch.empty((self.num_embeddings, self.embedding_dim), device=device, dtype=dtype))
    
    def reset_parameters(self, embed:nn.Embedding) -> None:
        self.weight.data.copy_(embed.weight.data[self.vocab_start_index:self.vocab_end_index])
        self._fill_padding_idx_with_zero()
    
    def _fill_padding_idx_with_zero(self) -> None:
        if self.padding_idx is not None and \
                self.padding_idx >= self.vocab_start_index and self.padding_idx < self.vocab_end_index:
            with torch.no_grad():
                self.weight[self.padding_idx - self.vocab_start_index].fill_(0)

    @torch.no_grad()
    def from_embed(embed: nn.Embedding):
        tp_config = qllm_tp_utils.get_tp_configs()
        global_rank = tp_config['global_rank']
        tp_index = tp_config['tp_index']
        split_k = tp_config['split_k']
        group = tp_config['group']

        num_embeddings = embed.num_embeddings
        embedding_dim = embed.embedding_dim
        padding_idx = embed.padding_idx

        if group is None:
            return embed # do nothing
        else:
            # do partition. since embedding is small, we load it first then partition directly
            num_embeddings_per_partition = tp.divide(num_embeddings, split_k)
            vocab_start_index = tp_index * num_embeddings_per_partition
            vocab_end_index = vocab_start_index + num_embeddings_per_partition
            device = embed.weight.device
            dtype = embed.weight.dtype
            # create a new embedding
            embed_1d = Embedding1D(num_embeddings_per_partition, embedding_dim, padding_idx, device=device, dtype=dtype, \
                                   vocab_start_index=vocab_start_index, vocab_end_index=vocab_end_index)
            embed_1d.reset_parameters(embed) # copy data from mo
            return embed_1d
    
    @torch.no_grad()
    def sole_forward(self, input_: torch.Tensor) -> torch.Tensor:
        input_mask = (input_ < self.vocab_start_index) | (input_ >= self.vocab_end_index)
        # Mask the input.
        masked_input = input_.clone() - self.vocab_start_index
        masked_input[input_mask] = 0
        output_parallel = F.embedding(masked_input, self.weight, self.padding_idx)
        # mask the output
        output_parallel[input_mask] = 0
        return output_parallel
    
    @torch.no_grad()
    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        input_mask = (input_ < self.vocab_start_index) | (input_ >= self.vocab_end_index)
        # Mask the input.
        masked_input = input_.clone() - self.vocab_start_index
        masked_input[input_mask] = 0
        output_parallel = F.embedding(masked_input, self.weight, self.padding_idx)
        output_parallel[input_mask] = 0
        # reduce input.
        group = qllm_tp_utils.get_tp_group()
        tp._all_reduce_sum(output_parallel, group)
        return output_parallel

