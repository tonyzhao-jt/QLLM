import torch.nn as nn 
import torch
# import dist
import torch.distributed as dist
from transformers import (
    BloomConfig,
    BloomForCausalLM,
    AutoTokenizer
)
# output decorator
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    CausalLMOutputWithCrossAttentions,
    BaseModelOutputWithPastAndCrossAttentions
)
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, LayerNorm, MSELoss
from accelerate import init_empty_weights
import copy
from torch.nn import functional as F
from transformers.models.bloom.modeling_bloom import (
    BloomForCausalLM,
    BloomModel,
    BloomBlock,
    BloomGelu
)
from transformers.utils import logging
logger = logging.get_logger(__name__)

import qllm
import qllm.tp as tp 
import qllm.tp.utils as qllm_tp_utils
import qllm.nn as qllm_nn
import lptorch
from lptorch import quantize_linear_module_with_bit, quantize_one_linear_module, ForwardTokenizer, AdaQTPConfig
from lptorch.utils import is_tensorcore_int8_available, get_capability
cap = get_capability()

import math

def dropout_add(x: torch.Tensor, residual: torch.Tensor, prob: float, training: bool) -> torch.Tensor:
    """
    Dropout add function

    Args:
        x (`torch.tensor`, *required*):
            input tensor
        residual (`torch.tensor`, *required*):
            esidual tensor
        prob (`float`, *required*):
            dropout probability
        training (`bool`, *required*):
            training mode
    """
    out = F.dropout(x, p=prob, training=training)
    out = residual + out
    return out


class BloomMLP(nn.Module):
    def __init__(self, config: BloomConfig):
        super().__init__()
        hidden_size = config.hidden_size

        self.pretraining_tp = config.pretraining_tp
        self.slow_but_exact = config.slow_but_exact
        self.dense_h_to_4h = nn.Linear(hidden_size, 4 * hidden_size)
        self.gelu_impl = BloomGelu()
        self.dense_4h_to_h = nn.Linear(4 * hidden_size, hidden_size)
        self.hidden_dropout = config.hidden_dropout

        self.enable_tp = False
        self.tp_comm_group = None
    

    def register_tp(self, bit, caliber, broadcast=True):
        tp_config = qllm_tp_utils.get_tp_configs()
        global_rank = tp_config['global_rank']
        tp_index = tp_config['tp_index']
        split_k = tp_config['split_k']
        group = tp_config['group']
        broadcast_group = tp_config['broadcast_group']
        tp_config_COL = AdaQTPConfig(split_k=split_k, global_rank=global_rank, tp_index=tp_index, split_type='COLUMN', comm_group=group)
        tp_config_ROW = AdaQTPConfig(split_k=split_k, global_rank=global_rank, tp_index=tp_index, split_type='ROW', comm_group=group)
        # first column then row
        self.dense_h_to_4h = quantize_one_linear_module(self.dense_h_to_4h, kernel_bit=bit, caliber=caliber, tp_config=tp_config_COL)
        self.dense_4h_to_h = quantize_one_linear_module(self.dense_4h_to_h, kernel_bit=bit, caliber=caliber, tp_config=tp_config_ROW)
        # enable tp
        self.enable_tp = True
        # self.tp_comm_group = group
        self.global_rank = global_rank
        self.tp_index = tp_index

        # partition along the head dim
        self.broadcast = broadcast_group


    def forward(self, hidden_states: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        if self.enable_tp and self.broadcast:
            group = qllm_tp_utils.get_tp_group()
            tp._broad_cast(hidden_states, self.global_rank, self.tp_index, group) # broadcast hidden states
            
        hidden_states = self.gelu_impl(self.dense_h_to_4h(hidden_states))

        if self.pretraining_tp > 1 and self.slow_but_exact:
            intermediate_output = torch.zeros_like(residual)
            slices = self.dense_4h_to_h.weight.shape[-1] / self.pretraining_tp
            for i in range(self.pretraining_tp):
                intermediate_output = intermediate_output + F.linear(
                    hidden_states[:, :, int(i * slices) : int((i + 1) * slices)],
                    self.dense_4h_to_h.weight[:, int(i * slices) : int((i + 1) * slices)],
                )
        else:
            intermediate_output = self.dense_4h_to_h(hidden_states)
        
        if self.enable_tp:
            # gather result
            group = qllm_tp_utils.get_tp_group()
            intermediate_output = tp._all_reduce_sum(intermediate_output, group)

        output = dropout_add(intermediate_output, residual, self.hidden_dropout, self.training)

        return output

class BloomAttention(nn.Module):
    def __init__(self, config: BloomConfig):
        super().__init__()

        self.pretraining_tp = config.pretraining_tp
        self.slow_but_exact = config.slow_but_exact

        self.hidden_size = config.hidden_size
        self.num_heads = config.n_head
        self.head_dim = self.hidden_size // self.num_heads
        self.split_size = self.hidden_size
        self.hidden_dropout = config.hidden_dropout

        if self.head_dim * self.num_heads != self.hidden_size:
            raise ValueError(
                f"`hidden_size` must be divisible by num_heads (got `hidden_size`: {self.hidden_size} and `num_heads`:"
                f" {self.num_heads})."
            )

        # Layer-wise attention scaling
        self.inv_norm_factor = 1.0 / math.sqrt(self.head_dim)
        self.beta = 1.0

        self.query_key_value = nn.Linear(self.hidden_size, 3 * self.hidden_size, bias=True)
        self.dense = nn.Linear(self.hidden_size, self.hidden_size)
        self.attention_dropout = nn.Dropout(config.attention_dropout)

        self.enable_tp = False
        self.tp_comm_group = None

        self.profile = False

        # kv related
        self.kv_cache = {}
        self.kv_status = {}
    
    def register_tp(self, bit, caliber, broadcast=True):
        assert len(self.kv_cache) == 0, "register_tp must be called before kv initialization"
        tp_config = qllm_tp_utils.get_tp_configs()
        global_rank = tp_config['global_rank']
        tp_index = tp_config['tp_index']
        split_k = tp_config['split_k']
        group = tp_config['group']
        broadcast_group = tp_config['broadcast_group']
        tp_config_COL = AdaQTPConfig(split_k=split_k, global_rank=global_rank, tp_index=tp_index, split_type='COLUMN', comm_group=group)
        tp_config_ROW = AdaQTPConfig(split_k=split_k, global_rank=global_rank, tp_index=tp_index, split_type='ROW', comm_group=group)
        kvq = self.query_key_value
        # first column then row
        self.query_key_value = quantize_one_linear_module(kvq, kernel_bit=bit, caliber=caliber, tp_config=tp_config_COL)
        self.dense = quantize_one_linear_module(self.dense, kernel_bit=bit, caliber=caliber, tp_config=tp_config_ROW)
        # enable tp
        self.enable_tp = True
        # self.tp_comm_group = group
        self.global_rank = global_rank
        self.tp_index = tp_index

        # partition along the head dim
        self.head_dim = self.head_dim // split_k
        self.broadcast = broadcast_group

    @torch.no_grad()
    def update_kv_cache(self, key_value_pair, request_id):
        if len(self.kv_cache) == 0 or self.profile:
            return 
        # copy the key value pair to the cache
        # self.kv_cache[layer_idx][request_id] = key_value_pair
        prev_token_length = self.kv_status[request_id][0]
        prompt_length = self.kv_status[request_id][1]
        self.kv_cache[request_id][0][:, :, prev_token_length:prev_token_length + prompt_length].copy_(key_value_pair[0][:, :, prev_token_length:prev_token_length + prompt_length])
        self.kv_cache[request_id][1][:, prev_token_length:prev_token_length + prompt_length, :].copy_(key_value_pair[1][:, prev_token_length:prev_token_length + prompt_length, :])
        # update token length
        self.kv_status[request_id][0] += 1

    @torch.no_grad()
    def get_kv_cache(self, request_id):
        if len(self.kv_cache) == 0:
            return None # not initialized
        # based on the prompt_length + previous token length to fetch kv
        prev_token_length = self.kv_status[request_id][0]
        prompt_length = self.kv_status[request_id][1]
        kv_output_length = prompt_length + prev_token_length - 1

        if prev_token_length == 0:
            return None
        # for bloom
            # concatenate along seq_length dimension:
            #  - key: [batch_size * self.num_heads, head_dim, kv_length]
            #  - value: [batch_size * self.num_heads, kv_length, head_dim]
        kv = (
            self.kv_cache[request_id][0][:, :, :kv_output_length],
            self.kv_cache[request_id][1][:, :kv_output_length, :]
        )
        return kv
    
    @torch.no_grad()
    def init_kv_cache(self, b, prompt_length, token_to_generate, request_id, torch_dtype=torch.float16):
        max_seq_len = prompt_length + token_to_generate
        k_shape = (b * self.num_heads, self.head_dim, max_seq_len)
        v_shape = (b * self.num_heads, max_seq_len, self.head_dim)
        params = list(self.query_key_value.parameters())
        device = params[0].device
        self.kv_cache[request_id] = (
            torch.empty(k_shape, dtype=torch_dtype, device=device),
            torch.empty(v_shape, dtype=torch_dtype, device=device)
        ) 
        self.kv_status[request_id] = [0, prompt_length]

    def _split_heads(self, fused_qkv: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Split the last dimension into (num_heads, head_dim) without making any copies, results share same memory
        storage as `fused_qkv`

        Args:
            fused_qkv (`torch.tensor`, *required*): [batch_size, seq_length, num_heads * 3 * head_dim]

        Returns:
            query: [batch_size, seq_length, num_heads, head_dim] key: [batch_size, seq_length, num_heads, head_dim]
            value: [batch_size, seq_length, num_heads, head_dim]
        """
        batch_size, seq_length, three_times_hidden_size = fused_qkv.shape
        fused_qkv = fused_qkv.view(batch_size, seq_length, self.num_heads, 3, self.head_dim)
        return fused_qkv[..., 0, :], fused_qkv[..., 1, :], fused_qkv[..., 2, :]

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Merge heads together over the last dimenstion

        Args:
            x: (`torch.tensor`, *required*): [batch_size * num_heads, seq_length, head_dim]

        Returns:
            torch.tensor: [batch_size, seq_length, num_heads * head_dim]
        """
        # What we want to achieve is:
        # batch_size * num_heads, seq_length, head_dim -> batch_size, seq_length, num_heads * head_dim
        batch_size_and_num_heads, seq_length, _ = x.shape
        batch_size = batch_size_and_num_heads // self.num_heads

        # First view to decompose the batch size
        # batch_size * num_heads, seq_length, head_dim -> batch_size, num_heads, seq_length, head_dim
        x = x.view(batch_size, self.num_heads, seq_length, self.head_dim)

        # batch_size, num_heads, seq_length, head_dim -> batch_size, seq_length, num_heads, head_dim
        x = x.permute(0, 2, 1, 3)

        # batch_size, seq_length, num_heads, head_dim -> batch_size, seq_length, num_heads * head_dim
        return x.reshape(batch_size, seq_length, self.num_heads * self.head_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        alibi: torch.Tensor,
        attention_mask: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        request_id: int = 1,
    ):
        if self.enable_tp and self.broadcast:
            group = qllm_tp_utils.get_tp_group()
            tp._broad_cast(hidden_states, self.global_rank, self.tp_index, group) # broadcast hidden states
        # when tensor parallel, split hidden_states at the front, gather at the end.
        fused_qkv = self.query_key_value(hidden_states)  # [batch_size, seq_length, 3 x hidden_size]

        # 3 x [batch_size, seq_length, num_heads, head_dim]
        (query_layer, key_layer, value_layer) = self._split_heads(fused_qkv)

        batch_size, q_length, _, _ = query_layer.shape

        query_layer = query_layer.transpose(1, 2).reshape(batch_size * self.num_heads, q_length, self.head_dim)
        key_layer = key_layer.permute(0, 2, 3, 1).reshape(batch_size * self.num_heads, self.head_dim, q_length)
        value_layer = value_layer.transpose(1, 2).reshape(batch_size * self.num_heads, q_length, self.head_dim)
        layer_past = self.get_kv_cache(request_id) 
        if layer_past is not None:
            past_key, past_value = layer_past
            # concatenate along seq_length dimension:
            #  - key: [batch_size * self.num_heads, head_dim, kv_length]
            #  - value: [batch_size * self.num_heads, kv_length, head_dim]
            key_layer = torch.cat((past_key, key_layer), dim=2)
            value_layer = torch.cat((past_value, value_layer), dim=1)
            
        # update kv cache
        self.update_kv_cache((key_layer, value_layer), request_id)

        _, _, kv_length = key_layer.shape

        # [batch_size * num_heads, q_length, kv_length]
        # we use `torch.Tensor.baddbmm` instead of `torch.baddbmm` as the latter isn't supported by TorchScript v1.11
        matmul_result = alibi.baddbmm(
            batch1=query_layer,
            batch2=key_layer,
            beta=self.beta,
            alpha=self.inv_norm_factor,
        )

        # change view to [batch_size, num_heads, q_length, kv_length]
        attention_scores = matmul_result.view(batch_size, self.num_heads, q_length, kv_length)

        # cast attention scores to fp32, compute scaled softmax and cast back to initial dtype - [batch_size, num_heads, q_length, kv_length]
        input_dtype = attention_scores.dtype
        # `float16` has a minimum value of -65504.0, whereas `bfloat16` and `float32` have a minimum value of `-3.4e+38`
        if input_dtype == torch.float16:
            attention_scores = attention_scores.to(torch.float)
        attn_weights = torch.masked_fill(attention_scores, attention_mask, torch.finfo(attention_scores.dtype).min)
        attention_probs = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(input_dtype)

        # [batch_size, num_heads, q_length, kv_length]
        attention_probs = self.attention_dropout(attention_probs)

        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        # change view [batch_size x num_heads, q_length, kv_length]
        attention_probs_reshaped = attention_probs.view(batch_size * self.num_heads, q_length, kv_length)

        # matmul: [batch_size * num_heads, q_length, head_dim]
        context_layer = torch.bmm(attention_probs_reshaped, value_layer)

        # change view [batch_size, num_heads, q_length, head_dim]
        context_layer = self._merge_heads(context_layer)

        # aggregate results across tp ranks. See here: https://github.com/pytorch/pytorch/issues/76232
        if self.pretraining_tp > 1 and self.slow_but_exact:
            slices = self.hidden_size / self.pretraining_tp
            output_tensor = torch.zeros_like(context_layer)
            for i in range(self.pretraining_tp):
                output_tensor = output_tensor + F.linear(
                    context_layer[:, :, int(i * slices) : int((i + 1) * slices)],
                    self.dense.weight[:, int(i * slices) : int((i + 1) * slices)],
                )
        else:
            output_tensor = self.dense(context_layer)

        if self.enable_tp:
            # gather result
            group = qllm_tp_utils.get_tp_group()
            output_tensor = tp._all_reduce_sum(output_tensor, group)
        
        output_tensor = dropout_add(output_tensor, residual, self.hidden_dropout, self.training)

        outputs = (output_tensor, )
        # outputs = (output_tensor, present)
        # if output_attentions:
        #     outputs += (attention_probs,)

        return outputs


class BloomBlockSharded(nn.Module):
    def __init__(self, config: BloomConfig):
        super().__init__()
        hidden_size = config.hidden_size

        self.input_layernorm = LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.num_heads = config.n_head
        self.self_attention = BloomAttention(config)
        self.post_attention_layernorm = LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

        self.mlp = BloomMLP(config)

        self.apply_residual_connection_post_layernorm = config.apply_residual_connection_post_layernorm
        self.hidden_dropout = config.hidden_dropout

        # shard info
        self.partitions = [0, 1]
    
    def set_partition(self, partitions):
        self.partitions = [i for i in partitions]
    
    def is_only_self_attention(self):
        return self.partitions == [0]
    
    def has_self_attention(self):
        return 0 in self.partitions

    @torch.no_grad()
    def verify_kernel(self):
        test_result = []
        self_attention_fake_bits, ffn_fake_bits = 16, 16
        for idx, part_idx in enumerate(self.partitions):
            bit = self.bits[idx]
            if part_idx == 0:
                self_attention_fake_bits = 16 if bit != '8:tc' else 8
            if part_idx == 1:
                ffn_fake_bits = 16 if bit != '8:tc' else 8
        
        for part, input_shape in self.test_input_shape.items():
            if part == 'self_attention':
                try:
                    fake_input = torch.randn(input_shape).cuda().half() if self_attention_fake_bits == 16 else \
                        torch.randint(-128, 127, input_shape, dtype=torch.int8).cuda()
                    test_attn = self.self_attention.cuda()
                    fake_output = test_attn(fake_input)
                    del test_attn, fake_output
                    test_result.append(True)
                except Exception as e:
                    print(e)
                    test_result.append(False)
            if part == 'FFN':
                try:
                    fake_input = torch.randn(input_shape).cuda().half() if ffn_fake_bits == 16 else \
                            torch.randint(-128, 127, input_shape, dtype=torch.int8).cuda()
                    test_fc1 = self.mlp.dense_h_to_4h.cuda()
                    test_fc2 = self.mlp.dense_4h_to_h.cuda()
                    # test
                    fake_output = test_fc1(fake_input)
                    fake_output = test_fc2(fake_output)
                    del test_fc1, test_fc2, fake_output, fake_input
                    test_result.append(True)
                except Exception as e:
                    print(e)
                    test_result.append(False)        
        
        return test_result

    @torch.no_grad()
    def shard(self, shard_strategy):
        self.set_partition(shard_strategy.get('shard', []))
        bits = shard_strategy.get('bits', [])
        self.bits = bits
        assert len(self.partitions) == len(bits), "shard and bitwidths should have the same length"

        # remove layer
        if 0 not in self.partitions:
            self.input_layernorm = None
            self.self_attention = None

        if 1 not in self.partitions:
            self.mlp = None
            self.post_attention_layernorm = None
        
        self.test_input_shape = {}
        caliber = lptorch.inner_caliber
        # Do Quantization Here
        for idx, partition in enumerate(self.partitions):
            if partition == 0:
                # deal with quantization here.
                bit = bits[idx]
                if not caliber.fake:
                    input_shape = caliber.get_module_input_shape(self.self_attention.query_key_value)
                    self.test_input_shape['self_attention'] = input_shape
                else:
                    for k, v in caliber.named_fake_input_shape.items():
                        if 'self_attention' in k:
                            if 'query_key_value' in k:
                                self.test_input_shape['self_attention'] = v
                            fake_calib_data = caliber.named_fake_calib_data[k]
                            if "query" in k:
                                unique_id = caliber.man_set_unique_id(self.self_attention.query_key_value)
                                caliber.man_set_module_calib_data(self.self_attention.query_key_value, fake_calib_data)
                            elif 'dense' in k:
                                unique_id = caliber.man_set_unique_id(self.self_attention.dense)
                                caliber.man_set_module_calib_data(self.self_attention.dense, fake_calib_data)
                
                # quantize
                if bit == "8:tc":
                    # remove the 8:tc for the moment, since it is not availble in small batchsize
                    print("use 8:tc-li instead" )
                    bit = "8:tc-li"
                if bit == "8:tc-li":
                    bit = 8
                    use_calib = True
                else:
                    use_calib = False

                # check if the tp_config is inside the shard_strategy
                if 'tp_config' in shard_strategy:
                    # quantize with tp sharding
                    comm_group = qllm._globals.__TENSOR__MODEL_PARALLEL__GROUP__
                    tp_config = shard_strategy['tp_config']
                    index = tp_config['index']
                    k = tp_config['k']
                    self.self_attention.register_tp(bit, caliber)
                else:
                    quantize_linear_module_with_bit(self.self_attention, bit, caliber=caliber) if use_calib else \
                        quantize_linear_module_with_bit(self.self_attention, bit)
                
            elif partition == 1:
                bit = bits[idx]
                # case that only ffn exists, prepare a forward quantizer here
                if not caliber.fake:
                    input_shape = caliber.get_module_input_shape(self.mlp.dense_h_to_4h)
                    self.test_input_shape['FFN'] = input_shape
                else:
                    for k, v in caliber.named_fake_input_shape.items():
                        if 'h_to_4h' in k:
                            self.test_input_shape['FFN'] = v
                            fake_calib_data = caliber.named_fake_calib_data[k]
                            unique_id = caliber.man_set_unique_id(self.mlp.dense_h_to_4h)
                            caliber.man_set_module_calib_data(self.mlp.dense_h_to_4h, fake_calib_data)
                        elif 'h_to_4h' in k:
                            fake_calib_data = caliber.named_fake_calib_data[k]
                            unique_id = caliber.man_set_unique_id(self.mlp.dense_4h_to_h)
                            caliber.man_set_module_calib_data(self.mlp.dense_4h_to_h, fake_calib_data)

                # quantize
                if bit == "8:tc":
                    # remove the 8:tc for the moment, since it is not availble in small batchsize
                    print("use 8:tc-li instead" )
                    bit = "8:tc-li"
                if bit == "8:tc-li":
                    bit = 8                    
                    use_calib = True
                else:
                    use_calib = False

                # check if the tp_config is inside the shard_strategy
                if 'tp_config' in shard_strategy:
                    # quantize with tp sharding
                    comm_group = qllm._globals.__TENSOR__MODEL_PARALLEL__GROUP__
                    tp_config = shard_strategy['tp_config']
                    index = tp_config['index']
                    k = tp_config['k']
                    self.mlp.register_tp(bit, caliber)
                else:
                    quantize_linear_module_with_bit(self.mlp, bit, caliber=caliber) if use_calib else \
                        quantize_linear_module_with_bit(self.mlp, bit)

    @torch.no_grad()
    def SELFATTEN_PART(self, hidden_states:torch.Tensor, attention_mask, layer_head_mask, alibi=None, request_id=1):
        # Layer norm at the beginning of the transformer layer.
        layernorm_output = self.input_layernorm(hidden_states)

        # Layer norm post the self attention.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = hidden_states

        # Self attention.
        attn_outputs = self.self_attention(
            layernorm_output,
            residual,
            attention_mask=attention_mask,
            alibi=alibi,
            head_mask=layer_head_mask,
            output_attentions=False, # didn't ouput attention
            request_id=request_id
        )

        attention_output = attn_outputs[0]

        outputs = attn_outputs[1:]

        return (attention_output,) + outputs # present key value
    
    @torch.no_grad()
    def FFN_PART(self, hidden_states:torch.Tensor):
        attention_output = hidden_states
        layernorm_output = self.post_attention_layernorm(attention_output)
        # Get residual
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = attention_output

        # MLP.
        output = self.mlp(layernorm_output, residual)

        return (output, )  

    @torch.no_grad()
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
        alibi: Optional[torch.Tensor] = None,
        request_id: int = 1,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:

        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (`torch.FloatTensor`, *optional*): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """
        if 0 in self.partitions:
            hidden_states = self.SELFATTEN_PART(hidden_states, attention_mask, head_mask, \
                                                 alibi=alibi, request_id=request_id)
            # hidden states here are tuple
            if 1 in self.partitions:
                hidden_states = hidden_states[0]
                outputs = self.FFN_PART(hidden_states)
                return outputs
            else:
                return hidden_states
        # Fully Connected
        if 1 in self.partitions:
            outputs = self.FFN_PART(hidden_states)
            return outputs
        raise ValueError("No partition is specified")
    
# act like OPTDecoder in usage.
class BloomModelSeq(BloomModel):
    def __init__(self, config: BloomConfig):
        with init_empty_weights():
            super().__init__(config)

        self.embed_dim = config.hidden_size
        self.num_heads = config.n_head

        # Embedding + LN Embedding
        self.word_embeddings = nn.Embedding(config.vocab_size, self.embed_dim)
        self.word_embeddings_layernorm = LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

        # Transformer blocks
        import os 
        set_decoder_meta = os.environ.get('SET_DECODERS_META', "0")
        if set_decoder_meta == "0":
            self.h = nn.ModuleList([BloomBlockSharded(config) for _ in range(config.num_hidden_layers)])
        else:
            # init empty
            self.h = nn.ModuleList([None for _ in range(config.num_hidden_layers)])
        
        # Final Layer Norm
        self.ln_f = LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()
    
    @torch.no_grad()
    def forward_pre(self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        past_key_values_length: int = 0):
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if past_key_values is None:
            past_key_values = tuple([None] * len(self.h))

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape batch_size x num_heads x N x N
        # head_mask has shape n_layer x batch x num_heads x N x N
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        hidden_states = self.word_embeddings_layernorm(inputs_embeds)

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # Compute alibi tensor: check build_alibi_tensor documentation
        seq_length_with_past = seq_length
        past_key_values_length = 0
        if past_key_values[0] is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length_with_past), device=hidden_states.device)
        else:
            attention_mask = attention_mask.to(hidden_states.device)

        alibi = self.build_alibi_tensor(attention_mask, self.num_heads, dtype=hidden_states.dtype)

        causal_mask = self._prepare_attn_mask(
            attention_mask,
            input_shape=(batch_size, seq_length),
            past_key_values_length=past_key_values_length,
        )

        return {
            "hidden_states": hidden_states,
            "attention_mask": causal_mask,
            "head_mask": head_mask,
            "past_key_values": past_key_values,
            "output_hidden_states": output_hidden_states,
            "output_attentions": output_attentions,
            "next_decoder_cache": None,
            "alibi": alibi
        }
    
    @torch.no_grad()
    def forward_post(self, hidden_states):
        # if self.input_len is not None:
        #     hidden_states = hidden_states[:, :self.input_len, :]
                # Add last hidden state
        hidden_states = self.ln_f(hidden_states)
        return hidden_states

    def get_decoder_layer_num(self):
        return len(self.h)

    def check_is_meta(self, layer):
        for name, param in layer.named_parameters():
            if 'weight' in name:
                return param.is_meta

    # inplace sharding
    @torch.no_grad()
    def _shard_decoders(self, sharding_strategy, device=None):
        layer_idxs = list(sharding_strategy.keys())
        for i in range(self.get_decoder_layer_num()):
            if i not in layer_idxs:
                del_ele = self.h[i]
                self.h[i] = None
                del del_ele
        # for each layer, start sharding
        for layer_idx in layer_idxs:
            layer = self.h[layer_idx]
            if layer is None:
                self.h[layer_idx] = BloomBlockSharded(self.config).to(torch.float16)
                print("create layer:", layer_idx, end="|")
            self.h[layer_idx].shard(sharding_strategy[layer_idx])
            self.h[layer_idx].eval() # use eval mode
            # directly move to device
            if device is not None:
                self.h[layer_idx] = self.h[layer_idx].to(device)
    
    def load_layer_weight(self, shard_weight_dir):
        pass
        # for layer in self.h:
        #     if layer is not None:
        #         layer.load_weight()

    @torch.no_grad()
    def verify_decoder_layers(self):
        for layer in self.h:
            if layer is not None:
               return layer.verify_kernel() # we only need to verify one layer

    def _delete_all_other_modules(self):
        del self.word_embeddings
        del self.word_embeddings_layernorm
        del self.ln_f

    # kv related operations
    @torch.no_grad()
    def init_kv_cache(self, b, prompt_length, token_to_generate, request_id):

        for idx, block in enumerate(self.h):
            if block is None:
                continue
            bits = block.bits[0]
            if bits == '8:tc':
                torch_dtype = torch.int8 # store in int8
            else:
                torch_dtype = torch.float16
            if block.has_self_attention():
                block.self_attention.init_kv_cache(b, prompt_length, token_to_generate, request_id, torch_dtype=torch_dtype)
    
    @torch.no_grad()
    def get_all_kv_cache_dict(self, request_id=None):
        return_dict = {}
        for idx, block in enumerate(self.h):
            if block is None:
                continue
            if block.has_self_attention():
                if hasattr(block.self_attention, 'kv_cache'):
                    if request_id is not None:
                        return_dict[idx] = block.self_attention.kv_cache[request_id]
                    else:
                        return_dict[idx] = block.self_attention.kv_cache
        return return_dict
    
    @torch.no_grad()
    def clear_request_kv_cache(self, request_id=1):
        for idx, block in enumerate(self.h):
            if block is None:
                continue
            if block.has_self_attention():
                if hasattr(block.self_attention, 'kv_cache'):
                    del block.self_attention.kv_cache[request_id]

    @torch.no_grad()
    def clear_all_kv_cache(self):
        for idx, block in enumerate(self.h):
            if block is None:
                continue
            if block.has_self_attention():
                if hasattr(block.self_attention, 'kv_cache'):
                    block.self_attention.kv_cache = {}

    @torch.no_grad()
    def forward(
        self,
        hidden_states: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.LongTensor] = None,
        alibi: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        request_id: Optional[int] = 1,
        **deprecated_arguments,
    ) -> Union[Tuple[torch.Tensor, ...], BaseModelOutputWithPastAndCrossAttentions]:
        mask_and_cache = (attention_mask, head_mask, alibi, use_cache) 
        for idx, block in enumerate(self.h):
            if block is None:
                continue

            outputs = block(
                hidden_states,
                attention_mask=attention_mask,
                head_mask=head_mask[idx],
                output_attentions=output_attentions,
                alibi=alibi,
                request_id=request_id,
            )

            if block.is_only_self_attention():
                hidden_states = (outputs[0], ) # 
                return hidden_states + mask_and_cache + (request_id, ) 

            hidden_states = outputs[0]

        # Add last hidden state
        # hidden_states = self.ln_f(hidden_states)

        return (hidden_states, ) + mask_and_cache + (request_id, )

class BloomForCausalLMSeq(BloomForCausalLM):
    def __init__(self, config: BloomConfig):
        with init_empty_weights():
            super().__init__(config)
        self.transformer = BloomModelSeq(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()
    
    '''
        Same for all LLMs
    '''
    @torch.no_grad()
    def preprocess_one_token(self, new_input_ids, next_tokens, use_cache=True, request_id=1):
        with torch.no_grad():
            batch_size, seq_length = new_input_ids.shape
            embed_tokens = self.transformer.word_embeddings
            # embed the new input tokens
            inputs_embeds = embed_tokens(next_tokens)
            inputs_embeds = inputs_embeds.view(batch_size, 1, -1)
            hidden_states = self.transformer.word_embeddings_layernorm(inputs_embeds)

            head_mask = self.transformer.get_head_mask(None, self.config.n_layer)

            seq_length_with_past = seq_length
            past_key_values_length = seq_length_with_past - 1
            input_shape = (batch_size, 1)
            attention_mask = torch.ones((batch_size, seq_length_with_past), device=hidden_states.device)
            num_heads = self.config.num_attention_heads
            alibi = self.transformer.build_alibi_tensor(attention_mask, num_heads, dtype=hidden_states.dtype)
            causal_mask = self.transformer._prepare_attn_mask(
                attention_mask,
                input_shape=input_shape,
                past_key_values_length=past_key_values_length,
            )
            mask_and_cache = (causal_mask, head_mask, alibi, use_cache) 
            request_token = (hidden_states,) + mask_and_cache + (request_id, )
            return request_token
    
    @torch.no_grad()
    def preprocess(self, input_ids=None, attention_mask=None, head_mask=None, past_key_values=None, inputs_embeds=None,
                   use_cache=None, output_attentions=None, output_hidden_states=None, return_dict=None, request_id=1, \
                    past_key_values_length: int = 0, **kwargs):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        pre_result = self.transformer.forward_pre(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            past_key_values_length=past_key_values_length,
        )
        self.return_dict = return_dict

        # add broadcast here for dist
        attention_mask = pre_result['attention_mask']
        head_mask = pre_result['head_mask']
        alibi = pre_result['alibi']
        use_cache = use_cache
        mask_and_cache = (attention_mask, head_mask, alibi, use_cache) 
        pre_result = (pre_result['hidden_states'],) + mask_and_cache + (request_id, )

        return pre_result
    
    def get_decoder_layer_num(self):
        return self.transformer.get_decoder_layer_num()
    # model sharders
    @torch.no_grad()
    def _verify_shard_strategy(self, shard_strategies):
        all_decode_ids = set()
        decoder_layer_num = self.get_decoder_layer_num()
        for idx, shard_strategy in shard_strategies.items():
            decode_ids = shard_strategy.keys()
            all_decode_ids = all_decode_ids.union(decode_ids)
        assert len(list(all_decode_ids)) == decoder_layer_num, f"MUST EQUAL {len(list(all_decode_ids))}/{len(self.model.decoder.layers)}"
    
    @torch.no_grad()
    def _shard_model_current(self, shard_strategy, device=None):
        self.transformer._shard_decoders(shard_strategy, device)
        self.transformer._delete_all_other_modules()
        del self.lm_head

    # inplace sharding
    @torch.no_grad()
    def _shard_model(self, shard_strategies, shard_idx):
        self.is_master = True if shard_idx == 0 else False
        if self.is_master:
            self._verify_shard_strategy(shard_strategies)
        current_shard_strategy = shard_strategies[shard_idx]
        self._shard_model_current(current_shard_strategy)
    
    @torch.no_grad()
    def _pure_pre_and_post(self):
        sharded_model = copy.deepcopy(self)
        sharded_model.transformer._shard_decoders({}) # didn't delete the embeddings, etc.
        sharded_model.transformer.word_embeddings = qllm_nn.Embedding1D.from_embed(sharded_model.transformer.word_embeddings)
        # don't handle the positional embedding, since it is small. follow collosal implementation
        # handle the lm_head
        self.lm_head = qllm_nn.Classifier1D.from_classi(self.lm_head, broadcast=True)
        return sharded_model

    # return model instance with copy
    @torch.no_grad()
    def shard_model(self, shard_strategies, shard_idx):
        # copy a model
        sharded_model = copy.deepcopy(self)
        sharded_model._shard_model(shard_strategies, shard_idx)
        return sharded_model
    
    @torch.no_grad()
    def decoder_layers_to_device(self, device):
        self.transformer.h = self.transformer.h.to(device)

    # rewrite the forward function
    @torch.no_grad()
    def decode(self, pre_result):
        def shard_launcher(prev_result, module_shard):
            # length_prev_result = len(prev_result)
            hidden_states = prev_result[0]
            attention_mask, head_mask, alibi, use_cache, request_id = prev_result[1:]
            res = module_shard(
                hidden_states=hidden_states,
                next_decoder_cache=None,
                attention_mask=attention_mask,
                head_mask=head_mask,
                past_key_values=None,
                use_cache=use_cache,
                alibi=alibi,
                request_id=request_id,
            )
            return res

        return shard_launcher(pre_result, self.transformer)

    @torch.no_grad()
    def init_kv_cache(self, b, prompt_length, token_to_generate, request_id):
        return self.transformer.init_kv_cache(b, prompt_length, token_to_generate, request_id)
    
    @torch.no_grad()
    def postprocess(self, results, labels=None):
        transformer_outputs = results
        return_dict = self.return_dict
        hidden_states = transformer_outputs[0]

        hidden_states = self.transformer.forward_post(hidden_states)

        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            batch_size, seq_length, vocab_size = shift_logits.shape
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(batch_size * seq_length, vocab_size), shift_labels.view(batch_size * seq_length)
            )

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=None,
            hidden_states=hidden_states,
            attentions=None,
        )
    
    @torch.no_grad()
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **deprecated_arguments,
    ) -> Union[Tuple[torch.Tensor], CausalLMOutputWithCrossAttentions]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        if deprecated_arguments.pop("position_ids", False) is not False:
            # `position_ids` could have been `torch.Tensor` or `None` so defaulting pop to `False` allows to detect if users were passing explicitly `None`
            warnings.warn(
                "`position_ids` have no functionality in BLOOM and will be removed in v5.0.0. You can safely ignore"
                " passing `position_ids`.",
                FutureWarning,
            )
        if len(deprecated_arguments) > 0:
            raise ValueError(f"Got unexpected arguments: {deprecated_arguments}")

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        self.return_dict = return_dict

        token_input = self.preprocess(input_ids, use_cache=True, request_id=1)
        transformer_outputs = self.decode(token_input)
        return self.postprocess(transformer_outputs)
    
    

if __name__ == '__main__':
    from qllm.models import bloom
    from qllm.utils import get_model_size, to_device_recursive
    bloom_basic, tokenizer = bloom.load_pretained_model_from_net('bigscience/bloom-560m')
    # bloom_basic2, tokenizer = bloom.load_pretained_model_from_net('bigscience/bloom-560m')
    test_bloom = BloomForCausalLMSeq.from_pretrained('bigscience/bloom-560m', torch_dtype=torch.float16)
    max_length = 128
    batched_ids = tokenizer.batch_encode_plus(["Hi, where is my dog. ", "Just test performance. How about you. ", \
                                                "The quick brown fox jumps over the lazy dog. It's a beautiful day outside, the sun is shining and the birds are chirping. I feel like going for a"], \
                                                padding='max_length', max_length=max_length, return_tensors="pt")

    # tocuda, since load using fp16
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batched_ids = to_device_recursive(dict(batched_ids), device)
    test_bloom = test_bloom.to(device)
    bloom_basic = bloom_basic.to(device)
    # bloom_basic2 = bloom_basic2.cuda()
    test_bloom.eval()
    bloom_basic.eval()

    with torch.no_grad():
        bloom_basic_out = bloom_basic(**batched_ids)
        # bloom_basic_out2 = bloom_basic2(input_ids)
    
    request_token = test_bloom.preprocess(**batched_ids, use_cache=True, request_id=1)
    # init kv
    num_tokens_to_generate = 1
    bs, prompt_length = batched_ids['input_ids'].shape
    test_bloom.init_kv_cache(bs, prompt_length, num_tokens_to_generate, request_id=1)
    with torch.no_grad():
        intermediate = test_bloom.decode(request_token)
        test_bloom_seq_out = test_bloom.postprocess(intermediate)
    
    # print(bloom_basic_out.logits - bloom_basic_out2.logits)
    print(test_bloom_seq_out.logits - bloom_basic_out.logits)