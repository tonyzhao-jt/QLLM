# replace the huggingface implemented layers into the QLLM implementation for sequential execution
import torch.nn as nn 
import torch
from transformers import (
    OPTConfig,
    OPTForCausalLM,
    AutoTokenizer,
    
)
# original ones
from transformers.models.opt.modeling_opt import (
    OPTAttention,
    OPTDecoderLayer,
    OPTForCausalLM,
    OPTLearnedPositionalEmbedding,
    OPTModel,
    OPTDecoder,
    OPTPreTrainedModel,
    _make_causal_mask, _expand_mask
)
# output decorator
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    ModelOutput
)

from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.utils import logging
from transformers.activations import ACT2FN
logger = logging.get_logger(__name__)

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import random
from dataclasses import dataclass
import copy
from accelerate import init_empty_weights

import lptorch
from lptorch import quantize_linear_module_with_bit, quantize_one_linear_module, ForwardTokenizer
from lptorch.utils import is_tensorcore_int8_available


from lptorch.torch_int.nn.bmm import BMM_S8T_S8N_S8T, BMM_S8T_S8N_F32T
from lptorch.torch_int.nn.linear import W8A8BFP32OFP32Linear, W8A8B8O8Linear, W8A8B8O8LinearReLU
from lptorch.torch_int.nn.fused import LayerNormQ

class Int8OPTAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )

        self.attention_weight_scale = 1.0

        self.qk_bmm = BMM_S8T_S8N_F32T(1.0)
        self.pv_bmm = BMM_S8T_S8N_S8T(1.0)

        self.k_proj = W8A8B8O8Linear(embed_dim, embed_dim)
        self.v_proj = W8A8B8O8Linear(embed_dim, embed_dim)
        self.q_proj = W8A8B8O8Linear(embed_dim, embed_dim)
        self.out_proj = W8A8BFP32OFP32Linear(embed_dim, embed_dim)

        self.fwd_tokenizer = ForwardTokenizer(16, 8)

    @staticmethod
    @torch.no_grad()
    def from_float(module: OPTAttention,
                   input_scale: float,
                   q_output_scale: float,
                   k_output_scale: float,
                   v_output_scale: float,
                   out_input_scale: float):
        int8_module = Int8OPTAttention(module.embed_dim, module.num_heads)
        # Fuse the scaling into the q_proj output scale
        q_output_scale = q_output_scale * module.scaling
        module.q_proj.weight *= module.scaling
        module.q_proj.bias *= module.scaling
        int8_module.q_proj = W8A8B8O8Linear.from_float(
            module.q_proj, input_scale, q_output_scale)
        int8_module.k_proj = W8A8B8O8Linear.from_float(
            module.k_proj, input_scale, k_output_scale)
        int8_module.v_proj = W8A8B8O8Linear.from_float(
            module.v_proj, input_scale, v_output_scale)
        int8_module.out_proj = W8A8BFP32OFP32Linear.from_float(
            module.out_proj, out_input_scale)
        int8_module.qk_bmm = BMM_S8T_S8N_F32T.from_scale(
            q_output_scale, k_output_scale)

        # alpha = s_prob * s_v / s_out, where s_prob = 1 / 127
        int8_module.pv_bmm = BMM_S8T_S8N_S8T.from_scale(
            1.0 / 127, v_output_scale, out_input_scale)
        return int8_module

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    @torch.no_grad()
    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""
        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None

        bsz, tgt_len, _ = hidden_states.size()
        # add a quantizer here
        # get query proj
        if hidden_states.dtype != torch.int8:
            hidden_states = self.fwd_tokenizer(hidden_states)
        query_states = self.q_proj(hidden_states)
        # get key, value proj
        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(
            query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        # for int8, N(src_length) need to be multiple of 16, padding here
        # pad_len = (16 - src_len % 16) % 16
        # padding = torch.zeros(key_states.size(0), pad_len, key_states.size(2), dtype=key_states.dtype, device=key_states.device)
        # key_states = torch.cat([key_states, padding], dim=1)

        # value_length = value_states.size(1)
        # value_pad_length = (16 - value_states.size(1) % 16) % 16 
        # padding_v = torch.zeros(value_states.size(0), value_pad_length, value_states.size(2), dtype=key_states.dtype, device=key_states.device)
        # value_states = torch.cat([value_states, padding_v], dim=1)
        # src_len = src_len + pad_len
        # print(query_states.shape, key_states.shape)
        # print(query_states.dtype, key_states.dtype)
        # import pdb; pdb.set_trace()
        # the bmm can only done in fp16. requires strict alginment and padding
        # attn_weights = self.qk_bmm(query_states, key_states)
        if src_len % 16:
            attn_weights = torch.bmm(query_states.half(), key_states.transpose(1,2).half())
        else:
            attn_weights = self.qk_bmm(query_states, key_states)

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(
                bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = torch.max(attn_weights, torch.tensor(
                torch.finfo(attn_weights.dtype).min))
            attn_weights = attn_weights.view(
                bsz * self.num_heads, tgt_len, src_len)

        attn_probs = nn.functional.softmax(attn_weights, dim=-1)

        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is"
                    f" {layer_head_mask.size()}"
                )
            attn_probs = layer_head_mask.view(
                1, -1, 1, 1) * attn_probs.view(bsz, self.num_heads, tgt_len, src_len)
            attn_probs = attn_probs.view(
                bsz * self.num_heads, tgt_len, src_len)

        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to be reshaped
            # twice and have to be reused in the following
            attn_probs_reshaped = attn_probs.view(
                bsz, self.num_heads, tgt_len, src_len)
            attn_probs = attn_probs_reshaped.view(
                bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_probs_reshaped = None

        # (A_row V_row)_row = (A_row V_col ^T)_row

        if src_len % 16:
            attn_output = torch.bmm(attn_probs.half(), value_states.half())
            attn_output = attn_output.to(torch.int8)
        else:
            attn_probs.mul_(127).round_()
            attn_probs = attn_probs.to(torch.int8)
            value_states = value_states.transpose(1, 2).contiguous()
            attn_output = self.pv_bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(
            bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)

        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned aross GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(
            bsz, tgt_len, self.embed_dim).contiguous()
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_probs_reshaped, past_key_value

class OPTAttentionSeq(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None

        bsz, tgt_len, _ = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        # get key, value proj
        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)
        
        src_len = key_states.size(1)
        # print(query_states.size(), key_states.size())
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min))
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        # upcast to fp32 if the weights are in fp16. Please see https://github.com/huggingface/transformers/pull/17437
        if attn_weights.dtype == torch.float16:
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(torch.float16)
        else:
            attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is"
                    f" {layer_head_mask.size()}"
                )
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to be reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)

        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned aross GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped, past_key_value

class OPTDecoderLayerSharded(nn.Module):
    def __init__(self, config: OPTConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = OPTAttentionSeq(
            embed_dim=self.embed_dim,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            bias=config.enable_bias,
        )
        self.do_layer_norm_before = config.do_layer_norm_before
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]

        self.self_attn_layer_norm = nn.LayerNorm(
            self.embed_dim, elementwise_affine=config.layer_norm_elementwise_affine
        )
        self.fc1 = nn.Linear(self.embed_dim, config.ffn_dim, bias=config.enable_bias)
        self.fc2 = nn.Linear(config.ffn_dim, self.embed_dim, bias=config.enable_bias)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim, elementwise_affine=config.layer_norm_elementwise_affine)

        self.partitions = [0,1] # by default, contain both self attention and FFN

    def set_partition(self, partitions):
        self.partitions = [i for i in partitions]
    
    def is_only_self_attention(self):
        return self.partitions == [0]

    def has_self_attention(self):
        return 0 in self.partitions

    # FOR CUTLASS (tensorcore), it may not be available to run on all GPUs.
    # Need to verify whether it is available on the current GPU.
    def verify_kernel(self):
        test_result = []
        self_attn_fake_bits, ffn_fake_bits = 16, 16
        for idx, part_idx in enumerate(self.partitions):
            bit = self.bits[idx]
            if part_idx == 0:
                self_attn_fake_bits = 16 if bit != '8:tc' else 8
            if part_idx == 1:
                ffn_fake_bits = 16 if bit != '8:tc' else 8
        
        for part, input_shape in self.test_input_shape.items():
            if part == 'self_attn':
                try:
                    fake_input = torch.randn(input_shape).cuda().half() if self_attn_fake_bits == 16 else \
                        torch.randint(-128, 127, input_shape, dtype=torch.int8).cuda()
                    test_attn = self.self_attn.cuda()
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
                    test_fc1 = self.fc1.cuda()
                    test_ln = self.final_layer_norm.cuda()
                    test_fc2 = self.fc2.cuda()
                    # test
                    fake_output = test_fc1(test_ln(fake_input))
                    fake_output = test_fc2(fake_output)
                    del test_fc1, test_ln, test_fc2, fake_output, fake_input
                    test_result.append(True)
                except Exception as e:
                    print(e)
                    test_result.append(False)        
        
        return test_result
    
    def shard(self, shard_strategy):
        self.set_partition(shard_strategy.get('shard', []))
        bits = shard_strategy.get('bits', [])
        self.bits = bits
        assert len(self.partitions) == len(bits), "shard and bitwidths should have the same length"

        if 0 not in self.partitions:
            self.self_attn = None
            self.self_attn_layer_norm = None

        if 1 not in self.partitions:
            self.fc1 = None
            self.fc2 = None
            self.final_layer_norm = None
        
        self.test_input_shape = {}
        caliber = lptorch.inner_caliber
        # Do Quantization Here
        for idx, partition in enumerate(self.partitions):
            if partition == 0:
                # deal with quantization here.
                bit = bits[idx]
                input_shape = caliber.get_module_input_shape(self.self_attn.q_proj)
                self.test_input_shape['self_attn'] = input_shape
                if bit == "8:tc": # tensorcore int8
                    assert is_tensorcore_int8_available() and caliber.get_module_calib_data(self.self_attn.q_proj) is not None, \
                        "Tensorcore is not available on this GPU, or the calibration data is not available"
                    attn_input_scale, q_output_scale = caliber.get_module_calib_data(self.self_attn.q_proj)
                    _, k_output_scale = caliber.get_module_calib_data(self.self_attn.k_proj)
                    _, v_output_scale = caliber.get_module_calib_data(self.self_attn.v_proj)
                    _, out_input_scale = caliber.get_module_calib_data(self.self_attn.out_proj)
                    self.self_attn = Int8OPTAttention.from_float(
                        self.self_attn, attn_input_scale, q_output_scale, k_output_scale, v_output_scale, out_input_scale)
                    
                    if self.do_layer_norm_before:
                        self.self_attn_layer_norm = LayerNormQ.from_float(self.self_attn_layer_norm, attn_input_scale)
                    else:
                        self.self_attn_layer_norm = LayerNormQ.from_float(self.self_attn_layer_norm, out_input_scale)
                # TODO: for moment, we disable the torchint by passing none to caliber
                elif bit == "8:tc-li":
                    bit = 8
                    quantize_linear_module_with_bit(self.self_attn, bit, caliber=caliber)
                else:   
                    quantize_linear_module_with_bit(self.self_attn, bit, caliber=None)
                
            elif partition == 1:
                bit = bits[idx]
                input_shape = caliber.get_module_input_shape(self.fc1)
                self.test_input_shape['FFN'] = input_shape
                if bit == "8:tc":
                    assert is_tensorcore_int8_available() and caliber.get_module_calib_data(self.fc1) is not None, \
                        "Tensorcore is not available on this GPU, or the calibration data is not available"
                    fc1_input_scale, fc1_output_scale = caliber.get_module_calib_data(self.fc1)
                    fc2_input_scale, fc2_output_scale = caliber.get_module_calib_data(self.fc2)
                    self.final_layer_norm = LayerNormQ.from_float(self.final_layer_norm, fc1_input_scale)
                    self.fc1 = W8A8B8O8LinearReLU.from_float(
                    self.fc1, fc1_input_scale, fc2_input_scale)
                    self.fc2 = W8A8BFP32OFP32Linear.from_float(
                    self.fc2, fc2_input_scale)
                elif bit == '8:tc-li':
                    bit = 8
                    self.fc1 = quantize_one_linear_module(self.fc1, kernel_bit=bit, caliber=caliber)
                    self.fc2 = quantize_one_linear_module(self.fc2, kernel_bit=bit, caliber=caliber)
                else:
                    # logger.info("use non-kernel quantization for f1f2")
                    # adaptive quantization for BITSANDBYTES and GPTQ
                    self.fc1 = quantize_one_linear_module(self.fc1, kernel_bit=bit, caliber=None)
                    self.fc2 = quantize_one_linear_module(self.fc2, kernel_bit=bit, caliber=None)

    def SELFATTEN_PART(self, hidden_states:torch.Tensor, attention_mask, past_key_value, layer_head_mask, output_attentions=False, use_cache=False):
        residual = hidden_states
        # 125m, 1.7B, ..., 175B applies layer norm BEFORE attention
        if self.do_layer_norm_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        # 350m applies layer norm AFTER attention
        if not self.do_layer_norm_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)
        
        # use kv cache
        if use_cache:
            outputs = (hidden_states, present_key_value)  
        else:
            outputs = (hidden_states,)

        return outputs
    
    def FFN_PART(self, hidden_states:torch.Tensor):
        hidden_states = hidden_states 
        # Fully Connected
        hidden_states_shape = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, hidden_states.size(-1))
        residual = hidden_states

        # 125m, 1.7B, ..., 175B applies layer norm BEFORE attention
        if self.do_layer_norm_before:
            hidden_states = self.final_layer_norm(hidden_states)

        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)

        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        hidden_states = (residual + hidden_states).view(hidden_states_shape)

        # 350m applies layer norm AFTER attention
        if not self.do_layer_norm_before:
            hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)

        return outputs

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
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
            hidden_states = self.SELFATTEN_PART(hidden_states, attention_mask, past_key_value, layer_head_mask, \
                                                 output_attentions=output_attentions, use_cache=use_cache)
            # hidden states here are tuple
            if 1 in self.partitions:
                if use_cache:
                    present_key_value = hidden_states[1] # tuple of two tensors (k and v)
                hidden_states = hidden_states[0]
                outputs = self.FFN_PART(hidden_states)
                return outputs if not use_cache else outputs + present_key_value
            else:
                return hidden_states
        # Fully Connected
        if 1 in self.partitions:
            outputs = self.FFN_PART(hidden_states)
            return outputs
        raise ValueError("No partition is specified")

class OPTDecoderSeq(OPTPreTrainedModel):
    def __init__(self, config: OPTConfig):
        super().__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.layerdrop
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_position_embeddings
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.word_embed_proj_dim, self.padding_idx)
        self.embed_positions = OPTLearnedPositionalEmbedding(config.max_position_embeddings, config.hidden_size)

        if config.word_embed_proj_dim != config.hidden_size:
            self.project_out = nn.Linear(config.hidden_size, config.word_embed_proj_dim, bias=False)
        else:
            self.project_out = None

        if config.word_embed_proj_dim != config.hidden_size:
            self.project_in = nn.Linear(config.word_embed_proj_dim, config.hidden_size, bias=False)
        else:
            self.project_in = None

        # Note that the only purpose of `config._remove_final_layer_norm` is to keep backward compatibility
        # with checkpoints that have been fine-tuned before transformers v4.20.1
        # see https://github.com/facebookresearch/metaseq/pull/164
        if config.do_layer_norm_before and not config._remove_final_layer_norm:
            self.final_layer_norm = nn.LayerNorm(
                config.hidden_size, elementwise_affine=config.layer_norm_elementwise_affine
            )
        else:
            self.final_layer_norm = None
        
        import os 
        set_decoder_meta = os.environ.get('SET_DECODERS_META', "0")
        if set_decoder_meta == "0":
            self.layers = nn.ModuleList([OPTDecoderLayerSharded(config) for _ in range(config.num_hidden_layers)])
        else:
            with init_empty_weights():
                self.layers = nn.ModuleList([OPTDecoderLayerSharded(config) for _ in range(config.num_hidden_layers)])
        # self.layers = nn.ModuleList([OPTDecoderLayer(config) for _ in range(config.num_hidden_layers)])

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init() # init weights

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    # Copied from transformers.models.bart.modeling_bart.BartDecoder._prepare_decoder_attention_mask
    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
                inputs_embeds.device
            )
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask
            
    
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
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        # past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # embed positions
        if attention_mask is None:
            attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.bool, device=inputs_embeds.device)
        pos_embeds = self.embed_positions(attention_mask, past_key_values_length)
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, input_shape, inputs_embeds, past_key_values_length
        )

        if self.project_in is not None:
            inputs_embeds = self.project_in(inputs_embeds)

        hidden_states = inputs_embeds + pos_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # check if head_mask has a correct number of layers specified if desired
        for attn_mask, mask_name in zip([head_mask], ["head_mask"]):
            if attn_mask is not None:
                if attn_mask.size()[0] != (len(self.layers)):
                    raise ValueError(
                        f"The `{mask_name}` should be specified for {len(self.layers)} layers, but it is for"
                        f" {head_mask.size()[0]}."
                    )
        next_decoder_cache = () if use_cache else None
        
        return {
            "hidden_states": hidden_states,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "past_key_values": past_key_values,
            "output_hidden_states": output_hidden_states,
            "output_attentions": output_attentions,
            "next_decoder_cache": next_decoder_cache,
        }

    def forward_post(self, hidden_states, next_decoder_cache=None, use_cache=False, return_dict=False):
        if self.final_layer_norm is not None:
            hidden_states = self.final_layer_norm(hidden_states)

        if self.project_out is not None:
            hidden_states = self.project_out(hidden_states)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, (), ()] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states= (),
            attentions=()
        )
    
    def get_decoder_layer_num(self):
        return len(self.layers)

    def check_is_meta(self, layer):
        for name, param in layer.named_parameters():
            if 'weight' in name:
                return param.is_meta

    # inplace sharding
    def _shard_decoders(self, sharding_strategy):
        layer_idxs = list(sharding_strategy.keys())
        for i in range(self.get_decoder_layer_num()):
            if i not in layer_idxs:
                del_ele = self.layers[i]
                self.layers[i] = None
                del del_ele
        # for each layer, start sharding
        for layer_idx in layer_idxs:
            layer = self.layers[layer_idx]
            if self.check_is_meta(layer):
                self.layers[layer_idx] = OPTDecoderLayerSharded(self.config).to(torch.float16)
            self.layers[layer_idx].shard(sharding_strategy[layer_idx])
    
    def verify_decoder_layers(self):
        for layer in self.layers:
            if layer is not None:
               return layer.verify_kernel() # we only need to verify one layer

    def _delete_all_other_modules(self):
        del self.embed_tokens
        del self.embed_positions
        del self.final_layer_norm
        del self.project_in
        del self.project_out
    
    def init_kv_cache(self, b, prompt_length, token_to_generate, request_id):
        if not hasattr(self, 'kv_cache'):
            self.kv_cache = {}
        max_seq_len = prompt_length + token_to_generate
        for layer_idx, decoder_layer in enumerate(self.layers):
            if decoder_layer is None:
                continue
            if not decoder_layer.has_self_attention():
                continue
            bits = decoder_layer.bits[0]
            if bits == '8:tc':
                kv_bit = 8
                torch_dtype = torch.int8
            else:
                kv_bit = 16
                torch_dtype = torch.float16
            hidden_size, head_num = self.config.hidden_size, self.config.num_attention_heads
            # size should be b x i x h
            if layer_idx not in self.kv_cache:
                self.kv_cache[layer_idx] = {}
            self.kv_cache[layer_idx][request_id] = [
                # size should be 
                torch.empty((b, head_num, max_seq_len, hidden_size // head_num), dtype=torch_dtype, device=self.device),
                torch.empty((b, head_num, max_seq_len, hidden_size // head_num), dtype=torch_dtype, device=self.device),
                0, # use to count the token generation
                prompt_length
            ]
    
    def update_kv_cache(self, layer_idx, key_value_pair, request_id):
        if not hasattr(self, 'kv_cache'):
            assert False, "kv_cache not initialized"
        # copy the key value pair to the cache
        # self.kv_cache[layer_idx][request_id] = key_value_pair
        self.kv_cache[layer_idx][request_id][2] += 1
        self.kv_cache[layer_idx][request_id][0][:, :, :self.kv_cache[layer_idx][request_id][2] + self.kv_cache[layer_idx][request_id][3] - 1, :].copy_(key_value_pair[0])
        self.kv_cache[layer_idx][request_id][1][:, :, :self.kv_cache[layer_idx][request_id][2] + \
                                                    self.kv_cache[layer_idx][request_id][3] - 1, :].copy_(key_value_pair[1])


    def get_kv_cache(self, layer_idx, request_id):
        if not hasattr(self, 'kv_cache'):
            assert False, "kv_cache not initialized"
        # based on the prompt_length + previous token length to fetch kv
        prompt_length = self.kv_cache[layer_idx][request_id][3]
        generated_token_length = self.kv_cache[layer_idx][request_id][2]
        kv_output_length = prompt_length + generated_token_length - 1
        if generated_token_length == 0:
            return None
        kv = (
            self.kv_cache[layer_idx][request_id][0][:, :, :kv_output_length, :],
            self.kv_cache[layer_idx][request_id][1][:, :, :kv_output_length, :]
        )
        return kv
    
    def verbose_kv_cache(self):
        print(self.kv_cache)
    
    def clear_request_kv_cache(self, request_id=1):
        for layer_idx in self.kv_cache:
            if request_id in self.kv_cache[layer_idx]:
                del self.kv_cache[layer_idx][request_id]

    def clear_all_kv_cache(self):
        self.kv_cache = {}

    # only remains inference related part
    def forward(
        self,
        hidden_states: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        next_decoder_cache: Optional[List[torch.FloatTensor]] = None,
        request_id: Optional[int] = 1,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        
        mask_and_cache = (attention_mask, head_mask, use_cache) 
        
        for idx, decoder_layer in enumerate(self.layers):
            if decoder_layer is None:
                continue
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if decoder_layer.has_self_attention():
                past_key_value = self.get_kv_cache(idx, request_id)
            else:
                past_key_value = None
            # past_key_value = past_key_values[idx] if past_key_values is not None else None
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                past_key_value=past_key_value,
                output_attentions=False,
                use_cache=use_cache,
            )
            if decoder_layer.has_self_attention(): # store the kv together with the selfattn
                if use_cache:
                    # the output will contain a present key value
                    if type(layer_outputs[1]) == tuple:
                        self.update_kv_cache(idx, layer_outputs[1], request_id)
                    else:
                        self.update_kv_cache(idx, layer_outputs[1:], request_id) # update for further usage

            if decoder_layer.is_only_self_attention():
                hidden_states = (layer_outputs[0], ) # 
                return hidden_states + mask_and_cache + (request_id, ) 
            
            # has only FFN or FFN + selfattn
            hidden_states = layer_outputs[0]

        return (hidden_states, ) + mask_and_cache + (request_id, )
        
# in generation, actually did nothing with model function. 
# only to decoder
class OPTModelSeq(OPTModel):
    def __init__(self, config: OPTConfig):
        with init_empty_weights():
            super().__init__(config)
        self.decoder = OPTDecoderSeq(config)
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_outputs

        return BaseModelOutputWithPast(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            hidden_states=decoder_outputs.hidden_states,
            attentions=decoder_outputs.attentions,
        )

class OPTForCausalLMSeq(OPTForCausalLM):
    def __init__(self, config):
        with init_empty_weights():
            super().__init__(config)
        self.model = OPTModelSeq(config)
        # the lm_head weight is automatically tied to the embed tokens weight
        self.lm_head = nn.Linear(config.word_embed_proj_dim, config.vocab_size, bias=False)
        # Initialize weights and apply final processing
        self.post_init()

    def preprocess_one_token(self, new_input_ids, next_tokens, use_cache=True, request_id=1):
        with torch.no_grad():
            input_ids_seq_length = new_input_ids.shape[1] - 1
            embed_tokens = self.model.decoder.embed_tokens
            pos_embeds = self.model.decoder.embed_positions
            # embed the new input tokens

            inputs_embeds = embed_tokens(new_input_ids)
            next_token_embeds = embed_tokens(next_tokens)
            bs = next_token_embeds.size(0)
            input_shape = (bs, 1)
            next_token_embeds = next_token_embeds.view(bs, 1, -1)
            attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.bool, device=inputs_embeds.device)
            p_embeds = pos_embeds(attention_mask, input_ids_seq_length)
            attention_mask = self.model.decoder._prepare_decoder_attention_mask(attention_mask, input_shape, inputs_embeds, input_ids_seq_length)
            # attention_mask[:, -1] = torch.finfo(attention_mask.dtype).min
            if self.model.decoder.project_in is not None:
                next_token_embeds = self.model.decoder.project_in(next_token_embeds)
            next_token_embeds = next_token_embeds + p_embeds
            request_token = (next_token_embeds, attention_mask) + (None, use_cache, request_id)
            return request_token

    def preprocess(self, input_ids=None, attention_mask=None, head_mask=None, past_key_values=None, inputs_embeds=None,
                   use_cache=None, output_attentions=None, output_hidden_states=None, return_dict=None, request_id=1, \
                    past_key_values_length: int = 0, **kwargs):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        pre_result = self.model.decoder.forward_pre(
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

        # add broadcast here for dist
        attention_mask = pre_result['attention_mask']
        head_mask = pre_result['head_mask']
        use_cache = use_cache
        mask_and_cache = (attention_mask, head_mask, use_cache) 
        pre_result = (pre_result['hidden_states'],) + mask_and_cache + (request_id, )

        self.return_dict = return_dict

        return pre_result
    
    # model sharders
    def _verify_shard_strategy(self, shard_strategies):
        all_decode_ids = set()
        for idx, shard_strategy in shard_strategies.items():
            decode_ids = shard_strategy.keys()
            all_decode_ids = all_decode_ids.union(decode_ids)
        assert len(list(all_decode_ids)) == len(self.model.decoder.layers), f"MUST EQUAL {len(list(all_decode_ids))}/{len(self.model.decoder.layers)}"
    
    def _shard_model_current(self, shard_strategy):
        self.model.decoder._shard_decoders(shard_strategy)
        self.model.decoder._delete_all_other_modules()

    # inplace sharding
    def _shard_model(self, shard_strategies, shard_idx):
        self.is_master = True if shard_idx == 0 else False
        if self.is_master:
            self._verify_shard_strategy(shard_strategies)
        current_shard_strategy = shard_strategies[shard_idx]
        self._shard_model_current(current_shard_strategy)
    
    def _pure_pre_and_post(self):
        sharded_model = copy.deepcopy(self)
        sharded_model.model.decoder._shard_decoders({}) # didn't delete the embeddings, etc.
        return sharded_model

    # return model instance with copy
    def shard_model(self, shard_strategies, shard_idx):
        # copy a model
        sharded_model = copy.deepcopy(self)
        sharded_model._shard_model(shard_strategies, shard_idx)
        return sharded_model
    
    def decoder_layers_to_device(self, device):
        self.model.decoder.layers = self.model.decoder.layers.to(device)

    # rewrite the forward function
    def decode(self, pre_result):
        with torch.no_grad():
            def shard_launcher(prev_result, module_shard):
                # length_prev_result = len(prev_result)
                hidden_states = prev_result[0]
                attention_mask, head_mask, use_cache, request_id = prev_result[1:]
                return module_shard(
                    hidden_states=hidden_states,
                    next_decoder_cache=None,
                    attention_mask=attention_mask,
                    head_mask=head_mask,
                    past_key_values=None,
                    use_cache=use_cache,
                    request_id=request_id,
                )

            return shard_launcher(pre_result, self.model.decoder)

    def init_kv_cache(self, b, prompt_length, token_to_generate, request_id):
        return self.model.decoder.init_kv_cache(b, prompt_length, token_to_generate, request_id)
    
    def postprocess(self, results, labels=None):
        with torch.no_grad():
            # head_mask, attention_mask should be passed with the pre_result
            # past_key_value should be stored within the model
            return_dict = self.return_dict
            hidden_states, attention_mask, head_mask, use_cache, request_id = results
            next_decoder_cache = None
            outputs = self.model.decoder.forward_post(
                hidden_states, next_decoder_cache, use_cache=use_cache, return_dict=return_dict
            )
            logits = self.lm_head(outputs[0]).contiguous()
            loss = None
            if labels is not None:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

            if not return_dict:
                output = (logits,) + outputs[1:]
                return (loss,) + output if loss is not None else output

            return CausalLMOutputWithPast(
                loss=loss,
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        pre_result = self.preprocess(
            input_ids, attention_mask, head_mask, past_key_values, inputs_embeds, use_cache, output_attentions,
            output_hidden_states, return_dict)
        
        results = self.decode(pre_result)
        
        return self.postprocess(results, labels)

        

if __name__ == '__main__':
    from qllm.models import opt 
    from qllm.utils import get_model_size
    opt_125M, tokenizer = opt.load_pretained_model_from_net('facebook/opt-125m')
    # sample text
    input_ids = tokenizer.encode("Hi, where is my dog", return_tensors="pt")
    model = OPTForCausalLMSeq.from_pretrained("facebook/opt-125m")
    model_2 = OPTForCausalLMSeq.from_pretrained("facebook/opt-125m")
    model_3 = OPTForCausalLMSeq.from_pretrained("facebook/opt-125m")

    sharding_strategy = {
        0: {
            0: {'shard': [0, 1], 'bits': [8, 8]},
            1: {'shard': [0, 1], 'bits': [8, 8]},
            2: {'shard': [0, 1], 'bits': [8, 8]},
            3: {'shard': [0, 1], 'bits': [8, 8]},
            4: {'shard': [0, 1], 'bits': [8, 8]},
            5: {'shard': [0, 1], 'bits': [8, 8]},
            6: {'shard': [0], 'bits': [8]},
        },
        1: {
            6: {'shard': [1], 'bits': [8]},
            7: {'shard': [0,1], 'bits': [8, 8]},
            8: {'shard': [0,1], 'bits': [8, 8]},
        },
        2: {
            9: {'shard': [0,1], 'bits': [8, 8]},
            10: {'shard': [0,1], 'bits': [8, 8]},
            11: {'shard': [0,1], 'bits': [8, 8]},
        }
    }
    model._shard_model(sharding_strategy, 0)
    model_2._shard_model(sharding_strategy, 1)
    model_3._shard_model(sharding_strategy, 2)

    # print model 1, 2, 3 size in MB
    print("Model 1 size: ", get_model_size(model, 'MB'))
    print("Model 2 size: ", get_model_size(model_2, 'MB'))
    print("Model 3 size: ", get_model_size(model_3, 'MB'))

    with torch.no_grad():
        res_2 = opt_125M(input_ids)
        pre_result = model.preprocess(input_ids, use_cache=True)

        # simulate the broadcast operation
        model_2.other_decode_params = model.other_decode_params
        model_3.other_decode_params = model.other_decode_params

        intermediate_results = model.decode(pre_result)
        intermediate_results = model_2.decode(intermediate_results)
        intermediate_results = model_3.decode(intermediate_results)
        
        res_1 = model.postprocess(intermediate_results, None)

    print(res_1.logits - res_2.logits)