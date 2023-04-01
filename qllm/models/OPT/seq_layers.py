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


class OPTDecoderLayerSharded(nn.Module):
    def __init__(self, config: OPTConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = OPTAttention(
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
    

    def shard(self, shard_strategy):
        shard = shard_strategy['shard']
        bits = shard_strategy['bits']
        assert len(shard) == len(bits), "shard and bitwidths should have the same length"
        self.set_partition(shard)
        # DO quantization and remove the unsed parameters here
    
    def FFN_PART(self, hidden_states, use_cache=False):
        if use_cache:
            hidden_states, present_key_value = hidden_states
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

        if use_cache:
            outputs += (present_key_value,)
        
        return outputs

    def SELFATTEN_PART(self, hidden_states, attention_mask, past_key_value, layer_head_mask, output_attentions=False):
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
        return hidden_states, present_key_value


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
            hidden_states = self.SELFATTEN_PART(hidden_states, attention_mask, past_key_value, layer_head_mask, output_attentions=output_attentions)
            if 1 in self.partitions:
                outputs = self.FFN_PART(hidden_states, use_cache=use_cache)
                return outputs
            else:
                return hidden_states
        # Fully Connected
        if 1 in self.partitions:
            outputs = self.FFN_PART(hidden_states, use_cache=use_cache)
            return outputs
        raise ValueError("No partition is specified")

class OPTDecoderSeq(OPTDecoder):
    def __init__(self, config: OPTConfig):
        super().__init__(config)
        self.layers = nn.ModuleList([OPTDecoderLayerSharded(config) for _ in range(config.num_hidden_layers)])
    
    def forward_pre(self, 
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,):
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

        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

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

    def forward_post(self, hidden_states, next_decoder_cache, use_cache=False, return_dict=False):
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

    def shard_decoders(self, sharding_strategy):
        layer_idxs = list(sharding_strategy.keys())
        for i in range(self.get_decoder_layer_num()):
            if i not in layer_idxs:
                self.layers[i] = None
        # for each layer, start sharding
        for layer_idx in layer_idxs:
            self.layers[layer_idx].shard(sharding_strategy[layer_idx])

    # only remains inference related part
    def forward(
        self,
        hidden_states: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        next_decoder_cache: Optional[List[torch.FloatTensor]] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        
        for idx, decoder_layer in enumerate(self.layers):
            if decoder_layer is None:
                continue
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            past_key_value = past_key_values[idx] if past_key_values is not None else None
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                past_key_value=past_key_value,
                output_attentions=False,
                use_cache=use_cache,
            )
            if decoder_layer.is_only_self_attention():
                return layer_outputs + (next_decoder_cache,) # contains hidden_states and present_key_value
                                     # but is till ok to pass to next decoder layer

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[1],)

        return (hidden_states, next_decoder_cache)
        
# in generation, actually did nothing with model function. 
# only to decoder
class OPTModelSeq(OPTModel):
    def __init__(self, config: OPTConfig):
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
        super().__init__(config)
        self.model = OPTModelSeq(config)
        # the lm_head weight is automatically tied to the embed tokens weight
        self.lm_head = nn.Linear(config.word_embed_proj_dim, config.vocab_size, bias=False)
        # Initialize weights and apply final processing
        self.post_init()
    
    def preprocess(self, input_ids=None, attention_mask=None, head_mask=None, past_key_values=None, inputs_embeds=None,
                   use_cache=None, output_attentions=None, output_hidden_states=None, return_dict=None):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        return input_ids, attention_mask, head_mask, past_key_values, inputs_embeds, use_cache, output_attentions, \
               output_hidden_states, return_dict
    
    # hidden_states = self.decoder.embed_tokens(input_ids)
    # pos_embeds = self.decoder.embed_positions(attention_mask)
    # hidden_states = hidden_states + pos_embeds

    # rewrite the forward function
    def decode(self, input_ids, attention_mask, head_mask, past_key_values, inputs_embeds, use_cache, output_attentions,
               output_hidden_states, return_dict):
        
        # return self.model.decoder(
        #     input_ids=input_ids,
        #     attention_mask=attention_mask,
        #     head_mask=head_mask,
        #     past_key_values=past_key_values,
        #     inputs_embeds=inputs_embeds,
        #     use_cache=use_cache,
        #     output_attentions=output_attentions,
        #     output_hidden_states=output_hidden_states,
        #     return_dict=return_dict,
        # )
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
        )

        # broadcast here
        past_key_values = pre_result['past_key_values']
        attention_mask = pre_result['attention_mask']
        head_mask = pre_result['head_mask']
        # shard
        # self.model.decoder.shard_decoders()
        # result = self.model.decoder(
        #     hidden_states=pre_result['hidden_states'],
        #     next_decoder_cache=pre_result['next_decoder_cache'],
        #     head_mask=head_mask,
        #     attention_mask=attention_mask,
        #     past_key_values=past_key_values,
        #     use_cache=use_cache,
        # )
        # # decoder layer number
        # decoder_num = self.model.decoder.get_decoder_layer_num()

        # SUM, 12
        shard_1_strategy = {
            0: {'shard': [0, 1], 'bits': [8, 8]},
            1: {'shard': [0, 1], 'bits': [8, 8]},
            2: {'shard': [0, 1], 'bits': [8, 8]},
            3: {'shard': [0, 1], 'bits': [8, 8]},
            4: {'shard': [0, 1], 'bits': [8, 8]},
            5: {'shard': [0, 1], 'bits': [8, 8]},
            6: {'shard': [0], 'bits': [8]},
        }

        shard_2_strategy = {
            6: {'shard': [1], 'bits': [8]},
            7: {'shard': [0,1], 'bits': [8, 8]},
            8: {'shard': [0,1], 'bits': [8, 8]},
            9: {'shard': [0,1], 'bits': [8, 8]},
            10: {'shard': [0,1], 'bits': [8, 8]},
            11: {'shard': [0,1], 'bits': [8, 8]},
        }

        def shard_launcher(prev_result, module_shard):
            length_prev_result = len(prev_result)
            if length_prev_result == 3: # ATTN
                return module_shard(
                    hidden_states=prev_result[:2],
                    next_decoder_cache=prev_result[2],
                    head_mask=head_mask,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                )
            else:
                return module_shard(
                    hidden_states=prev_result[0],
                    next_decoder_cache=prev_result[1],
                    head_mask=head_mask,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                )
        
        import copy
        decoder_1 = copy.deepcopy(self.model.decoder)
        decoder_2 = copy.deepcopy(self.model.decoder)

        decoder_1.shard_decoders(shard_1_strategy)
        decoder_2.shard_decoders(shard_2_strategy)

        pre_result = (pre_result['hidden_states'],pre_result['next_decoder_cache'])
        pre_result = shard_launcher(pre_result, decoder_1)
        pre_result = shard_launcher(pre_result, decoder_2)
        result = pre_result

        # result = decoder_1(
        #     hidden_states=pre_result['hidden_states'],
        #     next_decoder_cache=pre_result['next_decoder_cache'],
        #     head_mask=head_mask,
        #     attention_mask=attention_mask,
        #     past_key_values=past_key_values,
        #     use_cache=use_cache,
        # )

        # result = decoder_2(
        #     hidden_states=result[:2],
        #     next_decoder_cache=result[2],
        #     head_mask=head_mask,
        #     attention_mask=attention_mask,
        #     past_key_values=past_key_values,
        #     use_cache=use_cache,
        # )

        after_result = self.model.decoder.forward_post(
            *result, use_cache=use_cache, return_dict=return_dict
        )
        return after_result

    def postprocess(self, outputs, labels=None):
        logits = self.lm_head(outputs[0]).contiguous()
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

        return logits, loss

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
        input_ids, attention_mask, head_mask, past_key_values, inputs_embeds, use_cache, output_attentions, \
        output_hidden_states, return_dict = self.preprocess(
            input_ids, attention_mask, head_mask, past_key_values, inputs_embeds, use_cache, output_attentions,
            output_hidden_states, return_dict)

        outputs = self.decode(input_ids, attention_mask, head_mask, past_key_values, inputs_embeds, use_cache,
                              output_attentions, output_hidden_states, return_dict)
        logits, loss = self.postprocess(outputs, labels)

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

if __name__ == '__main__':
    model = OPTForCausalLMSeq.from_pretrained("facebook/opt-125m")
    from qllm.models import opt 
    opt_125M, tokenizer = opt.load_pretained_model_from_net('facebook/opt-125m')
    # sample text
    input_ids = tokenizer.encode("Hi, where is my dog", return_tensors="pt")

    with torch.no_grad():
        res_1 = model.generate(input_ids)
        res_2 = opt_125M.generate(input_ids)

    print(res_1 - res_2)