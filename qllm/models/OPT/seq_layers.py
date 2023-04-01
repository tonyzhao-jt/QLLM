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
    CausalLMOutputWithPast
)

from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.utils import logging
logger = logging.get_logger(__name__)

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import random

class OPTDecoderSeq(OPTDecoder):
    def __init__(self, config: OPTConfig):
        super().__init__(config)
    
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

    def shard_decoders(self,
                       start_idx: int=0,
                       end_idx: int=0):
        if end_idx == 0:
            end_idx = self.get_decoder_layer_num()
        # make sure the start_idx and end_idx are valid
        assert start_idx >= 0 and start_idx < end_idx and end_idx <= self.get_decoder_layer_num()
        # make all other layers to be None
        for i in range(self.get_decoder_layer_num()):
            if i < start_idx or i >= end_idx:
                self.layers[i] = None
    
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

        
        import copy
        decoder_1 = copy.deepcopy(self.model.decoder)
        decoder_2 = copy.deepcopy(self.model.decoder)

        decoder_1.shard_decoders(0, 5)
        decoder_2.shard_decoders(5)
    
        result = decoder_1(
            hidden_states=pre_result['hidden_states'],
            next_decoder_cache=pre_result['next_decoder_cache'],
            head_mask=head_mask,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )

        result = decoder_2(
            hidden_states=result[0],
            next_decoder_cache=result[1],
            head_mask=head_mask,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )

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