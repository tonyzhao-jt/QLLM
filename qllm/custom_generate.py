# Sequential execute the model but produce the same result as the original generate
import torch 
import torch.nn as nn 
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from transformers import LogitsProcessorList, StoppingCriteriaList
import warnings
import copy
class SequentialGenerate:
    def __init__(
            self,
            model:nn.Module,
            input_ids:torch.LongTensor,
            model_layers:Optional[List[nn.Module]]=None,
            generation_config=None,
            prepare_inputs_for_generation=None,
            _update_model_kwargs_for_generation=None,
            model_config=None) -> None:
        
        self.generation_config = generation_config if generation_config is not None else model.generation_config
        self.prepare_inputs_for_generation = prepare_inputs_for_generation if prepare_inputs_for_generation is not None else model.prepare_inputs_for_generation
        self._update_model_kwargs_for_generation = _update_model_kwargs_for_generation if _update_model_kwargs_for_generation is not None else model._update_model_kwargs_for_generation
        self.config = model.config if model_config is None else model_config
        # deep copy
        self.generation_config = copy.deepcopy(self.generation_config)
        self.prepare_inputs_for_generation = copy.deepcopy(self.prepare_inputs_for_generation)
        self._update_model_kwargs_for_generation = copy.deepcopy(self._update_model_kwargs_for_generation)
        self.config = copy.deepcopy(self.config)

        # generate logits processor and stopping criteria
        generation_config = self.generation_config
        logits_processor = LogitsProcessorList()
        inputs_tensor, model_input_name, model_kwargs = model._prepare_model_inputs(
            input_ids, generation_config.bos_token_id, {}
        )
        input_ids_seq_length = input_ids.shape[-1]
        # 8. prepare distribution pre_processing samplers
        logits_processor = model._get_logits_processor(
            generation_config=generation_config,
            input_ids_seq_length=input_ids_seq_length,
            encoder_input_ids=inputs_tensor,
            prefix_allowed_tokens_fn=None,
            logits_processor=logits_processor,
        )

        stopping_criteria = StoppingCriteriaList()
        # 9. prepare stopping criteria
        stopping_criteria = model._get_stopping_criteria(
            generation_config=generation_config, stopping_criteria=stopping_criteria
        )

        self.logits_processor = logits_processor
        self.stopping_criteria = stopping_criteria

        self.seq_execute = False
        if model_layers is not None: # enable sequential execution
            self.seq_execute = True
            self.model_layers = model_layers
        else:
            self.seq_execute = False
            self.model = model

    def enqueue_seq_layers(self, **kwargs):
        pass
    
    # use the same interface as the transformers, but decompose the while loop
    # without specify the output type
    # remove the support for sync gpus
    def greedy_search(self,
        input_ids: torch.LongTensor,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        # synced_gpus: Optional[bool] = False,
        **model_kwargs,):
        
        # init values
        logits_processor = logits_processor if logits_processor is not None else self.logits_processor
        stopping_criteria = stopping_criteria if stopping_criteria is not None else self.stopping_criteria
        # if max_length is not None:
        #     warnings.warn(
        #         "`max_length` is deprecated in this function, use"
        #         " `stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])` instead.",
        #         UserWarning,
        #     )
        #     stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
        pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        eos_token_id_tensor = torch.tensor(eos_token_id).to(input_ids.device) if eos_token_id is not None else None
        output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
        output_attentions = (
            output_attentions if output_attentions is not None else self.generation_config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate
            if return_dict_in_generate is not None
            else self.generation_config.return_dict_in_generate
        )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        # keep track of which sequences are already finished
        unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)

        return self.greedy_search_subloop(input_ids,
                            output_attentions,
                            output_hidden_states,
                            logits_processor,
                            return_dict_in_generate,
                            output_scores,
                            eos_token_id,
                            pad_token_id,
                            eos_token_id_tensor,
                            stopping_criteria,
                            unfinished_sequences,
                            scores, decoder_attentions, cross_attentions, decoder_hidden_states,
                            model_kwargs
                            )
        

        # if return_dict_in_generate:
        #     if self.config.is_encoder_decoder:
        #         return GreedySearchEncoderDecoderOutput(
        #             sequences=input_ids,
        #             scores=scores,
        #             encoder_attentions=encoder_attentions,
        #             encoder_hidden_states=encoder_hidden_states,
        #             decoder_attentions=decoder_attentions,
        #             cross_attentions=cross_attentions,
        #             decoder_hidden_states=decoder_hidden_states,
        #         )
        #     else:
        #         return GreedySearchDecoderOnlyOutput(
        #             sequences=input_ids,
        #             scores=scores,
        #             attentions=decoder_attentions,
        #             hidden_states=decoder_hidden_states,
        #         )
        # else:

    def greedy_search_subloop(self,
                          input_ids,
                          output_attentions,
                          output_hidden_states,
                          logits_processor,
                          return_dict_in_generate,
                          output_scores,
                          eos_token_id,
                          pad_token_id,
                          eos_token_id_tensor,
                          stopping_criteria,
                          unfinished_sequences,
                          scores, decoder_attentions, cross_attentions, decoder_hidden_states,
                          model_kwargs):
        # prepare model inputs
        model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

        # forward pass to get next token
        if not self.seq_execute:
            outputs = self.model(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
        else:
            outputs = self.enqueue(model_inputs)

        next_token_logits = outputs.logits[:, -1, :]

        # pre-process distribution
        next_tokens_scores = logits_processor(input_ids, next_token_logits)

        # Store scores, attentions and hidden_states when required
        if return_dict_in_generate:
            if output_scores:
                scores += (next_tokens_scores,)
            if output_attentions:
                decoder_attentions += (
                    (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                )
                if self.config.is_encoder_decoder:
                    cross_attentions += (outputs.cross_attentions,)

            if output_hidden_states:
                decoder_hidden_states += (
                    (outputs.decoder_hidden_states,)
                    if self.config.is_encoder_decoder
                    else (outputs.hidden_states,)
                )

        # argmax
        next_tokens = torch.argmax(next_tokens_scores, dim=-1)

        # finished sentences should have their next token be a padding token
        if eos_token_id is not None:
            if pad_token_id is None:
                raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

        # update generated ids, model inputs, and length for next step
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        model_kwargs = self._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
        )

        # if eos_token was found in one sentence, set sentence to finished
        if eos_token_id_tensor is not None:
            unfinished_sequences = unfinished_sequences.mul(
                next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
            )

        # stop when each sentence is finished, or if we exceed the maximum length
        if unfinished_sequences.max() == 0 or stopping_criteria(input_ids, scores):
            return input_ids

        # continue the loop
        return self.greedy_search_subloop(input_ids,
                                        output_attentions,
                                        output_hidden_states,
                                        logits_processor,
                                        return_dict_in_generate,
                                        output_scores,
                                        eos_token_id,
                                        pad_token_id,
                                        eos_token_id_tensor,
                                        stopping_criteria,
                                        unfinished_sequences,
                                        scores, decoder_attentions, cross_attentions, decoder_hidden_states,
                                        model_kwargs)


