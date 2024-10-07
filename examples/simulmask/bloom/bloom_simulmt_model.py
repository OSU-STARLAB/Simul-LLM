from transformers import BloomModel, BloomPreTrainedModel, BloomConfig, BloomForCausalLM
import torch.nn as nn
from torch.nn import functional as F
import math
import torch
from torch.nn import CrossEntropyLoss
from transformers.utils import (
    add_code_sample_docstrings,
    add_start_docstrings_to_model_forward,
    add_start_docstrings,
    logging,
)
from typing import Tuple, Union, Optional
from transformers.models.falcon.modeling_falcon import(
    _expand_mask,
    _make_causal_mask
)
from transformers.models.bloom.modeling_bloom import(
    build_alibi_tensor, 
    BLOOM_INPUTS_DOCSTRING, 
    _CHECKPOINT_FOR_DOC,
    _CONFIG_FOR_DOC,
    BLOOM_START_DOCSTRING,
    BloomAttention,
    BloomBlock,
    dropout_add
)  
from transformers.modeling_outputs import(
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions
)

from transformers.generation.utils import GreedySearchOutput
from transformers.generation.stopping_criteria import StoppingCriteriaList
from examples.simulmask.utils import build_simul_alibi_tensor, make_mask_batch

logger = logging.get_logger(__name__)
    
@add_start_docstrings(
    "The SimulMT Bloom Model transformer outputting raw hidden-states without any specific head on top.",
    BLOOM_START_DOCSTRING,
)
class BloomModelSimulMT(BloomModel):
    def __init__(self, config: BloomConfig, **kwargs):
        super().__init__(config)
        self.tokenizer = kwargs['tokenizer']
        self.waitk = kwargs['waitk']
        self.no_pos = kwargs['no_pos']
        self.old_alibi = kwargs.get('old_alibi', False)
        self.attmask_type = kwargs['attmask_type']
        self.set_prompt_lengths(kwargs['prompt1'], kwargs['prompt2'])
        if not self.no_pos and (self.attmask_type == 'simul') and not self.old_alibi:
            self.h = nn.ModuleList([BloomBlock(config) for _ in range(config.num_hidden_layers)])

    def set_prompt_lengths(self, prompt1, prompt2):
        self.tok_prompt1 = self.tokenizer([prompt1], truncation=True, padding=False, max_length=30, return_overflowing_tokens=False, return_length=False) 
        self.tok_prompt2 = self.tokenizer([prompt2], truncation=True, padding=False, max_length=30, return_overflowing_tokens=False, return_length=False)
        self.p1_len = len(self.tok_prompt1["input_ids"][0])
        self.p2_len = len(self.tok_prompt2["input_ids"][0])

    def _prepare_attn_mask(
        self, attention_mask: torch.Tensor, input_shape: Tuple[int, int], past_key_values_length: int, input_ids: torch.Tensor
    ) -> torch.BoolTensor:
        # Create a causal mask
        # The attention mask we receive as input should cover the whole extended sequence, including any past
        # cache, so its shape should be [batch_size, seq_length + past_key_values_length]
        # The output shape will be [batch_size, 1, seq_length, seq_length + past_key_values_length]
        if input_shape[1] + past_key_values_length != attention_mask.shape[1]:
            raise ValueError(
                "Attention mask shape should be (batch_size, seq_length + past_key_values_length)"
                f" but is {attention_mask.shape} with input_ids shape {input_shape} and past length"
                f" {past_key_values_length}."
            )
        combined_attention_mask = None
        device = attention_mask.device
        bsz, seq_length = input_shape

        if seq_length > 1:
            if self.attmask_type == 'causal':
                combined_attention_mask = _make_causal_mask(input_shape, device=device, past_key_values_length=past_key_values_length)
            elif self.attmask_type == 'full':
                combined_attention_mask = None
            else:
                combined_attention_mask = make_mask_batch(input_ids=input_ids, att_len=seq_length, device=device, p1_len=self.p1_len, p2_len=self.p2_len, tokenizer=self.tokenizer, waitk=self.waitk)
        
        # [batch_size, seq_length + past_key_values_length] -> [batch_size, 1, seq_length, seq_length + past_key_values_length]
        expanded_attn_mask = _expand_mask(attention_mask, past_key_values_length=past_key_values_length)
        combined_attention_mask = (
            expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask | combined_attention_mask
        )

        return combined_attention_mask
        
    @add_start_docstrings_to_model_forward(BLOOM_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPastAndCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor, ...], BaseModelOutputWithPastAndCrossAttentions]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

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

        causal_mask = self._prepare_attn_mask(
            attention_mask,
            input_shape=(batch_size, seq_length),
            past_key_values_length=past_key_values_length,
            input_ids=input_ids
        )

        if not self.no_pos and self.attmask_type == 'simul' and not self.old_alibi:
            alibi = build_simul_alibi_tensor(causal_mask, self.num_heads, dtype=hidden_states.dtype).reshape(-1, seq_length, seq_length)
        else:
            alibi = build_alibi_tensor(attention_mask, self.num_heads, dtype=hidden_states.dtype)
        if self.no_pos:
            alibi = torch.zeros(alibi.shape, device=alibi.device, dtype=alibi.dtype)


        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, use_cache=use_cache, output_attentions=output_attentions)

                    return custom_forward

                outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    alibi,
                    causal_mask,
                    layer_past,
                    head_mask[i],
                )
            else:
                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=causal_mask,
                    head_mask=head_mask[i],
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    alibi=alibi,
                )

            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)

        # Add last hidden state
        hidden_states = self.ln_f(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, presents, all_hidden_states, all_self_attentions] if v is not None)

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )
    
@add_start_docstrings(
    "The Bloom SimulMT Model transformer with a language modeling head on top (linear layer with weights tied to the input embeddings).",
    BLOOM_START_DOCSTRING,
)
class BloomForCausalLMSimulMT(BloomForCausalLM, BloomPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: BloomConfig, **kwargs):
        if kwargs['no_pos']:
            config.alibi = True
        BloomPreTrainedModel.__init__(self, config)
        self.tokenizer = kwargs['tokenizer']

        self.transformer = BloomModelSimulMT(config, **kwargs)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(BLOOM_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=CausalLMOutputWithCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
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

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(lm_logits.device)
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
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
    
    def prepare_inputs_for_generation_out(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> dict:
        cut_input_ids = input_ids
        if past_key_values is not None:
            past_key_length = past_key_values[0][0].shape[2]
            cut_input_ids = input_ids[:, past_key_length:]
            #Handles case where tokenizer merges two token predictions into one
            if cut_input_ids.shape[1] == 0:
                cut_input_ids = input_ids[:, -1:]

        return {
            "input_ids": cut_input_ids,
            "past_key_values": past_key_values,
            "use_cache": True,
        }

    def generate_out(
        self,
        input_ids: torch.LongTensor,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        **model_kwargs,
    ) -> Union[GreedySearchOutput, torch.LongTensor]:

        token_count = 0
        while True:
            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation_out(input_ids, **model_kwargs)

            # forward pass to get next token
            outputs = self(
                **model_inputs,
                return_dict=True,
            )

            next_token_logits = outputs.logits[:, -1, :]

            # argmax
            next_tokens = torch.argmax(next_token_logits, dim=-1)
            token_count += 1
            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)

            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )
            is_done, back_track_kv = stopping_criteria[0](input_ids, token_count)
            if is_done:
                # Handles case where we need to backtrack after predicting one word too far
                if back_track_kv:
                    past_key_values = []
                    for key, val in outputs.past_key_values:
                        past_key_values.append((key[:, :, :-1], val[:, :-1]))
                    outputs.past_key_values = tuple(past_key_values)
                break

        return {
            'prediction': input_ids[0][-token_count:-1] if back_track_kv else input_ids[0][-token_count:],
            'sequences': input_ids,
            'past_key_values': outputs.past_key_values
        }