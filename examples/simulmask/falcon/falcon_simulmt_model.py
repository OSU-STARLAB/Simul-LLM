from transformers import FalconModel, FalconPreTrainedModel, FalconConfig, FalconForCausalLM
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
    _make_causal_mask, 
    _expand_mask, build_alibi_tensor, 
    FALCON_INPUTS_DOCSTRING, 
    _CHECKPOINT_FOR_DOC,
    _CONFIG_FOR_DOC,
    FALCON_START_DOCSTRING,
    FalconAttention,
    FalconDecoderLayer
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
    "The SimulMT Falcon Model transformer outputting raw hidden-states without any specific head on top.",
    FALCON_START_DOCSTRING,
)
class FalconModelSimulMT(FalconModel):
    def __init__(self, config: FalconConfig, **kwargs):
        super().__init__(config)
        self.tokenizer = kwargs['tokenizer']
        self.waitk = kwargs['waitk']
        self.no_pos = kwargs['no_pos']
        self.old_alibi = kwargs.get('old_alibi', False)
        self.attmask_type = kwargs['attmask_type']
        self.set_prompt_lengths(kwargs['prompt1'], kwargs['prompt2'])
        if self.use_alibi and not self.no_pos and (self.attmask_type == 'simul') and not self.old_alibi:
            self.h = nn.ModuleList([FalconDecoderLayerSimulMT(config) for _ in range(config.num_hidden_layers)])

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
        
    @add_start_docstrings_to_model_forward(FALCON_INPUTS_DOCSTRING)
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
        output_attentions = output_attentions if output_attentions is not None  else self.config.output_attentions
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
        else:
            past_key_values = self._convert_to_rw_cache(past_key_values)

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape batch_size x num_heads x N x N
        # head_mask has shape n_layer x batch x num_heads x N x N
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        hidden_states = inputs_embeds

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        # Compute alibi tensor: check build_alibi_tensor documentation
        past_key_values_length = 0
        if past_key_values[0] is not None:
            past_key_values_length = past_key_values[0][0].shape[1]  # 1 because RW-cache, not standard format
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length + past_key_values_length), device=hidden_states.device)
            padding_mask = None
        else:
            attention_mask = attention_mask.to(hidden_states.device)

            if 0 in attention_mask:
                padding_mask = attention_mask
            else:
                padding_mask = None

        causal_mask = self._prepare_attn_mask(
            attention_mask,
            input_shape=(batch_size, seq_length),
            past_key_values_length=past_key_values_length,
            input_ids=input_ids
        )

        if self.use_alibi:
            if not self.no_pos and self.attmask_type == 'simul' and not self.old_alibi:
                alibi = build_simul_alibi_tensor(causal_mask, self.num_heads, dtype=hidden_states.dtype)
            else:
                alibi = build_alibi_tensor(attention_mask, self.num_heads, dtype=hidden_states.dtype)
            if self.no_pos:
                alibi = torch.zeros(alibi.shape, device=alibi.device, dtype=alibi.dtype)
        else:
            alibi = None
            if position_ids is None:
                device = input_ids.device if input_ids is not None else inputs_embeds.device
                position_ids = torch.arange(
                    past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
                )
                position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
            else:
                position_ids = position_ids.view(-1, seq_length).long()


        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

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
                    position_ids,
                    head_mask[i],
                    padding_mask,
                )
            else:
                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    head_mask=head_mask[i],
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    alibi=alibi,
                    padding_mask=padding_mask,
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

        if presents is not None:
            presents = self._convert_cache_to_standard_format(presents, batch_size)

        if not return_dict:
            return tuple(v for v in [hidden_states, presents, all_hidden_states, all_self_attentions] if v is not None)

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )
@add_start_docstrings(
    "The Falcon SimulMT Model transformer with a language modeling head on top (linear layer with weights tied to the input embeddings).",
    FALCON_START_DOCSTRING,
)
class FalconForCausalLMSimulMT(FalconForCausalLM, FalconPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: FalconConfig, **kwargs):
        if kwargs['no_pos']:
            config.alibi = True
        FalconPreTrainedModel.__init__(self, config)
        self.tokenizer = kwargs['tokenizer']

        self.transformer = FalconModelSimulMT(config, **kwargs)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], CausalLMOutputWithCrossAttentions]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
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
        r"""
        Generates sequences of token ids for models with a language modeling head using **greedy decoding** and can be
        used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

        <Tip warning={true}>

        In most cases, you do not need to call [`~generation.GenerationMixin.greedy_search`] directly. Use generate()
        instead. For an overview of generation strategies and code examples, check the [following
        guide](../generation_strategies).

        </Tip>


        Parameters:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            stopping_criteria (`StoppingCriteriaList`, *optional*):
                An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
                used to tell if the generation loop should stop.
            max_new_tokens (`int`, *optional*, defaults to 20):
                **DEPRECATED**. Use `logits_processor` or `stopping_criteria` directly to cap the number of generated
                tokens. The maximum length of the sequence to be generated.
            model_kwargs:
                Additional model specific keyword arguments will be forwarded to the `forward` function of the model.
                If model is an encoder-decoder model the kwargs should include `encoder_outputs`.

        Return:
            [`~generation.GreedySearchDecoderOnlyOutput`], [`~generation.GreedySearchEncoderDecoderOutput`] or
            `torch.LongTensor`: A `torch.LongTensor` containing the generated tokens (default behaviour) or a
            [`~generation.GreedySearchDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
            `return_dict_in_generate=True` or a [`~generation.GreedySearchEncoderDecoderOutput`] if
            `model.config.is_encoder_decoder=True`.

        Examples:

        ```python
        >>> from transformers import (
        ...     AutoTokenizer,
        ...     AutoModelForCausalLM,
        ...     LogitsProcessorList,
        ...     MinLengthLogitsProcessor,
        ...     StoppingCriteriaList,
        ...     MaxLengthCriteria,
        ... )

        >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
        >>> model = AutoModelForCausalLM.from_pretrained("gpt2")

        >>> # set pad_token_id to eos_token_id because GPT2 does not have a PAD token
        >>> model.generation_config.pad_token_id = model.generation_config.eos_token_id

        >>> input_prompt = "It might be possible to"
        >>> input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids

        >>> # instantiate logits processors
        >>> logits_processor = LogitsProcessorList(
        ...     [
        ...         MinLengthLogitsProcessor(10, eos_token_id=model.generation_config.eos_token_id),
        ...     ]
        ... )
        >>> stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(max_length=20)])

        >>> outputs = model.greedy_search(
        ...     input_ids, logits_processor=logits_processor, stopping_criteria=stopping_criteria
        ... )

        >>> tokenizer.batch_decode(outputs, skip_special_tokens=True)
        ["It might be possible to get a better understanding of the nature of the problem, but it's not"]
        ```"""
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
                        past_key_values.append((key[:, :, :-1], val[:, :, :-1]))
                    outputs.past_key_values = tuple(past_key_values)
                break

        return {
            'prediction': input_ids[0][-token_count:-1] if back_track_kv else input_ids[0][-token_count:],
            'sequences': input_ids,
            'past_key_values': outputs.past_key_values
        }

class FalconDecoderLayerSimulMT(FalconDecoderLayer):
    def __init__(self, config: FalconConfig):
        super().__init__(config)
        self.self_attention = (
            FalconAttentionSimulMT(config)
        )

class FalconAttentionSimulMT(FalconAttention):
    def __init__(self, config: FalconConfig):
        super().__init__(config)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        alibi: Optional[torch.Tensor],
        attention_mask: torch.Tensor,
        position_ids: Optional[torch.LongTensor] = None,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        head_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        padding_mask: Optional[torch.LongTensor] = None,
    ):
        fused_qkv = self.query_key_value(hidden_states)  # [batch_size, seq_length, 3 x hidden_size]
        num_kv_heads = self.num_heads if self.new_decoder_architecture else self.num_kv_heads
        # 3 x [batch_size, seq_length, num_heads, head_dim]
        (query_layer, key_layer, value_layer) = self._split_heads(fused_qkv)

        batch_size, query_length, _, _ = query_layer.shape

        query_layer = query_layer.transpose(1, 2).reshape(batch_size * self.num_heads, query_length, self.head_dim)
        key_layer = key_layer.transpose(1, 2).reshape(
            batch_size * num_kv_heads,
            query_length,
            self.head_dim,
        )
        value_layer = value_layer.transpose(1, 2).reshape(batch_size * num_kv_heads, query_length, self.head_dim)

        past_kv_length = 0 if layer_past is None else layer_past[0].shape[1]
        query_layer, key_layer = self.maybe_rotary(query_layer, key_layer, past_kv_length, position_ids)

        if layer_past is not None:
            past_key, past_value = layer_past
            # concatenate along seq_length dimension:
            #  - key: [batch_size * self.num_heads, kv_length, head_dim]
            #  - value: [batch_size * self.num_heads, kv_length, head_dim]
            key_layer = torch.cat((past_key, key_layer), dim=1)
            value_layer = torch.cat((past_value, value_layer), dim=1)

        _, kv_length, _ = key_layer.shape
        if use_cache:
            present = (key_layer, value_layer)
        else:
            present = None

        float_min = torch.finfo(query_layer.dtype).min
        attention_mask_float = (attention_mask * 1.0).masked_fill(attention_mask, float_min).to(query_layer.dtype)

        query_layer_ = query_layer.reshape(batch_size, self.num_heads, -1, self.head_dim)
        key_layer_ = key_layer.reshape(batch_size, num_kv_heads, -1, self.head_dim)
        value_layer_ = value_layer.reshape(batch_size, num_kv_heads, -1, self.head_dim)

        if alibi is None:
            if hasattr(F, "scaled_dot_product_attention") and not output_attentions:
                # TODO: deprecate this once we add FA2 support in Falcon
                logger.warning_once(
                    "The current implementation of Falcon calls `torch.scaled_dot_product_attention` directly, this will be deprecated in the"
                    " future in favor of the `BetterTransformer` API. Please install the latest optimum library with `pip install -U optimum` and call "
                    "`model.to_bettertransformer()` to benefit from `torch.scaled_dot_product_attention` and future performance optimizations."
                )

                attn_output = F.scaled_dot_product_attention(
                    query_layer_, key_layer_, value_layer_, attention_mask_float, 0.0, is_causal=False
                )
                attention_scores = None
            else:
                attention_scores = query_layer_ @ key_layer_.transpose(-1, -2)
                attention_scores /= math.sqrt(self.head_dim)

                attention_scores = F.softmax(
                    attention_scores + attention_mask_float, dim=-1, dtype=hidden_states.dtype
                )
                attn_output = attention_scores @ value_layer_

            attn_output = attn_output.view(batch_size, self.num_heads, query_length, self.head_dim)
            attn_output = attn_output.permute(0, 2, 1, 3)
            attn_output = attn_output.reshape(batch_size, query_length, self.num_heads * self.head_dim)

            output_tensor = self.dense(attn_output)

            if output_attentions:
                return output_tensor, present, attention_scores
            else:
                return output_tensor, present

        else:
            matmul_result = query_layer_ @ key_layer_.transpose(-1, -2)

            # change view to [batch_size, num_heads, q_length, kv_length]
            attention_scores = matmul_result.view(batch_size, self.num_heads, query_length, kv_length)

            # cast attention scores to fp32, compute scaled softmax and cast back to initial dtype - [batch_size, num_heads, q_length, kv_length]
            input_dtype = attention_scores.dtype
            # `float16` has a minimum value of -65504.0, whereas `bfloat16` and `float32` have a minimum value of `-3.4e+38`
            if input_dtype == torch.float16 or input_dtype == torch.bfloat16:
                attention_scores = attention_scores.to(torch.float32)
            # Matt (HF) note: We could possibly use F.scaled_dot_product_attention here too, by
            # adding (alibi * self.inv_norm_factor) to attention_mask_float. I think this would be mathematically
            # equivalent and more performant, but there might be a numerical difference. If you're reading this
            # and you'd like to experiment and maybe file a PR, feel free!
            attention_logits = attention_scores + alibi
            attention_logits *= self.inv_norm_factor
            attention_probs = F.softmax(attention_logits + attention_mask_float, dim=-1, dtype=hidden_states.dtype)
            # [batch_size, num_heads, q_length, kv_length]
            attention_probs = self.attention_dropout(attention_probs)

            if head_mask is not None:
                attention_probs = attention_probs * head_mask

            # change view [batch_size, num_heads, q_length, kv_length]
            attention_probs_reshaped = attention_probs.view(batch_size, self.num_heads, query_length, kv_length)

            # matmul: [batch_size * num_heads, q_length, head_dim]
            context_layer = (attention_probs_reshaped @ value_layer_).flatten(0, 1)

            # change view [batch_size, q_length, num_heads * head_dim]
            context_layer = self._merge_heads(context_layer)

            output_tensor = self.dense(context_layer)

            if output_attentions:
                return output_tensor, present, attention_probs
            else:
                return output_tensor, present