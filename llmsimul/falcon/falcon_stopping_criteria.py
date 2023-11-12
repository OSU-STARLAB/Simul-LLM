import torch
from transformers.generation.stopping_criteria import StoppingCriteria
from typing import Optional

'''
Custom stopping criteria, necessary to maximize inference throughput. Takes a maximum generated sequence
length and takes a "stop" token in the form of the '}' character, which indicates an end-bound for the
target token in the current evaluation framework. 

Could've been passed as a list of the existing criteria and a custom one, but just making a single criteria
seemed easier. Intended only for LLM use, at the moment.
'''

class StopTokenAndMaxLengthCriteria(StoppingCriteria):
    '''
    Custom `StoppingCriteria` that can be used to stop generation whenever the full generated number of tokens exceeds 
    `max_length`, including the input sequence. Intended to be used with a specified starting sequence length and
    a specified maximum number of new tokens.

    Args:
        start_length (`int`):
            The specified input prompt length.
        max_new_tokens (`int`):
            The maximum number of new tokens that may be generated as part of the model's decoding.
        max_position_embeddings (`int`, *optional*):
            The maximum model length, as defined by the model's `config.max_position_embeddings` attribute.
        eoseq_ids (`torch.LongTensor` of shape [1, len(eoseq_tokens)]):
            The list of specified token ids that result in stopping generation for the entire batch. 
            Any number of eoseq tokens may be specified.
    '''
    def __init__(
            self,
            start_length: int,
            max_new_tokens: int,
            eoseq_ids: torch.LongTensor,
            max_position_embeddings: Optional[int] = None,
        ):
        
        self.start_length = start_length
        self.max_new_tokens = max_new_tokens
        self.max_position_embeddings = max_position_embeddings
        self.eoseq_ids = eoseq_ids

        self.max_length = self.start_length + self.max_new_tokens


    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        cur_len = input_ids.shape[-1]
        is_done = cur_len >= self.max_length or torch.all(torch.any(torch.isin(input_ids[:, -1], self.eoseq_ids), -1)).item()
        if self.max_position_embeddings is not None and not is_done and cur_len >= self.max_position_embeddings:
            logger.warning_once(
                "This is a friendly reminder - the current text generation call will exceed the model's predefined "
                f"maximum length ({self.max_position_embeddings}). Depending on the model, you may observe "
                "exceptions, performance degradation, or nothing at all."
            )
        return is_done
