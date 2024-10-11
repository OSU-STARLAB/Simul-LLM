import torch
from transformers.generation.stopping_criteria import StoppingCriteria

'''
Custom stopping criteria, necessary to maximize inference throughput. Takes a maximum generated sequence
length and takes a "stop" token in the form of the '}' character, which indicates an end-bound for the
target token in the current evaluation framework. 

Could've been passed as a list of the existing criteria and a custom one, but just making a single criteria
seemed easier. Intended only for LLM use, at the moment.
'''
class StopTokenCriteria(StoppingCriteria):
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
            max_new_tokens: int,
            tokenizer,
        ):
        
        self.max_new_tokens = max_new_tokens
        self.tokenizer = tokenizer


    def __call__(self, input_ids: torch.LongTensor, token_count, **kwargs) -> bool:
        token_pred = self.tokenizer.decode(input_ids[0][-1:])
        token_preds = self.tokenizer.decode(input_ids[0][-token_count:])
        is_done = False
        back_track_kv = False
        terminating_tok = [".", ",", ":", ";", "?", "!"]
        if ' ' in token_preds[1:] or token_pred in terminating_tok or 'endoftext' in token_pred or token_count >= self.max_new_tokens:
            is_done = True
        if ' ' == token_pred[0] and token_count >= 2:
            back_track_kv = True
        return is_done, back_track_kv
