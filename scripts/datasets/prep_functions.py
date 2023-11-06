import os
from typing import List


def words_to_sntc(wordlist: List[str]) -> str:
    """
    :param wordlist: List[str] of words, may include punctuation
    """
    return ' '.join(wordlist).strip()


def format_wait_k_multi_sentences(source: str, target: str, k: int, 
                                  eos_token: str = '<|endoftext|>',
                                  empty_repr: str = '') -> List[List[str]]:
    """
    :param source: 1+ sentences in a source lange
    :param target: 1+ sentences that are a translation of the source
    :param k: how many source words to wait before predicting the next translated word 
    :param eos_token: EOS token added as the output when inputted with a complete translation
    :param empty_repr: Representation of what an "empty" value should look like

    What this function does: given a source sentence and target translation, formats multiple
        wait-k distributions of prompt content. 

    The output is a list of tuples in the form [current_source, current_target, target_token]    
    """
    
    wait_k_prompts = []
        
    src_words = source.split(' ')
    tgt_words = target.split(' ')
    src_wordcount = len(src_words)
    tgt_wordcount = len(tgt_words)
    assert(src_wordcount > 0)
    
    for i in range(tgt_wordcount):  # For every target word, pair it with the source words k words ahead

        current_source = words_to_sntc(src_words[0:min(src_wordcount + 1, i + (k - 1) + 1)])

        current_translation = words_to_sntc(tgt_words[0:i]) or empty_repr

        next_word = tgt_words[i]

        new_entry = [current_source, current_translation, next_word]
        wait_k_prompts.append(new_entry)

        # # We're at the last target word, but source still has words left after k
        # if i + 1 == tgt_wordcount and i + k < src_wordcount:  

        #     # If want to include an example that has the fullness of the source--
        #     full_source = source
        #     current_translation = words_to_sntc(tgt_words[:-1])
        #     last_word = tgt_words[-1]
        #     new_entry = (full_source, current_translation, last_word)
        #     wait_k_prompts.append(new_entry)

    # Finally, include an example with an EOS Token
    new_entry = [source, target, eos_token]
    wait_k_prompts.append(new_entry)

    return wait_k_prompts


def safe_open_w(path):
    '''
    Open "path" for writing, creating any parent directories that are needed

    From https://stackoverflow.com/questions/23793987/write-a-file-to-a-directory-that-doesnt-exist
    '''
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return open(path, 'w')
    