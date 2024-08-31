import torch
import math

"""
Finds the index of a specific token in the input_ids tensor.
"""
def get_special_index(input_ids, token, device, tokenizer):
    token_ids = tokenizer([token], truncation=True, padding=False, max_length=30, return_overflowing_tokens=False, return_length=False) 
    token_end_idx_list = []
    for i in range(len(input_ids)):
        for idx in torch.where(input_ids[i] == token_ids['input_ids'][0][0])[0]:
            if (token_ids['input_ids'][0] == input_ids[i][idx : idx + len(token_ids['input_ids'][0])].tolist()):
                token_end_idx_list.append((idx + len(token_ids['input_ids'][0]) - 1).item())
        if len(token_end_idx_list) < i+1:
            token_end_idx_list.append(input_ids.size(1)-1)

    return torch.tensor(token_end_idx_list, device=device)

""""
Finds the lengths in tokens of the source and target sequences.  Also finds the number of tokens in the first k words of the source sequence.
"""
def find_lengths(input_ids, device, p1_len, p2_len, tokenizer, waitk):
    assistant_index = get_special_index(input_ids, "\nAssistant:", device, tokenizer)
    
    s_lens = assistant_index - p1_len - p2_len + 1

    eos_index = get_special_index(input_ids, "<|endoftext|>", device, tokenizer)
    
    t_lens = eos_index - assistant_index
    t_lens[t_lens < 0] = 0
    
    len_dict = {'s_len': s_lens.tolist(), 't_len': t_lens.tolist()}

    t_word_batch = []
    s_word_batch = []
    waitk_lens = []
    for ele, a_i, s_len, t_len in zip(input_ids, assistant_index, s_lens, t_lens):
        t_word_lens = []
        s_word_lens = []
        target = tokenizer.decode(ele[a_i:a_i+t_len]).split()
        source = tokenizer.decode(ele[p1_len:p1_len+s_len]).split()
        for word in target:
            t_word_lens.append(len(tokenizer(' ' + word).input_ids))
        for word in source:
            s_word_lens.append(len(tokenizer(' ' + word).input_ids))
        t_word_batch.append(t_word_lens)
        s_word_batch.append(s_word_lens)
        waitk_lens.append(sum(s_word_lens[:waitk]))
    len_dict['t_word_lens'] = t_word_batch
    len_dict['s_word_lens'] = s_word_batch
    len_dict['waitk'] = waitk_lens

    len_dict_list = [dict(zip(len_dict.keys(), values)) for values in zip(*len_dict.values())]
    return len_dict_list

"""
Creates a simulmask for each batch in input_ids.
"""
def make_mask_batch(input_ids, att_len, device, p1_len, p2_len, tokenizer, waitk):
        len_dict_list = find_lengths(input_ids, device, p1_len, p2_len, tokenizer, waitk)

        att_mask_list = []
        for len_dict in len_dict_list:
            att_mask = make_simul_mask(len_dict, att_len, device)
            att_mask_list.append(att_mask)
        
        batch_att_mask = torch.stack(att_mask_list)[:, None, :, :]

        return batch_att_mask

"""
A helper function for constructing the simulmask.
"""
def create_uni_tri(len_dict, device):
    s_len, t_len, s_word_lens, t_word_lens, waitk = len_dict['s_len'], len_dict['t_len'], len_dict['s_word_lens'], len_dict['t_word_lens'], len_dict['waitk']
    tri_mask = torch.zeros((t_len, s_len), dtype=torch.bool, device=device)
    vert_pos = 0
    horz_pos = waitk
    for s_word_len, t_word_len in zip(s_word_lens[waitk:], t_word_lens):
        if vert_pos + t_word_len > t_len or horz_pos+s_word_len > s_len:
            break
        tri_mask[vert_pos:vert_pos+t_word_len, horz_pos:] = torch.ones(t_word_len, s_len-horz_pos, dtype=torch.bool, device=device)
        vert_pos += t_word_len
        horz_pos += s_word_len
    return tri_mask

"""
Creates a simulmask for a single batch.
"""
def make_simul_mask(len_dict, att_len, device, p1_len=0, p2_len=0):
    s_len, t_len, waitk = len_dict['s_len'], len_dict['t_len'], len_dict['waitk']
    mask1_pos = p1_len
    mask2_pos = mask1_pos + s_len + p2_len

    att_mask = torch.triu(torch.ones((att_len, att_len), dtype=torch.bool, device=device), diagonal=1)
    
    if waitk < s_len and t_len != 0:
        #Creates the attention mask for the target queries and source keys
        tri_mask = create_uni_tri(len_dict, device)
        att_mask[mask2_pos-1:mask2_pos+t_len-1, mask1_pos:mask1_pos+s_len] = tri_mask

        #Prevent prompt 2 from attending to source
        if p2_len > 1:
            att_mask[mask1_pos+s_len:mask2_pos-1, mask1_pos+waitk:mask1_pos+s_len] = torch.ones((p2_len-1, s_len-waitk), dtype=torch.bool, device=device)
    
    return att_mask

"""
Constructs the modified alibi tensor for the simulmask.
"""
def build_simul_alibi_tensor(attention_mask, num_heads, dtype):
    closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
    base = torch.tensor(
        2 ** (-(2 ** -(math.log2(closest_power_of_2) - 3))), device=attention_mask.device, dtype=torch.float32
    )
    powers = torch.arange(1, 1 + closest_power_of_2, device=attention_mask.device, dtype=torch.int32)
    slopes = torch.pow(base, powers)

    if closest_power_of_2 != num_heads:
        extra_base = torch.tensor(
            2 ** (-(2 ** -(math.log2(2 * closest_power_of_2) - 3))), device=attention_mask.device, dtype=torch.float32
        )
        num_remaining_heads = min(closest_power_of_2, num_heads - closest_power_of_2)
        extra_powers = torch.arange(1, 1 + 2 * num_remaining_heads, 2, device=attention_mask.device, dtype=torch.int32)
        slopes = torch.cat([slopes, torch.pow(extra_base, extra_powers)], dim=0)

    arange_tensor = ((~attention_mask).cumsum(dim=-1) - 1) * (~attention_mask)
    alibi = slopes[None, :, None, None].bfloat16() * arange_tensor
    return alibi.to(dtype)