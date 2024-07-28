"""
    RALCP and other beam rescoring options should be placed here to avoid
    bloating agent code. These beam rescoring options are almost always
    improvements when engaging in chunk-wise Speculative Beam Search.
"""

# a bit inefficient, technically slightly erroneous for agreement thresholds of
# less than 0.5 so we assume that it should be above that
def ralcp_sort(model_output, ralcp_thresh):
    ralcp_candidates = model_output
    ref_len = len(model_output)
    voting_dict = {}
    idx = 0
    min_len = len(model_output[0])
    for i in range(1, len(model_output)):
        min_len = min(min_len, len(model_output[i]))

    while idx < min_len:
        
        # find most commonly agreed upon candidates, heuristic for longest common prefix
        # can technicall miss the longest common prefix if the agreement threshold is below 0.5
        for i in range(len(ralcp_candidates)):
            if ralcp_candidates[i][idx] not in voting_dict.keys():
                voting_dict[ralcp_candidates[i][idx]] = [i]
            else:
                voting_dict[ralcp_candidates[i][idx]].append(i)
        
        # find the top line of agreement, part of heuristic. we assume thresholds of 0.5 or
        # higher, to ensure that only one agreement line is possible
        top_opt = 0
        top_votes = len(voting_dict[ralcp_candidates[0][idx]])
        for i in range(1, len(ralcp_candidates)):
            if top_votes < len(voting_dict[ralcp_candidates[0][idx]]):
                top_opt = i
                top_votes = len(voting_dict[ralcp_candidates[0][idx]])

        # check to make sure agreement threshold is met
        if float(top_votes / ref_len) > ralcp_thresh:
            temp_list = []
            for indx in voting_dict[ralcp_candidates[top_opt][idx]]:
                temp_list.append(model_output[indx])
            model_output = temp_list

            if len(model_output) == 1:
                return model_output

        else:
            model_output = model_output[0]
            #print(model_output)
            return model_output
        
        idx += 1
        voting_dict = {}
        ralcp_candidates = model_output

    return model_output[0]
