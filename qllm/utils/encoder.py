import torch 
# hf didn't support batch encode plus
# pad_token_id = 1 by default
def batch_encode_plus(tokenizer, prompts, return_tensors=None, max_length=None):
    # hf didn't provide the left padding for batched tokens
    # do manual padding
    interested_keys = ['input_ids', 'attention_mask', 'token_type_ids']
    sample_out = []
    for prompt in prompts:
        token_i = tokenizer(prompt, return_tensors="pt")
        for k in token_i.keys():
            if k in interested_keys:
                if k == 'input_ids':
                    # padd left, add 1
                    token_i[k] = torch.cat([torch.ones(max_length - token_i[k].shape[1], dtype=torch.long), token_i[k][0]])
                elif k == 'attention_mask':
                    # padd left, add 0
                    token_i[k] = torch.cat([torch.zeros(max_length - token_i[k].shape[1], dtype=torch.long), token_i[k][0]])
                # if the token_i dim is 1, then use view to make it 1,dim
                if len(token_i[k].shape) == 1:
                    token_i[k] = token_i[k].view(1, -1)
        sample_out.append(token_i)
    # merge along the batch dimension
    batched_out = {}
    for sample in sample_out:
        for key in sample:
            if key not in batched_out:
                batched_out[key] = []
            batched_out[key].append(sample[key])
    for key in batched_out:
        batched_out[key] = torch.cat(batched_out[key], dim=0)
    return batched_out