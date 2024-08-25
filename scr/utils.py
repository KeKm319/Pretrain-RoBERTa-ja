import torch
import random

def random_masking(input_ids, vocab_size, MASK):
    # ランダム配列の作成
    rand = torch.rand(input_ids.shape)
    # 各トークンにて15%の確率で変換
    mask_arr = rand < 0.15
    # Excluding special token
    special_token_mask = (input_ids > 8)
    mask_arr = mask_arr * special_token_mask
    selection=torch.flatten(mask_arr.nonzero()).tolist()
    
    """
    random mask
    The [MASK] token 80% of the time
    A random token 10% of the time
    The original token 0% of the time
    """
    
    for mask in selection:
        r = random.random()
        if r < 0.8:
            input_ids[mask] = MASK
        elif r < 0.9:
            input_ids[mask] = random.randint(9, vocab_size-1)
        else:
            pass
    
    return input_ids, mask_arr