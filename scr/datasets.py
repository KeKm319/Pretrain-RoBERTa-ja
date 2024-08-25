import torch
from torch.utils.data import Dataset
from scr.utils import random_masking

class Roberta_datasets(Dataset):
    def __init__(self, input_ids, tokenizer, max_len, vocab_size, PAD, MASK):
        self.input_ids = input_ids
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.PAD = PAD
        self.MASK = MASK

    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, index):
        x = torch.tensor(self.input_ids[index])
        pad = torch.ones(self.max_len, device=x.device, dtype=x.dtype)*self.PAD
        pad[:x.size(0)] = x
        x = pad
        pos = torch.arange(1, self.max_len+1)
        token_ids = torch.zeros_like(x)
        mask = (x == self.PAD)
        labels = x.detach().clone()
        x, mask_arr = random_masking(x, self.vocab_size, self.MASK)
        mask_labels = labels*mask_arr
        
        return x, pos, token_ids, mask, mask_labels