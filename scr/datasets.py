import torch
from torch.utils.data import Dataset
from scr.utils import random_masking

class Roberta_datasets(Dataset):
    def __iniit__(self, texts, tokenizer, max_len, PAD):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.PAD = PAD

    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, index):
        x = torch.tensor(self.texts[index])
        pad = torch.ones(self.max_len, device=x.device, dtype=x.dtype)*self.PAD
        pad[:x.size(0)] = x
        x = pad
        pos = torch.arange(1, self.max_len+1)
        token_ids = torch.zeros_like(x)
        mask = (x == self.PAD)
        labels = x.detach().clone()
        x, mask_arr = random_masking(x)
        mask_labels = labels*mask_arr
        
        return x, pos, token_ids, mask, mask_labels