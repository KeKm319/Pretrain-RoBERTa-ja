import torch
import torch.nn as nn
from embeddings import Embeddings
from encoder import Encoder

class MLM(nn.Module):
    def __init__(self, vocab_size, hidden_dim):
        super(MLM, self).__init__()
        self.out_linear = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        """
            args:
                x: torch.tensor size=(batch_size, seq_length, hidden_dim)
            return:
                torch.tensor size=(batch_size, seq_len, hidden_dim)
        """
        return self.out_linear(x)

class Model(nn.Module):
    def __init__(self, vocab_size, max_len, num_layers, num_attn_heads, hidden_dim, dropout=0.1):
        super(Model, self).__init__()
        self.embedding = Embeddings(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            max_len=max_len,
            dropout=dropout
        )
        
        self.roberta = Encoder(num_layers=num_layers, num_heads=num_attn_heads, hidden_dim=hidden_dim, dropout=dropout)
        self.mlm = MLM(vocab_size, hidden_dim)
    def forward(self, input_ids, pos_ids, token_type_ids, mask=None, return_weight=False):
        """
        args:
            input_ids: torch.tensor size=(batch_size, seq_length) 入力系列
            pos_ids: torch.tensor size=(batch_size, seq_length)
            token_type_ids: torch.tensor size=(batch_size, seq_length) value=(0)
        return:
            torch.tensor size=(batch_size, seq_length, vocab_size)
        """
        x = self.embedding(input_ids, pos_ids, token_type_ids)
        out, weights = self.roberta(x)
        out = self.mlm(out)

        if not return_weight:
            return out
        else:
            return out, weights