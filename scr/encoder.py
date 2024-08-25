import torch
import torch.nn as nn

class Attention(nn.Module):
    # Multi-Head-Attention + Residual-Connection & Layer-Normalization
    def __init__(self, num_heads=12, hidden_dim=768, dropout=0.1):
        super(Attention, self).__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.MHA = nn.MultiheadAttention(
            embed_dim = hidden_dim,
            num_heads = num_heads,
            dropout = dropout,
            batch_first = True
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, x, mask=None):
        """
            args:
                x: torch.tensor size=(batch_size, seq_length, emb_dim)
                mask: torch.tensor size=(batch_size, seq_length)
                    if token == [PAD] then mask = True
            return:
                torch.tensor size=(batch_size, seq_length, hidden_dim)
                weight: Attention_weight
        """
        residual = x
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        if mask is None:
            mask = torch.zeros_like(x, dtype=torch.bool)
        
        # (scale dot product attention) + (layer noermalization) + (projection)
        out, weight = self.MHA(query=q, key=k, value=v, key_padding_mask=mask)
        return self.layer_norm(out+residual), weight
        
class Intermediate(nn.Module):
    # Feed-Forward-Network + Residual-Connection & Layer-Normalization
    def __init__(self, hidden_dim=768, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(hidden_dim, hidden_dim*4)
        self.act_fn = nn.GELU()
        self.linear2 = nn.Linear(hidden_dim*4, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
            args:
                x: torch.tensor size=(batch_size, seq_length, hidden_dim)
            return:
                torch.tensor size=(batch_size, seq_length, hidden_dim)
        """
        residual = x
        out = self.act_fn(self.linear1(x))
        out = self.linear2(out)
        return self.dropout(self.layer_norm(out+residual))

class Layer(nn.Module):
    def __init__(self, num_heads=12, hidden_dim=768, dropout=0.1):
        super().__init__()
        self.attention = Attention(num_heads, hidden_dim, dropout)
        self.intermediate = Intermediate(hidden_dim, dropout)
    
    def forward(self, x, mask=None):
        """
            args:
                x: torch.tensor size=(batch_size, seq_length)
            return:
                torch.tensor size=(batch_size, seq_length, hidden_size)
        """
        out, weight = self.attention(x, mask)
        return self.intermediate(out), weight
    
class Encoder(nn.Module):
    def __init__(self, num_layers=12, num_heads=12, hidden_dim=768, dropout=0.1):
        super(Encoder, self).__init__()
        self.layer = nn.ModuleList([
            Layer(num_heads, hidden_dim, dropout)
            for _ in range(num_layers)
        ])
    
    def forward(self, x, mask=None):
        """
            args:
                x: torch.tensor size=(batch_size, seq_length)
                mask: torch.tensor size=(batch_size, seq_length)
            return:
                torch.tensor size=(batch_size, seq_length, hidden_size)
        """
        weights = []
        for layer in self.layer:
            x, weight = layer(x, mask)
            weights += [weight]

        return x, weights