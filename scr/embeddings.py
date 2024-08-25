import torch
import torch.nn as nn
import numpy as np

def position_encoding_init(n_position, d_pos_vec):
    """
    Positional Encodingのための行列の初期化を行う
    args:
        param n_position: int, 系列長
        param d_pos_vec: int, 隠れ層の次元数

    return: 
        torch.tensor, size=(n_position, d_pos_vec)
    """
    # PADがある単語の位置はpos=0にしておき、position_encも0にする
    position_enc = np.array([
        [pos / np.power(10000, 2 * (j // 2) / d_pos_vec) for j in range(d_pos_vec)]
        if pos != 0 else np.zeros(d_pos_vec) for pos in range(n_position)])

    position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2]) # dim 2i
    position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2]) # dim 2i+1
    return torch.tensor(position_enc, dtype=torch.float)

class Embeddings(nn.Module):
    def __init__(self, vocab_size, emb_dim, max_len, dropout=0.1):
        super(Embeddings, self).__init__()
        self.word_embs = nn.Embedding(vocab_size, emb_dim, padding_idx=3)
        self.pos_enc =nn.Embedding(max_len, emb_dim, padding_idx=0)
        self.seg_emb = nn.Embedding(2, emb_dim)
        self.layer_norm = nn.LayerNorm(emb_dim)
        self.dropout = nn.Dropout(dropout)
        # position encodingの重みを三角関数で初期化
        self.pos_enc.weight.data = position_encoding_init(max_len, emb_dim)
    def forward(self, input_ids, pos_ids, token_type_ids):
        """
        args:
            input_ids: torch.tensor size=(batch_size, seq_length) 入力系列
            pos_ids: torch.tensor size=(batch_size, seq_length)
            token_type_ids: torch.tensor size=(batch_size, seq_length) value=(0)
        return:
            torch.tensor size=(batch_size, seq_length, emb_dim)
        """
        word_embs = self.word_embs(input_ids)
        pos_enc = self.pos_enc(pos_ids)
        seg_emb = self.seg_emb(token_type_ids)

        return self.dropout(self.layer_norm(word_embs + pos_enc + seg_emb))