import sentencepiece as spm

class Tokenizer():
    def __init__(self, path):
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(path)
    def vocab_size(self):
        return self.sp.GetPieceSize()
    def label_2_id(self, label):
        return self.sp.PieceToId(label)
    def id_2_label(self, id):
        return self.sp.IdToPiece(int(id))
    def text_2_token(self, sentence):
        return self.sp.EncodeAsPieces(sentence)
    def encode_texts(self, sentence):
        input_ids = self.sp.encode(sentence)
        return input_ids