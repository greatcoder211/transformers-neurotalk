import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):
    def __init__(self, vocab_size, d_model):
        super(InputEmbeddings, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len=500):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return x


torch.manual_seed(42)
torch.set_printoptions(precision=2, sci_mode=False, linewidth=120)

vocab_size = 100
d_model = 8
seq_len = 16

input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 4, 14, 15]])

embedder = InputEmbeddings(vocab_size, d_model)
pe_layer = PositionalEncoding(d_model)

base_embeddings = embedder(input_ids)

print("--- 1. BASE EMBEDDINGS FOR WORD 'koła' (ID: 4) ---")
print(f"Position 3  (koła):\n {base_embeddings[0, 3, :].detach().numpy()}")
print(f"Position 13 (koła):\n {base_embeddings[0, 13, :].detach().numpy()}\n")

print("--- 2. POSITIONAL ENCODING VECTORS TO ADD ---")
print(f"Positional modifier for index 3:\n  {pe_layer.pe[0, 3, :].detach().numpy()}")
print(f"Positional modifier for index 13:\n {pe_layer.pe[0, 13, :].detach().numpy()}\n")

final_embeddings = pe_layer(base_embeddings)

print("--- 3. FINAL VECTORS AFTER ADDITION (Base + PE) ---")
print(f"Final Position 3  (koła):\n {final_embeddings[0, 3, :].detach().numpy()}")
print(f"Final Position 13 (koła):\n {final_embeddings[0, 13, :].detach().numpy()}")