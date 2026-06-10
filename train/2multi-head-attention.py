import torch
import torch.nn as nn
import math
import numpy as np

torch.manual_seed(42)
torch.set_printoptions(precision=2, sci_mode=False, linewidth=150)

vocab_size = 100
d_model = 8
seq_len = 16

input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 4, 14, 15]])
embedding_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
base_embeddings = embedding_layer(input_ids)

pos_encoding = torch.zeros(1, seq_len, d_model)
position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
pos_encoding[0, :, 0::2] = torch.sin(position * div_term)
pos_encoding[0, :, 1::2] = torch.cos(position * div_term)
final_embeddings = base_embeddings + pos_encoding

W_q = nn.Linear(d_model, d_model, bias=False)
W_k = nn.Linear(d_model, d_model, bias=False)
W_v = nn.Linear(d_model, d_model, bias=False)

Q = W_q(final_embeddings)
K = W_k(final_embeddings)
V = W_v(final_embeddings)

d_k = K.size(-1)
attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

#hardcode, normally backpropagation would do
# koła -> naukowego
attention_scores[0, 3, 4] += 50.0
# koła -> samochodzie
attention_scores[0, 13, 15] += 50.0
# złapał -> gumę
attention_scores[0, 8, 9] += 30.0
# gumę -> złapał
attention_scores[0, 9, 8] += 30.0

attention_weights = torch.softmax(attention_scores, dim=-1)
attention_output = torch.matmul(attention_weights, V)

np.set_printoptions(formatter={'float': '{: 0.2f}'.format}, linewidth=150)

print("--- FULL ATTENTION MATRIX (16x16) ---")
print(attention_weights[0].detach().numpy())
