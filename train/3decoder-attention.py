import torch
import torch.nn as nn
import math
import numpy as np

torch.manual_seed(42)

vocab_size = 100
d_model = 8
decoder_seq_len = 5

#0 - <SOS>(first token)
target_ids = torch.tensor([[0, 25, 42, 18, 99]])

embedding_layer_dec = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
base_dec_embeddings = embedding_layer_dec(target_ids)

pos_encoding_dec = torch.zeros(1, decoder_seq_len, d_model)
position_dec = torch.arange(0, decoder_seq_len, dtype=torch.float).unsqueeze(1)
div_term_dec = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
pos_encoding_dec[0, :, 0::2] = torch.sin(position_dec * div_term_dec)
pos_encoding_dec[0, :, 1::2] = torch.cos(position_dec * div_term_dec)

final_dec_embeddings = base_dec_embeddings + pos_encoding_dec

W_q_dec = nn.Linear(d_model, d_model, bias=False)
W_k_dec = nn.Linear(d_model, d_model, bias=False)
W_v_dec = nn.Linear(d_model, d_model, bias=False)

Q_dec = W_q_dec(final_dec_embeddings)
K_dec = W_k_dec(final_dec_embeddings)
V_dec = W_v_dec(final_dec_embeddings)

d_k_dec = K_dec.size(-1)
raw_attention_scores_dec = torch.matmul(Q_dec, K_dec.transpose(-2, -1)) / math.sqrt(d_k_dec)

causal_mask = torch.tril(torch.ones(decoder_seq_len, decoder_seq_len))

masked_attention_scores = raw_attention_scores_dec.masked_fill(causal_mask == 0, float('-inf'))

attention_weights_dec = torch.softmax(masked_attention_scores, dim=-1)

np.set_printoptions(formatter={'float': '{: 0.2f}'.format}, linewidth=150)
torch.set_printoptions(precision=2, sci_mode=False, linewidth=150)

print("--- RAW DECODER ATTENTION SCORES (5x5) ---")
print(raw_attention_scores_dec[0].detach().numpy())

print("\n--- CAUSAL MASK (5x5) ---")
print(causal_mask.numpy())

print("\n--- MASKED ATTENTION SCORES (-inf applied) ---")
print(masked_attention_scores[0].detach().numpy())

print("\n--- FINAL DECODER ATTENTION WEIGHTS (Softmax) ---")
print(attention_weights_dec[0].detach().numpy())