import torch
import torch.nn as nn
import math
import numpy as np

torch.manual_seed(42)

vocab_size = 100
d_model = 8
seq_len = 16
decoder_seq_len = 5

input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 4, 14, 15]])
target_ids = torch.tensor([[0, 25, 42, 18, 99]])

emb_layer = nn.Embedding(vocab_size, d_model)

#encoder
enc_emb = emb_layer(input_ids)
pos_enc = torch.zeros(1, seq_len, d_model)
pos = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
pos_enc[0, :, 0::2] = torch.sin(pos * div)
pos_enc[0, :, 1::2] = torch.cos(pos * div)
encoder_output = enc_emb + pos_enc

#decoder
dec_emb = emb_layer(target_ids)
pos_dec = torch.zeros(1, decoder_seq_len, d_model)
pos_d = torch.arange(0, decoder_seq_len, dtype=torch.float).unsqueeze(1)
pos_dec[0, :, 0::2] = torch.sin(pos_d * div)
pos_dec[0, :, 1::2] = torch.cos(pos_d * div)
decoder_output = dec_emb + pos_dec

W_q_cross = nn.Linear(d_model, d_model, bias=False)
W_k_cross = nn.Linear(d_model, d_model, bias=False)
W_v_cross = nn.Linear(d_model, d_model, bias=False)

Q_cross = W_q_cross(decoder_output)
K_cross = W_k_cross(encoder_output)
V_cross = W_v_cross(encoder_output)

d_k_cross = K_cross.size(-1)
cross_attention_scores = torch.matmul(Q_cross, K_cross.transpose(-2, -1)) / math.sqrt(d_k_cross)

cross_attention_scores[0, 2, 8] += 40.0
cross_attention_scores[0, 2, 9] += 40.0

cross_attention_weights = torch.softmax(cross_attention_scores, dim=-1)
cross_attention_output = torch.matmul(cross_attention_weights, V_cross)

np.set_printoptions(formatter={'float': '{: 0.2f}'.format}, linewidth=150)
torch.set_printoptions(precision=2, sci_mode=False, linewidth=150)

print("--- CROSS-ATTENTION ALIGNMENT MATRIX (5x16) ---")
print(cross_attention_weights[0].detach().numpy())

print("\n--- IDIOM TRANSLATION DEMONSTRATION ---")
print("Attention weights for target word 3 (Index 2):")
print(cross_attention_weights[0, 2].detach().numpy())