import torch
import torch.nn as nn

# 1. Hiperparametry (Hyperparameters) z publikacji "Attention Is All You Need"
d_model = 512       # Grubość wektora słowa
num_heads = 8       # Liczba głów Atencji
seq_length = 5      # Długość udawanego zdania (np. 5 słów)
batch_size = 1      # Liczba zdań przetwarzanych naraz

# 2. Inicjalizacja warstwy Multi-Head Attention
# Używamy batch_first=True, aby wymiary tensora były ułożone logicznie: (Batch, Sequence, Dimension)
mha_layer = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)

# 3. Generowanie sztucznych danych wejściowych (Mock Input Data)
# Udajemy, że mamy zdanie, które przeszło już przez Embedding i Positional Encoding.
# Tworzymy tensor wypełniony losowymi ułamkami.
input_sequence = torch.rand(batch_size, seq_length, d_model)

print("--- INPUT ---")
print("Input sequence shape:")
print(input_sequence.shape)

# 4. Mechanizm Self-Attention w praktyce
# Zgodnie z teorią: Query, Key i Value pochodzą z tego samego źródła (naszego zdania).
attention_output, attention_weights = mha_layer(
    query=input_sequence,
    key=input_sequence,
    value=input_sequence
)

print("\n--- OUTPUT ---")
print("Attention output shape (Contextualized vectors):")
print(attention_output.shape)

print("\nAttention weights shape (Correlation percentages):")
print(attention_weights.shape)