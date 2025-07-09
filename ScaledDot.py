import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import allclose
import math

class ScaledDotProduct(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, query, key, value):
        d_k = query.size(-1)
        scores = query @ key.transpose(-2,-1) / d_k**0.5
        attn_weights = torch.softmax(scores, dim = -1)
        output = attn_weights @ value 

        return output, attn_weights
    


# Instantiate your module
my_attn = ScaledDotProduct()

# Create dummy input
torch.manual_seed(0)
X = torch.randn(2, 3, 6)  # (batch_size=2, seq_len=3, embed_dim=6)

# Run PyTorch built-in attention
# Note: need to set `is_causal=False` to match behavior
torch_output = F.scaled_dot_product_attention(X, X, X, is_causal=False)

# Run your implementation
my_output, my_weights = my_attn(X, X, X)

# Compare outputs
print("Are outputs close?", allclose(torch_output, my_output, atol=1e-6))
