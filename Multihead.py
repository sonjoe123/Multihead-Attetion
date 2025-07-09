import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import allclose
import math


        
def ScaledDotProduct(self, query, key, value):
    d_k = query.size(-1)
    scores = query @ key.transpose(-2,-1) / d_k**0.5
    attn_weights = torch.softmax(scores, dim = -1)
    output = attn_weights @ value 

    return output, attn_weights

class AttentionBlock(nn.Module):
    def __init__(self, input_dim:int, output_dim:int, bias = False):
        super().__init__()
        self.W_query = nn.Linear(input_dim, output_dim, bias = bias)
        self.W_key = nn.Linear(input_dim, output_dim, bias = bias)
        self.W_value = nn.Linear(input_dim, output_dim, bias = bias)

    def forward (self, query, key, value):
        Q = self.W_query(query)
        K = self.W_key(key)
        V = self.W_value(value)

        output, attn_weights = ScaledDotProduct(Q, K, V)

        return output, attn_weights
