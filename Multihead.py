import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import allclose
import math


        
def ScaledDotProduct(query, key, value):
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
    

class MyMultiHeadAttention (nn.Module): 
    def __init__(self, embed_dim: int, num_heads:int, projection_bias = False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        head_dim = embed_dim // num_heads

        self.head_blocks = nn.ModuleList([
            AttentionBlock(input_dim = embed_dim, output_dim = head_dim, bias = projection_bias)
            for _ in range(self.num_heads)
        ])     

        self.projection = nn.Linear(embed_dim, embed_dim, bias = projection_bias)


    def forward(self, query, key, value):
        attn_list = []
        attn_weights_list = []

        for head in self.head_blocks:
            attn, attn_weights = head(query, key, value)
            attn_list.append(attn)
            attn_weights_list.append(attn_weights)


        attn = torch.concat(attn_list, dim= 2) 
        attn_weights = torch.stack(attn_weights_list).mean(dim = 0)

        return self.projection(attn), attn_weights
