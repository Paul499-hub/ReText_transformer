import torch.nn as nn
import torch 
# Modules
from modules.configs.Memory_v0 import ModelConfig

class Head(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        # Scale (/sqrt(d_head_size))
        self.scale = self.config.d_head_size ** -0.5
        # Search
        self.Q = nn.Linear(config.C, config.d_head_size, bias=False)
        self.K = nn.Linear(config.C, config.d_head_size, bias=False)
        self.V = nn.Linear(config.C, config.d_head_size, bias=False)
        # Mask
        self.register_buffer('tril', torch.tril(torch.ones(config.T, config.T)))
        # Dropout
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        if self.config.asserts: assert x.dim() == 3
        B,T,C = x.shape
        if self.config.asserts: assert (self.config.B==B) 
        if self.config.asserts: assert (self.config.T>=T) 
        if self.config.asserts: assert (self.config.C==C)
        # QKt / sqrt(d) -> softmax
        Q = self.Q(x)
        K = self.K(x)
        V = self.V(x)
        expected = (B,T, self.config.d_head_size)
        if self.config.asserts: assert (Q.shape==expected) 
        if self.config.asserts: assert (K.shape==expected) 
        if self.config.asserts: assert (V.shape==expected)
        # Attention matrix
        out = Q @ K.transpose(-2,-1)
        if self.config.asserts: assert out.shape == (B,T,T)
        out = out * self.scale
        out = out.masked_fill(self.tril[:T,:T]==0, float('-inf'))
        out = torch.softmax(out, dim=-1)
        out = self.dropout(out)
        out = out @ V # B,T,T @ B,T,d_head_size --> B,T, d_head_size
        if self.config.asserts: assert out.shape == (B,T, self.config.d_head_size)
        return out
