import torch.nn.functional as F
import torch.nn as nn
import torch 
# Modules
from modules.configs.Musician import ModelConfig

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
        self.register_buffer('tril', torch.tril(torch.ones(config.T*config.K, config.T*config.K)))
        # Dropout
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        if self.config.asserts: assert x.dim() == 3
        B,S,C = x.shape
        if self.config.asserts: assert (self.config.B==B) 
        if self.config.asserts: assert (self.config.T*self.config.K >= S) 
        if self.config.asserts: assert (self.config.C==C)
        # QKt / sqrt(d) -> softmax
        Q = self.Q(x)
        K = self.K(x)
        V = self.V(x)
        expected = (B, S, self.config.d_head_size)
        if self.config.asserts: assert (Q.shape==expected) 
        if self.config.asserts: assert (K.shape==expected) 
        if self.config.asserts: assert (V.shape==expected)
        # Attention matrix
        out = Q @ K.transpose(-2,-1)
        if self.config.asserts: assert out.shape == (B,S,S)        
        out = out * self.scale
        out = out.masked_fill(self.tril[:S,:S]==0, float('-inf'))
        out = torch.softmax(out, dim=-1)
        out = self.dropout(out)
        out = out @ V # B,S,S @ B,S,d_head_size --> B,S,d_head_size
        # XSA - remove the value 'self' vector from final attn output by using it's orthogonoal
        if self.config.XSA:
            Vn = F.normalize(V, dim=-1) # Value normalized, scaled its length to 1
            out = out - (out * Vn).sum(dim=-1, keepdim=True) * Vn
        if self.config.asserts: assert out.shape == (B,S, self.config.d_head_size)
        return out
