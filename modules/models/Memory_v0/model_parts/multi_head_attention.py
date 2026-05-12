import torch.nn as nn
import torch 
# Modules
from modules.models.Memory_v0.model_parts.attention_head import Head
from modules.configs.Memory_v0 import ModelConfig

class MultiHeadedAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.heads = nn.ModuleList( [Head(config) for _ in range(config.num_heads)] )
        self.proj = nn.Linear( config.d_head_size * config.num_heads, config.C ) 
        self.dropout = nn.Dropout(config.dropout)
    def forward(self, x):
        B,T,C = x.shape
        if self.config.asserts: assert (self.config.B==B) 
        if self.config.asserts: assert (self.config.T>=T) 
        if self.config.asserts: assert (self.config.C==C)
        out = torch.cat( [h(x) for h in self.heads], dim=-1 )
        if self.config.asserts: assert out.shape == (B,T, self.config.d_head_size * self.config.num_heads )
        out = self.dropout(self.proj(out))
        if self.config.asserts: assert out.shape == (B,T,C)
        return out