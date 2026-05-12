import torch.nn as nn
import torch
# Modules
from modules.configs.Memory_v0 import ModelConfig
from modules.models.Memory_v0.model_parts.feed_forward import FeedForward
from modules.models.Memory_v0.model_parts.multi_head_attention import MultiHeadedAttention

class Block(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.mha = MultiHeadedAttention(config)
        self.ff = FeedForward(config)
        self.ln_mha = nn.LayerNorm(config.C)
        self.ln_ff = nn.LayerNorm(config.C)

    def forward(self, x):
        B,T,C = x.shape
        if self.config.asserts: assert (self.config.B==B) 
        if self.config.asserts: assert (self.config.T>=T) 
        if self.config.asserts: assert (self.config.C==C)
        x = x + self.mha( self.ln_mha(x) )
        x = x + self.ff( self.ln_ff(x) )
        if self.config.asserts: assert x.shape == (B,T,C)
        return x