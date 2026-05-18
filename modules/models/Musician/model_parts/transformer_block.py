import torch.nn as nn
import torch
# Modules
from modules.configs.Musician import ModelConfig
from modules.models.Musician.model_parts.feed_forward import FeedForward
from modules.models.Musician.model_parts.multi_head_attention import MultiHeadedAttention

class Block(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.mha = MultiHeadedAttention(config)
        self.ff = FeedForward(config)
        self.ln_mha = nn.LayerNorm(config.C)
        self.ln_ff = nn.LayerNorm(config.C)

    def forward(self, x):
        B,S,C = x.shape
        if self.config.asserts: assert (self.config.B==B) 
        if self.config.asserts: assert (self.config.T * self.config.K >= S) 
        if self.config.asserts: assert (self.config.C==C)
        x = x + self.mha( self.ln_mha(x) )
        x = x + self.ff( self.ln_ff(x) )
        if self.config.asserts: assert x.shape == (B,S,C)
        return x