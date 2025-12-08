import torch
import torch.nn as nn

from modules.models.Math.model_parts.multi_head_math import MultiHeadMath
from modules.models.Math.model_parts.feed_forward import FeedForward
from modules.test_config import config, ModelConfig

class Block(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.mhm = MultiHeadMath(config)
        self.ff = FeedForward(config)
        self.ln = nn.LayerNorm(config.d_channel)
        self.ln2 = nn.LayerNorm(config.d_channel)
    
    def forward(self, x):
        # RESIDUAL + LayerNorm
        x = x + self.mhm( self.ln(x) )
        x = x + self.ff( self.ln2(x) )
        return x
    
def example_usage():
    block = Block(config)
    out = block.forward( torch.randn( config.d_batch, config.d_context, config.d_channel ))
    print(f't_block shape: {out.shape}') # Expected shape: [B,T,C]
    # Got B,T,C 
#example_usage()