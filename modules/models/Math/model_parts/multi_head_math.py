import torch
import torch.nn as nn
from modules.models.Math.model_parts.head import Head
from modules.test_config import config, ModelConfig


if False: # Parent script
    class Block(nn.Module):
        def __init__(self, config: ModelConfig):
            super().__init__()
            self.mhm = MultiHeadMath(config)
            self.ff = FeedForward(config)
            self.ln = nn.LayerNorm(config.d_channel)
            self.ln2 = nn.LayerNorm(config.d_channel)
        
        def forward(self, x):
            # RESIDUAL + LayerNorm
            x = x + self.mha( self.ln(x) )
            x = x + self.ff( self.ln2(x) )
            return x

class MultiHeadMath(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        #self.heads = nn.ModuleList(  [Head(config) for _ in range(config.num_heads)] )
        self.heads = nn.ModuleList([
            Head(config, math_operation='identity'), # <--- takes math operation as a param
            Head(config, math_operation='square'),
            Head(config, math_operation='tanh'),
            Head(config, math_operation='add'),
            Head(config, math_operation='mult'),
            Head(config, math_operation='mean'),
            Head(config, math_operation='max'),
        ])
        self.proj = nn.Linear(config.d_channel * len(self.heads), config.d_channel)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x):
        out = torch.cat([ h(x) for h in self.heads ], dim=-1 ) # shape: [B,T, num_heads * C ]
        out = self.dropout(self.proj(out))                     # shape: [B,T, C] 
        return out

def example_usage():
    mha = MultiHeadMath(config)
    out = mha.forward( torch.randn( config.d_batch,config.d_context, config.d_channel ) )
    print( f'mhm shape: {out.shape}' )
    # Expecting B,T,C 
    # mhm shape: torch.Size([1, 32, 64]) ---> [B,T,C] 
# example_usage()