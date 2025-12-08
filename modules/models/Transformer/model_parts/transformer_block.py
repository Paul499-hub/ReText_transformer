import torch
import torch.nn as nn

from modules.models.Transformer.model_parts.multi_head_attention import MultiHeadAttention
from modules.models.Transformer.model_parts.feed_forward import FeedForward
from modules.model_config import ModelConfig

    # cfg = ModelConfig(
    #     # Dimension params
    #     d_batch=1,
    #     d_context=32,
    #     d_channel=64,
    #     d_vocab=enc.n_vocab,
    #     d_head_size=2,
    #     # Other
    #     num_t_blocks=2,
    #     num_heads=2,
    #     dropout=0.1
    # )


class Block(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.mha = MultiHeadAttention(config)
        self.ff = FeedForward(config)
        self.ln = nn.LayerNorm(config.d_channel)
        self.ln2 = nn.LayerNorm(config.d_channel)
    
    def forward(self, x):
        x = x + self.mha( self.ln(x) )
        x = x + self.ff( self.ln2(x) )
        return x
    
def example_usage():
    from test_config import config
    t_block = Block(config)
    out = t_block.forward( torch.randn( config.d_batch, config.d_context, config.d_channel ))
    print(f't_block shape: {out.shape}') # Expected shape: [B,T,C]
    # Got B,T,C 
#example_usage()