import torch
import torch.nn as nn
from modules.models.Transformer.model_parts.attention_head import Head
from modules.model_config import ModelConfig

# config = ModelConfig(
#     # Dimension params
#     d_batch=1,
#     d_context=32,
#     d_channel=64,
#     d_vocab=50257,
#     d_head_size=2,
#     # Other
#     num_t_blocks=2,
#     num_heads=2,
#     dropout=0.1
# )

class MultiHeadAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.heads = nn.ModuleList(  [Head(config) for _ in range(config.num_heads)] )
        self.proj = nn.Linear(config.d_head_size * config.num_heads, config.d_channel)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x):
        out = torch.cat([ h(x) for h in self.heads ], dim=-1 ) # shape: [B,T, num_heads*D(d_head_size)]
        out = self.dropout(self.proj(out))                     # shape: [B,T, C(d_channel)] 
        return out

def example_usage():
    from test_config import config
    mha = MultiHeadAttention(config)
    out = mha.forward( torch.randn( config.d_batch,config.d_context, config.d_channel ) )
    print( f'mha shape: {out.shape}' )
    # Expecting B,T,C 
    #torch.Size([1, 32, 64]) ---> [B,T,C] 
#example_usage()