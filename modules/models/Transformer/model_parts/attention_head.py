import torch 
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from modules.model_config import ModelConfig

class Head(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        # Define layers
        self.l_query = nn.Linear( config.d_channel, config.d_head_size, bias=False )
        self.l_key = nn.Linear( config.d_channel, config.d_head_size, bias=False )
        self.l_value = nn.Linear( config.d_channel, config.d_head_size, bias=False )
        # Define mask
        self.register_buffer('tril', torch.tril(torch.ones(config.d_context, config.d_context)))
        # Define dropout 
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x):
        B, T, C = x.shape  
        # Pass though laters
        k = self.l_key(x)                                   # shape: [B, T, D]
        q = self.l_query(x)                                 # shape: [B, T, D]
        v = self.l_value(x)                                 # shape: [B, T, D]

        # Attention matrix -> softmax -> scaling by **-0.5
        a_m = ( q @ k.transpose(-2,-1) ) * self.config.d_head_size ** -0.5   # shape: [B, T, T]
        # Mask
        a_m = a_m.masked_fill(self.tril[:T,:T] == 0, float('-inf'))
        #print(a_m)
        # tensor([[[-0.0264,    -inf,    -inf,    -inf,    -inf,    -inf],
        #  [-0.1640,  0.0070,    -inf,    -inf,    -inf,    -inf],
        #  [ 0.0386,  0.2616,  0.1238,    -inf,    -inf,    -inf],
        #  [-0.0195, -0.2204, -0.2095,  0.3883,    -inf,    -inf],
        #  [-0.0367, -0.0720, -0.0968,  0.0502, -0.0395,    -inf],
        #  [-0.1780, -0.0788, -0.4246,  0.2954, -0.2197, -0.3527]]],
        # Softmax
        a_m = F.softmax(a_m, dim=-1)
        #print(a_m)
        #     tensor([[[1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        #      [0.5255, 0.4745, 0.0000, 0.0000, 0.0000, 0.0000],
        #      [0.5138, 0.2598, 0.2264, 0.0000, 0.0000, 0.0000],
        #      [0.3268, 0.2050, 0.2658, 0.2023, 0.0000, 0.0000],
        #      [0.1838, 0.1701, 0.2206, 0.1898, 0.2357, 0.0000],
        #      [0.1474, 0.1320, 0.3232, 0.1314, 0.1615, 0.1046]]],
        #    grad_fn=<SoftmaxBackward0>)
        # Dropout 
        a_m = self.dropout(a_m)
        # Dot with v 
        out = a_m @ v                                       # shape: [B, T, D]
        return out # shape: [B,T,D]


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

def example_usage():
    from test_config import config
    a_head = Head(config)
    out = a_head.forward( torch.randn(size=( config.d_batch, config.d_context, config.d_channel)) )
    print(f'attention head shape: {out.shape}') # Expected [B, T , D]
    # torch.Size([1, 32, 2])  -- Got: [B, T, D]
#example_usage()

# Quick attention head example 
def quick_attention_head_example():
    B,T,C = 4,8,32
    x = torch.randn(B,T,C)
    head_size=16
    key_lin = nn.Linear(C, head_size, bias=False)
    query_lin = nn.Linear(C, head_size, bias=False)
    value_lin = nn.Linear(C, head_size, bias=False)
    # Pass trough layers
    k = key_lin(x) 
    q = query_lin(x)
    v = value_lin(x)

    out = q @ k.transpose(-2,-1) * head_size**-0.5
    out = F.softmax(out, dim=-1)
    out = out @ v

    print(out.shape)
    # torch.Size([4, 8, 16])