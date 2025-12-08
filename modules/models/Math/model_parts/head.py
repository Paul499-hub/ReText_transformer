import torch 
import torch.nn as nn
import torch.nn.functional as F

from modules.test_config import config, ModelConfig

class Head(nn.Module):
    def __init__(self, config: ModelConfig, math_operation:str):
        super().__init__()
        self.config = config
        self.m_op = math_operation     
        # Math Layer instead --- No masking cause no auto - teacher forcing.
        if self.m_op != 'identity':
            self.l_math = nn.Linear(config.d_channel, config.d_channel, bias=False)
            # Define dropout 
            self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x):
        B, T, C = x.shape  

        # identity is trivial
        if self.m_op == "identity":
            return x
        
        # for all other ops compute projected vector
        m = self.l_math(x)   # [B, T, C]

        # Apply math operations 
        if self.m_op == "square":
            out = m**2
        if self.m_op == "tanh":
            out = torch.tanh(m)
        if self.m_op == "add":
            out = x + m 
        if self.m_op == "mult":
            out = x * m
        if self.m_op == "mean":
            out = torch.mean(torch.stack([x, m], dim=-1), dim=-1)
        if self.m_op == "max":
            #return torch.max(x, m).values
            out = torch.maximum(x, m)

        if False: # Test print
            print('\n---')
            print(f"Operation: {self.m_op}")
            print(f"x sample (first element): {x[0,0,:5]}")
            print(f"m sample (first element): {m[0,0,:5]}")
            print(f"Output sample (first element): {out[0,0,:5]}")  # print first 5 values of first token
        return out


def example_usage():
    m_head = Head(config, math_operation='square')
    out = m_head.forward( torch.randn(size=( config.d_batch, config.d_context, config.d_channel)) )
    print(f'head shape: {out.shape}') # Expected [B, T , C]
    # torch.Size([1, 32, 2])  -- Got: [B, T, C]
#example_usage()


def test_math_ops():
    ops = ["identity", "square", "tanh", "add", "mult", "mean", "max"]
    x = torch.randn(config.d_batch, config.d_context, config.d_channel)

    for op in ops:
        head = Head(config, math_operation=op)
        out = head.forward(x)
#test_math_ops()
