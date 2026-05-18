import torch 
import torch.nn as nn
# Modules
from modules.configs.Musician import ModelConfig 

class FeedForward(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config=config
        self.ff = nn.Sequential(
            nn.Linear(config.C, 4*config.C),
            nn.GELU(),
            nn.Linear(4*config.C, config.C),
            nn.Dropout(config.dropout)
        )
    
    def forward(self, x):
        B,S,C = x.shape
        if self.config.asserts: assert (B == self.config.B ) and (S <= self.config.T * self.config.K) and (C == self.config.C)
        out = self.ff(x)
        if self.config.asserts: assert out.shape == (B,S,C)
        return out


def _test_ff(config):
    ff = FeedForward(config)
    ff.eval()
    x = torch.randn(config.B, config.T, config.C)
    out = ff.forward(x)
    assert out.shape == (config.B, config.T, config.C)
    assert out.dtype == x.dtype
    assert torch.isfinite(out).all()
    print(f'out:{out}')