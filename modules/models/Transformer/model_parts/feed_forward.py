import torch
import torch.nn as nn

from modules.model_config import ModelConfig

class FeedForward(nn.Module):
    def __init__(self, config:ModelConfig):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear( config.d_channel, 4 * config.d_channel ),
            nn.GELU(),
            nn.Linear( 4 * config.d_channel, config.d_channel ),
            nn.Dropout( config.dropout )
        )
        
    def forward(self, x):
        return self.network(x)
    
def example_usage():
    from test_config import config
    ff = FeedForward(config)
    out = ff.forward( torch.randn( config.d_batch, config.d_context-5, config.d_channel ))
    print( f'ff shape: {out.shape}') # Expected B T C 
    # GOT [B, T, C]
    # Works with T - 5 aswell, context_length less than max.
#example_usage()