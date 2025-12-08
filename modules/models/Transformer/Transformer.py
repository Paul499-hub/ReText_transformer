import torch
import torch.nn as nn
import tiktoken
import torch.nn.functional as F

from modules.models.Transformer.model_parts.transformer_block import Block
from modules.model_config import ModelConfig

class Transformer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.token_embeddings = nn.Embedding( config.d_vocab, config.d_channel)
        self.positional_embeddings = nn.Embedding( config.d_context, config.d_channel )
        self.t_blocks = nn.Sequential(*[Block( self.config ) for _ in range( config.num_blocks )])
        self.ln = nn.LayerNorm(config.d_channel)
        self.output_layer = nn.Linear( config.d_channel, config.d_vocab )
        
    def forward(self, x:torch.Tensor, targets = None):
        """
        `x` is a tokenized string tensor: like this: torch.tensor([enc.encode('hello world')]) 
        - shape: [B, T] (T could be less than self.config.d_context)

        `targets` is the same as `x` but right-shifted
        - shape: [B, T] (T could be less than self.config.d_context)
        """
        B,T = x.shape
        assert B == self.config.d_batch 
        assert T <= self.config.d_context
        tok_emb = self.token_embeddings(x)                                      # shape: [B,T,C]
        pos_emb = self.positional_embeddings(torch.arange(T, device=x.device))  # shape: [T,C]
        out = tok_emb + pos_emb.unsqueeze(0)                                    # [B,T,C] + [1,T,C] = [B,T,C]
        out = self.t_blocks(out)
        out = self.ln(out)
        out = self.output_layer(out)                                            # shape: [B,T,V]
        # INFERENCE
        if targets == None:
            return out
        # TRAINING-CALCULATE LOSS
        else:
            B, T, V = out.shape
            loss = F.cross_entropy(out.reshape(B*T, V), targets.reshape(B*T))
            return out, loss
        
# test_config.py
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
    enc = tiktoken.get_encoding('gpt2')
    from test_config import config
    model = Transformer(config)
    # Inference example
    input_tensor = torch.tensor( [enc.encode('hello world')] ) # shape: [B, T]
    out = model.forward( input_tensor )
    print(f' ---------- TEST: Inference ----------')
    print(f'input tensor:{input_tensor}')
    print(f'out:{out}')
    print(f'{out.shape}') #  B, T, V  
    # Training example 
    input_tensor = torch.tensor( [[1,2,3]] )                                   
    out, loss = model.forward( 
                                x=input_tensor,                     # shape: [B, T]
                                targets=torch.tensor( [[2,3,4]] )   # shape: [B, T]
                            )
    print(f' ---------- TEST: Training ----------')
    print(f'model out: {out}')
    print(f'loss: {loss}')

    # -------- CONSOLE OUTPUT ------------ 
    # PS C:\Users\Saulius\Desktop\ReText\modules> uv run model.py
    #  ---------- TEST: Inference ----------
    # input tensor:tensor([[31373,   995]])
    # out:tensor([[[-0.0302, -0.2141,  0.4198,  ..., -0.9277,  0.1073, -0.7822],
    #          [-1.0282, -0.8396,  0.6167,  ..., -0.1573,  0.2821, -0.2977]]],
    #        grad_fn=<ViewBackward0>)
    # torch.Size([1, 2, 50257])
    #  ---------- TEST: Training ----------
    # model out: tensor([[[-0.5367,  0.7192,  1.0213,  ..., -0.7903,  0.2292,  0.0171],
    #          [-0.7257, -0.1950,  0.8076,  ..., -0.1298,  0.3109,  0.2797],
    #          [-0.0321,  0.7886,  0.4116,  ...,  0.4695, -0.9084, -0.0106]]],
    #        grad_fn=<ViewBackward0>)
    # loss: 11.264859199523926
    # PS C:\Users\Saulius\Desktop\ReText\modules>
        

# example_usage()