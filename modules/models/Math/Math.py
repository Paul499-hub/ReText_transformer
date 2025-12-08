import torch
import torch.nn as nn
import tiktoken
import torch.nn.functional as F

from modules.models.Math.model_parts.math_block import Block
from modules.test_config import config, ModelConfig


# Prediction must give one single new word - non autoregressive (no teacher forcing)
# Since im giving verious math operations to the model. IT might just pass data between layers 
# in a way that 'READS' waht the next token is from the front of the sequence.
# So input must be array of tokens, output is ONE SINGLE token. 

class SymbolicAlgebraEngine(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.token_embeddings = nn.Embedding( config.d_vocab, config.d_channel)
        self.positional_embeddings = nn.Embedding( config.d_context, config.d_channel )
        self.blocks = nn.Sequential(*[Block( self.config ) for _ in range( config.num_blocks )])
        # Final translate layer: map aggregated vector to vocab
        self.translate_layer = nn.Linear(config.d_channel, config.d_vocab)  # [B,C] -> [B,V]
        # Attention weights for aggregation
        self.attn_scores = nn.Linear(config.d_channel, 1)  # [B, T, C] -> [B, T, 1]

    def forward(self, x:torch.Tensor, targets = None):
        """
        `x` is a tokenized string tensor: like this: torch.tensor([enc.encode('hello world')]) 
        - shape: [B, T] (T could be less than self.config.d_context)
        """
        B,T = x.shape
        #print(f'input x shape:{x.shape}')
        # input x shape:torch.Size([1, 2])

        assert B == self.config.d_batch 
        assert T <= self.config.d_context
        tok_emb = self.token_embeddings(x)                                      # shape: [B,T,C]
        pos_emb = self.positional_embeddings(torch.arange(T, device=x.device))  # shape: [T,C]
        out = tok_emb + pos_emb.unsqueeze(0)                                    # [B,T,C] + [1,T,C] = [B,T,C]
        out = self.blocks(out)                  # [B,T,C]  
        #print(f'blocks out shape:{out.shape}')
        # blocks out shape:torch.Size([1, 2, 64]) --> B,T,C

        # Compute attention weights
        scores = self.attn_scores(out)         # [B, T, 1]
        weights = torch.softmax(scores, dim=1) # [B, T, 1]
        # Weighted sum over sequence
        aggregated = torch.sum(out * weights, dim=1)  # [B, C]

        # Map to vocab
        out = self.translate_layer(aggregated)  # [B,V]
        #print(f'translate_layer:{out.shape}')
        # translate_layer:torch.Size([1, 50257])

        # INFERENCE
        if targets == None:
            return out
        # TRAINING-CALCULATE LOSS
        else:
            B, V = out.shape
            loss = F.cross_entropy(out, targets)
            return out, loss
        
def example_usage():
    enc = tiktoken.get_encoding('gpt2')
    model = SymbolicAlgebraEngine(config)
    # Inference example
    input_tensor = torch.tensor( [enc.encode('hello world')] ) # shape: [B, T]
    out = model.forward( input_tensor )
    print(f' ---------- TEST: Inference ----------')
    print(f'input tensor:{input_tensor}')
    print(f'out:{out}')
    print(f'{out.shape}') #  B, V  
    # Training example 
    input_tensor = torch.tensor( [[1,2,3]] )                                   
    out, loss = model.forward( 
                                x=input_tensor,                     # shape: [B, T]
                                targets=torch.tensor( [4] )   # shape: [B]
                            )
    print(f' ---------- TEST: Training ----------')
    print(f'model out: {out}')
    print(f'loss: {loss}')

    # -------- CONSOLE OUTPUT ------------ 
    # PS C:\Users\Saulius\Desktop\ReText\modules\models\Math> uv run Math.py
    #  ---------- TEST: Inference ----------
    # input tensor:tensor([[31373,   995]])
    # out:tensor([[ 0.4662,  0.0066,  0.0278,  ..., -0.3675, -0.8608, -0.4335]],
    #        grad_fn=<AddmmBackward0>)
    # torch.Size([1, 50257])
    #  ---------- TEST: Training ----------
    # model out: tensor([[ 0.6972, -0.5176, -0.2329,  ..., -0.0944,  0.3191,  0.1585]],
    #        grad_fn=<AddmmBackward0>)
    # loss: 11.862524032592773
    # PS C:\Users\Saulius\Desktop\ReText\modules\models\Math>
        

example_usage()