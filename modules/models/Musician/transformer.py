import torch.nn.functional as F
import torch.nn as nn
import torch 
# Modules
from modules.configs.Musician import ModelConfig
from modules.models.Musician.model_parts.transformer_block import Block

class Transformer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.d_vocab, config.C)
        self.cod_emb = nn.Embedding(config.K, config.C)
        self.pos_emb = nn.Embedding(config.T, config.C)
        self.t_blocks = nn.Sequential(*[Block(config) for _ in range(config.t_blocks)])
        self.proj = nn.Linear(config.C, config.d_vocab)

    def forward(self, x:torch.tensor, targets=None):
        x = x.to(self.config.device)
        B,K,T = x.shape
        C = self.config.C
        if self.config.asserts: assert (B == self.config.B)
        if self.config.asserts: assert (K == self.config.K) 
        if self.config.asserts: assert (T <= self.config.T)
        # ------ Generate token embeddings
        tok_emb = self.tok_emb(x)
        if self.config.asserts: assert tok_emb.shape == (B,K,T,C)
        # ----- Generate codebook embeddings
        k_ids = torch.arange(K, device=x.device) 
        cod_emb = self.cod_emb(k_ids)
        if self.config.asserts: assert cod_emb.shape == (K,C)
        cod_emb = cod_emb.view(1, K, 1, C) 
        if self.config.asserts: assert cod_emb.shape == (1,K,1,C)
        # ---- Generate pos embeddings with arrange
        pos_emb = torch.arange(T, device = x.device)
        if self.config.asserts: assert pos_emb.shape == (T,)
        pos_emb = self.pos_emb(pos_emb)
        if self.config.asserts: assert pos_emb.shape == (T,C)
        pos_emb = pos_emb.view(1, 1, T, C)
        if self.config.asserts: assert pos_emb.shape == (1,1,T,C)
        # ---- Add pos_emb + tok_emb (broadcasts along Batch dim[0] -- 1) 
        out = tok_emb + pos_emb + cod_emb
        if self.config.asserts: assert out.shape == (B,K,T,C)
        out = out.permute(0,2,1,3)
        if self.config.asserts: assert out.shape == (B,T,K,C)
        out = out.reshape(B, K * T, C)
        if self.config.asserts: assert out.shape == (B,T*K,C)
        # Pass into multi headed attention T-Blocks
        out = self.t_blocks(out)
        if self.config.asserts: assert out.shape == (B,T*K,C)
        # Project into vocabulary 
        out = self.proj(out)
        if self.config.asserts: assert out.shape == (B,T*K, self.config.d_vocab)
        # INFERENCE / TRAINING
        if targets is None:
            return out 
        else:
            targets = targets.to(self.config.device)
            B,S,V = out.shape # V = d_vocab
            if self.config.asserts: assert targets.shape[0] == B
            if self.config.asserts: assert targets.shape[1] == S
            loss = F.cross_entropy(
                out.reshape(B*S, V), 
                targets.reshape(B*S),
                ignore_index=-100
            )
            return out, loss
