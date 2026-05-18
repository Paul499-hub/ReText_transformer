from dataclasses import dataclass 
import torch

@dataclass 
class ModelConfig:
    # Model parameters
    B:int # Batch size
    T:int # Context length
    K:int # Codebook (sound feature dimension)
    C:int # Channel dimension, dimensions for each token
    d_vocab:int # How many different token in vocabulary
    t_blocks:int # Transformer blocks
    num_heads:int # Number of attention heads per mha
    d_head_size:int 
    device: torch.device
    # Other
    dropout: float
    asserts: bool
    XSA: bool # Exclusice self attention
