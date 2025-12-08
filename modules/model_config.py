from dataclasses import dataclass

@dataclass
class ModelConfig:
    # Dimension parameters
    d_batch: int   
    d_context: int
    d_channel: int 
    d_vocab: int   
    d_head_size: int
    # Other
    num_blocks: int
    num_heads: int
    dropout: float



    