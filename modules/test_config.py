from modules.model_config import ModelConfig
config = ModelConfig(
    # Dimension params
    d_batch=1,
    d_context=32,
    d_channel=64,
    d_vocab=50257,
    d_head_size=2,
    # Other
    num_blocks=2,
    num_heads=2,
    dropout=0.1,
)
