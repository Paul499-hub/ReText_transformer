import torch
# Modules
from modules.encoders.MathEncoder import MathEncoder
from modules.configs.Memory_v0 import ModelConfig

# Memory_v0 
def generate_model_input(enc:MathEncoder, config:ModelConfig, device, ignore=False):
    math_samples_batch = [enc.get_math_eq() for _ in range(config.B)]
    q_batch = []
    a_batch = []
    for el in math_samples_batch:
        q_str = el[:-1]
        a_str = el[1:]
        if ignore: eq_idx = a_str.index('=')
        assert len(q_str) == len(a_str)
        q_batch.append(enc.encode(q_str))
        a_encoded = enc.encode(a_str)
        if ignore:
            a_encoded[:eq_idx] = [-100] * eq_idx
        a_batch.append(a_encoded)
    q_0_str = math_samples_batch[0][:-1]
    a_0_str = math_samples_batch[0][1:]
    q_batch = torch.tensor( q_batch, device=device )
    a_batch = torch.tensor( a_batch, device=device )
    assert q_batch.shape[0] == config.B
    assert q_batch.shape[1] == config.T
    assert a_batch.shape[0] == config.B
    assert a_batch.shape[1] == config.T
    return q_batch, a_batch , q_0_str, a_0_str