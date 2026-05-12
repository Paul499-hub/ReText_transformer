import torch
# Modules
from modules.models.Memory_v0.model_parts.feed_forward import _test_ff
from modules.encoders.MathEncoder import MathEncoder

def run_tests(enc, model, config):
    _test_ff(config)
    #test_Memory_v0_transformer(enc, model, config)
    test_Memory_v0_transformer_addition(enc, model, config)

# Memory_v0 Transformer test
def test_Memory_v0_transformer(enc, model, config):
    input = torch.tensor([ enc.encode('my name is dog'), enc.encode('my name is dog')  ])
    print(f'input shape: {input.shape}')
    out = model.forward(input)
    print(f'out shape: {out.shape}')
    assert out.shape[0] == config.B
    assert out.shape[1] <= config.T
    assert out.shape[2] == config.d_vocab

def test_Memory_v0_transformer_addition(enc, model, config):
    input = torch.tensor([ enc.encode('15+19'), enc.encode('15+19')  ])
    print(f'input shape: {input.shape}')
    out = model.forward(input)
    print(f'out shape: {out.shape}')
    assert out.shape[0] == config.B
    assert out.shape[1] <= config.T
    assert out.shape[2] == config.d_vocab

def test_addition_endocder(enc:MathEncoder):
    task = "152+137="
    encoded = enc.encode(task)
    print(f'[encoded]:{encoded} \n original: {task}')
    decoded = enc.decode(encoded)
    print(f'[decoded]:{decoded} \n original: {task}')