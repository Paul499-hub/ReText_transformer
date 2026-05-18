from encodec import EncodecModel
import torch
import time
import sys
# Modules
from modules.configs.Musician import ModelConfig
from modules.models.Musician.transformer import Transformer
from modules.utils.Sound import make_test_wav, tokenize_audio
from modules.utils.find_project_root import _find_project_root

# --------- Launch --------------------------
# Set-ExecutionPolicy -Scope Process Bypass
# .\.venv\Scripts\activate
# python -m modules.Memory_v0

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print( f' CUDA VER:{torch.version.cuda} py ver: {sys.version} pytorch ver: {torch.__version__} device:{device}')
project_root = _find_project_root(__file__)
sound_file_path = project_root / 'datasets' / 'test.wav'
encodec_model = EncodecModel.encodec_model_24khz()
config = ModelConfig(
    B = 1,
    K = 8, # codebook (sound feature dimension)
    T = 200,
    C = 512,
    d_vocab = encodec_model.quantizer.bins,
    t_blocks = 6,
    num_heads = 8,
    d_head_size = 64,
    dropout = 0.1,
    device = device,
    asserts = False,
    XSA = True,
)
lr = 1e-4 
model = Transformer(config).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr = lr)
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {trainable_params} --> {trainable_params/1e6:.2f}M --> {trainable_params/1e9:.3f}B ")
print(f'ROOT_DIR:{project_root}')
print(f"model device: {next(model.parameters()).device}")


#make_test_wav(sound_file_path)
audio_tokens = tokenize_audio(sound_file_path, encodec_model) # 1 batch 8 rows/codebooks 225 columns/frames = [B,K,T]

# flat_tokens = audio_tokens.permute(0,2,1).reshape(audio_tokens.shape[0], -1)
# assert flat_tokens.shape == (B, K*T)

def extract_q_a_from_full_sample(sample: torch.Tensor, config:ModelConfig, asserts:bool=False, prints:bool=False):
    # Note: We can make B K T work by model predicting one codebook entry at a time, but we pad rest of K frame, and pass
    # as input from 0 codebook, just longer + padded sequence now. Input then is [B,K,T+1] whitch is codebook divisable.
    B, K, T_sample = sample.shape # [1, 8, 225]
    if asserts: assert B == config.B
    if asserts: assert K == config.K
    if asserts: assert T_sample >= config.T + 1 
    # create input tensor B,K,T 
    q = sample[:,:,:config.T].clone() # [B,K,T]
    if asserts: assert q.shape == (B, K, config.T)
    # Permute and flatten Q
    # q = q.permute(0,2,1).reshape(B, config.T*K) # <------ DISABLED, ( model expects B,K,T )
    # Permute and flatten A
    a = sample[:,:, :config.T+1] # [B,K,T+1] = torch.Size([1, 8, 201])
    if asserts: assert a.shape == (B,K, config.T+1)
    a = a.permute(0,2,1)
    if asserts: assert a.shape == (B, config.T+1, K)
    a = a.reshape(B, (config.T+1) * K )
    if asserts: assert a.shape == (B, (config.T+1)*K )
    # Remove 7 (K-1) 
    a = a[:,:-(config.K-1)]
    if asserts: assert a.shape == (B, config.T*K+1)
    # create target right shifted
    a = a[:,1:]
    if asserts: assert a.shape == (B, config.T*K)
    if prints: # --- TEST input flatten to compare q a
        print(f'a shape:{a.shape}') #  [B, T*K+1 ] = torch.Size([1, 1600])
        print(f' q shape: {q.shape} | q:{q}') # q shape: torch.Size([1, 8, 200])
        q = q.permute(0,2,1)
        q = q.reshape(B, config.T * K)
        print(f'🧠 [model input][L-SHIFT] q shape: {q.shape} | q:{q[0,-10:]}')
        print(f'✍️ [targets][R-SHIFT]     a shape:{a.shape} | a:{a[0,-10:]}') #  [B, T*K+1 ] = torch.Size([1, 1600])
    return q, a

q, a = extract_q_a_from_full_sample( audio_tokens, config, True, True)
print(f'>>>>>>>> q shape:{q.shape} | a shape: {a.shape}')

if False: # Training loop
    q, a = extract_q_a_from_full_sample( audio_tokens, config, True, False)
    print(f'>>>>>>>> q shape:{q.shape} | a shape: {a.shape}')
    # >>>>>>>> q shape:torch.Size([1, 8, 200]) | a shape: torch.Size([1, 1600])

    out = model.forward(q)
    print(f'🦊 out shape:{out.shape}')
    #🦊 out shape:torch.Size([1, 1600, 1024])

    for i in range(100000):
        out, loss = model.forward(q,a)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if i % 100 == 0:
            print(f'🦊 out shape:{out.shape} | loss:{loss}')

# TODO - REdesign model to be able to continue off random codebook K dimension offset
#[B,K,T] input is easy to embed but bad for autoregressive generation.
#Flat [B,S] input is needed so generated single codebook tokens can be appended and fed back into the model.
        

