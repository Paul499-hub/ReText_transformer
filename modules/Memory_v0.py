import torch.nn as nn
import tiktoken
import torch
import time
import sys
import os
# Modules
from modules.utils.ModelInputGenerators import generate_model_input
from modules.models.Memory_v0.transformer import Transformer
from modules.test import run_tests, test_addition_endocder
from modules.encoders.MathEncoder import MathEncoder
from modules.configs.Memory_v0 import ModelConfig
from modules.utils.SaveLoader import SaveLoader
from modules.plotter.pyplot import Plotter
from modules.utils.Monitors import Monitor

# --------- Launch --------------------------
# Set-ExecutionPolicy -Scope Process Bypass
# .\.venv\Scripts\activate
# python -m modules.Memory_v0

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print( f' CUDA VER:{torch.version.cuda} py ver: {sys.version} pytorch ver: {torch.__version__} device:{device}')
enc = MathEncoder()
#enc = tiktoken.get_encoding('gpt2') # 50257
config = ModelConfig(
    B = 128,
    T = 11,
    C = 512,
    d_vocab = enc.n_vocab,
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
# model = torch.compile(model)
save_loader = SaveLoader(model, "Math_Addition_XSA")
save_loader.create_dirs()
save_loader.load_model_from_file(load_from_dir=save_loader.save_dir/'Math_Addition_Task_0.02B.pt') #<-- 😱 LOAD EXISTING MODEL
optimizer = torch.optim.AdamW(model.parameters(), lr = lr)
plotter = Plotter()
scaler = torch.cuda.amp.GradScaler() # AMP fp32 -> fp16 scaler (less precision where needed) (backward)
monitor = Monitor()
use_AMP = False
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {trainable_params} --> {trainable_params/1e6:.2f}M --> {trainable_params/1e9:.3f}B ")
print(f'ROOT_DIR:{save_loader.root_dir}')
print(f"model device: {next(model.parameters()).device}")

# ---------------------------------------------------- TRAINING LOOP -----------------------------------------------
if False:
    losses=[]
    loss_avg=[]
    s_t = time.time()
    stop_step = 1000000  # <-- step = 1 training example of shape [T]
    print_q_a_rate = config.B * 200
    save_rate = config.B * 600
    step_counter = 0
    tokens_seen_window = 0
    try:
        while True:
            step_counter += config.B  #  individual examples seen
            if step_counter > stop_step:
                break
            q_batch, a_batch, q_0_str, a_0_str = generate_model_input(enc, config, device, True)
            if use_AMP:
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    logits, loss = model.forward(q_batch, a_batch)
            else:
                logits, loss = model.forward(q_batch, a_batch)
            losses.append(loss.item()) # <--- batch mean loss item
            tokens_seen_window += q_batch.numel()
            plotter.maybe_loss_graph(step_counter, save_rate, loss_avg, save_loader.root_dir/'plt'/f'{save_loader.full_name}.png' )
            s_t, tokens_seen_window = monitor.maybe_log(
                            save_loader, enc, logits, loss, loss_avg, losses,
                            q_0_str, a_0_str, s_t, tokens_seen_window,
                            step_counter, stop_step, print_q_a_rate)
            monitor.maybe_save(save_loader,step_counter, save_rate)
            if use_AMP:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
    except KeyboardInterrupt:
        print(f'Training interrupted.')


# ------------------------------ INFERENCE ----------------
if True:
    config.B = 1 # Set batch size to 1
    while True:
        u_in = input("User input >>>")
        while True:
            u_in_enc = enc.encode(u_in)
            u_in_tensor = torch.tensor([u_in_enc], device=device)
            logits = model.forward(u_in_tensor)
            probs = torch.softmax(logits, dim=-1)
            pred_ids = torch.argmax(probs, dim=-1)
            ai_text_full = enc.decode(pred_ids[0].tolist()) # <-- batch aware text decode
            last_ai_char = ai_text_full[-1]
            #print(f'original input to model [str]:{u_in}')
            # print(f'🦑⏳ All ai generated tokens:{ai_text_full}')
            print(f'    🦑 Latest ai generated token | {last_ai_char}')
            u_in += last_ai_char
            print(f'        🔌 updated input str | {u_in}')
            if input('🚧 Continue generation? [y/n] >>>') in ['n', 'N', 'no', 'exit']: break
        if input('🚧 Continue inference? [y/n] >>>') in ['n', 'N', 'no', 'exit']: break



# ---- TESTS ----
# test_addition_endocder(enc)
# run_tests(enc, model, config)
# q_batch, a_batch , q_0_str, a_0_str = generate_model_input(enc,config,device,True)
# print(f'\n q_batch >>> {q_batch} \n a_batch >>> {a_batch} \n q_0_str >>> {q_0_str} \n a_0_str >>> {a_0_str}')