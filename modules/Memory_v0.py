import torch.nn as nn
import tiktoken
import torch
import time
import sys
import os
# Modules
from modules.test import run_tests, test_addition_endocder
from modules.models.Memory_v0.transformer import Transformer
from modules.encoders.MathEncoder import MathEncoder
from modules.configs.Memory_v0 import ModelConfig
from modules.utils.SaveLoader import SaveLoader
from modules.plotter.pyplot import Plotter
from modules.utils.ModelInputGenerators import generate_model_input
# --------- Launch --------------------------
# Set-ExecutionPolicy -Scope Process Bypass
# .\.venv\Scripts\activate
# python -m modules.Memory_v0

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print( f' CUDA VER:{torch.version.cuda} py ver: {sys.version} pytorch ver: {torch.__version__} device:{device}')
enc = MathEncoder()
#enc = tiktoken.get_encoding('gpt2') # 50257
config = ModelConfig(
    B = 2,
    T = 11,
    C = 512,
    d_vocab = enc.n_vocab,
    t_blocks = 6,
    num_heads = 8,
    d_head_size = 64,
    dropout = 0.1,
    device = device,
    asserts = False
)
lr = 1e-4 
model = Transformer(config).to(device)
save_loader = SaveLoader(model, "Math_Addition_Task_i2")
save_loader.create_dirs()
#save_loader.load_model_from_file(load_from_dir=save_loader.save_dir/'Math_Addition_Task_0.02B.pt') #<-- LOAD EXISTING MODEL
optimizer = torch.optim.AdamW(model.parameters(), lr = lr)
plotter = Plotter()
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {trainable_params} --> {trainable_params/1e6:.2f}M --> {trainable_params/1e9:.3f}B ")
print(f'ROOT_DIR:{save_loader.root_dir}')
print(f"model device: {next(model.parameters()).device}")

# ---------------------------------------------------- TRAINING LOOP -----------------------------------------------
if True:
    losses=[]
    loss_avg=[]
    s_t = time.time()
    stop_step = 100000  # <-- step = 1 training example of shape [T]
    print_q_a_rate = config.B * 200
    save_rate = config.B * 600
    while_counter = 0
    step_counter = 0
    tokens_seen_window = 0
    try:
        while True:
            while_counter+=1                         #  optimizer steps / batches
            step_counter = while_counter * config.B  #  individual examples seen
            if step_counter > stop_step:
                break
            q_batch, a_batch, q_0_str, a_0_str = generate_model_input(enc, config, device, True)
            logits, loss = model.forward(q_batch, a_batch)
            losses.append(loss.item()) # <--- batch mean loss item
            tokens_seen_window += q_batch.numel()
            # Show info, save plots
            if step_counter % print_q_a_rate == 0:
                time_elapsed_window = time.time() - s_t
                s_t = time.time()
                tokens_seen_save = tokens_seen_window
                tokens_seen_window = 0
                loss_avg.append(sum(losses) / len(losses))
                l_num = len(losses)
                losses = []
                probs = torch.softmax(logits, dim=-1)
                pred_ids = torch.argmax(probs, dim=-1)
                text = enc.decode(pred_ids[0].tolist()) # <-- batch aware text decode
                progress = (step_counter / stop_step) * 100
                # Print
                log_text = '\n'.join([
                    '\n',
                    "=" * 60,
                    f"🦵 steps {step_counter}/{stop_step} | {progress:.2f}%",
                    f"⏱️ elapsed_window: {time_elapsed_window:2f}s",
                    f'⌚ tok/s: {tokens_seen_save/ time_elapsed_window:.2f}',
                    f"❓ Q : {q_0_str}",
                    f" ---> 🎯 A : {a_0_str}",
                    f" ---> 🦑 P : {text}",
                    f"📉 loss: {loss.item():.4f}",
                    f"📊 avg : {loss_avg[-1]:.4f}",
                    "=" * 60
                ])
                print(f'[progress] step {step_counter}/{stop_step} | {progress:.2f}%')
                save_loader.append_log(log_text)
            if step_counter % save_rate == 0:
                save_loader.save_model()
            # Backprop
            loss.backward()
            # Update good -> step
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
    except KeyboardInterrupt:
        print(f'Training interrupted.')
    plotter.show_loss_graph(loss_avg, print_q_a_rate )


# ------------------------------ INFERENCE ----------------
if False:
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