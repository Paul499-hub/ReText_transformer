import torch.nn as nn
import torch
import tiktoken
import torch.nn.functional as F
from typing import List, Iterator
import random
import sys
import os
from copy import deepcopy
import time
# Modules
from modules.batcher import RandomBatcher
from modules.models.Transformer.Transformer import Transformer
from modules.model_config import ModelConfig
from torch.nn.utils import clip_grad_norm_

# https://hollymoviehd.cc/


def stopwatch_factory(enabled):
    def decorator(func):
        def wrapper(*args, **kwargs):
            if enabled:
                time_s = time.time()
                print(f'Timer started for function named:{func.__name__}')
            out = func(*args, **kwargs)
            if enabled:
                time_e = time.time()
                print(f' --- [Time]:{time_e - time_s}')
            return out
        return wrapper
    return decorator

# --- Helper functions
def accumulate_bucket(train_bucket:List[int], max_bucket_size_tok:int, new_int_encoded_text:List[int])->bool:
    # Type assert
    assert isinstance(train_bucket, List)
    if len(train_bucket)>0:
        assert isinstance(train_bucket[0], int)
    # Type assert
    assert isinstance(new_int_encoded_text, List)
    if len(new_int_encoded_text)>0:
        assert isinstance(new_int_encoded_text[0], int)
    # Length check and append new data 
    if len(train_bucket) < max_bucket_size_tok:
        train_bucket += new_int_encoded_text
        return True
    return False


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print( f' CUDA VER:{torch.version.cuda} py ver: {sys.version} pytorch ver: {torch.__version__} device:{device}')
enc = tiktoken.get_encoding('gpt2')
config = ModelConfig(
    # Dimension params
    d_batch=1,
    d_context=512,
    d_channel=512,
    d_vocab=enc.n_vocab, # -> 50257
    d_head_size= int(512/16),
    # Other
    num_blocks=10,
    num_heads=16,
    dropout=0.1
)
lr = 1e-4 
bucket_size = 80 
train_bucket = []
losses = []
model = Transformer(config).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr = lr)
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
step = 0
epoch = 0
print(f"Trainable parameters: {trainable_params}")
# Millions
print(f"Trainable parameters: {trainable_params/1e6:.2f}M")
# Billions
print(f"Trainable parameters: {trainable_params/1e9:.3f}B")
# Loading
os.makedirs('saved', exist_ok=True)
os.makedirs('log', exist_ok=True)
training_name = 'ReText_Harry' 
full_name = f'{training_name}_size-{trainable_params/1e9:.3f}B_B-{config.d_batch}_T-{config.d_context}_C-{config.d_channel}_nh-{config.num_heads}_tb-{config.num_blocks}_dr-{config.dropout}_V-{config.d_vocab}_lr-{lr}'
SAVE_MODEL_DIR = os.getcwd() + f'/saved/{full_name}'
continue_training = False
CONTINUE_TRAINING_FROM_DIR = os.getcwd() +'/saved/ReText_Harry_size-0.083B_B-1_T-512_C-512_nh-16_tb-10_dr-0.1_V-50257_lr-0.0001'
LOG_DIR = f'log/Log_{full_name}.txt'
# LOAD MODEL FROM FILE
if continue_training: 
  print(f'[Loading trained model...]')
  model.load_state_dict(torch.load(CONTINUE_TRAINING_FROM_DIR))
else:
    # XAVIER
    if True:
        for p in model.parameters():
            if p.dim() >1:
                nn.init.xavier_uniform_(p) 
model: Transformer = model
# Load dataset
with open('datasets/clean_harry.txt', 'r', encoding = 'utf-8') as f:
    train = f.read().split('|')
    
# Launch training loop
while True:
    epoch += 1           # Epoch counter
    c_el=0               # el counter
    time_s = time.time() # Timer start for bucke accumulation
    # Training loop
    for el in train:
        c_el+=1
        if accumulate_bucket(
                                train_bucket, 
                                max_bucket_size_tok=bucket_size*config.d_context, 
                                new_int_encoded_text=enc.encode(el) # <--  Token indexes accumulates to train_bucket 
                            ): 
            pass
        # Once bucket has accumulted 
        else:
            print(f' Bucket accumulated in :{time.time() - time_s}')
            # Timer start for training
            time_s = time.time() 
            in_len = 0
            # Convert to tensor
            train_bucket:torch.Tensor = torch.tensor(train_bucket, dtype = torch.long, device=device)
            # Shuffle and iterate
            rbat = RandomBatcher(train_bucket, config.d_context+1, config.d_batch) # context_length (T) is +1 because of right-shift
            while True:
                # Get q, a 
                q_a = rbat.random_batch_torch() # shape: [B,T+1]
                if q_a is None:
                    break
                q:torch.Tensor = q_a[:,:-1]     # shape: [B,T]
                a:torch.Tensor = q_a[:,1:]      # shape: [B,T]
                # Pass to model 
                out, loss = model.forward(q, a)
                # Show progress - loss
                losses.append(loss.item())
                step+=1
                if step % 10 == 0:
                    print(f"Epoch -> {epoch} | Step -> {step} | Loss -> {sum(losses[-100:])/100:.4f} | Tok/s -> {(rbat.ptr*config.d_context)/(time.time()-time_s):.3f}")
                    for b in q_a:
                        b_text = enc.decode( b.cpu().tolist() )
                        # Print model input
                        if False: 
                            print(f'\n --- TRAINING INPUT q_a TRANSLATED ----')
                            print(b_text)
                            print(f'\n\n -- TRAINING INPUT RAW TENSOR ---- ')
                            print(q_a)
                if step % 500 == 0:
                    print(f"Step {step}, Loss: {sum(losses[-100:])/100:.4f}")
                    # Greedy decode the current batch
                    with torch.no_grad():
                        logits = out  # shape [B, T, V]
                        greedy_tokens = torch.argmax(logits, dim=-1)  # [B, T] --> torch.Size([1, 512])
                        for batch in greedy_tokens:
                            b_text = enc.decode( batch.cpu().tolist() )
                            if True: # Print
                                print(f'\n\n ~~~ MODEL TEXT ~~~')
                                print(b_text)
                                #print(f'\n ~~~ MODEL RAW ~~~')
                                #print( batch)
                                #print( batch.shape)
                        g_out = ''.join(b_text)
                        with open(LOG_DIR, 'a', encoding='utf-8') as f:
                            print( f'\n Greedy output [epoch -> {epoch} | step -> {step}]: {g_out}', file=f)
                            print(f' Bucket ptr (token idx):{rbat.ptr} _max:{rbat._max} --> {(rbat.ptr/rbat._max)*100:.4f}% bucket complete... ')
                            print(f' Total training elements (sentences): {len(train)} current: {c_el} --> { (c_el/len(train)*100):.4f}% dataset complete...')
                            torch.save( deepcopy(model.state_dict()) , SAVE_MODEL_DIR  )
    
                # Skip update on NaN/Inf loss
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"[WARN] NaN/Inf loss at step {step} -> {float(loss)}; skipping update")
                    optimizer.zero_grad(set_to_none=True)
                    continue
                # Hyperparameters for stability
                max_grad_norm = 1.0
                loss_spike_threshold = 2.0
                spike_multiplier = 5.0 
                # Backprop
                loss.backward()
                # Clip grads and get norm
                total_norm = clip_grad_norm_(model.parameters(), max_grad_norm)
                # Detect bad grad norm (Nan/Inf) or extremely large norm
                if torch.isnan(total_norm) or torch.isinf(total_norm) or total_norm > 1e6:
                    print(f"[WARN] Bad grad norm {total_norm} at step {step}; skipping update")
                    optimizer.zero_grad(set_to_none=True)
                    continue
                # Skip update if loss is a big spike vs recent losses
                recent_count = min(len(losses), 100)
                recent_avg = sum(losses[-recent_count:]) / recent_count if recent_count>0 else float(loss)
                if loss.item() > max(loss_spike_threshold, spike_multiplier * recent_avg):
                    print(f"[WARN] Loss spike {loss.item():.4f} (recent avg {recent_avg:.4f}) at step {step}; skipping update")
                    optimizer.zero_grad(set_to_none=True)
                    continue

                # Update good -> step
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                # Log grad norm occasionally
                if step % 500 == 0:
                    print(f"[INFO] step {step} loss {loss.item():.4f} grad_norm {total_norm:.4f}")
            # Empty bucket
            train_bucket=[]
            losses = [] 

# Inference
if True:
    with torch.no_grad():
        _input = input('   User >>>')
        print('<<<')
        while True:
            # Input
            model.eval() # Turn off dropout layer
            print(f'Passing to model: {_input}')
            out = model.forward( torch.tensor(enc.encode(_input), device=device).unsqueeze(0)  ) # B, T -- Batch is 1 so unsqueeze 
            # Output
            logits = out
            greedy = torch.argmax(logits, dim=-1)
            for batch in greedy:
                last_pred = enc.decode([batch[-1].item()])
                print(f'      Model [-1] >>>{last_pred}')
                print('<<<')
                _input += last_pred
            input(f'[System]: continue?')

# Cata cleaning 
if False:
    for el in train:
        step+=1
        if step % 10000 == 0:
            print(f'el count:{step}')
        el = el.replace('“','"').replace('”','"')
        el = el.replace("’","'").replace("‘","'")
        with open('clean_harry','a', encoding='utf-8') as f:
            f.write( f'{el}|')
        encoded = enc.encode(el)
        #print(f'el text:{el}')
        #print(f'text encoded: {enc.encode(el)}')
        #print(f'text decoded: {enc.decode(encoded)}')
        for num in encoded:
            if num == 564:
                print(f'--FOUND: {num} --DECODING--> {enc.decode([num])} ')
                exit()



