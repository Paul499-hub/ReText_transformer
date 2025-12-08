import torch
import torch.nn as nn
from torch.nn import functional as F
import os
from copy import deepcopy
import sys
import random
import tiktoken
from torch.nn.utils import clip_grad_norm_
import matplotlib.pyplot as plt
import time
from torch.utils.data import DataLoader, Dataset, RandomSampler
from datasets import load_dataset
from nltk.translate.bleu_score import sentence_bleu
# import transformers #< -- 4.36.0.dev0 git+https://github.com/huggingface/transformers.git
# import huggingface_hub # < -- 0.19.4
# #from transformers import LLaMAForCausalLM, LlamaTokenizer

# Python 3.9.12
# CUDA 11.8
#conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia
#OR 
#poetry run pip install torch==2.2.0+cu118 torchvision==0.17.0+cu118 torchaudio==2.2.0+cu118 -f https://download.pytorch.org/whl/cu118/torch_stable.html

# ------------------------------------------- COMMAND TO RUN SCRIPT BELOW  ---------
#PS C:\Users\Saulius\Desktop\GODFolder\AiPytorchTut\YoutubeTransformer> C:\Users\Saulius\anaconda3\envs\ReTextENV\python.exe ReText.py

# BELU
if False: 
    # Reference and candidate sentences
    reference = [['the', 'cat', 'is', 'on', 'the', 'mat']]
    candidate = ['the', 'cat', 'is', 'on', 'the', 'mat']
    # Compute BLEU score
    bleu_score = sentence_bleu(reference, candidate)
    print(f'BLEU Score: {bleu_score}')

vocab_type ='gtp' # char/gtp/orca

class batch_pos_emb(nn.Module):
    def __init__(self, batch_size:int, max_sen_len:int, w_emb_d:int,  ):
        super(batch_pos_emb,self).__init__()
        # INPUT IS BACH * t * k . WORDS SHOULD ALREADY BE EMBEDDED INTO (6) DIMENSIONS
        self.initial = torch.zeros(batch_size, max_sen_len, w_emb_d, device=device)
        self.initial.requires_grad=False
        # EVERY POS UNTIL MAX SEN LEN
        pos = torch.arange(max_sen_len, device=device).unsqueeze(-1)
        # EVERY SECOND DIM 
        _2i = torch.arange(w_emb_d, device=device)[0::2] # [0,2,4]
        math1 = 10000 ** (_2i / w_emb_d ) # 10 000 ** ( 0.3, 0.6, 0,9)  == 24 , 334 , 7000 CLOCK HANDLES 
        math2 = pos / math1  
        self.initial[:, :, 0::2]= torch.sin(math2)
        self.initial[:, :, 1::2]= torch.cos(math2)
        #print(f' check if pos embedding matrix (before addition with x) is correct : \n {self.initial}')

    def forward(self, x:torch.tensor):
        return (x + self.initial)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding_layer = nn.Embedding(vocab_size,n_embed) #CONVERTS INTEGERS INTO N-DIMENSIONAL VECTORS (RELATIVE IDX WILL BE CLOSER)
        self.positional_embedding = nn.Embedding(context_size,n_embed) # context size because we embed positions from 0 to context size.
        self.t_blocks = nn.Sequential(*[Block() for _ in range(t_blocks)])
        self.ln = nn.LayerNorm(n_embed)
        self.l_one = nn.Linear(n_embed, vocab_size) 

    def forward(self, q, a=None):
        B,T = q.shape
        tok_emb = self.embedding_layer(q) # [batch_size, context_size] -> [batch_size, context_size, n_embed]
        pos_emb = self.positional_embedding( torch.arange(T, device=device) ) # (T,C)
        x = tok_emb+pos_emb
        x= self.t_blocks(x)
        x=self.ln(x)
        out = self.l_one(x) # [batch_size, context_size, l_one:out(vocab_size)]
        if a is None:
            return out
        else:
            B , T , C = out.shape
            out = out.view(B*T,C) # LOGITS FOR CROSSENTROPY HAVE TO NOT HAVE BATCH DIM (SECOND DIM MUST BE C)
            a = a.view(B*T) # TARGETS DONT HAVE A CHANNEL DIM (NOT EMBEDED)
            loss = F.cross_entropy(out, a)
            perplexity = torch.exp(torch.mean(loss))
            return out, loss , perplexity
    
    def generate(self,  q:torch.tensor, n_generate_chars:int):
        s = q   #<-- s = FULL LENGTH RESPONSE q = SLIDING 
        for _ in range(n_generate_chars):
            q = q[:, -context_size:]  #<- SLIDING WINDOW CONTEXT
            logits = self.forward(q) 
            # CHOOSE LAST TIME DIM... B,T,C -> B,C
            logits = logits[:,-1,:] 
            # APPLY TEMPERATURE SCALING
            scaled_logits = logits / temperature
            # PROBABILITY -> CHOICE B,C -> B,1
            probs = F.softmax(scaled_logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1) 
            #APPEND TO LAST PREDICTION
            s = torch.cat( (s, next_idx), dim=-1)#<- FULL GENERATED RESPONSE
            q = torch.cat((q, next_idx ), dim = -1) #(B, T+1)#<- SLIDING WINDOW CONTEXT
        return s  #<- TENSOR , LIST OF ALL GENERATED INDICES

class Head(nn.Module):
    def __init__(self):
        super().__init__()
        if False: # < -- SINGLE LAYER KVQ 
            self.lkey = nn.Linear(n_embed, head_size, bias=False)
            self.lquery = nn.Linear(n_embed, head_size, bias=False)
            self.lvalue = nn.Linear(n_embed, head_size, bias=False)
        # MULTI LINEAR ATTENTION KVQ 
        if True: 
            self.lkey = nn.Sequential(nn.Linear(n_embed,n_embed),
                                       nn.GELU(),
                                       nn.Linear(n_embed, head_size, bias=False))
            self.lquery = nn.Sequential(   nn.Linear(n_embed,n_embed),
                                            nn.GELU(),
                                            nn.Linear(n_embed, head_size, bias=False),)
            self.lvalue = nn.Sequential(nn.Linear(n_embed,n_embed),
                                        nn.GELU(),
                                        nn.Linear(n_embed, head_size, bias=False),)
        self.register_buffer('tril', torch.tril(torch.ones(context_size,context_size)))
        self.dropout = nn.Dropout(dropout)
    
    def forward(self,x):
        B,T,C = x.shape
        k, v, q = self.lkey(x), self.lvalue(x), self.lquery(x) # B , T 
        out = q @ k.transpose(-2,-1) * head_size **-0.5 # B T head_size x B head_size T = B T T
        out = out.masked_fill(self.tril[:T,:T] == 0, float('-inf'))
        out = F.softmax(out, dim=-1)
        out = self.dropout(out)
        out = out @ v # B,T,T  x B T head_size = B T head_size
        return out
        
class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = nn.ModuleList( [Head() for _ in range(num_heads)] )
        self.proj = nn.Linear(n_embed,n_embed)
        self.dropout = nn.Dropout(dropout)
    def forward(self,x):
        out = torch.cat([ h(x) for h in self.heads ] , dim=-1)
        out = self.dropout(self.proj(out)) 
        return out

class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.GELU(),
            nn.Linear( 4 * n_embed , n_embed ),
            nn.Dropout(dropout),
        )

    def forward(self,x):
        return self.network(x)

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.mha = MultiHeadAttention()
        self.ff = FeedForward()
        self.ln = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)
    def forward(self,x):
        x = x + self.mha( self.ln(x) ) # <------ RESIDUAL CONNECTION x+
        x = x + self.ff( self.ln2(x) ) # <------ RESIDUAL CONNECTION x+
        return x


# -------- EXPERIMENT
class LineExperiment(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(n_hidden_ex, n_hidden_ex)
        self.ac = nn.GELU()

    def forward(self,x):
        out = self.lin(x)
        out = self.ac(out)
        return out

class BlockExperiment(nn.Module):
    def __init__(self):
        super().__init__()
        self.ffwd = nn.Sequential( *[LineExperiment() for _ in range(ffwd_width)] )
        self.proj = nn.Linear(n_hidden_ex * 2, n_hidden_ex)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x):
        out = self.ffwd(x)
        out = torch.cat( [x, out], dim = -1 )
        out = self.proj(out)
        out = self.dropout(out)
        return out

class ModelExperiment(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding_layer = nn.Embedding(vocab_size,n_embed_ex) #CONVERTS INTEGERS INTO N-DIMENSIONAL VECTORS (RELATIVE IDX WILL BE CLOSER)
        self.expand_proj = nn.Linear(context_size*n_embed_ex, n_hidden_ex, bias=False)
        self.blocks = nn.Sequential(*[BlockExperiment() for _ in range(blocks_ex)])
        self.proj = nn.Linear(n_hidden_ex, vocab_size)

    def forward(self , q, a=None):
        B,T = q.shape # [batch, context_size]
        padding_size = max(0, context_size-T)  # Calculate the padding size
        q = F.pad(q, (0, padding_size), value = float(integerate(' ')[0]) )
        out = self.embedding_layer(q)
        out = out.view(B, context_size * n_embed_ex)
        out = self.expand_proj(out) #[batch_size, n_hidden_ex] 
        out = self.blocks(out) # [batch_size, n_hidden_ex]
        out = self.proj(out) # [batch_size, vocab_size] 
        # FIX CODE BELOW BECAUSE DIMENSIONS NOW DIFFER , NO SEPERATE T
        if a is None:
            return out
        else:
            # B , T , C = out.shape
            # out = out.view(B*T,C) # LOGITS FOR CROSSENTROPY HAVE TO NOT HAVE BATCH DIM (SECOND DIM MUST BE C)
            a = a[:,-1] # BECAUSE MY MODEL PREDICT WITHOUT TIME DIM - ONE WORD
            loss = F.cross_entropy(out, a)
            perplexity = torch.exp(torch.mean(loss))
            return out, loss , perplexity
        pass

    def generate(self,  q:torch.tensor, n_generate_chars:int):
        s = q   #<-- s = FULL LENGTH RESPONSE q = SLIDING 
        for _ in range(n_generate_chars):
            q = q[:, -context_size:]  #<- SLIDING WINDOW CONTEXT
            logits = self.forward(q) 
            # CHOOSE LAST TIME DIM... B,T,C -> B,C
            # logits = logits[:,-1,:] 
            # APPLY TEMPERATURE SCALING
            scaled_logits = logits / temperature
            # PROBABILITY -> CHOICE B,C -> B,1
            probs = F.softmax(scaled_logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1) 
            #APPEND TO LAST PREDICTION
            s = torch.cat( (s, next_idx), dim=-1)#<- FULL GENERATED RESPONSE
            q = torch.cat((q, next_idx ), dim = -1) #(B, T+1)#<- SLIDING WINDOW CONTEXT
        return s  #<- TENSOR , LIST OF ALL GENERATED INDICES

#---- PYTORCH TRANSFORMER 

class TransformerDecoderOnly(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward):
        super(TransformerDecoderOnly, self).__init__()

        # Create positional/word embeddings
        self.embedding_layer = nn.Embedding(vocab_size, n_embed)
        self.learned_positional_embedding = nn.Embedding(context_size,n_embed) # context size because we embed positions from 0 to context size.
        self.positional_embedding = batch_pos_emb(batch_size=batch_size, max_sen_len=context_size, w_emb_d=n_embed)

        # Create a decoder layer
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward)

        # Create the Transformer decoder using the specified number of layers
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        #Create last linear layer
        self.last_linear = nn.Linear(n_embed, vocab_size) #<- CHOOSE WORD FROM VOCABULARY

    def forward(self, tgt:torch.tensor, a=None):
        B,T = tgt.shape
        # EMBEDDING
        tok_emb = self.embedding_layer(tgt) 
        pos_emb = self.learned_positional_embedding(torch.arange(T,device=device)) 
        embedded_tgt = tok_emb + pos_emb #<- [BATCH, TIME, CHANNEL]
        embedded_tgt = embedded_tgt.transpose(0,1) #<- [TIME, BATCH, CHANNEL]
        # MASK
        memory = torch.zeros(embedded_tgt.size(0), embedded_tgt.size(1), embedded_tgt.size(2), device=device)
        tgt_mask = self.generate_tgt_mask(embedded_tgt.size(0))
        
        output = self.transformer_decoder(embedded_tgt,memory=memory, tgt_mask=tgt_mask )
        output = self.last_linear(output)

        if a is None:
            return output
        else:
            T,B,C = output.shape
            output = output.view(B*T,C) # LOGITS FOR CROSSENTROPY HAVE TO NOT HAVE BATCH DIM (SECOND DIM MUST BE C)
            a = a.view(B*T) # TARGETS DONT HAVE A CHANNEL DIM (NOT EMBEDED)
            loss = F.cross_entropy(output, a)
            return output, loss

    def generate_tgt_mask(self, sz):
        # Generate a square subsequent mask
        mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def generate(self, x:torch.tensor, n_generate_chars:int):
        
        s=x #<- B,T ONE BATCH ONLY

        for _ in range(n_generate_chars):
            
            #clip context size (if i want to generate MORE than context size)
            x = x[ : , -context_size: ]
            # pass trough model + APPLY TEMPERATURE SCALING
            logits = self.forward(x) #<- T , B , C  
            logits = logits[-1,:,:] # <- B, C
            scaled_logits = logits / temperature
            # convert to probabilities + choose
            probs = F.softmax(scaled_logits,dim=-1)
            chosen_idx = torch.multinomial(probs, num_samples=1) #<- B, 1
            
            x = torch.cat( (x,chosen_idx), dim=-1) #<- SLIDING CONTEXT WINDOW
            s = torch.cat( (s,chosen_idx), dim=-1) #<- FULL GENERATED TEXT SAVE
        return s
       
#---- PYTORCH TRANSFORMER 


# ===== FN [START]

def stringify(list_int):
    if vocab_type=='char': # OLD VOCAB SOLUTION 
        return ''.join([ itos[el]  for el in list_int ])
    elif vocab_type=='gtp':
        return enc.decode(list_int)

def integerate(str):
    if vocab_type=='char': # OLD VOCAB SOLUTION 
        return [ stoi[el] for el in str]
    if vocab_type=='gtp':
        return enc.encode(str)
    
def get_batch(split , del_from_list=True , fine_tunning = False): #<-- INCREMENTS c_epoch
    global nlt,nlv,c_epoch, nlt_max,lr
    if fine_tunning==False:
        if split == 'train':
            data=train_data
            if len(nlt) <= batch_size: #<------- IF EVERY DATAROW SEEN, EPOCH HAS PASSED
                nlt = list(range( random.randint(0, rand_offset), len(data)-context_size-1))[::skip_every]
                nlt_max=len(nlt)
                random.shuffle(nlt)
                c_epoch=c_epoch+1
                if False: #<-- FLIP LEARNING RATE - CYCLIC LEARNING RATE CHANGE
                    new_lr=lr_big if optimizer.param_groups[0]['lr']==lr_small else lr_small #FLIP LR
                    if c_epoch>warmup_epochs:
                        lr=lr_small
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr
                        print("[Learning rate update]:", optimizer.param_groups[0]['lr'])      
            selected_idx = [ nlt.pop() for _ in range(batch_size) ] if del_from_list==True else [ nlt[ random.randint(0, len(nlt)-1) ] for _ in range(batch_size) ]
        else:
            data=val_data
            if len(nlv) <= batch_size:
                nlv=list(range(0,len(data)-context_size-1))
                random.shuffle(nlv)
            selected_idx = [ nlv.pop() for _ in range(batch_size) ] if del_from_list==True else random.sample(nlv, batch_size)
        q = torch.stack([ data[el:el+context_size] for el in selected_idx ])  # [2,4,6], [7,8,7] 
        a =  torch.stack([ data[el+1:el+context_size+1] for el in selected_idx])  # L SHIFTED [4,6,9], [8,7,11]
        return q , a # returns tensor shape [ batch_size , context_size ]
    else:
        return q,a #<- batch of q batch of correlating a

@torch.no_grad()
def avg_loss_perplexity( n_avg):
    out={}
    out_p={}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(n_avg)
        perplexityes = torch.zeros(n_avg)
        for k in range(n_avg):
            q , a = get_batch(split, del_from_list=False)
            pred, loss, perplexity = model(q,a)
            losses[k] = loss.item()
            perplexityes[k] = perplexity 
        out[split] = losses.mean()
        out_p[split] = perplexityes.mean()
    model.train()
    return out , out_p

@torch.no_grad()
def infer(g_str, n_generate_chars):
    model.eval()
    res = model.generate( torch.tensor([ integerate(g_str) ],dtype=torch.long, device=device), n_generate_chars=n_generate_chars)
    for el in res:
        print( f'[O_O]:{stringify( el.tolist())}' )
    model.train()

def merge_datasets():
    max_line_length = 300
    whitelist = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ .?!():[]<>"
    joined = []
    print('reading...')
    skip = True
    with open( 'datasets/Emotion_classify_Data.csv', 'r', newline='') as f:
        csv_reader = csv.reader(f)
        for row in csv_reader:
            if skip==True:
                skip=False
            else:
                column_one = row[0]
                column_two = row[1]
                result = ''.join([ch for ch in (str(column_one) + ' ::: ' + str(column_two)) if ch in whitelist ])[-max_line_length:]
                if len(result) > 1:
                    joined.append( result )
    print(f'read example:{joined[-1]} \n')
    with open( 'datasets/twitter_training.csv', 'r', newline='', encoding='utf-8') as f:
        csv_reader = csv.reader(f)
        for row in csv_reader:
            column_one = row[3]
            column_two = row[2]
            result = ''.join([ch for ch in (str(column_one) + ' ::: ' + str(column_two)) if ch in whitelist ])[-max_line_length:]
            if len(result) > 1:
                joined.append( result )
    print(f'read example:{joined[-1]} \n')
    skip = True
    with open( 'datasets/movie.csv', 'r', newline='', encoding='utf-8') as f:
        csv_reader = csv.reader(f)
        for row in csv_reader:
            if skip==True:
                skip=False
            else:
                column_one = row[0]
                column_two = 'Positive' if row[1]==1 else 'Negative'
                result = ''.join([ch for ch in (str(column_one) + ' ::: ' + str(column_two)) if ch in whitelist ])[-max_line_length:]
                if len(result) > 1:
                    joined.append( result )
    print(f'read example:{joined[-1]} \n')
    skip = True
    with open( 'datasets/EcoPreprocessed.csv', 'r', newline='', encoding='utf-8') as f:
        csv_reader = csv.reader(f)
        for row in csv_reader:
            if skip==True:
                skip=False
            else:
                column_one = row[1]
                column_two = row[3]
                result = ''.join([ch for ch in (str(column_one) + ' ::: ' + str(column_two)) if ch in whitelist ])[-max_line_length:]
                if len(result) > 1:
                    joined.append( result )
    print(f'read example:{joined[-1]} \n')
    random.shuffle(joined)
    print('writing...')
    with open('datasets/new_merged.csv', 'w', newline='') as nf:
        csv_writer = csv.writer(nf, quoting=csv.QUOTE_NONE)
        for j in joined:
            csv_writer.writerow([j])
    print('done')

def save_plt( plt_name, step, value , plt_limit = 10):
    global plot_dict
    if plt_name in plot_dict:
        plot_lists = plot_dict[plt_name]
        plot_lists[0].append(step)
        plot_lists[1].append(value)
    else:
        plot_dict[plt_name] = [[step],[value]]
    os.makedirs("plt", exist_ok=True)  # create folder if it doesn’t exist
    plt.plot( plot_dict[plt_name][0][-plt_limit:] , plot_dict[plt_name][1][-plt_limit:] )
    plt.grid(True)
    plt.savefig(f'plt/{plt_name}_plot_epoch_{c_epoch}step_{steps}')
    plt.clf()

def measure_nlt_time():
    global nlt_first_second, time_list 
    res=0
    nlt_first_second.append( len(nlt) )
    time_list.append(time.time())
    if len(nlt_first_second) >= 2 :
        nlt_first_second = nlt_first_second[-2:] 
        time_list = time_list[-2:]
        time_diff = time_list[1] - time_list[0] 
        nlt_diff = nlt_first_second[0] - nlt_first_second[1] # nlt gets smaller time gets bigger so we flip
        if nlt_diff !=0:
            s_per_nlt = time_diff/nlt_diff
        else:
            s_per_nlt=1
        res = 1/s_per_nlt
    return res
    
def whitelist_filter(g_str, whitelist):
    return ''.join([ el for el in g_str if el in whitelist])

def print_gradients(grad):
    print(grad)

# ------------- TOKENIZERS -------------------------[START]

if vocab_type=='char': # OLD VOCAB stoi,itos,chars list 
    #chars = sorted(list(set(alltext))) # SET : UNIQUE ELEMENTS, MUTABLE, UNORDERED
    chars = [' ', '!', '&', '(', ')', ',', '.', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '|', '‘', '’', '“', '”']
    vocab_size = len(chars) 
    itos = { ch:idx for ch,idx in enumerate(chars)}
    stoi = { idx:ch for ch,idx in enumerate(chars)}
if vocab_type=='gtp':
    enc = tiktoken.get_encoding('gpt2')
    vocab_size = enc.n_vocab
if vocab_type=='intel':
    #tokenizer = LlamaTokenizer.from_pretrained('Intel/neural-chat-7b-v3-1')
    tokenizer = transformers.AutoTokenizer.from_pretrained('Intel/neural-chat-7b-v3-1')

# ------------- TOKENIZERS -------------------------[END]

SAVE_MODEL_DIR = os.getcwd() + '/Re'
torch.cuda.empty_cache()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print( f' CUDA VER:{torch.version.cuda} py ver: {sys.version} pytorch ver: {torch.__version__} device:{device}')
alltext = ''
n = None # SPLIT 90% OF TEXT INTO TRAINING SET/VALIDATION SET
c_epoch=0
nlt_max=0
dataset_load_stime_start=0
nlt=[]
nlv=[]
train_data = None
val_data = None
temperature = 0.1 # 0.1 (GREEDY) -> 2.0 (RANDOM)
context_size = 256 
batch_size = 4
n_embed = 10*64
num_heads = 10
head_size = n_embed//num_heads
t_blocks = 8
dropout = 0.1 
feed_forward_expand_factor = 4 
n_epoch = 1 #<-- PRE model has to be +1 BECAUSE GETBATCH INCREMENTS C_EPOCH AUTOMATICALLY
warmup_epochs = 8
rand_offset = 0

# MY TRANSFORMER
if True:
    model = Model().to(device)
#-------- EXPERIMENT
if False:
    #vocab_size=vocab_size # FROM ORIGINAL
    #context_size = context_size #FROM ORIGINAL
    n_embed_ex = 64
    n_hidden_ex = n_embed_ex * int(context_size * 1.4) 
    ffwd_width = 2
    blocks_ex = 6
    #dropout=dropout #FROM ORIGINAL
    #batch_size=batch_size #FROM ORIGINAL
# -- PYTORCH TRANSFORMER MODEL
if False: 
    model = TransformerDecoderOnly(d_model=n_embed, nhead=num_heads, num_layers=t_blocks, dim_feedforward=4*n_embed).to(device)
# -- INTEL MODEL
if False:  
    model_name = 'Intel/neural-chat-7b-v3-1'
    model = transformers.AutoModelForCausalLM.from_pretrained(model_name)
    #model = LlamaForCausalLM.from_pretrained(model_name)
if False: # <--- MY EXPERIMENTAL MODEL 
    model = ModelExperiment().to(device)

warmup_lr = 3e-4  #<-- USE 10x SMALLER FOR FINE TUNING
weight_decay = 1e-4
optimizer = torch.optim.AdamW(model.parameters(), lr=warmup_lr*0.1 , weight_decay=weight_decay) #STARTING WITH SMALL BECAUSE EPOCH FLIPS LR (EVEN 0'th)
max_grad_norm = 5.0  
skip_every = context_size # SINCE EVERY TOKEN IS A STARTING POINT FOR SLICING DATA TO MAKE THE INPUT BACH ELEMENT, SOME WORDS OR TOKENS IN DATASET WILL REPEAT ALMOST context_size TIMES. 
generation_starter = torch.tensor([integerate(' ')] , dtype=torch.long, device=device)
# LR SCHEDULER 
if False: 
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, verbose=True)
feed_forward_layers_manual = 0
training_name = 'ReText_BOOKS2' 
SAVE_MODEL_DIR = os.getcwd() + f'/{training_name}_fl-{feed_forward_layers_manual}_bs-{batch_size}_cs-{context_size}_ne-{n_embed}_nh-{num_heads}_tb-{t_blocks}_dr-{dropout}_fe-{feed_forward_expand_factor}_vs-{vocab_size}_wlr-{warmup_lr}'
continue_training = True
CONTINUE_TRAINING_FROM_DIR = os.getcwd() +'/ReText_BOOKS2_fl-0_bs-4_cs-256_ne-640_nh-10_tb-8_dr-0.1_fe-4_vs-50257_wlr-0.0003'
plt_x,plt_y = [],[]
fig,ax = plt.subplots()
plot_dict={}  
glob_min_loss=1000000.0
nlt_first_second=[]
time_list=[]
timer_start=None
timer_stop=None
nlt_max=0
model_fine_tuning=False
FT_q_batch=[]
FT_a_batch=[]
steps_total=0
ups = f'''{training_name}_fl-{feed_forward_layers_manual 
}_PT_bs-{batch_size}_cs-{context_size
}_ne-{n_embed}_nh-{num_heads
}_tb-{t_blocks}_dr-{dropout
}_fe-{feed_forward_expand_factor}_vs-{vocab_size}_wlr-{warmup_lr}''' #< --- TEXT FILE UPLOAD NAME
print(f' Upload file name: {ups}')
# CONVERT MODEL TO CPU AND SAVE
if False: 
    CONTINUE_TRAINING_FROM_DIR = os.getcwd() +'/ReText_PT_fl-0_bs-4_cs-256_ne-504_nh-12_tb-12_dr-0.1_fe-4_vs-50257_wlr-0.0003'
    model.load_state_dict(torch.load(CONTINUE_TRAINING_FROM_DIR))
    model.to('cpu')
    #print( f'[O_O] \n:{stringify(model.generate(generation_starter,n_generate_chars=context_size)[0].tolist())}\n')
    torch.save( deepcopy(model.state_dict()) , CONTINUE_TRAINING_FROM_DIR + '_CPU'  )
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

print( f'[O_O] \n:{stringify(model.generate(generation_starter,n_generate_chars=context_size)[0].tolist())}\n')
model.train()

# TRAINING LOOP  
if False:     
    # ---------------------------- DATASET LOAD [START]
    if True: #<---- HARRY
        with open('datasets/Harry_Potter_all_char_separated.txt','r', encoding='utf-8') as f:
            train=f.read()
            train=train.split('|')
    if False:# <- Q A + FOLLOW UP FINE TUNING DATASET
        dataset = load_dataset("stingning/ultrachat")
        train = dataset['train']
    if False: # <- GENERAL PRETRAINING DATASET
        if False:
            dataset = load_dataset("Skylion007/openwebtext")
            train = dataset['train']
        if True:
            train = load_dataset("Skylion007/openwebtext", split='train', num_proc=4 ) #cache_dir = os.path.join(os.getcwd(), 'dataset_cache')
    if False: # <- GENERAL PRETRAINING DATASET (BOOK)
        #train = load_dataset("bookcorpus", split='train', num_proc=4)
        #train = load_dataset("bookcorpusopen", split="train")  # downloads automatically
        #train = load_dataset("Skylion007/openwebtext", split="train")
        train = load_dataset("PleIAs/common_corpus", split="train")# <---100GB
    if False:# < -  Q A DATASET FINE TUNING  LINK -> https://huggingface.co/datasets/HuggingFaceH4/no_robots
        dataset = load_dataset("HuggingFaceH4/no_robots")
        train = dataset['train_sft']
    # ---------------------------- DATASET LOAD [END]
    algo_start_time = time.time()
    for n_train_epoch in range(100):
        print(f' --- > [NEW ENTIRE DATASET FINAL {n_train_epoch}]')
        c=0 # <- CURRENT DATASET BATCH COUNTER (TOTAL NUMBER OF DOCUMENTS)
        e=0 # <- CURRENT DATASET BATCH COUNTER ( NUMBER OF TEXT ACCUMULATED BATCHES)
        for el in train: 
            if True: #< -- ACCUMULATE
                if model_fine_tuning==True: #<-- FILTER & ACCUMULATE ALLTEXT
                    c=c+1
                    if len(el['messages'])>2: #<- ONLY ONE Q A PAIR FOR NOW
                        #continue #<-- COMMENT FOR HARRY POTTER DS 
                        pass 
                    if False:
                        print(f"\n\n PROMPT {el['messages'][0]['content']}" )
                        print( f" RESP {el['messages'][1]['content']}" )
                    FT_q = integerate( el['messages'][0]['content'] )
                    FT_a = integerate( el['messages'][1]['content'] )
                    if len(FT_q) > context_size or len(FT_a) > context_size:
                        #continue #<-- COMMENT FOR HARRY POTTER DS 
                        pass
                    FT_q = FT_q + [220] * (context_size - len(FT_q))# <- PAD ' ' 
                    FT_a = FT_a + [220] * (context_size - len(FT_a))# <- PAD
                    FT_q_batch.append( torch.tensor(FT_q, device=device)  ) 
                    FT_a_batch.append( torch.tensor(FT_a, device=device)  ) 
                    if len(FT_q_batch)<batch_size * 16: #<--- HOW MANY MODEL BATCHES I WANT TO LOAD IN DATASET BATCH
                        #continue  #<-- COMMENT FOR HARRY POTTER DS 
                        pass
                    len_accumulated = len(FT_q_batch) * context_size
                    print(len(FT_q_batch))
                else:
                    c=c+1 #<-- INDEX OF EL FROM TRAIN
                    if True:
                        alltext = alltext + el + '\n'
                    if False:
                        alltext = alltext + el['text'] + '\n'
                    if False:
                        alltext = alltext + whitelist_filter( el['text'] , chars ) + ' '
                    if len(alltext) < context_size * batch_size * 35 * 5 : # <------ ACCUMULATE DATA
                        continue   
            e=e+1 #<-- NUMBER OF ACUMULATED DATASET BATCHES
            c_epoch = 0
            n =  int(0.90 * len(alltext)) # SPLIT 90% OF TEXT INTO TRAINING SET/VALIDATION SET
            nlt,nlv=[],[]#<- NUMBER LIST TRAINING / VAL 
            train_data = torch.tensor( integerate(alltext[:n]) , device=device ) # device = device
            val_data = torch.tensor( integerate(alltext[n:]), device=device )
            if len(val_data) <= context_size:
                print(f'----------------------> WARNING. VALIDATION DATA LENGTH SMALLER THAN CONTEXT SIZE !')
                sys.exit()
            print(f' train data len:{len(train_data)} val data len: {len(val_data)}')
            alltext = '' # <- RELEASE ALLTEXT MEMORY 
            print(f' \n [DATASET BATCH LOADED {e}] {c}/{len(train)} t_data_length:{len(train_data)} v_data_length:{len(val_data)} ds_load_time:{ time.time()-dataset_load_stime_start}')
            start_time = time.time()
            for steps in range(99999):
                steps_total=steps_total+1
                if c_epoch>n_epoch: #<-- BREAK OUT OF steps LOOP
                    end_time=time.time()
                    t_dif=end_time-start_time
                    if model_fine_tuning==True: # <-- PRINT DATASET COMPLETED TIME STRING
                        print(f' dataset completed in :{t_dif} s | dataset token length:{ len_accumulated } | tok/s = { len_accumulated/t_dif }')
                    else:
                        print(f'dataset completed in :{t_dif} s | dataset token length:{len(train_data)} | tok/s = { len(train_data)/t_dif }')
                    dataset_load_stime_start=time.time()
                    break # <-- BREAK OUT OF steps LOOP
                if e % 6==0 and steps == 0: #<--- MODEL SAVE, WRITE MODEL PREDICTION TO FILE
                    print(f'[WRITING TO] -> MODEL_OUTPUT_{ups}.txt')
                    with open(f'MODEL_OUTPUT_{ups}.txt', 'a', encoding='utf-8') as f:
                        if model_fine_tuning==True: #<-- PRINT MODEL PREDICTION TO FILE
                            question = 'What sound does a cow make ?'
                            print( f'[O_O] e:{e} \n:{stringify(model.generate( torch.tensor([integerate(question)],device=device),n_generate_chars=context_size)[0].tolist())}\n', file=f)
                        else:
                            with torch.no_grad():
                                model.eval()
                                print( f'[O_O] time:{time.time()-algo_start_time:.1f}s e:{e} \n:{stringify(model.generate(generation_starter,n_generate_chars=context_size)[0].tolist())}\n', file=f)
                                model.train()
                        torch.save( deepcopy(model.state_dict()) , SAVE_MODEL_DIR  )
                if True and e % 30 == 0 and steps == 0: #<---- LEARNING RATE CHANGE (WARMUP)
                    if e <= 300: #<- THIS NUMBER HAS TO BE INITAL e_step x 10
                        new_lr =  warmup_lr  * (e / 300 ) #<- THIS NUMBER HAS TO BE INITAL e_step x 10
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = new_lr
                        print("[Learning rate update]:", optimizer.param_groups[0]['lr']) 
                if steps % 30==0 and steps!=0: # <----- NEW NLTPS 
                    nltps = measure_nlt_time()
                    print(f'epoch:{c_epoch} step:{steps} nlt:{len(nlt)} / {nlt_max} --> { round( (len(nlt)/nlt_max) * 100 ,2)  }% left to go... nltPS:{nltps}')
                if True and e % 5==0 and steps == 0: #< ---- PRINT LOSS + PLOT
                    if True:
                        losses, perplexities = avg_loss_perplexity(50)
                        print( f' c_epoch:{c_epoch} step:{steps} train loss:{losses["train"]:.4f}, val loss:{losses["val"]:.4f} global min loss:{glob_min_loss}')
                        print(f' perplexity train:{ perplexities["train"]} perplexity validation:{perplexities["val"]}')
                    print(f'[system uptime]:{ time.time() - algo_start_time  }s')
                    if True:
                        save_plt('LOSS', steps_total, losses['val']) #<-PLT WITH PICTURES
                        save_plt('PER', steps_total, perplexities['val'])
                if False: #<---- AUTOGRAD DIAGNOSTICS
                    with torch.autograd.profiler.profile(enabled=True, use_cuda=True) as prof:
                        print(f' autograd.profiler output: {prof} \n')
                if model_fine_tuning==True: #<-- GET q a BATCH 
                    FT_q_batch_stack = torch.stack( FT_q_batch[:batch_size] ) #<-- LOAD PREPADED TENSOR BATCH INTO CUDA
                    FT_a_batch_stack = torch.stack( FT_a_batch[:batch_size] ) #<-- LOAD PREPADED TENSOR BATCH INTO CUDA
                    q , a = FT_q_batch_stack, FT_a_batch_stack
                    FT_q_batch, FT_a_batch = FT_q_batch[batch_size:], FT_a_batch[batch_size:]#<--- DELETE SELECTED
                    if len(FT_q_batch) < batch_size:
                        c_epoch=c_epoch+1 
                else:
                    q , a = get_batch('train')#<------- GET BATCH TIME: 0.0009
                pred, loss , _ = model(q , a)#<--- 0.160s
                if True:
                    clip_grad_norm_(model.parameters(), max_grad_norm)#<---- 0.009s
                #optimizer.zero_grad(set_to_none=True)#<--- 0.002s
                for param in model.parameters():
                    param.grad = None
                loss.backward()# <---- 0.210s
                if False: #<-- SAVE MIN MAX MEDIAN GRADIENT TO FILE
                    if steps%1500==0 and steps !=0:
                        # Access and analyze gradients
                        all_gradients = []
                        for param in model.parameters():
                            if param.grad is not None:
                                all_gradients.extend(param.grad.data.cpu().numpy().flatten())

                        # Compute max, min, and median gradients
                        max_gradient = torch.max(torch.tensor(all_gradients))
                        min_gradient = torch.min(torch.tensor(all_gradients))
                        median_gradient = torch.median(torch.tensor(all_gradients))
                        with open(f'plt/GRADIENT_OUTPUT_{ups}.txt', 'a', encoding='utf-8') as f:
                            print(f"Step {steps} - Max Gradient: {max_gradient.item()}, Min Gradient: {min_gradient.item()}, Median Gradient: {median_gradient.item()}", file=f) 
                        pass
                optimizer.step() #<---- 0.018s
                #<------------------- FORWARD BACKWARD PASS TIME: 0.32s
            if model_fine_tuning==True: #<-- RESET VARIABLES
                FT_q_batch=[]
                FT_a_batch=[]
                nlt_first_second=[]
                time_list=[]
            else:
                alltext = '' #<-- RESET 
                nlt=[]
                nlv=[]
                nlt_first_second=[]
                time_list=[]

# INFERENCE
if True:
    infer(g_str='Magic', n_generate_chars=200)


















#https://towardsdatascience.com/testing-your-pytorch-models-with-torcheck-cb689ecbc08c
#conda install torchcheck
#torcheck.register(optimizer)
#torcheck.add_module_changing_check(model, module_name="my_model") #<- 1. Parameters change/not change















