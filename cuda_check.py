import torch
import sys 

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print( f' CUDA VER:{torch.version.cuda} py ver: {sys.version} pytorch ver: {torch.__version__} device:{device}')