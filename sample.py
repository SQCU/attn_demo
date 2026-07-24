#sample.py
#ripped straight from the nanogpt thank you karpathy
"""
Sample from a trained model
"""
import os
import pickle
from contextlib import nullcontext
import torch
import torch.nn as nn
import tiktoken
#no gpt2s here haha
import pgptlformer
from sampler_utils import ar_sample

#wacky env stuff:
#import tritonpathsetter
#tritonpathsetter.set_cuda_paths()

import pdb;

# ---
init_from ='resume' 
#out_dir = 'out'
out_dir = os.path.join('logs','re-pqt-rmsXrmsx3-ATTNII_fast-af1c5037-28b9-4ce3-b351-c78046f90ee7')
input_text = "FILE:prompt.txt" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 10 # number of samples to draw
max_new_tokens = 500 # number of tokens generated in each sample
temperature = 1.0 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
torch_compile = True # use PyTorch 2.0 + triton wheels to compile the model
maximum_context = 1024 # more tokens than this will be cropped from the decoder model's context

#wizard spell to get this script's path
LOCAL_DIR = os.path.dirname(__file__)

#args = Hyperparameters()

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

#default model config:
bad_default_prefab = {"vocab_size":50304, "num_layers":4,"dim":256,"dim_head":32,"headcount":8,"ff_mult":4, 
"lambda":True,"layerwisenorm":"rmsnorm","qknorm":"identitynorm", "training_seqlen":512}

# model
if init_from == 'resume':
    # init from a model saved in a specific directory
    #ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    out_dir = os.path.join(LOCAL_DIR, out_dir) #compatibility
    ckpt_path = os.path.join(out_dir, 'state_step040500.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    if 'model_args' in checkpoint.keys():
        tformer_cfg = checkpoint['model_args']
    else:
        tformer_cfg = bad_default_prefab
    model = pgptlformer.PGPT_Lformer(tformer_cfg)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    total_params = sum(p.numel() for p in model.parameters())
    print("Total parameters:", total_params)

model.eval()
model.to(device)
if torch_compile:
    model = torch.compile(model) # requires PyTorch 2.0 (optional)

# look for the meta pickle in case it is available in the dataset folder
load_meta = False
#???
# skip.

# ok let's assume gpt-2 encodings by default
print("No meta.pkl found, assuming GPT-2 encodings...")
enc = tiktoken.get_encoding("gpt2")
encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
decode = lambda l: enc.decode(l)

#define sampler as external to model bc it really is yknow
#...and it now lives in exactly one place. this used to be a local copy doing
#`logits, _, _z = model(...)` -- a 3-unpack of a 4-tuple, i.e. ValueError on the first
#token, for as long as forward_arg has had loss_per_sequence. see sampler_utils.ar_sample.
nlm_decode = ar_sample

# encode the beginning of the prompt
# the really weird overloading of text as a container for pathstrings is from nanogpt not me i promise.
if input_text.startswith('FILE:'):
    with open(input_text[5:], 'r', encoding='utf-8') as f:
        input_text = f.read()
input_ids = encode(input_text)
x = (torch.tensor(input_ids, dtype=torch.long, device=device)[None, ...])

# run generation
with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            y = nlm_decode(model, x, max_new_tokens, max_seq = maximum_context, temperature=temperature, top_k=top_k)
            print(decode(y[0].tolist()))
            print('---------------')

