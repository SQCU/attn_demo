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

# ---
init_from ='resume' 
#out_dir = 'out'
out_dir = os.path.join('logs','dyn_qkrmsnorm-3824bd2b-dd4b-4737-b074-57574f2cd8fc')
input_text = "FILE:prompt.txt" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 10 # number of samples to draw
max_new_tokens = 500 # number of tokens generated in each sample
temperature = 1.0 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 + triton wheels to compile the model
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
    ckpt_path = os.path.join(out_dir, 'state_step006250.pt')
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

model.eval()
model.to(device)
if compile:
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
def nlm_decode(model, idx, max_new_tokens, max_seq, temperature=1.0, top_k=None):
    
    for _ in range(max_new_tokens):
        # if the sequence context is growing too long we must crop it at max_seq
        idx_cond = idx if idx.size(1) <= max_seq else idx[:, -max_seq:]
        #forward requesting logits not loss:
        logits, _ = model(idx_cond, return_logits=True)
        # pluck the logits at the final step and scale by desired temperature
        logits = logits[:, -1, :] / temperature
        # optionally crop the logits to only the top k options
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
        # apply softmax to convert logits to (normalized) probabilities
        probs = nn.functional.softmax(logits, dim=-1)
        # sample from the distribution
        idx_next = torch.multinomial(probs, num_samples=1)
        # append sampled index to the running sequence and continue
        idx = torch.cat((idx, idx_next), dim=1)
    
    return idx

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

