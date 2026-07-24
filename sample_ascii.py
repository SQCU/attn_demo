#sample_ascii.py
"""
Sample from a trained model
"""
import os
from contextlib import nullcontext
import torch
import torch.nn as nn
# --- MODIFIED: No more tiktoken! ---
# import tiktoken 
import pgptlformer
from ascii_tokenizer import SimpleASCIITokenizer
from sampler_utils import ar_sample

# ---
# --- CONFIGURATION ---
# The output directory from your training run.
out_dir = os.path.join('logs','ascii-eos-L4-D768-A-361c85ac-3c92-4014-9e28-6d8481c5d96d') # <-- IMPORTANT: UPDATE THIS TO YOUR RUN'S DIRECTORY
# The specific checkpoint to use. 'state_step004500.pt' if you ran for 4500 steps.
checkpoint_name = 'state_step004500.pt' 
# Prompt for the model. Can be a string or a file path.
# prepend with "<eos>" if you trained with <eos> delimiters i guess!
input_text = "<eos>Once upon a time," # or "FILE:prompt.txt"
num_samples = 5
max_new_tokens = 500
temperature = 1.0 # < 1.0 makes model less random, > 1.0 more random
top_k = 200 # Restrict sampling to the top k most likely tokens
seed = 1337
device = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
torch_compile = True
maximum_context = 1024
# --- END CONFIGURATION ---

#wizard spell to get this script's path
LOCAL_DIR = os.path.dirname(__file__) if __file__ else '.'

# Boilerplate
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# the ASCII tokenizer used to be re-declared right here with range(256), while
# ascii_tokenizer.py (which loader.py's rollout capture imports) declares range(128) and
# data/prepare_ascii.py *emits* range(128). two tokenizers, one name, different vocabularies,
# and the equality assert below hid it: configs declare vocab_size 256 because 2401.14489
# wants vocab%64==0, not because there are 256 symbols. ids 128..255 are GEMM padding the
# model can emit but the data never contains. one tokenizer now, and the assert checks what
# it actually needs to check.

# model loading
ckpt_path = os.path.join(LOCAL_DIR, out_dir, checkpoint_name)
checkpoint = torch.load(ckpt_path, map_location=device)

# --- THIS IS KEY: The script automatically loads the correct model config! ---
# Your choice to save 'model_args' in the checkpoint was excellent.
# It makes this sampling script general-purpose.
if 'model_args' not in checkpoint:
    raise ValueError("Checkpoint must contain 'model_args' to configure the model.")
tformer_cfg = checkpoint['model_args']

# --- MODIFIED: Instantiate our new tokenizer ---
# Instead of tiktoken, we use our own.
enc = SimpleASCIITokenizer()
# the model's vocab must COVER the tokenizer's, not equal it: the extra ids are alignment
# padding. an equality assert here is how the 128-vs-256 split went unnoticed.
assert tformer_cfg['vocab_size'] >= enc.vocab_size, \
    f"Vocab size mismatch! Model trained with {tformer_cfg['vocab_size']}, but tokenizer needs at least {enc.vocab_size}."
encode = enc.encode
decode = enc.decode

# Load model
model = pgptlformer.PGPT_Lformer(tformer_cfg)
state_dict = checkpoint['model']
# The prefix removal is necessary if the model was compiled or run with DDP
unwanted_prefix = '_orig_mod.'
for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)

model.eval()
model.to(device)
if torch_compile:
    model = torch.compile(model)

# The sampling function is tokenizer-agnostic, so it is shared rather than copied.
# (the local copy did `logits, _, _ = model(...)`: a 3-unpack of a 4-tuple. dead since
#  forward_arg grew loss_per_sequence.)
nlm_decode = ar_sample

# Handle prompt from file or string
if input_text.startswith('FILE:'):
    with open(input_text[5:], 'r', encoding='utf-8') as f:
        input_text = f.read()
input_ids = encode(input_text)
x = (torch.tensor(input_ids, dtype=torch.long, device=device)[None, ...])

# Run generation
print(f"--- Sampling with prompt: ---\n{input_text}")
with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            print(f"\n--- SAMPLE {k+1} ---")
            y = nlm_decode(model, x, max_new_tokens, max_seq = maximum_context, temperature=temperature, top_k=top_k)
            print(decode(y[0].tolist()))