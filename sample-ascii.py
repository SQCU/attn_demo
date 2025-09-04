#sample-ascii.py
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

# --- NEW: Self-contained ASCII Tokenizer ---
class SimpleASCIITokenizer:
    """ A simple tokenizer for the first 256 ASCII characters. """
    def __init__(self):
        # We use range(256) to match the vocab_size from your training run
        self.chars = [chr(i) for i in range(256)]
        self.vocab_size = len(self.chars)
        self.char_to_int = {ch: i for i, ch in enumerate(self.chars)}
        self.int_to_char = {i: ch for i, ch in enumerate(self.chars)}

    def encode(self, text):
        """Converts a string to a list of integer token IDs."""
        return [self.char_to_int.get(char, 0) for char in text]

    def decode(self, tokens):
        """Converts a list of integer token IDs back to a string."""
        return "".join([self.int_to_char.get(token, '') for token in tokens])

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
# We must ensure the loaded model's vocab size matches our tokenizer.
assert tformer_cfg['vocab_size'] == enc.vocab_size, \
    f"Vocab size mismatch! Model trained with {tformer_cfg['vocab_size']}, but tokenizer has {enc.vocab_size}."
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

# The sampling function `nlm_decode` remains COMPLETELY UNCHANGED as it's tokenizer-agnostic
def nlm_decode(model, idx, max_new_tokens, max_seq, temperature=1.0, top_k=None):
    for _ in range(max_new_tokens):
        idx_cond = idx if idx.size(1) <= max_seq else idx[:, -max_seq:]
        logits, _, _ = model(idx_cond, return_logits=True)
        logits = logits[:, -1, :] / temperature
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
        probs = nn.functional.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx

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