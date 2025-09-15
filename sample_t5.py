# sample_t5.py
# uv run sample_t5.py
"""
Sample from a trained T5-style model
"""
import os
from contextlib import nullcontext
import torch
import torch.nn as nn
import pgptlformer

# ---
# --- CONFIGURATION ---
# The output directory from your T5 training run.
out_dir = 'logs/ascii-chart5-L4-D768-mkii-c1932d6c-1962-493d-b0b7-78e84e30e4e5' # <-- IMPORTANT: UPDATE THIS TO YOUR RUN'S DIRECTORY
checkpoint_name = 'state_step002000.pt'

# Prompt for the model. The script will format this for T5.
input_text = "Once upon a time,"

num_samples = 5
max_new_tokens = 500
temperature = 1.0 # Temperature is often lower for T5 generation
top_k = 200
seed = 1337
device = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
torch_compile = True # Compiling the new methods is a good idea
# --- END CONFIGURATION ---

# Boilerplate
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# --- NEW: T5-Aware ASCII Tokenizer ---
class T5ASCIITokenizer:
    """ A tokenizer that understands T5 special tokens. """
    def __init__(self, model_config):
        self.pad_token_id = model_config['pad_token_id']
        self.eos_token_id = model_config['eos_token_id']
        self.mask_start_id = model_config['mask_token_start_id']
        self.vocab_size = model_config['vocab_size']
        
        # Build vocabulary
        self.int_to_char = {i: chr(i) for i in range(128)} # Base ASCII
        self.int_to_char[self.pad_token_id] = "<pad>"
        self.int_to_char[self.eos_token_id] = "<eos>"
        # Add sentinel mask tokens
        for i in range(100): # Assuming we use up to 100 mask tokens
             self.int_to_char[self.mask_start_id + i] = f"<mask_{i}>"
        
        self.char_to_int = {v: k for k, v in self.int_to_char.items()}

    def encode(self, text):
        # Simple word-based splitting for special tokens
        import re
        parts = re.split(r'(<pad>|<eos>|<mask_\d+>)', text)
        tokens = []
        for part in parts:
            if part in self.char_to_int:
                tokens.append(self.char_to_int[part])
            else:
                tokens.extend([self.char_to_int.get(char, 0) for char in part])
        return tokens

    def decode(self, tokens):
        return "".join([self.int_to_char.get(token, f'[UNK:{token}]') for token in tokens])

# Load checkpoint and config
ckpt_path = os.path.join(out_dir, checkpoint_name)
checkpoint = torch.load(ckpt_path, map_location=device)
tformer_cfg = checkpoint['model_args']

assert tformer_cfg.get("is_t5", False), "This script is for T5 models only. Use sample_ascii.py for GPT models."

# Instantiate our new T5 tokenizer
enc = T5ASCIITokenizer(tformer_cfg)
encode = enc.encode
decode = enc.decode

# Load model
model = pgptlformer.PGPT_Lformer(tformer_cfg)
state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'
for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)

model.eval()
model.to(device)
if torch_compile:
    # Compile the new inference-specific methods
    model.encode = torch.compile(model.encode)
    model.decode_step = torch.compile(model.decode_step)

# --- NEW: T5 Generation Function ---
def t5_decode(model, encoder_input_ids, max_new_tokens, temperature=1.0, top_k=None):
    # Get special token IDs from the tokenizer
    pad_id = enc.pad_token_id
    eos_id = enc.eos_token_id
    
    # 1. ENCODE the prompt (happens only once)
    encoder_padding_mask = (encoder_input_ids != pad_id)
    encoder_hidden_states = model.encode(encoder_input_ids, encoder_padding_mask)
    
    # 2. Initialize the DECODER sequence
    # It starts with the pad_token_id, which acts as the "start" token for T5.
    decoder_input_ids = torch.tensor([[pad_id]], dtype=torch.long, device=device)

    # 3. DECODE token by token (autoregressive loop)
    for _ in range(max_new_tokens):
        logits = model.decode_step(decoder_input_ids, encoder_hidden_states, encoder_padding_mask)
        logits = logits.squeeze(1) / temperature # Shape: [B, Vocab] -> [1, Vocab]
        
        # Optional Top-K sampling
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
        
        probs = nn.functional.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        
        # Stop if the model generates the end-of-sequence token
        if idx_next.item() == eos_id:
            break
            
        # Append the sampled token to the decoder's input sequence
        decoder_input_ids = torch.cat((decoder_input_ids, idx_next), dim=1)
        
    # Return the generated sequence, excluding the initial start token
    return decoder_input_ids[:, 1:]

# --- Format the prompt for T5's denoising objective ---
# We ask the model to "fill in the blank" after our prompt.
prompt = f"{input_text} <mask_0>"
input_ids = encode(prompt)
encoder_input_ids = torch.tensor(input_ids, dtype=torch.long, device=device)[None, ...]

# Run generation
print(f"--- Sampling with T5 formatted prompt: ---\n{prompt}")
with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            print(f"\n--- SAMPLE {k+1} ---")
            y = t5_decode(model, encoder_input_ids, max_new_tokens, temperature=temperature, top_k=top_k)
            # The output is the generated text that "fills in" for <mask_0>
            generated_text = decode(y[0].tolist())
            print(generated_text)