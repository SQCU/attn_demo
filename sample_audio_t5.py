"""
# sample_audio_t5.py (v3 - Parallel Rollouts)
# Generates long-form audio in parallel for multiple samples.
#
# --- USAGE ---
# uv run sample_audio_t5.py --out_dir "logs/your_run_dir" --seed 420 --num_iterations 12
"""
import os
import sys
import subprocess # <-- ADD for running the decode script
from contextlib import nullcontext
import torch
import torch.nn as nn
import pgptlformer
import argparse
from tqdm import tqdm
from prompt_utils import AudioPromptGenerator
from datetime import datetime

# --- CONFIGURATION VIA ARGPARSE ---
parser = argparse.ArgumentParser(description="Sample long-form audio from a trained T5 model.")
parser.add_argument('--out_dir', type=str, required=True, help="The output directory from your audio T5 training run.")
parser.add_argument('--checkpoint_name', type=str, default='state_step002000.pt', help="Name of the checkpoint file.")
parser.add_argument('--num_samples', type=int, default=4, help="How many different audio clips to generate.")
parser.add_argument('--prompt_length', type=int, default=256, help="Length of the context window for each generation step.")
parser.add_argument('--tokens_per_step', type=int, default=256, help="How many new tokens to generate in each iteration.")
parser.add_argument('--num_iterations', type=int, default=12, help="Number of generation steps to perform for a long rollout.")
parser.add_argument('--temperature', type=float, default=1.0, help="Temperature for sampling.")
parser.add_argument('--top_k', type=int, default=250, help="Top-k for sampling.")
parser.add_argument('--seed', type=int, default=420420, help="Random seed.")
parser.add_argument('--device', type=str, default='cuda', help="Device to use ('cuda' or 'cpu').")
parser.add_argument('--compile', action='store_true', help="Enable torch.compile.")
parser.add_argument('--npz_file', type=str, default="data/measureformer/mproj_ii.npz")
parser.add_argument('--parquet_file', type=str, default="data/measureformer_analysis/2025-09-13_18-40-31_9a4c.parquet")

parser.add_argument('--autodecode', action='store_true', help="Automatically run the decoding script after generation.")
args = parser.parse_args()
# --- END CONFIGURATION ---

# Your correct boilerplate additions
run_id = datetime.now().strftime('%y%m%d%H%M%S')
print(f"Generated a unique ID for this run's output files: {run_id}")
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in args.device else 'cpu'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# Load checkpoint and config from the training run
print(f"Loading model from: {args.out_dir}")
ckpt_path = os.path.join(args.out_dir, args.checkpoint_name)
checkpoint = torch.load(ckpt_path, map_location=args.device)
model_config = checkpoint['model_args']

# Ensure this is a T5 model
assert model_config.get("is_t5", False), "This script is for T5 models only."

# Load model state (Your correct version)
model = pgptlformer.PGPT_Lformer(model_config)
state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'
for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)
model.eval().to(args.device)

if args.compile:
    print("Compiling model for inference...")
    model.encode = torch.compile(model.encode)
    model.decode_step = torch.compile(model.decode_step)

# --- Instantiate the Audio Prompt Generator ---
print("Initializing Audio Prompt Generator to get seed prompts...")
prompt_generator = AudioPromptGenerator(npz_path=args.npz_file, parquet_path=args.parquet_file)

# Get the initial batch of seed prompts
seed_prompts = prompt_generator.get_prompts(
    num_prompts=args.num_samples,
    prompt_length=args.prompt_length
).to(args.device)

# --- Fully Batched T5 Decode Function ---
def t5_decode_fully_batched(model, encoder_input_ids, max_new, temp, top_k):
    pad_id, eos_id = model_config['pad_token_id'], model_config['eos_token_id']
    batch_size = encoder_input_ids.shape[0]
    
    decoder_input_ids = torch.full((batch_size, 1), pad_id, dtype=torch.long, device=args.device)
    encoder_padding_mask = (encoder_input_ids != pad_id)
    encoder_hidden_states = model.encode(encoder_input_ids, encoder_padding_mask)
    
    # This mask tracks which sequences in the batch have finished
    has_finished = torch.zeros(batch_size, dtype=torch.bool, device=args.device)

    for _ in range(max_new):
        logits = model.decode_step(decoder_input_ids, encoder_hidden_states, encoder_padding_mask)
        logits = logits[:, -1, :] / temp
        
        if top_k is not None:
            k = min(top_k, logits.size(-1))
            v, _ = torch.topk(logits, k)
            threshold = v[:, -1].unsqueeze(-1)
            logits[logits < threshold] = -float('Inf')
            
        probs = nn.functional.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        
        # For sequences that have already finished, append padding instead of new tokens
        idx_next[has_finished] = pad_id
        # Update the finished mask for any sequences that just generated EOS
        has_finished |= (idx_next.squeeze() == eos_id)
        
        decoder_input_ids = torch.cat((decoder_input_ids, idx_next), dim=1)
        
        # If all sequences in the batch are done, we can exit early
        if has_finished.all():
            break
            
    return decoder_input_ids[:, 1:]

# --- Main PARALLEL Long-Form Generation Loop ---
print(f"\n--- Generating {args.num_samples} long-form samples in parallel over {args.num_iterations} iterations ---")
# Start with the entire batch of seed prompts
current_sequences = seed_prompts

with torch.no_grad():
    with ctx:
        # The single loop now processes all samples at once
        for _ in tqdm(range(args.num_iterations), desc="Rollout Progress"):
            # The context is the last `prompt_length` tokens from ALL sequences
            context = current_sequences[:, -args.prompt_length:]
            
            # Format for T5: append <mask_0> to each sequence in the batch
            mask_tokens = torch.full((args.num_samples, 1), model_config['mask_token_start_id'], dtype=torch.long, device=args.device)
            encoder_inputs = torch.cat([context, mask_tokens], dim=1)

            # Generate the next chunk of tokens for the ENTIRE batch
            new_tokens = t5_decode_fully_batched(
                model,
                encoder_inputs,
                args.tokens_per_step,
                args.temperature,
                args.top_k
            )

            # Append the new tokens to their respective sequences in the batch
            current_sequences = torch.cat([current_sequences, new_tokens], dim=1)

# The final result is the `current_sequences` tensor.
# Convert the batched tensor into a list of individual tensors for saving,
# which matches what the decoder script expects.
final_long_sequences = [seq for seq in current_sequences]

# --- SAVE THE RESULTS ---
output_filename = f"{run_id}_long_form_tokens_{os.path.basename(args.out_dir)}.pt"
print(f"\nSaving generated token sequences to: {output_filename}")

torch.save({
    'prompts': seed_prompts.cpu(),
    'generated': [seq.cpu() for seq in final_long_sequences],
    'model_config': model_config,
}, output_filename)

print("\n--- Generation Complete ---")
# --- NEW: AUTO-DECODING LOGIC ---
if args.autodecode:
    print(f"\n--- Auto-decoding enabled. Running decode_audio.py on '{output_filename}' ---")
    
    # Construct the command to run the decode script.
    # Using sys.executable is a robust way to ensure we use the same python environment.
    command = [
        sys.executable, 'decode_audio.py',
        '--input_file', output_filename
    ]
    
    try:
        # Run the command and let its output stream to the console.
        # check=True will raise an error if the decoding script fails.
        subprocess.run(command, check=True)
        print("\n--- Auto-decoding finished successfully. ---")
    except FileNotFoundError:
        print("\n[ERROR] Could not find 'decode_audio.py' in the current directory.")
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] The decoding script failed with exit code {e.returncode}.")
else:
    # The original message is now the 'else' case
    print(f"To listen, use the updated decoding script: 'uv run decode_audio.py --input_file {output_filename}'.")