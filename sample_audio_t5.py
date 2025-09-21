"""
# sample_audio_t5.py (v3 - Parallel Rollouts)
# Generates long-form audio in parallel for multiple samples.
#
# --- USAGE ---
# uv run sample_audio_t5.py --out_dir "logs/mformer-t5-l6-mk1-21d0cffb-48ce-4195-880a-e6c9ca5b1a0b" --seed 420 --num_iterations 12 --parquet_file data/measureformer_analysis/2025-09-17_04-08-55_fbe2.parquet --autodecode
# uv run sample_audio_t5.py --out_dir logs/mformer-t5-l6-mk2-ce62b19e-4718-4f15-bdee-af0ace4e7cc2 --checkpoint_name state_step002000.pt --seed 420 --num_iterations 12 --parquet_file data/measureformer_analysis/2025-09-17_04-08-55_fbe2.parquet --autodecode
# uv run sample_audio_t5.py --out_dir logs/mformer-t5-l6-mk2-ce62b19e-4718-4f15-bdee-af0ace4e7cc2 --checkpoint_name state_step010000.pt --seed 420 --num_iterations 12 --parquet_file data/measureformer_analysis/2025-09-17_04-08-55_fbe2.parquet --autodecode
# uv run sample_audio_t5.py --out_dir logs/mformer-t5-l6-mk2-ce62b19e-4718-4f15-bdee-af0ace4e7cc2 --checkpoint_name state_step010000.pt --seed 480 --num_iterations 12 --infill_prob 0.8 --parquet_file data/measureformer_analysis/2025-09-17_04-08-55_fbe2.parquet --autodecode --ood_audio_file "data/ood/Raft Ride (Alpha Mix) - The Legend of Zelda： Link's Awakening [gVFJlwr6INI].mp3"
# uv run sample_audio_t5.py --out_dir logs/mformer-t5-l6-mk2-ce62b19e-4718-4f15-bdee-af0ace4e7cc2 --checkpoint_name state_step010000.pt --seed 480 --num_iterations 12 --infill_prob 0.8 --parquet_file data/measureformer_analysis/2025-09-17_04-08-55_fbe2.parquet --autodecode --ood_audio_file "data/ood/Web Mystery (Dreamcast) - A Maze-MkWUe_bhNr4.mp3"
# uv run sample_audio_t5.py --out_dir logs/mformer-t5-l6-mk2-ce62b19e-4718-4f15-bdee-af0ace4e7cc2 --checkpoint_name state_step010000.pt --seed 480 --num_iterations 12 --infill_prob 0.8 --parquet_file data/measureformer_analysis/2025-09-17_04-08-55_fbe2.parquet --autodecode --ood_audio_file "data/ood/04 Glarthir - Poison (feat. Weebam-Na).mp3"
# uv run sample_audio_t5.py --out_dir logs/mformer-t5-l6-mk2-ce62b19e-4718-4f15-bdee-af0ace4e7cc2 --checkpoint_name state_step010000.pt --seed 480 --num_iterations 12 --infill_prob 0.8 --parquet_file data/measureformer_analysis/2025-09-17_04-08-55_fbe2.parquet --autodecode --ood_audio_file "data/ood/06. Never Stop The Fucking Rave.flac"
# uv run sample_audio_t5.py --out_dir logs/mformer-t5-l6-mk2-ce62b19e-4718-4f15-bdee-af0ace4e7cc2 --checkpoint_name state_step010000.pt --seed 420 --num_iterations 3 --parquet_file data/measureformer_analysis/2025-09-17_04-08-55_fbe2.parquet --autodecode --seed 480 --num_iterations 12 --infill_prob 0.8
# cpu debug run?
#  uv run sample_audio_t5.py --out_dir logs/mformer-t5-l6-mk2-ce62b19e-4718-4f15-bdee-af0ace4e7cc2 --checkpoint_name state_step010000.pt --seed 420 --num_iterations 3 --parquet_file data/measureformer_analysis/2025-09-17_04-08-55_fbe2.parquet --autodecode --seed 
480 --num_iterations 3 --infill_prob 0.8 --device cpu --num_samples 1
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
from prompt_utils import AudioPromptGenerator, OODAudioPromptGenerator
from datetime import datetime
import random

# --- CONFIGURATION VIA ARGPARSE ---
parser = argparse.ArgumentParser(description="Sample long-form audio from a trained T5 model.")
parser.add_argument('--out_dir', type=str, required=True, help="The output directory from your audio T5 training run.")
parser.add_argument('--checkpoint_name', type=str, default='state_step002000.pt', help="Name of the checkpoint file.")
parser.add_argument('--num_samples', type=int, default=4, help="How many different audio clips to generate.")
parser.add_argument('--prompt_length', type=int, default=128, help="Length of the context window for each generation step.")
parser.add_argument('--tokens_per_step', type=int, default=512, help="How many new tokens to generate in each iteration.")
parser.add_argument('--num_iterations', type=int, default=12, help="Number of generation steps to perform for a long rollout.")
parser.add_argument('--temperature', type=float, default=1.0, help="Temperature for sampling.")
parser.add_argument('--top_k', type=int, default=250, help="Top-k for sampling.")
parser.add_argument('--seed', type=int, default=420420, help="Random seed.")
parser.add_argument('--device', type=str, default='cuda', help="Device to use ('cuda' or 'cpu').")
parser.add_argument('--compile', action='store_true', help="Enable torch.compile.")
parser.add_argument('--npz_file', type=str, default="data/measureformer/mproj_ii.npz")
parser.add_argument('--parquet_file', type=str, default="data/measureformer_analysis/2025-09-13_18-40-31_9a4c.parquet")
parser.add_argument('--autodecode', action='store_true', help="Automatically run the decoding script after generation.")
parser.add_argument('--ood_audio_file', type=str, default=None, help="Path to an arbitrary audio file to use for generating prompts.")
parser.add_argument('--infill', action='store_true', help="Enable in-filling mode instead of continuation.")
parser.add_argument('--infill_length', type=int, default=512, help="Number of tokens to generate for the in-filled section.")
parser.add_argument('--infill_prob', type=float, default=0.0, help="Probability (0.0 to 1.0) of performing an in-fill step instead of continuation.")
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
#print(state_dict)
model.load_state_dict(state_dict)
model.eval().to(args.device)

if args.compile:
    print("Compiling model for inference...")
    model.encode = torch.compile(model.encode)
    model.decode_step = torch.compile(model.decode_step)

# --- Instantiate the Audio Prompt Generator ---
if args.ood_audio_file:
    if not os.path.exists(args.ood_audio_file):
        print(f"Error: OOD audio file not found at '{args.ood_audio_file}'")
        sys.exit(1)
    # The new on-the-fly generator for arbitrary audio
    prompt_generator = OODAudioPromptGenerator(
        ood_audio_path=args.ood_audio_file,
        device=args.device
    )
    # Give the output file a more descriptive name
    output_basename = os.path.splitext(os.path.basename(args.ood_audio_file))[0]
    output_filename_stem = f"{run_id}_ood_{output_basename}"
else:
    # The original generator that uses pre-computed training data files
    print("Initializing Audio Prompt Generator to get seed prompts from training data...")
    prompt_generator = AudioPromptGenerator(npz_path=args.npz_file, parquet_path=args.parquet_file)
    output_filename_stem = f"{run_id}_long_form_tokens_{os.path.basename(args.out_dir)}"

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

# --- NEW: T5 IN-FILLING DECODE FUNCTION ---
def t5_infill_batched(model, prefix_ids, postfix_ids, max_new, temp, top_k):
    """
    Performs batched in-filling using the T5 architecture.
    """
    pad_id = model_config['pad_token_id']
    eos_id = model_config['eos_token_id']
    mask_id = model_config['mask_token_start_id']
    batch_size = prefix_ids.shape[0]
    device = prefix_ids.device

    # 1. Construct the encoder input: [prefix, <mask_0>, postfix]
    mask_token = torch.full((batch_size, 1), mask_id, dtype=torch.long, device=device)
    encoder_input_ids = torch.cat([prefix_ids, mask_token, postfix_ids], dim=1)
    
    # The rest of the generation logic is identical to continuation
    encoder_padding_mask = (encoder_input_ids != pad_id)
    encoder_hidden_states = model.encode(encoder_input_ids, encoder_padding_mask)
    
    # The decoder starts by being prompted with what to fill in: <mask_0>
    decoder_input_ids = torch.full((batch_size, 1), mask_id, dtype=torch.long, device=device)
    
    has_finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

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
        
        idx_next[has_finished] = pad_id
        has_finished |= (idx_next.squeeze() == eos_id)
        
        decoder_input_ids = torch.cat((decoder_input_ids, idx_next), dim=1)
        
        if has_finished.all():
            break
            
    # 2. Post-process: remove the initial <mask_0> and any <eos>/padding
    output_sequences = []
    for i in range(batch_size):
        # Slice off the starting <mask_0> token
        seq = decoder_input_ids[i, 1:]
        # Find the first EOS token
        eos_idx = (seq == eos_id).nonzero(as_tuple=True)[0]
        if len(eos_idx) > 0:
            seq = seq[:eos_idx[0]]
        output_sequences.append(seq)
        
    return output_sequences

# --- NEW: High-Level Continuation Wrapper ---
def t5_continue_step(model, context, max_new, temp, top_k):
    """
    UNIFORM SIGNATURE: Takes a context and generates a continuation.
    """
    batch_size = context.shape[0]
    device = context.device
    mask_id = model_config['mask_token_start_id']
    
    # Construct encoder input: [context, <mask_0>]
    mask_tokens = torch.full((batch_size, 1), mask_id, dtype=torch.long, device=device)
    encoder_input_ids = torch.cat([context, mask_tokens], dim=1)
    
    return _t5_decoder_engine(model, encoder_input_ids, max_new, temp, top_k)

# --- NEW: High-Level In-filling Wrapper ---
def t5_infill_step(model, context, max_new, temp, top_k):
    """
    UNIFORM SIGNATURE: Takes a context, splits it, and generates an in-fill.
    """
    prefix_len = context.shape[1] // 2
    
    prefix_batch = context[:, :prefix_len]
    postfix_batch = context[:, -prefix_len:] # Use -prefix_len to be symmetrical
    
    batch_size = context.shape[0]
    device = context.device
    mask_id = model_config['mask_token_start_id']
    
    # Construct encoder input: [prefix, <mask_0>, postfix]
    mask_tokens = torch.full((batch_size, 1), mask_id, dtype=torch.long, device=device)
    encoder_input_ids = torch.cat([prefix_batch, mask_tokens, postfix_batch], dim=1)

    return _t5_decoder_engine(model, encoder_input_ids, max_new, temp, top_k)

def _t5_decoder_engine(model, encoder_input_ids, max_new, temp, top_k):
    """
    The low-level decoder engine. Takes a pre-formatted encoder input
    and generates the corresponding sequence. This is the unified backend
    for both continuation and in-filling.
    """
    pad_id = model_config['pad_token_id']
    eos_id = model_config['eos_token_id']
    mask_id = model_config['mask_token_start_id']
    batch_size = encoder_input_ids.shape[0]
    device = encoder_input_ids.device
    
    encoder_padding_mask = (encoder_input_ids != pad_id)
    encoder_hidden_states = model.encode(encoder_input_ids, encoder_padding_mask)
    
    # The decoder always starts by being prompted with <mask_0>
    decoder_input_ids = torch.full((batch_size, 1), mask_id, dtype=torch.long, device=device)
    has_finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

    for _ in range(max_new):
        logits = model.decode_step(decoder_input_ids, encoder_hidden_states, encoder_padding_mask)
        logits = logits[:, -1, :] / temp
        
        if top_k is not None:
            k = min(top_k, logits.size(-1))
            v, _ = torch.topk(logits, k)
            logits[logits < v[:, [-1]]] = -float('Inf')
            
        probs = nn.functional.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        
        idx_next[has_finished] = pad_id
        has_finished |= (idx_next.squeeze() == eos_id)
        decoder_input_ids = torch.cat((decoder_input_ids, idx_next), dim=1)
        if has_finished.all():
            break
            
    # Post-process: return a list of clean tensors (variable length)
    output_sequences = []
    for i in range(batch_size):
        seq = decoder_input_ids[i, 1:] # Slice off the starting <mask_0>
        eos_idx = (seq == eos_id).nonzero(as_tuple=True)[0]
        if len(eos_idx) > 0:
            seq = seq[:eos_idx[0]]
        output_sequences.append(seq)
        
    return output_sequences

# --- Main PARALLEL Long-Form Generation Loop ---
print(f"\n--- Generating {args.num_samples} long-form samples in parallel over {args.num_iterations} iterations ---")
# Start with the entire batch of seed prompts
initial_sequences = prompt_generator.get_prompts(
    num_prompts=args.num_samples,
    prompt_length=args.prompt_length
).to(args.device)

current_sequences = initial_sequences
pad_id = model_config['pad_token_id']
batch_size = args.num_samples

with torch.no_grad():
    with ctx:
        for i in tqdm(range(args.num_iterations), desc="Rollout Progress"):
            context = current_sequences[:, -args.prompt_length:]
            
            if random.random() < args.infill_prob:
                print(f"\nStep {i+1}: Performing ADDITIVE in-fill...")
                
                # --- THIS IS THE FINAL, CORRECTED STITCHING LOGIC FOR ADDITIVE IN-FILL ---
                # 1. The in-fill generation itself is correct.
                # It generates the "improvisation" between the prefix and postfix of the context.
                infilled_tokens_list = t5_infill_step(
                    model, context, args.infill_length, args.temperature, args.top_k
                )

                # 2. Extract the 'postfix' from the original context. This is the recurring motif.
                prefix_len = context.shape[1] // 2
                postfix_motif = context[:, -prefix_len:]

                # 3. Assemble the part to be APPENDED.
                # This part is [generated_improvisation] + [recurring_motif].
                next_sequences_list = []
                for j in range(batch_size):
                    part_to_append = torch.cat([infilled_tokens_list[j], postfix_motif[j]])
                    
                    # The new tape is simply the old tape plus the new part.
                    full_extended_sequence = torch.cat([current_sequences[j], part_to_append])
                    next_sequences_list.append(full_extended_sequence)
                
                # 4. Pad all sequences in the batch to the new maximum length.
                current_sequences = torch.nn.utils.rnn.pad_sequence(
                    next_sequences_list, batch_first=True, padding_value=pad_id
                )

            else: # Continuation logic (this was already correct)
                print(f"\nStep {i+1}: Performing continuation...")
                new_tokens_list = t5_continue_step(
                    model, context, args.tokens_per_step, args.temperature, args.top_k
                )

                padded_new_tokens = torch.nn.utils.rnn.pad_sequence(
                    new_tokens_list, batch_first=True, padding_value=pad_id
                )
                current_sequences = torch.cat([current_sequences, padded_new_tokens], dim=1)

final_long_sequences = current_sequences

# --- STAGE 3: UNIFIED SAVING ---
# Determine filename based on initialization mode
suffix = "infilled_rollout" if args.infill else "continued_rollout"
output_filename = f"tencache\{output_filename_stem}_{suffix}.pt"

print(f"\nSaving generated token sequences to: {output_filename}")
torch.save({
    # The format is now consistent regardless of mode
    'initial_context': initial_sequences.cpu(),
    'final_rollout': final_long_sequences.cpu(),
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
        '--input_file', output_filename, 
        "--device", args.device
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