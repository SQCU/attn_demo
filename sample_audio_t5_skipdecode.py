"""
# sample_audio_t5_skipdecode.py
# Generates audio token sequences with a T5 model and immediately decodes them
# into high-fidelity .wav files using the BPE-based skip-decoder.
# This is the primary script for generating listenable audio.
#
# --- USAGE ---
# uv run sample_audio_t5_skipdecode.py --out_dir=logs/your_run_dir
# uv run sample_audio_t5_skipdecode.py --out_dir "logs/mformer-t5-l6-mk1-21d0cffb-48ce-4195-880a-e6c9ca5b1a0b"
"""
import os
import torch
import torchaudio
import argparse
import numpy as np
import pandas as pd
import pickle
import random
import json
from contextlib import nullcontext
import torch.nn as nn
from tqdm import tqdm

import pgptlformer
from prompt_utils import AudioPromptGenerator
from encodec import EncodecModel

# --- Configuration via Argparse ---
parser = argparse.ArgumentParser(description="Generate and Skip-Decode audio from a T5 model.")
# Model/Generation args
parser.add_argument('--out_dir', type=str, required=True, help="The output directory from your audio T5 training run.")
parser.add_argument('--checkpoint_name', type=str, default='state_step002000.pt', help="Name of the checkpoint file.")
parser.add_argument('--num_samples', type=int, default=4, help="How many different audio clips to generate.")
parser.add_argument('--prompt_length', type=int, default=128, help="Length of the audio prompt in tokens (~6.8 seconds).")
parser.add_argument('--max_new_tokens', type=int, default=512, help="How many new tokens to generate (~6.8 seconds).")
parser.add_argument('--temperature', type=float, default=1.0, help="Temperature for sampling.")
parser.add_argument('--top_k', type=int, default=250, help="Top-k for sampling.")
parser.add_argument('--seed', type=int, default=420, help="Random seed.")
parser.add_argument('--device', type=str, default='cuda', help="Device to use ('cuda' or 'cpu').")
parser.add_argument('--compile', action='store_true', help="Enable torch.compile (default: off).")
# Skip-Decoder Data Paths
parser.add_argument('--raw_audio_path', type=str, default="data/measureformer/MORGPROJ_II.wav", help="Path to the original, raw audio file.")
parser.add_argument('--npz_token_path', type=str, default="data/measureformer/mproj_ii.npz", help="Path to the original token file for prompts.")
parser.add_argument('--parquet_file', type=str, default="data/measureformer_analysis/2025-09-13_18-40-31_9a4c.parquet", help="Path to the priority scores for prompts.")
parser.add_argument('--bpe_vocab_path', type=str, default="data/measureformer/audio_bpe_vocab.json", help="Path to the learned BPE vocabulary.")
parser.add_argument('--bpe_index_path', type=str, default="data/measureformer/skip_decoder_bpe_index.pkl", help="Path to the pre-built BPE n-gram index.")
# Output arg
parser.add_argument('--output_dir', type=str, default='generated_audio', help="Directory to save the final output .wav files.")
args = parser.parse_args()

# --- The BPE-Aware SkipDecoder Class ---
# --- NEW: The Final "Nearest Neighbor" GreedySkipDecoder Class ---
class GreedySkipDecoder:
    def __init__(self, vocab_path, index_path, raw_audio_path, npz_token_path, device='cuda'):
        print("\n--- Initializing Nearest Neighbor GreedySkipDecoder ---")
        
        print(f"  - Loading BPE vocabulary from: {vocab_path}")
        with open(vocab_path, 'r') as f:
            bpe_vocab = {int(k): tuple(v) for k, v in json.load(f).items()}

        print(f"  - Loading BPE index from: {index_path}")
        with open(index_path, 'rb') as f:
            self.bpe_index = pickle.load(f)

        print(f"  - Pre-computing BPE word matrix for similarity search...")
        self.seq_to_id = {v: k for k, v in bpe_vocab.items()}
        # Get all fused token sequences (our "dictionary" of audio words)
        self.bpe_words = [torch.tensor(seq, dtype=torch.float32) for seq, i in self.seq_to_id.items() if i >= 1024]
        
        print(f"  - Loading raw audio from: {raw_audio_path}")
        self.wav, self.sr = torchaudio.load(raw_audio_path)
        # --- FIX for channel mismatch ---
        # Ensure raw audio is stereo for consistency.
        if self.wav.shape[0] == 1:
            print("    - Warning: Raw audio is mono. Duplicating channel to create stereo.")
            self.wav = self.wav.repeat(2, 1)
        
        self.samples_per_token = self.wav.shape[1] / len(np.load(npz_token_path)['tokens'][0])
        print(f"  - Calculated {self.samples_per_token:.2f} audio samples per token.")
        
        self.device = device

    def _find_best_match(self, query_sequence):
        """Finds the BPE word in our vocabulary with the highest cosine similarity to the query."""
        query_tensor = torch.tensor(query_sequence, dtype=torch.float32)
        best_sim = -1.0
        best_match_seq = None

        for word in self.bpe_words:
            # Pad the shorter tensor to match the length of the longer one
            if len(query_tensor) > len(word):
                padded_word = torch.nn.functional.pad(word, (0, len(query_tensor) - len(word)))
                padded_query = query_tensor
            else:
                padded_query = torch.nn.functional.pad(query_tensor, (0, len(word) - len(query_tensor)))
                padded_word = word
            
            sim = torch.nn.functional.cosine_similarity(padded_query, padded_word, dim=0)
            if sim > best_sim:
                best_sim = sim
                best_match_seq = tuple(word.long().tolist())
                
        return best_match_seq

    def decode(self, generated_tokens, crossfade_ms=5):
        output_waveform = torch.tensor([], dtype=torch.float32)
        tokens_np = generated_tokens.cpu().numpy()
        
        i = 0
        pbar = tqdm(total=len(tokens_np), desc="  - Greedy Skip-Decoding")
        while i < len(tokens_np):
            match_found = False
            # Search for the longest possible BPE match
            longest_possible_len = min(len(tokens_np) - i, max(len(s) for s in self.bpe_words))
            
            for length in range(longest_possible_len, 0, -1):
                sub_sequence = tuple(tokens_np[i : i + length])
                
                if sub_sequence in self.seq_to_id:
                    # --- EXACT MATCH FOUND! ---
                    fused_id = self.seq_to_id[sub_sequence]
                    if fused_id in self.bpe_index and len(self.bpe_index[fused_id]) > 0:
                        original_token_start_idx = random.choice(self.bpe_index[fused_id])
                        start_sample = int(original_token_start_idx * self.samples_per_token)
                        end_sample = int((original_token_start_idx + length) * self.samples_per_token)
                        audio_chunk = self.wav[:, start_sample:end_sample].clone()
                        i += length
                        pbar.update(length)
                        match_found = True
                        break
            
            # If no exact match was found, use the "Nearest Neighbor" fallback
            if not match_found:
                # Use a small lookahead window for the query
                query_len = min(8, len(tokens_np) - i)
                query_sequence = tokens_np[i : i + query_len]
                
                # --- HEURISTIC FALLBACK LOGIC ---
                best_match_seq = self._find_best_match(query_sequence)
                
                fused_id = self.seq_to_id[best_match_seq]
                length = len(best_match_seq)
                
                original_token_start_idx = random.choice(self.bpe_index[fused_id])
                start_sample = int(original_token_start_idx * self.samples_per_token)
                end_sample = int((original_token_start_idx + length) * self.samples_per_token)
                audio_chunk = self.wav[:, start_sample:end_sample].clone()
                
                # Advance by the length of the query, not the matched chunk, to avoid skipping too much
                i += query_len
                pbar.update(query_len)

            # ... (Stitching and crossfading logic is now safe and remains the same) ...
            if output_waveform.numel() > 0:
                fade_len = min(int(self.sr * (crossfade_ms / 1000.0)), audio_chunk.shape[1], output_waveform.shape[1])
                if fade_len > 0:
                    fade_out = torch.linspace(1.0, 0.0, fade_len)
                    output_waveform[:, -fade_len:] *= fade_out
                    fade_in = torch.linspace(0.0, 1.0, fade_len)
                    audio_chunk[:, :fade_len] *= fade_in
                    overlapped_part = output_waveform[:, -fade_len:] + audio_chunk[:, :fade_len]
                    output_waveform = torch.cat([output_waveform[:, :-fade_len], overlapped_part, audio_chunk[:, fade_len:]], dim=1)
                else:
                    output_waveform = torch.cat([output_waveform, audio_chunk], dim=1)
            else:
                output_waveform = audio_chunk
        pbar.close()
        return output_waveform.unsqueeze(0)

from scipy.ndimage import gaussian_filter1d
import tempfile

def create_smoothed_priority_file(original_parquet_path, sigma=2.5):
    """
    Reads priority scores, applies a smoothing kernel, and saves the result
    to a temporary file. Returns the path to this temporary file.
    """
    print(f"\n--- Intercepting priority scores for debouncing (sigma={sigma}) ---")
    
    # 1. Read the original priority scores
    df = pd.read_parquet(original_parquet_path)
    raw_scores = df['priority_score'].values
    print("  - Original scores loaded.")

    # 2. Apply the smoothing kernel ("debounce")
    smoothed_scores = gaussian_filter1d(raw_scores, sigma=sigma)
    print("  - Smoothing kernel applied.")

    # 3. Create a new DataFrame and save to a temporary file
    smoothed_df = pd.DataFrame({'priority_score': smoothed_scores})
    
    # Use NamedTemporaryFile to get a stable path for the file's lifetime
    # delete=False is crucial so we can close it and still use the path.
    temp_file = tempfile.NamedTemporaryFile(suffix=".parquet", delete=False)
    temp_path = temp_file.name
    temp_file.close() # Close the file handle so pandas can write to it.

    smoothed_df.to_parquet(temp_path)
    print(f"  - Smoothed scores saved to temporary file: {os.path.basename(temp_path)}")
    
    # 4. Return the path to the new temporary file
    return temp_path

# --- T5 Generation Function (from sample_audio_t5.py) ---
def t5_audio_decode(model, model_config, encoder_input_ids, max_new_tokens, temperature, top_k):
    # ... (This function is identical to the one in the previous script) ...
    # ... (It's included here for completeness) ...
    pad_id = model_config['pad_token_id']
    eos_id = model_config['eos_token_id']
    batch_size = encoder_input_ids.shape[0]
    decoder_input_ids = torch.full((batch_size, 1), pad_id, dtype=torch.long, device=args.device)
    encoder_padding_mask = (encoder_input_ids != pad_id)
    encoder_hidden_states = model.encode(encoder_input_ids, encoder_padding_mask)
    has_finished = torch.zeros(batch_size, dtype=torch.bool, device=args.device)
    for _ in range(max_new_tokens):
        logits = model.decode_step(decoder_input_ids, encoder_hidden_states, encoder_padding_mask)
        logits = logits[:, -1, :] / temperature
        if top_k is not None:
            k = min(top_k, logits.size(-1))
            v, _ = torch.topk(logits, k)
            threshold = v[:, -1].unsqueeze(-1)
            logits[logits < threshold] = -float('Inf')
        probs = nn.functional.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx_next[has_finished] = pad_id
        has_finished |= (idx_next.squeeze(-1) == eos_id)
        decoder_input_ids = torch.cat((decoder_input_ids, idx_next), dim=1)
        if has_finished.all():
            break
    return decoder_input_ids[:, 1:]

# --- Main Script Logic ---
# 1. Boilerplate and Model Loading
torch.manual_seed(args.seed)

from datetime import datetime
run_id = datetime.now().strftime('%y%m%d%H%M%S') # e.g., "223104" for 10:31:04 PM
print(f"Generated a unique ID for this run's output files: {run_id}")

print(f"Loading model from: {args.out_dir}")
ckpt_path = os.path.join(args.out_dir, args.checkpoint_name)
# Load the entire checkpoint dictionary. Since you trained this model,
# it's a trusted source, and we can safely use weights_only=False.
checkpoint = torch.load(ckpt_path, map_location=args.device)
# Correctly extract the config and state dict from the dictionary
model_config = checkpoint['model_args']
state_dict = checkpoint['model']
# Instantiate the model with the correct config
model = pgptlformer.PGPT_Lformer(model_config)
# Clean the state_dict keys (for DataParallel/compile artifacts)
unwanted_prefix = '_orig_mod.'
for k, v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
# Load the cleaned state_dict
model.load_state_dict(state_dict)
model.eval()
model.to(args.device)
if args.compile:
    print("Compiling model for inference...")
    model.encode = torch.compile(model.encode)
    model.decode_step = torch.compile(model.decode_step)

# 2. Generate Prompts
temp_parquet_path = create_smoothed_priority_file(
            args.parquet_file, 
            sigma=2.5 # You can make this an argparse parameter later if you like it
        )
prompt_generator = AudioPromptGenerator(npz_path=args.npz_token_path, parquet_path=temp_parquet_path)
prompt_tokens = prompt_generator.get_prompts(num_prompts=args.num_samples, prompt_length=args.prompt_length).to(args.device)

# 3. Generate Token Sequences with T5 Model
infill_point = args.prompt_length // 2
encoder_input_ids = prompt_tokens.clone()
encoder_input_ids[:, infill_point:] = model_config['mask_token_start_id']
print(f"\n--- Generating {args.num_samples} token sequences with T5 model... ---")
with torch.no_grad(), torch.amp.autocast(device_type='cuda' if 'cuda' in args.device else 'cpu', dtype=torch.bfloat16):
    generated_sequences = t5_audio_decode(model, model_config, encoder_input_ids, args.max_new_tokens, args.temperature, args.top_k)
print("Token generation complete.")

# 4. Instantiate the SkipDecoder
skip_decoder = GreedySkipDecoder(
    index_path=args.bpe_index_path, vocab_path=args.bpe_vocab_path,
    raw_audio_path=args.raw_audio_path, npz_token_path=args.npz_token_path,
    device=args.device
)

# 5. Decode each generated sequence into a WAV file
os.makedirs(args.output_dir, exist_ok=True)
print(f"\n--- Decoding {args.num_samples} sequences to WAV files in '{args.output_dir}' ---")
for i in range(args.num_samples):
    print(f"Processing sample {i+1}/{args.num_samples}...")
    # Create the final, full token sequence to be decoded
    final_tokens = torch.cat([prompt_tokens[i, :infill_point], generated_sequences[i]])
    # Perform the skip-decoding
    waveform = skip_decoder.decode(final_tokens)
    # Save the high-fidelity audio
    base_run_name = os.path.basename(args.out_dir)
    output_filename = os.path.join(
        args.output_dir, 
        f"{run_id}_sample_{i}_{base_run_name}.wav"
    )
    #output_filename = os.path.join(args.output_dir, f"sample_{i}_{os.path.basename(args.out_dir)}.wav")
    torchaudio.save(output_filename, waveform.squeeze(0).cpu(), sample_rate=skip_decoder.sr)
    print(f"  -> Saved to {output_filename}")

print("\n--- All tasks complete ---")