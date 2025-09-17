# learn_audio_bpe.py
# Implements the BPE algorithm on a stream of Encodec tokens to learn a 
# vocabulary of common, reusable audio events.
#
# --- USAGE ---
# uv run learn_audio_bpe.py --vocab_size 4096
# in al ldeep learning situations try to aim for powers of 2 ;)
import numpy as np
import argparse
from collections import defaultdict, Counter
import json
from tqdm import tqdm

def get_stats(ids):
    """Counts all adjacent pairs in a list of integers."""
    counts = Counter()
    for pair in zip(ids, ids[1:]):
        counts[pair] += 1
    return counts

def merge(ids, pair, idx):
    """Replaces all occurrences of a pair with a new token index."""
    new_ids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and (ids[i], ids[i+1]) == pair:
            new_ids.append(idx)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1
    return new_ids

# --- Configuration ---
parser = argparse.ArgumentParser(description="Learn a BPE vocabulary from audio tokens.")
parser.add_argument('--npz_file', type=str, default="data/measureformer/mproj_ii.npz")
parser.add_argument('--output_prefix', type=str, default="data/measureformer/audio_bpe")
parser.add_argument('--vocab_size', type=int, default=1500, help="Target vocabulary size (base tokens + new merges).")
args = parser.parse_args()

# --- Main Logic ---
print(f"Loading tokens from {args.npz_file}...")
tokens = np.load(args.npz_file)['tokens'][0, :].tolist()

num_base_tokens = 1024 # Encodec vocabulary
target_vocab_size = args.vocab_size
num_merges = target_vocab_size - num_base_tokens

merges = {} # (tok1, tok2) -> new_tok_id
vocab = {i: (i,) for i in range(num_base_tokens)} # new_tok_id -> (constituents)

print(f"Starting BPE training. Will perform {num_merges} merges.")
pbar = tqdm(range(num_merges), desc="Learning BPE Merges")
for i in pbar:
    stats = get_stats(tokens)
    if not stats:
        break # No more pairs to merge
    
    # Find the most frequent pair
    best_pair = max(stats, key=stats.get)
    
    # The ID for our new merged token
    new_token_id = num_base_tokens + i
    
    # Perform the merge
    tokens = merge(tokens, best_pair, new_token_id)
    
    # Store the merge rule and the new vocab entry
    merges[best_pair] = new_token_id
    vocab[new_token_id] = vocab[best_pair[0]] + vocab[best_pair[1]]
    
    pbar.set_description(f"Merge {i+1}/{num_merges}: {best_pair} -> {new_token_id}")

print(f"\nBPE training complete. Final vocabulary size: {len(vocab)}")

# --- Save the vocabulary ---
# The vocab maps the new "fused-token" IDs to their original constituent tokens.
# This is essential for the SkipDecoder.
vocab_file = f"{args.output_prefix}_vocab.json"
print(f"Saving vocabulary to {vocab_file}...")
# Convert tuples to lists for JSON compatibility
json_vocab = {k: list(v) for k, v in vocab.items()}
with open(vocab_file, 'w') as f:
    json.dump(json_vocab, f)

# The merges are useful for re-tokenizing but not strictly needed by the decoder.
merges_file = f"{args.output_prefix}_merges.txt"
print(f"Saving merge rules to {merges_file}...")
with open(merges_file, 'w') as f:
    for pair, idx in merges.items():
        f.write(f"{pair[0]} {pair[1]}\n")

print("\n--- BPE Learning Complete ---")