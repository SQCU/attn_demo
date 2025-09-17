# create_bpe_index.py (Version 3.0 - Unicode Stringification)
# Uses the Aho-Corasick algorithm on a unicode string representation of tokens.
# This is the correct and fast method for default pyahocorasick installations.
#
# --- USAGE ---
# uv run create_bpe_index.py
import numpy as np
import json
import pickle
from tqdm import tqdm
from collections import defaultdict
import ahocorasick

# --- Configuration ---
NPZ_FILE = "data/measureformer/mproj_ii.npz"
BPE_VOCAB_FILE = "data/measureformer/audio_bpe_vocab.json"
OUTPUT_INDEX_FILE = "data/measureformer/skip_decoder_bpe_index.pkl"
BASE_VOCAB_SIZE = 1024

# --- Main Logic ---
print("Loading original tokens and BPE vocabulary...")
# Load tokens as a standard integer array.
tokens = np.load(NPZ_FILE, mmap_mode='r')['tokens'][0, :]
with open(BPE_VOCAB_FILE, 'r') as f:
    bpe_vocab = {int(k): tuple(v) for k, v in json.load(f).items()}

fused_tokens = {k: v for k, v in bpe_vocab.items() if k >= BASE_VOCAB_SIZE}
print(f"Found {len(fused_tokens)} fused-tokens in the vocabulary to index.")

# 1. Build the Aho-Corasick Automaton using UNICODE STRINGS
print("Building Aho-Corasick automaton...")
A = ahocorasick.Automaton()
for fused_id, constituent_tokens in tqdm(fused_tokens.items(), desc="Adding patterns to automaton"):
    # --- THE FIX ---
    # Convert the tuple of ints to a unicode string by mapping each int to its character.
    pattern_str = "".join(map(chr, constituent_tokens))
    A.add_word(pattern_str, fused_id)
A.make_automaton()
print("Automaton built.")

bpe_index = defaultdict(list)

# 2. Convert the entire token stream to a single unicode string and iterate ONCE
print("Converting token stream to a single unicode string for searching...")
# This will be a very long string, but it's memory-efficient in Python 3.
token_str = "".join(map(chr, tokens))

print("Scanning token stream with automaton (this will be fast)...")
# A.iter() on a string returns (end_index, value).
# The end_index is a character index, which is now a 1-to-1 match for our token index.
for end_index, fused_id in tqdm(A.iter(token_str), total=len(token_str)):
    # The returned index is the *end* of the match. We need the *start*.
    constituent_len = len(fused_tokens[fused_id])
    start_token_index = end_index - constituent_len + 1
    
    bpe_index[fused_id].append(start_token_index)

print(f"\nIndexing complete. The index contains {len(bpe_index)} entries.")

print(f"Saving BPE-based index to {OUTPUT_INDEX_FILE}...")
with open(OUTPUT_INDEX_FILE, 'wb') as f:
    pickle.dump(dict(bpe_index), f)

print("\n--- BPE Index Creation Complete ---")