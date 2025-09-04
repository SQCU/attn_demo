# prepare_ascii.py
# e.g. 
# uv run data\prepare_ascii.py --trainfile data\txt\TinyStoriesV2-GPT4-train.parquet --valfile data\txt\TinyStoriesV2-GPT4-valid.parquet -p tinystories-ascii
# This command will create a directory named `tinystories-ascii` containing the `.bin` files, 
# where each token ID corresponds to an ASCII character value.
import os
import argparse
import multiprocessing as mp
import numpy as np
from datasets import load_dataset
from tqdm import tqdm

# --- NEW: SimpleASCIITokenizer Class ---
# We define the tokenizer directly in this script for simplicity.
class SimpleASCIITokenizer:
    """
    A simple tokenizer for the first 128 ASCII characters.
    """
    def __init__(self):
        # Create a vocabulary of the first 128 ASCII characters
        self.chars = [chr(i) for i in range(128)]
        self.vocab_size = len(self.chars)
        
        # Create mappings from characters to integers and vice-versa
        self.char_to_int = {char: i for i, char in enumerate(self.chars)}
        self.int_to_char = {i: char for i, char in enumerate(self.chars)}

    def encode(self, text):
        """Converts a string to a list of integer token IDs."""
        # For any character outside the 128 ASCII range, we can map it to a default,
        # like 0 (NULL character), or simply ignore it. Here, we default to 0.
        return [self.char_to_int.get(char, 0) for char in text]

    def decode(self, tokens):
        """Converts a list of integer token IDs back to a string."""
        return "".join([self.int_to_char.get(token, '') for token in tokens])

# --- UNCHANGED: The robust binary writing function from your script ---
def write_datafile(filename, toks):
    """
    Writes token indices to a .bin file in the required format.
    """
    assert len(toks) < 2**31, "You are trying to tokenize too much data (over 2.1 billion tokens)"
    
    header = np.zeros(256, dtype=np.int32)
    header[0] = 20240520    # Magic number for modded_ngpt
    header[1] = 1           # Version
    header[2] = len(toks)   # Number of tokens
    
    if not isinstance(toks, np.ndarray) or not toks.dtype == np.uint16:
        maxtok = 2**16
        assert all(0 <= t < maxtok for t in toks), "Token dictionary is too large for uint16" 
        toks_np = np.array(toks, dtype=np.uint16)
    else:
        toks_np = toks
        
    print(f"Writing {len(toks):,} tokens to {filename}")
    with open(filename, "wb") as f:
        f.write(header.tobytes())
        f.write(toks_np.tobytes())

# --- MODIFIED: The tokenize function now uses our ASCII tokenizer ---
# We make the tokenizer a global variable to be accessible by the multiprocessing pool
tokenizer = SimpleASCIITokenizer()

def tokenize_document(doc):
    """
    Takes a document from the Hugging Face dataset and returns a numpy array of ASCII token IDs.
    """
    # The original script prepended an <|endoftext|> token. For a simple ASCII model,
    # we can treat the text as a continuous stream. Document boundaries are implicitly handled
    # by how the data is laid out in the shards.
    # character-level-modeling means a long eos is annoying. lets do something semi-compact:
    
    eos_span = tokenizer.encode('<eos>')
    tokens = []
    tokens.extend(eos_span)
    text = doc['text']
    tokens.extend(tokenizer.encode(text))
    
    tokens_np = np.array(tokens, dtype=np.uint16)
    
    # The validation check is still useful, ensuring our tokens fit in uint16
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "Token IDs are out of uint16 range"
    return tokens_np

# --- MAIN EXECUTION LOGIC (Largely Unchanged) ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ASCII dataset preprocessing for modded_nanogpt")
    parser.add_argument("-t", "--trainfile", type=str, required=True, help="Path to the training .parquet file.")
    parser.add_argument("-v", "--valfile", type=str, required=True, help="Path to the validation .parquet file.")
    parser.add_argument("-s", "--shard_size", type=int, default=10**8, help="Size of each shard in tokens.")
    parser.add_argument("-p", "--projectname", type=str, default="tinystories-ascii", help="Project name for the output directory.")
    args = parser.parse_args()

    # Create the output directory
    DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), args.projectname)
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)

    # Use a sensible number of processes
    nprocs = max(1, os.cpu_count() - 2)

    # Load the datasets using Hugging Face datasets
    tset = load_dataset("parquet", split='train', data_files={'train': args.trainfile}, num_proc=nprocs)
    vset = load_dataset("parquet", split='val', data_files={'val': args.valfile}, num_proc=nprocs)
    datasets = {"train": tset, "val": vset}
    print(f"Loaded datasets: {datasets}")
    print(f"Using an ASCII tokenizer with vocab size: {tokenizer.vocab_size}")

    # Process each split (train and val)
    for split, dataset in datasets.items():
        with mp.Pool(nprocs) as pool:
            shard_index = 0
            all_tokens_np = np.empty((args.shard_size,), dtype=np.uint16)
            token_count = 0
            progress_bar = None

            for tokens in pool.imap(tokenize_document, dataset, chunksize=16000):
                if token_count + len(tokens) < args.shard_size:
                    all_tokens_np[token_count : token_count + len(tokens)] = tokens
                    token_count += len(tokens)
                    if progress_bar is None:
                        progress_bar = tqdm(total=args.shard_size, unit="tokens", desc=f"Shard {shard_index} ({split})")
                    progress_bar.update(len(tokens))
                else:
                    remainder = args.shard_size - token_count
                    all_tokens_np[token_count : token_count + remainder] = tokens[:remainder]
                    progress_bar.update(remainder)
                    
                    filename = os.path.join(DATA_CACHE_DIR, f"{args.projectname}_{split}_{shard_index:06d}.bin")
                    write_datafile(filename, all_tokens_np)
                    
                    shard_index += 1
                    progress_bar.close()
                    progress_bar = None
                    
                    all_tokens_np[0 : len(tokens) - remainder] = tokens[remainder:]
                    token_count = len(tokens) - remainder
            
            if token_count != 0:
                filename = os.path.join(DATA_CACHE_DIR, f"{args.projectname}_{split}_{shard_index:06d}.bin")
                write_datafile(filename, all_tokens_np[:token_count])