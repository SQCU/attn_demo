#prepare.py 
###
### HOW TO USE THIS BAD BOY
### python prepare.py --trainfile TinyStoriesV2-GPT4-train.txt --valfile TinyStoriesV2-GPT4-valid.txt -p tinystories-gpt4
### wait
### receive collection of nanogpt style packed binaries

import os
import argparse
import multiprocessing as mp

from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset # huggingface datasets

#from modded_ngpt
def write_datafile(filename, toks):
    """
    write token indices to bin for c-likes
    a: header of 256x int32s
    b: sequence of unit16 tokens
    """
    assert len(toks) < 2**31, "ur trying to tokenize too much (over 2.1b)"
    # write header
    header = np.zeros(256, dtype=np.int32)
    header[0] = 20240520    # modded_ngpt magic
    header[1] = 1           # version
    header[2] = len(toks)   # number of tokens after the 256*4 bytes of header (each 2 bytes as uint16)
    # constrruct tokens npy listy
    if not isinstance(toks, np.ndarray) or not toks.dtype == np.uint16:
        # validate element width
        maxtok = 2**16
        assert all(0 <= t < maxtok for t in toks), "token dict 2 big 4 uint16" 
        toks_np = np.array(toks, dtype=np.uint16)
    else:
        toks_np = toks
    #write 2 file
    print(f"writing {len(toks):,} tokens to {filename}")
    with open(filename, "wb") as f:
        f.write(header.tobytes())
        f.write(toks_np.tobytes())

# selection
parser = argparse.ArgumentParser(description="modded_nanogpt dataset preprocessing")
parser.add_argument("-t", "--trainfile", type=str, default="Tinystories-train.txt", help="Which dataset to use for train.")
parser.add_argument("-v", "--valfile", type=str, default="Tinystories-valid.txt", help="Which dataset to use for train.")
parser.add_argument("-s", "--shard_size", type=int, default=10**8, help="Size of each shard in tokens")
parser.add_argument("-p", "--projectname", type=str, default="tinystories", help="what project we on?")
args = parser.parse_args()

if args.trainfile:
    local_dir = args.trainfile.rsplit(sep="-train.txt",maxsplit=1)[0]  #... just... don't ask...
else:
    #emergency filler
    local_dir = "generic_bin"



enc = tiktoken.get_encoding("gpt2") #should match tstories
eot = enc._special_tokens['<|endoftext|>'] # end of text token

def tokenize(doc):  #this can parse multiple 'documents', but will it?
    tokens = [eot] # the wonderful pre-delimiter <|endoftext|>

    #tokens.extend(enc.encode_ordinary(doc["text"]))
    # ... is this for a different dataset format? probably!
    tokens.extend(enc.encode_ordinary(doc['text']))
    tokens_np = np.array(tokens)
    maxtok = 2**16
    assert (0 <= tokens_np).all() and (tokens_np < maxtok).all(), "token dict 2 big 4 uint16"
    tokens_np_uint16 = tokens_np.astype(np.uint16)
    return tokens_np_uint16

    
# sensible execution guard against ram explosion:
if __name__ == '__main__':
    nprocs = max(1, os.cpu_count() - 4) # don't hog the entire system
    # -2 isn't enough since there's a host process too.
    #this is a mess.
    #data_files = {'train':args.trainfile, 'val':args.valfile}
    #datasets = load_dataset("text", split='train',
    #data_dir=os.path.dirname(__file__), data_files=args.trainfile, num_proc=nprocs)

    tset = load_dataset("text", split='train', data_dir=os.path.dirname(__file__), data_files={'train':args.trainfile}, num_proc=nprocs)
    vset = load_dataset("text", split='val', data_dir=os.path.dirname(__file__), data_files={'val':args.valfile}, num_proc=nprocs)
    datasets = {"train": tset, "val":vset}
    print(datasets)

    #os-independent bin cache dir creation code
    DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)

    for split, dataset in datasets.items():
        with mp.Pool(nprocs) as pool:
            shard_index = 0
            # preallocate shard space
            all_tokens_np = np.empty((args.shard_size,), dtype=np.uint16)
            token_count = 0
            progress_bar = None
            for tokens in pool.imap(tokenize, dataset, chunksize=32000):

                #space in shard for new entries?
                if token_count + len(tokens) < args.shard_size:
                    # simple append
                    all_tokens_np[token_count:token_count+len(tokens)] = tokens
                    token_count += len(tokens)
                    #proggy!!
                    if progress_bar is None:
                        progress_bar = tqdm(total=args.shard_size, unit="tokens", desc=f"shard(🦈):{shard_index}")
                    progress_bar.update(len(tokens))
                else:
                    # write filled shard and start new shard
                    #split = "val" if shard_index == 0  else "train"
                        # sus on that last part. 
                        #split = "train"
                    filename = os.path.join(DATA_CACHE_DIR, f"{args.projectname}_{split}_{shard_index:06d}.bin")
                    # peel document into whatsoever fits into shard; remainder yeet-yooted to next shard.
                    remainder = args.shard_size - token_count
                    progress_bar.update(remainder)
                    all_tokens_np[token_count:token_count+remainder] = tokens[:remainder]
                    write_datafile(filename, all_tokens_np)
                    shard_index +=1
                    progress_bar = None # death drive mentioned? # beyond the progress principle?
                    # slop that remainder on in yeehaw
                    all_tokens_np[0:len(tokens) - remainder] = tokens[remainder:]
                    token_count = len(tokens) - remainder
                
            if token_count != 0:
                #split = "val" if shard_index == 0  else "train"
                    # sus on that last part. 
                    #split = "train"
                filename = os.path.join(DATA_CACHE_DIR, f"{args.projectname}_{split}_{shard_index:06d}.bin")
                write_datafile(filename, all_tokens_np[:token_count])
    

