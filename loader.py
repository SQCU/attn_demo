#loader.py
### canonically 
### torchrun --standalone --nproc_per_node=8 loader.py
### but for us, probably 
### set USE_LIBUV=0
### set RANK 
### set TORCH_CUDNN_SDPA_ENABLED=1
### torchrun --standalone --nproc_per_node=1 loader.py
### ...
### uv run python loader.py --config_file configs/ascii_chart5_model.json
import os
import sys
import uuid
import glob
import time
import json
import argparse
from dataclasses import dataclass, field, asdict, fields

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
import torch._inductor.config as tconfig #... as tconfig? what on earth was that? oh okay.
from torch.nn.parallel import DistributedDataParallel as DDP
from datetime import datetime

from config_utils import ConfigError, cfg_get, is_doc_key, validate_model_config
from prompt_utils import PromptGenerator
from sampler_utils import ar_sample
from t5_utils import T5BatchProcessor, AdaptiveCurriculumSampler # <-- ADD SAMPLER
import pgptlformer

# pandas / pyarrow / bitsandbytes are imported at their point of use, not here.
# reason: this file used to be pure top-level script -- it read its own source at import
# time, asserted torch.cuda.is_available() at module scope, and pulled in a cuda-only
# optimizer library. that made it impossible to import on any machine without a gpu, which
# in turn made every function in it untestable, which is why merge_config and the
# gradient-accumulation loop could stay broken for months. everything below is import-safe;
# the training run lives in main().

### modded-nanogpt distributed dataset loader
# -----------------------------------------------------------------------------
# their simple Distributed Data Loader

def _peek_data_shard(filename):
    # only reads the header, returns header data
    with open(filename, "rb") as f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        header = np.frombuffer(f.read(256*4), dtype=np.int32)
    if header[0] != 20240520:
        print("ERROR: magic number mismatch in the data .bin file!")
        print("---> HINT: Are you passing in a correct file with --input_bin?")
        print("---> HINT: Dataset encoding changed recently, re-run data prepro or refer again to README")
        print("---> HINT: For example re-run: `python dev/data/tinyshakespeare.py`, then re-try")
        exit(1)
    assert header[1] == 1, "unsupported version"
    ntok = header[2] # number of tokens (claimed)
    return ntok # for now just return the number of tokens

def _load_data_shard(filename):
    with open(filename, "rb") as f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        header = np.frombuffer(f.read(256*4), dtype=np.int32)
        assert header[0] == 20240520, "magic number mismatch in the data .bin file"
        assert header[1] == 1, "unsupported version"
        ntok = header[2] # number of tokens (claimed)
        # the rest of it are tokens, stored as uint16
        tokens = np.frombuffer(f.read(), dtype=np.uint16)
    assert len(tokens) == ntok, "number of tokens read does not match header?"
    return tokens

class DistributedDataLoader:
    def __init__(self, filename_pattern, B, T, process_rank, num_processes, device="cuda"):
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.B = B
        self.T = T
        # was a hardcoded .cuda() in next_batch(). taking the device makes the loader work on
        # a cpu box, which is what lets tests/test_training_semantics.py run a real training
        # loop instead of asserting about one.
        self.device = device

        # glob files that match the pattern
        self.files = sorted(glob.glob(filename_pattern))
        assert len(self.files) > 0, f"did not find any files that match the pattern {filename_pattern}"

        # load and validate all data shards, count number of tokens in total
        ntok_total = 0
        for fname in self.files:
            shard_ntok = _peek_data_shard(fname)
            assert shard_ntok >= num_processes * B * T + 1
            ntok_total += int(shard_ntok)
        self.ntok_total = ntok_total

        # kick things off
        self.reset()

    def reset(self):
        self.current_shard = 0
        self.current_position = self.process_rank * self.B * self.T
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def advance(self): # advance to next data shard
        self.current_shard = (self.current_shard + 1) % len(self.files)
        self.current_position = self.process_rank * self.B * self.T
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def next_batch(self):
        B = self.B
        T = self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        buf = torch.tensor(buf.astype(np.int32), dtype=torch.long)
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance current position and load next shard if necessary
        self.current_position += B * T * self.num_processes
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.advance()
        return x.to(self.device), y.to(self.device)

class IntelligentAudioDataLoader:
    def __init__(self, npz_path, parquet_path, B, T, process_rank, num_processes, device="cuda"):
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.B = B  # Per-device batch size
        self.T = T  # Sequence length
        self.device = device

        print(f"[Rank {self.process_rank}] Initializing IntelligentAudioDataLoader.")

        import pandas as pd # sure why not. import literally everythign that does anything with data. you go gemini2.5.
        # 1. Load the small priority scores into memory
        priority_df = pd.read_parquet(parquet_path)
        self.weights = torch.from_numpy(priority_df['priority_score'].values).float()
        
        if self.weights.sum() > 0:
            self.weights /= self.weights.sum()
        else:
            self.weights = torch.ones_like(self.weights) / len(self.weights)

        # 2. Open the large token array using memory-mapping
        print(f"[Rank {self.process_rank}] Memory-mapping tokens from {npz_path}...")
        self.tokens_memmap = np.load(npz_path, mmap_mode='r')['tokens'][0, :]

        self.total_tokens = len(self.tokens_memmap)
        self.num_chunks = len(self.weights)
        self.chunk_size_tokens = self.total_tokens // self.num_chunks

        print(f"[Rank {self.process_rank}] Setup complete. Total tokens: {self.total_tokens}")

    def next_batch(self):
        """
        Samples a batch of *clean* token indices using priority scores,
        then reads T+1 tokens from the memory-mapped file to produce (x, y).
        This maintains a consistent API with other data loaders.
        """
        # Each process samples its own indices independently
        selected_chunk_indices = torch.multinomial(self.weights, num_samples=self.B, replacement=True)
        
        # Prepare batch tensors on CPU
        batch_x_np = np.zeros((self.B, self.T), dtype=np.int64)
        batch_y_np = np.zeros((self.B, self.T), dtype=np.int64)

        for i, chunk_idx in enumerate(selected_chunk_indices):
            # Center the sequence on the chosen chunk's start
            start_token = (chunk_idx.item() * self.chunk_size_tokens) - (self.T // 2)
            
            # We need T+1 tokens to create x and y
            end_token = start_token + self.T + 1
            
            # Boundary checks
            if start_token < 0:
                start_token = 0
                end_token = start_token + self.T + 1

            if end_token > self.total_tokens:
                end_token = self.total_tokens
                start_token = end_token - (self.T + 1)
            
            # Read only the required slice from disk into a numpy array
            buf = self.tokens_memmap[start_token:end_token]
            
            # Create x and y from the buffer
            batch_x_np[i] = buf[:-1]
            batch_y_np[i] = buf[1:]

        # Convert the final numpy batches to torch tensors and move to GPU
        x = torch.from_numpy(batch_x_np).to(self.device)
        y = torch.from_numpy(batch_y_np).to(self.device)
        
        return x, y

    def reset(self):
        # This loader is stateless, but we need the method for API consistency
        pass
# -----------------------------------------------------------------------------
# (get_batch() lived here: "downgrade to poor man's data loader / maybe superfluous bc
#  distributed data loader started working / delete? [ ]". it was unreachable and could not
#  have run -- it read module-scope names `data_dir`, `device_type` and `device` that do not
#  exist in this file. checkbox ticked.)

# custom eval pipeline woooooooo~!
class OnlineRolloutSampler:
    """
    Captures autoregressive rollouts from a model during training
    and logs them to a Parquet file with rich metadata.
    """
    def __init__(self, out_dir: str, run_id: str):
        import pyarrow as pa
        self.log_file = os.path.join(out_dir, f"rollouts_{run_id}.parquet")
        self.writer = None
        self.schema = pa.schema([
            pa.field('step', pa.int64()),
            pa.field('timestamp', pa.timestamp('us')),
            pa.field('hyperparameters', pa.string()), # Store as JSON string
            pa.field('prompt', pa.string()),
            pa.field('raw_tokens', pa.list_(pa.int32())),
            pa.field('raw_logits', pa.binary()), # Store as pickled numpy array
            pa.field('decoded_text_raw', pa.string()),
            pa.field('decoded_text_cleaned', pa.string())
        ])
        print(f"OnlineRolloutSampler initialized. Logging to: {self.log_file}")

    def capture_and_log(self, model, prompts: list[str], tokenizer, metadata: dict,
                        max_new_tokens: int, temperature=1.0, top_k=200, device='cuda',
                        autocast_ctx=None):
        import pyarrow as pa
        import pyarrow.parquet as pq

        model.eval() # Ensure model is in eval mode
        prompt_tokens = [tokenizer.encode(p) for p in prompts]
        x = torch.tensor(prompt_tokens, dtype=torch.long, device=device)
        max_seq = metadata['hyperparameters']['model_config']['training_seqlen']
        # the inlined decode loop that used to live here did `logits, _, _ = model(...)`:
        # a 3-unpack of a 4-tuple, so the whole online-rollout-capture feature has been
        # raising ValueError on its first token since forward_arg grew loss_per_sequence.
        # shared, arity-proof sampler now. no eos_id is passed, so there is no per-token
        # stop check and therefore no per-token device sync in this loop at all.
        with torch.no_grad():
            with (autocast_ctx if autocast_ctx is not None else _null_ctx()):
                x = ar_sample(model, x, max_new_tokens=max_new_tokens, max_seq=max_seq,
                              temperature=temperature, top_k=top_k)

        # ONE device->host transfer for the whole batch of rollouts. it used to be
        # `x[i].tolist()` inside the row loop, i.e. one transfer per captured sequence.
        rollout_tokens = x.cpu().tolist()

        # 2. Prepare data for Parquet logging
        table_data = []
        for i, tokens_list in enumerate(rollout_tokens):

            raw_text = tokenizer.decode(tokens_list)
            # Simple cleaning: truncate at first <eos>
            cleaned_text = raw_text.split('<eos>')[0]

            row = {
                'step': metadata['step'],
                'timestamp': metadata['timestamp'],
                'hyperparameters': json.dumps(metadata['hyperparameters']),
                'prompt': prompts[i],
                'raw_tokens': tokens_list,
                'raw_logits': b'', # Placeholder
                'decoded_text_raw': raw_text,
                'decoded_text_cleaned': cleaned_text
            }
            table_data.append(row)
            
        # 3. Write to Parquet file
        table = pa.Table.from_pylist(table_data, schema=self.schema)
        if self.writer is None:
            self.writer = pq.ParquetWriter(self.log_file, self.schema)
        self.writer.write_table(table)
        print(f"Logged {len(prompts)} rollouts for step {metadata['step']}.")

def _null_ctx():
    from contextlib import nullcontext
    return nullcontext()

def format_tensor_log(tensor: torch.Tensor, precision: int = 2) -> str:
    """Formats a 1D tensor into a compact string like '[0.12 0.34 ...]'

    iterating a tensor yields 0-d tensors and formatting each one is a separate host read,
    so this is num_buckets device syncs if `tensor` is on the gpu. it is called every step.
    the bucket distribution is now kept on the host (see main(): the curriculum sampler's
    whole state lives there because scipy's brentq runs there), so this iterates cpu memory.
    the .tolist() makes that explicit and costs nothing when the tensor is already local.
    """
    return "[" + " ".join([f"{x:.{precision}f}" for x in tensor.tolist()]) + "]"


class DeferredScalarLog:
    """per-step logging scalars, read ONE STEP LATE, so the log line never stalls the pipe.

    `train_loss.item()` in a training loop is a full device drain per step: the host blocks
    until every queued kernel retires just to print four decimal places. the fix used by the
    modded-nanogpt lineage this trainer descends from is to copy the scalars asynchronously
    into pinned host memory and consume the PREVIOUS step's copy, which has had an entire
    training step's worth of queued work to land.

    consequence, stated plainly because it shows up in the logs: the numbers printed on the
    line labelled `step:N` are the numbers from step N-1. nothing that is COMPUTED changes --
    the loss tensors themselves are untouched and the optimizer never sees these values.
    the first logged line is skipped because there is no previous step to report.

    THE EVENT IS NOT OPTIONAL. `host.copy_(device_tensor, non_blocking=True)` is
    asynchronous, and pinned memory is exactly the case where torch will NOT insert an
    implicit sync for you. reading `host` a step later without waiting on anything is a
    data race: usually a step of queued work has covered it, and when it has not you print
    the step before last, or a torn mix of the two, and nothing anywhere says so. so the
    copy is followed by a recorded cuda Event and the read waits on it. the wait is
    essentially free precisely because the event is a full training step old -- that is the
    same argument that justifies deferring in the first place, and it is what makes the
    deferral correct rather than merely fast.

    keys are named rather than positional so that a caller logging a different SET of
    scalars (an extra objective term, say) cannot silently shift every column by one.
    """

    def __init__(self, keys, device):
        self.keys = list(keys)
        self.is_cuda = torch.device(device).type == "cuda"
        self.host = torch.zeros(len(self.keys), dtype=torch.float32,
                                pin_memory=self.is_cuda)
        self.event = torch.cuda.Event() if self.is_cuda else None
        self.pending_step = None

    def _read(self):
        if self.event is not None:
            self.event.synchronize()   # waits on a copy enqueued a full step ago
        return {k: float(self.host[i]) for i, k in enumerate(self.keys)}

    def push(self, step, scalars):
        """queue `step`'s scalars (name -> 0-d tensor or number); return the PREVIOUS
        submission as (step, {name: float}), or (None, None) the first time.

        one stack + one copy, rather than one small copy per scalar: N slot-wise
        device-to-device copies is N kernel launches to move N floats."""
        previous = (self.pending_step, self._read()) if self.pending_step is not None \
            else (None, None)
        stacked = torch.stack([
            (v.detach() if torch.is_tensor(v) else torch.tensor(float(v)))
            .to(torch.float32).reshape(())
            for v in (scalars[k] for k in self.keys)])
        self.host.copy_(stacked, non_blocking=self.is_cuda)
        if self.event is not None:
            self.event.record()
        self.pending_step = step
        return previous

    def drain(self):
        """the final submission is still in flight when the loop ends. without this the
        last step never gets logged at all."""
        if self.pending_step is None:
            return (None, None)
        out = (self.pending_step, self._read())
        self.pending_step = None
        return out

### modded-nanogpt
### either 24/16*20=30 batches per 4090 or 24/32*20=15 batches per 4090, 
### depending on what kind of v100 tinystories used. 
### stuff these w/:
### uv run loader.py --config_file configs/ascii_char_model.json
@dataclass
class Hyperparameters:
    # data hyperparams
    data_format : str = "bin" 
    input_bin : str = 'data/tinystories-pqt/tinystories-pqt_train_*.bin' # input .bin to train on
    input_val_bin : str = 'data/tinystories-pqt/tinystories-pqt_val_*.bin' # input .bin to eval validation loss on
    # those two trailing commas were load-bearing in the worst way: `x: str = "",` makes the
    # default the TUPLE ("",), not the empty string. audited the whole dataclass; these were
    # the only two, and test_hyperparameters_string_defaults_are_strings now checks all of it.
    input_npz: str = ""
    priority_scores_parquet: str = ""
    run_name : str = "re-pqt-rmsXrmsx3-ATTNII_fast"
    # optimization hyperparams
    batch_size : int = 4*64 # macrobatch size, in sequences, across all devices
    device_batch_size : int = 64 # batch size, in sequences, per device. try to increase/decrease by powers of 2
    sequence_length : int = 512 # sequence length, in tokens
    num_iterations : int = 4500 # number of iterations to run #target 8 hrs
    attack : int = 40 # 2*(1-betas)^-1
    release : int = 256 # number of iterations of linear warmup/warmdown for triangular or trapezoidal schedule
    weight_decay : float = 0
    # evaluation and logging hyperparams
    val_loss_every : int = 200 # every how many steps to evaluate val loss? 0 for only at the end
    val_tokens : int = 5242880 # how many tokens of validation data? it's important to keep this fixed for consistent comparisons
    save_every : int = 12500 # every how many steps to save the checkpoint? 0 for only at the end
    #btw rollout capture will cause cuda graph breaks 
    # so this requires writing attention masking and some other extra stuff to recover lost perf.
    capture_rollouts_every: int = 0  # 0 to disable, otherwise capture every N steps
    capture_rollout_prompts_file: str = "data/TinyStories-valid.txt" # Source for prompts
    capture_rollout_batch_size: int = 32 # How many rollouts to capture at once
    # supercompute boilerplate
    ddp_run : bool = False #this stuff is so nyannoying
    # these four had no type annotation, which in a dataclass means they are not fields at
    # all -- just class attributes. they were still settable and still read, but asdict()
    # skipped them, so the ACTIVE HYPERPARAMETERS block written into every logfile has never
    # recorded the device, whether the run was compiled, or the z-loss coefficient. annotated.
    device : str = "cuda" # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
    torch_compile : bool = True   #hahahaha
    use_z_loss : bool = True
    z_loss_coefficient : float = 1e-4
    # model arch boilerplate
    model_config: dict = field(default_factory=lambda: {
        #global hparams
        "vocab_size": 50304,    #magic number wrt one specific tokenizer
        "num_layers": 4,
        #layer hparams
        "dim": 768,
        "dim_head": 64,
        "headcount": 12,
        "ff_mult": 4,
        "lambda": True, # The key 'lambda' is perfectly fine in a dictionary
        "layerwisenorm": "rmsnorm",
        "qknorm": "dynamic_shape_rmsnorm",
        "is_t5": False, #default to autoregressive
        # was True here while pgptlformer's own cfg_flag(..., default=False) said False:
        # ONE key with TWO defaults, so a config file that omitted it got attention-II and
        # a dict built in code did not. aligned to the model's, which is the value that
        # decides what actually gets constructed. every file in configs/ now states the
        # key explicitly (MODEL_CONFIG_SCHEMA requires it), so this default is reachable
        # only by running loader.py with no --config_file at all.
        "attention_deux": False,
        "attention_deux_norm": "none",  # "none" | "mean" | "inv_sqrt_s", see readme
        "attn_gate": "none",            # "none" | "sigmoid", see readme
        "rotary_embedding_base": 1000,  # yes 1000. deliberate. see pgptlformer.py.
        # t5-only special ids. None here means "not a t5 config"; the t5 setup path below
        # raises if they are still None when is_t5 is true.
        "pad_token_id": None,
        "eos_token_id": None,
        "mask_token_start_id": None,
        "training_seqlen": 512 # This was hardcoded before, good to have it here
    })

# --- REVISED: Simplified config loading logic ---
def merge_config(args: "Hyperparameters", config_data: dict) -> "Hyperparameters":
    """merge a parsed json config into a Hyperparameters instance, loudly.

    the old version was:

        for key, value in config_data.items():
            if key in args.model_config:  args.model_config[key] = value
            elif hasattr(args, key):      setattr(args, key, value)
            else:                         print("WARNING: Unknown hyperparameter ...")

    three separate problems.
      1. the first branch matched TOP-LEVEL json keys against model_config's keys, so a
         top-level "vocab_size" would land in the model config and a nested one would not.
      2. `hasattr(args, "model_config")` is true, so the nested "model_config" dict from the
         json REPLACED the dataclass default wholesale. any key the json omitted did not
         fall back to the default -- it simply ceased to exist, and the model then read it
         with .get() and got None.
      3. a typo'd key printed a warning into a wall of startup spam and carried on training
         for eight hours with the setting you thought you had changed still at its default.
    """
    for key, value in config_data.items():
        # json has no comment syntax. '_'-prefixed keys are documentation -- a config that
        # cannot record WHY a value is what it is grows decisions nobody can reconstruct --
        # and they are dropped here rather than carried into the model.
        if is_doc_key(key):
            continue
        if key == "model_config":
            if not isinstance(value, dict):
                raise ConfigError(f"'model_config' must be an object, got {type(value).__name__}")
            unknown = sorted(k for k in set(value) - set(args.model_config)
                             if not is_doc_key(k))
            if unknown:
                raise ConfigError(
                    f"unknown model_config key(s) {unknown}. "
                    f"known keys: {sorted(args.model_config)}")
            args.model_config.update({k: v for k, v in value.items() if not is_doc_key(k)})
        elif key in {f.name for f in fields(args)}:
            setattr(args, key, value)
        else:
            raise ConfigError(
                f"unknown hyperparameter {key!r} in config file. "
                f"known keys: {sorted(f.name for f in fields(args))} + 'model_config'")

    # Ensure sequence_length is consistent between training params and model params
    args.model_config['training_seqlen'] = args.sequence_length
    # startup-time schema check on the MERGED config: types, allowed values, ranges, and
    # dim_head*headcount == dim, reported all at once. this runs once per process, before
    # any tensor exists. require_present=False because the dataclass has already supplied
    # every key -- "is it written in the FILE" is checked by tests/test_config_schema.py
    # against the file itself, which is the only place that question has an answer.
    validate_model_config(args.model_config, require_present=False)
    return args


def load_config(argv=None):
    parser = argparse.ArgumentParser(description="Train a PGPT-Lformer model.")
    parser.add_argument("--config_file", type=str, default="", help="Path to a JSON configuration file.")
    cli_args, _ = parser.parse_known_args(argv)

    args = Hyperparameters()

    config_path = cli_args.config_file or args.config_file
    if config_path:
        print(f"Loading configuration from: {config_path}")
        with open(config_path) as f:
            config_data = json.load(f)
        args = merge_config(args, config_data)
    else:
        args.model_config['training_seqlen'] = args.sequence_length

    return args



def _sync_if_cuda(device):
    """torch.cuda.synchronize(), but only where it means anything.

    every call to this is a full pipeline drain. all of them below sit on an INTERVAL
    boundary (checkpoint save, validation, start/end of the timed region) and exist to make
    the wall-clock measurement honest -- none of them are per-step. that is deliberate and
    it is the line: timing syncs are allowed on intervals, never in the inner loop.
    """
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize()


def build_optimizers(model, is_t5_model):
    # bitsandbytes is cuda-only, so it is imported here rather than at module scope. that
    # single import statement is the difference between "this file can be unit tested" and
    # "this file cannot be imported without a gpu".
    import bitsandbytes as bnb
    if is_t5_model:
        # modded-nanogpt optimizer inits
        adam1 = torch.optim.Adam([model.what_the_embedder_doin.weight], lr=0.3,    betas=(0.9, 0.95) )
        adam2 = torch.optim.Adam([model.tokenpicker_head.weight],       lr=0.002,  betas=(0.9, 0.95) )
        params = list(model.encoder.parameters()) + list(model.decoder.parameters())
    else:
        # modded-nanogpt optimizer inits
        adam1 = torch.optim.Adam([model.what_the_embedder_doin.weight], lr=0.3,    betas=(0.9, 0.95) )
        adam2 = torch.optim.Adam([model.tokenpicker_head.weight],       lr=0.002,  betas=(0.9, 0.95) )
        params = list(model.lambdaformer.blocks.parameters())
    matrix_params = [p for p in params if p.ndim == 2]
    scalar_params = [p for p in params if p.ndim < 2]
    adam3 = bnb.optim.Adam8bit(matrix_params, lr=0.02, betas=(0.9, 0.95) ) #tune this, sensitive
    adam4 = bnb.optim.Adam8bit(scalar_params, lr=0.02, betas=(0.9, 0.95) ) #???, less sensitive
    return [adam1, adam2, adam3, adam4]


def main(argv=None):
    with open(os.path.abspath(__file__)) as f:
        code = f.read() # read the code of this file, for logging
        # (this used to read sys.argv[0] at import time, which is only this file when the
        #  file is the entrypoint. under pytest it logged pytest's own source.)

    args = load_config(argv)

    # convenience variables
    B, T = args.device_batch_size, args.sequence_length

    # set up DDP (distributed data parallel). torchrun sets this env variable
    if args.ddp_run == True:
        assert torch.cuda.is_available(), "ddp_run requires cuda"
        dist.init_process_group(backend='nccl')
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        print(f"using device: {device}")
        master_process = (ddp_rank == 0) # this process will do logging, checkpointing etc.
    else:
        # if not ddp, we are running on a single gpu, and one process
        master_process = True
        ddp_local_rank = 0
        ddp_world_size = 1
        ddp_rank = 0
        device = args.device
    if torch.device(device).type == "cuda":
        assert torch.cuda.is_available(), f"device={device} but torch.cuda.is_available() is False"
    #tokens_per_iter = train_accumulation_steps * ddp_world_size * batch_size * block_size
    #print(f"tokens per iteration will be: {tokens_per_iter:,}")

    # calculate the number of steps to take in the val loop.
    assert args.val_tokens % (B * T * ddp_world_size) == 0
    val_steps = args.val_tokens // (B * T * ddp_world_size)
    # calculate the steps of gradient accumulation required to attain the desired global batch size.
    assert args.batch_size % (B * ddp_world_size) == 0
    train_accumulation_steps = args.batch_size // (B * ddp_world_size)

    # load tokens
    if args.data_format == "bin":
        if master_process:
            print(f"Using 'bin' data format from {args.input_bin}")
        train_loader = DistributedDataLoader(args.input_bin, B, T, ddp_rank, ddp_world_size, device=device)
        # Only create val_loader if val_tokens is specified
        val_loader = None
        if args.val_tokens > 0:
            val_loader = DistributedDataLoader(args.input_val_bin, B, T, ddp_rank, ddp_world_size, device=device)
        if master_process:
            print(f"Training DataLoader: total number of tokens: {train_loader.ntok_total} across {len(train_loader.files)} files")
            if val_loader is not None:
                # this used to dereference val_loader unconditionally, one line after the
                # branch that leaves it None when val_tokens == 0.
                print(f"Validation DataLoader: total number of tokens: {val_loader.ntok_total} across {len(val_loader.files)} files")
    elif args.data_format == "audio":
        if master_process:
            print(f"Using 'audio' data format from {args.input_npz}")
        train_loader = IntelligentAudioDataLoader(args.input_npz, args.priority_scores_parquet, B, T, ddp_rank, ddp_world_size, device=device)
        # For now, validation will just re-sample from the training distribution
        val_loader = train_loader
        # A proper val set would use a different npz/parquet file.
    else:
        raise ValueError(f"Unknown data_format: {args.data_format}. Must be 'bin' or 'audio'.")
    x, y = train_loader.next_batch()

    if master_process:
        print("Building model...")

    # --- REVISED: Model instantiation is now much cleaner ---
    # No more creating a config dict. We just pass the one from our args object.
    model = pgptlformer.PGPT_Lformer(args.model_config)
    if hasattr(tconfig, "coordinate_descent_tuning"):
        #torch._inductor.config as tconfig
        tconfig.coordinate_descent_tuning = True # suggested by @Chillee
    model = model.to(device)
    if args.torch_compile:
        model = torch.compile(model)

    is_t5_model = args.model_config["is_t5"]
    t5_processor = None
    curriculum_sampler = None
    if is_t5_model:
        if master_process:
            print("Model is in T5 mode. Initializing T5BatchProcessor.")
        # E.g., vocab_size = 256 for ASCII, pad=256, eos=257, mask_start=258
        pad_token_id = args.model_config['pad_token_id']
        eos_token_id = args.model_config['eos_token_id']
        mask_token_start_id = args.model_config['mask_token_start_id']
        vocab_size = args.model_config['vocab_size']

        if any(tid is None for tid in [pad_token_id, eos_token_id, mask_token_start_id]):
            raise ConfigError("When is_t5=True, the config file must specify 'pad_token_id', 'eos_token_id', and 'mask_token_start_id'.")

        t5_processor = T5BatchProcessor(
            mask_token_start_id=mask_token_start_id,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            vocab_size=vocab_size
        )
        curriculum_sampler = AdaptiveCurriculumSampler()

    # here we wrap model into DDP container
    if args.ddp_run:
        model = DDP(model, device_ids=[ddp_local_rank])
    #raw_model = model.modules() # always contains the "raw" unwrapped model
    device_type = torch.device(device).type
    ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=(device_type == "cuda"))

    if master_process:
        print("Model built.")

    # CUDNN attention is ~4ms faster than Flash, but doesn't get selected by default in PyTorch 2.5.1
    if device_type == "cuda":
        from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp
        enable_cudnn_sdp(True)
        enable_flash_sdp(True)
        enable_mem_efficient_sdp(True)
        enable_math_sdp(False)

    optim_ensemble = build_optimizers(model, is_t5_model)

    # lr scheduler
    def get_ASR_env(it):
        assert it <= args.num_iterations
        # A) the famous linear warmup back at it again
        if it < args.attack:
            return (it+1) / args.attack
        # S) constant sustain
        elif it < args.num_iterations - args.release:
            return 1.0
        # R) release
        else:
            release_ratio = (args.num_iterations - it) / args.release
            return release_ratio

    schedulers = [torch.optim.lr_scheduler.LambdaLR(opt, get_ASR_env) for opt in optim_ensemble]

    # begin logging
    logfile = None
    run_id = None
    tokenizer = None
    prompt_generator = None
    rollout_sampler = None
    if master_process:
        run_id = str(uuid.uuid4())
        if args.run_name is not None:
            sep="-"
            run_id = sep.join([args.run_name, run_id])

        logdir = 'logs/%s/' % run_id
        os.makedirs(logdir, exist_ok=True)
        logfile = 'logs/%s.txt' % run_id
        # create the log file
        if args.capture_rollouts_every > 0:
            print("Scroingling eval toingos. Kindly wait.")
            # NOTE: You'll need a tokenizer instance. We use the ASCII one from ascii_tokenizer.py
            # For a real run, this should be the same tokenizer used for training.
            from ascii_tokenizer import SimpleASCIITokenizer # Example, adjust as needed
            tokenizer = SimpleASCIITokenizer()

            prompt_generator = PromptGenerator(args.capture_rollout_prompts_file)
            rollout_sampler = OnlineRolloutSampler(logdir, run_id) # Log to the same run directory
            print("Toingles scroingled.")
        with open(logfile, "w") as f:
            # begin the log by printing this file (the Python code)
            f.write('='*100 + '\n')
            f.write(code)
            f.write('='*100 + '\n')
            # Log the final, active hyperparameters. asdict() handles the nested dict perfectly.
            f.write("ACTIVE HYPERPARAMETERS:\n")
            f.write(json.dumps(asdict(args), indent=4))
            f.write('\n' + '='*100 + '\n')
            # log information about the hardware/software environment this is running on
            # and print the full `nvidia-smi` to file
            f.write(f"Running pytorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}\nnvidia-smi:\n")
            import subprocess
            try:
                result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                f.write(f'{result.stdout}\n')
            except FileNotFoundError:
                f.write('nvidia-smi not present on this host\n')
            f.write('='*100 + '\n')

    training_time_ms = 0
    # start the clock
    _sync_if_cuda(device)
    t0 = time.time()
    # begin training
    train_loader.reset()

    # --- device-sync accounting for the logging path -------------------------------------
    # train_loss / aux_loss used to be .item()'d on the step that produced them, which is a
    # full device drain per step for the sake of a print statement. they are now copied
    # asynchronously into pinned host memory and read back one step late. see DeferredScalarLog.
    LOG_KEYS = ("train_loss", "aux_loss")
    step_log = DeferredScalarLog(LOG_KEYS, device)
    pending_log = None   # (step, aug_log) awaiting the scalars queued on the previous step

    for step in range(args.num_iterations + 1):
        last_step = (step == args.num_iterations)
        # This effectively ignores timing first 10 steps, which are slower for weird reasons.
        # Alternately, and slightly more correctly in terms of benchmarking, we could do 10
        # steps with dummy data first, and then re-initialize the model and reset the loader.
        if step == 10:
            training_time_ms = 0
            t0 = time.time()
        timed_steps = float('nan') if step <= 11 else (step - 10) + 1 # <= 11 to avoid bug in val

        if master_process and (last_step or (args.save_every != 0 and step % args.save_every == 0)):
            # stop the clock
            _sync_if_cuda(device)
            training_time_ms += 1000 * (time.time() - t0)
            # save the state of the training process
            log = dict(step=step, code=code, model=model.state_dict(), model_args=args.model_config, optim_ensemble=[opt.state_dict() for opt in optim_ensemble])
            torch.save(log, 'logs/%s/state_step%06d.pt' % (run_id, step))
            # start the clock again
            _sync_if_cuda(device)
            t0 = time.time()

        # once in a while evaluate the validation dataset
        if ((last_step and val_steps > 0) or (args.val_loss_every > 0 and step % args.val_loss_every == 0)):
            # stop the clock
            _sync_if_cuda(device)
            training_time_ms += 1000 * (time.time() - t0)
            # run validation batches
            model.eval()
            if val_loader is None:
                if master_process:
                    print("something weird about your validation loader happened. like it didn't exist?\nskipping validation~!")
            else:
                val_loader.reset()
                val_loss = torch.zeros((), device=device)
                val_aux_loss = torch.zeros((), device=device)
                for _ in range(val_steps):
                    x_val_continuous, y_val_continuous = val_loader.next_batch()
                    # --- Apply same conditional logic as in training ---
                    if is_t5_model:
                        masked_inputs, decoder_inputs, target_labels, \
                        encoder_mask, decoder_mask = t5_processor(x_val_continuous, avg_span_length=3, mask_prob=0.15)
                        model_args = {
                            "input_ids": masked_inputs.to(device), "decoder_input_ids": decoder_inputs.to(device),
                            "targets": target_labels.to(device), "encoder_padding_mask": encoder_mask.to(device),
                            "decoder_padding_mask": decoder_mask.to(device)
                        }
                    else: # Autoregressive mode
                        padding_mask = torch.ones_like(x_val_continuous, dtype=torch.bool)
                        model_args = {
                            "input_ids": x_val_continuous.to(device), "targets": y_val_continuous.to(device),
                            "padding_mask": padding_mask.to(device)
                        }
                    with ctx:
                        # arity-proof: the val loop used to 4-unpack, which is exactly the
                        # coupling that killed the samplers.
                        outputs = model(**model_args, return_logits=False, return_zloss=args.use_z_loss)
                        loss, z_loss = outputs[1], outputs[2]
                        # accumulate ON DEVICE. no .item() anywhere in the val loop.
                        val_loss += loss.detach()
                        if z_loss is not None:
                            val_aux_loss += z_loss.detach()*args.z_loss_coefficient
                if args.ddp_run:
                    dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
                    dist.all_reduce(val_aux_loss, op=dist.ReduceOp.AVG)
                val_loss /= val_steps
                val_aux_loss /= val_steps
                # log val loss to console and to logfile
                if master_process:
                    # ONE host transfer for the whole validation pass, and it is on the
                    # validation interval by construction.
                    v_loss, v_aux = torch.stack([val_loss, val_aux_loss]).cpu().tolist()
                    line = (f'step:{step}/{args.num_iterations} val_loss:{v_loss:.4f} '
                            f'val_aux_loss:{v_aux:.4f} train_time:{training_time_ms:.0f}ms '
                            f'step_avg:{training_time_ms/(timed_steps-1):.2f}ms')
                    print(line)
                    with open(logfile, "a") as f:
                        f.write(line + '\n')
            # start the clock again
            _sync_if_cuda(device)
            t0 = time.time()

        # every once in a (probably longer) while sample autoregressive rollouts from model
        if master_process and (args.capture_rollouts_every > 0 and (last_step or step % args.capture_rollouts_every == 0)):
            if is_t5_model:
                # this used to be `pass`, not a skip. it printed "Skipping online rollouts"
                # and then fell straight through into the autoregressive rollout code it had
                # just announced it was skipping.
                print("\n--- Skipping online rollouts: T5 model requires a different generation method ---")
            else:
                print("\n--- Capturing online rollouts ---")
                prompts = prompt_generator.get_prompts(args.capture_rollout_batch_size)

                metadata = {
                    'step': step,
                    'timestamp': datetime.now(),
                    'hyperparameters': asdict(args)
                }

                # The rollout length will be context length - prompt length
                max_new = args.sequence_length - 32 # Assuming 32-char prompts

                rollout_sampler.capture_and_log(
                    model.module if args.ddp_run else model, # unwrap DDP model
                    prompts,
                    tokenizer,
                    metadata,
                    max_new_tokens=max_new,
                    device=device,
                    autocast_ctx=ctx,
                )
                print("--- Finished capturing rollouts ---\n")

        # bit confusing: we want to make sure to eval on 0th iteration
        # but also after the very last iteration. so we loop for step <= num_iterations
        # instead of just < num_iterations (one extra due to <=), only to do
        # the validation/sampling one last time, and then we break right here as we're done.
        if last_step:
            break

        # NEW: compute task distribution per accumulated batch, not per microbatch
        bucket_distribution = None
        crsm_diag = None
        if is_t5_model:
            # the curriculum sampler's ENTIRE state is host-side: its ema tensors are cpu
            # tensors and get_distribution() runs scipy's brentq on numpy. asking for the
            # distribution on `device` only to (a) run one multinomial and (b) format it into
            # a log line meant shipping it across the bus and then reading it back a scalar
            # at a time. keep it where it already lives.
            crsm_diag = curriculum_sampler.get_distribution(device='cpu')
            bucket_distribution = crsm_diag['p_final']

        # --- train time ---
        model.train()
        # per-micro-step curriculum observations, kept ON DEVICE and drained exactly once
        # per optimizer step (see below). the old code did
        #     for i in range(loss_per_seq.size(0)):
        #         curriculum_sampler.update(bucket_indices[i].item(), loss_per_seq[i].item())
        # which is 2*device_batch_size host transfers PER MICRO-STEP -- 256 pipeline drains
        # per optimizer step at B=32, accum=4. it also reused the name `i`, clobbering the
        # gradient-accumulation counter (see below).
        pending_curriculum = []
        for micro_step in range(1, train_accumulation_steps+1):
            # batch item construction (if you're reviewing this code make this 10x faster ;) )
            if is_t5_model:
                # Process the continuous batch into a T5 objective
                # this is a very 176k token gem2.5 way of passing variables lol but whatever
                masked_inputs, decoder_inputs, target_labels, \
                encoder_mask, decoder_mask, bucket_indices = t5_processor.create_curriculum_batch(
                    x, bucket_distribution, curriculum_sampler)

                # Prepare keyword arguments for the model
                model_args = {
                    "input_ids": masked_inputs.to(device),
                    "decoder_input_ids": decoder_inputs.to(device),
                    "targets": target_labels.to(device),
                    "encoder_padding_mask": encoder_mask.to(device),
                    "decoder_padding_mask": decoder_mask.to(device)
                }
            else: # Autoregressive mode
                # Create a simple padding mask (assuming 0 is padding for AR, or none is used)
                # In your case, the .bin files are unpadded streams, so the mask is all ones.
                padding_mask = torch.ones_like(x, dtype=torch.bool)
                # Prepare keyword arguments for the model
                model_args = {
                    "input_ids": x.to(device),
                    "targets": y.to(device),
                    "padding_mask": padding_mask.to(device)
                }

            # network forward()
            with ctx:
                outputs = model(**model_args, return_logits=False, return_zloss=args.use_z_loss)
                loss, z_loss, loss_per_seq = outputs[1], outputs[2], outputs[3]
                train_loss = loss.detach()
                # loss_per_seq is literally already mean reduced along non-skipped indices, ergo
                # loss_per_seq is of shape [B]
                # we may now pass that right into curriculum_sampler; but NOT one scalar at a
                # time, and NOT from inside the autocast region.
                if is_t5_model:
                    pending_curriculum.append((bucket_indices, loss_per_seq.detach()))
                if z_loss is not None:
                    train_aux_loss = z_loss.detach()*args.z_loss_coefficient
                    loss = loss+z_loss*args.z_loss_coefficient
                else:
                    train_aux_loss = torch.zeros((), device=device)
            # backward pass
            if args.ddp_run and micro_step < train_accumulation_steps:
                with model.no_sync(): # there's no need to sync gradients every accumulation step
                    loss.backward()
            else:
                # this `else` did not exist. under ddp, the FINAL micro-step -- the one that
                # is supposed to trigger the gradient all-reduce -- never called backward()
                # at all, so every ddp run silently dropped 1/train_accumulation_steps of
                # its gradient and never synced. single-gpu runs were unaffected, which is
                # why it survived: nobody here has been running ddp.
                loss.backward() # just sync on the last step
            x, y = train_loader.next_batch()

        # ONE host transfer per optimizer step for the whole curriculum update, replaying the
        # updates in exactly the order the old per-item loop applied them. the sampler state
        # is only READ by get_distribution() at the top of the step, so moving the writes to
        # the end of the accumulation loop is state-identical -- same values, same order,
        # same count. this transfer is deliberate and irreducible: AdaptiveCurriculumSampler
        # runs scipy.optimize.root_scalar (brentq) on the host, so the losses have to land on
        # the host eventually. once per step, not once per sequence.
        if is_t5_model and pending_curriculum:
            all_buckets = torch.cat([b for b, _ in pending_curriculum]).tolist()
            all_losses = torch.cat([l.float() for _, l in pending_curriculum]).cpu().tolist()
            for bucket_idx, item_loss in zip(all_buckets, all_losses):
                curriculum_sampler.update(bucket_idx, item_loss)

        for p in model.parameters():    #grad accum normalization?
            if p.grad is not None:
                p.grad /= train_accumulation_steps
        # skip muon momentum warmup since we're adaming it
        #...
        # step the optimizers and schedulers
        for opt, sched in zip(optim_ensemble, schedulers):
            opt.step()
            sched.step()
        # null the gradients
        model.zero_grad(set_to_none=True)
        # --- train time is already over ---

        # logging
        #dist.all_reduce(train_loss, op=dist.ReduceOp.AVG) # all-reducing the training loss would be more correct in terms of logging, but slower
        if master_process:
            aug_log = ""
            if is_t5_model:
                # -- Tier 1: Concise `stdout` log --
                # all three of these are already host-side tensors (see get_distribution
                # above), so none of them sync.
                tgt_l = crsm_diag['target_loss'].item()
                exp_l = crsm_diag['expected_loss'].item()
                lmbda = crsm_diag['lambda']
                p_final_str = format_tensor_log(bucket_distribution, precision=2)
                aug_log += f"| crclm: Tgt_L:{tgt_l:.2f} Exp_L:{exp_l:.2f} λ:{lmbda:.2f} Dist:{p_final_str}"

            # queue this step's scalars, get the previous step's back.
            _, previous = step_log.push(step, {"train_loss": train_loss,
                                               "aux_loss": train_aux_loss})
            if pending_log is not None and previous is not None:
                prev_step, prev_aug, prev_time = pending_log
                prev_loss, prev_aux = previous["train_loss"], previous["aux_loss"]
                ultimate_log = (f"step:{prev_step+1}/{args.num_iterations} "
                                f"train_loss:{prev_loss:.4f} aux_loss:{prev_aux:.4f} "
                                f"train_time:{prev_time:.0f}ms "
                                f"step_avg:{prev_time/(float('nan') if prev_step <= 11 else (prev_step-10)+1):.2f}ms"
                                + prev_aug)
                print(ultimate_log)
                with open(logfile, "a") as f:
                    f.write(ultimate_log+"\n")
            pending_log = (step, aug_log, training_time_ms + 1000 * (time.time() - t0))

    if master_process:
        # drain the one deferred log line still in flight. this used to push a fake
        # (0.0, 0.0) record just to shift the pipeline; drain() reads it directly.
        if pending_log is not None:
            _, previous = step_log.drain()
            if previous is not None:
                prev_step, prev_aug, prev_time = pending_log
                prev_loss, prev_aux = previous["train_loss"], previous["aux_loss"]
                ultimate_log = (f"step:{prev_step+1}/{args.num_iterations} "
                                f"train_loss:{prev_loss:.4f} aux_loss:{prev_aux:.4f} "
                                f"train_time:{prev_time:.0f}ms" + prev_aug)
                print(ultimate_log)
                with open(logfile, "a") as f:
                    f.write(ultimate_log+"\n")
        if torch.device(device).type == "cuda":
            print(f"peak memory consumption: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB")

    # clean up nice
    if args.ddp_run:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
