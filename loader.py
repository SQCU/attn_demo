#loader.py
### canonically 
### torchrun --standalone --nproc_per_node=8 loader.py
### but for us, probably 
### set USE_LIBUV=0
### set RANK 
### set TORCH_CUDNN_SDPA_ENABLED=1
### torchrun --standalone --nproc_per_node=1 loader.py
import os
import sys
with open(sys.argv[0]) as f:
    code = f.read() # read the code of this file ASAP, for logging
import uuid
import glob
import time
from dataclasses import dataclass

import numpy as np
import torch
import bitsandbytes as bnb
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
import torch._inductor.config as config
from torch.nn.parallel import DistributedDataParallel as DDP

import pgptlformer

#wacky env stuff:
#import tritonpathsetter
#tritonpathsetter.set_cuda_paths()
#tritonpathsetter.add_cuda_files()

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
    def __init__(self, filename_pattern, B, T, process_rank, num_processes):
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.B = B
        self.T = T

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
        return x.cuda(), y.cuda()

# -----------------------------------------------------------------------------

# downgrade to poor man's data loader:
# maybe superfluous bc distributed data loader started working
# delete? [ ]
def get_batch(split):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    block_size = args.sequence_length
    batch_size = args.batch_size
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

### modded-nanogpt
### either 24/16*20=30 batches per 4090 or 24/32*20=15 batches per 4090, 
### depending on what kind of v100 tinystories used. 
@dataclass
class Hyperparameters:
    # data hyperparams
    input_bin : str = 'data/tinystories-pqt/tinystories-pqt_train_*.bin' # input .bin to train on
    input_val_bin : str = 'data/tinystories-pqt/tinystories-pqt_val_*.bin' # input .bin to eval validation loss on
    # optimization hyperparams
    batch_size : int = 4*64 # macrobatch size, in sequences, across all devices
    device_batch_size : int = 64 # batch size, in sequences, per device. try to increase/decrease by powers of 2
    sequence_length : int = 512 # sequence length, in tokens
    num_iterations : int = 40500 # number of iterations to run #target 8 hrs
    attack : int = 40 # 2*(1-betas)^-1
    release : int = 256 # number of iterations of linear warmup/warmdown for triangular or trapezoidal schedule
    weight_decay : float = 0
    # evaluation and logging hyperparams
    val_loss_every : int = 2000 # every how many steps to evaluate val loss? 0 for only at the end
    val_tokens : int = 5242880 # how many tokens of validation data? it's important to keep this fixed for consistent comparisons
    save_every : int = 12500 # every how many steps to save the checkpoint? 0 for only at the end
    run_name : str = "re-pqt-rmsXrmsx3-ATTNII_fast"
    # supercompute boilerplate
    ddp_run : bool = False #this stuff is so nyannoying
    device = "cuda" # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
    torch_compile = True   #hahahaha
    use_z_loss = True
    z_loss_coefficient = 1e-4
args = Hyperparameters()

# convenience variables
B, T = args.device_batch_size, args.sequence_length
# set up DDP (distributed data parallel). torchrun sets this env variable
assert torch.cuda.is_available()

if args.ddp_run == True:
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
    seed_offset = 0
    ddp_world_size = 1
    ddp_rank = 0
    device = args.device
#tokens_per_iter = train_accumulation_steps * ddp_world_size * batch_size * block_size
#print(f"tokens per iteration will be: {tokens_per_iter:,}")


# calculate the number of steps to take in the val loop.
assert args.val_tokens % (B * T * ddp_world_size) == 0
val_steps = args.val_tokens // (B * T * ddp_world_size)
# calculate the steps of gradient accumulation required to attain the desired global batch size.
assert args.batch_size % (B * ddp_world_size) == 0
train_accumulation_steps = args.batch_size // (B * ddp_world_size)

# load tokens 
train_loader = DistributedDataLoader(args.input_bin, B, T, ddp_rank, ddp_world_size)
val_loader = DistributedDataLoader(args.input_val_bin, B, T, ddp_rank, ddp_world_size)
if master_process:
    print(f"Training DataLoader: total number of tokens: {train_loader.ntok_total} across {len(train_loader.files)} files")
    print(f"Validation DataLoader: total number of tokens: {val_loader.ntok_total} across {len(val_loader.files)} files")
x, y = train_loader.next_batch()

if master_process:
    print("Building model...")

#tinystories
#num_vocab=50304 for non-tinystories models
#qknorm="identitynorm" for nonqknorm models
layer_prefab = {"dim":768,"dim_head":64,"headcount":12,"ff_mult":4, 
"lambda":True,"layerwisenorm":"rmsnorm","qknorm":"dynamic_shape_rmsnorm", 
"attention_deux":True, "training_seqlen":args.sequence_length}
#global_prefab = {"vocab_size":8192, "num_layers":4}
#weird errors
global_prefab = {"vocab_size":50304, "num_layers":4}
config = {}
config.update(layer_prefab)
config.update(global_prefab)

model = pgptlformer.PGPT_Lformer(config)
if hasattr(config, "coordinate_descent_tuning"):
    config.coordinate_descent_tuning = True # suggested by @Chillee
model = model.to(device)
if args.torch_compile:
    model = torch.compile(model)

# here we wrap model into DDP container
if args.ddp_run:
    model = DDP(model, device_ids=[ddp_local_rank])
#raw_model = model.modules() # always contains the "raw" unwrapped model
ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)

if master_process:
    print("Model built.")

# CUDNN attention is ~4ms faster than Flash, but doesn't get selected by default in PyTorch 2.5.1
from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp
enable_cudnn_sdp(True)
enable_flash_sdp(True)
enable_mem_efficient_sdp(True)
enable_math_sdp(False)

# modded-nanogpt optimizer inits
adam1 = torch.optim.Adam([model.lambdaformer.what_the_embedder_doin.weight], lr=0.3,    betas=(0.9, 0.95) )
adam2 = torch.optim.Adam([model.tokenpicker_head.weight],                    lr=0.002,  betas=(0.9, 0.95) )
params = list(model.lambdaformer.blocks.parameters())
matrix_params = [p for p in params if p.ndim == 2]
scalar_params = [p for p in params if p.ndim < 2]
adam3 = bnb.optim.Adam8bit(matrix_params, lr=0.02, betas=(0.9, 0.95) ) #tune this, sensitive
adam4 = bnb.optim.Adam8bit(scalar_params, lr=0.02, betas=(0.9, 0.95) ) #???, less sensitive
optim_ensemble = [adam1, adam2, adam3, adam4]

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
if master_process:
    run_id = str(uuid.uuid4())
    if args.run_name is not None:
        sep="-"
        run_id = sep.join([args.run_name, run_id])
    
    logdir = 'logs/%s/' % run_id
    os.makedirs(logdir, exist_ok=True)
    logfile = 'logs/%s.txt' % run_id
    # create the log file
    with open(logfile, "w") as f:
        # begin the log by printing this file (the Python code)
        f.write('='*100 + '\n')
        f.write(code)
        f.write('='*100 + '\n')
        # log information about the hardware/software environment this is running on
        # and print the full `nvidia-smi` to file
        f.write(f"Running pytorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}\nnvidia-smi:\n")
        import subprocess
        result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        f.write(f'{result.stdout}\n')
        f.write('='*100 + '\n')

training_time_ms = 0
# start the clock
torch.cuda.synchronize()
t0 = time.time()
# begin training
train_loader.reset()

for step in range(args.num_iterations + 1):
    last_step = (step == args.num_iterations)
    # This effectively ignores timing first 10 steps, which are slower for weird reasons.
    # Alternately, and slightly more correctly in terms of benchmarking, we could do 10
    # steps with dummy data first, and then re-initialize the model and reset the loader.
    if step == 10:
        training_time_ms = 0
        t0 = time.time()
    timed_steps = float('nan') if step <= 11 else (step - 10) + 1 # <= 11 to avoid bug in val

    # once in a while evaluate the validation dataset
    if (last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0)):
        # stop the clock
        torch.cuda.synchronize()
        training_time_ms += 1000 * (time.time() - t0)
        # run validation batches
        model.eval()
        val_loader.reset()
        val_loss = 0.0
        val_aux_loss = 0.0
        for _ in range(val_steps):
            x_val, y_val = val_loader.next_batch()
            with ctx: # of course, we'd like to use no_grad() here too, but that creates a torch.compile error for some reason
                _, loss, z_loss = model(x_val, y_val, return_logits=False, return_zloss=args.use_z_loss)
                val_loss += loss.detach()
                if z_loss is not None:
                    val_aux_loss += z_loss.detach()*args.z_loss_coefficient
                del loss, z_loss
        if args.ddp_run:
            dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
            dist.all_reduce(val_aux_loss, op=dist.ReduceOp.AVG)
        val_loss /= val_steps
        val_aux_loss /= val_steps
        # log val loss to console and to logfile
        if master_process:
            print(f'step:{step}/{args.num_iterations} val_loss:{val_loss:.4f} val_aux_loss:{val_aux_loss:.4f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/(timed_steps-1):.2f}ms')
            with open(logfile, "a") as f:
                f.write(f'step:{step}/{args.num_iterations} val_loss:{val_loss:.4f} val_aux_loss:{val_aux_loss:.4f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/(timed_steps-1):.2f}ms\n')
        # start the clock again
        torch.cuda.synchronize()
        t0 = time.time()

        if master_process and (last_step or (args.save_every != 0 and step % args.save_every == 0)):
            # stop the clock
            torch.cuda.synchronize()
            training_time_ms += 1000 * (time.time() - t0)
            # save the state of the training process
            log = dict(step=step, code=code, model=model.state_dict(), model_args=config, optim_ensemble=[opt.state_dict() for opt in optim_ensemble])
            torch.save(log, 'logs/%s/state_step%06d.pt' % (run_id, step))
            # start the clock again
            torch.cuda.synchronize()
            t0 = time.time()

    # bit confusing: we want to make sure to eval on 0th iteration
    # but also after the very last iteration. so we loop for step <= num_iterations
    # instead of just < num_iterations (one extra due to <=), only to do
    # the validation/sampling one last time, and then we break right here as we're done.
    if last_step:
        break

    # --- train time ---
    model.train()
    for i in range(1, train_accumulation_steps+1):
        # forward pass
        with ctx:
            _, loss, z_loss = model(x, y, return_logits=False, return_zloss=args.use_z_loss)
            train_loss = loss.detach()
            if z_loss is not None:
                train_aux_loss = z_loss.detach()*args.z_loss_coefficient
                loss = loss+z_loss*args.z_loss_coefficient
            else:
                train_aux_loss = 0
        # advance the dataset for the next batch
        x, y = train_loader.next_batch()
        # backward pass
        if args.ddp_run:
            if i < train_accumulation_steps:
                with model.no_sync(): # there's no need to sync gradients every accumulation step
                    loss.backward()
        else:
            loss.backward() # just sync on the last step

    for p in model.parameters():    #grad accum normalization?
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

     #dist.all_reduce(train_loss, op=dist.ReduceOp.AVG) # all-reducing the training loss would be more correct in terms of logging, but slower
    if master_process:
        approx_time = training_time_ms + 1000 * (time.time() - t0)
        print(f"step:{step+1}/{args.num_iterations} train_loss:{train_loss.item():.4f} aux_loss:{train_aux_loss.item():.4f} train_time:{approx_time:.0f}ms step_avg:{approx_time/timed_steps:.2f}ms")
        with open(logfile, "a") as f:
            f.write(f"step:{step+1}/{args.num_iterations} train_loss:{train_loss.item():.4f} aux_loss:{train_aux_loss.item():.4f} train_time:{approx_time:.0f}ms step_avg:{approx_time/timed_steps:.2f}ms\n")
if master_process:
    print(f"peak memory consumption: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB")

# clean up nice
if args.ddp_run:
    dist.destroy_process_group()