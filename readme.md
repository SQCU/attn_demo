# attn_demo

## forenote:
this is the fourth major patch or so!
we went from 'something like a nanogpt' to 'something a lot weirder than a nanogpt' to 'what's the best way to sample from raw audio streams to train a model to play drum and bass on a thirty minute deadline?'. 
there are *many* nonstandard tools and assumptions in these architecture studies. most of them are immediately and contextually validated by using them. others are strange enough that they need some explanation.
1: attention-II is strange. if this was work done for an employer or a research lab, you would see a report before code.
instead, code before report. such is the way of the world.

2: the 'palm_network_architecture-gpt_loss-lambda_resnet_attenuation' pgptl initialization no longer traces most of the weirder features. linear projections aren't optimized to run faster because of it, but the network design uses parallel network subunits (self-attention, cross-attention, FFNs) to reduce the number of layernorms per 'layer' in a model. does this mean that at maximum tensor parallelism, these networks could run 3x faster per layer than a serial (s-attn,x-attn,ffn) t5 encoder-decoder model? yes! get in touch if you want to experiment with weird hyperscaling architectures like that. or if you want to float me hardware to try it and find out what can be done with those latency budgets.

3: also there's unusual training features, like options for online capture of rollouts of text (not single tokens for 'tokenwise eval loss') during generative pretraining of boring old generative text models. this is very similar to what you'd expect from a real training environment like primeintellect verifiers. however, this is a pedagogical / research training environment, so this code path exists mostly to do statistics supporting like, 'scaling' studies or hyperparameter sweeps on neural networks nobody has ever trained or sampled from before. it probably wouldn't be that hard to extend this for RL!

a closing remark: these models have been tested, and actually work! they have also been tested and found to be compilable. compiling makes the models run good. you can train more model per second, (or serve more batches of user inputs per second) if you compile your models good. get in touch (you can find this account's handle on the search engines) if you are interested in hosting such models but do not instantly figure out your integrations from pure gnosis and skimming provided code.

## Getting Started

### Installation

It is recommended to use a virtual environment. `uv` is a fast and effective choice.

```
# Using uv
uv sync
```
just kidding!
```
#using uv in a cuda context
uv venv --seed
uv sync --with cuda
uv add flash-attn==2.7.2.post1 --no-build-isolation
#using uv in a cudaless context
uv venv --seed
uv sync
```
#### so you're running this on primeintellect for a live event...?
# in this demonstration i leak how i name my keys:
ssh -i ~/.ssh/primeintellect_ed25519 {user@node_ip} -p 22

#first: tmux or smth
tmux
# CONTROL+B %
#	this induces a vertical diptych
# thereon CONTROL+B left/right/up/down for terminal selection whatever. it's a modal interface.
# thereon CONTROL+B x to close the selected pane

# remote persistent t5 sequence->sequence inference server:
# ...write your remote node IP address INTO the project root config.yaml
# then transfer models, script, etc.
rsync -avz -e "ssh -i ~/.ssh/primeintellect_ed25519 -p 22" --exclude .git --exclude .venv mnt/c/dox/ai/attn_demo {user@node_ip}:~
# naively this can take a long time. like uh. well. it can take a while.

# redis support
sudo apt-get update
sudo apt-get install redis-server
sudo systemctl start redis-server
sudo systemctl status redis-server
# inferenceserv support
cd attn_demo
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --seed
uv sync --extra cuda
uv add flash-attn==2.7.2.post1 --no-build-isolation

# pane up server
    redis-cli ping (Make sure Redis is running; it should return PONG).
    Pane 1: uv run t5_service.py
    Pane 2: uv run encodec_service.py
    Pane 3: uv run serve_audio.py
# pane up client
	Terminal A: ssh -i ~/.ssh/primeintellect_ed25519 -L 6379:localhost:6379 {user@node_ip}
	Terminal B: uv run local_client.py fetch (to listen for results)
    Terminal C: uv run local_client.py monitor (to check the system)
    Terminal C: uv run local_client.py submit --seed 1337 (to kick off a job)


```

# Alternatively, using pip
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r reqqies.txt
# You may need to install bitsandbytes separately
pip install bitsandbytes
```
For GPU-accelerated attention, you can try installing Flash Attention. This can be time-consuming to compile.
```bash
# This can take a very long time (>1 hour on hobbyist computers)
uv add flash-attn --no-build-isolation
```

### Dry Run: Training a Text Model

1.  **Download a dataset**, such as [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories/), and place the text files in the `./data/` directory.

2.  **Prepare the data.** This will tokenize the raw text and create efficient binary shards.
    ```bash
    cd data
    python prepare.py --trainfile TinyStoriesV2-GPT4-train.txt --valfile TinyStoriesV2-GPT4-valid.txt -p tinystories-gpt4
    cd ..
    ```

3.  **Train the model.** The default `loader.py` script is configured for this task.
    ```bash
    # Examine the source and comments of 'loader.py' first!
    uv run python loader.py
    ```

### Sampling from a Trained Model

You can generate text from your trained model using `sample.py`.

You will need to edit the simplistic hardcoded checkpoint path in `sample.py` to match your trained `pgpt_lformer` checkpoint. Edit the `prompt.txt` file to change the sampling cue. `sample.py` is more of an existence proof than a tool; if you can read this sentence, you can write a better one!

## Data Flow & Project Workflows

Understanding the flow of data from raw files to model outputs is key. Below are the primary workflows supported by this project.

### 1. Training & Sampling an Autoregressive Text Model (GPT-style)

This is the classic workflow for training a model to predict the next token.

#### Training Data Flow

```
1. Raw Text Files (.txt)
     │
     └─> data/prepare.py
           │
           ├─ Tokenizes text using tiktoken (GPT-2)
           └─ Creates sharded, memory-mapped binary files
     │
     └─> Sharded Dataset (.bin)
           │
           └─> loader.py (in autoregressive mode)
                 │
                 ├─ DistributedDataLoader reads token chunks
                 └─ Trains the PGPT-Lformer model
```

#### Inference Data Flow

```
1. Trained Model Checkpoint (.pt)
     │
     └─> sample.py
           │
           ├─ Loads the model and a text prompt
           └─ Generates a continuation autoregressively
     │
     └─> Generated Text (console output)
```

### 2. Training & Sampling a T5 Model on Neural Audio

This is the most complex and powerful workflow, designed for generative audio tasks.

#### Training Data Flow

This is a two-phase process to create a structurally-aware dataset.

```
PHASE 1: Tokenization
1. Raw Audio File (.wav, .mp3, etc.)
     │
     └─> encodec_index.py
           │
           ├─ Encodes audio into discrete neural codes (multi-stream)
           └─ Saves as a compressed NumPy array
     │
     └─> Tokenized Audio Artifact (.npz)

PHASE 2: Structural Analysis & Scoring
2. Tokenized Audio (.npz) + Raw Audio (.wav)
     │
     └─> mformer_dataset.py
           │
           ├─ Performs spectral analysis on the raw audio to get features (flux, RMS)
           ├─ Computes high-level signals (novelty, stability) from features
           └─ Generates a priority score for each audio chunk
     │
     └─> Priority Scores (.parquet)

PHASE 3: Training
3. Tokenized Audio (.npz) + Priority Scores (.parquet)
     │
     └─> loader.py (in T5 audio mode)
           │
           ├─ IntelligentAudioDataLoader samples high-priority audio chunks
           ├─ T5BatchProcessor creates denoising tasks (masked spans)
           ├─ AdaptiveCurriculumSampler adjusts task difficulty
           └─ Trains the T5 encoder-decoder model
```

#### Inference Data Flow (The OOD Pipeline)

This workflow allows you to test the model on any audio file, using the same structural analysis logic from training to generate interesting prompts.

```
1. User provides an arbitrary audio file (e.g., song.mp3)
     │
     └─> sample_audio_t5.py
           │
           ├─ Instantiates OODAudioPromptGenerator
           ├─ Runs the entire Tokenization + Structural Analysis pipeline in memory
           ├─ Samples high-priority sections to use as prompts (prefix/postfix)
           │
           ├─ For each iteration in the rollout:
           │    ├─ Dynamically chooses a strategy:
           │    │   - Continuation: Predict what comes next.
           │    │   - In-filling: Improvise between a prefix and a recurring postfix.
           │    └─ Stitches the newly generated tokens onto the main sequence tape
           │
           └─ Saves the final token sequence (.pt)
     │
     └─> decode_audio.py
           │
           ├─ Loads the token sequence and the Encodec model
           └─ Decodes the tokens back into an audible waveform
     │
     └─> Final Audio (.wav)
```

### 3. Training & Sampling a T5 Model on ASCII Text

This workflow demonstrates the T5 objective on an extremely obvious debugging dataset/task, focusing on architecture study over multiple choice examination benchmarks. this is 'rollout oriented ml research'.

#### Training Data Flow

```
1. Raw Text Files (.txt or .parquet)
     │
     └─> data/prepare_ascii.py
           │
           ├─ Converts all text to ASCII bytes (tokens 0-255)
           └─ Creates sharded, memory-mapped binary files
     │
     └─> Sharded ASCII Dataset (.bin)
           │
           └─> loader.py (in T5 ASCII mode via config file)
                 │
                 ├─ T5BatchProcessor creates denoising tasks
                 └─ Trains the T5 encoder-decoder model
```

#### Inference Data Flow

Similar to the audio workflow, but simpler. The model performs denoising/in-filling on ASCII character streams.

## Advanced Topics

### ATTENTION-II

[redacted]

### Compilation (Triton / Torch Inductor)

if you enjoy compiling code, you will *love* running triton. expect a 4x reduction in gpu memory utilization and a 4x increase in training speed if you compile your models. however, compiling is literal; you must have a c++ compiler configured in your system. a lot of the project notes attached to this repository will guide you towards a combination of dependencies which *permit* compilation, but compilation is never a sure thing in contemporary computing.

edit the torch_compile flags in sample.py and loader.py to 'False' if configuring compilers isn't your jam.

## License Notice

All code presented *without* license :)
Do not construe the availability of this source code for authorization of any sort! Or in fact any warranty or guarantee to the behavior, meaning, appropriate deployment, or social value of the tools demonstrated herein.

Get in touch personally if you think that code licensing is something that matters in your folkway, and we can work out something, together, which is better than what you originally had in mind. peace out~ <3