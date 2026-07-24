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
uv sync --extra cuda
uv add flash-attn==2.7.2.post1 --no-build-isolation
#using uv in a cudaless context
uv venv --seed
uv sync
```

#### so you're running this on primeintellect for a live event...?
```
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

### Running the tests

```
uv run python -m pytest tests -q
```

there are tests now. this repo's recurring failure has never been the mechanism -- the
mechanisms are ambitious and mostly correct -- it is the **instrument**: something reading
the mechanism that was itself never checked. a presence test standing in for a value test,
a 3-unpack of a 4-tuple, a loss that deletes the token every sampler stops on, a pair of
transposes around a no-op. all of it survived because nothing in this repo could be imported
without a gpu, let alone run.

so: `loader.py`, `prompt_utils.py`, `t5_utils.py` and `sampler_utils.py` are now importable
on a laptop (heavy and cuda-only imports moved to their point of use; the training script
body lives in `main()`), and everything below runs on cpu in a few seconds.

- `test_forward_paths.py` -- every forward path x arity x z-loss x attention-II, plus a
  short real sampling loop, plus a model deliberately returning a **five**-tuple to prove
  the call sites cannot break that way again.
- `test_config_gating.py` -- every knob read by value; `"attention_deux": false` disables it;
  malformed and partial configs fail loudly; every shipped config in `configs/` builds.
- `test_model_internals.py` -- norms and scales against reference implementations.
- `test_attention_gate.py` -- gate-off bit-identity, gate-on init behaviour, gate coverage
  across all four attention schemas.
- `test_attention_ii_scaling.py` -- the S-scaling result, asserted.
- `test_t5_batch_processor.py` -- EOS survives into the loss and moves the number.
- `test_training_semantics.py` -- gradient accumulation semantics, the ddp micro-step fix,
  and the device-sync lint (which is itself tested against a planted offender).
- `test_audio_branch.py` -- the shared decode loop at batch 1 and batch 2, sync counts, and
  ast-level checks over the modules that need the audio stack to import.
- `test_compile.py` -- all 12 combinations of (gate x attention-II x qknorm), for both the
  AR and T5 forwards, traced with `torch.compile(fullgraph=True)`. `fullgraph` is the
  assertion: dynamo raises rather than silently falling back, so a graph break in a new knob
  fails the suite. backend is `eager`, because what is under test is the trace, not codegen
  -- inductor needs a c++ toolchain and this has to run on a laptop.
- `test_flex_masks.py` -- 37 tests over `flex_masks.py`, each one asserting one of the two
  safety directions of the block-mask contract against a dense reference built *in the
  test* at tiny shapes. see "BLOCK MASKS FROM BOUNDS" below.
- `test_lints.py` -- the two lints, each with a planted violation, because a lint that
  finds nothing because it is broken passes every test ever written.

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

...fine. [unredacted], because it turns out you cannot ablate a thing you cannot measure,
and this one has never been measurable.

**what it is.** a second QK pair (`queryBproj`/`keyBproj`) is projected and rotary-embedded
exactly like the first, its scores are computed with no softmax, and the result is matmul'd
against an all-ones V and added to the attention output *before* `attnoutproj`.

it is not a bias. V is all ones, so every row of `scores @ V` is one number repeated across
`dim_head` -- the row sum. push that through `attnoutproj` and the term is

```
Q · K_B^T · 1 · O   =   (per-head scalar) x (that head's column-sum of O)
```

a **context-conditioned scalar gain along a learned per-head direction in the residual
stream**. there is no information in the V slot at all; the entire content of the term is
"how much does this query agree with everything it can see", and the direction it writes
along is learned. that is a strange but perfectly legitimate member of the attention family.
`tests/test_attention_gate.py::test_attention_ii_is_a_scalar_gain_along_a_learned_direction`
asserts exactly this decomposition, so the framing above is checked rather than claimed.

**its one real defect: normalization.** the softmax path returns a convex combination, so
its magnitude does not care how many keys there are. attention-II returns a plain SUM over
up to S scored keys, so it does. the two get added together, and only one of them is
comparable with itself across sequence lengths. that is why every attention-II ablation this
repo has run is uninterpretable: the term's contribution at S=512 is not the same object as
its contribution at S=64.

#### the gate

`"attn_gate": "sigmoid"` adds a post-attention gate `g = sigmoid(W_g x)` with
`W_g : dim -> headcount`, broadcast over `dim_head` and multiplied into the attention output
before the out-projection.

it is applied **uniformly to every attention schema** -- standard SDPA self-attention,
cross-attention, the l2norm/cosine path, and attention-II each get their own gate, built by
the same factory from the same config key. this is the point: a gate that existed on one
path and not another would confound every A/B run against it.

- **default is `"none"`, and `"none"` is free.** no parameters, no ops, and the forward pass
  is *bit-identical* to the pre-gate model -- `torch.equal`, not `allclose`, over logits,
  loss, z-loss and per-sequence loss, for AR and T5, with and without attention-II, across
  three qknorm settings. existing checkpoints load `strict=True` and evaluate unchanged.
- **init is zero.** `W_g = 0`, `b_g = 0`, so `sigmoid(0) = 0.5` exactly: at step 0 every gate
  is a constant halving with no input dependence. this is deliberately *not* identity-at-init
  -- a gate that starts at 1.0 needs a large positive bias, which starts the sigmoid
  saturated and its gradient near zero. 0.5 is the sigmoid's maximum-derivative point, so
  the gate learns from the first step.
- **turning it on does not perturb the trunk.** constructing an `nn.Linear` draws from the
  global rng, so a naively-added gate would shift every parameter initialized after it and
  the ablation would be confounded by init noise. the gate is constructed inside a
  `fork_rng`, so gate-on and gate-off models at the same seed differ *only* by the gate
  tensors. paired A/B.
- **compile-friendly.** one `Linear`, one `sigmoid`, one broadcast multiply. the on/off
  choice is a python-level branch resolved at construction, so no data-dependent control
  flow reaches the traced graph.

#### the measurement: does the gate alone make attention-II commensurable?

no. measured, not argued. `uv run python tools/measure_attention_ii.py` reports the rms
magnitude of each term as it enters `attnoutproj`, at dim=768, 12 heads, dim_head=64,
causal mask, seed 1337:

| config | S=64 | S=128 | S=256 | S=512 | S=1024 | ratio growth |
|---|---|---|---|---|---|---|
| **as shipped** (gate none, no row norm) | 26.8 | 49.2 | 91.2 | 169.3 | **316.6** | **11.8x** |
| gate=sigmoid, no row norm | 26.8 | 49.2 | 91.2 | 169.3 | 316.6 | 11.8x |
| row norm = 1/sqrt(S) | 3.35 | 4.35 | 5.70 | 7.48 | 9.90 | 2.95x |
| **row norm = mean** | 1.26 | 1.29 | 1.26 | 1.18 | **1.19** | **0.95x** |
| gate=sigmoid + row norm = mean | 1.26 | 1.29 | 1.26 | 1.18 | 1.19 | 0.95x |

(the numbers are `rms(attention-II) / rms(softmax)`. underlying magnitudes: as shipped, the
softmax term falls 0.204 -> 0.071 as it averages over more keys while the attention-II term
climbs 5.47 -> 22.6.)

three findings:

1. **as shipped, attention-II is 27x the softmax term at S=64 and 317x at S=1024.** it does
   not merely dominate the residual contribution, it dominates it *increasingly*, so the
   architecture at one sequence length is not the architecture at another.
2. **the sigmoid gate alone changes the ratio by exactly nothing.** it halves both terms at
   init, because it is applied uniformly -- which is the correct design and also means it
   cannot, on its own, buy commensurability. a sigmoid multiplies by something in (0,1); it
   bounds a term, it cannot cancel an S-dependence. the gate gives the network a *learnable*
   knob, not a *structural* fix.
3. **the row sum needs an explicit normalization, and `mean` is the one that works.**
   dividing by the number of unmasked keys per query row -- the row-exact analogue of the
   softmax denominator, not a global 1/S, because under a causal mask row `i` sums `i+1`
   terms and not `S` -- holds the ratio flat at ~1.2 across a 16x span of sequence lengths.
   `1/sqrt(S)` makes the attention-II term itself S-invariant (0.684 -> 0.707) but leaves the
   ratio drifting 3x, because the softmax term is *itself* shrinking like 1/sqrt(S).

so the ablation you actually want is `"attention_deux_norm": "mean"` **and**
`"attn_gate": "sigmoid"`: `mean` makes the term commensurable with the softmax path across
S, and the gate bounds its output and lets the network learn how much of it it wants. with
qknorm bounding the score magnitude on the way in and both of those on the way out,
attention-II is hopelessly normalized and its distinct properties are finally the only thing
varying.

both default to the legacy values (`"none"` / `"none"`), so nothing about the existing
checkpoints or the existing numbers moves until you ask it to.

`tests/test_attention_ii_scaling.py` asserts all three findings.

### rotary base

`rotary_embedding_base` defaults to **1000**, not the usual 10000. this is deliberate
(`shhh don't tell anyone about the rotemb base`) and it has not been changed. it is now a
named config key with a documented default instead of a magic number buried in a keyword
argument, so it is possible to ablate it on purpose. do not "fix" it to 10000 without
rerunning something.

### norms, honestly

- `qknorm` normalizes over **`[dim_head]`** -- one head's worth. it used to be declared as
  `[headcount, dim_head]`, which makes `nn.RMSNorm`/`nn.LayerNorm` reduce over the head axis
  too, coupling every head to every other. that is not qk-norm. the `dynamic_shape_*` norms
  were never affected (they always reduced over the last axis only), and every config in
  `configs/` uses `dynamic_shape_rmsnorm`, so no shipped run changes. selecting
  `"qknorm": "layernorm"` now gets you the actual 2302.05442 spec the header comment has
  been describing since the first commit.
- `dynamic_shape_rmsnorm`/`dynamic_shape_layernorm` had a `transpose(1,2)` on the way in and
  another on the way out around a `size()[3:]`. for a `[B,T,H,D]` tensor that slice is `(D,)`
  in either layout, so the pair of transposes was a no-op: these are, and always were, plain
  norms over head_dim. simplified, with a bit-identity test against the old implementation.
  they keep the property that motivated them -- normalized_shape is read off the runtime
  tensor, so sampling can change B and T freely.
- the `l2norm` "bootleg cosine attention" path applied its learned scalar to the attention
  *output* with `scale=1` hardcoded. outside the softmax that is a per-block output rescale,
  not a temperature; it did nothing the name implies. the scalar is now folded into Q, which
  is exactly equivalent to passing it as `scale=` and keeps the parameter differentiable.
  that path also passed `is_causal=True` and discarded `attention_mask` entirely, which
  silently causal-masked the **bidirectional T5 encoder** and ignored padding in cross-
  attention. it honors the mask it is given now, and it is checked against an explicit
  softmax reference implementation.
- `cross_attn_norm` was constructed and never called, with the disabling line commented out
  (`nice try gemini 2.5! i won't surrender so easily`). **deleted, deliberately.** the entire
  premise of the pgptl parallel block is ONE pre-norm feeding self-attention, cross-attention
  and the FFN -- that is where the "3x fewer layernorms per layer" claim in the forenote
  comes from. a second norm in front of cross-attention is the serial design this
  architecture exists to avoid. it carried zero parameters under every config in `configs/`
  (`layerwisenorm: rmsnorm` -> `elementwise_affine=False`), so no shipped `state_dict` key
  changes; there is a test for that.

### The T5 target stream: what is in the loss

`T5BatchProcessor` used to build the loss labels by overwriting **sentinels and EOS** with
`pad_token_id` so `ignore_index` would skip them. the consequence is worth stating plainly:
**the model was never trained to emit EOS**, and every sampler in this repo -- `sample_t5.py`,
`sample_audio_t5.py`, `sample_audio_t5_skipdecode.py`, `t5_service.py` -- terminates on EOS.
generation could only ever stop by exhausting `max_new`, and has for a year. the sampler was
reading a stop signal the trainer had removed.

the treatment now:

- **EOS is in the loss.** it is the one token the samplers depend on.
- **sentinels are out of the loss, by default and on purpose.** the k-th sentinel in a target
  stream is always `mask_token_start_id + k` -- a deterministic counter carrying no
  information about the data. training on it buys nothing and deflates the reported loss by
  padding it with free tokens. `T5BatchProcessor(train_on_sentinels=True)` gives the
  T5-canonical treatment (original T5 trains on the whole target) if you want to ablate it.
- **the range is derived, not hardcoded.** it used to be `mask_token_start_id + 100`, while
  the masker's own guardrail permits `vocab_size - mask_token_start_id` sentinels -- 1022 for
  the audio configs. sentinels 100..1021 fell through and *were* trained on, so the real
  policy was "sentinels are excluded, except the ones that aren't", which is worse than
  either policy. one definition (`max_spans`) now serves both.
- the decoder's start-of-sequence slot is spelled with `pad_token_id`, so
  `(decoder_input_ids != pad_token_id)` marked the whole first **column** invisible: nothing
  could attend to the start of the sequence. it did not NaN (torch 2.5.1's safe-softmax
  returns an all-zero row), it just silently contributed nothing.

### BLOCK MASKS FROM BOUNDS

`flex_masks.py` builds `flex_attention` `BlockMask`es from cheap conservative bounds,
without ever materializing a dense `[B, H, Q, KV]` mask. it knows nothing about any model:
it is about mask STRUCTURE -- prefix bounds, sliding windows, block-diagonal groups, and
their unions, intersections and key-space concatenations. those are the shapes that show up
in padded batching, packed multi-user prefill, prefix-bidirectional attention,
sliding-window attention and grouped/block-parallel decoding alike.

#### the contract: full is a SUBSET, partial is a SUPERSET

this is the whole reason the module can be cheap and still be correct. a `BlockMask` carries
two block lists and **they have opposite safety directions**:

| list | requirement | what happens if you get it wrong |
| --- | --- | --- |
| `full_kv_num_blocks` / `full_kv_indices` | must be a **SUBSET** of the truly-all-allowed blocks | flex **skips `mask_mod` entirely** inside a full block, so over-claiming FULL silently attends forbidden keys. this is the unsafe direction. |
| `kv_num_blocks` / `kv_indices` (partial) | must be a **SUPERSET** of every block containing any allowed element | flex evaluates `mask_mod` inside these, so an over-listed block costs kernel time and nothing else. under-claiming silently drops attention. |

the corollary is the thing worth remembering:

> **demoting a would-be-full block to partial is always safe.** it costs speed, never
> correctness, because `mask_mod` runs inside it and filters exactly.

so a builder does not need exact block occupancy. it needs a cheap LOWER bound on full and a
cheap UPPER bound on live, and the `mask_mod` -- which runs in-kernel, on registers, and
never touches HBM -- makes up the difference. that is what lets every function in the module
be `O(n_q_blocks * n_kv_blocks)` instead of `O(Q * KV)`.

two consequences that look like bugs and are not:

- a **sliding window** claims no full blocks by default. a kv block is all-allowed only if
  it sits inside *every* query's window, which needs `window >= block + (hi - lo)`; with the
  usual `window < block` that is impossible. pass `allow_full=True` with `key_all` when you
  genuinely have wide windows.
- the per-q-block union of windows is **not contiguous** once the bounds inside a block are
  spaced further apart than the window. the convex hull `[lo - window + 1, hi]` over-lists,
  and that is legal: a superset is exactly what partial must be.
  `test_gapped_window_union_is_a_legal_superset_not_a_bug` asserts the gaps are real, the
  hull covers them, and the reconstruction is still exact.

#### why this is not optional: the traffic

the obvious way to derive block occupancy is `any()`/`all()` over a materialized dense mask.
exact, structurally impossible to get wrong -- and bandwidth-pessimized in precisely the way
`flex_attention` exists to avoid. at a representative training shape (B=128, H=12, Q=640,
KV=512, BLOCK=128):

| description | elements | bytes |
| --- | --- | --- |
| dense `[B, H, Q, KV]` bool mask | 128 x 12 x 640 x 512 = **503,316,480** | **503.3 MB** |
| block occupancy, H=1 (partial + full bools) | 2 x 128 x 5 x 4 = **5,120** | 5.1 KB |
| the int32 CSR flex actually consumes (both orientations, partial + full) | | 50.2 KB |
| **block total** | | **55.3 KB** |

**9,102x**, per layer, per step. against the occupancy alone the ratio is the asymptotic
`BLOCK^2 * H / 2 = 128*128*12/2 = 98,304`; the CSR is what costs the rest. masks are built
with `H=1` and broadcast across heads, because every pattern in the module is a function of
(batch, query position, key position) only.

`mask_traffic_report()` computes those numbers rather than asserting them by hand, and
`test_mask_traffic_report_matches_the_readme_arithmetic` pins this table.

#### enforcement

- `lint_mask_traffic.py` bans dense-mask materialization two ways, because neither alone is
  sufficient. **statically**, an AST pass flags `create_block_mask` / `create_mask` /
  `_ordered_to_dense` and any `mask_mod` that subscripts a tensor with both a query and a key
  index; it is cheap and it cannot see through helper functions. **dynamically**,
  `AllocationAudit` is a `TorchFunctionMode` that watches every tensor produced inside a
  block and fails the moment anything `Q*KV`-shaped appears, no matter how it was spelled.
- `lint_attention_dtypes.py` is an AST pass over dtype-widening sites on the attention path,
  checked against `lint_allowlist_attention_dtypes.json` -- a data file in which every entry
  carries a **numerical** justification. a new widening site with no recorded reason fails
  the suite. it is a static pass and not a runtime assertion on purpose: a runtime dtype
  check would cost a graph break and a device sync on every step of every run forever, to
  catch a mistake that is made in the source at edit time.

both lints take their file set and allowlist set as parameters and run standalone:

```
uv run python lint_mask_traffic.py
uv run python lint_attention_dtypes.py
```

#### `BlockMask` is constructed directly

not via `create_block_mask` (it evaluates `mask_mod` over the whole `Q*KV` grid, and is not
dynamo-traceable in torch 2.5.1 -- `inspect.signature`) and not via
`BlockMask.from_kv_blocks` (which round-trips the CSR back through a block-level dense tensor
via `_ordered_to_dense` purely to transpose it, and runs per-call validation -- by the same
rule that keeps checks out of hot paths, that validation belongs in the equivalence test).
the q-orientation lists flex's backward needs are the packing of the transposed occupancy,
asserted bit-identical to torch's `_transpose_ordered`. measured on torch 2.5.1, building a
`BlockMask` this way inside a `torch.compile` region produces **zero dynamo graph breaks**,
so the mask build does not need an eager island at all.

#### the obvious next step, deliberately not taken

**the T5 encoder and decoder still build their padding and causal masks densely, and they
should not.** both are pure block structure -- a padding mask is `key_block_validity()`, a
causal mask is `prefix_blocks()` with `bound = q_idx`, and encoder-decoder cross-attention is
`prefix_blocks()` with an unbounded prefix -- so the whole family is expressible in the
primitives above with no new mechanism.

it is not done here. moving the model's attention onto `flex_attention` changes the numerics
(a different kernel, a different reduction order), changes the compile surface, and needs its
own before/after on real checkpoints. that is a reviewed piece of work of its own, not a
rider on the module that makes it possible. `flex_masks.py` ships tested and unused by the
model, which is the honest state of it.

### Known-broken and left that way

`sample_audio_t5_skipdecode.py` is a janky mess and the author says so. it has not been
rewritten. one live `KeyError` in its fallback path was guarded, and that is all.

its nearest-neighbour matcher takes the **cosine similarity between sequences of integer
token IDs**, zero-padded to a common length. that is not a distance between sounds. encodec
codebook indices are categorical labels; index 300 is not "closer" to index 301 than to index
7, and their dot product means nothing. the mechanism is conceptually empty and no amount of
fixing the code around it changes that. it is written down here rather than repaired, because
repairing it means designing a real acoustic distance, and that is a different piece of work.

### Device-host synchronization: the inventory

every `.item()`, `.tolist()`, `.cpu()`, `.numpy()`, python-side `if` on a tensor value, and
`torch.cuda.synchronize()` is a pipeline drain: the gpu empties and the host waits. in a
training loop that is pure loss. this is the accounting, because you cannot fix what you
have not counted. **no speedup number appears here** -- the work was done and validated on a
cpu box, so there is nothing honest to report but the counts.

`B` is `device_batch_size`, `A` is `train_accumulation_steps`. the audio configs run B=32,
A=4.

#### removed

| where | what forced it | fired | now |
|---|---|---|---|
| `loader.py` per-sequence curriculum loop | `bucket_indices[i].item()` + `loss_per_seq[i].item()` | **2B per micro-step** = 256/step at B=32,A=4 | one batched drain per optimizer step (**1/step**), replaying the same updates in the same order |
| `t5_utils.create_curriculum_batch` | `bucket_indices[i].item()` + `batch_x[i].tolist()` inside the per-sequence loop | **2B per micro-step** = 256/step | 2 transfers per micro-step, hoisted out of the loop (**8/step**) |
| `loader.py` per-step log line | `train_loss.item()`, `train_aux_loss.item()` | 2/step | copied async into pinned host memory, consumed **one step late** (0 blocking) |
| `loader.py` curriculum log line | `format_tensor_log` iterating a device tensor | num_buckets/step = 8/step | distribution kept on host (where the sampler already lives), 0 |
| `loader.py` validation loop | `val_loss += loss.detach()` accumulated then `:.4f`-formatted | 2 per val micro-step | accumulated on device, **1 drain per validation pass** |
| `loader.py` rollout capture | `x[i].tolist()` per captured sequence | B per capture | 1 per capture |
| `sampler_utils.ar_sample` (was inlined ×3) | none -- no eos handling existed | 0 | 0, and eos support added without adding one |
| `sampler_utils.t5_decode` (was inlined ×5) | `if has_finished.all()` | **1 per generated token** | 1 per `stop_check_every` (32) tokens; `stop_check_every=0` never syncs |
| `sample_t5.py` | `if idx_next.item() == eos_id` | **1 per generated token** | same shared loop |

for a 512-token T5 rollout at batch 1, that last row alone is 512 drains -> 16.

#### kept, deliberately

| where | why | frequency |
|---|---|---|
| `loader.py` curriculum drain | `AdaptiveCurriculumSampler` runs `scipy.optimize.root_scalar` (brentq): a host-side scalar root find with a data-dependent iteration count. it **cannot** be made device-resident and pretending otherwise would mean contorting the code around a sync that still happens. the sampler's whole state is kept on cpu so the solver itself costs no transfer; the losses have to come to it. | 1 per optimizer step, batched |
| `loader.py` `torch.cuda.synchronize()` | bounding the timed region honestly. all four calls sit on an **interval boundary** (checkpoint save, validation, start/end of timing) and none are per-step. | on save/val intervals |
| `loader.py` validation drain | one `torch.stack([...]).cpu()` for the whole pass, on the validation interval | 1 per validation |
| `torch.cuda.max_memory_allocated()` | end of run | once |

#### identified, NOT changed

`DistributedDataLoader.next_batch` and `IntelligentAudioDataLoader.next_batch` build tensors
in unpinned host memory and copy them across blocking, once per micro-step. pinning plus
`non_blocking=True` is the standard fix and is semantics-identical, but it is a cuda-only
change that cannot be validated on this machine, so it is written down rather than shipped.

#### what changed in the output

the per-step training log line now prints the loss from **step N-1** on the line labelled
step N. that is the whole cost of the deferred-scalar trick, and it is a change to what is
*printed*, never to what is *computed*: the loss tensors are untouched and no optimizer ever
sees these values. the first line is skipped because there is no previous step, and the last
one is drained after the loop. validation lines are unaffected and still exact.

#### the guard

`tests/test_training_semantics.py` walks `loader.main()` and `sampler_utils.ar_sample` with
`ast` and fails if a `.item()`/`.tolist()`/`.cpu()`/`.numpy()`/`cuda.synchronize()` appears
inside an inner loop. the lint is itself checked against a planted offender, because a lint
that passes by finding nothing anywhere is exactly the kind of instrument this repo keeps
getting caught by. sync counts in the sampling loops are asserted directly, by patching
`Tensor.all`/`Tensor.item` and counting calls.

### Compilation (Triton / Torch Inductor)

if you enjoy compiling code, you will *love* running triton. expect a 4x reduction in gpu memory utilization and a 4x increase in training speed if you compile your models. however, compiling is literal; you must have a c++ compiler configured in your system. a lot of the project notes attached to this repository will guide you towards a combination of dependencies which *permit* compilation, but compilation is never a sure thing in contemporary computing.

edit the torch_compile flags in sample.py and loader.py to 'False' if configuring compilers isn't your jam.

## License Notice

All code presented *without* license :)
Do not construe the availability of this source code for authorization of any sort! Or in fact any warranty or guarantee to the behavior, meaning, appropriate deployment, or social value of the tools demonstrated herein.

Get in touch personally if you think that code licensing is something that matters in your folkway, and we can work out something, together, which is better than what you originally had in mind. peace out~ <3