"""measure the magnitude of the attention-II term against the softmax term, as a function
of sequence length.

    uv run python tools/measure_attention_ii.py

the question this answers, and the reason attention-II has never been measurable:

the softmax path produces a convex combination of value vectors, so its magnitude is
bounded by the magnitude of V regardless of how many keys there are. the attention-II path
has NO softmax -- its rows are plain sums of up to S scaled scores against an all-ones V --
so its magnitude grows with S. two terms are added together in the residual stream and only
one of them is commensurable with itself across sequence lengths.

the follow-up question, which is what the gate is for: does a post-attention sigmoid gate
fix that on its own? a sigmoid can only multiply by something in (0,1), so it can shrink the
term but it cannot remove an S-dependence. the numbers below are what settles it, rather
than the argument.

everything here runs on cpu in a couple of seconds. no gpu, no training, no timings --
these are magnitudes, which are exact and portable, not throughput, which is neither.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pgptlformer
from pgptlformer import create_attention_mask

SEQ_LENS = (64, 128, 256, 512, 1024)
DIM, DIM_HEAD, HEADS = 768, 64, 12
BATCH = 2
SEED = 1337


def make_block(**overrides):
    cfg = {
        "vocab_size": 256, "num_layers": 1, "dim": DIM, "dim_head": DIM_HEAD,
        "headcount": HEADS, "ff_mult": 4, "lambda": True,
        "layerwisenorm": "rmsnorm", "qknorm": "dynamic_shape_rmsnorm",
        "attention_deux": True, "training_seqlen": 512,
    }
    cfg.update(overrides)
    torch.manual_seed(SEED)
    return pgptlformer.vit22_tformer(cfg, is_decoder=False)


@torch.no_grad()
def split_terms(block, seq_len):
    """run the block's self-attention and return the softmax term and the attention-II term
    separately, both as they are just before attnoutproj (i.e. where the gate applies)."""
    torch.manual_seed(SEED)
    x = torch.randn(BATCH, seq_len, DIM)
    # the block input is pre-normed in forward(), so norm it here too or the magnitudes are
    # a statement about randn rather than about attention.
    x = block.layerwisenorm(x)
    mask = create_attention_mask(torch.ones(BATCH, seq_len, dtype=torch.bool), is_causal=True)

    B, T, _ = x.shape
    q = block.queryproj(x).view(B, T, HEADS, DIM_HEAD)
    k = block.keyproj(x).view(B, T, HEADS, DIM_HEAD)
    v = block.valueproj(x).view(B, T, HEADS, DIM_HEAD)
    cos, sin = block.rotary(q)
    q, k = block.projnorm(q), block.projnorm(k)
    q = pgptlformer.apply_rotarizer_emb(q, cos, sin)
    k = pgptlformer.apply_rotarizer_emb(k, cos, sin)
    softmax_term = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
        attn_mask=mask[:, None, :, :], scale=block.scale)
    softmax_term = block._apply_gate(block.self_attn_gate, softmax_term, x)

    bq = block.queryBproj(x).view(B, T, HEADS, DIM_HEAD)
    bk = block.keyBproj(x).view(B, T, HEADS, DIM_HEAD)
    bq, bk = block.projnorm(bq), block.projnorm(bk)
    bq = pgptlformer.apply_rotarizer_emb(bq, cos, sin)
    bk = pgptlformer.apply_rotarizer_emb(bk, cos, sin)
    dud = torch.ones_like(v)
    ii_term = pgptlformer.scaled_dot_product_attn_bias(
        bq.transpose(1, 2), bk.transpose(1, 2), dud.transpose(1, 2),
        attn_mask=mask.to(q.dtype), scale=block.scale,
        row_norm=block.attention_II_norm, training=False)
    ii_term = block._apply_gate(block.attention_II_gate, ii_term, x)
    return softmax_term, ii_term


def rms(t):
    return t.float().pow(2).mean().sqrt().item()


def sweep(label, **overrides):
    block = make_block(**overrides)
    block.eval()
    rows = []
    for s in SEQ_LENS:
        soft, ii = split_terms(block, s)
        rows.append((s, rms(soft), rms(ii), rms(ii) / rms(soft)))
    return label, rows


def main():
    torch.set_grad_enabled(False)
    configs = [
        ("gate=none  attn_ii_norm=none        (as shipped)", dict()),
        ("gate=sigmoid attn_ii_norm=none", dict(attn_gate="sigmoid")),
        ("gate=none  attn_ii_norm=inv_sqrt_s", dict(attention_deux_norm="inv_sqrt_s")),
        ("gate=none  attn_ii_norm=mean", dict(attention_deux_norm="mean")),
        ("gate=sigmoid attn_ii_norm=mean", dict(attn_gate="sigmoid", attention_deux_norm="mean")),
    ]
    print(f"dim={DIM} heads={HEADS} dim_head={DIM_HEAD} batch={BATCH} seed={SEED}, causal mask")
    print("rms magnitude of each term as it enters attnoutproj\n")
    for label, overrides in configs:
        _, rows = sweep(label, **overrides)
        print(label)
        print(f"    {'S':>6} {'softmax':>10} {'attn-II':>10} {'ratio':>10} {'ratio/S=64':>11}")
        base = rows[0][3]
        for s, soft, ii, ratio in rows:
            print(f"    {s:>6} {soft:>10.4f} {ii:>10.4f} {ratio:>10.3f} {ratio/base:>11.2f}x")
        print()


if __name__ == "__main__":
    main()
