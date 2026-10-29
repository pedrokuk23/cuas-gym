"""Gradient diagnostic for SFT: per-module LoRA gradient norms on a few training examples, without updating.

    CUAS_DTYPE=float32 python3 -m cuas_rlvr.diag_grad --model mlx-community/Qwen2.5-7B-Instruct-4bit \
        --data data/mlx_sft_v2 [--lora-keys self_attn.q_proj,self_attn.k_proj,self_attn.v_proj,self_attn.o_proj]
"""
import argparse
import json
import os

import numpy as np

from .mlx_common import load_model
from .sft_mlx import encode


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True); p.add_argument("--data", required=True)
    p.add_argument("--n", type=int, default=3); p.add_argument("--lora-keys", default=None)
    a = p.parse_args()
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten
    keys = a.lora_keys.split(",") if a.lora_keys else None
    model, tok, cfg = load_model(a.model, None, 16, 16, 20.0, lora_keys=keys)
    rows = [encode(tok, json.loads(l)["messages"]) for l in open(os.path.join(a.data, "train.jsonl"))][:a.n]

    def loss_fn(model, inp, tgt, msk):
        logits = model(inp).astype(mx.float32)
        lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        tl = mx.take_along_axis(lp, tgt[..., None], axis=-1).squeeze(-1)
        return -(tl * msk).sum() / msk.sum()

    vg = nn.value_and_grad(model, loss_fn)
    print("dtype", os.environ.get("CUAS_DTYPE", "native"), "keys", keys or "all linear")
    for k, (pr, co) in enumerate(rows):
        s = pr + co
        inp = mx.array(np.array(s[:-1], np.int32)[None]); tgt = mx.array(np.array(s[1:], np.int32)[None])
        m = np.zeros(len(s) - 1, np.float32); m[len(pr) - 1:] = 1.0
        l, g = vg(model, inp, tgt, mx.array(m[None]))
        mx.eval(l, g)
        norms = {n: float(mx.sqrt((v.astype(mx.float32) ** 2).sum())) for n, v in tree_flatten(g)}
        tot = float(np.sqrt(sum(x * x for x in norms.values() if np.isfinite(x))))
        bad = [n for n, x in norms.items() if not np.isfinite(x)]
        top = sorted(((x, n) for n, x in norms.items() if np.isfinite(x)), reverse=True)[:6]
        med = float(np.median([x for x in norms.values() if np.isfinite(x)]))
        print(f"ex {k}: tokens {len(s)} loss {float(l):.4f} total {tot:.3g} median {med:.3g} non-finite {len(bad)}")
        for x, n in top:
            print(f"    {x:.3g}  {n}")


if __name__ == "__main__":
    main()
