"""Supervised fine-tuning with LoRA in MLX, with the safeguards mlx_lm's trainer lacks.

Loss: mean negative log-likelihood of the assistant answer (prompt masked), computed in float32. Each update clips the
gradient norm and is skipped when the loss or the gradient is not finite, so one overflow cannot ruin the run.
Checkpoints are saved as NNNNNNN_adapters.safetensors (the layout of mlx_lm, used by select_sft.py).

    python3 -m cuas_rlvr.sft_mlx --model mlx-community/Qwen2.5-7B-Instruct-4bit --data data/mlx_sft_v2 \
        --iters 560 --out ckpt/v2_q7b_sft
"""
import argparse
import json
import math
import os

os.environ.setdefault("CUAS_DTYPE", "float32")  # see mlx_common.load_model
import random
import time

import numpy as np

from .mlx_common import load_model, save_adapter


def encode(tok, msgs):
    p = tok.apply_chat_template(msgs[:-1], add_generation_prompt=True)
    p = list(p) if not isinstance(p, str) else tok.encode(p)
    c = tok.encode(msgs[-1]["content"], add_special_tokens=False) if hasattr(tok, "encode") else []
    eos = tok.eos_token_id if getattr(tok, "eos_token_id", None) is not None else list(tok.eos_token_ids)[0]
    return p, list(c) + [eos]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True); p.add_argument("--data", required=True); p.add_argument("--out", required=True)
    p.add_argument("--iters", type=int, default=560); p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--clip", type=float, default=1.0); p.add_argument("--max-len", type=int, default=3072)
    p.add_argument("--lora-layers", type=int, default=16); p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--lora-scale", type=float, default=20.0)
    p.add_argument("--lora-keys", default=None, help="comma-separated module names (default: all linear layers)")
    p.add_argument("--eval-every", type=int, default=50); p.add_argument("--save-every", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.optimizers import clip_grad_norm
    from mlx.utils import tree_flatten

    random.seed(a.seed); mx.random.seed(a.seed)
    model, tok, cfg = load_model(a.model, None, a.lora_layers, a.lora_rank, a.lora_scale,
                                 lora_keys=a.lora_keys.split(",") if a.lora_keys else None)
    load = lambda f: [encode(tok, json.loads(l)["messages"]) for l in open(os.path.join(a.data, f))]
    train, valid = load("train.jsonl"), load("valid.jsonl")

    def arrays(pr, co):
        s = (pr + co)[:a.max_len]
        inp = mx.array(np.array(s[:-1], np.int32)[None])
        tgt = mx.array(np.array(s[1:], np.int32)[None])
        m = np.zeros(len(s) - 1, np.float32); m[len(pr) - 1:] = 1.0
        return inp, tgt, mx.array(m[None])

    def loss_fn(model, inp, tgt, msk):
        logits = model(inp).astype(mx.float32)
        lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        tl = mx.take_along_axis(lp, tgt[..., None], axis=-1).squeeze(-1)
        return -(tl * msk).sum() / msk.sum()

    vg = nn.value_and_grad(model, loss_fn)
    opt = optim.Adam(learning_rate=a.lr)
    os.makedirs(a.out, exist_ok=True)
    log = open(os.path.join(a.out, "log.jsonl"), "w")

    def val_loss():
        ls = []
        for pr, co in valid:
            l = loss_fn(model, *arrays(pr, co)); mx.eval(l); ls.append(float(l))
        return float(np.mean(ls))

    v0 = val_loss()
    print(f"iter 0: val {v0:.4f}", flush=True)
    log.write(json.dumps({"iter": 0, "val": v0}) + "\n")
    order, skipped, run, t0 = [], 0, [], time.time()
    for it in range(1, a.iters + 1):
        if not order:
            order = random.sample(range(len(train)), len(train))
        pr, co = train[order.pop()]
        l, g = vg(model, *arrays(pr, co))
        mx.eval(l, g)
        lv = float(l)
        g, gn = clip_grad_norm(g, a.clip)
        gn = float(gn)
        if math.isfinite(lv) and math.isfinite(gn):
            opt.update(model, g)
            mx.eval(model.trainable_parameters(), opt.state)
            run.append(lv)
        else:
            skipped += 1
        rec = {"iter": it, "loss": lv, "grad_norm": gn, "skipped": skipped}
        if it % 10 == 0:
            print(f"iter {it}: loss {np.mean(run[-10:]) if run else float('nan'):.4f} grad {gn:.2f} skipped {skipped} "
                  f"[{(time.time() - t0) / 60:.1f} min]", flush=True)
        if it % a.eval_every == 0 or it == a.iters:
            rec["val"] = val_loss()
            print(f"iter {it}: val {rec['val']:.4f}", flush=True)
        log.write(json.dumps(rec) + "\n"); log.flush()
        if it % a.save_every == 0 or it == a.iters:
            mx.save_safetensors(os.path.join(a.out, f"{it:07d}_adapters.safetensors"), dict(tree_flatten(model.trainable_parameters())))
            save_adapter(model, cfg, a.out)
    print(f"done; skipped updates: {skipped}")


if __name__ == "__main__":
    main()
