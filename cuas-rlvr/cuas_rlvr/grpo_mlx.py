"""GRPO with LoRA on Apple silicon (MLX), with the CUAS-Gym verifier as the reward.

For each training instance, G answers are sampled at temperature 1; their rewards are normalized within the group
(advantage A_i = (r_i - mean) / (std + eps)) and the policy is updated on-policy with
    loss = - sum_i A_i * sum_t log pi(y_it | x, y_i<t) / sum_i |y_i|
(one update per group, so the importance ratio is 1 and clipping is inactive; no KL term, as in the article).
Groups with identical rewards carry no signal and are skipped.

    python3 -m cuas_rlvr.grpo_mlx --model mlx-community/Qwen2.5-7B-Instruct-4bit --adapter ckpt/sft \
        --reward hybrid --steps 300 --out ckpt/grpo
"""
import argparse
import json
import os

os.environ.setdefault("CUAS_DTYPE", "float32")  # see mlx_common.load_model
import random
import time

import numpy as np

from cuas_gym.cli import ids, load
from .mlx_common import chat_ids, load_model, save_adapter
from cuas_gym.prompt import parse_answer
from cuas_gym.rewards import hybrid_reward


def reward_of(v, text, mode, lam):
    r = hybrid_reward(v, parse_answer(text), lam)
    if not r["valid"]:
        return 0.0, r
    return {"hybrid": r["r"], "outcome": r["r_out"], "process": r["r_proc"]}[mode], r


def make_loss(nn, mx):
    def loss_fn(model, inputs, targets, mask, adv, ntok):
        logits = model(inputs).astype(mx.float32)
        lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        tok = mx.take_along_axis(lp, targets[..., None], axis=-1).squeeze(-1)
        per_seq = (tok * mask).sum(axis=1)
        return -(adv * per_seq).sum() / ntok
    return loss_fn


def batch_arrays(mx, prompt, comps):
    """Inputs/targets/mask for prompt + completion sequences (completion tokens only in the mask)."""
    seqs = [prompt + c for c in comps]
    L = max(len(s) for s in seqs)
    inp = np.zeros((len(seqs), L - 1), np.int32)
    tgt = np.zeros((len(seqs), L - 1), np.int32)
    msk = np.zeros((len(seqs), L - 1), np.float32)
    for i, s in enumerate(seqs):
        n = len(s)
        inp[i, :n - 1] = s[:-1]
        tgt[i, :n - 1] = s[1:]
        msk[i, len(prompt) - 1:n - 1] = 1.0
    return mx.array(inp), mx.array(tgt), mx.array(msk)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True); p.add_argument("--adapter", help="start from this LoRA adapter (SFT)")
    p.add_argument("--data", default="data/gym"); p.add_argument("--out", required=True)
    p.add_argument("--reward", choices=["hybrid", "outcome", "process"], default="hybrid"); p.add_argument("--lam", type=float, default=0.5)
    p.add_argument("--steps", type=int, default=300); p.add_argument("--group", type=int, default=8)
    p.add_argument("--micro", type=int, default=2, help="sequences per backward pass (memory)")
    p.add_argument("--lr", type=float, default=2e-6); p.add_argument("--max-tokens", type=int, default=768)
    p.add_argument("--temp", type=float, default=1.0)
    p.add_argument("--lora-layers", type=int, default=16); p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--lora-scale", type=float, default=20.0)
    p.add_argument("--save-every", type=int, default=25); p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_map
    from mlx.optimizers import clip_grad_norm
    from mlx_lm.generate import batch_generate
    from mlx_lm.sample_utils import make_sampler

    random.seed(a.seed); np.random.seed(a.seed); mx.random.seed(a.seed)
    model, tok, cfg = load_model(a.model, a.adapter, a.lora_layers, a.lora_rank, a.lora_scale)
    opt = optim.Adam(learning_rate=a.lr)
    vg = nn.value_and_grad(model, make_loss(nn, mx))
    sampler = make_sampler(temp=a.temp)
    train = ids(a.data, "train")
    order = []
    os.makedirs(a.out, exist_ok=True)
    log = open(os.path.join(a.out, "log.jsonl"), "w")   # a run always starts from the SFT adapter, so its log restarts
    with open(os.path.join(a.out, "args.json"), "w") as f:
        json.dump(vars(a), f, indent=1)
    t0 = time.time()
    for step in range(1, a.steps + 1):
        if not order:
            order = random.sample(train, len(train))
        iid = order.pop()
        inst, v = load(a.data, iid)
        prompt = chat_ids(tok, inst)
        tg = time.time()
        res = batch_generate(model, tok, [prompt] * a.group, max_tokens=a.max_tokens, sampler=sampler, return_token_ids=True)
        tgen = time.time() - tg
        comps = [list(c) for c in res.token_ids]
        scored = [reward_of(v, t, a.reward, a.lam) for t in res.texts]
        R = np.array([s[0] for s in scored])
        rec = {"step": step, "iid": iid, "r_mean": float(R.mean()), "r_max": float(R.max()), "r_std": float(R.std()),
               "valid": float(np.mean([s[1]["valid"] for s in scored])), "qr": float(np.mean([s[1]["r_out"] for s in scored])),
               "proc": float(np.mean([s[1]["r_proc"] for s in scored])), "tokens": float(np.mean([len(c) for c in comps])),
               "t_gen": tgen}
        if R.std() > 1e-6:
            adv = (R - R.mean()) / (R.std() + 1e-4)
            ntok = float(sum(len(c) for c in comps))
            grads, loss = None, 0.0
            tb = time.time()
            for i in range(0, a.group, a.micro):
                idx = list(range(i, min(i + a.micro, a.group)))
                inp, tgt, msk = batch_arrays(mx, prompt, [comps[j] for j in idx])
                l, g = vg(model, inp, tgt, msk, mx.array(adv[idx], dtype=mx.float32), ntok)
                mx.eval(l, g)
                loss += float(l)
                grads = g if grads is None else tree_map(lambda x, y: x + y, grads, g)
            if not np.isfinite(loss):
                raise SystemExit(f"step {step}: non-finite loss, stopping (adapter of the last saved step is intact)")
            grads, gnorm = clip_grad_norm(grads, 1.0)
            opt.update(model, grads)
            mx.eval(model.trainable_parameters(), opt.state)
            rec.update(loss=loss, grad_norm=float(gnorm), t_bwd=time.time() - tb, updated=True)
        else:
            rec.update(updated=False)
        rec["elapsed"] = time.time() - t0
        log.write(json.dumps(rec) + "\n"); log.flush()
        print(f"step {step}: r {rec['r_mean']:.3f} (max {rec['r_max']:.3f}) valid {rec['valid']:.2f} QR {rec['qr']:.3f} "
              f"tok {rec['tokens']:.0f} gen {tgen:.0f}s {'upd' if rec['updated'] else 'skip'}  [{rec['elapsed'] / 60:.1f} min]", flush=True)
        if step % a.save_every == 0 or step == a.steps:
            save_adapter(model, cfg, a.out)
            save_adapter(model, cfg, os.path.join(a.out, f"step{step:04d}"))


if __name__ == "__main__":
    main()
