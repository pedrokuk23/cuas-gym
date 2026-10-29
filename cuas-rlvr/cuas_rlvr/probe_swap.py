"""Range-swap probe: does the model follow the data-sheet ranges or the type names?

For each version-2 instance of a split in which both EO/IR and passive RF have a nonzero range, the prompt is rebuilt
with the two ranges swapped (type names, costs and physics unchanged). The model answers both prompts with greedy
decoding. A model that reasons from the ranges shifts its choices toward the type that received the longer range;
a model that reasons from the names does not. Scores are computed on the true detection matrix.

    python3 -m cuas_rlvr.probe_swap --model mlx-community/Qwen2.5-7B-Instruct-4bit --adapter ckpt/v2_q7b_grpo_s0 \
        --data data/gym2 --split test_id --out results/v2/probe_swap_grpo_s0.jsonl
"""
import argparse
import copy
import json
import os
import time

import numpy as np

from cuas_gym.cli import ids, load
from .mlx_common import chat_ids, load_model
from cuas_gym.prompt import parse_answer
from cuas_gym.rewards import hybrid_reward


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True); p.add_argument("--adapter")
    p.add_argument("--data", default="data/gym2"); p.add_argument("--split", default="test_id")
    p.add_argument("--a", default="eo"); p.add_argument("--b", default="rf")
    p.add_argument("--max-tokens", type=int, default=1536); p.add_argument("--batch", type=int, default=8)
    p.add_argument("--out", required=True); p.add_argument("--n", type=int, default=1000)
    a = p.parse_args()
    from mlx_lm.generate import batch_generate
    from mlx_lm.sample_utils import make_sampler
    model, tok, _ = load_model(a.model, a.adapter)
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(l)["iid"] for l in open(a.out)}
    jobs = []
    for iid in ids(a.data, a.split)[:a.n]:
        inst, v = load(a.data, iid)
        if a.a in inst.types and a.b in inst.types and inst.drange[a.a] > 0 and inst.drange[a.b] > 0 and iid not in done:
            jobs.append((iid, inst, v))
    print(f"{len(jobs)} instances with both {a.a} and {a.b} usable", flush=True)
    sampler = make_sampler(temp=0.0)
    t0 = time.time()
    with open(a.out, "a") as f:
        for k in range(0, len(jobs), a.batch // 2):
            chunk = jobs[k:k + a.batch // 2]
            prompts, keys = [], []
            for iid, inst, v in chunk:
                sw = copy.copy(inst); sw.drange = dict(inst.drange)
                sw.drange[a.a], sw.drange[a.b] = inst.drange[a.b], inst.drange[a.a]
                prompts += [chat_ids(tok, inst), chat_ids(tok, sw)]
                keys += [(iid, "orig"), (iid, "swap")]
            res = batch_generate(model, tok, prompts, max_tokens=a.max_tokens, sampler=sampler)
            rec = {}
            for (iid, kind), text in zip(keys, res.texts):
                inst, v = next((i, vv) for ii, i, vv in chunk if ii == iid)
                act = parse_answer(text)
                r = hybrid_reward(v, act)
                ts = [v.alias.get(str(t).lower(), str(t).lower()) for _, t in act] if act else []   # letters -> types when anonymized
                rec.setdefault(iid, {"iid": iid, "drange": inst.drange, "cost": inst.cost})[kind] = {
                    "valid": r["valid"], "qr": r["r_out"], "n_a": ts.count(a.a), "n_b": ts.count(a.b), "action": act, "text": text}
            for iid in rec:
                f.write(json.dumps(rec[iid]) + "\n")
            f.flush()
            print(f"[{k + len(chunk)}/{len(jobs)}] {time.time() - t0:.0f} s", flush=True)
    recs = [json.loads(l) for l in open(a.out)]
    for kind in ("orig", "swap"):
        n_a = sum(r[kind]["n_a"] for r in recs); n_b = sum(r[kind]["n_b"] for r in recs)
        print(f"{kind}: QR {np.mean([r[kind]['qr'] for r in recs]):.3f} valid {np.mean([r[kind]['valid'] for r in recs]):.2f} "
              f"share of {a.a} among {a.a}+{a.b}: {n_a / max(1, n_a + n_b):.2f}")
    longer_a = [r for r in recs if r["drange"][a.a] > r["drange"][a.b]]
    print(f"instances where {a.a} has the longer range originally: {len(longer_a)}/{len(recs)}")


if __name__ == "__main__":
    main()
