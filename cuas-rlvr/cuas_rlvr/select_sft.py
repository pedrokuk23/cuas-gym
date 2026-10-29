"""Choose the SFT checkpoint by the quality of its answers, not by its loss.

Each saved checkpoint generates answers (greedy) for the held-out validation instances of the SFT split; the one with
the highest mean quality ratio (invalid answers count as zero) is copied to --out. A low loss does not prevent
degenerate answers such as repeated placements, which only generation reveals.

    python3 -m cuas_rlvr.select_sft --model ... --data data/gym2 --src data/sft_train_v2.jsonl \
        --ckpt ckpt/v2_q7b_sft --out ckpt/v2_q7b_sft_best
"""
import argparse
import glob
import json
import os
import random
import shutil
import tempfile

import numpy as np

from cuas_gym.cli import load
from .mlx_common import chat_ids, load_model
from cuas_gym.prompt import parse_answer
from cuas_gym.rewards import hybrid_reward


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True); p.add_argument("--data", required=True); p.add_argument("--src", required=True)
    p.add_argument("--ckpt", required=True); p.add_argument("--out", required=True); p.add_argument("--valid", type=int, default=20)
    p.add_argument("--max-tokens", type=int, default=1024); p.add_argument("--min-iter", type=int, default=100)
    a = p.parse_args()
    from mlx_lm.generate import batch_generate
    rows = [json.loads(l) for l in open(a.src)]
    random.Random(0).shuffle(rows)                       # same split as mlx_sft_data.py
    vids = [r["iid"] for r in rows[:a.valid]]
    insts = {i: load(a.data, i) for i in vids}
    files = sorted(glob.glob(os.path.join(a.ckpt, "*_adapters.safetensors")))
    files = [f for f in files if int(os.path.basename(f)[:7]) >= a.min_iter] + [os.path.join(a.ckpt, "adapters.safetensors")]
    results = []
    for f in files:
        tmp = tempfile.mkdtemp()
        shutil.copyfile(f, os.path.join(tmp, "adapters.safetensors"))
        shutil.copyfile(os.path.join(a.ckpt, "adapter_config.json"), os.path.join(tmp, "adapter_config.json"))
        model, tok, _ = load_model(a.model, tmp)
        prompts = [chat_ids(tok, insts[i][0]) for i in vids]
        res = batch_generate(model, tok, prompts, max_tokens=a.max_tokens)
        q, v = [], []
        for i, text in zip(vids, res.texts):
            r = hybrid_reward(insts[i][1], parse_answer(text))
            q.append(r["r_out"]); v.append(r["valid"])
        name = os.path.basename(f)
        results.append((float(np.mean(q)), float(np.mean(v)), name, f))
        print(f"{name}: valid {np.mean(v):.2f}  QR {np.mean(q):.3f}", flush=True)
        del model
    best = max(results)
    os.makedirs(a.out, exist_ok=True)
    shutil.copyfile(best[3], os.path.join(a.out, "adapters.safetensors"))
    shutil.copyfile(os.path.join(a.ckpt, "adapter_config.json"), os.path.join(a.out, "adapter_config.json"))
    with open(os.path.join(a.out, "README.txt"), "w") as fh:
        fh.write(f"Chosen by validation answers: {best[2]} (QR {best[0]:.3f}, valid {best[1]:.2f})\n")
        for r in results:
            fh.write(f"{r[2]}: QR {r[0]:.3f} valid {r[1]:.2f}\n")
    print(f"chosen {best[2]} (QR {best[0]:.3f}, valid {best[1]:.2f})")


if __name__ == "__main__":
    main()
