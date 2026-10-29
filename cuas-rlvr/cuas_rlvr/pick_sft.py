"""Pick the SFT checkpoint with the lowest validation loss before any non-finite loss, from an mlx_lm lora log.

    python3 -m cuas_rlvr.pick_sft --log results/logs/sft.log --ckpt ckpt/v2_q7b_sft --out ckpt/v2_q7b_sft_best
"""
import argparse
import math
import os
import re
import shutil


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--log", required=True); p.add_argument("--ckpt", required=True); p.add_argument("--out", required=True)
    a = p.parse_args()
    txt = re.sub(r"\x1b\[[0-9;]*m", "", open(a.log, errors="ignore").read()).replace("\r", "\n")
    vals, bad = {}, None
    for line in txt.splitlines():
        m = re.match(r"\s*(\d+)\s+val\s+(\S+)", line)
        t = re.match(r"\s*(\d+)\s+(nan|inf|-?\d+\.\d+)\s", line)
        if m:
            it, v = int(m.group(1)), m.group(2)
            if v in ("nan", "inf") or not math.isfinite(float(v)):
                bad = it if bad is None else min(bad, it)
            else:
                vals[it] = float(v)
        elif t and t.group(2) in ("nan", "inf"):
            bad = int(t.group(1)) if bad is None else min(bad, int(t.group(1)))
    cands = []
    for it, v in vals.items():
        f = os.path.join(a.ckpt, f"{it:07d}_adapters.safetensors")
        if os.path.exists(f) and (bad is None or it < bad):
            cands.append((v, it, f))
    final = os.path.join(a.ckpt, "adapters.safetensors")
    if bad is None and vals:
        last = max(vals)
        cands.append((vals[last], last, final))
    if not cands:
        raise SystemExit("no usable SFT checkpoint found")
    v, it, f = min(cands)
    os.makedirs(a.out, exist_ok=True)
    shutil.copyfile(f, os.path.join(a.out, "adapters.safetensors"))
    shutil.copyfile(os.path.join(a.ckpt, "adapter_config.json"), os.path.join(a.out, "adapter_config.json"))
    with open(os.path.join(a.out, "README.txt"), "w") as fh:
        fh.write(f"SFT checkpoint at iteration {it}, validation loss {v:.4f}; first non-finite loss at iteration {bad}\n")
    print(f"picked iteration {it} (val {v:.4f}); first non-finite at {bad}")


if __name__ == "__main__":
    main()
