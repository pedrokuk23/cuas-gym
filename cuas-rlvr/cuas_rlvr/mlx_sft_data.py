"""Writes the SFT data in the folder layout of mlx_lm.lora (train.jsonl / valid.jsonl with "messages")."""
import argparse
import json
import os
import random


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", default="data/sft_train.jsonl"); p.add_argument("--out", default="data/mlx_sft")
    p.add_argument("--valid", type=int, default=20)
    a = p.parse_args()
    rows = [json.loads(l) for l in open(a.src)]
    random.Random(0).shuffle(rows)
    os.makedirs(a.out, exist_ok=True)
    for name, part in (("valid", rows[:a.valid]), ("train", rows[a.valid:])):
        with open(os.path.join(a.out, name + ".jsonl"), "w") as f:
            for r in part:
                f.write(json.dumps({"messages": r["messages"]}) + "\n")
    print(f"{len(rows) - a.valid} train, {a.valid} valid -> {a.out}")


if __name__ == "__main__":
    main()
