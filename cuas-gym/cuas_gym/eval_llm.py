"""Zero-/few-shot evaluation of a language model on CUAS-Gym.

Works with any OpenAI-compatible chat endpoint (OpenAI, vLLM / SGLang serving an open model, Ollama, OpenRouter) and
with the Anthropic Messages API. Only the standard library is needed.

Optional connector: it is used only when run by hand from the command line (no training
script calls it). Nothing is sent anywhere unless a provider is configured: either an API key in the
environment (OPENAI_API_KEY / ANTHROPIC_API_KEY, supplied by the user; never stored in the code) or --base-url pointing
to a model server you run (vLLM, SGLang, Ollama). Without one of these the script stops before any request.

    export OPENAI_API_KEY=...            # or ANTHROPIC_API_KEY
    python3 -m cuas_gym.eval_llm --data data/gym --split test_id --provider openai --model gpt-4.1 --n 60 \
        --out results/llm_gpt41_test_id.jsonl
    # open model served locally by vLLM:  vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000
    python3 -m cuas_gym.eval_llm --provider openai --base-url http://localhost:8000/v1 --model Qwen/Qwen2.5-7B-Instruct ...
"""
import argparse
import json
import os
import time
import urllib.request

import numpy as np

from .cli import ids, load
from .prompt import SYSTEM, build_prompt, parse_answer
from .rewards import hybrid_reward


def check_config(provider, base_url):
    """Refuse to run without an explicit configuration: no default service is ever called."""
    key = "ANTHROPIC_API_KEY" if provider == "anthropic" else "OPENAI_API_KEY"
    if base_url:
        if not base_url.startswith(("http://", "https://")):
            raise SystemExit("--base-url must be an http(s) URL")
        if provider == "anthropic" and not os.environ.get(key):
            raise SystemExit(f"provider anthropic needs {key} in the environment")
        return
    if not os.environ.get(key):
        raise SystemExit(f"No LLM provider configured: set {key} (your own key) or pass --base-url of a model server you run. "
                         "Nothing was sent.")


def chat(provider, model, system, user, base_url=None, temperature=0.0, max_tokens=4096):
    check_config(provider, base_url)
    if provider == "anthropic":
        url = (base_url or "https://api.anthropic.com") + "/v1/messages"
        body = {"model": model, "max_tokens": max_tokens, "temperature": temperature, "system": system,
                "messages": [{"role": "user", "content": user}]}
        hdr = {"x-api-key": os.environ["ANTHROPIC_API_KEY"], "anthropic-version": "2023-06-01", "content-type": "application/json"}
    else:
        url = (base_url or "https://api.openai.com/v1") + "/chat/completions"
        body = {"model": model, "temperature": temperature, "max_tokens": max_tokens,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        hdr = {"Authorization": "Bearer " + os.environ.get("OPENAI_API_KEY", "none"), "content-type": "application/json"}
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=hdr)
    for k in range(5):
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                d = json.load(r)
            if provider == "anthropic":
                return "".join(c.get("text", "") for c in d["content"]), d.get("usage", {})
            return d["choices"][0]["message"]["content"], d.get("usage", {})
        except Exception as e:  # noqa: BLE001 (retry on any transport error)
            if k == 4:
                raise
            time.sleep(5 * (k + 1))


def few_shot_block(data, k):
    """k solved training instances (reference solution) shown before the task."""
    out = []
    for iid in ids(data, "train")[:k]:
        inst, v = load(data, iid)
        ans = {"placements": [{"site": s, "type": t} for s, t in v.action_of(v.reference()["rows"])]}
        out.append("Example task:\n" + build_prompt(inst) + "\nExample answer:\n<answer>" + json.dumps(ans) + "</answer>\n")
    return "\n".join(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/gym"); p.add_argument("--split", default="test_id")
    p.add_argument("--provider", choices=["openai", "anthropic"], default="openai"); p.add_argument("--model", required=True)
    p.add_argument("--base-url"); p.add_argument("--n", type=int, default=60); p.add_argument("--shots", type=int, default=0)
    p.add_argument("--samples", type=int, default=1, help="answers per instance (best-of-k uses temperature 1.0)")
    p.add_argument("--no-attrs", action="store_true", help="hide terrain attributes (ablation)")
    p.add_argument("--lam", type=float, default=0.5); p.add_argument("--out", required=True)
    a = p.parse_args()
    check_config(a.provider, a.base_url)
    shots = few_shot_block(a.data, a.shots) if a.shots else ""
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(l)["iid"] + f"#{json.loads(l)['k']}" for l in open(a.out)}
    with open(a.out, "a") as f:
        for iid in ids(a.data, a.split)[:a.n]:
            inst, v = load(a.data, iid)
            user = (shots + "\nNow solve this task:\n" if shots else "") + build_prompt(inst, attrs=not a.no_attrs)
            for k in range(a.samples):
                if f"{iid}#{k}" in done:
                    continue
                t = time.time()
                text, usage = chat(a.provider, a.model, SYSTEM, user, a.base_url, 0.0 if a.samples == 1 else 1.0)
                act = parse_answer(text)
                r = hybrid_reward(v, act, a.lam)
                sc = v.score(act) if r["valid"] else {"Fn": 0.0, "cov90": 0.0}
                rec = {"iid": iid, "k": k, "model": a.model, "valid": r["valid"], "error": r["error"], "qr": r["r_out"],
                       "r_proc": r["r_proc"], "Fn": sc["Fn"], "cov90": sc["cov90"], "action": act, "time_s": time.time() - t,
                       "usage": usage, "text": text}
                f.write(json.dumps(rec) + "\n"); f.flush()
                print(f"{iid}#{k}: valid {r['valid']} QR {r['r_out']:.3f} proc {r['r_proc']:.3f}", flush=True)
    recs = [json.loads(l) for l in open(a.out)]
    q = np.array([r["qr"] for r in recs])
    print(f"n={len(recs)} valid={np.mean([r['valid'] for r in recs]):.3f} QR mean={q.mean():.3f}")


if __name__ == "__main__":
    main()
