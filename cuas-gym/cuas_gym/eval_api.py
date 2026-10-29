"""Evaluation of a model served through an OpenAI-compatible chat endpoint (vLLM, etc.) on CUAS-Gym.

    export CUAS_API_KEY=...            # never put the token in a file of the repository
    python3 -m cuas_gym.eval_api --base-url https://<your-server>/v1 --model qwen3.8-27b-nvfp4 \
        --data data/gym2 --split test_id --out results/api/qwen3p8_27b_think_test_id.jsonl
    ... --no-think                          # chat_template_kwargs.enable_thinking=false (Qwen3 family)

Same record format as eval_mlx.py, so report/report_v2/probe_types read the output unchanged. Greedy by default
(--temp 0); --samples k with --temp 1.0 for best-of-k; --workers concurrent requests; resumes an interrupted run.
Reasoning content returned in a separate field (vLLM reasoning parsers) is kept in "reasoning" and not parsed for the answer.
"""
import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import requests

from .cli import ids, load
from .prompt import SYSTEM, build_prompt, parse_answer
from .rewards import hybrid_reward


def ask(base_url, key, model, msgs, max_tokens, temp, think, timeout):
    """Streams the completion: the Cloudflare tunnel in front of the server cuts idle responses at ~100 s (HTTP 524),
    and a long answer takes longer than that; a stream keeps the connection alive."""
    body = {"model": model, "messages": msgs, "max_tokens": max_tokens, "temperature": temp,
            "stream": True, "stream_options": {"include_usage": True}}
    if think is not None:
        body["chat_template_kwargs"] = {"enable_thinking": bool(think)}
    for attempt in range(5):
        try:
            with requests.post(f"{base_url.rstrip('/')}/chat/completions", json=body, timeout=(30, timeout), stream=True,
                               headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"}) as r:
                if r.status_code in (429, 500, 502, 503, 504, 524):
                    raise RuntimeError(f"HTTP {r.status_code}: {r.text[:120]}")
                r.raise_for_status()
                text, reasoning, finish, usage = [], [], None, {}
                for line in r.iter_lines(decode_unicode=True):
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    d = json.loads(data)
                    if d.get("usage"):
                        usage = d["usage"]
                    for ch in d.get("choices", []):
                        delta = ch.get("delta") or {}
                        if delta.get("content"):
                            text.append(delta["content"])
                        rc = delta.get("reasoning_content") or delta.get("reasoning")
                        if rc:
                            reasoning.append(rc)
                        if ch.get("finish_reason"):
                            finish = ch["finish_reason"]
                return {"text": "".join(text), "reasoning": "".join(reasoning), "finish": finish, "usage": usage}
        except (requests.RequestException, RuntimeError, KeyError, ValueError) as e:
            if attempt == 4:
                return {"text": "", "reasoning": "", "finish": f"error: {str(e)[:160]}", "usage": {}}
            time.sleep(10 * (attempt + 1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default=os.environ.get("CUAS_API_URL"))
    p.add_argument("--model", required=True)
    p.add_argument("--data", default="data/gym2"); p.add_argument("--split", default="test_id")
    p.add_argument("--n", type=int, default=1000); p.add_argument("--samples", type=int, default=1)
    p.add_argument("--temp", type=float, default=0.0); p.add_argument("--max-tokens", type=int, default=8192)
    p.add_argument("--no-think", action="store_true"); p.add_argument("--think", action="store_true")
    p.add_argument("--workers", type=int, default=8); p.add_argument("--timeout", type=float, default=600)
    p.add_argument("--no-attrs", action="store_true"); p.add_argument("--lam", type=float, default=0.5)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    if not a.base_url:
        raise SystemExit("pass --base-url or set CUAS_API_URL (an OpenAI-compatible endpoint)")
    key = os.environ.get("CUAS_API_KEY")
    if not key:
        raise SystemExit("set CUAS_API_KEY in the environment")
    think = False if a.no_think else (True if a.think else None)
    name = a.model + ("" if think is None else ("+think" if think else "+nothink"))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    done = set()
    if os.path.exists(a.out):  # resume: keep the finished answers, drop records that failed with a transport error
        keep = [json.loads(l) for l in open(a.out) if l.strip()]
        keep = [r for r in keep if not str(r.get("finish", "")).startswith("error")]
        with open(a.out, "w") as f:
            for r in keep:
                f.write(json.dumps(r) + "\n")
        done = {(r["iid"], r["k"]) for r in keep}
    jobs = [(iid, k) for iid in ids(a.data, a.split)[:a.n] for k in range(a.samples) if (iid, k) not in done]
    t0 = time.time(); n_done = 0

    def run(job):
        iid, k = job
        inst, v = load(a.data, iid)
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": build_prompt(inst, attrs=not a.no_attrs)}]
        t = time.time()
        o = ask(a.base_url, key, a.model, msgs, a.max_tokens, a.temp, think, a.timeout)
        text = o["text"]
        act = parse_answer(text)
        r = hybrid_reward(v, act, a.lam)
        sc = v.score(act) if r["valid"] else {"Fn": 0.0, "cov90": 0.0}
        u = o["usage"]
        return {"iid": iid, "k": k, "model": name, "valid": r["valid"], "error": r["error"], "qr": r["r_out"],
                "r_proc": r["r_proc"], "Fn": sc["Fn"], "cov90": sc["cov90"], "action": act,
                "n_tokens": u.get("completion_tokens", 0), "prompt_tokens": u.get("prompt_tokens", 0),
                "finish": o["finish"], "time_s": time.time() - t, "text": text, "reasoning": o["reasoning"]}

    with open(a.out, "a") as f, ThreadPoolExecutor(a.workers) as ex:
        futs = [ex.submit(run, j) for j in jobs]
        for fu in as_completed(futs):
            rec = fu.result(); f.write(json.dumps(rec) + "\n"); f.flush(); n_done += 1
            if n_done % a.workers == 0 or n_done == len(jobs):
                recs = [json.loads(l) for l in open(a.out)]
                print(f"[{n_done}/{len(jobs)}] {time.time() - t0:.0f} s, mean QR so far {np.mean([r['qr'] for r in recs]):.3f}, "
                      f"valid {np.mean([r['valid'] for r in recs]):.2f}, tokens {np.mean([r['n_tokens'] for r in recs]):.0f}", flush=True)
    recs = [json.loads(l) for l in open(a.out)]
    trunc = sum(r["finish"] == "length" for r in recs); errs = sum(str(r["finish"]).startswith("error") for r in recs)
    print(f"{a.split}: n={len(recs)} valid={np.mean([r['valid'] for r in recs]):.3f} QR mean={np.mean([r['qr'] for r in recs]):.3f} "
          f"tokens={np.mean([r['n_tokens'] for r in recs]):.0f} truncated={trunc} errors={errs}")


if __name__ == "__main__":
    main()
