#!/bin/bash
# Re-evaluates the Qwen3.8-27B adapters on test_scale with a small batch (batch 8 runs out of memory on a 32 GB GPU
# for the larger instances). Run from the repository root after gpu/run_gpu.sh.
cd "$(dirname "$0")/.."
LOG=results/gpu_logs/scale27.log; mkdir -p results/gpu_logs
for v in q38_27b q38_27banon; do
  for s in 0 1 2; do
    out=results/gpu/${v}_grpo_s${s}_test_scale.jsonl
    if [ -s "$out" ] && [ "$(wc -l < "$out")" -ge 40 ]; then echo "exists $out" >> $LOG; continue; fi
    rm -f "$out"
    A=""; [ $v = q38_27banon ] && A="CUAS_ANON=1"
    echo "=== $v s$s $(date)" >> $LOG
    env $A PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 -m gpu.eval_hf --model Qwen/Qwen3.8-27B \
      --adapter ckpt/${v}_grpo_s${s}/final --data data/gym2 --split test_scale --no-think --max-tokens 2048 \
      --batch 2 --load-4bit --out "$out" >> $LOG 2>&1
  done
done
