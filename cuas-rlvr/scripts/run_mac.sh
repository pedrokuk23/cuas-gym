#!/bin/bash
# CUAS-Gym on Apple silicon (MLX). Run from the Terminal app on the Mac (not in a VM):
#   cd cuas-rlvr   (repository root; run as ./scripts/<name>.sh)
#   ./run_mac.sh setup        # once: virtual environment with MLX
#   ./run_mac.sh smoke        # 5 instances, checks everything and prints the speed
#   ./run_mac.sh all          # baseline -> SFT -> GRPO -> evaluations (many hours; the Mac must stay on)
# Single stages: baseline | train (all but baseline) | after-sft | sft | eval-sft | grpo | eval-grpo | grpo-outcome | eval-grpo-outcome | status
# Other model:  MODEL=mlx-community/Qwen2.5-1.5B-Instruct-4bit TAG=q1p5 ./run_mac.sh all
set -e
cd "$(dirname "$0")/.."
MODEL="${MODEL:-mlx-community/Qwen2.5-7B-Instruct-4bit}"
TAG="${TAG:-q7b}"
STEPS="${STEPS:-300}"
SFT_ITERS="${SFT_ITERS:-560}"
PY=".venv/bin/python"
mkdir -p results/logs ckpt
SPLITS="test_id test_terrain test_sensor test_scale"
KEEP="caffeinate -i"
# SFT adapter used by the later stages: the best checkpoint when it exists (ckpt/<tag>_sft_best), else the final one
if [ -z "$SFT_AD" ]; then
  if [ -d "ckpt/${TAG}_sft_best" ]; then SFT_AD="ckpt/${TAG}_sft_best"; else SFT_AD="ckpt/${TAG}_sft"; fi
fi   # keeps the Mac awake while a stage runs

log() { echo "[$(date '+%H:%M:%S')] $*"; }

evaluate() {  # $1 = label, $2 = adapter folder or empty
  for s in $SPLITS; do
    log "evaluating $1 on $s"
    if [ -n "$2" ]; then AD="--adapter $2"; else AD=""; fi
    $KEEP $PY -m cuas_rlvr.eval_mlx --model "$MODEL" $AD --split "$s" --out "results/mlx_${TAG}_$1_$s.jsonl" \
      2>&1 | tee -a "results/logs/eval_${TAG}_$1.log" | grep -E "^test|mean QR" | tail -1
  done
}

case "$1" in
  setup)
    command -v python3 >/dev/null || { echo "python3 not found: install it (brew install python@3.12)"; exit 1; }
    python3 -c 'import sys; assert sys.version_info >= (3, 10), "Python 3.10+ needed: brew install python@3.12"'
    [ "$(uname -m)" = "arm64" ] || { echo "run this on the Mac itself (Apple silicon), not in a VM"; exit 1; }
    python3 -m venv .venv
    $PY -m pip install -q --upgrade pip
    $PY -m pip install -q "mlx-lm==0.32.2" numpy pyyaml
    $PY -c "import mlx.core as mx; print('MLX', mx.__version__, mx.default_device())"
    log "downloading $MODEL (about 4 GB for the 7B model)"
    $PY -c "from mlx_lm import load; load('$MODEL'); print('model ok')"
    ;;
  smoke)
    $PY -m cuas_gym.tests.test_gym | tail -1
    $PY -m cuas_rlvr.eval_mlx --model "$MODEL" --split test_id --n 5 --batch 5 --out "results/smoke_${TAG}_$(date +%m%d%H%M).jsonl"
    log "smoke test done; answers in results/smoke_${TAG}_*.jsonl"
    ;;
  baseline) evaluate base "" ;;
  sft)
    $PY -m cuas_gym.sft_data --data data/gym --out data/sft_train.jsonl
    $PY -m cuas_rlvr.mlx_sft_data --src data/sft_train.jsonl --out data/mlx_sft --valid 20
    log "SFT: $SFT_ITERS iterations"
    $KEEP $PY -m mlx_lm lora -c cuas_rlvr/mlx_lora.yaml --model "$MODEL" --train --data data/mlx_sft \
      --iters "$SFT_ITERS" --batch-size 1 --num-layers 16 --mask-prompt --learning-rate "${SFT_LR:-1e-5}" --max-seq-length 3072 \
      --steps-per-eval 50 --val-batches 20 --save-every 100 --adapter-path "ckpt/${TAG}_sft" 2>&1 | tee "results/logs/sft_${TAG}.log"
    if grep -q " nan" "results/logs/sft_${TAG}.log"; then echo "SFT diverged (NaN loss): stopping before GRPO"; exit 1; fi
    ;;
  eval-sft) evaluate sft "$SFT_AD" ;;
  grpo)
    log "GRPO (hybrid reward): $STEPS steps"
    $KEEP $PY -m cuas_rlvr.grpo_mlx --model "$MODEL" --adapter "$SFT_AD" --reward hybrid --lam 0.5 \
      --steps "$STEPS" --out "ckpt/${TAG}_grpo" 2>&1 | tee -a "results/logs/grpo_${TAG}.log"
    ;;
  eval-grpo) evaluate grpo "ckpt/${TAG}_grpo" ;;
  grpo-outcome)
    log "GRPO (outcome reward, ablation): $STEPS steps"
    $KEEP $PY -m cuas_rlvr.grpo_mlx --model "$MODEL" --adapter "$SFT_AD" --reward outcome \
      --steps "$STEPS" --out "ckpt/${TAG}_grpo_outcome" 2>&1 | tee -a "results/logs/grpo_outcome_${TAG}.log"
    ;;
  eval-grpo-outcome) evaluate grpo_outcome "ckpt/${TAG}_grpo_outcome" ;;
  after-sft)
    log "using SFT adapter $SFT_AD"
    "$0" eval-sft; "$0" grpo; "$0" eval-grpo; "$0" grpo-outcome; "$0" eval-grpo-outcome
    log "stages after SFT done"
    ;;
  train)
    "$0" sft; "$0" eval-sft; "$0" grpo; "$0" eval-grpo; "$0" grpo-outcome; "$0" eval-grpo-outcome
    log "training stages done"
    ;;
  all)
    "$0" baseline; "$0" sft; "$0" eval-sft; "$0" grpo; "$0" eval-grpo; "$0" grpo-outcome; "$0" eval-grpo-outcome
    log "all stages done"
    ;;
  status)
    for f in results/mlx_*.jsonl; do [ -f "$f" ] && $PY -c "
import json,sys,numpy as np; r=[json.loads(l) for l in open('$f')]
print(f\"{'$f':60s} n={len(r):4d} valid={np.mean([x['valid'] for x in r]):.2f} QR={np.mean([x['qr'] for x in r]):.3f}\")"; done
    for d in ckpt/*_grpo*; do [ -f "$d/log.jsonl" ] && echo "$d: $(wc -l < "$d/log.jsonl") steps, last: $(tail -1 "$d/log.jsonl" | cut -c1-160)"; done
    ;;
  *) sed -n 2,9p "$0" ;;
esac
