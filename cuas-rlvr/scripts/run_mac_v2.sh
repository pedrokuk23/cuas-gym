#!/bin/bash
# CUAS-Gym v2 (sensor variants with data-sheet ranges) on Apple silicon. From the Terminal app on the Mac:
#   cd cuas-rlvr   (repository root; run as ./scripts/<name>.sh)
#   ./run_mac_v2.sh all        # baseline, SFT, GRPO (outcome reward, seeds 0 1 2), evaluations (grpo-hybrid: optional)
#   ./run_mac_v2.sh status
# Single stages: baseline | sft | select-sft | from-sft | after-sft | eval-sft | grpo SEED | eval-grpo SEED | grpo-hybrid | eval-grpo-hybrid | seeds
#   probes | anon | after-seeds   (diagnostics of the unseen-sensor result and the anonymized-inventory experiment)
# Stages resume: finished evaluations are skipped, and a GRPO run with a finished adapter is not repeated.
set -e
cd "$(dirname "$0")/.."
MODEL="${MODEL:-mlx-community/Qwen2.5-7B-Instruct-4bit}"
TAG="${TAG:-v2_q7b}"
DATA=data/gym2
RES=results/v2
STEPS="${STEPS:-300}"
SFT_ITERS="${SFT_ITERS:-560}"
PY=".venv/bin/python"
KEEP="caffeinate -i"
SPLITS="test_id test_terrain test_sensor test_scale"
mkdir -p "$RES/logs" ckpt
SFT_AD="ckpt/${TAG}_sft_best"
SFT_SRC="${SFT_SRC:-data/sft_train_v2.jsonl}"
SFT_DIR="${SFT_DIR:-data/mlx_sft_v2}"
log() { echo "[$(date '+%H:%M:%S')] $*"; }

evaluate() {  # $1 label, $2 adapter (or empty)
  for s in $SPLITS; do
    log "evaluating $1 on $s"
    if [ -n "$2" ]; then AD="--adapter $2"; else AD=""; fi
    $KEEP $PY -m cuas_rlvr.eval_mlx --model "$MODEL" $AD --data "$DATA" --split "$s" --out "$RES/mlx_${TAG}_$1_$s.jsonl" \
      2>&1 | tee -a "$RES/logs/eval_${TAG}_$1.log" | grep -E "^test" | tail -1
  done
}

grpo_run() {  # $1 reward, $2 seed, $3 out dir
  if [ -f "$3/step$(printf %04d "$STEPS")/adapters.safetensors" ]; then log "$3 already finished"; return; fi
  log "GRPO ($1 reward, seed $2): $STEPS steps -> $3"
  $KEEP $PY -m cuas_rlvr.grpo_mlx --model "$MODEL" --adapter "$SFT_AD" --data "$DATA" --reward "$1" --lam 0.5 \
    --seed "$2" --steps "$STEPS" --out "$3" 2>&1 | tee -a "$RES/logs/grpo_${TAG}_$1_s$2.log"
}

case "$1" in
  baseline) evaluate base "" ;;
  sft)
    $PY -m cuas_gym.sft_data --data "$DATA" --out "$SFT_SRC"
    $PY -m cuas_rlvr.mlx_sft_data --src "$SFT_SRC" --out "$SFT_DIR" --valid 20
    log "SFT: $SFT_ITERS iterations (own MLX trainer: float32 loss, gradient clipping, skips non-finite updates)"
    $KEEP $PY -m cuas_rlvr.sft_mlx --model "$MODEL" --data "$SFT_DIR" --iters "$SFT_ITERS" --lr 1e-5 \
      --out "ckpt/${TAG}_sft" 2>&1 | tee "$RES/logs/sft_${TAG}.log"
    "$0" select-sft
    ;;
  select-sft)
    log "choosing the SFT checkpoint by the quality of its validation answers"
    $KEEP $PY -m cuas_rlvr.select_sft --model "$MODEL" --data "$DATA" --src "$SFT_SRC" \
      --ckpt "ckpt/${TAG}_sft" --out "$SFT_AD" 2>&1 | tee "$RES/logs/select_sft_${TAG}.log" | grep -E "valid|chosen"
    ;;
  from-sft)
    "$0" sft; "$0" eval-sft; "$0" seeds
    log "all v2 stages from SFT done" ;;
  after-sft)
    "$0" select-sft; "$0" eval-sft; "$0" seeds
    log "all v2 stages after SFT done" ;;
  eval-sft) evaluate sft "$SFT_AD" ;;
  grpo) grpo_run outcome "${2:-0}" "ckpt/${TAG}_grpo_s${2:-0}" ;;
  eval-grpo) evaluate "grpo_s${2:-0}" "ckpt/${TAG}_grpo_s${2:-0}" ;;
  grpo-hybrid) grpo_run hybrid 0 "ckpt/${TAG}_grpo_hybrid_s0" ;;
  eval-grpo-hybrid) evaluate grpo_hybrid_s0 "ckpt/${TAG}_grpo_hybrid_s0" ;;
  seeds)
    for sd in 0 1 2; do "$0" grpo $sd; "$0" eval-grpo $sd; done ;;
  all)
    "$0" baseline; "$0" sft; "$0" eval-sft; "$0" seeds
    log "all v2 stages done" ;;
  probes)   # diagnostics on the trained models (~1.5 h): ceilings, range-swap probe, exploration probe
    $PY -m cuas_gym.ceiling --data "$DATA" --split test_sensor --type ac --out "$RES/ceiling_test_sensor_ac.json"
    $PY -m cuas_gym.ceiling --data "$DATA" --split test_id --type radar --out "$RES/ceiling_test_id_radar.json"
    for ad in "ckpt/${TAG}_grpo_s0:grpo_s0" "$SFT_AD:sft"; do
      log "range-swap probe (${ad#*:})"
      $KEEP $PY -m cuas_rlvr.probe_swap --model "$MODEL" --adapter "${ad%%:*}" --data "$DATA" --split test_id \
        --out "$RES/probe_swap_${TAG}_${ad#*:}.jsonl" 2>&1 | tee -a "$RES/logs/probes_${TAG}.log" | grep -E "^(orig|swap|instances)"
    done
    log "exploration probe: 8 samples at temperature 1 on the unseen-sensor split"
    $KEEP $PY -m cuas_rlvr.eval_mlx --model "$MODEL" --adapter "ckpt/${TAG}_grpo_s0" --data "$DATA" --split test_sensor \
      --samples 8 --temp 1.0 --out "$RES/probe_explore_${TAG}_grpo_s0.jsonl" 2>&1 | tee -a "$RES/logs/probes_${TAG}.log" | tail -1
    $PY -m cuas_rlvr.probe_types "$RES/probe_explore_${TAG}_grpo_s0.jsonl" $RES/mlx_${TAG}_*_test_sensor.jsonl \
      --ceiling "$RES/ceiling_test_sensor_ac.json" --type ac 2>&1 | tee -a "$RES/logs/probes_${TAG}.log"
    ;;
  anon)     # anonymized inventory (letters + properties): base, SFT, eval, GRPO seed 0, eval (~10 h); seeds 1 and 2: scripts/extend.sh
    export CUAS_ANON=1 TAG=v2anon_q7b SFT_SRC=data/sft_train_v2anon.jsonl SFT_DIR=data/mlx_sft_v2anon
    "$0" baseline; "$0" sft; "$0" eval-sft; "$0" grpo 0; "$0" eval-grpo 0
    $PY -m cuas_rlvr.probe_types $RES/mlx_${TAG}_*_test_sensor.jsonl --ceiling "$RES/ceiling_test_sensor_ac.json" --type ac --anon
    log "anonymized-inventory stages done" ;;
  after-seeds)  # once the three seeds are done: redo the truncated scale evaluations, probes, then the anonymized run
    "$0" baseline; "$0" eval-sft; "$0" probes; "$0" anon ;;
  status)
    for f in $RES/mlx_*.jsonl; do [ -f "$f" ] && $PY -c "
import json,numpy as np; r=[json.loads(l) for l in open('$f')]
print(f\"{'$f':62s} n={len(r):4d} valid={np.mean([x['valid'] for x in r]):.2f} QR={np.mean([x['qr'] for x in r]):.3f}\")"; done
    for d in ckpt/${TAG}_grpo*; do [ -f "$d/log.jsonl" ] && echo "$d: $(wc -l < "$d/log.jsonl") steps"; done
    ;;
  *) sed -n 2,7p "$0" ;;
esac
