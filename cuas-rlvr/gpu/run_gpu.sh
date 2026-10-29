#!/bin/bash
# CUAS-Gym v2 on a CUDA server: baseline -> SFT -> checkpoint selection -> GRPO (3 seeds) -> evaluation,
# named inventory first, then the anonymized inventory. Every stage skips work whose output already exists,
# so the script can be re-run after an interruption.
#
#   cd cuas-rlvr && tmux new -s train
#   ./gpu/run_gpu.sh                  # all stages, MODEL=Qwen/Qwen3-8B
#   MODEL=Qwen/Qwen3.8-27B TAG=q38_27b LOAD4BIT=1 SKIP_BASE=1 ./gpu/run_gpu.sh && ./gpu/scale27.sh   # QLoRA for the 27B
#   ./gpu/run_gpu.sh status           # one-line summary of what exists
set -e -o pipefail   # a failing Python stage stops the script (grep alone may match nothing)
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
MODEL=${MODEL:-Qwen/Qwen3-8B}
TAG=${TAG:-q3_8b}
SEEDS=${SEEDS:-"0 1 2"}
STEPS=${STEPS:-300}
LOAD4BIT=${LOAD4BIT:-0}; Q=""; [ "$LOAD4BIT" = 1 ] && Q="--load-4bit"
SKIP_BASE=${SKIP_BASE:-0}   # 1: skip the zero-shot evaluation (e.g. measured elsewhere)
DATA=data/gym2
RES=results/gpu; LOG=$RES/logs; mkdir -p "$LOG" ckpt
SPLITS="test_id test_terrain test_sensor test_scale"
ev() {  # ev <out> <extra args>
  local out=$1; shift
  # complete only when the file has one record per expected instance (20 for checkpoint selection, else the split size)
  local n=0 exp="" prev=""; for x in "$@"; do case "$prev" in --split|--ids) exp=$x;; esac; prev=$x; done
  complete() {  # every expected instance id present (the split's ids, or the held-out ids for checkpoint selection)
    python3 - "$1" "$DATA" "$exp" <<'PY' 2>/dev/null
import json, sys
from cuas_gym.cli import ids
got = {json.loads(l)["iid"] for l in open(sys.argv[1])}
spec = sys.argv[3]
want = set(json.load(open(spec))) if spec.endswith(".json") else set(ids(sys.argv[2], spec))
sys.exit(0 if want and want <= got else 1)
PY
  }
  [ -s "$out" ] && complete "$out" && { echo "   (complete) $out"; return; }
  set -o pipefail
  python3 -m gpu.eval_hf --model "$MODEL" --data $DATA --no-think --max-tokens 2048 --batch 8 $Q --out "$out" "$@" 2>&1 | tee "$LOG/$(basename "$out" .jsonl).log" | tail -1 \
    || { echo "FAILED: $out"; return 1; }
  complete "$out" || { echo "INCOMPLETE: $out"; return 1; }
}

if [ "$1" = status ]; then
  nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader
  for f in $RES/*.jsonl; do [ -f "$f" ] && printf "%-55s %4d\n" "$(basename "$f")" "$(wc -l < "$f")"; done
  for d in ckpt/${TAG}*; do [ -d "$d" ] && echo "$d: $(ls "$d" | tr '\n' ' ')"; done
  exit 0
fi

run_variant() {  # run_variant <tag> <sft jsonl>
  local T=$1 SFT=$2
  echo "=== [$T] baseline ($MODEL, no thinking)"
  [ "$SKIP_BASE" = 1 ] || for sp in $SPLITS; do ev "$RES/${T}_base_${sp}.jsonl" --split $sp; done

  echo "=== [$T] SFT"
  if [ ! -d "ckpt/${T}_sft/final" ]; then
    python3 -m gpu.sft_trl --model "$MODEL" --data "$SFT" --out "ckpt/${T}_sft" --epochs 3 --lr 1e-4 --batch 2 --accum 4 \
      --max-len 4096 --save-every 10 --no-think $Q 2>&1 | tee "$LOG/sft_${T}.log" | { grep -E "train_runtime|Error|error" || :; }
  fi

  echo "=== [$T] checkpoint selection on the valid split"
  if [ ! -e "ckpt/${T}_sft/best" ]; then
    for c in ckpt/${T}_sft/checkpoint-* ckpt/${T}_sft/final; do
      [ -d "$c" ] || continue
      ev "$RES/select_${T}_$(basename "$c").jsonl" --adapter "$c" --ids "ckpt/${T}_sft/valid_ids.json" --max-tokens 1536
    done
    best=$(python3 - "$RES" "$T" <<'PY'
import glob, json, sys, numpy as np
res, t = sys.argv[1], sys.argv[2]
sc = {}
for f in glob.glob(f"{res}/select_{t}_*.jsonl"):
    rs = [json.loads(l) for l in open(f)]
    sc[f.split(f"select_{t}_")[1][:-6]] = np.mean([r["qr"] for r in rs])
for k, v in sorted(sc.items(), key=lambda x: -x[1]): print(k, round(v, 3), file=sys.stderr)
print(max(sc, key=sc.get))
PY
)
    echo "   chosen: $best"; ln -sfn "$best" "ckpt/${T}_sft/best"
  fi

  echo "=== [$T] GRPO, seeds: $SEEDS"
  for s in $SEEDS; do
    out="ckpt/${T}_grpo_s$s"
    if [ ! -d "$out/final" ]; then
      python3 -m gpu.grpo_trl --model "$MODEL" --adapter "ckpt/${T}_sft/best" --data $DATA --out "$out" --reward outcome \
        --steps $STEPS --group 8 --micro 2 --lr 1e-5 --max-tokens 1024 --seed $s --no-think $Q 2>&1 | tee "$LOG/grpo_${T}_s$s.log" | { grep -E "train_runtime|Error|error" || :; }
    fi
    for sp in $SPLITS; do ev "$RES/${T}_grpo_s${s}_${sp}.jsonl" --adapter "$out/final" --split $sp; done
  done
}

run_variant "$TAG" data/sft_train_v2.jsonl
CUAS_ANON=1 run_variant "${TAG}anon" data/sft_train_v2anon.jsonl
echo "=== done: $(date)"
