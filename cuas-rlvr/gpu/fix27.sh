#!/bin/bash
# Control experiment: 27B, property-based inventory, GRPO with truncated answers kept in the loss (reward 0)
# instead of masked; everything else as in run_gpu.sh (same SFT start, seeds, 300 steps, 1024 tokens, NF4).
set -o pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false CUAS_ANON=1
MODEL=Qwen/Qwen3.8-27B; T=q38_27banon; DATA=data/gym2; RES=results/gpu; LOG=$RES/logs; mkdir -p $LOG
for s in 0 1 2; do
  out=ckpt/${T}_grpofix_s$s
  if [ ! -d $out/final ]; then
    echo "train seed $s $(date)"
    python3 -m gpu.grpo_trl --model $MODEL --adapter ckpt/${T}_sft/best --data $DATA --out $out --reward outcome \
      --steps 300 --group 8 --micro 2 --lr 1e-5 --max-tokens 1024 --seed $s --no-think --load-4bit --penalize-truncated \
      > $LOG/grpofix_${T}_s$s.log 2>&1 || { echo "FAIL train $s"; continue; }
  fi
  for sp in test_id test_terrain test_sensor test_scale; do
    b=8; [ $sp = test_scale ] && b=2
    f=$RES/${T}_grpofix_s${s}_${sp}.jsonl
    [ -s $f ] && [ $(wc -l < $f) -ge 40 ] && continue
    python3 -m gpu.eval_hf --model $MODEL --data $DATA --no-think --max-tokens 2048 --batch $b --load-4bit \
      --adapter $out/final --split $sp --out $f > $LOG/$(basename $f .jsonl).log 2>&1 || echo "FAIL eval $s $sp"
  done
  echo "done seed $s $(date)"
done
echo "ALLDONE $(date)"
