#!/bin/bash
# Extensions after probe_anon.sh: (1) range-swap probe on named seeds 1 and 2 (~30 min);
# (2) anonymized inventory, GRPO seeds 1 and 2 with evaluations (~12 h). Run from the repository root: ./scripts/extend.sh
set -e
cd "$(dirname "$0")/.."
M=mlx-community/Qwen2.5-7B-Instruct-4bit; PY=.venv/bin/python; RES=results/v2
for s in 1 2; do
  echo "[$(date '+%H:%M:%S')] range-swap probe, named seed $s"
  caffeinate -i $PY -m cuas_rlvr.probe_swap --model $M --adapter ckpt/v2_q7b_grpo_s$s --data data/gym2 --split test_id \
    --out $RES/probe_swap_v2_q7b_grpo_s$s.jsonl 2>&1 | tee -a $RES/logs/probes_v2_q7b.log | grep -E "^(orig|swap)"
done
export CUAS_ANON=1 TAG=v2anon_q7b SFT_SRC=data/sft_train_v2anon.jsonl SFT_DIR=data/mlx_sft_v2anon
for s in 1 2; do ./scripts/run_mac_v2.sh grpo $s; ./scripts/run_mac_v2.sh eval-grpo $s; done
$PY -m cuas_rlvr.probe_types $RES/mlx_v2anon_q7b_grpo_s*_test_sensor.jsonl --ceiling $RES/ceiling_test_sensor_ac.json --type ac --anon | tee -a $RES/logs/probes_v2anon_q7b.log
echo "[$(date '+%H:%M:%S')] DONE"
