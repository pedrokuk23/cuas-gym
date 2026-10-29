#!/bin/bash
# Range-swap probe and exploration probe on the anonymized-inventory model (~1 h)
cd "$(dirname "$0")/.."; export CUAS_ANON=1
M=mlx-community/Qwen2.5-7B-Instruct-4bit; PY=.venv/bin/python; RES=results/v2
caffeinate -i $PY -m cuas_rlvr.probe_swap --model $M --adapter ckpt/v2anon_q7b_grpo_s0 --data data/gym2 --split test_id \
  --out $RES/probe_swap_v2anon_q7b_grpo_s0.jsonl 2>&1 | tee -a $RES/logs/probes_v2anon_q7b.log | grep -E "^(orig|swap|[0-9]+ inst)"
caffeinate -i $PY -m cuas_rlvr.eval_mlx --model $M --adapter ckpt/v2anon_q7b_grpo_s0 --data data/gym2 --split test_sensor \
  --samples 8 --temp 1.0 --out $RES/probe_explore_v2anon_q7b_grpo_s0.jsonl 2>&1 | tee -a $RES/logs/probes_v2anon_q7b.log | tail -1
$PY -m cuas_rlvr.probe_types $RES/probe_explore_v2anon_q7b_grpo_s0.jsonl --ceiling $RES/ceiling_test_sensor_ac.json --type ac --anon | tee -a $RES/logs/probes_v2anon_q7b.log
echo DONE
