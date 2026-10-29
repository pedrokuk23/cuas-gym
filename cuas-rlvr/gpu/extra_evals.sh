#!/bin/bash
# Extra evaluations (no training):
#   1. Qwen3.8-27B zero-shot with the same 4-bit NF4 backend used for the trained 27B adapters (the paper's zero-shot
#      row was measured on an NVFP4 vLLM server), so that training and backend effects can be separated.
#   2. Test-split scores of the selected 27B SFT checkpoints (names and properties), missing from the paper.
#   3. Single-request latency (batch 1) on the first 20 in-distribution instances, for the trained 8B and 27B and the
#      untrained 27B, to replace the batch-amortized times.
# Run from the repository root, in tmux, after `git pull`:   ./gpu/extra_evals.sh 2>&1 | tee results/gpu_logs/extra_evals.log
# Every step resumes; a step is complete only when its file has one line per expected instance.
set -u
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
M27=Qwen/Qwen3.8-27B; M8=Qwen/Qwen3-8B; DATA=data/gym2; RES=results/gpu_extra; mkdir -p $RES results/gpu_logs
SPLITS="test_id test_terrain test_sensor test_scale"
expected() { python3 -c "from cuas_gym.cli import ids; import sys; print(len(ids('$DATA', sys.argv[1])))" "$1"; }
complete() { [ -s "$1" ] && python3 -c "import json,sys; ids={json.loads(l)['iid'] for l in open(sys.argv[1])}; sys.exit(0 if len(ids)>=int(sys.argv[2]) else 1)" "$1" "$2" 2>/dev/null; }  # unique ids, valid JSON
ev() {  # ev <out> <expected lines> <args...>
  local out=$1 n=$2; shift 2
  if complete "$out" "$n"; then echo "   (complete) $out"; return 0; fi
  echo "=== $(date '+%F %T') $out"
  python3 -m gpu.eval_hf --data $DATA --no-think --out "$out" "$@" || { echo "FAILED: $out"; return 1; }
  complete "$out" "$n" || { echo "INCOMPLETE: $out ($(wc -l < "$out") of $n)"; return 1; }
}
fail=0
# 2. SFT checkpoints of the 27B (short answers)
for v in q38_27b q38_27banon; do
  if [ $v = q38_27banon ]; then export CUAS_ANON=1; else export CUAS_ANON=0; fi
  for sp in $SPLITS; do
    n=$(expected $sp); b=8; [ $sp = test_scale ] && b=2
    ev $RES/${v}_sft_${sp}.jsonl $n --model $M27 --adapter ckpt/${v}_sft/best --load-4bit --split $sp --max-tokens 2048 --batch $b || fail=1
  done
done
export CUAS_ANON=0
# 1. Zero-shot 27B on NF4, same 10240-token budget as the vLLM run (long answers: small batches)
for sp in $SPLITS; do
  n=$(expected $sp); b=4; [ $sp = test_scale ] && b=2
  ev $RES/q38_27b_base_nf4_${sp}.jsonl $n --model $M27 --load-4bit --split $sp --max-tokens 10240 --batch $b || fail=1
done
# 3. Single-request latency, 20 in-distribution instances
python3 -c "from cuas_gym.cli import ids; import json; json.dump(ids('$DATA','test_id')[:20], open('$RES/lat_ids.json','w'))"
ev $RES/lat1_q3_8b_grpo_s0.jsonl 20 --model $M8 --adapter ckpt/q3_8b_grpo_s0/final --ids $RES/lat_ids.json --max-tokens 2048 --batch 1 || fail=1
ev $RES/lat1_q38_27b_grpo_s0.jsonl 20 --model $M27 --adapter ckpt/q38_27b_grpo_s0/final --load-4bit --ids $RES/lat_ids.json --max-tokens 2048 --batch 1 || fail=1
ev $RES/lat1_q38_27b_base_nf4.jsonl 20 --model $M27 --load-4bit --ids $RES/lat_ids.json --max-tokens 10240 --batch 1 || fail=1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader > $RES/gpu_info.txt
python3 -c "import torch,transformers,peft,bitsandbytes as b; print('torch',torch.__version__,'transformers',transformers.__version__,'peft',peft.__version__,'bnb',b.__version__)" >> $RES/gpu_info.txt
tar czf results/gpu_extra.tgz $RES results/gpu_logs/extra_evals.log 2>/dev/null
[ $fail = 0 ] && echo "=== all extra evaluations complete: results/gpu_extra.tgz" || echo "=== some runs failed or are incomplete; re-run the script to resume"
