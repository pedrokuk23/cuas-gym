#!/bin/bash
# Zero-shot baseline of a served model on CUAS-Gym v2 (4 splits), thinking off and on.
#   export CUAS_API_KEY=<token>   (in the Terminal, not in a file)
#   ./run_api_base.sh [smoke]         smoke = 2 instances per split, thinking off only
set -e
cd "$(dirname "$0")/.."
: "${CUAS_API_KEY:?export CUAS_API_KEY first}"
URL=${CUAS_API_URL:?export CUAS_API_URL (OpenAI-compatible endpoint, e.g. https://host/v1)}
MODEL=${MODEL:-qwen3.8-27b-nvfp4}
PY=${PY:-.venv/bin/python}
TAG=${TAG:-q38_27b}
N=1000; MODES="nothink think"
[ "$1" = smoke ] && { N=2; MODES="nothink"; }
mkdir -p results/api/logs
for mode in $MODES; do
  if [ $mode = think ]; then flag=--think; mt=24576; else flag=--no-think; mt=10240; fi
  for sp in test_id test_terrain test_sensor test_scale; do
    out=results/api/${TAG}_base_${mode}_${sp}.jsonl
    echo "== $mode $sp -> $out"
    $PY -m cuas_gym.eval_api --base-url "$URL" --model "$MODEL" --data data/gym2 --split $sp --n $N \
      $flag --max-tokens $mt --workers 8 --out "$out" 2>&1 | tee results/api/logs/${TAG}_base_${mode}_${sp}.log | tail -1
  done
done
