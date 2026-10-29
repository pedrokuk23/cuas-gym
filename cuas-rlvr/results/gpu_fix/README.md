Control run of the supplementary material (truncated answers on the GPU): the three property-based Qwen3.8-27B GRPO
seeds trained again with answers cut at the 1024-token limit kept in the loss at zero reward (`gpu/fix27.sh`,
`gpu/grpo_trl.py --penalize-truncated`), evaluated with greedy decoding and 2048 tokens.

- `q38_27banon_grpofix_s<seed>_<split>.jsonl`: every answer of the control runs on the four test splits (same record
  format as `results/v2/`); `logs/`: training and evaluation logs.
- `training_stats.txt`: output of `python3 -m cuas_rlvr.truncation_report --ckpt ckpt` (share of sampled answers cut at
  the limit, sampled validity and mean answer length before and after step 120, all GPU GRPO runs); it reads the TRL
  `trainer_state.json` and `log.jsonl` in `ckpt/<run>/`.
- `fixdump.txt`: compact view, one line per test instance: split-id, QR of seeds 0-2, validity of seeds 0-2, answer tokens of seeds 0-2.
- `summary.json`: per-split seed mean, seed SD, bootstrap 95% CI, paired differences from the original runs and from the
  completed geometric heuristic (10,000 resamples of instances).

