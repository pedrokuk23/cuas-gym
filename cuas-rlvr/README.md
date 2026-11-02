# CUAS-RLVR: code and data of the article

Code, training data, model outputs and training logs that reproduce the experiments of

> P. Kukulka de Albuquerque, A. R. Kuroswiski, L. Galembeck and E. C. de F. Morais, *Learning to Plan Heterogeneous
> Counter-UAS Sensor Networks Over Terrain With Physics-Verified Rewards*, submitted to the IEEE Open Journal of the Computer Society, 2026.

The environment (instances, verifier, rewards, prompts, baselines, physics module) is the `cuas-gym/` folder of the
same repository; this folder holds everything built on top of it: supervised fine-tuning (SFT) and group relative policy optimization
(GRPO) with the verifier as reward, on Apple silicon (MLX) and on a CUDA GPU (PyTorch, TRL, PEFT), the evaluation and
probe scripts, the scripts that write the article's tables and figures, and every model output behind the numbers in
the article.

## Setup

```
git clone https://github.com/pedrokuk23/cuas-gym.git cuas
pip install -e cuas/cuas-gym
cd cuas/cuas-rlvr      # all commands below run from this folder
# data/gym and data/gym2 are relative links to ../cuas-gym/data/gym and ../cuas-gym/data/gym2 (instances stored once);
# without symbolic links, pass --data ../cuas-gym/data/gym2 instead
```

Re-creating the tables and the statistics from the released outputs needs only numpy (and matplotlib for the
figures); training needs the requirements below.

## Where each number of the article comes from

| Article element | Source (paths relative to `cuas-rlvr/`) |
|---|---|
| Table 1 (sensor models) | equations and parameters in the supplementary material, Section S1 and Table S1; detection matrices in `../cuas-gym/data/gym2` |
| Table 2 (training configurations) | `scripts/run_mac_v2.sh` (Qwen2.5-7B), `gpu/run_gpu.sh` (Qwen3-8B, Qwen3.8-27B), `cuas_rlvr/mlx_lora.yaml` |
| Table 3 (planning quality with CIs; validity, seed SD and matrix lookups in Tables S5, S9, S10, S12, S13) | `paper_artifacts/tables/v2/baselines.tex`, `llm*.tex`; regenerate with `python3 -m cuas_gym.cli baselines ...` and the `report_v2` commands below; the 27B zero-shot row (NF4 backend) is printed by `report_extra` |
| Table 4 (answer length, latency, amortized time) | `python3 -m cuas_rlvr.cost_table --results results/v2 --latency results/latency --out out/tables/v2` writes `cost.tex` (identical to `paper_artifacts/tables/v2/cost.tex`) from `results/latency/lat1_*.jsonl` (batch 1) and `n_tokens`/`time_s` of `results/v2/*.jsonl` (batched); 27B NF4 zero-shot and SFT outputs: `results/v2/mlx_q38_27b_basenf4_*`, `mlx_q38_27b*_sft_*` (from `gpu/extra_evals.sh`, raw outputs in `results/gpu_extra/`) |
| Table 5 (use of the unseen acoustic array) | `report_v2` output (decisive instances), `cuas_rlvr.probe_types`, `results/v2/baselines_test_sensor.csv` |
| Table 6 (probes) | `scripts/run_mac_v2.sh probes`, `scripts/probe_anon.sh`, `scripts/extend.sh`; outputs `results/v2/probe_*.jsonl` |
| Fig. 1 (unseen-sensor instance, 7B planners) | `python3 -m cuas_rlvr.figure_map_rationale --data data/gym2 --iid test_sensor-000000 --panel "Reference=ref" --panel "7B, names=named:results/v2/mlx_v2_q7b_grpo_s0_test_sensor.jsonl" --panel "7B, properties=anon:results/v2/mlx_v2anon_q7b_grpo_s0_test_sensor.jsonl" --out paper_artifacts/figures/map_7b.pdf` (needs the Copernicus tile) |
| Fig. 2 (unseen-terrain instance, 27B planners and completed heuristic) | same script, `--iid test_terrain-000002 --panel "27B, names=named:results/v2/mlx_q38_27b_grpo_s0_test_terrain.jsonl" --panel "27B, properties=anon:results/v2/mlx_q38_27banon_grpo_s0_test_terrain.jsonl" --panel "Geometric, completed=geo" --out paper_artifacts/figures/map_27b.pdf` |
| Fig. 3 (training curves) | `python3 -m cuas_rlvr.figure_curves --ckpt ckpt --mark q38_27banon:0 --out training_curves_all.pdf` |
| Truncation control (supplementary, property-based 27B) | `gpu/fix27.sh` (GRPO with `--penalize-truncated`); answers of the three control seeds in `results/gpu_fix/*.jsonl`; training statistics `python3 -m cuas_rlvr.truncation_report --ckpt ckpt` (`results/gpu_fix/training_stats.txt`) and their summary with paired bootstrap intervals in `results/gpu_fix/summary.json` |
| Learned surrogate + greedy (Table 3, Table 5, Tables S6--S7) | `python3 -m cuas_rlvr.surrogate --mode properties --seed 0` (seeds 0-2; also `--mode names`), outputs `results/v2/surrogate/surrogate_<mode>_s<seed>.csv`; tables and results by threat: `python3 -m cuas_rlvr.surrogate_report --results results/v2 --data data/gym2 --out out/tables/v2` (needs scikit-learn) |
| Table S11 (27B zero-shot, NVFP4/vLLM) | `report_extra` writes `q38.tex` (identical to `paper_artifacts/tables/v2/q38.tex`) |
| Table S13 (27B zero-shot on two serving backends) | printed by `report_extra` |
| Radar detection curve, sensitivity (supplementary, Section S1) | `python3 -m cuas_rlvr.radar_sensitivity compute --split <split> --matrices ../radar_exact` (each test split; needs the Copernicus tile and scipy), then `python3 -m cuas_rlvr.radar_sensitivity compare --matrices ../radar_exact --out results/v2/radar_sensitivity.json`; released result: `results/v2/radar_sensitivity.json` |
| Paired differences and bootstrap intervals quoted in the text | printed by the `report_v2` commands (`mean_vs_geo`, `mean_vs_geo_complete`, `anon_vs_*`) |
| Preliminary environment (supplementary tables) | `python3 -m cuas_rlvr.report --data data/gym --results results` |

**Scope of reproduction.** Re-scoring every stored answer and regenerating every table and figure of results needs only
this repository (numpy, matplotlib). Re-running inference or training needs the listed models and
hardware (the LoRA adapters are not released; the scripts rebuild them). Regenerating the instances, their terrain attributes and detection matrices, and the model checks uses the physics
module in `../cuas-gym` (`python3 -m cuas_gym.regen_check` recomputes all 520 version-2 matrices identically).

**Comparators.** A learned surrogate (`cuas_rlvr.surrogate`) predicts the detection matrix from the attributes of the observation with gradient-boosted trees fitted on the training instances and runs greedy search on the prediction. Two simulator-free geometric heuristics are reported: `geometric`, which stops when its binary
coverage discs add no uncovered weight, and `geometric_complete`, which keeps those choices and spends the remaining
budget by total disc weight per cost. `gpu/extra_evals.sh` runs the 27B zero-shot evaluation on the NF4 training
backend, the 27B SFT test evaluations, and the batch-1 latency measurements.

## Reproducing the tables from the released outputs

```
python3 -m cuas_rlvr.report    --data data/gym  --results results --out out          # version 1 (preliminary environment, Section VI-D, Table S8)
python3 -m cuas_rlvr.report_v2 --results results/v2 --data data/gym2 --ckpt ckpt --out out                        # Qwen2.5-7B
python3 -m cuas_rlvr.report_v2 --results results/v2 --data data/gym2 --ckpt ckpt --out out \
    --tag q3_8b --anon-tag q3_8banon --base-name Qwen3-8B --suffix _q3_8b                                             # Qwen3-8B
python3 -m cuas_rlvr.report_v2 --results results/v2 --data data/gym2 --ckpt ckpt --out out \
    --tag q38_27b --anon-tag q38_27banon --base-name Qwen3.8-27B --suffix _q38_27b                                    # Qwen3.8-27B
python3 -m cuas_rlvr.probe_types results/v2/mlx_v2_q7b_grpo_s0_test_sensor.jsonl --data data/gym2               # unseen-sensor use
```

```
# reference certificates of the five decisive instances (branch and bound without acoustic arrays, module of cuas-gym)
python3 -m cuas_gym.ceiling --data data/gym2 --split test_sensor --type ac --certify --below 0.9 \
    --from-json results/v2/ceiling_test_sensor_ac.json --out results/v2/ceiling_certificates_ac.json
# Table S11 (q38.tex); 27B on the NF4 backend, SFT vs GRPO, array use (valid answers only), single-request latency
python3 -m cuas_rlvr.report_extra --results results/v2 --latency results/latency --data data/gym2 --out out/tables/v2
# Table 4 (cost.tex)
python3 -m cuas_rlvr.cost_table --results results/v2 --latency results/latency --out out/tables/v2
```

Each command prints the paired differences and bootstrap intervals quoted in the text and writes the LaTeX tables
(`out/tables/v2/*.tex` and, for version 1, `out/*.tex`, identical to `paper_artifacts/tables/`) and the training-curve figures.

## Re-running the training

**Apple silicon (Qwen2.5-7B-Instruct, 4-bit, MLX; about 6 h per GRPO seed on a 48 GB M5 Pro):**

```
python3 -m venv .venv && .venv/bin/pip install -r requirements-mlx.txt && .venv/bin/pip install -e ../cuas-gym
./scripts/run_mac_v2.sh all           # zero-shot, SFT, checkpoint selection, GRPO seeds 0-2, evaluations
./scripts/run_mac_v2.sh anon          # the same with the anonymized inventory, GRPO seed 0
./scripts/extend.sh                   # anonymized GRPO seeds 1 and 2, range-swap probe on named seeds 1 and 2
./scripts/run_mac_v2.sh probes        # exploration and range-swap probes
```

Training runs in float32 (`CUAS_DTYPE=float32`, set by the training scripts): with 16-bit activations the
gradients through the 4-bit layers are wrong by orders of magnitude and LoRA training diverges (`cuas_rlvr/diag_grad.py`).

**CUDA GPU (Qwen3-8B bf16 LoRA, Qwen3.8-27B QLoRA; one RTX 5090 32 GB):**

```
pip install -r gpu/requirements.txt && pip install -e ../cuas-gym
./gpu/run_gpu.sh                                                     # Qwen3-8B: about 1.4 h per GRPO seed
MODEL=Qwen/Qwen3.8-27B TAG=q38_27b LOAD4BIT=1 SKIP_BASE=1 ./gpu/run_gpu.sh && ./gpu/scale27.sh   # about 5 h per seed
```

`run_gpu.sh` runs, for the named and then the anonymized inventory: zero-shot evaluation, SFT (280 examples, 20 held out),
checkpoint selection by generated-answer quality, GRPO with the outcome reward (300 steps, 8 samples per instance,
three seeds) and evaluation on the four test splits. It writes `results/gpu/` and `ckpt/<run>/`; the released copies of
those outputs are in `results/v2/` under the names `mlx_<tag>_<stage>_<split>.jsonl` used by `report_v2`.

**Zero-shot through an API (the untrained 27B model, NVFP4 serving, Tables S11 and S13):**

```
export CUAS_API_KEY=...        # never store the key in a file
./scripts/run_api_base.sh          # OpenAI-compatible endpoint; thinking mode off and on
```

## Contents

| Path | Content |
|---|---|
| `cuas_rlvr/` | MLX training (`sft_mlx`, `grpo_mlx`, `mlx_common`, `select_sft`, `mlx_sft_data`), evaluation (`eval_mlx`), probes (`probe_swap`, `probe_types`), gradient diagnosis (`diag_grad`), tables and figures (`report`, `report_v2`, `report_extra`, `cost_table`, `radar_sensitivity`, `surrogate_report`, `truncation_report`, `figure_curves`, `figure_instance`, `figure_map_rationale`), supervised checkpoint selection (`pick_sft`), learned surrogate baseline (`surrogate`) |
| `gpu/` | PyTorch/TRL pipeline: `sft_trl`, `grpo_trl` (verifier as GRPO reward), `eval_hf`, `merge_lora`, `run_gpu.sh`, `scale27.sh` |
| `scripts/` | MLX pipelines (`run_mac.sh` version 1, `run_mac_v2.sh` version 2), probes, API evaluation |
| `data/` | SFT data built from the reference solutions (`sft_train.jsonl` v1, `sft_train_v2.jsonl`, `sft_train_v2anon.jsonl`) |
| `results/` | version-1 outputs and baselines (`mlx_q7b_*`, `baselines_*.csv`, `warm_start_q7b.json`) |
| `results/v2/` | version-2 outputs of every model, stage, seed and split; baselines; unseen-sensor ceilings; probe outputs; MLX logs |
| `results/api/` | Qwen3.8-27B zero-shot outputs, thinking off and on |
| `results/gpu_logs/`, `results/gpu_selection/` | logs of the GPU runs and the checkpoint-selection answers |
| `results/validation/` | validation of the physics module against the MathWorks Boulder scenario (`python3 -m cuas_gym.physics.tests.boulder`) |
| `ckpt/<run>/log.jsonl` | per-step GRPO and SFT logs (reward, validity, quality ratio); adapters are not included |
| `paper_artifacts/` | the tables and figures of the article and supplementary material (`tables/v2/`); version-1 tables in `tables/` |

Every output record holds the instance id, the model's full answer text, the parsed placement, validity, quality
ratio, process reward and answer length, so any number in the article can be recomputed or checked.

## Models

Qwen2.5-7B-Instruct (MLX 4-bit conversion), Qwen3-8B and Qwen3.8-27B, all released by Alibaba Cloud under the Apache
License 2.0. The trained LoRA adapters are not included because of their size; they can be re-created with the scripts
above, or obtained from the authors on request.

## Licences

Code: MIT (`../LICENSE`). Data and results (`data/`, `results/`, `ckpt/`, `paper_artifacts/`): CC BY 4.0. The instances
they refer to derive from the Copernicus DEM GLO-30 (© DLR e.V. 2010–2014 and © Airbus Defence and Space GmbH
2014–2018, provided under COPERNICUS by the European Union and ESA; all rights reserved).
