# CUAS-Gym: learning to plan counter-UAS sensor networks over terrain

This repository holds everything behind the article

> P. Kukulka de Albuquerque, A. R. Kuroswiski, L. Galembeck and E. C. de F. Morais, *Learning to Plan Heterogeneous
> Counter-UAS Sensor Networks Over Terrain With Physics-Verified Rewards*, submitted to the IEEE Open Journal of the
> Computer Society, 2026.

It has two parts, in two folders of this single repository:

| Folder | Content | Needs |
|---|---|---|
| [`cuas-gym/`](cuas-gym/) | **The benchmark.** Placement instances on the Copernicus GLO-30 elevation model with their detection matrices and solver-derived references; the verifier and rewards; the text observation (named or property-based inventory) and answer parser; the comparators (random, geometric heuristic, greedy, local search, genetic algorithm, branch and bound); and the **physics module** (terrain, grid, line of sight, radar, EO/IR, passive RF, Remote ID and acoustic models) that generates the instances and recomputes every stored matrix. | Python ≥ 3.9, numpy |
| [`cuas-rlvr/`](cuas-rlvr/) | **The experiments of the article.** SFT and GRPO training with the verifier as reward (MLX on Apple silicon; PyTorch/TRL on a CUDA GPU), evaluation and probe scripts, the learned-surrogate baseline, supervised data, every model output behind the reported numbers, training logs, and the scripts that regenerate the article's tables and figures. | numpy, matplotlib (re-scoring and tables); scikit-learn (surrogate); MLX or PyTorch/TRL (training) |

`cuas-rlvr/data/gym` and `cuas-rlvr/data/gym2` are relative links to the instances in `cuas-gym/data/`, so the
instances are stored once. On a system without symbolic links (for example Windows without developer mode), pass
`--data ../cuas-gym/data/gym2` to the scripts instead.

## Quick start

```
git clone <this repository> cuas
cd cuas
pip install -e cuas-gym
(cd cuas-gym && python3 -m cuas_gym.tests.test_gym)                      # 24 checks
(cd cuas-gym && python3 -m cuas_gym.regen_check --data data/gym2)         # recomputes the 520 version-2 matrices: identical
(cd cuas-rlvr && python3 -m cuas_rlvr.report_v2 --results results/v2 --data data/gym2 --ckpt ckpt --out /tmp/out)
cmp /tmp/out/tables/v2/llm.tex cuas-rlvr/paper_artifacts/tables/v2/llm.tex  # same table as in the article
```

`cuas-gym/README.md` describes the benchmark and how the detection matrices are computed; `cuas-rlvr/README.md`
maps every table and figure of the article to the command and the files that produce it.

## Licences

Code: MIT (`LICENSE`). Data, results and benchmark artifacts: CC BY 4.0 (`LICENSE-DATA.md`). The terrain attributes
and detection matrices derive from the Copernicus DEM GLO-30 (© DLR e.V. 2010–2014 and © Airbus Defence and Space GmbH
2014–2018, provided under COPERNICUS by the European Union and ESA; all rights reserved).

## Citation

See `CITATION.cff`. The repository is archived on Zenodo (DOI given in `CITATION.cff` and in the article).
