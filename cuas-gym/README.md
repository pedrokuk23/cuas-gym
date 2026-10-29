# CUAS-Gym

A training and evaluation environment for language models on **multi-sensor placement over real terrain** with a
**physics-verified reward**. It accompanies the article *Learning to Plan Heterogeneous Counter-UAS Sensor Networks Over
Terrain With Physics-Verified Rewards* (Kukulka de Albuquerque, Kuroswiski, Galembeck and Morais, submitted 2026); the
code and data that reproduce the article's experiments are in the `cuas-rlvr/` folder of the same repository.

Each instance is a budgeted placement problem: a region of interest on the Copernicus GLO-30 elevation model, 10–40
candidate sites, an inventory of counter-drone sensor types (radar, EO/IR camera, Remote ID receiver, passive RF
receiver and, in one test split, an acoustic array) with costs, a budget, a target altitude and a threat drone. The
detection probability of every configuration (site × sensor type) in every cell was computed once by the physics
module `cuas_gym.physics` (terrain, line of sight, sensor models; see below) and is stored with the instance, so the verifier scores **any placement exactly and in
microseconds** with the fusion rule P(≥1) = 1 − Π(1 − Pd). The language model never sees that matrix: it sees a text
description of the drone, the region, the inventory (with data-sheet ranges in version 2) and, per site, its
coordinates, elevation, height above the region's median and horizon angle in eight directions.

**Needs:** Python ≥ 3.9 and numpy. Nothing else is needed to load the released instances, score answers, compute the
rewards, run the baselines and build the supervised data. Generating instances and recomputing their detection
matrices also use only numpy, with the Copernicus tile S24 W046 shipped in `data/dem/`.

## How the detection matrix is computed

The region of interest is covered by square cells of 300 m; every cell that touches the region's polygon is kept.
The ground height of a cell is the mean of 2 × 2 samples of the Copernicus GLO-30 model inside it (bilinear
interpolation), and the target flies at that height plus the instance's altitude above ground (60, 90 or 120 m).
Each sensor stands on a 10 m mast on the ground of its site. Line of sight from each site to each cell is computed on
the full-resolution model (about 30 m posting) with an effective earth radius of 4/3 × 6371 km: for each cell the
engine stores the lowest visible target height, and sensors that need line of sight see the cell only if the target
is above it. Each sensor type then maps range, geometry, environment and target to a detection probability (radar:
radar equation and Shnidman's Swerling approximation; EO/IR: point-target contrast model; passive RF and Remote ID:
link budget with free-space and knife-edge diffraction loss; acoustic: spreading, air absorption, barrier and ground
attenuation). The equations and parameters are in the supplementary material of the article.

## Install

```
git clone <this repository> cuas          # one repository: cuas-gym/ and cuas-rlvr/
pip install -e cuas/cuas-gym
cd cuas/cuas-gym && python3 -m cuas_gym.tests.test_gym        # 24 checks, incl. instances generated afresh
```

## Quick start

```python
from cuas_gym.cli import ids, load
from cuas_gym import build_prompt, parse_answer, hybrid_reward

inst, v = load("data/gym2", ids("data/gym2", "test_id")[0])
prompt = build_prompt(inst)                       # what the model sees
answer = '<answer>{"placements": [{"site": 3, "type": "rf"}, {"site": 7, "type": "eo"}]}</answer>'
r = hybrid_reward(v, parse_answer(answer))        # {'r_out': quality ratio F(S)/F(S*), 'r_proc': ..., 'valid': ...}
```

`r_out` is the quality ratio against the stored reference solution (proven optimal by branch and bound on 88–100% of
the standard instances); invalid answers (unknown site or type, repeated placement, budget exceeded) score zero.

## Data

| Folder | Version | Splits (instances) |
|---|---|---|
| `data/gym` | 1: default sensor parameters | train 300, test_id 60, test_terrain 60, test_sensor 60, test_scale 40 |
| `data/gym2` | 2: one variant per sensor type, cost growing with performance, data-sheet range in the prompt | same |

Each instance is three files: `<iid>.json` (geometry, sites with terrain attributes, inventory, budget, altitude,
weights, drone), `<iid>.npz` (detection matrix `P[config, cell]`, cell weights, cell centres) and `<iid>.ref.json`
(reference solution and whether it is proven optimal).

| Split | Terrain | Inventory | Sites | Budget |
|---|---|---|---|---|
| train | A: Paraíba valley (rolling) | radar, EO/IR, Remote ID, RF | 10–18 | 8–22 |
| test_id | A (new instances) | same | 10–18 | 8–22 |
| test_terrain | B: Serra do Mar and coast (steep) | same | 10–18 | 8–22 |
| test_sensor | A | + acoustic array (never in training) | 10–18 | 8–22 |
| test_scale | A | same | 28–40 | 30–50 |

## Commands

```
python3 -m cuas_gym.cli baselines --data data/gym2 --split test_id --out baselines_test_id.csv
python3 -m cuas_gym.sft_data --data data/gym2 --out sft_train_v2.jsonl          # supervised data from the references
CUAS_ANON=1 python3 -m cuas_gym.sft_data --data data/gym2 --out sft_train_v2anon.jsonl
python3 -m cuas_gym.ceiling --data data/gym2 --split test_sensor --type ac --out ceiling.json
python3 -m cuas_gym.eval_api --base-url <OpenAI-compatible URL> --model <name> --data data/gym2 --split test_id --out out.jsonl
python3 -m cuas_gym.cli generate --split train --n 300 --version 2 --out data/gym2
python3 -m cuas_gym.regen_check --data data/gym2            # recompute every stored matrix and compare (520/520 identical)
python3 -m cuas_gym.physics.tests.test_models               # unit checks of the sensor models
python3 -m cuas_gym.physics.tests.validation_table          # Table S2: checks against independent references
python3 -m cuas_gym.physics.tests.boulder --tile <Copernicus N39 W106 tile>   # MathWorks Boulder radar example
```

`CUAS_ANON=1` switches the observation to the **anonymized inventory**: each sensor type becomes a letter, permuted
per instance, described only by its cost, open-terrain range against the drone and line-of-sight need; the verifier
maps the letters back.

| File | Content |
|---|---|
| `instances.py` | instance data structure, splits, generation, detection matrix, save/load |
| `physics/` | terrain (GeoTIFF reader, bilinear sampling), grid, line of sight on the 4/3 earth, sensor detection models (radar, EO/IR, passive RF, Remote ID, acoustic), drone library |
| `regen_check.py` | recomputes released instances and compares them with the stored matrices |
| `verifier.py` | F(S), marginal gains, answer validation |
| `rewards.py` | outcome (quality ratio), process (gain per cost against the best step), hybrid |
| `baselines.py` | random, simulator-free geometric heuristic (`geometric`, and `geometric_complete`, which spends the remaining budget), greedy, local search, genetic algorithm, branch and bound |
| `prompt.py` | observation text (named or anonymized inventory) and answer parser |
| `catalog.py` | the three threat drones of the instances |
| `ceiling.py` | best placement without a given sensor type: how much a split depends on it |
| `sft_data.py` | supervised examples (short rationale + reference answer) |
| `eval_api.py`, `eval_llm.py` | evaluation of a model behind an OpenAI-compatible or Anthropic API |
| `variants.py`, `datasheet_cache.json` | version-2 sensor variants and their data-sheet ranges |

## Licences and attribution

Code: MIT (see `LICENSE` at the repository root). Instance data (`data/`): CC BY 4.0 (see `data/LICENSE`). The terrain attributes and the
detection matrices are derived from the Copernicus DEM GLO-30: © DLR e.V. 2010–2014 and © Airbus Defence and Space GmbH
2014–2018, provided under COPERNICUS by the European Union and ESA; all rights reserved. The tile in `data/dem/` is redistributed
under the Copernicus DEM licence with that attribution (see `data/dem/ATTRIBUTION.txt`).

## Citation

See `CITATION.cff` at the repository root.
