"""Checks of CUAS-Gym (run from the repository root: python3 -m cuas_gym.tests.test_gym).

With the terrain tile in data/dem/, instances are also generated afresh and the verifier is checked against the
physics module's own analysis; without the tile, the checks run on released instances from data/."""
import itertools
import sys

import numpy as np

from cuas_gym import Verifier, build_prompt, generate, hybrid_reward, parse_answer
from cuas_gym import baselines as B
import os
from cuas_gym.instances import DEM_DIR, scenario_dict, tile
from cuas_gym.cli import ids, load
from cuas_gym.physics import Study
from cuas_gym.physics.scenario import Scenario
HAVE_TILE = os.path.isdir(DEM_DIR) and any(f.endswith(".tif") for f in os.listdir(DEM_DIR))


def test_matches_physics(inst, v, rng):
    """F(S) of the verifier equals the weighted fused probability computed by the physics module's own analysis."""
    rows = B.random_feasible(v, rng)
    chosen = {inst.configs[r] for r in rows}
    d = scenario_dict(inst)
    for s in d["sensors"]:
        sid, t = s["name"].split("-")
        s["deployed"] = (int(sid[1:]), t) in chosen
    st = Study(Scenario(d), [tile()], download=False)
    r = st.analyze(alt=inst.alt)
    por = r["agg"]["por"][r["grid"].idx]
    F_phys = float(inst.w @ por)
    assert abs(F_phys - v.F(rows)) <= 1e-3 * max(1.0, F_phys), (F_phys, v.F(rows))


def test_submodular(v, rng, trials=200):
    n = len(v.cost)
    for _ in range(trials):
        A = list(rng.choice(n, rng.integers(0, 4), replace=False))
        extra = [i for i in rng.choice(n, 3, replace=False) if i not in A]
        Bset = A + extra[:-1]
        e = extra[-1]
        gA = v.F(A + [e]) - v.F(A)
        gB = v.F(Bset + [e]) - v.F(Bset)
        assert gA >= gB - 1e-6, (gA, gB)


def test_bnb_exact(v):
    """On a reduced instance, branch and bound equals brute force."""
    keep = list(range(min(12, len(v.cost))))
    v2 = Verifier.__new__(Verifier)
    v2.__dict__.update(v.__dict__)
    v2.P, v2.cost, v2.evals = v.P[keep], v.cost[keep], 0
    best = 0.0
    for k in range(1, 6):
        for S in itertools.combinations(range(len(keep)), k):
            if v2.cost[list(S)].sum() <= v2.budget:
                best = max(best, v2.F(list(S)))
    rows, proven = B.branch_and_bound(v2, B.greedy(v2), 60)
    assert proven and abs(v2.F(rows) - best) < 1e-6, (v2.F(rows), best)


def test_rewards_and_parser(inst, v):
    ref = v.action_of(v.reference()["rows"])
    txt = "reasoning...\n<answer>" + '{"placements": [' + ", ".join(f'{{"site": {s}, "type": "{t}"}}' for s, t in ref) + "]}</answer>"
    a = parse_answer(txt)
    assert a == [tuple(x) for x in ref]
    r = hybrid_reward(v, a)
    assert r["valid"] and abs(r["r_out"] - 1) < 1e-4 and 0 <= r["r_proc"] <= 1
    assert hybrid_reward(v, [(999, "radar")])["r"] == 0.0
    assert hybrid_reward(v, parse_answer("no answer"))["r"] == 0.0
    over = [(s["id"], t) for s in inst.sites for t in inst.types]
    assert not hybrid_reward(v, over)["valid"]           # exceeds the budget
    assert "Candidate sites" in build_prompt(inst) and "horizon" not in build_prompt(inst, attrs=False)


def main():
    rng = np.random.default_rng(1)
    n = 0
    cases = [("train", 0, 1), ("train", 1, 1), ("test_terrain", 0, 1), ("test_sensor", 0, 1),
             ("train", 0, 2), ("test_sensor", 3, 2)]
    for split, seed, ver in cases:
        if HAVE_TILE:
            inst = generate(seed, split, version=ver)
            v = Verifier(inst)
        else:
            data = "data/gym" if ver == 1 else "data/gym2"
            inst, v = load(data, ids(data, split)[seed])
        checks = [lambda: test_submodular(v, rng), lambda: test_bnb_exact(v), lambda: test_rewards_and_parser(inst, v)]
        if HAVE_TILE:
            checks.insert(0, lambda: test_matches_physics(inst, v, rng))
        for t in checks:
            t(); n += 1
        print(f"ok {split}-{seed} v{ver}" + ("" if HAVE_TILE else " (released instance)"))
    print(f"{n} checks passed")


if __name__ == "__main__":
    sys.exit(main())
