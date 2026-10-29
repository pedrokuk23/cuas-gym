"""Rewards of the article: outcome (quality ratio), process (marginal gain against the greedy step) and hybrid."""
import numpy as np


def outcome_reward(v, rows):
    """F(S) / F(S*) for a valid placement."""
    return v.F(rows) / max(v.reference()["F"], 1e-12)


def process_reward(v, rows):
    """Mean over the steps of min(1, (gain_i / cost_i) / best affordable gain-per-cost at step i)."""
    q = np.ones(v.P.shape[1])
    left, used, score = v.budget, set(), []
    for r in rows:
        g_all = v.all_gains(q)
        ok = np.array([(i not in used) and v.cost[i] <= left + 1e-9 for i in range(len(v.cost))])
        ratio = np.where(ok, g_all / v.cost, -np.inf)
        best = float(ratio.max())
        mine = float(g_all[r] / v.cost[r])
        score.append(1.0 if best <= 1e-12 else min(1.0, max(0.0, mine / best)))
        q *= 1.0 - v.P[r]
        used.add(r)
        left -= v.cost[r]
    return float(np.mean(score)) if score else 0.0


def hybrid_reward(v, action, lam=0.5, format_ok=True):
    """r = 1[valid] (r_out + lam * r_proc). Invalid or unparsable answers get 0."""
    if not format_ok or action is None:
        return {"r": 0.0, "r_out": 0.0, "r_proc": 0.0, "valid": False, "error": "unparsable answer"}
    rows, err = v.rows_of(action)
    if err:
        return {"r": 0.0, "r_out": 0.0, "r_proc": 0.0, "valid": False, "error": err}
    ro, rp = outcome_reward(v, rows), process_reward(v, rows)
    return {"r": ro + lam * rp, "r_out": ro, "r_proc": rp, "valid": True, "error": None}
