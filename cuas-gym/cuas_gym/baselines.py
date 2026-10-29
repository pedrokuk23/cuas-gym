"""Classical baselines and the reference solver.

- random_feasible: random affordable placement (sanity floor)
- greedy: lazy greedy by gain per cost and by gain, plus the best single affordable sensor; the best of the three
  keeps the constant-factor guarantee of budgeted submodular maximization
- local_search: swap (remove one, add affordable ones greedily) until no improvement
- genetic: steady-state genetic algorithm on feasible sets
- branch_and_bound: exact search with a fractional-knapsack bound on the marginal gains (submodularity); proves
  optimality when it finishes within the time limit
- reference: best of the above, with a flag telling whether it is proven optimal
"""
import heapq
import time

import numpy as np


def _fill(v, rows, by_ratio=True, q=None):
    """Greedy completion of `rows` within the budget."""
    rows = list(rows)
    q = v.miss(rows) if q is None else q.copy()
    left = v.budget - float(v.cost[rows].sum()) if rows else v.budget
    used = set(rows)
    while True:
        g = v.all_gains(q)
        best, bi = 0.0, -1
        for i in range(len(g)):
            if i in used or v.cost[i] > left + 1e-9 or g[i] <= 1e-12:
                continue
            s = g[i] / v.cost[i] if by_ratio else g[i]
            if s > best:
                best, bi = s, i
        if bi < 0:
            return rows
        rows.append(bi); used.add(bi); left -= v.cost[bi]; q *= 1.0 - v.P[bi]


def lazy_greedy(v, by_ratio=True):
    """Lazy evaluation (Minoux): marginal gains only decrease, so stale upper bounds in a heap are re-evaluated lazily."""
    q = np.ones(v.P.shape[1])
    g0 = v.all_gains(q)
    key = (lambda i, g: g / v.cost[i]) if by_ratio else (lambda i, g: g)
    heap = [(-key(i, g0[i]), i, 0) for i in range(len(g0))]
    heapq.heapify(heap)
    rows, left, it = [], v.budget, 0
    while heap:
        negk, i, stamp = heapq.heappop(heap)
        if v.cost[i] > left + 1e-9:
            continue
        if stamp == it:
            if -negk <= 1e-12:
                break
            rows.append(i); left -= v.cost[i]; q *= 1.0 - v.P[i]; it += 1
        else:
            g = float(v.w @ (q * v.P[i]))
            v.evals += 1   # one marginal-gain lookup in the detection matrix
            heapq.heappush(heap, (-key(i, g), i, it))
    return rows


def greedy(v):
    cands = [lazy_greedy(v, True), lazy_greedy(v, False)]
    single = [i for i in range(len(v.cost)) if v.cost[i] <= v.budget]
    if single:
        b = max(single, key=lambda i: v.F([i]))
        cands.append(_fill(v, [b]))
    return max(cands, key=v.F)


def random_feasible(v, rng):
    order = rng.permutation(len(v.cost))
    rows, left = [], v.budget
    for i in order:
        if v.cost[i] <= left + 1e-9:
            rows.append(int(i)); left -= v.cost[i]
    return rows


def local_search(v, rows, max_rounds=50):
    rows = list(rows)
    best = v.F(rows)
    for _ in range(max_rounds):
        improved = False
        for j in range(len(rows)):
            base = rows[:j] + rows[j + 1:]
            q = v.miss(base)
            left = v.budget - float(v.cost[base].sum())
            g = v.all_gains(q)
            for i in np.argsort(-g):
                if i in rows or v.cost[i] > left + 1e-9:
                    continue
                cand = _fill(v, base + [int(i)], True)
                f = v.F(cand)
                if f > best + 1e-9:
                    rows, best, improved = cand, f, True
                    break
            if improved:
                break
        if not improved:
            return rows
    return rows


def genetic(v, rng, pop=40, gens=80, pmut=0.3):
    n = len(v.cost)

    def repair(s):
        s = list(dict.fromkeys(int(i) for i in s))
        rng.shuffle(s)
        out, left = [], v.budget
        for i in s:
            if v.cost[i] <= left + 1e-9:
                out.append(i); left -= v.cost[i]
        return _fill(v, out, True) if rng.random() < 0.5 else out

    P = [repair(random_feasible(v, rng)) for _ in range(pop)]
    fit = [v.F(s) for s in P]
    for _ in range(gens * pop // 2):
        a, b = (P[max(rng.choice(pop, 3, replace=False), key=lambda k: fit[k])] for _ in range(2))
        child = [i for i in set(a) | set(b) if rng.random() < 0.5]
        if rng.random() < pmut:
            child.append(int(rng.integers(n)))
        if child and rng.random() < pmut:
            child.pop(int(rng.integers(len(child))))
        child = repair(child)
        f = v.F(child)
        worst = int(np.argmin(fit))
        if f > fit[worst]:
            P[worst], fit[worst] = child, f
    return P[int(np.argmax(fit))]


def branch_and_bound(v, incumbent, time_limit=20.0):
    """Exact maximization of F within the budget. Bound: F(S) + fractional knapsack of the current marginal gains,
    valid because F is submodular (gains of a set never exceed the sum of individual marginal gains)."""
    t0 = time.time()
    n = len(v.cost)
    best_rows, best = list(incumbent), v.F(incumbent)
    order = np.argsort(-v.all_gains(np.ones(v.P.shape[1])) / v.cost)
    done = [True]

    def bound(q, fS, start, left):
        g = v.all_gains(q)
        idx = [i for i in order[start:] if v.cost[i] <= left + 1e-9 and g[i] > 0]
        idx.sort(key=lambda i: -g[i] / v.cost[i])
        ub, cap = fS, left
        for i in idx:
            if v.cost[i] <= cap:
                ub += g[i]; cap -= v.cost[i]
            else:
                ub += g[i] * cap / v.cost[i]
                break
        return ub

    def dfs(start, rows, q, fS, left):
        nonlocal best, best_rows
        if time.time() - t0 > time_limit:
            done[0] = False
            return
        if fS > best + 1e-9:
            best, best_rows = fS, list(rows)
        if bound(q, fS, start, left) <= best + 1e-9:
            return
        for k in range(start, n):
            i = order[k]
            if v.cost[i] > left + 1e-9:
                continue
            q2 = q * (1.0 - v.P[i])
            dfs(k + 1, rows + [int(i)], q2, float(v.w @ (1.0 - q2)), left - v.cost[i])
            if not done[0]:
                return

    dfs(0, [], np.ones(v.P.shape[1]), 0.0, v.budget)
    return best_rows, done[0]


def reference(v, time_limit=20.0, seed=0):
    rng = np.random.default_rng(seed)
    t = time.time()
    G = greedy(v)
    L = local_search(v, G)
    A = local_search(v, genetic(v, rng))
    inc = max([G, L, A], key=v.F)
    B, proven = branch_and_bound(v, inc, time_limit)
    best = max([inc, B], key=v.F)
    return {"rows": best, "F": v.F(best), "proven": bool(proven), "time": time.time() - t,
            "F_greedy": v.F(G), "F_local": v.F(L), "F_ga": v.F(A)}


def _geometric_cover(v):
    """Binary coverage discs of the simulator-free heuristic: one boolean row per configuration over the ROI cells.
    Uses only what the language model sees (ROI geometry, sites, horizon angles, inventory, data-sheet ranges, drone
    type), never the detection matrix. Each sensor covers a disc whose radius is its nominal (version 1) or data-sheet
    (version 2) range, shrunk for line-of-sight sensors by the site's mean horizon angle; types that cannot detect the
    drone cover nothing."""
    inst = v.inst
    from .catalog import DRONES
    d = next(x for x in DRONES if x["key"] == inst.drone)
    useless = {"rid"} if d["ridPt"] is None else set()
    if d["c2Pt"] is None:
        useless.add("rf")
    xy, w = np.asarray(inst.cell_xy, float), np.asarray(inst.w, float)
    site = {s["id"]: s for s in inst.sites}
    cover = []
    for (sid, t) in inst.configs:
        s = site[sid]
        if t in useless:
            cover.append(np.zeros(len(w), bool)); continue
        r = inst.rmax[t] if getattr(inst, "version", 1) != 2 else inst.drange[t] * 1000.0
        if r <= 0:
            cover.append(np.zeros(len(w), bool)); continue
        if inst.needlos[t]:
            hz = float(np.mean(np.maximum(s["horizon"], 0)))
            r *= 1.0 / (1.0 + hz)          # 0 deg -> full range, 1 deg -> half, 3 deg -> quarter
        dx, dy = xy[:, 0] - s["x"] * 1000, xy[:, 1] - s["y"] * 1000
        cover.append(dx * dx + dy * dy <= r * r)
    return np.array(cover), w


def geometric(v, n_pts=None):
    """Simulator-free expert heuristic (original design): greedy on new covered weight per cost of the binary
    discs of _geometric_cover. It stops when no affordable disc adds uncovered weight, which can leave budget unspent."""
    cover, w = _geometric_cover(v)
    covered = np.zeros(len(w), bool)
    rows, left = [], v.budget
    while True:
        gain = (cover & ~covered) @ w
        ok = [(i not in rows) and v.cost[i] <= left + 1e-9 and gain[i] > 0 for i in range(len(gain))]
        if not any(ok):
            break
        i = max((i for i in range(len(gain)) if ok[i]), key=lambda i: gain[i] / v.cost[i])
        rows.append(i); left -= v.cost[i]; covered |= cover[i]
    return rows if rows else random_feasible(v, np.random.default_rng(0))


def geometric_complete(v):
    """Geometric heuristic with budget completion: keeps the choices of `geometric` and then spends the remaining
    budget on the affordable configurations with the largest total disc weight per cost (redundant coverage still
    raises the fused probability). Fixed rule, no tuning; still never reads the detection matrix. It is the stronger
    simulator-free comparator."""
    rows = list(geometric(v))
    cover, w = _geometric_cover(v)
    total = cover @ w
    left = v.budget - float(v.cost[rows].sum())
    while True:
        ok = [i for i in range(len(total)) if i not in rows and v.cost[i] <= left + 1e-9 and total[i] > 0]
        if not ok:
            return rows
        i = max(ok, key=lambda i: total[i] / v.cost[i])
        rows.append(i); left -= v.cost[i]


def greedy_order(v, rows):
    """Reorder a set so that each next sensor has the largest marginal gain per cost among the remaining ones."""
    rows, out, q = list(rows), [], np.ones(v.P.shape[1])
    while rows:
        g = [(float(v.w @ (q * v.P[r])) / v.cost[r], r) for r in rows]
        _, r = max(g)
        out.append(r); rows.remove(r); q *= 1.0 - v.P[r]
    return out
