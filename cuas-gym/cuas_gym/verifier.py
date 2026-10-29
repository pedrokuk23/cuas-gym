"""Exact scoring of a placement with the fusion rule of the physics module.

F(S) = sum_v w_v [1 - prod_{c in S} (1 - P[c, v])]          (weighted fused probability of detection)

Under this rule F is monotone and submodular, so the marginal gain of each sensor is well defined and cheap to compute.
"""
import numpy as np


class Verifier:
    def __init__(self, inst):
        self.inst = inst
        self.P = np.asarray(inst.P, np.float64)
        self.w = np.asarray(inst.w, np.float64)
        self.W = float(self.w.sum())
        self.row = {(int(s), str(t)): i for i, (s, t) in enumerate(inst.configs)}
        from .prompt import anon_on, anon_map
        self.alias = {lab: t for t, lab in anon_map(inst).items()} if anon_on() else {}   # letter -> type
        self.cost = np.array([inst.cost[t] for _, t in inst.configs], float)
        self.budget = float(inst.budget)
        self._ref = None
        self.evals = 0   # simulator-based evaluations of a placement or of a marginal gain (lookups in the precomputed detection matrix)

    # --- core ---------------------------------------------------------------------------------------------------------
    def miss(self, rows):
        """prod (1 - P) over the rows: probability that no chosen sensor detects, per cell."""
        q = np.ones(self.P.shape[1])
        for r in rows:
            q *= 1.0 - self.P[r]
        return q

    def F(self, rows):
        self.evals += 1
        return float(self.w @ (1.0 - self.miss(rows)))

    def gains(self, rows):
        """Marginal gain of each row in the given order."""
        q, out = np.ones(self.P.shape[1]), []
        for r in rows:
            g = float(self.w @ (q * self.P[r]))
            out.append(g)
            q *= 1.0 - self.P[r]
        return out

    def all_gains(self, q):
        """Marginal gain of every configuration given the current miss vector q."""
        self.evals += self.P.shape[0]
        return self.P @ (q * self.w)

    # --- actions ------------------------------------------------------------------------------------------------------
    def rows_of(self, action):
        """action: list of (site, type). Returns (rows, error message or None)."""
        rows, seen = [], set()
        for a in action:
            try:
                typ = str(a[1]).strip().lower()
                key = (int(a[0]), self.alias.get(typ, typ))
            except (TypeError, ValueError, IndexError):
                return None, f"malformed placement {a!r}"
            if key not in self.row:
                return None, f"unknown site/type {key}"
            if key in seen:
                return None, f"duplicate placement {key}"
            seen.add(key)
            rows.append(self.row[key])
        if not rows:
            return None, "empty placement"
        c = float(self.cost[rows].sum())
        if c > self.budget + 1e-9:
            return None, f"cost {c:g} exceeds budget {self.budget:g}"
        return rows, None

    def score(self, action, pd_thr=0.9):
        rows, err = self.rows_of(action)
        if err:
            return {"valid": False, "error": err, "F": 0.0, "Fn": 0.0, "cost": 0.0, "gains": [], "cov90": 0.0}
        pfus = 1.0 - self.miss(rows)
        F = float(self.w @ pfus)
        return {"valid": True, "error": None, "F": F, "Fn": F / self.W, "cost": float(self.cost[rows].sum()),
                "gains": self.gains(rows), "rows": rows, "cov90": float((pfus >= pd_thr).mean())}

    def action_of(self, rows):
        return [list(self.inst.configs[r]) for r in rows]

    # --- reference ----------------------------------------------------------------------------------------------------
    def reference(self):
        """Best known solution (exact enumeration when small, else greedy + local search); cached."""
        if self._ref is None:
            from .baselines import reference
            self._ref = reference(self)
        return self._ref
