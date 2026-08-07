"""EXP-066: hybrid prior-guided, chance-node-aware KR tree (search-validation arm).

Implements the user-specified structural priorities:
  * explicit decision/chance (afterstate) separation — min/max only at DECISION
    nodes (via to-move-perspective selection); chance nodes AVERAGE outcomes
  * double progressive widening: intended actions AND execution outcomes widen
    independently (outcomes: n_allowed = ceil(visits^out_alpha), capped)
  * minimum-noise-evidence gate: no new action opens until every "serious"
    opened action (top half by KR value) has >= min_evidence effective samples
  * policy-prior-guided opening + PUCT-style exploration bonus over KR-shared
    statistics (kernel regression across nearby actions; bandwidth in
    EXECUTION-NOISE units, i.e. the physically-meaningful metric)
  * mixed leaves: value head for cheap evaluations, every `rollout_every`-th
    leaf corrected by a terminal rollout (rules-grounded)
  * anytime: report the chosen root action at each simulator-call budget
    checkpoint (budget = authoritative-simulator calls; NN evals uncounted)

Kept wave-free (sequential visits): intended for the CPU-JAX backend where
single-throw sims are ~1-5 ms; the benchmark counts sim CALLS, not wall time.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional

import numpy as np

from .. import env_bridge

NOISE_STD = np.array([0.0123, 0.003, 2.0, 0.015], dtype=np.float64)


def _kr(acts: np.ndarray, edge_n: np.ndarray, edge_sum: np.ndarray, bw: float):
    """Kernel-regressed value + effective count over opened actions.
    Distance in execution-noise units (dn); K = exp(-0.5 (dn/bw)^2)."""
    z = acts / NOISE_STD[None]
    d2 = ((z[:, None, :] - z[None, :, :]) ** 2).sum(-1)
    K = np.exp(-0.5 * d2 / max(bw, 1e-6) ** 2)
    W = K @ edge_n
    S = K @ edge_sum
    with np.errstate(invalid="ignore", divide="ignore"):
        V = np.where(W > 0, S / np.maximum(W, 1e-9), 0.0)
    return V, W


class _Budget:
    def __init__(self):
        self.sims = 0


class _Decision:
    __slots__ = ("x", "c", "h", "pool", "prior", "n_open", "edge_n", "edge_sum",
                 "chance", "visits", "leaf_evals")

    def __init__(self, x, c, h, pool, prior):
        self.x, self.c, self.h = x, c, int(h)
        self.pool, self.prior = pool, prior          # candidate actions + opening priors
        self.n_open = 0
        self.edge_n = np.zeros(len(pool), np.float64)
        self.edge_sum = np.zeros(len(pool), np.float64)
        self.chance: Dict[int, "_Chance"] = {}
        self.visits = 0
        self.leaf_evals = 0


class _Chance:
    __slots__ = ("action", "outcomes", "out_visits")

    def __init__(self, action):
        self.action = action
        self.outcomes: List[_Decision] = []
        self.out_visits: List[int] = []


class HybridTree:
    def __init__(self, x, c, h, sie, pool, pool_prior, *,
                 sample_fn: Callable, value_fn: Callable, rollout_fn: Callable,
                 noise, rng, c_puct=1.5, bw=1.0, min_evidence=8,
                 out_alpha=1.0, out_cap=16, inner_pool=16, rollout_every=4):
        self.sie = int(sie)
        self.sample_fn = sample_fn        # (x, c, n) -> [n,4] policy proposals
        self.value_fn = value_fn          # (x, c) -> V for c's to-move block
        self.rollout_fn = rollout_fn      # (x, c, h, persp, budget) -> rules value (root persp)
        self.noise, self.rng = noise, rng
        self.c_puct, self.bw = float(c_puct), float(bw)
        self.min_evidence = int(min_evidence)
        self.out_alpha, self.out_cap = float(out_alpha), int(out_cap)
        self.inner_pool, self.rollout_every = int(inner_pool), int(rollout_every)
        self.root_persp = int(round(c[2]))
        self.budget = _Budget()
        self.root = _Decision(np.asarray(x, np.float32), np.asarray(c, np.float32),
                              int(h), np.asarray(pool, np.float32),
                              np.asarray(pool_prior, np.float64))
        self._open_action(self.root)      # always start with the top-prior action

    # ---------------- node machinery ---------------- #
    def _open_action(self, node: _Decision):
        if node.n_open < len(node.pool):
            node.n_open += 1

    def _maybe_widen(self, node: _Decision):
        if node.n_open >= len(node.pool):
            return
        k = node.n_open
        acts = node.pool[:k]
        V, W = _kr(acts.astype(np.float64), node.edge_n[:k], node.edge_sum[:k], self.bw)
        if k == 0:
            self._open_action(node)
            return
        serious = np.argsort(V)[::-1][: max(1, k // 2)]
        if np.all(W[serious] >= self.min_evidence):
            self._open_action(node)

    def _select(self, node: _Decision) -> int:
        k = node.n_open
        acts = node.pool[:k].astype(np.float64)
        V, W = _kr(acts, node.edge_n[:k], node.edge_sum[:k], self.bw)
        to_move = int(round(node.c[2]))
        v_tm = V if to_move == self.root_persp else -V     # min/max only via to-move persp
        tot = max(W.sum(), 1.0)
        score = v_tm + self.c_puct * node.prior[:k] * np.sqrt(tot) / (1.0 + W)
        return int(np.argmax(score))

    def _leaf_value(self, node: _Decision) -> float:
        """Mixed leaf: V-head normally; every rollout_every-th eval a terminal rollout."""
        node.leaf_evals += 1
        if node.h <= 0:
            return float(env_bridge.score_end(node.x, self.root_persp))
        if node.leaf_evals % self.rollout_every == 0:
            return float(self.rollout_fn(node.x, node.c, node.h, self.root_persp, self.budget))
        v = float(self.value_fn(node.x, node.c))
        return v if int(round(node.c[2])) == self.root_persp else -v

    def _visit(self, node: _Decision) -> float:
        node.visits += 1
        if node.h <= 0:
            return float(env_bridge.score_end(node.x, self.root_persp))
        self._maybe_widen(node)
        i = self._select(node)
        ch = node.chance.get(i)
        if ch is None:
            ch = node.chance[i] = _Chance(node.pool[i])
        # --- chance node: double widening over outcomes ---
        allowed = min(self.out_cap, sum(ch.out_visits) + 1) if self.out_alpha >= 1.0 else \
            min(self.out_cap, max(1, int(np.ceil((sum(ch.out_visits) + 1) ** self.out_alpha))))
        if len(ch.outcomes) < allowed:
            a = self.noise.sample_batch(ch.action[None], 1).reshape(4).astype(np.float32)
            post, _ = env_bridge.apply_legality(
                node.x, env_bridge.simulate_one(node.x, node.c, a)[None], node.h, node.c)
            self.budget.sims += 1
            nc = env_bridge.next_condition(node.c, self.sie)
            child_pool = self.sample_fn(post[0], nc, self.inner_pool)
            child_prior = np.full(len(child_pool), 1.0 / len(child_pool))
            child = _Decision(post[0], nc, node.h - 1, child_pool, child_prior)
            self._open_action(child)
            ch.outcomes.append(child)
            ch.out_visits.append(1)
            v = self._leaf_value(child)
        else:
            j = int(np.argmin(ch.out_visits))
            ch.out_visits[j] += 1
            v = self._visit(ch.outcomes[j])          # deepen an existing outcome
        node.edge_n[i] += 1.0
        node.edge_sum[i] += v
        return v

    # ---------------- driver ---------------- #
    def run(self, checkpoints: List[int]) -> Dict[int, np.ndarray]:
        """Run until each simulator-call checkpoint; return {budget: chosen root action}."""
        out: Dict[int, np.ndarray] = {}
        for cp in sorted(checkpoints):
            stall = 0
            while self.budget.sims < cp:
                before = self.budget.sims
                self._visit(self.root)
                stall = stall + 1 if self.budget.sims == before else 0
                if stall > 20000:   # saturated tree: no sim-consuming work left
                    break
            k = self.root.n_open
            V, W = _kr(self.root.pool[:k].astype(np.float64),
                       self.root.edge_n[:k], self.root.edge_sum[:k], self.bw)
            ok = W >= min(self.min_evidence, W.max())
            idx = int(np.argmax(np.where(ok, V, -1e18)))
            out[cp] = self.root.pool[idx].copy()
        return out
