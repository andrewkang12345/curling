"""EXP-068: WAVE-BATCHED (vectorised) prior-guided, chance-aware search tree.

Same semantics as `hybrid_tree` — decision nodes OPTIMISE, chance nodes INTEGRATE
(i.i.d. draws from the true execution-noise model, equally weighted); double
progressive widening; minimum-noise-evidence gate before opening actions;
PUCT exploration over kernel-regressed statistics (bw -> 0 = plain PUCT);
anytime budget checkpoints in authoritative-simulator calls — but it selects
``wave`` paths per iteration under virtual loss and executes ALL of that wave's
simulator / policy / rollout work in BATCHES.

That is what makes deeper search affordable: the sequential tree paid ~0.3 s per
visit in per-node GPU round-trips (measured), which put 64k sims at ~172 min per
state. Here one wave of W paths costs ~one batched sim + one batched policy call
+ one lockstep rollout group.

New vs hybrid_tree: ``max_depth`` (search-depth cap). Nodes at the cap with
throws remaining are evaluated by a rules-grounded lockstep rollout to terminal
(no value head anywhere unless ``value_batch_fn`` is supplied), so h > 2 search
stays attributable to the tree rather than to V.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from .. import env_bridge

NOISE_STD = np.array([0.0123, 0.003, 2.0, 0.015], dtype=np.float64)


def _kr(acts: np.ndarray, n: np.ndarray, s: np.ndarray, bw: float):
    """Kernel-regressed value + effective count (bw -> 0 gives plain per-action stats)."""
    if bw <= 1e-6:
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(n > 0, s / np.maximum(n, 1e-9), 0.0), n.copy()
    z = acts / NOISE_STD[None]
    d2 = ((z[:, None, :] - z[None, :, :]) ** 2).sum(-1)
    K = np.exp(-0.5 * d2 / bw ** 2)
    W = K @ n
    S = K @ s
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(W > 0, S / np.maximum(W, 1e-9), 0.0), W


class _Budget:
    def __init__(self):
        self.sims = 0


class _Dec:
    """Decision node (a team is about to throw)."""
    __slots__ = ("x", "c", "h", "depth", "pool", "prior", "n_open",
                 "edge_n", "edge_sum", "edge_sq", "edge_vl", "chance",
                 "integrate_actions")

    def __init__(self, x, c, h, depth, integrate_actions=False):
        self.x, self.c, self.h, self.depth = x, c, int(h), int(depth)
        # Normal search nodes optimise for the team to move.  An opponent-model
        # node instead integrates actions sampled from the fixed opponent policy;
        # minimising over those samples would silently turn the oracle back into
        # free minimax search.
        self.integrate_actions = bool(integrate_actions)
        self.pool = None          # [M,4] candidate intents (None until assigned)
        self.prior = None
        self.n_open = 0
        self.edge_n = None
        self.edge_sum = None
        self.edge_sq = None
        self.edge_vl = None
        self.chance: Dict[int, "_Chance"] = {}

    def set_pool(self, pool, prior, open_all=False):
        self.pool = np.asarray(pool, np.float32)
        self.prior = np.asarray(prior, np.float64)
        m = len(self.pool)
        self.edge_n = np.zeros(m, np.float64)
        self.edge_sum = np.zeros(m, np.float64)
        self.edge_sq = np.zeros(m, np.float64)
        self.edge_vl = np.zeros(m, np.float64)
        self.n_open = m if open_all else 1  # model samples are all part of the expectation


class _Chance:
    """Chance node (an intended throw whose execution is random)."""
    __slots__ = ("action", "outcomes", "out_n", "out_vl")

    def __init__(self, action):
        self.action = np.asarray(action, np.float32)
        self.outcomes: List[_Dec] = []
        self.out_n: List[float] = []
        self.out_vl: List[float] = []


class VecTree:
    def __init__(self, x, c, h, sie, pool, prior, *,
                 sample_batch_fn: Callable, rollout_batch_fn: Callable,
                 noise, rng, c_puct: float = 1.5, bw: float = 1e-9,
                 min_evidence: int = 8, out_cap: int = 8, inner_pool: int = 8,
                 max_depth: int = 4, wave: int = 32, root_out_cap: int = 0,
                 rollout_mult: int = 1, opponent_batch_fn: Optional[Callable] = None,
                 opponent_block: Optional[int] = None, opponent_samples: int = 1):
        self.sie = int(sie)
        self.sample_batch_fn = sample_batch_fn   # (states[B,24], cond[3], n) -> [B,n,4]
        self.rollout_batch_fn = rollout_batch_fn  # (states[B,24], cond[3], h, persp) -> [B]
        self.noise, self.rng = noise, rng
        self.c_puct, self.bw = float(c_puct), float(bw)
        self.min_evidence, self.out_cap = int(min_evidence), int(out_cap)
        self.inner_pool, self.max_depth, self.wave = int(inner_pool), int(max_depth), int(wave)
        self.opponent_batch_fn = opponent_batch_fn
        self.opponent_block = None if opponent_block is None else int(opponent_block)
        self.opponent_samples = max(1, int(opponent_samples))
        # EXP-069 adjudicator: the ROOT chance node integrates over many more execution
        # outcomes (the root expectation is the quantity being reported), and rollout
        # cost is multiplied when the tail policy itself searches (value-greedy steps).
        self.root_out_cap = int(root_out_cap) if root_out_cap else int(out_cap)
        # GUARD (EXP-071 bug, 2026-08-10): the ROOT chance node's outcome cap bounds how
        # many execution-noise draws the reported root expectation ever integrates —
        # independent of budget. Leaving it at the interior default (8) while the
        # adjudicator used 64 made bigger budgets deepen subtrees around a root estimate
        # that never got more accurate, and regret INCREASED with budget. Warn loudly
        # rather than fail silently.
        if self.root_out_cap < 32:
            import warnings
            warnings.warn(f"VecTree root_out_cap={self.root_out_cap} < 32: root expectation "
                          f"integrates few noise draws; regret may not decrease with budget "
                          f"(see EXP-071).", RuntimeWarning, stacklevel=2)
        self.rollout_mult = max(1, int(rollout_mult))
        self.root_persp = int(round(c[2]))
        self.budget = _Budget()
        self.root = _Dec(np.asarray(x, np.float32), np.asarray(c, np.float32), int(h), 0)
        self.root.set_pool(pool, prior)

    # ------------------------------------------------------------------ #
    def _maybe_widen(self, node: _Dec):
        if node.n_open >= len(node.pool):
            return
        k = node.n_open
        V, W = _kr(node.pool[:k].astype(np.float64), node.edge_n[:k], node.edge_sum[:k], self.bw)
        serious = np.argsort(V)[::-1][: max(1, k // 2)]
        if np.all(W[serious] >= self.min_evidence):
            node.n_open += 1

    def _select(self, node: _Dec) -> int:
        k = node.n_open
        if node.integrate_actions:
            # Allocate visits in proportion to the sampled-policy prior.  The
            # parent backup then estimates E_{a~pi_opp}[return], rather than the
            # worst sampled reply.  Virtual visits keep a wave balanced too.
            visits = node.edge_n[:k] + node.edge_vl[:k]
            prior = node.prior[:k] / max(float(node.prior[:k].sum()), 1e-12)
            target = prior * (float(visits.sum()) + 1.0)
            return int(np.argmax(target - visits))
        V, W = _kr(node.pool[:k].astype(np.float64), node.edge_n[:k], node.edge_sum[:k], self.bw)
        W_eff = W + node.edge_vl[:k]                  # virtual loss suppresses re-selection
        to_move = int(round(node.c[2]))
        v_tm = V if to_move == self.root_persp else -V     # optimise in the mover's frame
        tot = max(W_eff.sum(), 1.0)
        return int(np.argmax(v_tm + self.c_puct * node.prior[:k] * np.sqrt(tot) / (1.0 + W_eff)))

    def _descend(self):
        """One path under virtual loss. Returns (path, kind, payload)."""
        node = self.root
        path: List[Tuple[_Dec, int, Optional[_Chance], Optional[int]]] = []
        while True:
            if node.h <= 0:
                return path, "term", node
            if node.depth >= self.max_depth:
                return path, "leaf", node
            if node.pool is None:                      # interior node awaiting its pool
                return path, "leaf", node
            self._maybe_widen(node)
            i = self._select(node)
            ch = node.chance.get(i)
            if ch is None:
                ch = node.chance[i] = _Chance(node.pool[i])
            node.edge_vl[i] += 1.0
            cap = self.root_out_cap if node is self.root else self.out_cap
            allowed = min(cap, int(sum(ch.out_n) + sum(ch.out_vl)) + 1)
            if len(ch.outcomes) < allowed:
                path.append((node, i, ch, None))
                return path, "expand", (node, ch)
            j = int(np.argmin(np.asarray(ch.out_n) + np.asarray(ch.out_vl)))
            ch.out_vl[j] += 1.0
            path.append((node, i, ch, j))
            node = ch.outcomes[j]

    def _backup(self, path, v: float):
        for node, i, ch, j in path:
            node.edge_vl[i] -= 1.0
            node.edge_n[i] += 1.0
            node.edge_sum[i] += v
            node.edge_sq[i] += v * v
            if j is None:                              # expansion: newest outcome
                ch.out_n[-1] += 1.0
            else:
                ch.out_vl[j] -= 1.0
                ch.out_n[j] += 1.0

    # ------------------------------------------------------------------ #
    def _run_wave(self):
        sels = [self._descend() for _ in range(self.wave)]

        # ---- 1. expansions: batched sim, grouped by parent depth (shared h/cond) ----
        new_nodes: Dict[int, _Dec] = {}
        by_depth = defaultdict(list)
        for idx, (path, kind, payload) in enumerate(sels):
            if kind == "expand":
                by_depth[payload[0].depth].append(idx)
        for depth, idxs in by_depth.items():
            parents = np.stack([sels[i][2][0].x for i in idxs])
            conds = np.stack([sels[i][2][0].c for i in idxs])
            acts = np.stack([sels[i][2][1].action for i in idxs])
            noisy = np.asarray(self.noise.sample_batch(acts, 1), np.float32).reshape(len(idxs), 4)
            posts = env_bridge.simulate_batched(parents, conds, noisy)
            h_par = sels[idxs[0]][2][0].h
            c_par = sels[idxs[0]][2][0].c
            for k, i in enumerate(idxs):               # legality is per-parent
                corrected, _ = env_bridge.apply_legality(parents[k], posts[k][None], h_par, c_par)
                posts[k] = corrected[0]
            self.budget.sims += len(idxs)
            nc = env_bridge.next_condition(c_par, self.sie)
            for k, i in enumerate(idxs):
                node, ch = sels[i][2]
                child_block = int(round(nc[2]))
                model_opponent = (self.opponent_batch_fn is not None and
                                  self.opponent_block is not None and
                                  child_block == self.opponent_block)
                child = _Dec(posts[k], nc, node.h - 1, node.depth + 1,
                             integrate_actions=model_opponent)
                ch.outcomes.append(child)
                ch.out_n.append(0.0)
                ch.out_vl.append(0.0)
                new_nodes[i] = child

        # ---- 2. evaluation targets ----
        targets: Dict[int, _Dec] = {}
        for idx, (path, kind, payload) in enumerate(sels):
            targets[idx] = new_nodes[idx] if kind == "expand" else payload
        vals: Dict[int, float] = {}
        roll = defaultdict(list)                        # (h, cond-key) -> [idx]
        for idx, node in targets.items():
            if node.h <= 0:
                vals[idx] = float(env_bridge.score_end(node.x, self.root_persp))
            else:
                roll[(node.h, tuple(np.round(node.c, 6)))].append(idx)
        for (h_g, _ck), idxs in roll.items():
            states = np.stack([targets[i].x for i in idxs])
            cond = targets[idxs[0]].c
            out = self.rollout_batch_fn(states, cond, h_g, self.root_persp)
            self.budget.sims += len(idxs) * h_g * self.rollout_mult
            for k, i in enumerate(idxs):
                vals[i] = float(out[k])

        # ---- 3. pools for new interior children (batched policy), grouped by depth ----
        need = [i for i, n in new_nodes.items()
                if n.h > 0 and n.depth < self.max_depth and n.pool is None]
        by_d2 = defaultdict(list)
        for i in need:
            by_d2[new_nodes[i].depth].append(i)
        for depth, idxs in by_d2.items():
            states = np.stack([new_nodes[i].x for i in idxs])
            cond = new_nodes[idxs[0]].c
            if new_nodes[idxs[0]].integrate_actions:
                pools = self.opponent_batch_fn(states, cond, new_nodes[idxs[0]].h,
                                               new_nodes[idxs[0]].depth,
                                               self.opponent_samples)
                pools = np.asarray(pools, np.float32).reshape(
                    len(states), self.opponent_samples, 4)
                unif = np.full(self.opponent_samples, 1.0 / self.opponent_samples)
            else:
                pools = self.sample_batch_fn(states, cond, self.inner_pool)
                unif = np.full(self.inner_pool, 1.0 / self.inner_pool)
            for k, i in enumerate(idxs):
                new_nodes[i].set_pool(pools[k], unif,
                                      open_all=new_nodes[i].integrate_actions)

        # ---- 4. backup ----
        for idx, (path, kind, payload) in enumerate(sels):
            self._backup(path, vals[idx])

    # ------------------------------------------------------------------ #
    def root_value(self) -> float:
        """Backed-up value of the root in ROOT perspective — with a single forced root
        action this IS E_noise[ value of that action under strong play by both sides ]."""
        n = self.root.edge_n[: self.root.n_open]
        s = self.root.edge_sum[: self.root.n_open]
        tot = float(n.sum())
        return float(s.sum() / tot) if tot > 0 else 0.0

    def root_stats(self):
        """(actions, Q, SE, n) over root actions with evidence — for distillation
        targets and the collect-time significance gate."""
        k = self.root.n_open
        n = self.root.edge_n[:k]
        keep = n > 0
        q = np.where(keep, self.root.edge_sum[:k] / np.maximum(n, 1), 0.0)
        var = np.maximum(np.where(keep, self.root.edge_sq[:k] / np.maximum(n, 1) - q ** 2, 0.0), 0.0)
        se = np.where(n > 1, np.sqrt(var / np.maximum(n, 1)), np.inf)
        return self.root.pool[:k][keep], q[keep], se[keep], n[keep]

    def best_action(self) -> np.ndarray:
        k = self.root.n_open
        V, W = _kr(self.root.pool[:k].astype(np.float64),
                   self.root.edge_n[:k], self.root.edge_sum[:k], self.bw)
        ok = W >= min(self.min_evidence, W.max() if W.size else 0)
        return self.root.pool[int(np.argmax(np.where(ok, V, -1e18)))].copy()

    def run(self, checkpoints: List[int]) -> Dict[int, np.ndarray]:
        out: Dict[int, np.ndarray] = {}
        for cp in sorted(checkpoints):
            stall = 0
            while self.budget.sims < cp:
                before = self.budget.sims
                self._run_wave()
                stall = stall + 1 if self.budget.sims == before else 0
                if stall > 50:
                    break
            out[cp] = self.best_action()
        return out
