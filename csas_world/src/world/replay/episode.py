"""Helpers for assembling K-step unroll records and n-step value targets.

Curling ends are episodic with a single terminal reward (the end-score
differential).  Value targets along a greedy rollout are the kernel-smoothed
search value at each visited state, evaluated from the to-move perspective, with
the sign alternating each ply (zero-sum).  ``gamma`` defaults to 1.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from .schema import empty_record


def nstep_value_targets(step_values: Sequence[float], rewards: Sequence[float],
                        gamma: float = 1.0, n: int = 5) -> np.ndarray:
    """n-step bootstrapped returns.

    step_values[k] = bootstrap value estimate at state k (to-move perspective).
    rewards[k]     = reward received transitioning k -> k+1.
    Returns an array aligned to step_values (length T).
    """
    T = len(step_values)
    out = np.zeros(T, dtype=np.float32)
    for k in range(T):
        g = 0.0
        discount = 1.0
        last = k
        for j in range(k, min(k + n, T - 1)):
            g += discount * float(rewards[j])
            discount *= gamma
            last = j + 1
        # bootstrap from the value at the horizon (sign already in perspective)
        g += discount * float(step_values[last])
        out[k] = g
    return out


def build_unroll_record(K: int, M: int, *, x0: np.ndarray, c0: np.ndarray,
                        actions_raw: np.ndarray, next_states: np.ndarray,
                        next_conds: np.ndarray, value_targets: np.ndarray,
                        rewards: np.ndarray, outcome_margin: float, source: int,
                        horizon: int, dist_actions_raw: Optional[np.ndarray] = None,
                        dist_weights: Optional[np.ndarray] = None,
                        live_mask_fn=None) -> Dict[str, np.ndarray]:
    """Assemble one fixed-shape record from a greedy rollout of length k_eff<=K."""
    r = empty_record(K, M)
    r["x0"] = x0.astype(np.float32)
    r["c0"] = c0.astype(np.float32)
    r["horizon"] = np.int64(horizon)
    r["source"] = np.int64(source)
    r["outcome_margin"] = np.float32(outcome_margin)
    r["outcome_mask"] = np.float32(1.0)

    k_eff = min(K, len(actions_raw))
    if k_eff > 0:
        r["a_raw"][:k_eff] = actions_raw[:k_eff]
        r["next_states"][:k_eff] = next_states[:k_eff]
        r["next_conds"][:k_eff] = next_conds[:k_eff]
        r["consistency_mask"][:k_eff] = 1.0
        if live_mask_fn is not None:
            for j in range(k_eff):
                r["next_live"][j] = live_mask_fn(next_states[j])
        r["reward_target"][:k_eff] = rewards[:k_eff]
        # reward target is meaningful for steps that transition the env
        r["reward_mask"][:k_eff] = 1.0

    # value targets for steps 0..k_eff (value at root + each rolled state)
    nv = min(K + 1, len(value_targets))
    r["value_target"][:nv] = value_targets[:nv]
    r["value_mask"][:nv] = 1.0

    # MCTS distillation candidates at the root
    if dist_actions_raw is not None and dist_weights is not None and len(dist_actions_raw) > 0:
        m = min(M, len(dist_actions_raw))
        r["dist_actions_raw"][:m] = dist_actions_raw[:m]
        w = dist_weights[:m].astype(np.float32)
        if w.sum() > 0:
            w = w / w.sum()
        r["dist_weights"][:m] = w
        r["dist_mask"] = np.float32(1.0)
    return r


__all__ = ["nstep_value_targets", "build_unroll_record"]
