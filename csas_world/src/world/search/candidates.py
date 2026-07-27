"""Candidate-throw generation for KR-UCT search / target collection.

Builds the mixed multi-source candidate pool used by the canonical pipeline:
  policy-sampled  + diverse structured grid (draws/takeouts/ticks on reachable
  stones) + hand-structured seeds + local perturbations + global uniform.
This realises the "~96 legal policy draws + ~96 diverse shots" set.

The learned-dynamics prefilter hook (``use_learned_model_prefilter``) is wired
but inert by default -- a place to later score candidates with G/V before
spending simulator time on them.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .. import env_bridge
from ..actions import ACTION_HIGH, ACTION_LOW, clip_raw
from ..config import SearchCfg

_LOCAL_STD = np.array([0.10, 0.045, 0.40, 0.055], dtype=np.float32)


def _dedup(actions: np.ndarray, decimals: int = 5) -> np.ndarray:
    if len(actions) == 0:
        return actions
    _, idx = np.unique(np.round(actions, decimals), axis=0, return_index=True)
    return actions[np.sort(idx)]


def generate_candidates(policy, action_mean_t, action_std_t, x: np.ndarray, c: np.ndarray,
                        cfg: SearchCfg, rng: np.random.Generator, device,
                        world_model=None) -> np.ndarray:
    """Returns a deduped (N,4) array of physical candidate actions."""
    from csas.search import _sample_actions

    parts = []
    if cfg.policy_candidates > 0:
        parts.append(_sample_actions(policy, action_mean_t, action_std_t, x, c,
                                     cfg.policy_candidates, device, cfg.temperature,
                                     cfg.std_scale, global_frac=0.0))
    if cfg.diverse_candidates > 0:
        parts.append(env_bridge.diverse_grid_actions(x, cfg.diverse_candidates))
    if cfg.structured_candidates > 0:
        parts.append(env_bridge.structured_actions(x, cfg.structured_candidates))
    if cfg.global_candidates > 0:
        parts.append(env_bridge.global_actions(cfg.global_candidates, rng))

    pool = np.concatenate([p for p in parts if len(p)], axis=0).astype(np.float32)
    if cfg.local_candidates > 0 and len(pool):
        n_seed = min(16, len(pool))
        seeds = pool[:n_seed]
        loc = env_bridge.local_perturbations(seeds, cfg.local_candidates, _LOCAL_STD, rng)
        pool = np.concatenate([pool, loc], axis=0)

    pool = clip_raw(pool)
    pool = _dedup(pool)

    if cfg.use_learned_model_prefilter and world_model is not None:
        pool = _learned_prefilter(world_model, x, c, pool, device,
                                  keep=cfg.policy_candidates + cfg.diverse_candidates)
    return pool


def _learned_prefilter(world_model, x: np.ndarray, c: np.ndarray, pool: np.ndarray,
                       device, keep: int) -> np.ndarray:
    """Rank candidates by learned G+V (cheap) and keep the top-``keep`` (ablation hook)."""
    import torch

    if len(pool) <= keep:
        return pool
    with torch.no_grad():
        xt = torch.as_tensor(x[None], dtype=torch.float32, device=device)
        ct = torch.as_tensor(c[None], dtype=torch.float32, device=device)
        h0 = world_model.encode(xt, ct)
        a_raw = torch.as_tensor(pool, dtype=torch.float32, device=device)
        a_box = world_model.raw_to_box(a_raw)
        h1 = world_model.step_dynamics(h0.expand(len(pool), -1), a_box)
        v = world_model.value_head.value(h1).cpu().numpy()
    top = np.argsort(-v)[:keep]
    return pool[top]


__all__ = ["generate_candidates"]
