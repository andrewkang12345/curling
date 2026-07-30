"""Bridge to the canonical ``csas`` simulator, scorer and candidate generators.

This is the ONLY module that touches JAX / the physics engine.  Everything
heavy is lazy-imported so that the pure-torch training path never initialises
JAX.  The simulator is the *authoritative* transition generator -- the learned
dynamics head is trained to imitate it but is never used to produce training
targets here.

All states are the canonical 24-vector (12 stones x [x,y], normalised by 4095).
All conditions are the 3-vector ``[shot_norm, team_order, stone_block]``.
All actions are physical ``[speed, angle, spin, y0]``.
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional, Tuple

import numpy as np

NUM_STONES = 12
STATE_DIM = NUM_STONES * 2  # 24
COND_DIM = 3

# --------------------------------------------------------------------------- #
# Real-curling boundary removal — DEFAULT ON since 2026-07-27 (EXP-052 era).
#
# Every simulator transition removes stones whose final CENTER is fully past
# the back line, or that touch a side board — which also makes the
# early-takeout legality rule bind the way real mixed-doubles rules intend.
# Short (hogged) stones are unchanged.
#
# Set WORLD_BOUNDARY_REMOVAL=0 to reproduce the HISTORICAL convention (the raw
# data grid's in-play mask almost never removed stones; takeout victims parked
# "spent" behind the house). All certified numbers up to and including az_v14d
# / EXP-050 were produced under the historical convention.
# --------------------------------------------------------------------------- #
BOUNDARY_REMOVAL = str(os.environ.get("WORLD_BOUNDARY_REMOVAL", "1")).lower() not in ("0", "false", "no")
BACK_LINE_REMOVE_M = 1.829 + 0.145   # back line tangent to the house + stone radius
SIDE_REMOVE_M = 2.375 - 0.145        # sheet half-width minus stone radius


def boundary_removal(posts_norm: np.ndarray) -> np.ndarray:
    """Kill stones past the back line / on the side boards (no-op unless the
    WORLD_BOUNDARY_REMOVAL env flag is set). Accepts (..., 24) normalised."""
    if not BOUNDARY_REMOVAL:
        return posts_norm
    posts = np.asarray(posts_norm, dtype=np.float32).copy()
    flat = posts.reshape(-1, NUM_STONES, 2) * 4095.0
    x, y = flat[..., 0], flat[..., 1]
    live = ((x > 0) | (y > 0)) & (x < 4095.0) & (y < 4095.0)
    along = (800.0 - y) * 0.003048
    lateral = (x - 750.0) * 0.003048
    kill = live & ((along > BACK_LINE_REMOVE_M) | (np.abs(lateral) > SIDE_REMOVE_M))
    flat[kill] = 4095.0
    return (flat / 4095.0).reshape(posts.shape).astype(np.float32)


# --------------------------------------------------------------------------- #
# Simulator transition  s' = simulate(s, a)
# --------------------------------------------------------------------------- #
def simulate(state_norm: np.ndarray, cond: np.ndarray, actions: np.ndarray) -> np.ndarray:
    """Apply one throw per action from a single root state.

    Parameters
    ----------
    state_norm : (24,) normalised pre-throw board.
    cond       : (3,)  condition (decides which block slot the new stone fills).
    actions    : (N, 4) physical actions.

    Returns
    -------
    (N, 24) normalised post-throw boards (authoritative physics).
    """
    from csas.search import _simulate_candidates

    actions = np.atleast_2d(np.asarray(actions, dtype=np.float32))
    return boundary_removal(np.asarray(_simulate_candidates(
        np.asarray(state_norm, dtype=np.float32),
        np.asarray(cond, dtype=np.float32),
        actions,
    ), dtype=np.float32))


def simulate_one(state_norm: np.ndarray, cond: np.ndarray, action: np.ndarray) -> np.ndarray:
    """Single (s, a) -> s' transition. Returns (24,)."""
    return simulate(state_norm, cond, np.asarray(action, dtype=np.float32)[None])[0]


def simulate_batched(states_norm: np.ndarray, conds: np.ndarray, actions: np.ndarray) -> np.ndarray:
    """Per-state single-action transition: states[B,24], conds[B,3], actions[B,4] -> [B,24].

    Buckets the batch by live-stone count (the JAX vmap needs a uniform n_prev) and
    calls the csas batched simulator per bucket. Used for batched policy rollouts.
    """
    from csas.search import _simulate_candidates_batched

    states = np.asarray(states_norm, dtype=np.float32).reshape(-1, STATE_DIM)
    conds = np.asarray(conds, dtype=np.float32).reshape(-1, COND_DIM)
    actions = np.asarray(actions, dtype=np.float32).reshape(-1, 4)
    B = states.shape[0]
    raw = states.reshape(B, NUM_STONES, 2) * 4095.0
    live_counts = np.array([int(((r[:, 0] > 0) | (r[:, 1] > 0)).sum() -
                                int(np.sum((r[:, 0] >= 4095.0) & (r[:, 1] >= 4095.0))))
                            for r in raw])
    # robust live count via the canonical helper
    from csas.common import in_play_raw
    live_counts = np.array([int(in_play_raw(r).sum()) for r in raw], dtype=np.int64)
    out = np.zeros((B, STATE_DIM), dtype=np.float32)
    for n_prev in np.unique(live_counts):
        idx = np.where(live_counts == n_prev)[0]
        res = _simulate_candidates_batched(states[idx], conds[idx], actions[idx][:, None, :])
        out[idx] = np.asarray(res, dtype=np.float32)[:, 0, :]
    return boundary_removal(out)


def score_end(state_norm: np.ndarray, perspective_block: int) -> float:
    """Signed end score for ``perspective_block`` (curling rules, +pts/-pts/0)."""
    from csas.generate_horizon_targets import score_end_value

    return float(score_end_value(np.asarray(state_norm, dtype=np.float32), int(perspective_block)))


def next_condition(cond: np.ndarray, shots_in_end: int) -> np.ndarray:
    from csas import common

    return np.asarray(common.next_condition(np.asarray(cond, dtype=np.float32), int(shots_in_end)),
                      dtype=np.float32)


def mixed_doubles_no_takeout_active(horizon: int) -> bool:
    from csas.generate_horizon_targets import mixed_doubles_no_takeout_active as _f

    return bool(_f(int(horizon)))


def apply_legality(pre_norm: np.ndarray, posts_norm: np.ndarray, horizon: int,
                   cond: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Enforce the 'no removing an opponent rock from play before the 4th thrown
    stone' rule (active for thrown stones 1-3, i.e. horizon >= 8).

    A throw is illegal ONLY if it knocks an opponent rock out of play; contact and
    movement that keep it in play are legal, as is removing one's own rock. Illegal
    posts are replaced by the pre-shot state (throw forfeited, displaced rocks
    restored). Returns (corrected_posts, illegal_mask).
    """
    from csas.generate_horizon_targets import replace_illegal_early_takeout_posts

    return replace_illegal_early_takeout_posts(
        np.asarray(pre_norm, dtype=np.float32),
        np.asarray(posts_norm, dtype=np.float32),
        int(horizon),
        np.asarray(cond, dtype=np.float32),
    )


def mask_illegal_scores(q: np.ndarray, illegal: np.ndarray, penalty: float = -1.0e6) -> np.ndarray:
    from csas.generate_horizon_targets import mask_illegal_action_scores

    return mask_illegal_action_scores(np.asarray(q, dtype=np.float64),
                                      np.asarray(illegal, dtype=bool), penalty)


def kr_smooth(actions: np.ndarray, values: np.ndarray, action_mean: np.ndarray,
              action_std: np.ndarray, bandwidth: float, uct_c: float) -> np.ndarray:
    from csas.search import kr_smooth_scores

    return kr_smooth_scores(np.asarray(actions, dtype=np.float64),
                            np.asarray(values, dtype=np.float64),
                            np.asarray(action_mean, dtype=np.float64),
                            np.asarray(action_std, dtype=np.float64),
                            float(bandwidth), float(uct_c))


def soft_topk(scores: np.ndarray, top_k: int, temperature: float) -> Tuple[np.ndarray, np.ndarray]:
    """Return (top_indices, softmax_weights) over the highest-scoring candidates."""
    from csas.generate_horizon_targets import _soft_topk

    return _soft_topk(np.asarray(scores, dtype=np.float64), int(top_k), float(temperature))


# --------------------------------------------------------------------------- #
# Candidate generation (mixed multi-source pool)
# --------------------------------------------------------------------------- #
def diverse_grid_actions(pre_state: np.ndarray, n_limit: int) -> np.ndarray:
    from csas.generate_horizon_targets import _diverse_grid_actions

    return np.asarray(_diverse_grid_actions(np.asarray(pre_state, dtype=np.float32), int(n_limit)),
                      dtype=np.float32)


def structured_actions(pre_state: np.ndarray, n_limit: int) -> np.ndarray:
    from csas.generate_horizon_targets import _generic_structured_actions

    return np.asarray(_generic_structured_actions(np.asarray(pre_state, dtype=np.float32), int(n_limit)),
                      dtype=np.float32)


def global_actions(n_samples: int, rng: np.random.Generator) -> np.ndarray:
    from csas.generate_horizon_targets import _global_fallback

    return np.asarray(_global_fallback(int(n_samples), rng), dtype=np.float32)


def local_perturbations(seed_actions: np.ndarray, n_samples: int, std: np.ndarray,
                        rng: np.random.Generator) -> np.ndarray:
    from csas.generate_horizon_targets import _local_perturbations

    return np.asarray(_local_perturbations(np.asarray(seed_actions, dtype=np.float32),
                                           int(n_samples), np.asarray(std, dtype=np.float32), rng),
                      dtype=np.float32)


# --------------------------------------------------------------------------- #
# Checkpoint loading + direct value evaluation (avoids csas.evaluate_states bug)
# --------------------------------------------------------------------------- #
def load_csas_policy(path: str, device):
    """Returns (policy_model, action_mean[4], action_std[4]) as numpy."""
    from csas.search import load_policy

    model, amean, astd = load_policy(path, device)
    return model, np.asarray(amean.detach().cpu().numpy(), dtype=np.float32), \
        np.asarray(astd.detach().cpu().numpy(), dtype=np.float32)


def load_csas_value(path: str, device):
    from csas.search import load_value_model

    return load_value_model(path, device)


def evaluate_value(value_model, states_norm: np.ndarray, cond: np.ndarray, device,
                   batch_size: int = 256) -> np.ndarray:
    """Mean predicted value per state (calls the gaussian value model directly).

    batch_size is small (256) because the GraphTF curl-arc edge features allocate
    O(batch * stones^2 * arcs) tensors -- larger batches OOM (e.g. noise-expanded
    candidate sets of ~2k states).

    We deliberately do NOT use ``csas.search.evaluate_states`` -- that function
    references ``sys.modules`` without importing ``sys`` and NameErrors on its
    precomputed-graph fast path.
    """
    import torch

    batch_size = int(os.environ.get("VALUE_EVAL_BATCH", batch_size))  # GPU-sharing knob
    states = np.atleast_2d(np.asarray(states_norm, dtype=np.float32))
    cond = np.asarray(cond, dtype=np.float32)
    c = np.broadcast_to(cond, (states.shape[0], COND_DIM)).astype(np.float32)
    out = np.empty(states.shape[0], dtype=np.float32)
    value_model.eval()
    with torch.no_grad():
        for i in range(0, states.shape[0], batch_size):
            xb = torch.as_tensor(states[i:i + batch_size], device=device)
            cb = torch.as_tensor(c[i:i + batch_size], device=device)
            res = value_model(xb, cb)
            mean = res[0] if isinstance(res, (tuple, list)) else res
            out[i:i + batch_size] = mean.squeeze(-1).float().cpu().numpy()
    return out


@lru_cache(maxsize=1)
def warm_jax() -> str:
    """Touch the simulator once so the first real call is not paying compile time."""
    import jax  # noqa: F401

    z = np.zeros(STATE_DIM, dtype=np.float32)
    c = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    simulate(z, c, np.array([[1.2, 0.0, 0.5, 0.0]], dtype=np.float32))
    import jax as _jax

    return _jax.default_backend()


__all__ = [
    "NUM_STONES", "STATE_DIM", "COND_DIM",
    "simulate", "simulate_one", "score_end", "next_condition",
    "mixed_doubles_no_takeout_active", "apply_legality", "mask_illegal_scores",
    "kr_smooth", "soft_topk",
    "diverse_grid_actions", "structured_actions", "global_actions", "local_perturbations",
    "load_csas_policy", "load_csas_value", "evaluate_value", "warm_jax",
]
