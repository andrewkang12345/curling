#!/usr/bin/env python3

# curling_inverse.py
# CEM inverse solver with a HARD loss that supports stones going off-sheet.
# The loss does rectangular matching:
# - Only predicted stones that end IN-BOUNDS are eligible to match to targets.
# - Targets are the IN-BOUNDS stones from your CSV next-state.
# - Any *unmatched target* gets a fixed miss penalty.
# - Any *unmatched in-bounds prediction* (i.e., stone that should have been removed but wasn't)
#   also gets a fixed penalty.
#
# This keeps the solver faithful when stones are knocked out (disappear).

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional
import logging
import numpy as np

import jax
import jax.numpy as jnp
from jax import random, lax

from curling_sim_jax import (
    CurlingParams,
    simulate_from_params,
    simulate_from_params_flex,
    phys_array_from_params,
    N_PHYS,
)

log = logging.getLogger("curling")

# ----------------------------
# Board extents (in *sim meters*)
# These match the CSV→meters mapping used in your viz scripts:
#   x_m = (800 - y_csv) * 0.003048         # along-sheet (button at 0)
#   y_m = (x_csv - 750) * 0.003048         # lateral (centerline at 0)
#
# CSV y ∈ [0,3000]  -> x_m ∈ [-6.7056, +2.4384]
# CSV x ∈ [0,1500]  -> y_m ∈ [-2.2860, +2.2860]
# ----------------------------
MIN_X = -6.7056
MAX_X =  2.4384
MIN_Y = -2.2860
MAX_Y =  2.2860

# Upper bounds (m^2). Effective penalty per stone becomes <= these caps.
MISS_TARGET_MAX    = 4.0
EXTRA_PRED_MAX     = 4.0
# Soft “how close to dead?” length scale (meters)
DEATH_MARGIN_SCALE = 0.20
BIG                 = 1e6  # internal masking for greedy assignment
MAX_SLOT_STONES     = 12
BLOCK_SLOT_STONES   = 6
BLOCK_PRED_STONES   = 7
TRANSITION_MOVE_THRESH_M = 0.06
UNCHANGED_STONE_W = 0.35
MOVED_STONE_W = 1.25
NEW_OR_REMOVED_STONE_W = 2.0
THROWN_STONE_EXTRA_W = 1.0
SLOT_HYBRID_SET_W = 0.25


# ----------------------------
# Bounds & helpers
# ----------------------------

@dataclass(frozen=True)
class SolveBounds:
    speed_min: float = 0.1
    speed_max: float = 3.0
    angle_min: float = -0.35
    angle_max: float = 0.35
    spin_min: float = -3.0
    spin_max: float = 3.0
    y0_min: float = -0.23
    y0_max: float = 0.23

def _make_bounds_arrays(b: SolveBounds):
    lo = jnp.array([b.speed_min, b.angle_min, b.spin_min, b.y0_min])
    hi = jnp.array([b.speed_max, b.angle_max, b.spin_max, b.y0_max])
    span = hi - lo
    return lo, hi, span

def _x_phys_from_x01(x01: jnp.ndarray, lo: jnp.ndarray, span: jnp.ndarray) -> jnp.ndarray:
    return lo + jnp.clip(x01, 0.0, 1.0) * span

def _x01_from_x_phys(x: jnp.ndarray, lo: jnp.ndarray, span: jnp.ndarray) -> jnp.ndarray:
    return jnp.clip((x - lo) / (span + 1e-8), 0.0, 1.0)

def _safe_metric(v: jnp.ndarray) -> jnp.ndarray:
    return jnp.nan_to_num(v, nan=1e9, posinf=1e9, neginf=1e9)

# ----------------------------
# HARD loss that allows stones to disappear (off-sheet)
# ----------------------------

def _in_bounds_mask(pos: jnp.ndarray) -> jnp.ndarray:
    """Boolean mask for stones that end within the board rectangle."""
    x = pos[:, 0]
    y = pos[:, 1]
    return (x > MIN_X) & (x < MAX_X) & (y > MIN_Y) & (y < MAX_Y)

def _rect_greedy_cost(pred_final: jnp.ndarray, target_xy: jnp.ndarray) -> jnp.ndarray:
    """
    Greedy rectangular assignment cost between IN-BOUNDS predictions and targets,
    with distance-aware penalties for unmatched items based on proximity to the boundary.
    """
    Np = pred_final.shape[0]
    Nt = target_xy.shape[0]

    # Which predictions are eligible to match (must be in-bounds)
    inb = _in_bounds_mask(pred_final)
    N_in = jnp.sum(inb.astype(jnp.int32))

    # Early-out: no targets: pay a soft penalty for each in-bounds predicted stone (unmatched)
    if Nt == 0:
        # Weight by how far inside those predictions are (near edge ⇒ smaller penalty)
        margins_pred = _signed_margin(pred_final)
        w_pred = _soft_weight_from_margin(margins_pred) * inb.astype(pred_final.dtype)
        return EXTRA_PRED_MAX * jnp.sum(w_pred)

    # Pairwise squared distances for (eligible rows) x (targets)
    d2 = jnp.sum((pred_final[:, None, :] - target_xy[None, :, :]) ** 2, axis=2)  # (Np, Nt)
    C = d2 + (~inb)[:, None] * BIG  # mask out-of-bounds prediction rows

    # Greedy select K = min(N_in, Nt) matches; track which rows/cols were used
    K = jnp.minimum(N_in, jnp.int32(Nt))
    used_r = jnp.zeros((Np,), dtype=jnp.bool_)
    used_c = jnp.zeros((Nt,), dtype=jnp.bool_)
    total  = jnp.array(0.0, dtype=pred_final.dtype)

    def body(s, carry):
        C_curr, total_curr, used_r_curr, used_c_curr = carry
        # argmin (global)
        idx_flat = jnp.argmin(C_curr)
        i = idx_flat // Nt
        j = idx_flat %  Nt
        val = C_curr[i, j]

        def _do_update(_):
            total_new = total_curr + val
            used_r_new = used_r_curr.at[i].set(True)
            used_c_new = used_c_curr.at[j].set(True)
            # Invalidate selected row/col
            C1 = C_curr.at[i, :].set(BIG)
            C1 = C1.at[:, j].set(BIG)
            return (C1, total_new, used_r_new, used_c_new)

        def _skip(_):
            return (C_curr, total_curr, used_r_curr, used_c_curr)

        # Only actually select while s < K (after that, loop is a no-op)
        return lax.cond(s < K, _do_update, _skip, operand=None)

    C0 = C
    carry0 = (C0, total, used_r, used_c)
    C_fin, matched_sum, used_r_fin, used_c_fin = lax.fori_loop(0, Nt, body, carry0)

    # Compute soft penalties for the unmatched ones
    # Targets are *by construction* in-bounds. Weight by how far inside each target is.
    tgt_margins = _signed_margin(target_xy)                    # (Nt,)
    tgt_w       = _soft_weight_from_margin(tgt_margins)        # (Nt,)
    tgt_unmatched_mask = (~used_c_fin).astype(pred_final.dtype)
    miss_target_pen = MISS_TARGET_MAX * jnp.sum(tgt_w * tgt_unmatched_mask)

    # In-bounds predicted stones left unmatched
    pred_margins = _signed_margin(pred_final)                  # (Np,)
    pred_w       = _soft_weight_from_margin(pred_margins)      # (Np,)
    pred_unmatched_mask = (inb & (~used_r_fin)).astype(pred_final.dtype)
    extra_pred_pen = EXTRA_PRED_MAX * jnp.sum(pred_w * pred_unmatched_mask)

    return matched_sum + miss_target_pen + extra_pred_pen


def _rect_greedy_cost_masked(
    pred_final: jnp.ndarray,
    pred_mask: jnp.ndarray,
    target_xy: jnp.ndarray,
    target_mask: jnp.ndarray,
) -> jnp.ndarray:
    """
    Fixed-shape rectangular assignment cost.
    `pred_mask` and `target_mask` mark which rows are active so JAX can reuse one
    compiled kernel across shots with different stone counts.
    """
    Np = pred_final.shape[0]
    Nt = target_xy.shape[0]

    pred_active = pred_mask & _in_bounds_mask(pred_final)
    target_active = target_mask

    d2 = jnp.sum((pred_final[:, None, :] - target_xy[None, :, :]) ** 2, axis=2)
    valid_pairs = pred_active[:, None] & target_active[None, :]
    C = jnp.where(valid_pairs, d2, BIG)

    K = jnp.minimum(
        jnp.sum(pred_active.astype(jnp.int32)),
        jnp.sum(target_active.astype(jnp.int32)),
    )

    used_r = jnp.zeros((Np,), dtype=jnp.bool_)
    used_c = jnp.zeros((Nt,), dtype=jnp.bool_)
    total = jnp.array(0.0, dtype=pred_final.dtype)

    def body(s, carry):
        C_curr, total_curr, used_r_curr, used_c_curr = carry
        idx_flat = jnp.argmin(C_curr)
        i = idx_flat // Nt
        j = idx_flat % Nt
        val = C_curr[i, j]

        def _do_update(_):
            total_new = total_curr + val
            used_r_new = used_r_curr.at[i].set(True)
            used_c_new = used_c_curr.at[j].set(True)
            C1 = C_curr.at[i, :].set(BIG)
            C1 = C1.at[:, j].set(BIG)
            return (C1, total_new, used_r_new, used_c_new)

        return lax.cond(s < K, _do_update, lambda _: carry, operand=None)

    _, matched_sum, used_r_fin, used_c_fin = lax.fori_loop(
        0,
        max(Np, Nt),
        body,
        (C, total, used_r, used_c),
    )

    tgt_margins = _signed_margin(target_xy)
    tgt_w = _soft_weight_from_margin(tgt_margins)
    tgt_unmatched = (target_active & (~used_c_fin)).astype(pred_final.dtype)
    miss_target_pen = MISS_TARGET_MAX * jnp.sum(tgt_w * tgt_unmatched)

    pred_margins = _signed_margin(pred_final)
    pred_w = _soft_weight_from_margin(pred_margins)
    pred_unmatched = (pred_active & (~used_r_fin)).astype(pred_final.dtype)
    extra_pred_pen = EXTRA_PRED_MAX * jnp.sum(pred_w * pred_unmatched)

    return matched_sum + miss_target_pen + extra_pred_pen


def _transition_block_weights(
    prev_block_pos: jnp.ndarray,
    prev_block_mask: jnp.ndarray,
    target_xy: jnp.ndarray,
    target_mask: jnp.ndarray,
    thrown_active: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    move_thresh2 = TRANSITION_MOVE_THRESH_M ** 2
    prev_to_target_d2 = jnp.sum((prev_block_pos - target_xy) ** 2, axis=1)

    unchanged = prev_block_mask & target_mask & (prev_to_target_d2 <= move_thresh2)
    moved = prev_block_mask & target_mask & (prev_to_target_d2 > move_thresh2)
    new_target = (~prev_block_mask) & target_mask
    removed_prev = prev_block_mask & (~target_mask)

    target_w = (
        unchanged.astype(target_xy.dtype) * UNCHANGED_STONE_W
        + moved.astype(target_xy.dtype) * MOVED_STONE_W
        + new_target.astype(target_xy.dtype) * NEW_OR_REMOVED_STONE_W
    )
    target_w = jnp.where(target_mask, target_w, 0.0)

    pred_slot_w = (
        unchanged.astype(prev_block_pos.dtype) * UNCHANGED_STONE_W
        + moved.astype(prev_block_pos.dtype) * 1.0
        + removed_prev.astype(prev_block_pos.dtype) * NEW_OR_REMOVED_STONE_W
    )
    pred_slot_w = jnp.where(prev_block_mask, pred_slot_w, 0.0)
    thrown_w = jnp.where(thrown_active, jnp.asarray([THROWN_STONE_EXTRA_W], dtype=prev_block_pos.dtype), jnp.asarray([0.0], dtype=prev_block_pos.dtype))
    pred_extra_w = jnp.concatenate([pred_slot_w, thrown_w], axis=0)
    return target_w, target_w, pred_extra_w


def _rect_optimal_cost_masked(
    pred_final: jnp.ndarray,
    pred_mask: jnp.ndarray,
    target_xy: jnp.ndarray,
    target_mask: jnp.ndarray,
    target_match_w: jnp.ndarray,
    target_miss_w: jnp.ndarray,
    pred_extra_w: jnp.ndarray,
) -> jnp.ndarray:
    pred_active = pred_mask & _in_bounds_mask(pred_final)
    target_active = target_mask

    d2 = jnp.sum((pred_final[:, None, :] - target_xy[None, :, :]) ** 2, axis=2)
    pair_cost = d2 * target_match_w[None, :]
    pred_extra_cost = EXTRA_PRED_MAX * pred_extra_w
    target_miss_cost = MISS_TARGET_MAX * target_miss_w

    Np = pred_final.shape[0]
    Nt = target_xy.shape[0]
    num_states = 1 << Nt
    mask_ids = jnp.arange(num_states, dtype=jnp.int32)
    bit_ids = (1 << jnp.arange(Nt, dtype=jnp.int32))
    inf = jnp.asarray(BIG * 10.0, dtype=pred_final.dtype)

    dp = jnp.full((num_states,), inf, dtype=pred_final.dtype).at[0].set(0.0)

    def step(dp_cur, i):
        def active_step(dp_in):
            dp_next = dp_in + pred_extra_cost[i]
            for j in range(Nt):
                bit = bit_ids[j]
                valid = target_active[j] & ((mask_ids & bit) == 0)
                cand = jnp.where(valid, dp_in + pair_cost[i, j], inf)
                dp_next = dp_next.at[mask_ids | bit].min(cand)
            return dp_next

        return lax.cond(pred_active[i], active_step, lambda x: x, dp_cur), None

    dp, _ = lax.scan(step, dp, jnp.arange(Np, dtype=jnp.int32))

    matched_targets = (mask_ids[:, None] & bit_ids[None, :]) != 0
    miss_mask = target_active[None, :] & (~matched_targets)
    total = dp + jnp.sum(jnp.where(miss_mask, target_miss_cost[None, :], 0.0), axis=1)
    return jnp.min(total)


def _rect_cost_masked(
    pred_final: jnp.ndarray,
    pred_mask: jnp.ndarray,
    target_xy: jnp.ndarray,
    target_mask: jnp.ndarray,
    target_match_w: jnp.ndarray,
    target_miss_w: jnp.ndarray,
    pred_extra_w: jnp.ndarray,
    assignment_mode: str,
) -> jnp.ndarray:
    pred_active = pred_mask & _in_bounds_mask(pred_final)
    target_active = target_mask

    if assignment_mode == "greedy":
        d2 = jnp.sum((pred_final[:, None, :] - target_xy[None, :, :]) ** 2, axis=2)
        weighted_target_xy = target_xy
        weighted_d2 = d2 * target_match_w[None, :]
        C = jnp.where(pred_active[:, None] & target_active[None, :], weighted_d2, BIG)

        Np = pred_final.shape[0]
        Nt = target_xy.shape[0]
        K = jnp.minimum(
            jnp.sum(pred_active.astype(jnp.int32)),
            jnp.sum(target_active.astype(jnp.int32)),
        )
        used_r = jnp.zeros((Np,), dtype=jnp.bool_)
        used_c = jnp.zeros((Nt,), dtype=jnp.bool_)
        total = jnp.array(0.0, dtype=pred_final.dtype)

        def body(s, carry):
            C_curr, total_curr, used_r_curr, used_c_curr = carry
            idx_flat = jnp.argmin(C_curr)
            i = idx_flat // Nt
            j = idx_flat % Nt
            val = C_curr[i, j]

            def _do_update(_):
                total_new = total_curr + val
                used_r_new = used_r_curr.at[i].set(True)
                used_c_new = used_c_curr.at[j].set(True)
                C1 = C_curr.at[i, :].set(BIG)
                C1 = C1.at[:, j].set(BIG)
                return (C1, total_new, used_r_new, used_c_new)

            return lax.cond(s < K, _do_update, lambda _: carry, operand=None)

        _, matched_sum, used_r_fin, used_c_fin = lax.fori_loop(
            0,
            max(pred_final.shape[0], target_xy.shape[0]),
            body,
            (C, total, used_r, used_c),
        )

        miss_target_pen = MISS_TARGET_MAX * jnp.sum(
            target_miss_w * (target_active & (~used_c_fin)).astype(pred_final.dtype)
        )
        extra_pred_pen = EXTRA_PRED_MAX * jnp.sum(
            pred_extra_w * (pred_active & (~used_r_fin)).astype(pred_final.dtype)
        )
        return matched_sum + miss_target_pen + extra_pred_pen

    return _rect_optimal_cost_masked(
        pred_final,
        pred_mask,
        target_xy,
        target_mask,
        target_match_w,
        target_miss_w,
        pred_extra_w,
    )


def _distance_cost(d2: jnp.ndarray, *, use_huber: bool = False, delta: float = 0.30) -> jnp.ndarray:
    if not use_huber:
        return d2
    d = jnp.sqrt(jnp.maximum(d2, 0.0) + 1e-8)
    return jnp.where(d <= delta, 0.5 * d2, delta * (d - 0.5 * delta))


def _slot_identity_block_cost(
    pred_final: jnp.ndarray,
    pred_mask: jnp.ndarray,
    prev_block_pos: jnp.ndarray,
    prev_block_mask: jnp.ndarray,
    target_xy: jnp.ndarray,
    target_mask: jnp.ndarray,
    thrown_active: jnp.ndarray,
    *,
    transition_reweight: bool = False,
    hybrid_set_weight: float = 0.0,
    use_huber: bool = False,
) -> jnp.ndarray:
    slot_pred = pred_final[:BLOCK_SLOT_STONES]
    slot_mask = pred_mask[:BLOCK_SLOT_STONES]
    thrown_pred = pred_final[BLOCK_SLOT_STONES:BLOCK_SLOT_STONES + 1]
    thrown_mask = pred_mask[BLOCK_SLOT_STONES]

    slot_inb = _in_bounds_mask(slot_pred)
    thrown_inb = _in_bounds_mask(thrown_pred)[0]
    target_margin_w = _soft_weight_from_margin(_signed_margin(target_xy))
    slot_margin_w = _soft_weight_from_margin(_signed_margin(slot_pred))
    thrown_margin_w = _soft_weight_from_margin(_signed_margin(thrown_pred))[0]

    move_thresh2 = TRANSITION_MOVE_THRESH_M ** 2
    prev_to_target_d2 = jnp.sum((prev_block_pos - target_xy) ** 2, axis=1)
    unchanged = prev_block_mask & target_mask & (prev_to_target_d2 <= move_thresh2)
    moved = prev_block_mask & target_mask & (prev_to_target_d2 > move_thresh2)
    removed_prev = prev_block_mask & (~target_mask)
    new_target = (~prev_block_mask) & target_mask

    dtype = pred_final.dtype
    if transition_reweight:
        surv_w = (
            unchanged.astype(dtype) * UNCHANGED_STONE_W
            + moved.astype(dtype) * MOVED_STONE_W
        )
        removed_w = removed_prev.astype(dtype) * NEW_OR_REMOVED_STONE_W
        new_w = new_target.astype(dtype) * (NEW_OR_REMOVED_STONE_W + THROWN_STONE_EXTRA_W)
    else:
        surv_w = (prev_block_mask & target_mask).astype(dtype)
        removed_w = removed_prev.astype(dtype)
        new_w = new_target.astype(dtype)

    surv_mask = prev_block_mask & target_mask
    surv_alive = surv_mask & slot_mask & slot_inb
    surv_missing = surv_mask & (~(slot_mask & slot_inb))
    surv_d2 = jnp.sum((slot_pred - target_xy) ** 2, axis=1)
    surv_cost = jnp.sum(jnp.where(surv_alive, surv_w * _distance_cost(surv_d2, use_huber=use_huber), 0.0))
    surv_miss_cost = MISS_TARGET_MAX * jnp.sum(jnp.where(surv_missing, surv_w * target_margin_w, 0.0))

    removed_alive = removed_prev & slot_mask & slot_inb
    removed_cost = EXTRA_PRED_MAX * jnp.sum(jnp.where(removed_alive, removed_w * slot_margin_w, 0.0))

    new_count = jnp.sum(new_target.astype(jnp.int32))
    thrown_should_exist = thrown_active & (new_count > 0)
    thrown_alive = thrown_active & thrown_mask & thrown_inb
    thrown_d2 = jnp.sum((target_xy - thrown_pred[0]) ** 2, axis=1)

    # If the next state has exactly one new target slot in this block, align the thrown stone to it.
    new_match_cost = jnp.sum(jnp.where(new_target, new_w * _distance_cost(thrown_d2, use_huber=use_huber), 0.0))
    new_miss_cost = MISS_TARGET_MAX * jnp.sum(jnp.where(new_target, new_w * target_margin_w, 0.0))
    thrown_extra_cost = EXTRA_PRED_MAX * jnp.where(
        thrown_alive & (~(new_count > 0)),
        (NEW_OR_REMOVED_STONE_W + THROWN_STONE_EXTRA_W) * thrown_margin_w,
        0.0,
    )

    total = surv_cost + surv_miss_cost + removed_cost
    total = total + jnp.where(thrown_should_exist, jnp.where(thrown_alive, new_match_cost, new_miss_cost), 0.0)
    total = total + thrown_extra_cost

    if hybrid_set_weight > 0.0:
        target_match_w = jnp.ones((target_xy.shape[0],), dtype=dtype)
        target_miss_w = target_margin_w
        pred_extra_w = jnp.concatenate([slot_margin_w, jnp.asarray([thrown_margin_w], dtype=dtype)], axis=0)
        total = total + hybrid_set_weight * _rect_cost_masked(
            pred_final,
            pred_mask,
            target_xy,
            target_mask,
            target_match_w,
            target_miss_w,
            pred_extra_w,
            "greedy",
        )

    return total

def _hard_loss_allow_deaths(p: CurlingParams,
                            prev_positions_button: jnp.ndarray,
                            target_positions_button: jnp.ndarray,
                            x: jnp.ndarray) -> jnp.ndarray:
    raise NotImplementedError(
        "Ownership-agnostic inverse loss has been removed. "
        "Use _hard_loss_allow_deaths_by_block instead."
    )

def _make_batched_hard_loss(p, prev, target):
    raise NotImplementedError(
        "Ownership-agnostic inverse loss has been removed. "
        "Use _make_batched_hard_loss_by_block instead."
    )


def _hard_loss_allow_deaths_by_block(
    p: CurlingParams,
    prev_positions_button: jnp.ndarray,
    prev_slot_mask: jnp.ndarray,
    thrower_block: jnp.ndarray,
    target_positions_block0: jnp.ndarray,
    target_mask_block0: jnp.ndarray,
    target_positions_block1: jnp.ndarray,
    target_mask_block1: jnp.ndarray,
    x: jnp.ndarray,
    *,
    loss_variant: str = "current",
) -> jnp.ndarray:
    """
    Ownership-aware hard loss:
    - simulate to final positions
    - keep a fixed 12-slot board plus the thrown stone
    - split predicted stones by ownership block (1..6 vs 7..12)
    - match each block only against the corresponding masked target block
    """
    pred_final = simulate_from_params(p, prev_positions_button, x, dynamic=False)
    pred_block0 = jnp.concatenate([pred_final[:BLOCK_SLOT_STONES], pred_final[12:13]], axis=0)
    pred_block1 = jnp.concatenate([pred_final[BLOCK_SLOT_STONES:MAX_SLOT_STONES], pred_final[12:13]], axis=0)
    pred_mask0 = jnp.concatenate(
        [prev_slot_mask[:BLOCK_SLOT_STONES], jnp.asarray([thrower_block == 0], dtype=jnp.bool_)],
        axis=0,
    )
    pred_mask1 = jnp.concatenate(
        [prev_slot_mask[BLOCK_SLOT_STONES:MAX_SLOT_STONES], jnp.asarray([thrower_block == 1], dtype=jnp.bool_)],
        axis=0,
    )
    assignment_mode = "greedy"
    transition_reweight = False
    slot_identity_mode = False
    hybrid_set_weight = 0.0
    use_huber = False
    if loss_variant == "optimal":
        assignment_mode = "optimal"
    elif loss_variant == "greedy_transition":
        transition_reweight = True
    elif loss_variant == "optimal_transition":
        assignment_mode = "optimal"
        transition_reweight = True
    elif loss_variant == "slot_identity":
        slot_identity_mode = True
    elif loss_variant == "slot_transition":
        slot_identity_mode = True
        transition_reweight = True
    elif loss_variant == "slot_identity_hybrid":
        slot_identity_mode = True
        hybrid_set_weight = SLOT_HYBRID_SET_W
    elif loss_variant == "slot_transition_hybrid":
        slot_identity_mode = True
        transition_reweight = True
        hybrid_set_weight = SLOT_HYBRID_SET_W
    elif loss_variant == "slot_transition_huber":
        slot_identity_mode = True
        transition_reweight = True
        use_huber = True
    elif loss_variant == "slot_transition_huber_hybrid":
        slot_identity_mode = True
        transition_reweight = True
        use_huber = True
        hybrid_set_weight = SLOT_HYBRID_SET_W

    if slot_identity_mode:
        return (
            _slot_identity_block_cost(
                pred_block0,
                pred_mask0,
                prev_positions_button[:BLOCK_SLOT_STONES],
                prev_slot_mask[:BLOCK_SLOT_STONES],
                target_positions_block0,
                target_mask_block0,
                thrower_block == 0,
                transition_reweight=transition_reweight,
                hybrid_set_weight=hybrid_set_weight,
                use_huber=use_huber,
            )
            + _slot_identity_block_cost(
                pred_block1,
                pred_mask1,
                prev_positions_button[BLOCK_SLOT_STONES:MAX_SLOT_STONES],
                prev_slot_mask[BLOCK_SLOT_STONES:MAX_SLOT_STONES],
                target_positions_block1,
                target_mask_block1,
                thrower_block == 1,
                transition_reweight=transition_reweight,
                hybrid_set_weight=hybrid_set_weight,
                use_huber=use_huber,
            )
        )

    target_miss_w0 = _soft_weight_from_margin(_signed_margin(target_positions_block0))
    target_miss_w1 = _soft_weight_from_margin(_signed_margin(target_positions_block1))
    pred_extra_w0 = _soft_weight_from_margin(_signed_margin(pred_block0))
    pred_extra_w1 = _soft_weight_from_margin(_signed_margin(pred_block1))
    target_match_w0 = jnp.ones((target_positions_block0.shape[0],), dtype=target_positions_block0.dtype)
    target_match_w1 = jnp.ones((target_positions_block1.shape[0],), dtype=target_positions_block1.dtype)

    if transition_reweight:
        slot_target_w0, slot_target_miss_w0, slot_pred_extra_w0 = _transition_block_weights(
            prev_positions_button[:BLOCK_SLOT_STONES],
            prev_slot_mask[:BLOCK_SLOT_STONES],
            target_positions_block0,
            target_mask_block0,
            thrower_block == 0,
        )
        slot_target_w1, slot_target_miss_w1, slot_pred_extra_w1 = _transition_block_weights(
            prev_positions_button[BLOCK_SLOT_STONES:MAX_SLOT_STONES],
            prev_slot_mask[BLOCK_SLOT_STONES:MAX_SLOT_STONES],
            target_positions_block1,
            target_mask_block1,
            thrower_block == 1,
        )
        target_match_w0 = slot_target_w0
        target_match_w1 = slot_target_w1
        target_miss_w0 = target_miss_w0 * slot_target_miss_w0
        target_miss_w1 = target_miss_w1 * slot_target_miss_w1
        pred_extra_w0 = pred_extra_w0 * slot_pred_extra_w0
        pred_extra_w1 = pred_extra_w1 * slot_pred_extra_w1

    return (
        _rect_cost_masked(
            pred_block0,
            pred_mask0,
            target_positions_block0,
            target_mask_block0,
            target_match_w0,
            target_miss_w0,
            pred_extra_w0,
            assignment_mode,
        )
        + _rect_cost_masked(
            pred_block1,
            pred_mask1,
            target_positions_block1,
            target_mask_block1,
            target_match_w1,
            target_miss_w1,
            pred_extra_w1,
            assignment_mode,
        )
    )


def build_batched_hard_loss_by_block(p: CurlingParams, *, loss_variant: str = "current"):
    def loss_batch(
        prev_positions_button: jnp.ndarray,
        prev_slot_mask: jnp.ndarray,
        thrower_block: jnp.ndarray,
        target_positions_block0: jnp.ndarray,
        target_mask_block0: jnp.ndarray,
        target_positions_block1: jnp.ndarray,
        target_mask_block1: jnp.ndarray,
        x_phys_batch: jnp.ndarray,
    ) -> jnp.ndarray:
        loss_one = lambda x_phys: _hard_loss_allow_deaths_by_block(
            p,
            prev_positions_button,
            prev_slot_mask,
            thrower_block,
            target_positions_block0,
            target_mask_block0,
            target_positions_block1,
            target_mask_block1,
            x_phys,
            loss_variant=loss_variant,
        )
        return jax.vmap(loss_one)(x_phys_batch)

    return jax.jit(loss_batch)


def _hard_loss_by_block_flex(
    p: CurlingParams,
    prev_positions_button: jnp.ndarray,
    prev_slot_mask: jnp.ndarray,
    thrower_block: jnp.ndarray,
    target_positions_block0: jnp.ndarray,
    target_mask_block0: jnp.ndarray,
    target_positions_block1: jnp.ndarray,
    target_mask_block1: jnp.ndarray,
    x: jnp.ndarray,
    phys: jnp.ndarray,
    *,
    loss_variant: str = "current",
) -> jnp.ndarray:
    """Same as _hard_loss_allow_deaths_by_block but with dynamic physics params."""
    pred_final = simulate_from_params_flex(p, prev_positions_button, x, phys)
    pred_block0 = jnp.concatenate([pred_final[:BLOCK_SLOT_STONES], pred_final[12:13]], axis=0)
    pred_block1 = jnp.concatenate([pred_final[BLOCK_SLOT_STONES:MAX_SLOT_STONES], pred_final[12:13]], axis=0)
    pred_mask0 = jnp.concatenate(
        [prev_slot_mask[:BLOCK_SLOT_STONES], jnp.asarray([thrower_block == 0], dtype=jnp.bool_)],
        axis=0,
    )
    pred_mask1 = jnp.concatenate(
        [prev_slot_mask[BLOCK_SLOT_STONES:MAX_SLOT_STONES], jnp.asarray([thrower_block == 1], dtype=jnp.bool_)],
        axis=0,
    )

    assignment_mode = "greedy"
    if loss_variant == "optimal":
        assignment_mode = "optimal"

    target_miss_w0 = _soft_weight_from_margin(_signed_margin(target_positions_block0))
    target_miss_w1 = _soft_weight_from_margin(_signed_margin(target_positions_block1))
    pred_extra_w0 = _soft_weight_from_margin(_signed_margin(pred_block0))
    pred_extra_w1 = _soft_weight_from_margin(_signed_margin(pred_block1))
    target_match_w0 = jnp.ones((target_positions_block0.shape[0],), dtype=target_positions_block0.dtype)
    target_match_w1 = jnp.ones((target_positions_block1.shape[0],), dtype=target_positions_block1.dtype)

    return (
        _rect_cost_masked(
            pred_block0, pred_mask0,
            target_positions_block0, target_mask_block0,
            target_match_w0, target_miss_w0, pred_extra_w0,
            assignment_mode,
        )
        + _rect_cost_masked(
            pred_block1, pred_mask1,
            target_positions_block1, target_mask_block1,
            target_match_w1, target_miss_w1, pred_extra_w1,
            assignment_mode,
        )
    )


def build_batched_hard_loss_by_block_flex(p: CurlingParams, *, loss_variant: str = "current"):
    """Returns a JIT-compiled loss fn that takes both throw params and physics params."""
    def loss_batch(
        prev_positions_button: jnp.ndarray,
        prev_slot_mask: jnp.ndarray,
        thrower_block: jnp.ndarray,
        target_positions_block0: jnp.ndarray,
        target_mask_block0: jnp.ndarray,
        target_positions_block1: jnp.ndarray,
        target_mask_block1: jnp.ndarray,
        x_phys_batch: jnp.ndarray,
        sim_phys_batch: jnp.ndarray,
    ) -> jnp.ndarray:
        loss_one = lambda x_phys, sim_phys: _hard_loss_by_block_flex(
            p,
            prev_positions_button,
            prev_slot_mask,
            thrower_block,
            target_positions_block0,
            target_mask_block0,
            target_positions_block1,
            target_mask_block1,
            x_phys,
            sim_phys,
            loss_variant=loss_variant,
        )
        return jax.vmap(loss_one)(x_phys_batch, sim_phys_batch)

    return jax.jit(loss_batch)


# Signed margin (meters) to the rectangle edges.
# Positive = in-bounds distance to nearest edge; Negative = overshoot outside.
def _signed_margin(pos: jnp.ndarray) -> jnp.ndarray:
    x = pos[:, 0]; y = pos[:, 1]
    m = jnp.minimum(jnp.minimum(x - MIN_X, MAX_X - x),
                    jnp.minimum(y - MIN_Y, MAX_Y - y))
    return m

# Monotone [0,1) weight: small near edge (margin≈0) → small penalty, grows with margin
def _soft_weight_from_margin(margin: jnp.ndarray, tau: float = DEATH_MARGIN_SCALE) -> jnp.ndarray:
    margin_pos = jnp.maximum(margin, 0.0)  # we only “reward closeness” for items inside
    return 1.0 - jnp.exp(-margin_pos / tau)

# ----------------------------
# Sampling utilities
# ----------------------------

def _sample_diag(key, mean, sigma, n):
    d = mean.shape[0]
    if n <= 0:
        return jnp.empty((0, d))
    eps = random.normal(key, (n, d))
    samp = mean[None, :] + eps * sigma[None, :]
    return jnp.clip(samp, 0.0, 1.0)

def _sample_full(key, mean, cov, n):
    d = mean.shape[0]
    if n <= 0:
        return jnp.empty((0, d))
    cov_np = np.asarray(cov)
    try:
        L = np.linalg.cholesky(cov_np + 1e-10 * np.eye(d))
    except np.linalg.LinAlgError:
        L = np.diag(np.sqrt(np.maximum(np.diag(cov_np), 1e-10)))
    eps = random.normal(key, (n, d))
    samp = mean[None, :] + jnp.asarray(eps @ L.T)
    return jnp.clip(samp, 0.0, 1.0)

# ----------------------------
# CEM Solver
# ----------------------------

def solve_inverse(
    p: CurlingParams,
    prev_positions_button: jnp.ndarray,    # (N_prev,2)
    target_positions_button: jnp.ndarray,  # (N_tgt,2)  <-- may be <= N_prev+1
    bounds: SolveBounds = SolveBounds(),
    *,
    pop_size: int = 96,
    generations: int = 30,
    elite_frac: float = 0.2,
    sigma_init: float = 0.20,
    sigma_floor: float = 0.01,
    ema_alpha: float = 0.7,
    use_full_cov: bool = False,
    mix_with_best_frac: float = 0.35,
    jitter_anchor: float = 0.002,
    key: jax.random.PRNGKey = jax.random.PRNGKey(0),
    init_x: Optional[jnp.ndarray] = None,
    loss_threshold: Optional[float] = 0.1,   # hard-loss early stop
    log_topk: int = 3,
) -> Tuple[jnp.ndarray, float]:
    raise NotImplementedError(
        "Ownership-agnostic inverse solve has been removed. "
        "Use solve_inverse_by_block instead."
    )


def solve_inverse_by_block(
    p: CurlingParams,
    prev_positions_button: jnp.ndarray,
    prev_slot_mask: jnp.ndarray,
    thrower_block: jnp.ndarray,
    target_positions_block0: jnp.ndarray,
    target_mask_block0: jnp.ndarray,
    target_positions_block1: jnp.ndarray,
    target_mask_block1: jnp.ndarray,
    bounds: SolveBounds = SolveBounds(),
    *,
    pop_size: int = 96,
    generations: int = 30,
    elite_frac: float = 0.2,
    sigma_init: float = 0.20,
    sigma_floor: float = 0.01,
    ema_alpha: float = 0.7,
    use_full_cov: bool = False,
    mix_with_best_frac: float = 0.35,
    jitter_anchor: float = 0.002,
    key: jax.random.PRNGKey = jax.random.PRNGKey(0),
    init_x: Optional[jnp.ndarray] = None,
    loss_threshold: Optional[float] = 0.1,
    log_topk: int = 3,
    eval_chunk_size: int = 400,
    batched_hard_fn=None,
) -> Tuple[jnp.ndarray, float]:
    """
    Ownership-aware CEM solve where block 0 and block 1 stones are matched separately.
    """
    log.info(
        "START BLOCK CEM: pop_size=%d, generations=%d, elite_frac=%.2f, sigma_init=%.3f, sigma_floor=%.3f, "
        "ema_alpha=%.2f, full_cov=%s, mix_with_best_frac=%.2f, jitter_anchor=%.4f, loss_threshold=%s",
        pop_size, generations, elite_frac, sigma_init, sigma_floor, ema_alpha,
        str(use_full_cov), mix_with_best_frac, jitter_anchor,
        ("None" if loss_threshold is None else f"{loss_threshold}")
    )

    lo, hi, span = _make_bounds_arrays(bounds)
    d = int(lo.shape[0])
    assert pop_size >= 4 and 0.0 < elite_frac < 1.0
    eval_chunk_size = max(1, min(int(eval_chunk_size), int(pop_size)))

    if init_x is not None:
        mean = _x01_from_x_phys(init_x, lo, span)
    else:
        mean = jnp.ones((d,)) * 0.5
    if use_full_cov:
        cov = jnp.eye(d) * (sigma_init ** 2)
    else:
        sigma = jnp.ones((d,)) * sigma_init

    if batched_hard_fn is None:
        batched_hard_fn = build_batched_hard_loss_by_block(p)

    def batched_hard(x_phys_batch: jnp.ndarray) -> jnp.ndarray:
        return batched_hard_fn(
            prev_positions_button,
            prev_slot_mask,
            thrower_block,
            target_positions_block0,
            target_mask_block0,
            target_positions_block1,
            target_mask_block1,
            x_phys_batch,
        )

    best_x01 = mean
    best_phys = _x_phys_from_x01(best_x01, lo, span)
    best_hard = float(batched_hard(best_phys[None, :])[0])
    log.info("INIT BLOCK | mean_x01=%s | best_hard=%.6f", list(map(float, mean)), best_hard)
    if (loss_threshold is not None) and (best_hard <= float(loss_threshold)):
        log.info("BLOCK EARLY-STOP at init | best_hard=%.6f <= %.6f", best_hard, float(loss_threshold))
        return best_phys, best_hard

    elites_k = max(1, int(round(pop_size * elite_frac)))
    key_loop = key

    for gen in range(generations):
        n_anchor = int(round(pop_size * mix_with_best_frac))
        n_model = pop_size - n_anchor - 1
        n_model = max(0, n_model)
        n_anchor = max(0, n_anchor)

        key_loop, k_model, k_anchor = random.split(key_loop, 3)
        if use_full_cov:
            cand_model = _sample_full(k_model, mean, cov, n_model)
            anchor_sigma = jnp.ones((d,)) * max(sigma_floor, jitter_anchor)
            cand_anchor = _sample_diag(k_anchor, best_x01, anchor_sigma, n_anchor)
        else:
            cand_model = _sample_diag(k_model, mean, sigma, n_model)
            anchor_sigma = jnp.maximum(sigma * 0.5, jnp.ones((d,)) * jitter_anchor)
            cand_anchor = _sample_diag(k_anchor, best_x01, anchor_sigma, n_anchor)

        anchor_exact = best_x01[None, :]
        # Evaluate the current anchor first so thresholded early-exit can cut
        # off the rest of the generation on already-solved shots.
        X01 = jnp.concatenate([anchor_exact, cand_anchor, cand_model], axis=0)
        X_phys = _x_phys_from_x01(X01, lo, span)

        losses_np = np.full((pop_size,), np.inf, dtype=np.float32)
        evaluated_n = 0
        stop_early = False
        for start in range(0, pop_size, eval_chunk_size):
            end = min(pop_size, start + eval_chunk_size)
            losses_chunk = _safe_metric(batched_hard(X_phys[start:end]))
            losses_chunk_np = np.asarray(losses_chunk)
            losses_np[start:end] = losses_chunk_np
            evaluated_n = end

            i_local = int(np.argmin(losses_chunk_np))
            chunk_best = float(losses_chunk_np[i_local])
            if chunk_best < best_hard:
                i0 = start + i_local
                best_hard = chunk_best
                best_x01 = X01[i0]
                best_phys = X_phys[i0]
                log.info(
                    "  NEW BLOCK BEST | gen=%d | cand=%d | hard=%.6f | x=%s",
                    gen,
                    i0,
                    best_hard,
                    list(map(float, best_phys)),
                )

            if (loss_threshold is not None) and (best_hard <= float(loss_threshold)):
                stop_early = True
                break

        order_eval = np.argsort(losses_np[:evaluated_n])
        topk = order_eval[:min(log_topk, evaluated_n)]
        log.info(
            "BLOCK GEN %03d | eval=%d/%d | best=%.6f | top%d: %s",
            gen,
            evaluated_n,
            pop_size,
            float(np.min(losses_np[:evaluated_n])),
            len(topk),
            ", ".join([f"{int(i)}:{losses_np[int(i)]:.6f}" for i in topk]),
        )

        if stop_early:
            log.info("BLOCK EARLY-STOP at gen=%d | best_hard=%.6f <= %.6f", gen, best_hard, float(loss_threshold))
            break

        order = np.argsort(losses_np)
        elites_idx = order[:elites_k]
        E = X01[elites_idx, :]
        elite_mean = jnp.mean(E, axis=0)

        if use_full_cov:
            E0 = E - elite_mean[None, :]
            cov_elite = (E0.T @ E0) / max(1, E.shape[0] - 1)
            mean = (1.0 - ema_alpha) * mean + ema_alpha * elite_mean
            cov = (1.0 - ema_alpha) * cov + ema_alpha * cov_elite
            diag = jnp.maximum(jnp.diag(cov), (sigma_floor ** 2))
            cov = cov - jnp.diag(jnp.diag(cov)) + jnp.diag(diag)
        else:
            elite_std = jnp.std(E, axis=0)
            mean = (1.0 - ema_alpha) * mean + ema_alpha * elite_mean
            sigma = (1.0 - ema_alpha) * sigma + ema_alpha * elite_std
            sigma = jnp.maximum(sigma, jnp.ones_like(sigma) * sigma_floor)

        mean = 0.9 * mean + 0.1 * best_x01

    log.info("FINISH BLOCK CEM | best_hard=%.6f | x_best=%s", best_hard, list(map(float, best_phys)))
    return best_phys, best_hard

# ----------------------------
# Small wrapper for convenience
# ----------------------------

def solve_and_simulate(
    p: CurlingParams,
    prev_positions_button: jnp.ndarray,
    target_positions_button: jnp.ndarray,
    **kwargs
):
    raise NotImplementedError(
        "Ownership-agnostic inverse solve has been removed. "
        "Use solve_and_simulate_by_block instead."
    )


def solve_and_simulate_by_block(
    p: CurlingParams,
    prev_positions_button: jnp.ndarray,
    prev_slot_mask: jnp.ndarray,
    thrower_block: jnp.ndarray,
    target_positions_block0: jnp.ndarray,
    target_mask_block0: jnp.ndarray,
    target_positions_block1: jnp.ndarray,
    target_mask_block1: jnp.ndarray,
    **kwargs
):
    x_phys, hard_loss = solve_inverse_by_block(
        p,
        prev_positions_button,
        prev_slot_mask,
        thrower_block,
        target_positions_block0,
        target_mask_block0,
        target_positions_block1,
        target_mask_block1,
        **kwargs,
    )
    pred_final = simulate_from_params(p, prev_positions_button, x_phys, dynamic=False)
    return x_phys, hard_loss, pred_final
