#!/usr/bin/env python3
"""
Inverse-solve the first thrown shot of each end:
  prev = canonical pre-placed stones (2 stones)
  next = observed post-shot-7 state from Stones.csv

Outputs stones_with_estimates for ShotID=7 rows, compatible with
the existing inverse pipeline output format.
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

os.environ["JAX_PLATFORMS"] = "cpu"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "inverse"))
sys.path.insert(0, BASE_DIR)

import jax
import jax.numpy as jnp
from curling_inverse import (
    CurlingParams,
    SolveBounds,
    solve_inverse_by_block,
    build_batched_hard_loss_by_block,
)
from sim_presets import contact_mild_params

STONES_CSV = os.path.join(BASE_DIR, "2026", "Stones.csv")
ENDS_CSV = os.path.join(BASE_DIR, "2026", "Ends.csv")
OUT_DIR = os.path.join(BASE_DIR, "preplaced_augment")

# Canonical pre-placed positions (meters, tee = origin)
CANON = {
    "standard":  {"guard": (-3.4016, 0.0),    "inhouse": ( 0.4572, 0.0)},
    "pp_right":  {"guard": (-3.4016, 1.0333), "inhouse": (-0.1524, 1.2192)},
    "pp_left":   {"guard": (-3.4016,-1.0333), "inhouse": (-0.1524,-1.2192)},
}

PAD_POS = np.array([50.0, 50.0], dtype=np.float32)
GUARD_X_THRESHOLD = -2.0
BLOCK_STONES = 6  # slots 1-6 = block 0, slots 7-12 = block 1


def csv_to_meters(x_csv, y_csv):
    return (800.0 - y_csv) * 0.003048, (x_csv - 750.0) * 0.003048


def build_prev_slots(mode, guard_slot):
    """Build 12-slot prev array from canonical pre-placed positions."""
    slots = np.tile(PAD_POS, (12, 1))
    mask = np.zeros(12, dtype=bool)
    canon = CANON[mode]
    gx, gy = canon["guard"]
    ix, iy = canon["inhouse"]
    inhouse_slot = 7 if guard_slot == 1 else 1
    slots[guard_slot - 1] = [gx, gy]
    mask[guard_slot - 1] = True
    slots[inhouse_slot - 1] = [ix, iy]
    mask[inhouse_slot - 1] = True
    return slots.astype(np.float32), mask


def build_target_blocks(next_state):
    """
    Build target block arrays from next_state dict {slot_id: (x,y)}.
    Block 0 = slots 1-6 + thrown stone (if block 0 thrower)
    Block 1 = slots 7-12 + thrown stone (if block 1 thrower)
    We include a 7th slot for the thrown stone in each block.
    """
    # Block 0: slots 1-6, Block 1: slots 7-12
    # +1 slot each for the newly thrown stone (slot 13 = thrown)
    b0 = np.tile(PAD_POS, (BLOCK_STONES + 1, 1))
    b0_mask = np.zeros(BLOCK_STONES + 1, dtype=bool)
    b1 = np.tile(PAD_POS, (BLOCK_STONES + 1, 1))
    b1_mask = np.zeros(BLOCK_STONES + 1, dtype=bool)

    for sid, (xm, ym) in next_state.items():
        if 1 <= sid <= 6:
            b0[sid - 1] = [xm, ym]
            b0_mask[sid - 1] = True
        elif 7 <= sid <= 12:
            b1[sid - 7] = [xm, ym]
            b1_mask[sid - 7] = True
    return b0.astype(np.float32), b0_mask, b1.astype(np.float32), b1_mask


def main():
    print("Loading data...")
    stones = pd.read_csv(STONES_CSV)
    ends = pd.read_csv(ENDS_CSV)
    EK = ["CompetitionID", "SessionID", "GameID", "EndID"]

    # Load classification to get mode + guard_slot
    class_df = pd.read_csv(os.path.join(OUT_DIR, "preplaced_classification.csv"))
    class_map = {}
    for _, r in class_df.iterrows():
        key = (r["CompetitionID"], r["SessionID"], r["GameID"], r["EndID"])
        class_map[key] = {"mode": r["mode"], "guard_slot": int(r["guard_slot"]) if pd.notna(r["guard_slot"]) else None}

    # Get post-shot-7 states from Stones.csv
    s7 = stones[stones["ShotID"] == 7].copy()

    # Physics params
    p_coarse = contact_mild_params(CurlingParams, dt=0.02, substeps=2, k_penalty=2.5e4)
    p_refine = contact_mild_params(CurlingParams, dt=0.02, substeps=2, k_penalty=2.5e4)
    bounds = SolveBounds()
    batched_hard_fn = build_batched_hard_loss_by_block(p_refine)

    results = []
    n_solved = 0
    n_skipped = 0

    for _, row in tqdm(s7.iterrows(), total=len(s7), desc="Solving first shots"):
        key = (row["CompetitionID"], row["SessionID"], row["GameID"], row["EndID"])
        info = class_map.get(key)
        if info is None or info["mode"] == "offsheet" or info["guard_slot"] is None:
            n_skipped += 1
            continue

        mode = info["mode"]
        guard_slot = info["guard_slot"]

        # Build prev state (canonical pre-placed)
        prev_slots, prev_slot_mask = build_prev_slots(mode, guard_slot)

        # Build next state (post-shot-7 from Stones.csv)
        next_state = {}
        for sid in range(1, 13):
            xc = row.get(f"stone_{sid}_x", 0)
            yc = row.get(f"stone_{sid}_y", 0)
            if pd.notna(xc) and pd.notna(yc) and 0 < xc < 4000 and 0 < yc < 4000:
                xm, ym = csv_to_meters(xc, yc)
                next_state[sid] = (xm, ym)

        if len(next_state) < 2:
            n_skipped += 1
            continue

        # Infer thrower block: which block gained a stone?
        prev_ids = set()
        if guard_slot == 1:
            prev_ids = {1, 7 if guard_slot == 1 else 1}
        inhouse_slot = 7 if guard_slot == 1 else 1
        prev_ids = {guard_slot, inhouse_slot}
        new_ids = set(next_state.keys()) - prev_ids
        if new_ids:
            new_id = min(new_ids)
            thrower_block = 0 if new_id <= 6 else 1
        else:
            # No new stone visible (went off sheet) — guess from TeamID
            thrower_block = 0  # fallback

        # Build target blocks
        b0, b0_mask, b1, b1_mask = build_target_blocks(next_state)

        # Solve
        prev_j = jnp.asarray(prev_slots, dtype=jnp.float32)
        prev_mask_j = jnp.asarray(prev_slot_mask, dtype=jnp.bool_)
        thrower_j = jnp.asarray(thrower_block, dtype=jnp.int32)
        b0_j = jnp.asarray(b0, dtype=jnp.float32)
        b0_mask_j = jnp.asarray(b0_mask, dtype=jnp.bool_)
        b1_j = jnp.asarray(b1, dtype=jnp.float32)
        b1_mask_j = jnp.asarray(b1_mask, dtype=jnp.bool_)

        try:
            x_best, hard_loss = solve_inverse_by_block(
                p_refine,
                prev_j, prev_mask_j, thrower_j,
                b0_j, b0_mask_j, b1_j, b1_mask_j,
                bounds,
                pop_size=96,
                generations=30,
                loss_threshold=0.1,
                batched_hard_fn=batched_hard_fn,
                key=jax.random.PRNGKey(int(row["EndID"]) * 100 + int(row.get("GameID", 0))),
            )
            x_np = np.asarray(x_best)
            hard_np = float(hard_loss)
            solver_ok = hard_np < 0.5
        except Exception as e:
            x_np = np.full(4, np.nan)
            hard_np = np.nan
            solver_ok = False

        out = {
            "CompetitionID": row["CompetitionID"],
            "SessionID": row["SessionID"],
            "GameID": row["GameID"],
            "EndID": row["EndID"],
            "ShotID": 7,
            "prev_N": 2,
            "next_total_N": len(next_state),
            "next_in_bounds_N": len(next_state),
            "est_speed": float(x_np[0]),
            "est_angle": float(x_np[1]),
            "est_spin": float(x_np[2]),
            "est_y0": float(x_np[3]),
            "hard_loss_coarse": np.nan,
            "hard_loss_refine": hard_np,
            "solver_ok": solver_ok,
            "solver_method": "preplaced_cem",
            "loss_variant": "current",
            "thrower_block": thrower_block,
            "mode": mode,
            "guard_slot": guard_slot,
        }
        # Add prev/next stone positions
        for sid in range(1, 13):
            out[f"prev_stone_{sid}_x_m"] = prev_slots[sid-1, 0] if prev_slot_mask[sid-1] else np.nan
            out[f"prev_stone_{sid}_y_m"] = prev_slots[sid-1, 1] if prev_slot_mask[sid-1] else np.nan
            if sid in next_state:
                out[f"next_stone_{sid}_x_m"] = next_state[sid][0]
                out[f"next_stone_{sid}_y_m"] = next_state[sid][1]
                out[f"next_stone_{sid}_inbounds"] = 1
            else:
                out[f"next_stone_{sid}_x_m"] = np.nan
                out[f"next_stone_{sid}_y_m"] = np.nan
                out[f"next_stone_{sid}_inbounds"] = 0

        results.append(out)
        n_solved += 1

    df_out = pd.DataFrame(results)
    out_path = os.path.join(OUT_DIR, "first_shots_inverse.csv")
    df_out.to_csv(out_path, index=False)

    n_ok = df_out["solver_ok"].sum() if len(df_out) else 0
    mean_loss = df_out["hard_loss_refine"].mean() if len(df_out) else np.nan
    med_loss = df_out["hard_loss_refine"].median() if len(df_out) else np.nan

    print(f"\nDone: {n_solved} solved, {n_skipped} skipped")
    print(f"solver_ok: {n_ok}/{n_solved} ({100*n_ok/max(1,n_solved):.1f}%)")
    print(f"hard_loss: mean={mean_loss:.4f}, median={med_loss:.4f}")
    print(f"Saved: {out_path}")

    # Also update augmented_throws.csv with the solved parameters
    aug_path = os.path.join(OUT_DIR, "augmented_throws.csv")
    if os.path.exists(aug_path) and len(df_out) > 0:
        aug = pd.read_csv(aug_path)
        merge_cols = ["CompetitionID", "SessionID", "GameID", "EndID", "ShotID"]
        update_cols = ["est_speed", "est_angle", "est_spin", "est_y0",
                       "hard_loss_refine", "solver_ok", "solver_method"]
        inv_sub = df_out[merge_cols + update_cols].copy()
        inv_sub = inv_sub.rename(columns={c: f"_inv_{c}" for c in update_cols})

        aug = pd.merge(aug, inv_sub, on=merge_cols, how="left")
        for c in update_cols:
            mask = aug[f"_inv_{c}"].notna()
            aug.loc[mask, c] = aug.loc[mask, f"_inv_{c}"]
            aug.drop(columns=[f"_inv_{c}"], inplace=True)

        aug.to_csv(aug_path, index=False)
        n_updated = mask.sum()
        print(f"Updated {n_updated} rows in {aug_path}")


if __name__ == "__main__":
    main()
