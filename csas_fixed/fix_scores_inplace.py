#!/usr/bin/env python3
"""
Fix shot_scores CSVs in-place:
1. Recompute v_next with 3-element conditioning (no is_hammer)
2. Compute v_prev via zero-sum sign-flip of previous shot's v_next
3. Update dv_obs, dv_mean, perspective columns
4. Remove is_hammer column
"""

from __future__ import annotations

import argparse
import math
import pathlib
import sys
import glob as globmod

import numpy as np
import pandas as pd

THIS_DIR = pathlib.Path(__file__).resolve().parent
sys.path.append(str(THIS_DIR / "valueModel"))
sys.path.append(str(THIS_DIR))

from score_shots_mc_seq import (
    load_value_model,
    compact_positions,
    normalize_raw_matrix,
    positions_m_to_raw_matrix,
    make_raw_defaults_for_state,
    infer_thrower_block,
    _coerce_context_dim,
    SHOT_KEY,
)

END_KEY = ["CompetitionID", "SessionID", "GameID", "EndID"]


def fix_one_csv(csv_path, inv_df, model_fn, cond_dim):
    df = pd.read_csv(csv_path)
    if len(df) == 0:
        return

    print(f"  {csv_path.name}: {len(df)} rows", flush=True)

    # Save old v_prev for dv_mean adjustment
    v_prev_old = df["v_prev"].astype(float).values.copy()

    # Sort by end+shot for sequential processing
    df = df.sort_values(SHOT_KEY).reset_index(drop=True)

    # Add ShotID_prev within each end
    df["ShotID_prev"] = df.groupby(END_KEY)["ShotID"].shift(1)

    # Compute ShotIndex within end
    df["ShotIndex"] = df.groupby(END_KEY).cumcount()

    # Merge inverse data for stone positions
    inv_stone_cols = [c for c in inv_df.columns if "stone_" in c and "_m" in c]
    inv_meta = [c for c in ["obs_throw_slot_id", "team_slot_block"] if c in inv_df.columns]
    inv_keep = SHOT_KEY + inv_stone_cols + inv_meta
    inv_keep = list(dict.fromkeys(inv_keep))
    inv_keep = [c for c in inv_keep if c in inv_df.columns]

    merged = df.merge(inv_df[inv_keep], on=SHOT_KEY, how="left")

    new_v_prev = np.full(len(merged), np.nan, dtype=np.float64)
    new_v_next = np.full(len(merged), np.nan, dtype=np.float64)

    v_next_cache = {}
    prev_end_key = None

    for idx in range(len(merged)):
        row = merged.iloc[idx]

        # Clear cache at end boundaries
        ek = tuple(row[c] for c in END_KEY)
        if ek != prev_end_key:
            v_next_cache.clear()
            prev_end_key = ek

        # Extract stone positions
        prev_mat = np.full((12, 2), np.nan, dtype=np.float32)
        next_mat = np.full((12, 2), np.nan, dtype=np.float32)
        for i in range(1, 13):
            for prefix, mat in [("prev", prev_mat), ("next", next_mat)]:
                xc = f"{prefix}_stone_{i}_x_m"
                yc = f"{prefix}_stone_{i}_y_m"
                if xc in row.index and pd.notna(row[xc]):
                    mat[i-1] = [float(row[xc]), float(row[yc])]

        _, prev_ids = compact_positions(prev_mat)
        _, next_ids = compact_positions(next_mat)

        obs_slot = float(row.get("obs_throw_slot_id", np.nan)) if "obs_throw_slot_id" in row.index else np.nan
        team_slot = float(row.get("team_slot_block", np.nan)) if "team_slot_block" in row.index else np.nan
        thrower_block = infer_thrower_block(prev_ids=prev_ids, next_ids=next_ids,
                                            obs_throw_slot_id=obs_slot, team_slot_block=team_slot)

        shot_norm_next = float(row.get("shot_norm_next", 0.0))
        shot_norm_prev = float(row.get("shot_norm_prev", 0.0))
        team_order = float(row.get("team_order", 0.0))
        stone_block = float(thrower_block)
        shot_index = float(row.get("ShotIndex", 0.0))

        # v_next with 3-element conditioning
        c_next = np.array([shot_norm_next, team_order, stone_block], dtype=np.float32)
        next_defaults = make_raw_defaults_for_state(shot_index, team_order, thrower_block)
        next_raw = normalize_raw_matrix(positions_m_to_raw_matrix(next_mat, raw_defaults=next_defaults))
        v_next = float(model_fn(next_raw, c_next))
        new_v_next[idx] = v_next

        # v_prev via zero-sum sign-flip
        sid_prev = row.get("ShotID_prev", np.nan)
        if pd.notna(sid_prev) and int(sid_prev) in v_next_cache:
            v_prev = -v_next_cache[int(sid_prev)]
        else:
            # No predecessor v_next to sign-flip from (first scored shot in end).
            # Keep original v_prev from the scoring run (imperfect but self-consistent).
            c_prev = np.array([shot_norm_prev, team_order, stone_block], dtype=np.float32)
            prev_defaults = make_raw_defaults_for_state(shot_index - 1.0, team_order, thrower_block)
            prev_raw = normalize_raw_matrix(positions_m_to_raw_matrix(prev_mat, raw_defaults=prev_defaults))
            v_prev = float(model_fn(prev_raw, c_prev))
        new_v_prev[idx] = v_prev

        if pd.notna(row.get("ShotID")):
            v_next_cache[int(row["ShotID"])] = v_next

    # Apply updates
    df["v_prev"] = new_v_prev
    df["v_next"] = new_v_next
    df["dv_obs"] = new_v_next - new_v_prev

    # Adjust dv_mean by v_prev delta
    v_prev_delta = v_prev_old - new_v_prev
    for col in ["dv_mean", "dv_p10", "dv_p50", "dv_p90", "cvar_10"]:
        if col in df.columns:
            df[col] = df[col].astype(float) + v_prev_delta

    # Recompute z_obs
    if "dv_mean" in df.columns and "dv_std" in df.columns:
        dv_obs = df["dv_obs"].astype(float).values
        dv_mean = df["dv_mean"].astype(float).values
        dv_std = df["dv_std"].astype(float).values
        with np.errstate(divide="ignore", invalid="ignore"):
            df["z_obs"] = np.where(np.isfinite(dv_std) & (dv_std > 1e-8),
                                   (dv_obs - dv_mean) / dv_std, np.nan)

    # Update perspective columns
    for src, dst in [("dv_obs", "dv_obs_thrower"), ("dv_mean", "dv_mean_thrower")]:
        if dst in df.columns:
            df[dst] = df[src]
    for src, dst in [("dv_obs", "dv_obs_opponent"), ("dv_mean", "dv_mean_opponent")]:
        if dst in df.columns:
            df[dst] = -df[src].astype(float)

    team_order_vals = df["team_order"].astype(float).values
    for base in ["dv_obs", "dv_mean"]:
        for to_val, suffix in [(0.0, "_team_order0"), (1.0, "_team_order1")]:
            col = f"{base}{suffix}"
            if col in df.columns:
                sign = np.where(team_order_vals == to_val, 1.0, -1.0)
                df[col] = df[base].astype(float).values * sign

    # Remove is_hammer and temp columns
    drop_cols = [c for c in ["is_hammer", "ShotID_prev", "ShotIndex"] if c in df.columns]
    df = df.drop(columns=drop_cols)

    df.to_csv(csv_path, index=False)

    dv_new = new_v_next - new_v_prev
    dv_old = df["v_next"].astype(float).values - v_prev_old  # approximate
    print(f"    Mean |v_prev delta|={np.nanmean(np.abs(v_prev_delta)):.4f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", default=str(THIS_DIR))
    ap.add_argument("--inverse-glob", default="inverse_current/stones_with_estimates.chunk*.csv")
    ap.add_argument("--holdout-ids", nargs="*", default=["0", "22230015", "23240026", "24250026"])
    args = ap.parse_args()

    base_dir = pathlib.Path(args.base_dir)

    # Load inverse data once
    inv_paths = sorted(globmod.glob(str(base_dir / args.inverse_glob)))
    if not inv_paths:
        print("ERROR: no inverse files found", flush=True)
        return
    print(f"Loading {len(inv_paths)} inverse chunks...", flush=True)
    inv_df = pd.concat([pd.read_csv(p) for p in inv_paths], ignore_index=True)
    print(f"  {len(inv_df)} inverse rows loaded", flush=True)

    for hid in args.holdout_ids:
        holdout_dir = base_dir / "holdouts" / str(hid)
        model_path = holdout_dir / "model" / "model.pt"
        if not model_path.exists():
            print(f"SKIP holdout {hid}: no model at {model_path}", flush=True)
            continue

        print(f"\n=== Holdout {hid} ===", flush=True)
        print(f"Loading model: {model_path}", flush=True)
        model_fn, cond_dim = load_value_model(model_path)
        print(f"  cond_dim={cond_dim}", flush=True)

        for name in ["shot_scores_local.csv", "shot_scores_global.csv"]:
            csv_path = holdout_dir / "scoring" / name
            if csv_path.exists():
                fix_one_csv(csv_path, inv_df, model_fn, cond_dim)

    print("\nAll done!", flush=True)


if __name__ == "__main__":
    main()
