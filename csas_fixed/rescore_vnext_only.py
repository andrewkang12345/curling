#!/usr/bin/env python3
"""
Re-evaluate ONLY v_next with the new SetTransformer model for each holdout.
Uses sign-flip for v_prev. Adjusts dv_mean by v_prev delta AND v_next delta.

Since execution_value = v_next - mean(v_sim), and we can't re-evaluate v_sim
(simulated boards aren't saved), we adjust:
  execution_value_new = v_next_new - mean(v_sim_old)
                      = v_next_new - (dv_mean_old + v_prev_old)

This gives self-consistent execution_value with the new model's v_next evaluation,
while acknowledging v_sim is from the old model.
"""

import sys
import pathlib
import glob as globmod
import numpy as np
import pandas as pd

THIS_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR / "valueModel"))
sys.path.insert(0, str(THIS_DIR / "valueModel" / "ablation"))

from score_shots_mc_seq import (
    load_value_model, compact_positions, normalize_raw_matrix,
    positions_m_to_raw_matrix, make_raw_defaults_for_state,
    infer_thrower_block, SHOT_KEY,
)

END_KEY = ["CompetitionID", "SessionID", "GameID", "EndID"]

HOLDOUTS = {"0": 0, "22230015": 22230015, "23240026": 23240026, "24250026": 24250026}


def rescore_csv(orig_path, out_path, inv_df, model_fn, cond_dim):
    df = pd.read_csv(orig_path)
    df = df.sort_values(SHOT_KEY).reset_index(drop=True)
    n = len(df)
    print(f"  {orig_path.name}: {n} rows", flush=True)

    # Save originals
    v_next_old = df["v_next"].astype(float).values.copy()
    v_prev_old = df["v_prev"].astype(float).values.copy()
    dv_mean_old = df["dv_mean"].astype(float).values.copy()

    # Merge with inverse data for stone positions
    inv_cols = SHOT_KEY + [c for c in inv_df.columns if "stone_" in c and "_m" in c]
    inv_cols += [c for c in ["obs_throw_slot_id", "team_slot_block"] if c in inv_df.columns]
    inv_cols = list(dict.fromkeys([c for c in inv_cols if c in inv_df.columns]))
    merged = df.merge(inv_df[inv_cols], on=SHOT_KEY, how="left")

    # Re-evaluate v_next with new model
    v_next_new = np.full(n, np.nan, dtype=np.float64)

    for idx in range(n):
        row = merged.iloc[idx]
        next_mat = np.full((12, 2), np.nan, dtype=np.float32)
        for i in range(1, 13):
            nx, ny = row.get(f"next_stone_{i}_x_m", np.nan), row.get(f"next_stone_{i}_y_m", np.nan)
            if pd.notna(nx):
                next_mat[i-1] = [float(nx), float(ny)]

        _, next_ids = compact_positions(next_mat)
        prev_mat = np.full((12, 2), np.nan, dtype=np.float32)
        for i in range(1, 13):
            px, py = row.get(f"prev_stone_{i}_x_m", np.nan), row.get(f"prev_stone_{i}_y_m", np.nan)
            if pd.notna(px):
                prev_mat[i-1] = [float(px), float(py)]
        _, prev_ids = compact_positions(prev_mat)

        obs_slot = float(row.get("obs_throw_slot_id", np.nan)) if "obs_throw_slot_id" in row.index else np.nan
        team_slot = float(row.get("team_slot_block", np.nan)) if "team_slot_block" in row.index else np.nan
        thrower_block = infer_thrower_block(prev_ids=prev_ids, next_ids=next_ids,
                                            obs_throw_slot_id=obs_slot, team_slot_block=team_slot)

        shot_norm_next = float(row.get("shot_norm_next", 0.0))
        team_order = float(row.get("team_order", 0.0))
        stone_block = float(thrower_block)
        shot_index = float(row.get("ShotIndex", 0.0)) if pd.notna(row.get("ShotIndex")) else 0.0

        c_next = np.array([shot_norm_next, team_order, stone_block], dtype=np.float32)
        next_defaults = make_raw_defaults_for_state(shot_index, team_order, thrower_block)
        next_raw = normalize_raw_matrix(positions_m_to_raw_matrix(next_mat, raw_defaults=next_defaults))
        v_next_new[idx] = float(model_fn(next_raw, c_next))

    # v_prev: sign-flip of previous shot's new v_next; keep original for first-in-end
    df["_prev_vnext_new"] = pd.Series(v_next_new).groupby(df[END_KEY].apply(tuple, axis=1)).shift(1).values
    has_prev = np.isfinite(df["_prev_vnext_new"].values)
    v_prev_new = np.where(has_prev, -df["_prev_vnext_new"].values, v_prev_old)

    # Update columns
    df["v_next"] = v_next_new
    df["v_prev"] = v_prev_new
    df["dv_obs"] = v_next_new - v_prev_new

    # Adjust dv_mean: dv_mean_old = mean(v_sim_old) - v_prev_old
    # We want: dv_mean_new = mean(v_sim_old) - v_prev_new = dv_mean_old + (v_prev_old - v_prev_new)
    v_prev_delta = v_prev_old - v_prev_new
    df["dv_mean"] = dv_mean_old + v_prev_delta
    for col in ["dv_p10", "dv_p50", "dv_p90", "cvar_10"]:
        if col in df.columns:
            df[col] = df[col].astype(float) + v_prev_delta

    # Recompute z_obs
    if "dv_std" in df.columns:
        dv_obs = df["dv_obs"].astype(float).values
        dv_mean = df["dv_mean"].astype(float).values
        dv_std = df["dv_std"].astype(float).values
        with np.errstate(divide="ignore", invalid="ignore"):
            df["z_obs"] = np.where(np.isfinite(dv_std) & (dv_std > 1e-8),
                                   (dv_obs - dv_mean) / dv_std, np.nan)

    # Perspective columns
    team_order_arr = df["team_order"].astype(float).values if "team_order" in df.columns else np.zeros(n)
    for src, dst in [("dv_obs", "dv_obs_thrower"), ("dv_mean", "dv_mean_thrower")]:
        if dst in df.columns: df[dst] = df[src]
    for src, dst in [("dv_obs", "dv_obs_opponent"), ("dv_mean", "dv_mean_opponent")]:
        if dst in df.columns: df[dst] = -df[src].astype(float)
    for base in ["dv_obs", "dv_mean"]:
        for to_val, suffix in [(0.0, "_team_order0"), (1.0, "_team_order1")]:
            col = f"{base}{suffix}"
            if col in df.columns:
                df[col] = df[base].astype(float).values * np.where(team_order_arr == to_val, 1.0, -1.0)

    if "is_hammer" in df.columns:
        df = df.drop(columns=["is_hammer"])
    df = df.drop(columns=["_prev_vnext_new"], errors="ignore")

    df.to_csv(out_path, index=False)
    v_next_delta = v_next_new - v_next_old
    print(f"    v_next delta: mean={np.nanmean(v_next_delta):.4f}, std={np.nanstd(v_next_delta):.4f}", flush=True)


def main():
    # Load inverse data
    inv_paths = sorted(globmod.glob(str(THIS_DIR / "inverse_current" / "stones_with_estimates.chunk*.csv")))
    print(f"Loading {len(inv_paths)} inverse chunks...", flush=True)
    inv_df = pd.concat([pd.read_csv(p) for p in inv_paths], ignore_index=True)
    print(f"  {len(inv_df)} rows", flush=True)

    for hid, comp_id in HOLDOUTS.items():
        print(f"\n=== Holdout {hid} (comp {comp_id}) ===", flush=True)
        model_path = THIS_DIR / "holdouts" / hid / "model" / "model.pt"
        print(f"Loading model: {model_path}", flush=True)
        model_fn, cond_dim = load_value_model(model_path, device="cpu")
        print(f"  cond_dim={cond_dim}", flush=True)

        scoring_dir = THIS_DIR / "holdouts" / hid / "scoring"
        for mode in ["local", "global"]:
            orig = scoring_dir / mode / f"shot_scores_comp_{comp_id}.csv"
            out = scoring_dir / f"shot_scores_{mode}.csv"
            if orig.exists():
                rescore_csv(orig, out, inv_df, model_fn, cond_dim)

    print("\nDone!", flush=True)


if __name__ == "__main__":
    main()
