#!/usr/bin/env python3
"""
Fix shot_scores using ONLY sign-flip for v_prev. No model re-evaluation.

Restores original v_next and v_sim-derived columns from the untouched
per-competition CSVs, then applies:
  v_prev = -v_next of previous shot (zero-sum sign-flip)
  dv_obs = original_v_next - sign_flipped_v_prev
  First-in-end shots (no predecessor) → NaN

This keeps everything self-consistent with the original scoring model.
"""

import pathlib
import math
import numpy as np
import pandas as pd

SHOT_KEY = ["CompetitionID", "SessionID", "GameID", "EndID", "ShotID"]
END_KEY = ["CompetitionID", "SessionID", "GameID", "EndID"]
BASE = pathlib.Path("/mnt/data/curling2/csas_fixed")

HOLDOUTS = {
    "0": 0,
    "22230015": 22230015,
    "23240026": 23240026,
    "24250026": 24250026,
}

# Columns derived from v_prev that need recalculation
VPREV_DEPENDENT = [
    "v_prev", "dv_obs",
    "dv_obs_thrower", "dv_obs_opponent",
    "dv_obs_team_order0", "dv_obs_team_order1",
]

# dv_mean and sample-derived columns need v_prev adjustment
DVMEAN_COLS = ["dv_mean", "dv_p10", "dv_p50", "dv_p90", "cvar_10",
               "dv_mean_thrower", "dv_mean_opponent",
               "dv_mean_team_order0", "dv_mean_team_order1"]


def fix_csv(orig_path, out_path):
    """Fix a single shot_scores CSV using sign-flip only."""
    df = pd.read_csv(orig_path)
    df = df.sort_values(SHOT_KEY).reset_index(drop=True)
    n = len(df)
    print(f"  {orig_path.name} → {out_path.name}: {n} rows", flush=True)

    # Keep original v_next (self-consistent with v_sim from same scoring run)
    v_next_orig = df["v_next"].astype(float).values.copy()
    v_prev_orig = df["v_prev"].astype(float).values.copy()
    dv_mean_orig = df["dv_mean"].astype(float).values.copy() if "dv_mean" in df.columns else None

    # Compute sign-flipped v_prev: -v_next of previous shot in same end
    df["_prev_v_next"] = df.groupby(END_KEY)["v_next"].shift(1)
    is_first = df["_prev_v_next"].isna()

    v_prev_new = np.where(is_first, np.nan, -df["_prev_v_next"].astype(float).values)
    dv_obs_new = v_next_orig - v_prev_new  # NaN for first-in-end

    df["v_prev"] = v_prev_new
    df["dv_obs"] = dv_obs_new

    # Adjust dv_mean by v_prev delta: dv_mean = mean(v_sim) - v_prev
    # dv_mean_new = mean(v_sim) - v_prev_new = dv_mean_orig + (v_prev_orig - v_prev_new)
    v_prev_delta = v_prev_orig - v_prev_new  # NaN for first-in-end
    if dv_mean_orig is not None:
        df["dv_mean"] = dv_mean_orig + v_prev_delta
    for col in ["dv_p10", "dv_p50", "dv_p90", "cvar_10"]:
        if col in df.columns:
            df[col] = df[col].astype(float) + v_prev_delta

    # dv_std unchanged (shift doesn't affect std)
    # se_mean unchanged

    # Recompute z_obs and percentile_obs
    # percentile_obs can't be recomputed without raw samples, but the rank doesn't
    # change (all samples shift by same delta). Keep original percentile_obs.
    # z_obs = (dv_obs - dv_mean) / dv_std
    if "dv_mean" in df.columns and "dv_std" in df.columns:
        dv_obs_arr = df["dv_obs"].astype(float).values
        dv_mean_arr = df["dv_mean"].astype(float).values
        dv_std_arr = df["dv_std"].astype(float).values
        with np.errstate(divide="ignore", invalid="ignore"):
            df["z_obs"] = np.where(
                np.isfinite(dv_std_arr) & (dv_std_arr > 1e-8),
                (dv_obs_arr - dv_mean_arr) / dv_std_arr, np.nan)

    # Perspective columns
    team_order = df["team_order"].astype(float).values if "team_order" in df.columns else np.zeros(n)

    if "dv_obs_thrower" in df.columns:
        df["dv_obs_thrower"] = df["dv_obs"]
    if "dv_obs_opponent" in df.columns:
        df["dv_obs_opponent"] = -df["dv_obs"].astype(float)
    if "dv_mean_thrower" in df.columns:
        df["dv_mean_thrower"] = df["dv_mean"]
    if "dv_mean_opponent" in df.columns:
        df["dv_mean_opponent"] = -df["dv_mean"].astype(float)

    for base in ["dv_obs", "dv_mean"]:
        for to_val, suffix in [(0.0, "_team_order0"), (1.0, "_team_order1")]:
            col = f"{base}{suffix}"
            if col in df.columns:
                sign = np.where(team_order == to_val, 1.0, -1.0)
                df[col] = df[base].astype(float).values * sign

    # Remove is_hammer if present
    if "is_hammer" in df.columns:
        df = df.drop(columns=["is_hammer"])

    # Clean up temp column
    df = df.drop(columns=["_prev_v_next"])

    # Write
    df.to_csv(out_path, index=False)

    n_first = int(is_first.sum())
    n_valid = n - n_first
    print(f"    {n_valid} shots with sign-flip v_prev, {n_first} first-in-end → NaN", flush=True)


def main():
    for hid_str, comp_id in HOLDOUTS.items():
        print(f"\n=== Holdout {hid_str} ===", flush=True)
        scoring_dir = BASE / "holdouts" / hid_str / "scoring"

        for mode in ["local", "global"]:
            orig = scoring_dir / mode / f"shot_scores_comp_{comp_id}.csv"
            out = scoring_dir / f"shot_scores_{mode}.csv"
            if orig.exists():
                fix_csv(orig, out)
            else:
                print(f"  SKIP: {orig} not found", flush=True)

    print("\nDone!", flush=True)


if __name__ == "__main__":
    main()
