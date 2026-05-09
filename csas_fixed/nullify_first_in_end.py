#!/usr/bin/env python3
"""
Nullify v_prev/dv_obs/dv_mean for first-in-end shots (ShotID=8) that have no
predecessor v_next to sign-flip from. These shots cannot be correctly scored
without the pre-throw state of ShotID=7.
"""

import pathlib
import numpy as np
import pandas as pd

SHOT_KEY = ["CompetitionID", "SessionID", "GameID", "EndID", "ShotID"]
END_KEY = ["CompetitionID", "SessionID", "GameID", "EndID"]

# Columns that depend on v_prev and must be nullified
NULLIFY_COLS = [
    "v_prev", "dv_obs", "dv_mean",
    "dv_std", "dv_p10", "dv_p50", "dv_p90", "cvar_10",
    "percentile_obs", "z_obs", "se_mean",
    "dv_obs_thrower", "dv_mean_thrower",
    "dv_obs_opponent", "dv_mean_opponent",
    "dv_obs_team_order0", "dv_mean_team_order0",
    "dv_obs_team_order1", "dv_mean_team_order1",
]

BASE = pathlib.Path("/mnt/data/curling2/csas_fixed")
HOLDOUT_IDS = ["0", "22230015", "23240026", "24250026"]


def nullify_first_in_end(csv_path):
    df = pd.read_csv(csv_path)
    df = df.sort_values(SHOT_KEY).reset_index(drop=True)

    # Identify first shot in each end
    is_first = df.groupby(END_KEY).cumcount() == 0
    n_first = is_first.sum()

    # Nullify all value-dependent columns for first-in-end shots
    for col in NULLIFY_COLS:
        if col in df.columns:
            df.loc[is_first, col] = np.nan

    df.to_csv(csv_path, index=False)
    print(f"  {csv_path.name}: nullified {n_first} first-in-end shots of {len(df)}", flush=True)


def main():
    for hid in HOLDOUT_IDS:
        print(f"=== Holdout {hid} ===", flush=True)
        scoring_dir = BASE / "holdouts" / hid / "scoring"
        for name in ["shot_scores_local.csv", "shot_scores_global.csv"]:
            csv_path = scoring_dir / name
            if csv_path.exists():
                nullify_first_in_end(csv_path)
    print("\nDone!", flush=True)


if __name__ == "__main__":
    main()
