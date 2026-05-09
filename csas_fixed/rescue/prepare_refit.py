#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


KEYS = ["CompetitionID", "SessionID", "GameID", "EndID", "ShotID"]


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare selective slot-identity inverse refit inputs.")
    ap.add_argument("--old-csv", required=True, help="Existing full inverse CSV to split into good/bad rows.")
    ap.add_argument("--out-dir", required=True, help="Output directory for keys and preserved good rows.")
    ap.add_argument("--threshold", type=float, default=0.1, help="Refit threshold on hard_loss_refine.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.old_csv, low_memory=False)
    bad = df[df["hard_loss_refine"] > args.threshold].copy()
    good = df[df["hard_loss_refine"] <= args.threshold].copy()

    bad_keys_csv = out_dir / "bad_keys_gt_threshold.csv"
    good_seed_csv = out_dir / "good_seed.csv"
    stats_json = out_dir / "stats.json"

    bad[KEYS].to_csv(bad_keys_csv, index=False)
    good.to_csv(good_seed_csv, index=False)

    stats = {
        "source_csv": str(Path(args.old_csv).resolve()),
        "threshold": args.threshold,
        "total_rows": int(len(df)),
        "good_rows": int(len(good)),
        "bad_rows": int(len(bad)),
        "good_fraction": float(len(good) / len(df)) if len(df) else 0.0,
        "bad_fraction": float(len(bad) / len(df)) if len(df) else 0.0,
    }
    stats_json.write_text(json.dumps(stats, indent=2) + "\n")

    print(json.dumps(stats, indent=2))
    print(f"[wrote] {bad_keys_csv}")
    print(f"[wrote] {good_seed_csv}")
    print(f"[wrote] {stats_json}")


if __name__ == "__main__":
    main()
