#!/usr/bin/env python3
"""Build the human-prior policy dataset from inverse-solver throw estimates."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset

from common import ACTION_COLS, FIXED_ROOT, KEY_COLS, POS_MAX, STONE_COLS, log, write_json
from dataset import ValueDataset
from train_holdout_models_cond3 import materialize, make_holdout_split


def load_inverse_estimates(inverse_glob: str, max_loss: float | None) -> pd.DataFrame:
    parts = [pd.read_csv(p) for p in sorted(Path().glob(inverse_glob))] if not Path(inverse_glob).is_absolute() else [
        pd.read_csv(p) for p in sorted(Path(inverse_glob).parent.glob(Path(inverse_glob).name))
    ]
    if not parts:
        raise FileNotFoundError(f"No inverse estimate files matched {inverse_glob}")
    inv = pd.concat(parts, ignore_index=True)
    keep = inv[ACTION_COLS].replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
    if "solver_ok" in inv.columns:
        keep &= inv["solver_ok"].astype(bool).to_numpy()
    if max_loss is not None and "hard_loss_refine" in inv.columns:
        keep &= pd.to_numeric(inv["hard_loss_refine"], errors="coerce").fillna(np.inf).to_numpy() <= max_loss
    return inv.loc[keep, KEY_COLS + ACTION_COLS + ["hard_loss_refine"]].copy()


def build_policy_tensors(
    fixed_root: Path = FIXED_ROOT,
    inverse_glob: str | None = None,
    holdout: int = 0,
    split: str = "train",
    max_loss: float | None = 0.08,
    val_end_frac: float = 0.10,
    split_seed: int = 123,
):
    ds = ValueDataset(
        str(fixed_root / "2026" / "Stones.csv"),
        str(fixed_root / "2026" / "Ends.csv"),
        augment_positions=False,
        augment_flip=False,
    )
    Xp, Xc, Y = materialize(ds)
    train_idx, val_idx, test_idx, _ = make_holdout_split(ds.df, holdout, val_end_frac, split_seed)
    split_idx = {"train": train_idx, "val": val_idx, "test": test_idx, "all": np.arange(len(ds.df))}[split]

    inverse_glob = inverse_glob or str(fixed_root / "inverse_current" / "stones_with_estimates.chunk*.csv")
    inv = load_inverse_estimates(inverse_glob, max_loss)
    group_cols = ["CompetitionID", "SessionID", "GameID", "EndID"]
    df_with_prev = ds.df.copy()
    df_with_prev["_prev_ds_idx"] = (
        pd.Series(df_with_prev.index, index=df_with_prev.index)
        .groupby([df_with_prev[c] for c in group_cols], sort=False)
        .shift(1)
    )
    frame = df_with_prev.iloc[split_idx].reset_index().rename(columns={"index": "_ds_idx"})
    merged = frame.merge(inv, on=KEY_COLS, how="inner")
    merged = merged[np.isfinite(merged["_prev_ds_idx"].to_numpy(dtype=np.float64))].copy()
    if merged.empty:
        raise RuntimeError(
            f"No policy rows after merging split={split} holdout={holdout} with inverse estimates and previous states."
        )

    idx = merged["_ds_idx"].to_numpy(dtype=np.int64)
    prev_idx = merged["_prev_ds_idx"].to_numpy(dtype=np.int64)
    x = Xp[prev_idx].float()
    c = Xc[idx].float()
    y = torch.tensor(merged[ACTION_COLS].to_numpy(dtype=np.float32))
    meta = merged[KEY_COLS + ["TeamID", "ShotIndex", "ShotsInEnd", "hard_loss_refine"]].copy()
    meta["InputShotID"] = merged["ShotID"].to_numpy()
    meta["InputStateSource"] = "previous_in_end"
    return x, c, y, Y[idx].float(), meta


def make_policy_dataset(*args, **kwargs):
    x, c, a, _, _ = build_policy_tensors(*args, **kwargs)
    mean = a.mean(0)
    std = a.std(0).clamp(min=1e-4)
    z = (a - mean) / std
    return TensorDataset(x, c, z), {"action_mean": mean.tolist(), "action_std": std.tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="checkpoints/policy_dataset_stats.json")
    ap.add_argument("--holdout", type=int, default=0)
    ap.add_argument("--max_loss", type=float, default=0.08)
    args = ap.parse_args()
    x, c, a, _, meta = build_policy_tensors(holdout=args.holdout, max_loss=args.max_loss)
    stats = {"rows": len(x), "action_mean": a.mean(0).tolist(), "action_std": a.std(0).tolist()}
    write_json(stats, Path(args.out))
    log(f"policy rows={len(x)} mean={stats['action_mean']} std={stats['action_std']}")


if __name__ == "__main__":
    main()
