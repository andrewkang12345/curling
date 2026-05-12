#!/usr/bin/env python3
"""Compare old vs distilled value models by throw number / end phase."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from common import FIXED_ROOT, log
from dataset import ValueDataset
from kr_uct_search import load_value_model
from train_holdout_models_cond3 import make_holdout_split, materialize


@torch.no_grad()
def predict(model, x, c, device, batch_size=4096):
    vals = []
    loader = DataLoader(TensorDataset(x, c), batch_size=batch_size, shuffle=False)
    for xb, cb in loader:
        mean, _ = model(xb.to(device), cb.to(device))
        vals.append(mean.squeeze(-1).cpu().numpy())
    return np.concatenate(vals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", type=int, default=0)
    ap.add_argument("--old", default="/mnt/data/curling2/csas_fixed/holdouts/0/model_settf_gaussian/model.pt")
    ap.add_argument("--new", default="checkpoints/value_search_distilled/model.pt")
    ap.add_argument("--out", default="logs/value_by_shot_comparison.csv")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    ds = ValueDataset(str(FIXED_ROOT / "2026" / "Stones.csv"), str(FIXED_ROOT / "2026" / "Ends.csv"), augment_positions=False, augment_flip=False)
    Xp, Xc, Y = materialize(ds)
    _, _, test_idx, _ = make_holdout_split(ds.df, args.holdout, 0.10, 123)
    old_model = load_value_model(args.old, device)
    new_model = load_value_model(args.new, device)
    x, c, y = Xp[test_idx], Xc[test_idx], Y[test_idx].squeeze(-1).numpy()
    old = predict(old_model, x, c, device)
    new = predict(new_model, x, c, device)
    frame = ds.df.iloc[test_idx][["CompetitionID", "ShotID", "ShotIndex", "ShotsInEnd", "shot_norm"]].copy()
    frame["y"] = y
    frame["old_pred"] = old
    frame["new_pred"] = new
    frame["phase"] = pd.cut(frame["shot_norm"], bins=[-0.01, 0.33, 0.66, 1.01], labels=["early", "mid", "late"])
    rows = []
    for name, sub in list(frame.groupby("phase", observed=False)) + [("all", frame)]:
        rows.append(
            {
                "holdout": int(args.holdout),
                "phase": str(name),
                "n": int(len(sub)),
                "old_mse": float(np.mean((sub["old_pred"] - sub["y"]) ** 2)),
                "new_mse": float(np.mean((sub["new_pred"] - sub["y"]) ** 2)),
            }
        )
    for shot_idx, sub in frame.groupby("ShotIndex"):
        rows.append(
            {
                "holdout": int(args.holdout),
                "phase": f"shot_{int(shot_idx)}",
                "n": int(len(sub)),
                "old_mse": float(np.mean((sub["old_pred"] - sub["y"]) ** 2)),
                "new_mse": float(np.mean((sub["new_pred"] - sub["y"]) ** 2)),
            }
        )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    log(pd.DataFrame(rows).to_string(index=False))
    log(f"saved {out}")


if __name__ == "__main__":
    main()
