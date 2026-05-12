#!/usr/bin/env python3
"""Generate search-improved value targets for distillation."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common import KEY_COLS, log, set_seed
from kr_uct_search import KRUctSearcher
from policy_dataset import build_policy_tensors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", type=int, default=0)
    ap.add_argument("--policy", default="checkpoints/policy_prior/model.pt")
    ap.add_argument("--value", default="/mnt/data/curling2/csas_fixed/holdouts/0/model_settf_gaussian/model.pt")
    ap.add_argument("--out", default="search_targets/holdout0_search_targets.csv")
    ap.add_argument("--split", default="train", choices=["train", "val", "test", "all"])
    ap.add_argument("--max_rows", type=int, default=0)
    ap.add_argument("--shard_index", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--candidates", type=int, default=256)
    ap.add_argument("--rollout_depth", type=int, default=1)
    ap.add_argument("--child_candidates", type=int, default=64)
    ap.add_argument("--kernel_bandwidth", type=float, default=0.75)
    ap.add_argument("--uct_c", type=float, default=0.05)
    ap.add_argument("--early_mid_oversample", type=float, default=1.5)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    set_seed(args.seed + args.shard_index)
    out = Path(args.out)
    log_path = out.with_suffix(".log")
    x, c, actions, y, meta = build_policy_tensors(holdout=args.holdout, split=args.split, max_loss=None)
    rows = np.arange(len(x))
    if args.early_mid_oversample > 1.0:
        shot_norm = c[:, 0].numpy()
        early = rows[shot_norm <= 0.60]
        extra = np.random.choice(early, size=int((args.early_mid_oversample - 1.0) * len(early)), replace=True) if len(early) else np.array([], dtype=np.int64)
        rows = np.concatenate([rows, extra])
    rows = rows[rows % args.num_shards == args.shard_index]
    if args.max_rows > 0:
        rows = rows[: args.max_rows]
    log(f"search rows={len(rows)} shard={args.shard_index}/{args.num_shards} device={args.device}", log_path)

    searcher = KRUctSearcher(
        args.policy,
        args.value,
        device=args.device,
        candidates=args.candidates,
        rollout_depth=args.rollout_depth,
        child_candidates=args.child_candidates,
        kernel_bandwidth=args.kernel_bandwidth,
        uct_c=args.uct_c,
    )
    records = []
    for n, idx in enumerate(rows, start=1):
        res = searcher.search(x[idx].numpy(), c[idx].numpy())
        target = float(res.get("rollout_value", res["best_value"]))
        rec = meta.iloc[int(idx)][KEY_COLS + ["TeamID", "ShotIndex", "ShotsInEnd"]].to_dict()
        rec.update(
            {
                "search_target": target,
                "root_best_value": float(res["best_value"]),
                "root_best_score": float(res["best_score"]),
                "root_mean_value": float(res["mean_value"]),
                "root_p90_value": float(res["p90_value"]),
                "human_value_target": float(y[idx].item()),
                "shot_norm": float(c[idx, 0].item()),
                "best_speed": float(res["best_action"][0]),
                "best_angle": float(res["best_action"][1]),
                "best_spin": float(res["best_action"][2]),
                "best_y0": float(res["best_action"][3]),
            }
        )
        records.append(rec)
        if n % 25 == 0:
            log(f"finished {n}/{len(rows)} latest_target={target:.4f}", log_path)
            pd.DataFrame(records).to_csv(out, index=False)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(out, index=False)
    log(f"saved {out} rows={len(records)}", log_path)


if __name__ == "__main__":
    main()
