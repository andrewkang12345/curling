#!/usr/bin/env python3
"""EXP-064 backfill: add rank_pos/rank_neg/rank_cond/rank_mask to existing corpora.

For each significance-gated record (dist_mask > 0, horizon >= 2), take the stored
top-1/top-2 target actions, simulate RANK_R noisy executions of each from
(x0, c0) with the authoritative sim (current rules), and store the post-states.
Writes AUGMENTED COPIES to --out-train/--out-val, mirroring the existing
train/val split dirs; originals are untouched.

  python scripts/backfill_rank_posts.py \
      --in-train artifacts/replay/az_v19_train --in-val artifacts/replay/az_v19_val \
      --out-train artifacts/replay/az_v19_rank_train --out-val artifacts/replay/az_v19_rank_val
"""
import argparse
import glob
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")
import world  # noqa: F401
from world import env_bridge
from world.replay.schema import RANK_R
from world.search.noise import make_noise

ap = argparse.ArgumentParser()
ap.add_argument("--in-train", default="artifacts/replay/az_v19_train")
ap.add_argument("--in-val", default="artifacts/replay/az_v19_val")
ap.add_argument("--out-train", default="artifacts/replay/az_v19_rank_train")
ap.add_argument("--out-val", default="artifacts/replay/az_v19_rank_val")
ap.add_argument("--seed", type=int, default=64)
args = ap.parse_args()

env_bridge.warm_jax()
NZ = make_noise("/mnt/data/curling2/csas_v3/configs/noise/v2_fullsheet.json", seed=args.seed)
print(f"[backfill] rules={'NEW' if env_bridge.BOUNDARY_REMOVAL else 'OLD'} R={RANK_R}", flush=True)

for in_dir, out_dir in ((args.in_train, args.out_train), (args.in_val, args.out_val)):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    n_rows = n_filled = 0
    for fp in sorted(glob.glob(f"{in_dir}/*.npz")):
        z = dict(np.load(fp, allow_pickle=True))
        N = len(z["dist_mask"])
        rank_pos = np.zeros((N, RANK_R, 24), np.float32)
        rank_neg = np.zeros((N, RANK_R, 24), np.float32)
        rank_cond = np.zeros((N, 3), np.float32)
        rank_mask = np.zeros((N,), np.float32)
        for i in range(N):
            if z["dist_mask"][i] <= 0 or int(z["horizon"][i]) < 2:
                continue
            w = np.asarray(z["dist_weights"][i])
            top = np.argsort(w)[::-1][:2]
            if w[top[1]] <= 0:
                continue
            x, c = z["x0"][i].astype(np.float32), z["c0"][i].astype(np.float32)
            h = int(z["horizon"][i])
            a1 = z["dist_actions_raw"][i][top[0]].astype(np.float32)
            a2 = z["dist_actions_raw"][i][top[1]].astype(np.float32)
            realized = NZ.sample_batch(np.stack([a1, a2]), RANK_R, crn=True).reshape(-1, 4)
            posts, _ = env_bridge.apply_legality(x, env_bridge.simulate(x, c, realized), h, c)
            posts = posts.reshape(2, RANK_R, 24)
            rank_pos[i], rank_neg[i] = posts[0], posts[1]
            rank_cond[i] = env_bridge.next_condition(c, 10)
            rank_mask[i] = 1.0
            n_filled += 1
        n_rows += N
        z.update(rank_pos=rank_pos, rank_neg=rank_neg, rank_cond=rank_cond, rank_mask=rank_mask)
        np.savez_compressed(Path(out_dir) / Path(fp).name, **z)
    print(f"[backfill] {in_dir} -> {out_dir}: {n_rows} rows, {n_filled} rank pairs", flush=True)
print("BACKFILL_DONE", flush=True)
