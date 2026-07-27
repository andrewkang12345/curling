#!/usr/bin/env python3
"""Eval a model vs the human prior at h10 (the pre-placed opening), SPLIT BY MODE
(standard / pp_left / pp_right), to test whether the model is weaker on the off-center
openings. NOISY winrate (robust selection + realized execution), both throw orders.
"""
import argparse
import math
import sys

sys.path.insert(0, "src")
import numpy as np

import world  # noqa: F401  (bootstrap GNN env + csas path)
from world import env_bridge
from world.config import Config
from world.eval.head_to_head import CsasPlayer, WorldPlayer, H2HRoot, head_to_head
from world.preplaced import board_norm, PREPLACED_HORIZON, PREPLACED_SHOTS_IN_END
from world.search.noise import make_noise
from csas.preplaced_value_data import load_preplaced_training_frame

ap = argparse.ArgumentParser()
ap.add_argument("--champion", required=True)
ap.add_argument("--device", default="cuda:0")
ap.add_argument("--split", default="val")
ap.add_argument("--max-per-mode", type=int, default=0, help="cap roots per mode (0=all); for speed on CPU")
ap.add_argument("--sel-noise-samples", type=int, default=-1, help="robust-selection K (-1=cfg default)")
ap.add_argument("--only-mode", default=None, help="evaluate just one mode (for parallel sharding across GPUs)")
args = ap.parse_args()

import torch
cfg = Config()
dev = torch.device(args.device)
env_bridge.warm_jax()
ncfg = cfg.csas_path(cfg.search.noise_config).as_posix()
sns = int(cfg.search.noise_samples) if args.sel_noise_samples < 0 else int(args.sel_noise_samples)
noisy = sns > 0          # sns=0 -> deterministic selection + execution (fast relative comparison)
h = PREPLACED_HORIZON
nzA = make_noise(ncfg, 1000 * h + 1) if noisy else None
nzB = make_noise(ncfg, 1000 * h + 2) if noisy else None
env_nz = make_noise(ncfg, 1000 * h + 9) if noisy else None
A = WorldPlayer(args.champion, dev, name="ours", noise=nzA, sel_noise_samples=sns if noisy else 0)
B = CsasPlayer(cfg.csas_path(cfg.paths.prior_policy_ckpt).as_posix(),
               cfg.csas_path(cfg.paths.prior_value_ckpt).as_posix(),
               dev, name="prior", noise=nzB, sel_noise_samples=sns if noisy else 0)

df = load_preplaced_training_frame()
comp = df["CompetitionID"].astype(int).to_numpy()
df = df[(comp == 0) if args.split == "val" else (comp != 0)]

print(f"[h10-by-mode] champion={args.champion}  split={args.split}  NOISY (robust select x{sns} + realized noise)")
print(f"{'mode':10s} {'n_roots':>7s} {'winrate':>9s} {'±SE':>6s}  {'w/hammer':>8s} {'no-hammer':>9s} {'dScore':>7s}")
modes = (args.only_mode,) if args.only_mode else ("standard", "pp_left", "pp_right")
for mode in modes:
    sub = df[df["mode"] == mode]
    if args.max_per_mode and len(sub) > args.max_per_mode:
        sub = sub.sample(n=args.max_per_mode, random_state=0)
    roots = [H2HRoot(board_norm(mode, int(r["guard_slot"])),
                     np.asarray([0.0, 0.0, float(round(float(r["thrower_block"])))], np.float32),
                     PREPLACED_SHOTS_IN_END, h) for _, r in sub.iterrows()]
    if not roots:
        print(f"{mode:10s} {'0':>7s}  (no roots)"); continue
    res = head_to_head(A, B, roots, env_noise=env_nz, realize_noise=noisy)
    w, n = res["winrate"], res["n_ends"]
    se = math.sqrt(max(w * (1 - w), 1e-9) / n)
    # order0 = ours as to-move (h10 even -> ours NO hammer); order1 = ours hammer
    no_ham, ham = res["winrate_order0"], res["winrate_order1"]
    print(f"{mode:10s} {len(roots):>7d} {w:>9.3f} {se:>6.3f}  {ham:>8.3f} {no_ham:>9.3f} {res['mean_margin']:>+7.3f}")
print("H10_BY_MODE_DONE")
