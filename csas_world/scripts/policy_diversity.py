#!/usr/bin/env python3
"""Measure how diverse / overlapping each policy's proposed candidates are.

At a set of real boards, draw the same n_candidates the search/eval uses (deployment
temperature 1.1, std_scale 1.2) from each policy and quantify their spread:
  * rel-spread : per-dim std of the candidates / the dataset-wide action std
                 (z-units). ~1 => candidates span the dataset's variation; <<1 =>
                 tightly clustered (overlapping).
  * pair-dist  : mean pairwise Euclidean distance between candidates in z-space.
Compared across the human prior, anchor_noisy, and az_v3/iter1.
"""
from __future__ import annotations

import numpy as np
import torch

import world  # noqa: F401  (bootstrap)
from world.eval.head_to_head import CsasPlayer, WorldPlayer, build_h2h_roots

CSAS = "/mnt/data/curling2/csas_v3"
HUMAN = f"{CSAS}/checkpoints/policy/human_prior_fullcov/best.pt"
VALUE = f"{CSAS}/checkpoints/value/holdout0/model.pt"
ANCHOR = "checkpoints/csas_world/anchor_noisy/model.pt"
LATEST = "checkpoints/csas_world/az_v3/iter1/model.pt"
DIMS = ["speed", "angle", "spin", "y0"]


def measure(player, states_conds, amean, astd, n=48):
    rel = []           # per-state per-dim std / astd
    pair = []          # per-state mean pairwise z-distance
    for x, c in states_conds:
        cands = np.asarray(player._sample_fn(x, c, n), np.float32)   # [n,4] physical
        if len(cands) < 2:
            continue
        rel.append(cands.std(axis=0) / astd)
        z = (cands - amean) / astd
        d = np.linalg.norm(z[:, None, :] - z[None, :, :], axis=-1)
        iu = np.triu_indices(len(z), k=1)
        pair.append(float(d[iu].mean()))
    rel = np.stack(rel)               # [S,4]
    return rel.mean(axis=0), float(rel.mean()), float(np.mean(pair))


def main():
    dev = torch.device("cuda:0")
    # states across the end
    scs = []
    for h in (3, 6, 9):
        for r in build_h2h_roots(CSAS, h, 10, split="val", seed=h):
            scs.append((r.x.astype(np.float32), r.c.astype(np.float32)))
    print(f"[diversity] {len(scs)} states; n_candidates=48; temp=1.1 std_scale=1.2\n")

    players = [
        ("human_prior", CsasPlayer(HUMAN, VALUE, dev, n_candidates=48, name="human")),
        ("anchor_noisy", WorldPlayer(ANCHOR, dev, n_candidates=48, name="anchor")),
        ("az_v3/iter1", WorldPlayer(LATEST, dev, n_candidates=48, name="v3")),
    ]
    # shared dataset action std/mean (identical across checkpoints' z-normalisation)
    amean = np.asarray(players[1][1].model.action_mean.cpu().numpy(), np.float32)
    astd = np.asarray(players[1][1].model.action_std.cpu().numpy(), np.float32)
    print(f"dataset action std [speed,angle,spin,y0] = {np.round(astd,4)}\n")

    hdr = f"{'policy':14s} {'rel-spread':>11s} {'pair-dist':>10s}   per-dim rel-spread " + str(DIMS)
    print(hdr); print("-" * len(hdr))
    for name, pl in players:
        per_dim, overall, pair = measure(pl, scs, amean, astd)
        print(f"{name:14s} {overall:11.3f} {pair:10.3f}   {np.round(per_dim,3).tolist()}")


if __name__ == "__main__":
    main()
