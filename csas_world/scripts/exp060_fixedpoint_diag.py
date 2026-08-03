#!/usr/bin/env python3
"""EXP-060: the fixed-point diagnostic — does the student still disagree with its teacher?

Distillation only teaches where teacher != student. For stored collection records
(x0, c0, teacher's chosen action = top-weighted dist target), compute the CURRENT
champion's own deployed selection (WorldPlayer, k=8) at the same state and measure
material disagreement (noise-normalised action distance dn > 2.5). Two student
selections per ply give the stochastic-selection baseline (student-vs-student
self-disagreement among near-ties) — the informative quantity is teacher-student
disagreement ABOVE that baseline, especially on significance-gated plies (the only
plies that train the policy).

  python scripts/exp060_fixedpoint_diag.py --shard-id K --num-shards 4
  python scripts/exp060_fixedpoint_diag.py --aggregate
"""
import argparse
import glob
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")
import world  # noqa: F401
from world import env_bridge

NOISE_STD = np.array([0.0123, 0.003, 2.0, 0.015], dtype=np.float64)

ap = argparse.ArgumentParser()
ap.add_argument("--corpus", default="artifacts/replay/az_v19_train,artifacts/replay/az_v19_val",
                help="comma list of shard dirs (screen-teacher corpus = the 9x-relevant one)")
ap.add_argument("--world", default="checkpoints/csas_world/az_v14d/best.pt")
ap.add_argument("--sel-noise", type=int, default=8)
ap.add_argument("--nonsig-sample", type=int, default=400)
ap.add_argument("--dn-thresh", type=float, default=2.5)
ap.add_argument("--device", default="cuda:0")
ap.add_argument("--out-dir", default="eval_out/exp060_fixedpoint")
ap.add_argument("--num-shards", type=int, default=1)
ap.add_argument("--shard-id", type=int, default=0)
ap.add_argument("--seed", type=int, default=60)
ap.add_argument("--aggregate", action="store_true")
args = ap.parse_args()

OUT = Path(args.out_dir)
OUT.mkdir(parents=True, exist_ok=True)


def aggregate():
    rows = []
    for f in sorted(OUT.glob("shard*.jsonl")):
        rows += [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    if not rows:
        print("no results yet")
        return
    print(f"EXP-060 fixed-point diagnostic over {len(rows)} plies "
          f"(student = az_v14d deployed selection k={args.sel_noise})\n")
    for label, sel in (("SIG-GATED plies (the training signal)", lambda r: r["sig"]),
                       ("non-sig plies", lambda r: not r["sig"])):
        sub = [r for r in rows if sel(r)]
        if not sub:
            continue
        ts = np.array([r["dn_ts"] for r in sub])       # teacher vs student
        ss = np.array([r["dn_ss"] for r in sub])       # student vs student (baseline)
        dts = float((ts > args.dn_thresh).mean())
        dss = float((ss > args.dn_thresh).mean())
        n = len(sub)
        se = math.sqrt(max(dts * (1 - dts), 1e-9) / n)
        print(f"== {label} (n={n})")
        print(f"   teacher-vs-student disagreement: {dts:.1%} ± {100*se:.1f}pp")
        print(f"   student-vs-student baseline:     {dss:.1%}")
        print(f"   EXCESS (headroom signal):        {dts - dss:+.1%}")
        for h in sorted(set(r["h"] for r in sub)):
            hh = [r for r in sub if r["h"] == h]
            t2 = float(np.mean([r["dn_ts"] > args.dn_thresh for r in hh]))
            s2 = float(np.mean([r["dn_ss"] > args.dn_thresh for r in hh]))
            print(f"      h{h:02d}: n={len(hh):4d}  ts {t2:.0%}  ss {s2:.0%}  excess {t2-s2:+.0%}")
        print()


if args.aggregate:
    aggregate()
    sys.exit(0)

# --------------------------------------------------------------------------- #
import torch

from world.eval.head_to_head import WorldPlayer
from world.search.noise import make_noise

device = torch.device(args.device)
env_bridge.warm_jax()
noise_path = "/mnt/data/curling2/csas_v3/configs/noise/v2_fullsheet.json"
student = WorldPlayer(args.world, device, name="student",
                      noise=make_noise(noise_path, seed=601 + args.shard_id),
                      sel_noise_samples=args.sel_noise)
student2 = WorldPlayer(args.world, device, name="student2",
                       noise=make_noise(noise_path, seed=907 + args.shard_id),
                       sel_noise_samples=args.sel_noise)
print(f"[exp060] shard {args.shard_id}/{args.num_shards} rules="
      f"{'NEW' if env_bridge.BOUNDARY_REMOVAL else 'OLD'}", flush=True)

# load corpus rows
X, C, TA, SIG, H = [], [], [], [], []
for d in args.corpus.split(","):
    for f in sorted(glob.glob(f"{d}/*.npz")):
        z = np.load(f, allow_pickle=True)
        w = np.asarray(z["dist_weights"])
        best = np.argmax(w, axis=1)
        X.append(np.asarray(z["x0"]))
        C.append(np.asarray(z["c0"]))
        TA.append(np.asarray(z["dist_actions_raw"])[np.arange(len(best)), best])
        SIG.append(np.asarray(z["dist_mask"]) > 0)
        H.append(np.asarray(z["horizon"]))
X = np.concatenate(X); C = np.concatenate(C); TA = np.concatenate(TA)
SIG = np.concatenate(SIG); H = np.concatenate(H).astype(int)

rng = np.random.default_rng(args.seed)
sig_idx = np.where(SIG)[0]
nonsig_idx = rng.choice(np.where(~SIG)[0], size=min(args.nonsig_sample, int((~SIG).sum())),
                        replace=False)
todo = np.concatenate([sig_idx, nonsig_idx])
print(f"[exp060] corpus rows {len(X)}: {len(sig_idx)} sig + {len(nonsig_idx)} sampled non-sig",
      flush=True)

out_path = OUT / f"shard{args.shard_id}.jsonl"
done = set()
if out_path.exists():
    done = {json.loads(l)["i"] for l in out_path.read_text().splitlines() if l.strip()}


def dn(a, b):
    return float(np.linalg.norm((np.asarray(a, np.float64) - np.asarray(b, np.float64)) / NOISE_STD))


for j, i in enumerate(todo):
    i = int(i)
    if j % args.num_shards != args.shard_id or i in done:
        continue
    x, c, h = X[i].astype(np.float32), C[i].astype(np.float32), int(H[i])
    block = int(round(c[2]))
    s1 = np.asarray(student.select_intended(x, c, h, 10, block), np.float32)
    s2 = np.asarray(student2.select_intended(x, c, h, 10, block), np.float32)
    rec = {"i": i, "h": h, "sig": bool(SIG[i]),
           "dn_ts": round(dn(TA[i], s1), 2), "dn_ss": round(dn(s1, s2), 2)}
    with out_path.open("a") as fh:
        fh.write(json.dumps(rec) + "\n")
    if j % 40 == 0:
        print(f"[exp060] {j}/{len(todo)}", flush=True)

print("EXP060_SHARD_DONE", flush=True)
