#!/usr/bin/env python3
"""EXP-053b: power extension of the depth certification (playouts only).

The EXP-053 searches are done and stored; only the PRIMARY pair (d3 vs the
compute-matched d2p control) is re-adjudicated with 8 additional paired guided
playouts per disagreement (pooled T=16). Roots are rebuilt deterministically
from the same seeds. Pre-registered extension rule (second sequential look):
certify depth iff pooled playout-resolved d3 wins at binomial p < 0.03.

  python scripts/exp053b_extend.py --shard-id K --num-shards 4 --device cuda:0
  python scripts/exp053b_extend.py --aggregate
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")
import world  # noqa: F401
from world import env_bridge
from world.config import Config, load_config

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/exp_037_sig_screen_tree.yaml")
ap.add_argument("--world", default="checkpoints/csas_world/az_v14d/best.pt")
ap.add_argument("--horizons", default="4,6,8,10")
ap.add_argument("--roots-per-h", type=int, default=40)
ap.add_argument("--device", default="cuda:0")
ap.add_argument("--out-dir", default="eval_out/exp053_depth")
ap.add_argument("--num-shards", type=int, default=1)
ap.add_argument("--shard-id", type=int, default=0)
ap.add_argument("--extra-playouts", type=int, default=8)
ap.add_argument("--seed", type=int, default=53)
ap.add_argument("--pair", default="d3_vs_d2p")
ap.add_argument("--aggregate", action="store_true")
args = ap.parse_args()

OUT = Path(args.out_dir)
A_NAME, B_NAME = args.pair.split("_vs_")


def load_rows():
    rows = []
    for f in sorted(OUT.glob("shard*.jsonl")):
        if "ext" in f.name:
            continue
        rows += [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    return rows


def pooled(m1, se1, n1, m2, se2, n2):
    """Exact two-group pooling of mean/SE from per-group stats."""
    s1sq, s2sq = (se1 * math.sqrt(n1)) ** 2, (se2 * math.sqrt(n2)) ** 2
    n = n1 + n2
    m = (n1 * m1 + n2 * m2) / n
    ss = (n1 - 1) * s1sq + (n2 - 1) * s2sq + n1 * (m1 - m) ** 2 + n2 * (m2 - m) ** 2
    var = ss / (n - 1)
    return m, math.sqrt(var / n)


def _binom_p(k, n):
    from math import comb
    if n == 0:
        return 1.0
    return sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n if k >= n / 2 else \
        sum(comb(n, i) for i in range(0, k + 1)) / 2 ** n


if args.aggregate:
    rows = load_rows()
    ext = {}
    for f in OUT.glob("ext_shard*.jsonl"):
        for l in f.read_text().splitlines():
            r = json.loads(l)
            ext[(r["h"], r["i"])] = r
    stats, per_h = [], {}
    for r in rows:
        adj = r.get("adj", {}).get(args.pair, {})
        if not adj.get("playout"):
            continue
        p0 = adj["playout"]
        e = ext.get((r["h"], r["i"]))
        if e:
            m, se = pooled(p0["delta"], p0["se"], p0["T"], e["delta"], e["se"], e["T"])
            n = p0["T"] + e["T"]
        else:
            m, se, n = p0["delta"], p0["se"], p0["T"]
        stats.append((m, se, n, r["h"]))
    res = [(m, h) for m, se, n, h in stats if abs(m) > 2 * se]
    wins = sum(1 for m, _ in res if m > 0)
    pooled_T = sorted(set(n for _, _, n, _ in stats))
    print(f"EXP-053b pooled ({args.pair}): {len(stats)} adjudicated pairs, T={pooled_T}")
    print(f"  resolved (|Δ|>2SE): {len(res)}  |  {A_NAME} wins {wins}/{len(res)}  "
          f"(binom p={_binom_p(wins, len(res)):.4f})")
    print(f"  mean Δ over all adjudicated: {np.mean([m for m,_,_,_ in stats]):+.4f}/end "
          f"± {np.std([m for m,_,_,_ in stats])/math.sqrt(len(stats)):.4f}")
    for h in sorted(set(h for _, h in res)):
        hh = [m for m, hx in res if hx == h]
        print(f"    h{h:02d}: {sum(1 for m in hh if m>0)}/{len(hh)} resolved wins")
    print(f"\n  PRE-REGISTERED EXTENSION VERDICT (p<0.03): "
          f"{'CERTIFIED — depth is a lever' if len(res) and _binom_p(wins, len(res)) < 0.03 and wins > len(res)/2 else 'NOT certified'}")
    sys.exit(0)

# --------------------------------------------------------------------------- #
import torch

from world.eval.head_to_head import WorldPlayer, build_h2h_roots
from world.preplaced import build_preplaced_h2h_roots
from world.search.noise import LocalNoise, make_noise

device = torch.device(args.device)
env_bridge.warm_jax()
cfg_full: Config = load_config(args.config)
noise_path = cfg_full.csas_path(cfg_full.search.noise_config).as_posix()
champion = WorldPlayer(args.world, device, name="playout",
                       noise=make_noise(noise_path, seed=97 + args.shard_id),
                       sel_noise_samples=2)
print(f"[exp053b] shard {args.shard_id}/{args.num_shards} rules="
      f"{'NEW' if env_bridge.BOUNDARY_REMOVAL else 'OLD'}", flush=True)


def guided_playout(x, c, hh, sie, first_action, seed):
    persp = int(round(c[2]))
    nz = LocalNoise(noise_path, seed=seed)
    st, cc, h = x.copy(), c.copy(), hh
    a = np.asarray(first_action, np.float32)
    while h >= 1:
        realized = nz.sample_batch(a[None], 1).reshape(4).astype(np.float32)
        post, _ = env_bridge.apply_legality(st, env_bridge.simulate_one(st, cc, realized)[None], h, cc)
        st, cc, h = post[0], env_bridge.next_condition(cc, sie), h - 1
        if h >= 1:
            a = np.asarray(champion.select_intended(st, cc, h, sie, int(round(cc[2]))), np.float32)
    return float(env_bridge.score_end(st, persp))


rows = load_rows()
roots_by_h = {}
for h in [int(v) for v in args.horizons.split(",")]:
    if h >= 10:
        roots_by_h[h] = build_preplaced_h2h_roots(h, args.roots_per_h, split="val", seed=args.seed + h)
    else:
        roots_by_h[h] = build_h2h_roots(cfg_full.paths.csas_v3_root, h, args.roots_per_h,
                                        split="val", seed=args.seed + h)

out_path = OUT / f"ext_shard{args.shard_id}.jsonl"
done = set()
if out_path.exists():
    for l in out_path.read_text().splitlines():
        r = json.loads(l)
        done.add((r["h"], r["i"]))

for r in rows:
    adj = r.get("adj", {}).get(args.pair, {})
    if not adj.get("playout") or (r["h"], r["i"]) in done:
        continue
    if r["i"] % args.num_shards != args.shard_id:
        continue
    root = roots_by_h[r["h"]][r["i"]]
    x, c = root.x.astype(np.float32), root.c.astype(np.float32)
    sie = int(root.shots_in_end)
    A = np.asarray(r["ops"][A_NAME]["action"], np.float32)
    B = np.asarray(r["ops"][B_NAME]["action"], np.float32)
    ds = []
    for t in range(8, 8 + args.extra_playouts):   # fresh seeds, disjoint from T=0..7
        seed = 100000 + 977 * t + args.shard_id
        ds.append(guided_playout(x, c, r["h"], sie, A, seed)
                  - guided_playout(x, c, r["h"], sie, B, seed))
    ds = np.asarray(ds, np.float64)
    rec = {"h": r["h"], "i": r["i"], "pair": args.pair,
           "delta": float(ds.mean()), "se": float(ds.std(ddof=1) / math.sqrt(len(ds))),
           "T": len(ds)}
    with out_path.open("a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(f"[exp053b] h{r['h']} i{r['i']} delta={rec['delta']:+.3f}±{rec['se']:.3f}", flush=True)

print("EXP053B_SHARD_DONE", flush=True)
