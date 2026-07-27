#!/usr/bin/env python3
"""EXP-053: is search DEPTH a lever? Per-ply certification of a depth-3 screen-beam
operator against the certified depth-2 screen_tree — including a COMPUTE-MATCHED
d2 control — with disagreements adjudicated by paired guided playouts.

Per root ply, three operators choose an action (all value-free, all proposing from
the same exported champion policy, all under the current default rules):
  d2   : collection operator (az_v12 screen_tree, exp_037 knobs)
  d2p  : same operator, budget raised to ~match d3 (stage-1 k_ego 48, tree 128 sims)
  d3   : recursive screen-beam, minimax backup, CRN screens (world.search.beam)

Adjudication of pairs that chose materially different shots (noise-normalised
action distance > --dn-thresh):
  * paired terminal-MC (k=64, CRN): cheap, but biased toward behavior-policy
    continuations (recorded as secondary evidence)
  * paired GUIDED PLAYOUTS (primary): play the end out T times from each action
    with the deployed champion (WorldPlayer, robust selection) moving BOTH sides,
    common noise streams across the two branches.

Pre-registered primary metric: among playout-resolved d3-vs-d2p disagreements
(|Δ| > 2·SE), the fraction won by d3 (binomial vs 0.5). Sharded + resumable
(JSONL per shard). Aggregate with --aggregate.

  python scripts/exp053_depth_cert.py --shard-id 0 --num-shards 4 --device cuda:0
  python scripts/exp053_depth_cert.py --aggregate
"""
import argparse
import copy
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")
import world  # noqa: F401
from world import env_bridge
from world.config import Config, load_config

NOISE_STD = np.array([0.0123, 0.003, 2.0, 0.015], dtype=np.float64)  # v2_fullsheet per-dim SDs

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/exp_037_sig_screen_tree.yaml")
ap.add_argument("--policy", default="checkpoints/csas_world/az_v15_L8/incumbent0_policy_csas.pt")
ap.add_argument("--world", default="checkpoints/csas_world/az_v14d/best.pt")
ap.add_argument("--horizons", default="4,6,8,10")
ap.add_argument("--roots-per-h", type=int, default=40)
ap.add_argument("--device", default="cuda:0")
ap.add_argument("--out-dir", default="eval_out/exp053_depth")
ap.add_argument("--num-shards", type=int, default=1)
ap.add_argument("--shard-id", type=int, default=0)
ap.add_argument("--playouts", type=int, default=8)
ap.add_argument("--dn-thresh", type=float, default=2.5)
ap.add_argument("--seed", type=int, default=53)
ap.add_argument("--aggregate", action="store_true")
args = ap.parse_args()

OUT = Path(args.out_dir)
OUT.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def aggregate():
    rows = []
    for f in sorted(OUT.glob("shard*.jsonl")):
        rows += [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    if not rows:
        print("no results yet")
        return
    print(f"EXP-053 aggregate over {len(rows)} plies "
          f"(horizons {sorted(set(r['h'] for r in rows))})\n")
    for op in ("d2", "d2p", "d3"):
        ts = [r["ops"][op]["seconds"] for r in rows if op in r["ops"]]
        print(f"  {op:4s} mean wall-clock {np.mean(ts):6.1f}s / ply")
    for pair in (("d3", "d2p"), ("d3", "d2"), ("d2p", "d2")):
        key = f"{pair[0]}_vs_{pair[1]}"
        sub = [r for r in rows if key in r.get("adj", {})]
        dns = [r["dn"].get(key, 0.0) for r in rows]
        n_dis = sum(1 for d in dns if d > args.dn_thresh)
        print(f"\n== {key}: disagreement rate {n_dis}/{len(rows)} "
              f"({100*n_dis/max(len(rows),1):.0f}%) at dn>{args.dn_thresh}")
        # paired MC
        mc = [(r["adj"][key]["mc"]["delta"], r["adj"][key]["mc"]["se"]) for r in sub
              if r["adj"][key].get("mc")]
        if mc:
            res = [d for d, s in mc if abs(d) > 2 * s]
            wins = sum(1 for d in res if d > 0)
            print(f"   MC(k=64,CRN):     {len(mc)} adjudicated, {len(res)} resolved, "
                  f"{pair[0]} wins {wins}/{len(res)}"
                  + (f"  (binom p={_binom_p(wins, len(res)):.3f})" if res else ""))
        # guided playouts (primary for d3 pairs)
        po = [(r["adj"][key]["playout"]["delta"], r["adj"][key]["playout"]["se"], r["h"])
              for r in sub if r["adj"][key].get("playout")]
        if po:
            res = [(d, h) for d, s, h in po if abs(d) > 2 * s]
            wins = sum(1 for d, _ in res if d > 0)
            mean_d = np.mean([d for d, _, _ in po])
            print(f"   PLAYOUTS(T={args.playouts}): {len(po)} adjudicated, {len(res)} resolved, "
                  f"{pair[0]} wins {wins}/{len(res)}"
                  + (f"  (binom p={_binom_p(wins, len(res)):.3f})" if res else "")
                  + f" | mean Δ={mean_d:+.3f}/end over all adjudicated")
            for h in sorted(set(h for _, h in res)):
                hh = [d for d, hx in res if hx == h]
                print(f"      h{h:02d}: {sum(1 for d in hh if d>0)}/{len(hh)} resolved wins")


def _binom_p(k, n):
    if n == 0:
        return 1.0
    from math import comb
    return sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n if k >= n / 2 else \
        sum(comb(n, i) for i in range(0, k + 1)) / 2 ** n


if args.aggregate:
    aggregate()
    sys.exit(0)

# --------------------------------------------------------------------------- #
# Setup
# --------------------------------------------------------------------------- #
import torch
from csas.search import load_policy

from world.eval.head_to_head import WorldPlayer
from world.preplaced import build_preplaced_h2h_roots
from world.eval.head_to_head import build_h2h_roots
from world.search.beam import screen_beam_choose, screen_tree_choose
from world.search.collect import _mc_rollout_terminal_batch
from world.search.noise import LocalNoise, make_noise

device = torch.device(args.device)
env_bridge.warm_jax()
cfg_full: Config = load_config(args.config)
cfg = cfg_full.search
cfg_p = copy.deepcopy(cfg)          # d2p: compute-matched control
cfg_p.noise_samples = 48
cfg_p.mcts_sims = 128

policy, amean_t, astd_t = load_policy(args.policy, device)
amean_np = amean_t.detach().cpu().numpy().astype(np.float64)
astd_np = astd_t.detach().cpu().numpy().astype(np.float64)
noise_path = cfg_full.csas_path(cfg.noise_config).as_posix()
NZ = make_noise(noise_path, seed=args.seed * 101 + args.shard_id)
champion = WorldPlayer(args.world, device, name="playout",
                       noise=make_noise(noise_path, seed=97 + args.shard_id),
                       sel_noise_samples=2)
print(f"[exp053] shard {args.shard_id}/{args.num_shards} rules="
      f"{'NEW' if env_bridge.BOUNDARY_REMOVAL else 'OLD'} device={args.device}", flush=True)



def dn_dist(a, b):
    return float(np.linalg.norm((np.asarray(a, np.float64) - np.asarray(b, np.float64)) / NOISE_STD))


def paired_mc(x, c, hh, sie, A, B, k=64):
    """Paired terminal-MC gap Q(A)-Q(B), CRN executions (root perspective)."""
    persp = int(round(c[2]))
    nc = env_bridge.next_condition(c, sie)
    realized = NZ.sample_batch(np.stack([A, B]).astype(np.float32), k, crn=True).reshape(-1, 4)
    posts, _ = env_bridge.apply_legality(x, env_bridge.simulate(x, c, realized), hh, c)
    rng = np.random.default_rng(1234)
    q = _mc_rollout_terminal_batch(policy, amean_t, astd_t, posts, nc, hh - 1, sie,
                                   persp, device, rng, NZ, cfg.rollout_temp, cfg.std_scale,
                                   value_model=None, n_search=1).reshape(2, k)
    d = q[0] - q[1]
    return {"delta": float(d.mean()), "se": float(d.std(ddof=1) / math.sqrt(k))}


def guided_playout(x, c, hh, sie, first_action, seed):
    """Play the end out: forced (noisy) first throw, then the deployed champion
    moves BOTH sides. Returns final margin from the root thrower's perspective."""
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


def paired_playouts(x, c, hh, sie, A, B, T):
    ds = []
    for t in range(T):
        seed = 100000 + 977 * t + args.shard_id
        ds.append(guided_playout(x, c, hh, sie, A, seed) - guided_playout(x, c, hh, sie, B, seed))
    ds = np.asarray(ds, np.float64)
    return {"delta": float(ds.mean()), "se": float(ds.std(ddof=1) / math.sqrt(len(ds))),
            "T": int(T)}


# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #
out_path = OUT / f"shard{args.shard_id}.jsonl"
done = set()
if out_path.exists():
    for l in out_path.read_text().splitlines():
        try:
            r = json.loads(l)
            done.add((r["h"], r["i"]))
        except Exception:
            pass

horizons = [int(h) for h in args.horizons.split(",")]
for h in horizons:
    if h >= 10:
        roots = build_preplaced_h2h_roots(h, args.roots_per_h, split="val", seed=args.seed + h)
    else:
        roots = build_h2h_roots(cfg_full.paths.csas_v3_root, h, args.roots_per_h,
                                split="val", seed=args.seed + h)
    for i, root in enumerate(roots):
        if i % args.num_shards != args.shard_id or (h, i) in done:
            continue
        x, c = root.x.astype(np.float32), root.c.astype(np.float32)
        sie = int(root.shots_in_end)
        rec = {"h": h, "i": i, "ops": {}, "dn": {}, "adj": {}}
        rng = np.random.default_rng(args.seed * 7919 + h * 131 + i)
        acts = {}
        for name, fn in (
            ("d2", lambda: screen_tree_choose(policy, amean_t, astd_t, amean_np, astd_np,
                                              x, c, h, sie, cfg, rng, device, NZ)),
            ("d2p", lambda: screen_tree_choose(policy, amean_t, astd_t, amean_np, astd_np,
                                               x, c, h, sie, cfg_p, rng, device, NZ)),
            ("d3", lambda: screen_beam_choose(policy, amean_t, astd_t, amean_np, astd_np,
                                              x, c, h, sie, cfg, rng, device, NZ)),
        ):
            t0 = time.time()
            res = fn()
            if res is None:
                break
            acts[name] = np.asarray(res["action"], np.float64)
            rec["ops"][name] = {"action": [round(float(v), 6) for v in res["action"]],
                                "q": round(res["q"], 4),
                                "seconds": round(time.time() - t0, 1)}
        if len(acts) < 3:
            continue
        for a_name, b_name in (("d3", "d2p"), ("d3", "d2"), ("d2p", "d2")):
            key = f"{a_name}_vs_{b_name}"
            dn = dn_dist(acts[a_name], acts[b_name])
            rec["dn"][key] = round(dn, 2)
            if dn <= args.dn_thresh:
                continue
            adj = {"mc": paired_mc(x, c, h, sie, acts[a_name].astype(np.float32),
                                   acts[b_name].astype(np.float32))}
            if a_name == "d3":   # playout adjudication for the primary + secondary pairs
                adj["playout"] = paired_playouts(x, c, h, sie, acts[a_name].astype(np.float32),
                                                 acts[b_name].astype(np.float32), args.playouts)
            rec["adj"][key] = adj
        with out_path.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(f"[exp053] h{h} root {i}: " +
              " ".join(f"{k}={v['seconds']}s" for k, v in rec["ops"].items()) +
              f" dn={rec['dn']}", flush=True)

print("EXP053_SHARD_DONE", flush=True)
