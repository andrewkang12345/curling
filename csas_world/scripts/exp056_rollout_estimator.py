#!/usr/bin/env python3
"""EXP-056: the Sage-3T question — is the lever the ROLLOUT ESTIMATOR, not tree depth?

2x2 factorial on how root candidates are scored, all arms sharing the SAME dense
candidate pool per ply (proposal variance removed from the comparison):

  RT  : raw-policy rollouts to terminal, k_ego 8           (current stage-1)
  RtT : raw-policy rollouts TRUNCATED @4 throws + champion-V leaf, k_ego 8
  ST  : SEARCHED rollouts (value-greedy steps, n_search=6) to terminal, k_ego 4
  StT : searched + truncated@4 + V leaf, k_ego 4           <- the Sage-3T analog

plus `record` = the full screen_tree operator of record (exp_037) as the
deployment reference. Adjudication: paired guided playouts T=20 on the PRIMARY
contrast StT-vs-record; paired raw-terminal MC (k=64, CRN) on the factorial
contrasts (each cell vs RT) and record-vs-RT. Mean adjudicated Δ is primary.

  python scripts/exp056_rollout_estimator.py --shard-id K --num-shards 4
  python scripts/exp056_rollout_estimator.py --aggregate
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")
import world  # noqa: F401
from world import env_bridge
from world.config import Config, load_config

NOISE_STD = np.array([0.0123, 0.003, 2.0, 0.015], dtype=np.float64)

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/exp_037_sig_screen_tree.yaml")
ap.add_argument("--policy", default="checkpoints/csas_world/az_v15_L8/incumbent0_policy_csas.pt")
ap.add_argument("--world", default="checkpoints/csas_world/az_v14d/best.pt")
ap.add_argument("--horizons", default="6,8,10")
ap.add_argument("--roots-per-h", type=int, default=32)
ap.add_argument("--device", default="cuda:0")
ap.add_argument("--out-dir", default="eval_out/exp056_rollout")
ap.add_argument("--num-shards", type=int, default=1)
ap.add_argument("--shard-id", type=int, default=0)
ap.add_argument("--playouts", type=int, default=20)
ap.add_argument("--dn-thresh", type=float, default=2.5)
ap.add_argument("--trunc", type=int, default=4)
ap.add_argument("--n-search", type=int, default=6)
ap.add_argument("--seed", type=int, default=56)
ap.add_argument("--aggregate", action="store_true")
args = ap.parse_args()

OUT = Path(args.out_dir)
OUT.mkdir(parents=True, exist_ok=True)

MC_PAIRS = (("StT", "record"), ("StT", "RT"), ("ST", "RT"), ("RtT", "RT"), ("record", "RT"))
PLAYOUT_PAIRS = (("StT", "record"),)


def _binom_p(k, n):
    from math import comb
    if n == 0:
        return 1.0
    return sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n if k >= n / 2 else \
        sum(comb(n, i) for i in range(0, k + 1)) / 2 ** n


def aggregate():
    rows = []
    for f in sorted(OUT.glob("shard*.jsonl")):
        rows += [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    if not rows:
        print("no results yet")
        return
    print(f"EXP-056 aggregate over {len(rows)} plies")
    for op in ("RT", "RtT", "ST", "StT", "record"):
        ts = [r["ops"][op]["seconds"] for r in rows if op in r["ops"]]
        if ts:
            print(f"  {op:7s} mean wall-clock {np.mean(ts):6.1f}s / ply")
    for a, b in MC_PAIRS:
        key = f"{a}_vs_{b}"
        sub = [r for r in rows if key in r.get("adj", {})]
        n_dis = sum(1 for r in rows if r["dn"].get(key, 0.0) > args.dn_thresh)
        print(f"\n== {key}: disagreements {n_dis}/{len(rows)}")
        for est in ("playout", "mc"):
            ds = [(r["adj"][key][est]["delta"], r["adj"][key][est]["se"], r["h"])
                  for r in sub if r["adj"][key].get(est)]
            if not ds:
                continue
            mean = np.mean([d for d, _, _ in ds])
            se_m = np.std([d for d, _, _ in ds], ddof=1) / math.sqrt(len(ds))
            res = [(d, h) for d, s, h in ds if abs(d) > 2 * s]
            wins = sum(1 for d, _ in res if d > 0)
            print(f"   [{est}] n={len(ds)} meanΔ={mean:+.4f} ± {se_m:.4f}/end "
                  f"(t={mean/max(se_m,1e-9):.2f})  resolved {wins}/{len(res)} "
                  f"(p={_binom_p(wins, len(res)):.3f})")
            for h in sorted(set(h for _, _, h in ds)):
                hh = [d for d, _, hx in ds if hx == h]
                print(f"      h{h:02d}: n={len(hh)} meanΔ={np.mean(hh):+.3f}")


if args.aggregate:
    aggregate()
    sys.exit(0)

# --------------------------------------------------------------------------- #
import torch
import torch.nn as nn
from csas.search import load_policy

from world.config import model_cfg_from_dict
from world.eval.head_to_head import WorldPlayer, build_h2h_roots
from world.model import WorldModel
from world.preplaced import build_preplaced_h2h_roots
from world.search.beam import screen_tree_choose
from world.search.candidates import generate_candidates
from world.search.collect import _mc_rollout_terminal_batch, score_candidates_terminal
from world.search.noise import LocalNoise, make_noise
from world.train.trainer import load_world_checkpoint

device = torch.device(args.device)
env_bridge.warm_jax()
cfg_full: Config = load_config(args.config)
cfg = cfg_full.search
noise_path = cfg_full.csas_path(cfg.noise_config).as_posix()
NZ = make_noise(noise_path, seed=args.seed * 101 + args.shard_id)
policy, amean_t, astd_t = load_policy(args.policy, device)
amean_np = amean_t.detach().cpu().numpy().astype(np.float64)
astd_np = astd_t.detach().cpu().numpy().astype(np.float64)

# champion value head (in-rollout search + truncated leaf) + playout adjudicator
ck = torch.load(args.world, map_location=device, weights_only=False)
_wm = WorldModel(model_cfg_from_dict(ck["model_cfg"])).to(device)
load_world_checkpoint(_wm, args.world, map_location=device)
_wm.eval()


class _WorldValue(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x, c):
        return self.m.value_head.value(self.m.encode(x, c))


VMODEL = _WorldValue(_wm)
champion = WorldPlayer(args.world, device, name="playout",
                       noise=make_noise(noise_path, seed=97 + args.shard_id),
                       sel_noise_samples=2)
print(f"[exp056] shard {args.shard_id}/{args.num_shards} rules="
      f"{'NEW' if env_bridge.BOUNDARY_REMOVAL else 'OLD'}", flush=True)


def screen_arm(x, c, h, sie, cands, k_ego, n_search, trunc, rng):
    """Score the SHARED candidate pool with one rollout-estimator config; return
    the argmax-legal action + q."""
    persp = int(round(c[2]))
    q, _posts, illegal, _se = score_candidates_terminal(
        policy, amean_t, astd_t, x, c, cands, h, sie, persp, device, rng, NZ,
        cfg.rollout_temp, cfg.std_scale,
        value_model=(VMODEL if n_search > 1 else None), n_search=n_search,
        k_ego=k_ego, return_std=True, crn=True,
        max_steps=trunc, leaf_value_model=(VMODEL if trunc else None))
    qm = np.where(np.asarray(illegal, bool), -1e18, q)
    w = int(np.argmax(qm))
    return np.asarray(cands[w], np.float32), float(q[w])


ARMS = {
    "RT":  dict(k_ego=8, n_search=1, trunc=0),
    "RtT": dict(k_ego=8, n_search=1, trunc=args.trunc),
    "ST":  dict(k_ego=4, n_search=args.n_search, trunc=0),
    "StT": dict(k_ego=4, n_search=args.n_search, trunc=args.trunc),
}


def dn_dist(a, b):
    return float(np.linalg.norm((np.asarray(a, np.float64) - np.asarray(b, np.float64)) / NOISE_STD))


def paired_mc(x, c, hh, sie, A, B, k=64):
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
        seed = 560000 + 977 * t + 31 * args.shard_id
        ds.append(guided_playout(x, c, hh, sie, A, seed) - guided_playout(x, c, hh, sie, B, seed))
    ds = np.asarray(ds, np.float64)
    return {"delta": float(ds.mean()), "se": float(ds.std(ddof=1) / math.sqrt(len(ds))),
            "T": int(T)}


out_path = OUT / f"shard{args.shard_id}.jsonl"
done = set()
if out_path.exists():
    for l in out_path.read_text().splitlines():
        try:
            r = json.loads(l)
            done.add((r["h"], r["i"]))
        except Exception:
            pass

for h in [int(v) for v in args.horizons.split(",")]:
    roots = (build_preplaced_h2h_roots(h, args.roots_per_h, split="val", seed=args.seed + h)
             if h >= 10 else
             build_h2h_roots(cfg_full.paths.csas_v3_root, h, args.roots_per_h,
                             split="val", seed=args.seed + h))
    for i, root in enumerate(roots):
        if i % args.num_shards != args.shard_id or (h, i) in done:
            continue
        x, c = root.x.astype(np.float32), root.c.astype(np.float32)
        sie = int(root.shots_in_end)
        rng = np.random.default_rng(args.seed * 7919 + h * 131 + i)
        cands = np.asarray(generate_candidates(policy, amean_t, astd_t, x, c, cfg, rng, device),
                           np.float32)   # ONE shared proposal pool for all factorial arms
        rec = {"h": h, "i": i, "ops": {}, "dn": {}, "adj": {}}
        acts = {}
        ok = True
        for name, kw in ARMS.items():
            t0 = time.time()
            a, q = screen_arm(x, c, h, sie, cands, rng=rng, **kw)
            acts[name] = np.asarray(a, np.float64)
            rec["ops"][name] = {"action": [round(float(v), 6) for v in a], "q": round(q, 4),
                                "seconds": round(time.time() - t0, 1)}
        t0 = time.time()
        r2 = screen_tree_choose(policy, amean_t, astd_t, amean_np, astd_np,
                                x, c, h, sie, cfg, rng, device, NZ)
        if r2 is None:
            continue
        acts["record"] = np.asarray(r2["action"], np.float64)
        rec["ops"]["record"] = {"action": [round(float(v), 6) for v in r2["action"]],
                                "q": round(r2["q"], 4), "seconds": round(time.time() - t0, 1)}
        for a_name, b_name in MC_PAIRS:
            key = f"{a_name}_vs_{b_name}"
            dn = dn_dist(acts[a_name], acts[b_name])
            rec["dn"][key] = round(dn, 2)
            if dn <= args.dn_thresh:
                continue
            adj = {"mc": paired_mc(x, c, h, sie, acts[a_name].astype(np.float32),
                                   acts[b_name].astype(np.float32))}
            if (a_name, b_name) in PLAYOUT_PAIRS:
                adj["playout"] = paired_playouts(x, c, h, sie, acts[a_name].astype(np.float32),
                                                 acts[b_name].astype(np.float32), args.playouts)
            rec["adj"][key] = adj
        with out_path.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(f"[exp056] h{h} root {i}: " +
              " ".join(f"{k}={v['seconds']}s" for k, v in rec["ops"].items()), flush=True)

print("EXP056_SHARD_DONE", flush=True)
