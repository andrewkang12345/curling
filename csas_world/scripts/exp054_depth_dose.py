#!/usr/bin/env python3
"""EXP-054: depth x compute DOSE-RESPONSE (the clean version of EXP-053).

Four operators choose on identical root plies (fresh val roots, new rules):
  d2_lo : screen_tree, exp_037 knobs           (ke 8,  48 tree sims)   ~60s/ply
  d3_lo : screen-beam, EXP-053 knobs           (ke 8,  beams 6/3)      ~120s/ply
  d2_hi : screen_tree, 4x budget               (ke 32, 192 tree sims)  ~320s/ply (a priori)
  d3_hi : screen-beam, ~wall-clock-matched     (ke 12, beams 8/4, opp 20, mine 16)

Adjudication (paired, CRN first throws):
  playouts T=16 (primary, deployed champion moves both sides): d3_hi vs d2_hi
    [PRIMARY, single look, binom p<0.05] and d3_lo vs d2_lo [EXP-053 replication]
  paired MC k=64 (secondary): those two + the within-depth dose contrasts
    d2_hi vs d2_lo, d3_hi vs d3_lo.

Pre-registered readouts: (1) PRIMARY d3_hi>d2_hi on playout-resolved wins (p<0.05);
(2) dose-response within depth (hi beats lo per MC); (3) interaction: does the
depth gap grow from lo to hi? Wall-clock recorded per operator for honest budget
accounting.

  python scripts/exp054_depth_dose.py --shard-id K --num-shards 4 --device cuda:0
  python scripts/exp054_depth_dose.py --aggregate
"""
import argparse
import copy
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
ap.add_argument("--out-dir", default="eval_out/exp054_dose")
ap.add_argument("--num-shards", type=int, default=1)
ap.add_argument("--shard-id", type=int, default=0)
ap.add_argument("--playouts", type=int, default=16)
ap.add_argument("--dn-thresh", type=float, default=2.5)
ap.add_argument("--seed", type=int, default=54)
ap.add_argument("--aggregate", action="store_true")
args = ap.parse_args()

OUT = Path(args.out_dir)
OUT.mkdir(parents=True, exist_ok=True)

PLAYOUT_PAIRS = (("d3_hi", "d2_hi"), ("d3_lo", "d2_lo"))
MC_PAIRS = PLAYOUT_PAIRS + (("d2_hi", "d2_lo"), ("d3_hi", "d3_lo"))


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
    print(f"EXP-054 aggregate over {len(rows)} plies\n")
    for op in ("d2_lo", "d3_lo", "d2_hi", "d3_hi"):
        ts = [r["ops"][op]["seconds"] for r in rows if op in r["ops"]]
        if ts:
            print(f"  {op:6s} mean wall-clock {np.mean(ts):6.1f}s / ply")
    for a, b in MC_PAIRS:
        key = f"{a}_vs_{b}"
        sub = [r for r in rows if key in r.get("adj", {})]
        n_dis = sum(1 for r in rows if r["dn"].get(key, 0.0) > args.dn_thresh)
        print(f"\n== {key}: disagreements {n_dis}/{len(rows)}")
        mc = [(r["adj"][key]["mc"]["delta"], r["adj"][key]["mc"]["se"]) for r in sub
              if r["adj"][key].get("mc")]
        if mc:
            res = [d for d, s in mc if abs(d) > 2 * s]
            wins = sum(1 for d in res if d > 0)
            print(f"   MC(k=64):    {len(mc)} adjudicated, {len(res)} resolved, "
                  f"{a} wins {wins}/{len(res)} (p={_binom_p(wins, len(res)):.3f}) "
                  f"meanΔ={np.mean([d for d,_ in mc]):+.3f}")
        po = [(r["adj"][key]["playout"]["delta"], r["adj"][key]["playout"]["se"], r["h"])
              for r in sub if r["adj"][key].get("playout")]
        if po:
            res = [(d, h) for d, s, h in po if abs(d) > 2 * s]
            wins = sum(1 for d, _ in res if d > 0)
            print(f"   PLAYOUTS(T={args.playouts}): {len(po)} adjudicated, {len(res)} resolved, "
                  f"{a} wins {wins}/{len(res)} (p={_binom_p(wins, len(res)):.4f}) "
                  f"meanΔ={np.mean([d for d,_,_ in po]):+.3f}/end")
            for h in sorted(set(h for _, h in res)):
                hh = [d for d, hx in res if hx == h]
                print(f"      h{h:02d}: {sum(1 for d in hh if d>0)}/{len(hh)} resolved wins")


if args.aggregate:
    aggregate()
    sys.exit(0)

# --------------------------------------------------------------------------- #
import torch
from csas.search import load_policy

from world.eval.head_to_head import WorldPlayer, build_h2h_roots
from world.preplaced import build_preplaced_h2h_roots
from world.search.beam import screen_beam_choose, screen_tree_choose
from world.search.collect import _mc_rollout_terminal_batch
from world.search.noise import LocalNoise, make_noise

device = torch.device(args.device)
env_bridge.warm_jax()
cfg_full: Config = load_config(args.config)
cfg_lo = cfg_full.search                       # d2_lo knobs (exp_037)
cfg_hi = copy.deepcopy(cfg_lo)
cfg_hi.noise_samples = 32
cfg_hi.mcts_sims = 192

policy, amean_t, astd_t = load_policy(args.policy, device)
amean_np = amean_t.detach().cpu().numpy().astype(np.float64)
astd_np = astd_t.detach().cpu().numpy().astype(np.float64)
noise_path = cfg_full.csas_path(cfg_lo.noise_config).as_posix()
NZ = make_noise(noise_path, seed=args.seed * 101 + args.shard_id)
champion = WorldPlayer(args.world, device, name="playout",
                       noise=make_noise(noise_path, seed=97 + args.shard_id),
                       sel_noise_samples=2)
print(f"[exp054] shard {args.shard_id}/{args.num_shards} rules="
      f"{'NEW' if env_bridge.BOUNDARY_REMOVAL else 'OLD'}", flush=True)


def dn_dist(a, b):
    return float(np.linalg.norm((np.asarray(a, np.float64) - np.asarray(b, np.float64)) / NOISE_STD))


def paired_mc(x, c, hh, sie, A, B, k=64):
    persp = int(round(c[2]))
    nc = env_bridge.next_condition(c, sie)
    realized = NZ.sample_batch(np.stack([A, B]).astype(np.float32), k, crn=True).reshape(-1, 4)
    posts, _ = env_bridge.apply_legality(x, env_bridge.simulate(x, c, realized), hh, c)
    rng = np.random.default_rng(1234)
    q = _mc_rollout_terminal_batch(policy, amean_t, astd_t, posts, nc, hh - 1, sie,
                                   persp, device, rng, NZ, cfg_lo.rollout_temp, cfg_lo.std_scale,
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
        seed = 540000 + 977 * t + 31 * args.shard_id
        ds.append(guided_playout(x, c, hh, sie, A, seed) - guided_playout(x, c, hh, sie, B, seed))
    ds = np.asarray(ds, np.float64)
    return {"delta": float(ds.mean()), "se": float(ds.std(ddof=1) / math.sqrt(len(ds))),
            "T": int(T)}


OPS = {
    "d2_lo": lambda x, c, h, sie, rng: screen_tree_choose(
        policy, amean_t, astd_t, amean_np, astd_np, x, c, h, sie, cfg_lo, rng, device, NZ),
    "d2_hi": lambda x, c, h, sie, rng: screen_tree_choose(
        policy, amean_t, astd_t, amean_np, astd_np, x, c, h, sie, cfg_hi, rng, device, NZ),
    "d3_lo": lambda x, c, h, sie, rng: screen_beam_choose(
        policy, amean_t, astd_t, amean_np, astd_np, x, c, h, sie, cfg_lo, rng, device, NZ),
    "d3_hi": lambda x, c, h, sie, rng: screen_beam_choose(
        policy, amean_t, astd_t, amean_np, astd_np, x, c, h, sie, cfg_lo, rng, device, NZ,
        beam_root=8, opp_cands=20, beam_opp=4, my_cands=16, k_ego=12),
}

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
        rng = np.random.default_rng(args.seed * 7919 + h * 131 + i)
        rec = {"h": h, "i": i, "ops": {}, "dn": {}, "adj": {}}
        acts = {}
        ok = True
        for name, fn in OPS.items():
            t0 = time.time()
            res = fn(x, c, h, sie, rng)
            if res is None:
                ok = False
                break
            acts[name] = np.asarray(res["action"], np.float64)
            rec["ops"][name] = {"action": [round(float(v), 6) for v in res["action"]],
                                "q": round(res["q"], 4),
                                "seconds": round(time.time() - t0, 1)}
        if not ok:
            continue
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
        print(f"[exp054] h{h} root {i}: " +
              " ".join(f"{k}={v['seconds']}s" for k, v in rec["ops"].items()), flush=True)

print("EXP054_SHARD_DONE", flush=True)
