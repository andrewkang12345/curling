#!/usr/bin/env python3
"""EXP-055: WHY is simulator-tree depth flat? Two pre-registered diagnostics.

--mode weak  (055a, amortization hypothesis): rerun the EXP-053 primary contrast
  (d3 beam vs compute-matched d2) with the HUMAN PRIOR as the proposal/rollout
  policy inside both operators (adjudication still uses the deployed champion).
  If depth's edge appears with a weak policy, the null is explained: our
  champion-distilled rollouts already amortize what an extra ply would compute.

--mode sb    (055b, spine-bias hypothesis): stochastic-branching d3 — ply 2
  branches on 3 realized-post medoids (the opponent replies to boards that
  actually happen) — evaluated on EXP-054's exact roots and adjudicated against
  the STORED d2_hi actions (playouts, primary) and d3_hi actions (paired MC,
  secondary). If depth's edge jumps, the deterministic spine was eating it.

Primary readout per mode: mean adjudicated playout Δ (t-test) over all
disagreements + resolved-wins binomial, T=20 paired guided playouts.

  python scripts/exp055_depth_diagnosis.py --mode weak --shard-id K --num-shards 4
  python scripts/exp055_depth_diagnosis.py --mode sb   --shard-id K --num-shards 4
  python scripts/exp055_depth_diagnosis.py --mode weak --aggregate
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
ap.add_argument("--mode", choices=["weak", "sb"], required=True)
ap.add_argument("--config", default="configs/exp_037_sig_screen_tree.yaml")
ap.add_argument("--champion-policy", default="checkpoints/csas_world/az_v15_L8/incumbent0_policy_csas.pt")
ap.add_argument("--world", default="checkpoints/csas_world/az_v14d/best.pt")
ap.add_argument("--exp054-dir", default="eval_out/exp054_dose")
ap.add_argument("--device", default="cuda:0")
ap.add_argument("--num-shards", type=int, default=1)
ap.add_argument("--shard-id", type=int, default=0)
ap.add_argument("--playouts", type=int, default=20)
ap.add_argument("--dn-thresh", type=float, default=2.5)
ap.add_argument("--roots-per-h", type=int, default=32)   # weak mode
ap.add_argument("--horizons", default="4,6,8,10")        # weak mode
ap.add_argument("--seed", type=int, default=55)
ap.add_argument("--aggregate", action="store_true")
args = ap.parse_args()

OUT = Path(f"eval_out/exp055_{args.mode}")
OUT.mkdir(parents=True, exist_ok=True)


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
    print(f"EXP-055 ({args.mode}) aggregate over {len(rows)} plies")
    keys = sorted({k for r in rows for k in r.get("adj", {})})
    for key in keys:
        sub = [r for r in rows if key in r["adj"]]
        for est in ("playout", "mc"):
            ds = [(r["adj"][key][est]["delta"], r["adj"][key][est]["se"], r["h"])
                  for r in sub if r["adj"][key].get(est)]
            if not ds:
                continue
            mean = np.mean([d for d, _, _ in ds])
            se_m = np.std([d for d, _, _ in ds], ddof=1) / math.sqrt(len(ds))
            res = [(d, h) for d, s, h in ds if abs(d) > 2 * s]
            wins = sum(1 for d, _ in res if d > 0)
            print(f"  {key} [{est}]: n={len(ds)}  meanΔ={mean:+.4f} ± {se_m:.4f}/end "
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
from csas.search import load_policy

from world.eval.head_to_head import WorldPlayer, build_h2h_roots
from world.preplaced import build_preplaced_h2h_roots
from world.search.beam import screen_beam_choose, screen_tree_choose
from world.search.collect import _mc_rollout_terminal_batch
from world.search.noise import LocalNoise, make_noise

device = torch.device(args.device)
env_bridge.warm_jax()
cfg_full: Config = load_config(args.config)
cfg = cfg_full.search
noise_path = cfg_full.csas_path(cfg.noise_config).as_posix()
NZ = make_noise(noise_path, seed=args.seed * 101 + args.shard_id)
champion = WorldPlayer(args.world, device, name="playout",
                       noise=make_noise(noise_path, seed=97 + args.shard_id),
                       sel_noise_samples=2)

if args.mode == "weak":
    pol_path = cfg_full.csas_path(cfg_full.paths.prior_policy_ckpt).as_posix()
else:
    pol_path = args.champion_policy
policy, amean_t, astd_t = load_policy(pol_path, device)
amean_np = amean_t.detach().cpu().numpy().astype(np.float64)
astd_np = astd_t.detach().cpu().numpy().astype(np.float64)
print(f"[exp055-{args.mode}] shard {args.shard_id}/{args.num_shards} "
      f"policy={pol_path} rules={'NEW' if env_bridge.BOUNDARY_REMOVAL else 'OLD'}", flush=True)


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
        seed = 550000 + 977 * t + 31 * args.shard_id
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


def emit(rec):
    with out_path.open("a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(f"[exp055-{args.mode}] h{rec['h']} i{rec['i']} " +
          " ".join(f"{k}={v['seconds']}s" for k, v in rec["ops"].items()), flush=True)


if args.mode == "weak":
    # EXP-053 primary contrast, prior-policy edition: d3 beam vs compute-matched d2
    cfg_p = copy.deepcopy(cfg)
    cfg_p.noise_samples = 48
    cfg_p.mcts_sims = 128
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
            rec = {"h": h, "i": i, "ops": {}, "dn": {}, "adj": {}}
            t0 = time.time()
            r2 = screen_tree_choose(policy, amean_t, astd_t, amean_np, astd_np,
                                    x, c, h, sie, cfg_p, rng, device, NZ)
            if r2 is None:
                continue
            rec["ops"]["d2p_w"] = {"action": [round(float(v), 6) for v in r2["action"]],
                                   "seconds": round(time.time() - t0, 1)}
            t0 = time.time()
            r3 = screen_beam_choose(policy, amean_t, astd_t, amean_np, astd_np,
                                    x, c, h, sie, cfg, rng, device, NZ)
            if r3 is None:
                continue
            rec["ops"]["d3_w"] = {"action": [round(float(v), 6) for v in r3["action"]],
                                  "seconds": round(time.time() - t0, 1)}
            A, B = np.asarray(r3["action"], np.float32), np.asarray(r2["action"], np.float32)
            dn = dn_dist(A, B)
            rec["dn"]["d3_w_vs_d2p_w"] = round(dn, 2)
            if dn > args.dn_thresh:
                rec["adj"]["d3_w_vs_d2p_w"] = {
                    "mc": paired_mc(x, c, h, sie, A, B),
                    "playout": paired_playouts(x, c, h, sie, A, B, args.playouts)}
            emit(rec)
else:
    # sb mode: recompute d3 with stochastic branching on EXP-054's roots; compare to
    # the STORED d2_hi (playouts, primary) and d3_hi (paired MC, secondary) actions.
    rows54 = []
    for f in sorted(Path(args.exp054_dir).glob("shard*.jsonl")):
        rows54 += [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    roots_by_h = {}
    for h in sorted({r["h"] for r in rows54}):
        roots_by_h[h] = (build_preplaced_h2h_roots(h, 32, split="val", seed=54 + h)
                         if h >= 10 else
                         build_h2h_roots(cfg_full.paths.csas_v3_root, h, 32,
                                         split="val", seed=54 + h))
    for r54 in rows54:
        h, i = r54["h"], r54["i"]
        if i % args.num_shards != args.shard_id or (h, i) in done:
            continue
        root = roots_by_h[h][i]
        x, c = root.x.astype(np.float32), root.c.astype(np.float32)
        sie = int(root.shots_in_end)
        rng = np.random.default_rng(args.seed * 7919 + h * 131 + i)
        rec = {"h": h, "i": i, "ops": {}, "dn": {}, "adj": {}}
        t0 = time.time()
        r3 = screen_beam_choose(policy, amean_t, astd_t, amean_np, astd_np,
                                x, c, h, sie, cfg, rng, device, NZ, branch_reps=3)
        if r3 is None:
            continue
        A = np.asarray(r3["action"], np.float32)
        rec["ops"]["d3_sb"] = {"action": [round(float(v), 6) for v in A],
                               "seconds": round(time.time() - t0, 1)}
        for other, est in (("d2_hi", "playout"), ("d3_hi", "mc")):
            B = np.asarray(r54["ops"][other]["action"], np.float32)
            key = f"d3_sb_vs_{other}"
            dn = dn_dist(A, B)
            rec["dn"][key] = round(dn, 2)
            if dn > args.dn_thresh:
                adj = {"mc": paired_mc(x, c, h, sie, A, B)}
                if est == "playout":
                    adj["playout"] = paired_playouts(x, c, h, sie, A, B, args.playouts)
                rec["adj"][key] = adj
        emit(rec)

print(f"EXP055_{args.mode.upper()}_SHARD_DONE", flush=True)
