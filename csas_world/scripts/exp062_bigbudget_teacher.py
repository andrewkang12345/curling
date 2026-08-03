#!/usr/bin/env python3
"""EXP-062: big-budget SELF-distillation teacher verification.

EXP-061 showed the screen_tree teacher is BEHIND the deployed selection (winner's
curse on 8-sample MC argmax). The proposed replacement teacher is the student's
own decision rule at a larger budget — more candidates x more noise samples,
value-head ranked (no MC curse; value-head bias shared with the student cancels
in the contrast). This experiment verifies per-decision superiority BEFORE any
collection cycle:

  S   = deployed selection, 48 cands x k=8      (the student)
  S'  = an independent second student draw       (null anchor: Δ ≈ 0 expected)
  T1  = 96 cands x k=32                          (~8x budget)
  T2  = 192 cands x k=64                         (~32x budget)

Per state (same distribution as EXP-061: on-distribution sig plies + hot-prefix
states): Δ(X − S) by paired terminal-MC k=64 CRN; guided playouts T=16 on a
subsample of T2-vs-S as confirmation. Pre-registered: T1 > S and T2 >= T1
(budget dose-response) certifies the teacher; then one collection+retrain cycle.

  python scripts/exp062_bigbudget_teacher.py --shard-id K --num-shards 4
  python scripts/exp062_bigbudget_teacher.py --aggregate
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
from world.config import Config, load_config

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/exp_037_sig_screen_tree.yaml")
ap.add_argument("--policy", default="checkpoints/csas_world/az_v15_L8/incumbent0_policy_csas.pt")
ap.add_argument("--world", default="checkpoints/csas_world/az_v14d/best.pt")
ap.add_argument("--n-prefix", type=int, default=160)
ap.add_argument("--n-control", type=int, default=80)
ap.add_argument("--playout-every", type=int, default=4, help="1/N states get playout confirm")
ap.add_argument("--playouts", type=int, default=16)
ap.add_argument("--mc-k", type=int, default=64)
ap.add_argument("--device", default="cuda:0")
ap.add_argument("--out-dir", default="eval_out/exp062_teacher")
ap.add_argument("--num-shards", type=int, default=4)
ap.add_argument("--shard-id", type=int, default=0)
ap.add_argument("--seed", type=int, default=62)
ap.add_argument("--aggregate", action="store_true")
args = ap.parse_args()

OUT = Path(args.out_dir)
OUT.mkdir(parents=True, exist_ok=True)
CONTRASTS = ("S2", "T1", "T2")


def aggregate():
    rows = []
    for f in sorted(OUT.glob("shard*.jsonl")):
        rows += [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    if not rows:
        print("no results yet")
        return
    print(f"EXP-062 big-budget teacher verification over {len(rows)} states\n")
    for name, label in (("S2", "S' second student draw (null anchor)"),
                        ("T1", "T1 = 96 cands x k=32 (~8x budget)"),
                        ("T2", "T2 = 192 cands x k=64 (~32x budget)")):
        for strat in ("all", "control", "prefix"):
            sub = [r for r in rows if f"d_{name}" in r and (strat == "all" or r["stratum"] == strat)]
            if not sub:
                continue
            d = np.array([r[f"d_{name}"] for r in sub])
            m, se = d.mean(), d.std(ddof=1) / math.sqrt(len(d))
            tag = f"Δ({name}−S) [{strat}]"
            print(f"  {tag:26s} n={len(sub):4d}  {m:+.4f} ± {se:.4f}/end (t={m/max(se,1e-9):+.2f})"
                  + (f"   <- {label}" if strat == "all" else ""))
        print()
    po = [r["po_T2"] for r in rows if r.get("po_T2") is not None]
    if po:
        d = np.array(po)
        m, se = d.mean(), d.std(ddof=1) / math.sqrt(len(d))
        print(f"  PLAYOUT confirm Δ(T2−S): n={len(d)}  {m:+.4f} ± {se:.4f}/end (t={m/max(se,1e-9):+.2f})")


if args.aggregate:
    aggregate()
    sys.exit(0)

# --------------------------------------------------------------------------- #
import torch
from csas.search import _sample_actions, load_policy

from world.eval.head_to_head import WorldPlayer
from world.preplaced import MODES, board_norm
from world.search.collect import _mc_rollout_terminal_batch
from world.search.noise import LocalNoise, make_noise

device = torch.device(args.device)
env_bridge.warm_jax()
cfg_full: Config = load_config(args.config)
cfg = cfg_full.search
noise_path = cfg_full.csas_path(cfg.noise_config).as_posix()
NZ = make_noise(noise_path, seed=args.seed * 101 + args.shard_id)
policy, amean_t, astd_t = load_policy(args.policy, device)

player = WorldPlayer(args.world, device, name="sel",
                     noise=make_noise(noise_path, seed=621 + args.shard_id),
                     sel_noise_samples=8)
NOISES = {k: make_noise(noise_path, seed=s + 31 * args.shard_id)
          for k, s in (("S", 1), ("S2", 2), ("T1", 3), ("T2", 4))}
BUDGETS = {"S": (48, 8), "S2": (48, 8), "T1": (96, 32), "T2": (192, 64)}


def select(tag, x, c, h):
    n, k = BUDGETS[tag]
    player.n, player.sel_noise_samples, player.noise = n, k, NOISES[tag]
    return np.asarray(player.select_intended(x, c, h, 10, int(round(c[2]))), np.float32)


def hot_prefix_state(rng):
    mode = MODES[rng.integers(0, len(MODES))]
    first_block = int(rng.integers(0, 2))
    x = board_norm(mode, 1 if first_block == 0 else 7)
    c = np.asarray([0.0, 0.0, float(first_block)], dtype=np.float32)
    P = int(rng.integers(2, 5))
    h = 10
    for _ in range(P):
        cands = np.asarray(_sample_actions(policy, amean_t, astd_t, x, c, 8, device,
                                           2.5, 2.5, 0.0), np.float32)
        a = NZ.sample_batch(cands[rng.integers(0, len(cands))][None], 1).reshape(4).astype(np.float32)
        post, _ = env_bridge.apply_legality(x, env_bridge.simulate_one(x, c, a)[None], h, c)
        x, c, h = post[0], env_bridge.next_condition(c, 10), h - 1
    return x, c, h


def q_gap(x, c, h, A, B):
    persp = int(round(c[2]))
    nc = env_bridge.next_condition(c, 10)
    realized = NZ.sample_batch(np.stack([A, B]).astype(np.float32), args.mc_k, crn=True).reshape(-1, 4)
    posts, _ = env_bridge.apply_legality(x, env_bridge.simulate(x, c, realized), h, c)
    rng = np.random.default_rng(1234)
    q = _mc_rollout_terminal_batch(policy, amean_t, astd_t, posts, nc, h - 1, 10,
                                   persp, device, rng, NZ, cfg.rollout_temp, cfg.std_scale,
                                   value_model=None, n_search=1).reshape(2, args.mc_k)
    return float((q[0] - q[1]).mean())


def guided_playout_gap(x, c, h, A, B, T):
    def run(first, seed):
        persp = int(round(c[2]))
        nz = LocalNoise(noise_path, seed=seed)
        st, cc, hh = x.copy(), c.copy(), h
        a = np.asarray(first, np.float32)
        while hh >= 1:
            realized = nz.sample_batch(a[None], 1).reshape(4).astype(np.float32)
            post, _ = env_bridge.apply_legality(st, env_bridge.simulate_one(st, cc, realized)[None], hh, cc)
            st, cc, hh = post[0], env_bridge.next_condition(cc, 10), hh - 1
            if hh >= 1:
                a = select("S", st, cc, hh)
        return float(env_bridge.score_end(st, persp))
    ds = [run(A, 620000 + 977 * t) - run(B, 620000 + 977 * t) for t in range(T)]
    return float(np.mean(ds))


# control states (az_v19 sig plies)
CX, CC, CH = [], [], []
for f in sorted(glob.glob("artifacts/replay/az_v19_train/*.npz"))[:20]:
    z = np.load(f, allow_pickle=True)
    m = np.asarray(z["dist_mask"]) > 0
    CX.append(np.asarray(z["x0"])[m]); CC.append(np.asarray(z["c0"])[m]); CH.append(np.asarray(z["horizon"])[m])
CX = np.concatenate(CX); CC = np.concatenate(CC); CH = np.concatenate(CH).astype(int)
ctrl_idx = np.random.default_rng(args.seed).choice(len(CX), size=min(args.n_control, len(CX)), replace=False)

out_path = OUT / f"shard{args.shard_id}.jsonl"
done = set()
if out_path.exists():
    done = {(json.loads(l)["stratum"], json.loads(l)["i"]) for l in out_path.read_text().splitlines() if l.strip()}

print(f"[exp062] shard {args.shard_id}/{args.num_shards} rules="
      f"{'NEW' if env_bridge.BOUNDARY_REMOVAL else 'OLD'}", flush=True)
jobs = [("control", int(i)) for i in ctrl_idx] + [("prefix", i) for i in range(args.n_prefix)]
for stratum, i in jobs:
    if i % args.num_shards != args.shard_id or (stratum, i) in done:
        continue
    rng = np.random.default_rng(args.seed * 7919 + i * 131 + (0 if stratum == "prefix" else 7))
    if stratum == "prefix":
        x, c, h = hot_prefix_state(rng)
    else:
        x, c, h = CX[i].astype(np.float32), CC[i].astype(np.float32), int(CH[i])
    S = select("S", x, c, h)
    acts = {k: select(k, x, c, h) for k in CONTRASTS}
    rec = {"stratum": stratum, "i": i, "h": h}
    for k in CONTRASTS:
        rec[f"d_{k}"] = round(q_gap(x, c, h, acts[k], S), 4)
    if i % args.playout_every == 0:
        rec["po_T2"] = round(guided_playout_gap(x, c, h, acts["T2"], S, args.playouts), 4)
    with out_path.open("a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(f"[exp062] {stratum} {i}: h{h} " +
          " ".join(f"{k}={rec[f'd_{k}']:+.3f}" for k in CONTRASTS), flush=True)

print("EXP062_SHARD_DONE", flush=True)
