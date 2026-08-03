#!/usr/bin/env python3
"""EXP-061: hot-prefix HEADROOM MAP — is there teachable signal OFF-distribution?

EXP-060 showed the policy is at its teacher's fixed point ON-distribution (excess
disagreement ~0) and that action-level comparisons are lotteries over a plateau.
The remaining distillation hope is off-distribution: states the policy never
visits in self-play. Mechanism (the user-approved (ii) design): HOT PREFIXES —
play the first P in {2,3,4} throws from a canonical pre-placed root with a
high-temperature policy under execution noise (states sound by construction),
then measure at the resulting state the VALUE GAP

    Δ = Q(teacher's choice) − Q(student's choice)

by paired terminal-MC (k=64, CRN), where teacher = the collection operator
(exp_037 screen_tree) and student = the champion's deployed selection (k=8).
Control stratum: the same measurement on sig-gated az_v19 corpus states
(on-distribution; expected Δ ≈ 0). Report by stratum + board-descriptor buckets
(coverage-control preview: stones in play / in house / guards).

  python scripts/exp061_headroom_map.py --shard-id K --num-shards 4
  python scripts/exp061_headroom_map.py --aggregate
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

NOISE_STD = np.array([0.0123, 0.003, 2.0, 0.015], dtype=np.float64)

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/exp_037_sig_screen_tree.yaml")
ap.add_argument("--policy", default="checkpoints/csas_world/az_v15_L8/incumbent0_policy_csas.pt")
ap.add_argument("--world", default="checkpoints/csas_world/az_v14d/best.pt")
ap.add_argument("--n-prefix", type=int, default=360, help="hot-prefix states (total)")
ap.add_argument("--n-control", type=int, default=120, help="on-distribution control states")
ap.add_argument("--hot-temp", type=float, default=2.5)
ap.add_argument("--hot-std", type=float, default=2.5)
ap.add_argument("--mc-k", type=int, default=64)
ap.add_argument("--device", default="cuda:0")
ap.add_argument("--out-dir", default="eval_out/exp061_headroom")
ap.add_argument("--num-shards", type=int, default=1)
ap.add_argument("--shard-id", type=int, default=0)
ap.add_argument("--seed", type=int, default=61)
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
    print(f"EXP-061 headroom map over {len(rows)} states\n")

    def stat(sub, label):
        if not sub:
            return
        d = np.array([r["delta"] for r in sub])
        m, se = d.mean(), d.std(ddof=1) / math.sqrt(len(d))
        print(f"  {label:34s} n={len(sub):4d}  Δ={m:+.4f} ± {se:.4f}/end (t={m/max(se,1e-9):+.2f})")

    stat([r for r in rows if r["stratum"] == "control"], "CONTROL (on-distribution sig plies)")
    pre = [r for r in rows if r["stratum"] == "prefix"]
    stat(pre, "HOT-PREFIX (off-distribution)")
    for p in sorted(set(r["P"] for r in pre)):
        stat([r for r in pre if r["P"] == p], f"  prefix length P={p} (h={10-p})")
    print("\n  by board descriptor (prefix states): stones-in-play / in-house / guards")
    for key in sorted(set((r["n_live"], r["n_house"], r["n_guard"]) for r in pre)):
        sub = [r for r in pre if (r["n_live"], r["n_house"], r["n_guard"]) == key]
        if len(sub) >= 8:
            stat(sub, f"  live={key[0]} house={key[1]} guard={key[2]}")


if args.aggregate:
    aggregate()
    sys.exit(0)

# --------------------------------------------------------------------------- #
import torch
from csas.common import raw_to_compact_m
from csas.search import _sample_actions, load_policy

from world.eval.head_to_head import WorldPlayer
from world.preplaced import MODES, board_norm
from world.search.beam import screen_tree_choose
from world.search.collect import _mc_rollout_terminal_batch
from world.search.noise import make_noise

device = torch.device(args.device)
env_bridge.warm_jax()
cfg_full: Config = load_config(args.config)
cfg = cfg_full.search
noise_path = cfg_full.csas_path(cfg.noise_config).as_posix()
NZ = make_noise(noise_path, seed=args.seed * 101 + args.shard_id)
policy, amean_t, astd_t = load_policy(args.policy, device)
amean_np = amean_t.detach().cpu().numpy().astype(np.float64)
astd_np = astd_t.detach().cpu().numpy().astype(np.float64)
student = WorldPlayer(args.world, device, name="student",
                      noise=make_noise(noise_path, seed=611 + args.shard_id),
                      sel_noise_samples=8)
print(f"[exp061] shard {args.shard_id}/{args.num_shards} rules="
      f"{'NEW' if env_bridge.BOUNDARY_REMOVAL else 'OLD'}", flush=True)


def descriptor(x):
    raw = np.asarray(x, np.float32).reshape(12, 2) * 4095.0
    from csas.common import in_play_raw
    live = in_play_raw(raw)
    cm = raw_to_compact_m(raw)
    n_live = int(live.sum())
    r = np.hypot(cm[:, 0], cm[:, 1])
    n_house = int((live & (r <= 1.974)).sum())
    n_guard = int((live & (cm[:, 0] < -1.974) & (np.abs(cm[:, 1]) < 2.375)).sum())
    return n_live, n_house, n_guard


def hot_prefix_state(rng):
    """Sound off-distribution state: P hot throws from a canonical root."""
    mode = MODES[rng.integers(0, len(MODES))]
    first_block = int(rng.integers(0, 2))
    guard_slot = 1 if first_block == 0 else 7
    x = board_norm(mode, guard_slot)
    c = np.asarray([0.0, 0.0, float(first_block)], dtype=np.float32)
    P = int(rng.integers(2, 5))
    h = 10
    for _ in range(P):
        cands = np.asarray(_sample_actions(policy, amean_t, astd_t, x, c, 8, device,
                                           args.hot_temp, args.hot_std, 0.0), np.float32)
        a = cands[rng.integers(0, len(cands))]                       # a hot SAMPLE, not argmax
        a = NZ.sample_batch(a[None], 1).reshape(4).astype(np.float32)
        post, _ = env_bridge.apply_legality(x, env_bridge.simulate_one(x, c, a)[None], h, c)
        x = post[0]
        c = env_bridge.next_condition(c, 10)
        h -= 1
    return x, c, h, P


def q_gap(x, c, h, A, B):
    """Paired terminal-MC Q(A) − Q(B), CRN executions."""
    persp = int(round(c[2]))
    nc = env_bridge.next_condition(c, 10)
    realized = NZ.sample_batch(np.stack([A, B]).astype(np.float32), args.mc_k,
                               crn=True).reshape(-1, 4)
    posts, _ = env_bridge.apply_legality(x, env_bridge.simulate(x, c, realized), h, c)
    rng = np.random.default_rng(1234)
    q = _mc_rollout_terminal_batch(policy, amean_t, astd_t, posts, nc, h - 1, 10,
                                   persp, device, rng, NZ, cfg.rollout_temp, cfg.std_scale,
                                   value_model=None, n_search=1).reshape(2, args.mc_k)
    d = q[0] - q[1]
    return float(d.mean()), float(d.std(ddof=1) / math.sqrt(args.mc_k))


# control states: sig-gated az_v19 corpus rows
CX, CC, CH = [], [], []
for f in sorted(glob.glob("artifacts/replay/az_v19_train/*.npz"))[:20]:
    z = np.load(f, allow_pickle=True)
    m = np.asarray(z["dist_mask"]) > 0
    CX.append(np.asarray(z["x0"])[m]); CC.append(np.asarray(z["c0"])[m])
    CH.append(np.asarray(z["horizon"])[m])
CX = np.concatenate(CX); CC = np.concatenate(CC); CH = np.concatenate(CH).astype(int)
rng0 = np.random.default_rng(args.seed)
ctrl_idx = rng0.choice(len(CX), size=min(args.n_control, len(CX)), replace=False)

out_path = OUT / f"shard{args.shard_id}.jsonl"
done = set()
if out_path.exists():
    done = {(json.loads(l)["stratum"], json.loads(l)["i"])
            for l in out_path.read_text().splitlines() if l.strip()}

jobs = [("control", int(i)) for i in ctrl_idx] + [("prefix", i) for i in range(args.n_prefix)]
for stratum, i in jobs:
    if i % args.num_shards != args.shard_id or (stratum, i) in done:
        continue
    rng = np.random.default_rng(args.seed * 7919 + i * 131 + (0 if stratum == "prefix" else 7))
    if stratum == "prefix":
        x, c, h, P = hot_prefix_state(rng)
    else:
        x, c, h, P = CX[i].astype(np.float32), CC[i].astype(np.float32), int(CH[i]), 0
    tr = screen_tree_choose(policy, amean_t, astd_t, amean_np, astd_np,
                            x, c, h, 10, cfg, rng, device, NZ)
    if tr is None:
        continue
    A = np.asarray(tr["action"], np.float32)
    B = np.asarray(student.select_intended(x, c, h, 10, int(round(c[2]))), np.float32)
    delta, se = q_gap(x, c, h, A, B)
    n_live, n_house, n_guard = descriptor(x)
    rec = {"stratum": stratum, "i": i, "h": h, "P": P, "delta": round(delta, 4),
           "se": round(se, 4), "n_live": n_live, "n_house": n_house, "n_guard": n_guard,
           "dn": round(float(np.linalg.norm((A - B) / NOISE_STD)), 2)}
    with out_path.open("a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(f"[exp061] {stratum} {i}: h{h} Δ={delta:+.3f}±{se:.3f}", flush=True)

print("EXP061_SHARD_DONE", flush=True)
