#!/usr/bin/env python3
"""EXP-068: 4-ply VECTORISED tree at h=4 and h=10 (deep-search validation).

EXP-066 validated a correctly-structured tree at h=2 (monotone held-out regret,
beats flat robust width at >=16k sims). Two things were left open: it was only
2 plies, and the sequential implementation cost ~172 min/state at 64k. This
experiment uses the wave-batched `VecTree` (src/world/search/vec_tree.py) to run
**4-ply search** at:

  * h=4  — 4 plies reaches TERMINAL, so leaves are exact rule scores (no value
           head anywhere: a clean deep-search correctness test);
  * h=10 — 4 plies then rules-grounded lockstep ROLLOUTS to terminal at the cap
           (still no value head in the tree).

Budgets: 1k / 4k / 16k simulator calls (user-specified; 64k is used only for the
reference searcher, which is a yardstick, not an arm).

Regret at h>2 cannot use exact expectimax (only h=2 admits it), so it is
HELD-OUT and adjudicated the way every certified number in this project is:
paired guided playouts with the CHAMPION continuation under CRN, over the union
of every arm's chosen actions plus the 64k reference choice. Regret(arm,B) =
best-in-set - chosen, both from that fresh evaluation.

  phases: states -> search -> adjudicate -> aggregate   (per --horizon)
  plus:   validate_h2  (vec_tree on EXP-066's states, scored against its
                        128-action reference table -> does vectorisation
                        reproduce the sequential tree's regret curve?)
"""
import argparse
import glob
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")
import world  # noqa: F401
from world import env_bridge
from world.config import Config, load_config

ap = argparse.ArgumentParser()
ap.add_argument("--phase", required=True,
                choices=["states", "search", "adjudicate", "aggregate", "validate_h2"])
ap.add_argument("--horizon", type=int, default=4)
ap.add_argument("--config", default="configs/exp_037_sig_screen_tree.yaml")
ap.add_argument("--policy", default="checkpoints/csas_world/az_v15_L8/incumbent0_policy_csas.pt")
ap.add_argument("--world", default="checkpoints/csas_world/az_v25_br/best.pt")
ap.add_argument("--n-states", type=int, default=30)
ap.add_argument("--budgets", default="1000,4000,16000")
ap.add_argument("--ref-budget", type=int, default=64000)
ap.add_argument("--playouts", type=int, default=64)
ap.add_argument("--max-depth", type=int, default=4)
ap.add_argument("--wave", type=int, default=32)
ap.add_argument("--device", default="cuda:0")
ap.add_argument("--out-dir", default="eval_out/exp068_deep")
ap.add_argument("--num-shards", type=int, default=1)
ap.add_argument("--shard-id", type=int, default=0)
ap.add_argument("--seed", type=int, default=68)
args = ap.parse_args()

OUT = Path(args.out_dir) / f"h{args.horizon:02d}"
OUT.mkdir(parents=True, exist_ok=True)
BUDGETS = [int(b) for b in args.budgets.split(",")]
POOL_POLICY, POOL_EXTRA = 96, 32


# ------------------------------------------------------------------ aggregate #
def aggregate():
    rows, adj = [], {}
    for f in OUT.glob("search_shard*.jsonl"):
        rows += [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    for f in OUT.glob("adj_shard*.jsonl"):
        for l in f.read_text().splitlines():
            r = json.loads(l)
            adj[r["sid"]] = r
    if not rows or not adj:
        print(f"incomplete: {len(rows)} search rows, {len(adj)} adjudicated states")
        return
    print(f"EXP-068 h={args.horizon} ({args.max_depth}-ply): {len(adj)} states adjudicated, "
          f"{len(rows)} arm rows")
    print(f"held-out: paired champion-continuation playouts T={args.playouts} (CRN)\n")
    for arm in sorted(set(r["arm"] for r in rows if r["arm"] != "ref")):
        line = []
        for B in BUDGETS:
            rs = []
            for r in rows:
                if r["arm"] != arm or str(B) not in r["chosen_idx"]:
                    continue
                a = adj.get(r["sid"])
                ci = str(r["chosen_idx"][str(B)])
                if a and ci in a["held"]:
                    rs.append(float(max(a["held"].values()) - a["held"][ci]))
            if rs:
                line.append(f"B={B//1000}k: {np.mean(rs):+.4f}±{np.std(rs)/math.sqrt(len(rs)):.4f}")
        print(f"  {arm:12s} " + "   ".join(line))
    rs = []
    for r in rows:
        if r["arm"] != "ref":
            continue
        a = adj.get(r["sid"])
        ci = str(r["chosen_idx"][str(args.ref_budget)])
        if a and ci in a["held"]:
            rs.append(float(max(a["held"].values()) - a["held"][ci]))
    if rs:
        print(f"  {'ref(64k)':12s} {np.mean(rs):+.4f}±{np.std(rs)/math.sqrt(len(rs)):.4f}  "
              f"(yardstick, not an arm)")
    print("\nvalidation: AGGREGATE regret must fall with budget (within SEs);")
    print("tree beating flat_width at 16k reproduces the EXP-066 finding at depth.")


if args.phase == "aggregate":
    aggregate()
    sys.exit(0)

# ------------------------------------------------------------------ shared    #
import torch
import torch.nn as nn
from csas.search import _sample_actions, _sample_actions_batch, load_policy

from world.config import model_cfg_from_dict
from world.eval.head_to_head import WorldPlayer
from world.model import WorldModel
from world.preplaced import MODES, board_norm
from world.search.candidates import generate_candidates
from world.search.collect import _mc_rollout_terminal_batch
from world.search.noise import LocalNoise, make_noise
from world.search.vec_tree import VecTree
from world.train.trainer import load_world_checkpoint

device = torch.device(args.device if torch.cuda.is_available() else "cpu")
env_bridge.warm_jax()
cfg_full: Config = load_config(args.config)
cfg = cfg_full.search
noise_path = cfg_full.csas_path(cfg.noise_config).as_posix()
NZ = make_noise(noise_path, seed=args.seed * 101 + args.shard_id)
policy, amean_t, astd_t = load_policy(args.policy, device)

ck = torch.load(args.world, map_location=device, weights_only=False)
_wm = WorldModel(model_cfg_from_dict(ck["model_cfg"])).to(device)
load_world_checkpoint(_wm, args.world, map_location=device)
_wm.eval()


class _V(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x, c):
        return self.m.value_head.value(self.m.encode(x, c))


VMODEL = _V(_wm)
STATES_NPZ = OUT / "states.npz"
SIE = 10


def sample_batch_fn(states, cond, n):
    cb = np.broadcast_to(np.asarray(cond, np.float32), (len(states), 3)).astype(np.float32)
    out = _sample_actions_batch(policy, amean_t, astd_t, np.asarray(states, np.float32), cb,
                                n, device, 1.1, 1.2, 0.0)
    return np.asarray(out, np.float32).reshape(len(states), n, 4)


def rollout_batch_fn(states, cond, h, persp):
    """Lockstep raw-policy rollout to terminal; rules-scored (no value head)."""
    rng = np.random.default_rng(int(args.seed) + 13)
    return _mc_rollout_terminal_batch(policy, amean_t, astd_t, np.asarray(states, np.float32),
                                      np.asarray(cond, np.float32), int(h), SIE, int(persp),
                                      device, rng, NZ, cfg.rollout_temp, cfg.std_scale)


# ------------------------------------------------------------------ states    #
if args.phase == "states":
    rng = np.random.default_rng(args.seed + args.horizon)
    xs, cs, pools = [], [], []
    for i in range(args.n_states):
        mode = MODES[rng.integers(0, len(MODES))]
        fb = int(rng.integers(0, 2))
        x = board_norm(mode, 1 if fb == 0 else 7)
        c = np.asarray([0.0, 0.0, float(fb)], np.float32)
        h = 10
        while h > args.horizon:                     # hot-prefix burn-in (sound by construction)
            cand = np.asarray(_sample_actions(policy, amean_t, astd_t, x, c, 8, device,
                                              2.0, 2.0, 0.0), np.float32)
            a = NZ.sample_batch(cand[rng.integers(0, len(cand))][None], 1).reshape(4).astype(np.float32)
            post, _ = env_bridge.apply_legality(x, env_bridge.simulate_one(x, c, a)[None], h, c)
            x, c, h = post[0], env_bridge.next_condition(c, SIE), h - 1
        pol = np.asarray(_sample_actions(policy, amean_t, astd_t, x, c, POOL_POLICY, device,
                                         1.1, 1.2, 0.0), np.float32)
        dense = np.asarray(generate_candidates(policy, amean_t, astd_t, x, c, cfg, rng, device),
                           np.float32)
        extra = dense[rng.choice(len(dense), size=min(POOL_EXTRA, len(dense)), replace=False)]
        xs.append(x); cs.append(c)
        pools.append(np.concatenate([pol, extra])[: POOL_POLICY + POOL_EXTRA])
        print(f"[states] h{args.horizon} {i}", flush=True)
    np.savez_compressed(STATES_NPZ, x=np.stack(xs), c=np.stack(cs), pool=np.stack(pools))
    print(f"[states] wrote {len(xs)} states -> {STATES_NPZ}")
    sys.exit(0)


# ------------------------------------------------------------------ h2 check  #
if args.phase == "validate_h2":
    """Does the vectorised tree reproduce the sequential tree's h=2 regret curve?
    Scored against EXP-066's 128-action expectimax reference table."""
    Z = np.load("eval_out/exp066_search/states.npz", allow_pickle=True)
    ref = {}
    for f in Path("eval_out/exp066_search").glob("ref_shard*.jsonl"):
        for l in f.read_text().splitlines():
            r = json.loads(l)
            ref[r["sid"]] = np.asarray(r["ref"])
    out_path = Path(args.out_dir) / f"validate_h2_shard{args.shard_id}.jsonl"
    done = {json.loads(l)["sid"] for l in out_path.read_text().splitlines()} if out_path.exists() else set()
    for sid in range(min(args.n_states, len(Z["x"]))):
        if sid % args.num_shards != args.shard_id or sid in done or sid not in ref:
            continue
        x, c = Z["x"][sid].astype(np.float32), Z["c"][sid].astype(np.float32)
        pool = Z["pool"][sid].astype(np.float32)
        prior = np.concatenate([np.full(POOL_POLICY, 0.8 / POOL_POLICY),
                                np.full(len(pool) - POOL_POLICY, 0.2 / max(len(pool) - POOL_POLICY, 1))])
        tree = VecTree(x, c, 2, SIE, pool, prior, sample_batch_fn=sample_batch_fn,
                       rollout_batch_fn=rollout_batch_fn, noise=NZ,
                       rng=np.random.default_rng(args.seed + sid), max_depth=2,
                       wave=args.wave)
        picks = tree.run(BUDGETS)
        idx = {str(B): int(np.argmin(np.linalg.norm(
            (pool - a) / np.array([0.0123, 0.003, 2.0, 0.015], np.float32), axis=1)))
            for B, a in picks.items()}
        reg = {B: float(ref[sid].max() - ref[sid][idx[str(B)]]) for B in BUDGETS}
        with out_path.open("a") as fh:
            fh.write(json.dumps({"sid": sid, "chosen_idx": idx,
                                 "regret": {str(k): round(v, 4) for k, v in reg.items()}}) + "\n")
        print(f"[val_h2] sid {sid}: " + " ".join(f"{B//1000}k={reg[B]:+.2f}" for B in BUDGETS),
              flush=True)
    print("EXP068_VALH2_SHARD_DONE", flush=True)
    sys.exit(0)


Z = np.load(STATES_NPZ, allow_pickle=True)
SX, SC, SPOOL = Z["x"], Z["c"], Z["pool"]
NSTD = np.array([0.0123, 0.003, 2.0, 0.015], np.float32)


def nearest_idx(pool, a):
    return int(np.argmin(np.linalg.norm((pool - np.asarray(a, np.float32)) / NSTD[None], axis=1)))


# ------------------------------------------------------------------ search    #
if args.phase == "search":
    out_path = OUT / f"search_shard{args.shard_id}.jsonl"
    done = set()
    if out_path.exists():
        done = {(json.loads(l)["sid"], json.loads(l)["arm"])
                for l in out_path.read_text().splitlines() if l.strip()}
    ARMS = ["vec_tree", "flat_width", "ref"]
    jobs = [(sid, arm) for sid in range(len(SX)) for arm in ARMS]
    for j, (sid, arm) in enumerate(jobs):
        if j % args.num_shards != args.shard_id or (sid, arm) in done:
            continue
        x, c, pool = SX[sid].astype(np.float32), SC[sid].astype(np.float32), SPOOL[sid].astype(np.float32)
        h = args.horizon
        chosen = {}
        if arm in ("vec_tree", "ref"):
            budgets = BUDGETS if arm == "vec_tree" else [args.ref_budget]
            prior = np.concatenate([np.full(POOL_POLICY, 0.8 / POOL_POLICY),
                                    np.full(len(pool) - POOL_POLICY,
                                            0.2 / max(len(pool) - POOL_POLICY, 1))])
            tree = VecTree(x, c, h, SIE, pool, prior, sample_batch_fn=sample_batch_fn,
                           rollout_batch_fn=rollout_batch_fn, noise=NZ,
                           rng=np.random.default_rng(args.seed * 31 + sid),
                           max_depth=args.max_depth, wave=args.wave)
            picks = tree.run(budgets)
            chosen = {str(B): nearest_idx(pool, a) for B, a in picks.items()}
        else:                                        # deployed-selector family, budget-matched
            nc = env_bridge.next_condition(c, SIE)
            for B in BUDGETS:
                k = int(np.clip(B // 128, 8, 64))
                m = min(len(pool), B // k)
                sub = pool[:m]
                realized = NZ.sample_batch(sub, k, crn=True).reshape(-1, 4)
                posts, ill = env_bridge.apply_legality(x, env_bridge.simulate(x, c, realized), h, c)
                q = (-np.asarray(env_bridge.evaluate_value(VMODEL, posts, nc, device))
                     ).reshape(m, k).mean(axis=1)
                q[ill.reshape(m, k).any(axis=1)] = -1e18
                chosen[str(B)] = int(np.argmax(q))
        with out_path.open("a") as fh:
            fh.write(json.dumps({"sid": sid, "arm": arm, "chosen_idx": chosen}) + "\n")
        print(f"[search] h{h} sid {sid} {arm} done", flush=True)
    print("EXP068_SEARCH_SHARD_DONE", flush=True)
    sys.exit(0)


# ------------------------------------------------------------------ adjudicate #
if args.phase == "adjudicate":
    champ = WorldPlayer(args.world, device, name="cont",
                        noise=make_noise(noise_path, seed=4242 + args.shard_id),
                        sel_noise_samples=8)
    rows = []
    for f in OUT.glob("search_shard*.jsonl"):
        rows += [json.loads(l) for l in f.read_text().splitlines() if l.strip()]

    def playout(x, c, h, first_action, seed):
        nz = LocalNoise(noise_path, seed=seed)
        persp = int(round(c[2]))
        st, cc, hh = x.copy(), c.copy(), int(h)
        a = np.asarray(first_action, np.float32)
        while hh >= 1:
            realized = nz.sample_batch(a[None], 1).reshape(4).astype(np.float32)
            post, _ = env_bridge.apply_legality(st, env_bridge.simulate_one(st, cc, realized)[None],
                                                hh, cc)
            st, cc, hh = post[0], env_bridge.next_condition(cc, SIE), hh - 1
            if hh >= 1:
                a = np.asarray(champ.select_intended(st, cc, hh, SIE, int(round(cc[2]))), np.float32)
        return float(env_bridge.score_end(st, persp))

    out_path = OUT / f"adj_shard{args.shard_id}.jsonl"
    done = {json.loads(l)["sid"] for l in out_path.read_text().splitlines()} if out_path.exists() else set()
    for sid in range(len(SX)):
        if sid % args.num_shards != args.shard_id or sid in done:
            continue
        x, c, pool = SX[sid].astype(np.float32), SC[sid].astype(np.float32), SPOOL[sid].astype(np.float32)
        idxs = sorted({int(v) for r in rows if r["sid"] == sid for v in r["chosen_idx"].values()})
        if not idxs:
            continue
        held = {}
        for i in idxs:                                # CRN: identical seeds across actions
            ds = [playout(x, c, args.horizon, pool[i], 680_000 + 977 * t) for t in range(args.playouts)]
            held[str(i)] = round(float(np.mean(ds)), 4)
        se = float(np.std([held[str(i)] for i in idxs], ddof=1)) if len(idxs) > 1 else 0.0
        with out_path.open("a") as fh:
            fh.write(json.dumps({"sid": sid, "held": held, "spread": round(se, 4)}) + "\n")
        print(f"[adj] h{args.horizon} sid {sid}: {len(idxs)} actions x {args.playouts} playouts",
              flush=True)
    print("EXP068_ADJ_SHARD_DONE", flush=True)
    sys.exit(0)
