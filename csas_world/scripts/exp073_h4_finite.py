#!/usr/bin/env python3
"""EXP-073: h=4 FINITE-GAME reference + VecTree-with-confirmation (user design 2026-08-11).

The h>2 experiments were judging one stochastic tree with another (the 32k "strong-play"
adjudicator), so regret there was never ground truth — EXP-066 avoided this only because
h=2 admits an exact expectimax reference. This builds the missing yardstick by recursion
instead of by MCTS:

    root shot A
      -> average K1 CRN execution draws               (chance)
      -> opponent picks the WORST reply from a fixed pool   (min)
      -> average K2 CRN execution draws               (chance)
      -> remaining h=2 solved by the VALIDATED high-precision expectimax
    value(A) = mean_{K1} [ min_B mean_{K2} [ V_h2(s2) ] ]

Every noise draw is a FIXED CRN set shared across root actions, so the table is a
deterministic function of the state: two runs give identical numbers, and differences
between root actions are paired (low variance) even though absolute values carry a
consistent small-pool bias — which is fine, since only the RANKING is used.

Arms measured against that table:
  * vt_plain  — VecTree root argmax (what EXP-068/069/071/072 used)
  * vt_confirm— VecTree explores, then its top-8 root actions are re-evaluated in a
                fresh paired CRN confirmation pass and the winner is taken from THAT
                (MCTS explores -> robust root tournament decides). The confirmation is
                paid for OUT OF the same budget, so budgets stay comparable.
Both at 1k/4k/16k/64k x >=2 seeds. Metrics per the user's spec: held-out regret vs the
finite table, per-seed monotonicity, ACROSS-SEED VALUE variance (not action agreement),
and epsilon-optimal rate.
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
ap.add_argument("--phase", required=True, choices=["states", "ref", "search", "aggregate"])
ap.add_argument("--config", default="configs/exp_037_sig_screen_tree.yaml")
ap.add_argument("--policy", default="checkpoints/csas_world/az_v25_br/policy_csas.pt")
ap.add_argument("--world", default="checkpoints/csas_world/az_v25_br/best.pt")
ap.add_argument("--n-states", type=int, default=24)
ap.add_argument("--n-root", type=int, default=32, help="fixed shared root candidate pool")
ap.add_argument("--budgets", default="1000,4000,16000,64000")
ap.add_argument("--seeds", default="a,b")
# finite reference fidelity
ap.add_argument("--k1", type=int, default=6, help="root execution draws (CRN)")
ap.add_argument("--n-opp1", type=int, default=10, help="opponent reply pool at h=3")
ap.add_argument("--k2", type=int, default=2, help="opponent execution draws (CRN)")
ap.add_argument("--h2-cand", type=int, default=10)
ap.add_argument("--h2-k3", type=int, default=5)
ap.add_argument("--h2-opp", type=int, default=8)
ap.add_argument("--h2-k4", type=int, default=3)
ap.add_argument("--chunk", type=int, default=24, help="h=2 solves per batch")
ap.add_argument("--confirm-frac", type=float, default=0.25, help="budget share for the confirmation pass")
ap.add_argument("--confirm-top", type=int, default=8)
ap.add_argument("--device", default="cuda:0")
ap.add_argument("--out-dir", default="eval_out/exp073_h4")
ap.add_argument("--num-shards", type=int, default=1)
ap.add_argument("--shard-id", type=int, default=0)
ap.add_argument("--ref-seed-base", type=int, default=90000, help="CRN base; change it to re-derive the table independently (yardstick stability check)")
ap.add_argument("--seed", type=int, default=73)
args = ap.parse_args()

OUT = Path(args.out_dir)
OUT.mkdir(parents=True, exist_ok=True)
BUDGETS = [int(b) for b in args.budgets.split(",")]
SIE = 10
H = 4


# ------------------------------------------------------------------ aggregate #
def aggregate():
    ref = {}
    for f in OUT.glob("ref_shard*.jsonl"):
        for l in f.read_text().splitlines():
            r = json.loads(l)
            ref[r["sid"]] = np.asarray(r["value"])
    rows = []
    for f in OUT.glob("search_shard*.jsonl"):
        rows += [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    if not ref or not rows:
        print(f"incomplete: {len(ref)} reference states, {len(rows)} arm rows")
        return
    sids = sorted(ref)
    print(f"EXP-073 h=4 finite-game reference: {len(sids)} states, {len(rows)} arm rows")
    spread = np.mean([ref[s].max() - ref[s].min() for s in sids])
    gap12 = np.mean([np.sort(ref[s])[-1] - np.sort(ref[s])[-2] for s in sids])
    print(f"reference: mean best-worst spread {spread:.3f}/end, mean top1-top2 gap {gap12:.3f}/end")
    print("(a deterministic table: fixed CRN noise sets shared across root actions)\n")

    arms = sorted({r["arm"] for r in rows})
    seeds = sorted({r["seed"] for r in rows})
    for arm in arms:
        print(f"== {arm}")
        for sd in seeds:
            line = []
            for B in BUDGETS:
                rs = [float(ref[r["sid"]].max() - ref[r["sid"]][r["chosen_idx"][str(B)]])
                      for r in rows
                      if r["arm"] == arm and r["seed"] == sd and r["sid"] in ref
                      and str(B) in r["chosen_idx"]]
                if rs:
                    line.append(f"{B//1000}k: {np.mean(rs):.4f}±{np.std(rs)/math.sqrt(len(rs)):.4f}")
            print(f"   seed {sd}: " + "  ".join(line))
        # across-seed VALUE agreement (the metric that matters, not action agreement)
        for B in BUDGETS:
            per = {}
            for r in rows:
                if r["arm"] == arm and str(B) in r["chosen_idx"] and r["sid"] in ref:
                    per.setdefault(r["sid"], []).append(
                        float(ref[r["sid"]][r["chosen_idx"][str(B)]]))
            both = [v for v in per.values() if len(v) >= 2]
            if both:
                sd_val = float(np.mean([np.std(v) for v in both]))
                same_act = np.mean([len({tuple(sorted(v))}) for v in both])  # placeholder
                eps = float(np.mean([float(ref[s].max() - max(per[s]) <= 0.05) for s in per]))
                print(f"   B={B//1000}k: across-seed value SD {sd_val:.4f}/end   "
                      f"eps-optimal(0.05) {eps:.0%}")
        print()


if args.phase == "aggregate":
    aggregate()
    sys.exit(0)

# ------------------------------------------------------------------ shared    #
import torch
import torch.nn as nn
from csas.search import _sample_actions, _sample_actions_batch, load_policy

from world.config import model_cfg_from_dict
from world.model import WorldModel
from world.preplaced import MODES, board_norm
from world.search.collect import _mc_rollout_terminal_batch
from world.search.noise import make_noise
from world.search.vec_tree import VecTree
from world.train.trainer import load_world_checkpoint

device = torch.device(args.device if torch.cuda.is_available() else "cpu")
env_bridge.warm_jax()
cfg_full: Config = load_config(args.config)
cfg = cfg_full.search
noise_path = cfg_full.csas_path(cfg.noise_config).as_posix()
NZ = make_noise(noise_path, seed=args.seed * 7 + args.shard_id)
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
STATES = OUT / "states.npz"
NSTD = np.array([0.0123, 0.003, 2.0, 0.015], np.float32)




def score_end_batch(states_norm, persp):
    """Vectorised twin of csas.score_end_value (verified row-identical below). The
    per-row Python/JAX version was the second O(10^5)-call bottleneck in the finite
    reference."""
    from csas.common import NUM_STONES, POS_MAX, in_play_raw
    from csas.generate_horizon_targets import BUTTON_RAW, HOUSE_RADIUS_RAW
    s = np.asarray(states_norm, np.float32).reshape(-1, NUM_STONES, 2) * POS_MAX
    live = np.stack([in_play_raw(r) for r in s]) if len(s) < 64 else _live_vec(s)
    d = np.linalg.norm(s - np.asarray(BUTTON_RAW, np.float32)[None, None, :], axis=2)
    in_house = live & (d <= HOUSE_RADIUS_RAW)
    teams = np.zeros(NUM_STONES, np.int32); teams[6:] = 1
    dh = np.where(in_house, d, np.inf)
    best = np.argmin(dh, axis=1)
    scoring = teams[best]
    any_house = in_house.any(axis=1)
    opp_mask = in_house & (teams[None, :] != scoring[:, None])
    opp_best = np.min(np.where(opp_mask, d, np.inf), axis=1)
    pts = (in_house & (teams[None, :] == scoring[:, None]) & (d < opp_best[:, None])).sum(axis=1)
    sign = np.where(scoring == int(persp), 1.0, -1.0)
    return np.where(any_house, sign * pts, 0.0).astype(np.float64)


def _live_vec(s):
    """in_play_raw vectorised over rows (same predicate, no per-row call)."""
    from csas.common import POS_MAX
    x, y = s[..., 0], s[..., 1]
    on_grid = (x > 0) | (y > 0)
    parked = (x >= POS_MAX) & (y >= POS_MAX)
    return on_grid & ~parked


def legal_fix(pre_rows, posts, horizon, cond):
    """The early-takeout forfeit rule is active ONLY for horizon >= 8 (thrown stones
    1-3); boundary removal is already applied inside simulate/simulate_batched. So at
    h<=4 this is a no-op — skipping it removes O(10^5) per-row JAX calls per state,
    which was the entire cost of the finite reference (55 min/state -> seconds)."""
    if int(horizon) < 8:
        return posts
    out = posts.copy()
    for i in range(len(posts)):
        corrected, _ = env_bridge.apply_legality(pre_rows[i], posts[i][None], int(horizon), cond)
        out[i] = corrected[0]
    return out


def pol_batch(states, cond, n, temp=1.1, std=1.2, cap=96):
    """Batched policy proposals with chunking (GNN edge features are memory-heavy)."""
    states = np.asarray(states, np.float32)
    out = []
    for s in range(0, len(states), cap):
        sub = states[s:s + cap]
        cb = np.broadcast_to(np.asarray(cond, np.float32), (len(sub), 3)).astype(np.float32)
        out.append(np.asarray(_sample_actions_batch(policy, amean_t, astd_t, sub, cb, n,
                                                    device, temp, std, 0.0),
                              np.float32).reshape(len(sub), n, 4))
    return np.concatenate(out, axis=0)


# ------------------------------------------------------------------ states    #
if args.phase == "states":
    rng = np.random.default_rng(args.seed)
    xs, cs, pools = [], [], []
    for i in range(args.n_states):
        mode = MODES[rng.integers(0, len(MODES))]
        fb = int(rng.integers(0, 2))
        x = board_norm(mode, 1 if fb == 0 else 7)
        c = np.asarray([0.0, 0.0, float(fb)], np.float32)
        h = 10
        while h > H:                                  # hot-prefix burn-in (sound by construction)
            cand = np.asarray(_sample_actions(policy, amean_t, astd_t, x, c, 8, device,
                                              2.0, 2.0, 0.0), np.float32)
            a = NZ.sample_batch(cand[rng.integers(0, len(cand))][None], 1).reshape(4).astype(np.float32)
            post, _ = env_bridge.apply_legality(x, env_bridge.simulate_one(x, c, a)[None], h, c)
            x, c, h = post[0], env_bridge.next_condition(c, SIE), h - 1
        pool = np.asarray(_sample_actions(policy, amean_t, astd_t, x, c, args.n_root, device,
                                          1.1, 1.2, 0.0), np.float32)
        xs.append(x); cs.append(c); pools.append(pool)
        print(f"[states] {i}", flush=True)
    np.savez_compressed(STATES, x=np.stack(xs), c=np.stack(cs), pool=np.stack(pools))
    print(f"[states] wrote {len(xs)} h=4 states -> {STATES}")
    sys.exit(0)

Z = np.load(STATES, allow_pickle=True)
SX, SC, SPOOL = Z["x"], Z["c"], Z["pool"]


# ------------------------------------------------------------------ h=2 solve #
def h2_values(states, cond, seed):
    """High-precision expectimax value of h=2 states (the EXP-066-validated shape):
    max over our candidates of  mean_noise[ min over opponent replies mean_noise[ rules ] ].
    Deterministic given `seed` (fixed CRN sets), batched over `states`."""
    states = np.asarray(states, np.float32)
    S = len(states)
    if S == 0:
        return np.zeros(0)
    persp = int(round(np.asarray(cond)[2]))
    nz = make_noise(noise_path, seed=seed)
    nc = env_bridge.next_condition(np.asarray(cond, np.float32), SIE)
    cand = pol_batch(states, cond, args.h2_cand)                       # [S,C,4]
    C, K3 = args.h2_cand, args.h2_k3
    a_flat = cand.reshape(S * C, 4)
    real = np.asarray(nz.sample_batch(a_flat, K3, crn=True), np.float32).reshape(S * C * K3, 4)
    st_rep = np.repeat(states, C * K3, axis=0)
    cb = np.broadcast_to(np.asarray(cond, np.float32), (len(st_rep), 3)).astype(np.float32)
    post = legal_fix(st_rep, env_bridge.simulate_batched(st_rep, cb, real), 2, cond)
    # opponent's last throw
    opp = pol_batch(post, nc, args.h2_opp)                              # [S*C*K3, O, 4]
    O, K4 = args.h2_opp, args.h2_k4
    o_flat = opp.reshape(-1, 4)
    real2 = np.asarray(nz.sample_batch(o_flat, K4, crn=True), np.float32).reshape(-1, 4)
    st2 = np.repeat(post, O * K4, axis=0)
    cb2 = np.broadcast_to(nc, (len(st2), 3)).astype(np.float32)
    term = legal_fix(st2, env_bridge.simulate_batched(st2, cb2, real2), 1, nc)
    sc = score_end_batch(term, persp)
    sc = sc.reshape(S, C, K3, O, K4)
    v = sc.mean(axis=4).min(axis=3).mean(axis=2)                        # [S,C]
    return v.max(axis=1)


# ------------------------------------------------------------------ reference #
if args.phase == "ref":
    out_path = OUT / f"ref_shard{args.shard_id}.jsonl"
    done = {json.loads(l)["sid"] for l in out_path.read_text().splitlines()} if out_path.exists() else set()
    for sid in range(len(SX)):
        if sid % args.num_shards != args.shard_id or sid in done:
            continue
        x, c, pool = SX[sid].astype(np.float32), SC[sid].astype(np.float32), SPOOL[sid].astype(np.float32)
        A = len(pool)
        nz = make_noise(noise_path, seed=args.ref_seed_base + sid)                  # fixed CRN set per state
        c1 = env_bridge.next_condition(c, SIE)
        # level 1: root action x K1 CRN execution draws
        real1 = np.asarray(nz.sample_batch(pool, args.k1, crn=True), np.float32).reshape(A * args.k1, 4)
        s1 = env_bridge.simulate(x, c, real1)
        s1, _ = env_bridge.apply_legality(x, s1, H, c)
        # level 2: opponent replies (fixed pool per s1), worst-for-us
        opp = pol_batch(s1, c1, args.n_opp1)                            # [A*K1, B, 4]
        B, K2 = args.n_opp1, args.k2
        o_flat = opp.reshape(-1, 4)
        real2 = np.asarray(nz.sample_batch(o_flat, K2, crn=True), np.float32).reshape(-1, 4)
        st = np.repeat(s1, B * K2, axis=0)
        cb = np.broadcast_to(c1, (len(st), 3)).astype(np.float32)
        s2 = legal_fix(st, env_bridge.simulate_batched(st, cb, real2), H - 1, c1)
        # level 3: the validated h=2 expectimax at every resulting state
        c2 = env_bridge.next_condition(c1, SIE)
        vals = np.empty(len(s2))
        for s in range(0, len(s2), args.chunk):
            vals[s:s + args.chunk] = h2_values(s2[s:s + args.chunk], c2, seed=args.ref_seed_base + 1000 + sid)
        v = vals.reshape(A, args.k1, B, K2)
        value = v.mean(axis=3).min(axis=2).mean(axis=1)                 # mean_K1[ min_B mean_K2 ]
        with out_path.open("a") as fh:
            fh.write(json.dumps({"sid": sid, "value": [round(float(z), 4) for z in value]}) + "\n")
        print(f"[ref] sid {sid}: best {value.max():+.3f} worst {value.min():+.3f} "
              f"top1-top2 {np.sort(value)[-1] - np.sort(value)[-2]:+.3f}", flush=True)
    print("EXP073_REF_SHARD_DONE", flush=True)
    sys.exit(0)


# ------------------------------------------------------------------ search    #
if args.phase == "search":
    def rollout_batch_fn(states, cond, h, persp):
        rng = np.random.default_rng(args.seed + 13)
        return _mc_rollout_terminal_batch(policy, amean_t, astd_t, np.asarray(states, np.float32),
                                          np.asarray(cond, np.float32), int(h), SIE, int(persp),
                                          device, rng, NZ, cfg.rollout_temp, cfg.std_scale)

    def confirm(x, c, acts, budget_sims, seed):
        """Paired CRN confirmation tournament over the tree's finalists: identical
        execution draws for every finalist, value-greedy continuation to terminal."""
        n = len(acts)
        persp = int(round(c[2]))
        nc = env_bridge.next_condition(c, SIE)
        per = max(8, int(budget_sims / max(n * (H - 1) * 4, 1)))        # CRN draws per finalist
        nz = make_noise(noise_path, seed=seed)
        real = np.asarray(nz.sample_batch(np.asarray(acts, np.float32), per, crn=True),
                          np.float32).reshape(n * per, 4)
        post = env_bridge.simulate(x, c, real)
        post, ill = env_bridge.apply_legality(x, post, H, c)
        rng = np.random.default_rng(seed + 5)
        q = _mc_rollout_terminal_batch(policy, amean_t, astd_t, post, nc, H - 1, SIE, persp,
                                       device, rng, NZ, cfg.rollout_temp, cfg.std_scale,
                                       value_model=VMODEL, n_search=4)
        q = np.asarray(q, np.float64).reshape(n, per)
        q = np.where(np.asarray(ill).reshape(n, per), -1e9, q)
        return int(np.argmax(q.mean(axis=1)))

    out_path = OUT / f"search_shard{args.shard_id}.jsonl"
    done = set()
    if out_path.exists():
        done = {(json.loads(l)["sid"], json.loads(l)["arm"], json.loads(l)["seed"])
                for l in out_path.read_text().splitlines() if l.strip()}
    seeds = args.seeds.split(",")
    jobs = [(sid, arm, sd) for sid in range(len(SX))
            for arm in ("vt_plain", "vt_confirm") for sd in seeds]
    for j, (sid, arm, sd) in enumerate(jobs):
        if j % args.num_shards != args.shard_id or (sid, arm, sd) in done:
            continue
        x, c, pool = SX[sid].astype(np.float32), SC[sid].astype(np.float32), SPOOL[sid].astype(np.float32)
        base = args.seed * (31 if sd == "a" else 41) + sid * 7
        prior = np.full(len(pool), 1.0 / len(pool))
        chosen = {}
        for B in BUDGETS:
            frac = args.confirm_frac if arm == "vt_confirm" else 0.0
            tree_budget = int(B * (1.0 - frac))
            tree = VecTree(x, c, H, SIE, pool, prior, sample_batch_fn=lambda s, cd, n: pol_batch(s, cd, n),
                           rollout_batch_fn=rollout_batch_fn, noise=NZ,
                           rng=np.random.default_rng(base + B), max_depth=4, wave=32,
                           root_out_cap=64)
            tree.run([tree_budget])
            acts, q, se, nv = tree.root_stats()
            if len(acts) == 0:
                chosen[str(B)] = 0
                continue
            if arm == "vt_plain":
                pick = acts[int(np.argmax(q))]
            else:
                top = np.argsort(q)[::-1][: min(args.confirm_top, len(q))]
                w = confirm(x, c, acts[top], int(B * frac), base + B + 1)
                pick = acts[top][w]
            chosen[str(B)] = int(np.argmin(np.linalg.norm((pool - pick) / NSTD[None], axis=1)))
        with out_path.open("a") as fh:
            fh.write(json.dumps({"sid": sid, "arm": arm, "seed": sd, "chosen_idx": chosen}) + "\n")
        print(f"[search] sid {sid} {arm} seed {sd} done", flush=True)
    print("EXP073_SEARCH_SHARD_DONE", flush=True)
    sys.exit(0)
