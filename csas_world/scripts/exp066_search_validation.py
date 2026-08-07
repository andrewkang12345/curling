#!/usr/bin/env python3
"""EXP-066: instrumented SEARCH-VALIDATION benchmark (user-specified design).

Question: was search ever given a fair, correctly-structured chance to beat the
robust flat selectors — or were our trees underfunded/mis-structured? Measure
SIMPLE REGRET vs a near-exact reference as a function of simulator-call budget;
the soundness criterion is MONOTONE regret reduction with budget.

Design (h=2 states: our throw + opponent's last throw -> rules score; the one
horizon where a chance-correct expectimax reference is computable to high
precision):
  * STATES: ~60 = 40 tactical (hot-prefix generated, congested: >=4 live stones
    or >=2 in house) + 20 control, all at h=2.
  * SHARED POOL per state: 96 policy samples + 32 structured/diverse = 128
    candidates; every arm chooses FROM THE POOL (proposal variance excluded).
  * REFERENCE (phase ref, GPU): two-stage chance-correct expectimax:
    stage A all 128 candidates at (32 root-CRN x [48 opp cands x 8 CRN]);
    stage B top-16 refined at (128 x [64 x 16]). Truth table Ref(a) + SE.
  * ARMS at budgets {1k, 4k, 16k, 64k} simulator calls:
      - flat_width  : bigsel-style robust width (m cands x k noise, value-ranked;
                      k = clip(B/128, 8, 64), m = min(128, B // k))
      - screen_tree : the operator of record, budget-scaled (noise_samples and
                      stage-2 sims scaled by B/2700)
      - hybrid_tree : the new prior-guided chance-aware KR tree (CPU phase;
                      anytime checkpoints from ONE 64k run)
  * METRIC: regret(arm, B) = max_a Ref(a) − Ref(chosen_a); mean over states,
    stratified tactical/control; monotonicity verdict per arm.

Phases (run separately: ref+flat on GPU-JAX, tree on CPU-JAX):
  python scripts/exp066_search_validation.py --phase states
  python scripts/exp066_search_validation.py --phase ref   --shard-id K --num-shards N
  python scripts/exp066_search_validation.py --phase flat  --shard-id K --num-shards N
  python scripts/exp066_search_validation.py --phase tree  --shard-id K --num-shards N
  python scripts/exp066_search_validation.py --phase aggregate
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
ap.add_argument("--phase", required=True,
                choices=["states", "ref", "flat", "tree", "adjudicate", "aggregate"])
ap.add_argument("--config", default="configs/exp_037_sig_screen_tree.yaml")
ap.add_argument("--policy", default="checkpoints/csas_world/az_v15_L8/incumbent0_policy_csas.pt")
ap.add_argument("--world", default="checkpoints/csas_world/az_v14d/best.pt")
ap.add_argument("--n-tactical", type=int, default=40)
ap.add_argument("--n-control", type=int, default=20)
ap.add_argument("--budgets", default="1000,4000,16000,64000")
ap.add_argument("--device", default="cuda:0")
ap.add_argument("--out-dir", default="eval_out/exp066_search")
ap.add_argument("--num-shards", type=int, default=1)
ap.add_argument("--shard-id", type=int, default=0)
ap.add_argument("--seed", type=int, default=66)
ap.add_argument("--ref-fast", action="store_true", help="tiny reference for smoke tests")
ap.add_argument("--state-subset", type=int, default=0, help="tree/adjudicate: use only sid < N")
ap.add_argument("--inner-pool", type=int, default=8, help="opponent candidates per outcome node")
ap.add_argument("--out-cap", type=int, default=8, help="max sampled outcomes per chance node")
args = ap.parse_args()

OUT = Path(args.out_dir)
OUT.mkdir(parents=True, exist_ok=True)
BUDGETS = [int(b) for b in args.budgets.split(",")]
H = 2
POOL_POLICY, POOL_EXTRA = 96, 32


# ---------------------------------------------------------------- aggregate #
def aggregate():
    ref = {}
    for f in OUT.glob("ref_shard*.jsonl"):
        for l in f.read_text().splitlines():
            r = json.loads(l)
            ref[r["sid"]] = r
    rows = []
    for f in list(OUT.glob("flat_shard*.jsonl")) + list(OUT.glob("tree_shard*.jsonl")):
        rows += [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    adj = {}
    for f in OUT.glob("adj_shard*.jsonl"):
        for l in f.read_text().splitlines():
            r = json.loads(l); adj[r["sid"]] = r
    if not ref or not rows:
        print(f"incomplete: {len(ref)} ref states, {len(rows)} arm rows")
        return
    if adj:
        print(f"regret source: HELD-OUT fresh evaluation ({len(adj)} states; curse-free)")
    print(f"EXP-066 search-validation: {len(ref)} states, {len(rows)} arm rows")
    med_se = np.median([r["ref_se_top"] for r in ref.values()])
    print(f"reference precision: median top-16 SE = {med_se:.4f}/end\n")
    tree_sids = {r["sid"] for r in rows if r["arm"].endswith("_term")}
    if tree_sids:
        rows = [r for r in rows if r["sid"] in tree_sids]
        print(f"common-subset comparison: {len(tree_sids)} states covered by all arms")
    arms = sorted(set(r["arm"] for r in rows))
    for arm in arms:
        print(f"== {arm}")
        for strat in ("all", "tactical", "control"):
            line = []
            for B in BUDGETS:
                rs = []
                for r in rows:
                    if r["arm"] != arm or str(B) not in r["chosen_idx"]:
                        continue
                    rf = ref.get(r["sid"])
                    if rf is None or (strat != "all" and rf["stratum"] != strat):
                        continue
                    a = adj.get(r["sid"])
                    ci = r["chosen_idx"][str(B)]
                    if a is not None and str(ci) in a["held"]:
                        rs.append(float(a["held"][str(a["ref_argmax"])] - a["held"][str(ci)]))
                    else:
                        vals = np.asarray(rf["ref"])
                        rs.append(float(vals.max() - vals[ci]))
                if rs:
                    line.append(f"B={B//1000}k: {np.mean(rs):+.4f}±{np.std(rs)/math.sqrt(len(rs)):.4f}")
            print(f"   [{strat:8s}] " + "   ".join(line))
        print()
    print("verdict rule: sound implementation <=> AGGREGATE mean regret decreases in B (within SEs —")
    print("per-state non-monotonicity is expected variance);")
    print("game-level ceiling <=> all arms' 64k regret ~ reference noise floor.")


if args.phase == "aggregate":
    aggregate()
    sys.exit(0)

# ---------------------------------------------------------------- shared    #
import torch
import torch.nn as nn
from csas.search import _sample_actions, load_policy

from world.config import model_cfg_from_dict
from world.model import WorldModel
from world.preplaced import MODES, board_norm
from world.search.candidates import generate_candidates
from world.search.noise import make_noise
from world.train.trainer import load_world_checkpoint

device = torch.device(args.device if torch.cuda.is_available() else "cpu")
env_bridge.warm_jax()
cfg_full: Config = load_config(args.config)
cfg = cfg_full.search
noise_path = cfg_full.csas_path(cfg.noise_config).as_posix()
NZ = make_noise(noise_path, seed=args.seed * 101 + args.shard_id)
policy, amean_t, astd_t = load_policy(args.policy, device)
amean_np = amean_t.detach().cpu().numpy().astype(np.float64)
astd_np = astd_t.detach().cpu().numpy().astype(np.float64)

ck = torch.load(args.world, map_location=device, weights_only=False)
_wm = WorldModel(model_cfg_from_dict(ck["model_cfg"])).to(device)
load_world_checkpoint(_wm, args.world, map_location=device)
_wm.eval()


def value_batch(states, cond):
    class _V(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x, c):
            return self.m.value_head.value(self.m.encode(x, c))
    return env_bridge.evaluate_value(_V(_wm), states, cond, device)


STATES_NPZ = OUT / "states.npz"


def descriptor(x):
    from csas.common import in_play_raw, raw_to_compact_m
    raw = np.asarray(x, np.float32).reshape(12, 2) * 4095.0
    live = in_play_raw(raw)
    cm = raw_to_compact_m(raw)
    r = np.hypot(cm[:, 0], cm[:, 1])
    return int(live.sum()), int((live & (r <= 1.974)).sum())


# ---------------------------------------------------------------- states    #
if args.phase == "states":
    rng = np.random.default_rng(args.seed)
    xs, cs, strata, pools = [], [], [], []

    def hot_to_h2(want_tactical):
        while True:
            mode = MODES[rng.integers(0, len(MODES))]
            fb = int(rng.integers(0, 2))
            x = board_norm(mode, 1 if fb == 0 else 7)
            c = np.asarray([0.0, 0.0, float(fb)], np.float32)
            h = 10
            while h > H:
                cands = np.asarray(_sample_actions(policy, amean_t, astd_t, x, c, 8, device,
                                                   2.0, 2.0, 0.0), np.float32)
                a = NZ.sample_batch(cands[rng.integers(0, len(cands))][None], 1).reshape(4).astype(np.float32)
                post, _ = env_bridge.apply_legality(x, env_bridge.simulate_one(x, c, a)[None], h, c)
                x, c, h = post[0], env_bridge.next_condition(c, 10), h - 1
            n_live, n_house = descriptor(x)
            tact = (n_live >= 4) or (n_house >= 2)
            if tact == want_tactical:
                return x, c

    for i in range(args.n_tactical + args.n_control):
        want = i < args.n_tactical
        x, c = hot_to_h2(want)
        pool = generate_candidates(policy, amean_t, astd_t, x, c, cfg, rng, device)
        pol = np.asarray(_sample_actions(policy, amean_t, astd_t, x, c, POOL_POLICY, device,
                                         1.1, 1.2, 0.0), np.float32)
        extra_idx = rng.choice(len(pool), size=min(POOL_EXTRA, len(pool)), replace=False)
        merged = np.concatenate([pol, np.asarray(pool, np.float32)[extra_idx]])[: POOL_POLICY + POOL_EXTRA]
        xs.append(x); cs.append(c); pools.append(merged)
        strata.append("tactical" if want else "control")
        print(f"[states] {i}: {strata[-1]}", flush=True)
    np.savez_compressed(STATES_NPZ, x=np.stack(xs), c=np.stack(cs),
                        pool=np.stack(pools), stratum=np.array(strata))
    print(f"[states] wrote {len(xs)} states -> {STATES_NPZ}")
    sys.exit(0)

Z = np.load(STATES_NPZ, allow_pickle=True)
SX, SC, SPOOL, SSTRAT = Z["x"], Z["c"], Z["pool"], Z["stratum"]
print(f"[exp066:{args.phase}] shard {args.shard_id}/{args.num_shards} "
      f"jax={env_bridge}", flush=True)


def opp_value_batch(posts, nc, n_opp, k_opp, rng):
    """Opponent's best last throw per post (chance-correct: mean over k CRN
    executions per opponent candidate, MAX from opponent's perspective).
    Returns value in ROOT (thrower-at-h2) perspective. Batched over posts."""
    persp_opp = int(round(nc[2]))
    out = np.empty(len(posts))
    for i, s in enumerate(posts):
        oc = np.asarray(_sample_actions(policy, amean_t, astd_t, s, nc, n_opp, device,
                                        1.1, 1.2, 0.0), np.float32)
        realized = NZ.sample_batch(oc, k_opp, crn=True).reshape(-1, 4)
        posts2, _ = env_bridge.apply_legality(s, env_bridge.simulate(s, nc, realized), 1, nc)
        sc = np.array([env_bridge.score_end(p, persp_opp) for p in posts2]).reshape(n_opp, k_opp)
        out[i] = -sc.mean(axis=1).max()      # opponent maximises its own; flip to root persp
    return out


def ref_value(x, c, actions, k_root, n_opp, k_opp, rng):
    """Chance-correct expectimax value of each root action (root persp)."""
    nc = env_bridge.next_condition(c, 10)
    vals = np.empty(len(actions)); ses = np.empty(len(actions))
    for i, a in enumerate(actions):
        realized = NZ.sample_batch(np.asarray(a, np.float32)[None], k_root).reshape(-1, 4)
        posts, _ = env_bridge.apply_legality(x, env_bridge.simulate(x, c, realized), H, c)
        v = opp_value_batch(posts, nc, n_opp, k_opp, rng)
        vals[i] = v.mean(); ses[i] = v.std(ddof=1) / math.sqrt(len(v))
    return vals, ses


if args.phase == "ref":
    out_path = OUT / f"ref_shard{args.shard_id}.jsonl"
    done = {json.loads(l)["sid"] for l in out_path.read_text().splitlines()} if out_path.exists() else set()
    for sid in range(len(SX)):
        if sid % args.num_shards != args.shard_id or sid in done:
            continue
        rng = np.random.default_rng(args.seed + sid)
        x, c, pool = SX[sid].astype(np.float32), SC[sid].astype(np.float32), SPOOL[sid].astype(np.float32)
        if args.ref_fast:
            vA, _ = ref_value(x, c, pool, k_root=4, n_opp=8, k_opp=2, rng=rng)
            top = np.argsort(vA)[::-1][:4]
            vB, seB = ref_value(x, c, pool[top], k_root=8, n_opp=16, k_opp=4, rng=rng)
        else:
            vA, _ = ref_value(x, c, pool, k_root=32, n_opp=48, k_opp=8, rng=rng)
            top = np.argsort(vA)[::-1][:16]
            vB, seB = ref_value(x, c, pool[top], k_root=128, n_opp=64, k_opp=16, rng=rng)
        ref = vA.copy()
        ref[top] = vB
        rec = {"sid": sid, "stratum": str(SSTRAT[sid]), "ref": [round(float(v), 4) for v in ref],
               "ref_se_top": round(float(seB.mean()), 4)}
        with out_path.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(f"[ref] sid {sid} done (top SE {rec['ref_se_top']})", flush=True)
    print("EXP066_REF_SHARD_DONE", flush=True)
    sys.exit(0)


def nearest_pool_idx(pool, a):
    d = np.linalg.norm((pool - np.asarray(a, np.float32)) / NOISE_STD_F[None], axis=1)
    return int(np.argmin(d))


NOISE_STD_F = np.array([0.0123, 0.003, 2.0, 0.015], dtype=np.float32)

if args.phase == "flat":
    from world.search.beam import screen_tree_choose
    out_path = OUT / f"flat_shard{args.shard_id}.jsonl"
    done = set()
    if out_path.exists():
        done = {(json.loads(l)["sid"], json.loads(l)["arm"]) for l in out_path.read_text().splitlines()}
    for sid in range(len(SX)):
        if sid % args.num_shards != args.shard_id:
            continue
        x, c, pool = SX[sid].astype(np.float32), SC[sid].astype(np.float32), SPOOL[sid].astype(np.float32)
        nc = env_bridge.next_condition(c, 10)
        # ---- flat_width (bigsel family, budget-scaled) ----
        if (sid, "flat_width") not in done:
            chosen = {}
            for B in BUDGETS:
                k = int(np.clip(B // 128, 8, 64))
                m = min(len(pool), B // k)
                sub = pool[:m]
                realized = NZ.sample_batch(sub, k, crn=True).reshape(-1, 4)
                posts, ill = env_bridge.apply_legality(x, env_bridge.simulate(x, c, realized), H, c)
                v = -value_batch(posts, nc)
                q = np.asarray(v).reshape(m, k).mean(axis=1)
                q[ill.reshape(m, k).any(axis=1)] = -1e18
                chosen[str(B)] = int(np.argmax(q))
            with out_path.open("a") as fh:
                fh.write(json.dumps({"sid": sid, "arm": "flat_width", "chosen_idx": chosen}) + "\n")
        # ---- screen_tree (operator of record, budget-scaled) ----
        if (sid, "screen_tree") not in done:
            chosen = {}
            for B in BUDGETS:
                import copy
                cfg_b = copy.deepcopy(cfg)
                scale = B / 2700.0
                cfg_b.noise_samples = int(np.clip(round(8 * scale), 2, 64))
                cfg_b.mcts_sims = int(np.clip(round(48 * scale), 16, 1024))
                rng = np.random.default_rng(args.seed * 31 + sid * 7 + B)
                r = screen_tree_choose(policy, amean_t, astd_t, amean_np, astd_np,
                                       x, c, H, 10, cfg_b, rng, device, NZ)
                chosen[str(B)] = nearest_pool_idx(pool, r["action"]) if r else 0
            with out_path.open("a") as fh:
                fh.write(json.dumps({"sid": sid, "arm": "screen_tree", "chosen_idx": chosen}) + "\n")
        print(f"[flat] sid {sid} done", flush=True)
    print("EXP066_FLAT_SHARD_DONE", flush=True)
    sys.exit(0)

if args.phase == "adjudicate":
    # Held-out simple-regret evaluation (user review 2026-08-07): the reference table's
    # argmax carries winner's curse; re-evaluate the reference-selected action AND every
    # arm's chosen action with a FRESH high-precision CRN evaluation (different seed).
    # regret(arm,B) = fresh(ref_argmax) - fresh(chosen) — held-out, curse-free.
    ref = {}
    for f in OUT.glob("ref_shard*.jsonl"):
        for l in f.read_text().splitlines():
            r = json.loads(l); ref[r["sid"]] = r
    rows = []
    for f in list(OUT.glob("flat_shard*.jsonl")) + list(OUT.glob("tree_shard*.jsonl")):
        rows += [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    out_path = OUT / f"adj_shard{args.shard_id}.jsonl"
    done = {json.loads(l)["sid"] for l in out_path.read_text().splitlines()} if out_path.exists() else set()
    n_states = min(args.state_subset or len(SX), len(SX))
    for sid in range(n_states):
        if sid % args.num_shards != args.shard_id or sid in done or sid not in ref:
            continue
        x, c, pool = SX[sid].astype(np.float32), SC[sid].astype(np.float32), SPOOL[sid].astype(np.float32)
        idxs = {int(np.argmax(ref[sid]["ref"]))}
        for r in rows:
            if r["sid"] == sid:
                idxs |= {int(v) for v in r["chosen_idx"].values()}
        idxs = sorted(idxs)
        rng = np.random.default_rng(999_000 + sid)          # fresh seed vs ref phase
        vals, ses = ref_value(x, c, pool[idxs], k_root=128, n_opp=64, k_opp=16, rng=rng)
        rec = {"sid": sid, "ref_argmax": int(np.argmax(ref[sid]["ref"])),
               "held": {str(i): round(float(v), 4) for i, v in zip(idxs, vals)},
               "held_se": round(float(np.mean(ses)), 4)}
        with out_path.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(f"[adj] sid {sid}: {len(idxs)} actions re-evaluated (se~{rec['held_se']})", flush=True)
    print("EXP066_ADJ_SHARD_DONE", flush=True)
    sys.exit(0)

if args.phase == "tree":
    from world.search.hybrid_tree import HybridTree
    # PRIMARY correctness arms (user review 2026-08-07): TERMINAL leaves only at h=2 —
    # no value head anywhere, so a failure is attributable to the tree, not V.
    # puct_term: bw->0 disables kernel sharing => plain prior-guided PUCT (diagnosis arm).
    # hybrid_v (V-leaf) is parked as the secondary practical-speed arm.
    TREE_ARMS = {"hybrid_term": dict(bw=1.0, rollout_every=1),
                 "puct_term":  dict(bw=1e-9, rollout_every=1)}
    out_path = OUT / f"tree_shard{args.shard_id}.jsonl"
    done = set()
    if out_path.exists():
        done = {(json.loads(l)["sid"], json.loads(l)["arm"]) for l in out_path.read_text().splitlines()}

    def sample_fn(x, c, n):
        return np.asarray(_sample_actions(policy, amean_t, astd_t, x, c, n, device,
                                          1.1, 1.2, 0.0), np.float32)

    def value_fn(x, c):
        return float(value_batch(x[None], c)[0])

    def rollout_fn(x, c, h, persp, budget):
        st, cc, hh = x.copy(), c.copy(), int(h)
        while hh >= 1:
            a = sample_fn(st, cc, 1)[0]
            a = NZ.sample_batch(a[None], 1).reshape(4).astype(np.float32)
            post, _ = env_bridge.apply_legality(st, env_bridge.simulate_one(st, cc, a)[None], hh, cc)
            budget.sims += 1
            st, cc, hh = post[0], env_bridge.next_condition(cc, 10), hh - 1
        return float(env_bridge.score_end(st, persp))

    n_states = min(args.state_subset or len(SX), len(SX))
    jobs = [(sid, arm) for sid in range(n_states) for arm in TREE_ARMS]
    for j, (sid, arm) in enumerate(jobs):
        if j % args.num_shards != args.shard_id or (sid, arm) in done:
            continue
        x, c, pool = SX[sid].astype(np.float32), SC[sid].astype(np.float32), SPOOL[sid].astype(np.float32)
        prior = np.concatenate([np.full(POOL_POLICY, 0.8 / POOL_POLICY),
                                np.full(len(pool) - POOL_POLICY, 0.2 / max(len(pool) - POOL_POLICY, 1))])
        rng = np.random.default_rng(args.seed * 17 + sid)
        kw = TREE_ARMS[arm]
        tree = HybridTree(x, c, H, 10, pool, prior, sample_fn=sample_fn, value_fn=value_fn,
                          rollout_fn=rollout_fn, noise=NZ, rng=rng,
                          bw=kw["bw"], rollout_every=kw["rollout_every"],
                          inner_pool=args.inner_pool, out_cap=args.out_cap)
        picks = tree.run(BUDGETS)
        chosen = {str(B): nearest_pool_idx(pool, a) for B, a in picks.items()}
        with out_path.open("a") as fh:
            fh.write(json.dumps({"sid": sid, "arm": arm, "chosen_idx": chosen}) + "\n")
        print(f"[tree] sid {sid} {arm} done ({tree.budget.sims} sims)", flush=True)
    print("EXP066_TREE_SHARD_DONE", flush=True)
    sys.exit(0)
