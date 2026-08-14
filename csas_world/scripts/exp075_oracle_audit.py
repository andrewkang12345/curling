#!/usr/bin/env python3
"""EXP-075: paired root-action audit for the EXP-074 response oracle.

The full-game EXP-074 result conflates three things: the action chosen by the
opponent-model tree, the fidelity of its approximate opponent, and retention of
that action by the distilled student.  This audit separates them.

Phases
------
states
    Play fresh deployed v25-v26 ends and save learner-to-move states.  Half of
    the games put the learner on the first ply and half put it on the second,
    yielding exactly ``n-per-parity`` states at every horizon.
actions
    Freeze three intended root actions per state: deployed v25, the original
    16k EXP-074 ``opp_vectree`` teacher, and deployed az_v28.
eval
    Force each frozen root action, then use deployed v25 (learner) and v26
    (opponent) 48-candidate x 8-noise continuation.  Every arm in a
    state/replicate receives identical RNG streams (paired CRN).
aggregate
    Treat the state, not the continuation replicate, as the sampling unit and
    report paired score differences by horizon/parity/depth bucket.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")
import world  # noqa: F401
from world import env_bridge


ARM_NAMES = ("v25", "search", "azv28")
SIE = 10
ACTION_SCALE = np.asarray([0.0123, 0.003, 2.0, 0.015], dtype=np.float64)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True,
                    choices=("states", "actions", "eval", "aggregate"))
    ap.add_argument("--config", default="configs/exp_074_opp_vt_targets.yaml")
    ap.add_argument("--policy", default="checkpoints/csas_world/az_v25_br/policy_csas.pt")
    ap.add_argument("--v25", default="checkpoints/csas_world/az_v25_br/best.pt")
    ap.add_argument("--v26", default="checkpoints/csas_world/az_v26_br2/best.pt")
    ap.add_argument("--azv28", default="checkpoints/csas_world/az_v28_oppbr_meta/best.pt")
    ap.add_argument("--out-dir", default="eval_out/exp075_oracle_audit")
    ap.add_argument("--n-per-parity", type=int, default=24,
                    help="fresh games in each learner throwing-order group; also states/horizon")
    ap.add_argument("--horizons", default="1,2,3,4,5,6,7,8,9,10")
    ap.add_argument("--budget", type=int, default=16000)
    ap.add_argument("--repeats", type=int, default=16,
                    help="paired exact continuation draws per frozen root action")
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=75000)
    ap.add_argument("--eval-seed", type=int, default=975000)
    return ap.parse_args()


ARGS = parse_args()
OUT = Path(ARGS.out_dir)
OUT.mkdir(parents=True, exist_ok=True)
HORIZONS = tuple(sorted({int(h) for h in ARGS.horizons.split(",")}))
if not HORIZONS or min(HORIZONS) < 1 or max(HORIZONS) > 10:
    raise ValueError(f"invalid horizons: {HORIZONS}")
if not (0 <= ARGS.shard_id < ARGS.num_shards):
    raise ValueError("shard-id must be in [0, num-shards)")


def _reset_torch(seed: int):
    import torch

    np.random.seed(int(seed) % (2**32 - 1))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _reset_noise(noise, seed: int):
    if noise is not None:
        noise.rng = np.random.default_rng(int(seed))


def _load_jsonl(paths):
    rows = []
    for path in paths:
        if not path.exists():
            continue
        for lineno, line in enumerate(path.read_text().splitlines(), start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except Exception as exc:
                raise RuntimeError(f"bad JSONL at {path}:{lineno}: {exc}") from exc
    return rows


def _append_jsonl(path: Path, row):
    with path.open("a") as fh:
        fh.write(json.dumps(row, sort_keys=True) + "\n")
        fh.flush()


def _noise_path(cfg_full):
    return cfg_full.csas_path(cfg_full.search.noise_config).as_posix()


def _load_player(ckpt, device, noise_path, seed, name):
    from world.eval.head_to_head import WorldPlayer
    from world.search.noise import make_noise

    return WorldPlayer(ckpt, device, n_candidates=48, name=name,
                       noise=make_noise(noise_path, seed), sel_noise_samples=8)


def make_states():
    import torch

    from world.config import load_config
    from world.preplaced import build_preplaced_roots
    from world.search.noise import make_noise

    state_path = OUT / "states.npz"
    if state_path.exists():
        z = np.load(state_path, allow_pickle=True)
        counts = Counter(map(int, z["horizon"]))
        expected = {h: ARGS.n_per_parity for h in range(1, 11)}
        if dict(sorted(counts.items())) != expected:
            raise RuntimeError(f"existing states file has wrong counts: {counts}")
        print(f"[states] reuse complete {state_path} ({len(z['horizon'])} states)", flush=True)
        return

    cfg_full = load_config(ARGS.config)
    noise_path = _noise_path(cfg_full)
    device = torch.device(ARGS.device if torch.cuda.is_available() else "cpu")
    env_bridge.warm_jax()
    learner = _load_player(ARGS.v25, device, noise_path, ARGS.seed + 11, "v25")
    opponent = _load_player(ARGS.v26, device, noise_path, ARGS.seed + 13, "v26")
    env_noise = make_noise(noise_path, ARGS.seed + 17)

    n_games = 2 * int(ARGS.n_per_parity)
    roots = build_preplaced_roots(10, n_games, split="val", seed=ARGS.seed,
                                  num_shards=1, shard_id=0, balance=True)
    if len(roots) != n_games:
        raise RuntimeError(f"requested {n_games} roots, got {len(roots)}")

    xs, cs, hs, games, learner_blocks, firsts = [], [], [], [], [], []
    for game_idx, root in enumerate(roots):
        learner_first = game_idx < ARGS.n_per_parity
        root_block = int(round(float(root.c[2])))
        learner_block = root_block if learner_first else 1 - root_block
        state = root.x.copy().astype(np.float32)
        cond = root.c.copy().astype(np.float32)
        hh = int(root.horizon)

        game_seed = ARGS.seed + 1000 + game_idx * 10007
        _reset_torch(game_seed)
        _reset_noise(learner.noise, game_seed + 11)
        _reset_noise(opponent.noise, game_seed + 13)
        _reset_noise(env_noise, game_seed + 17)

        while hh >= 1:
            block = int(round(float(cond[2])))
            if block == learner_block:
                xs.append(state.copy())
                cs.append(cond.copy())
                hs.append(hh)
                games.append(game_idx)
                learner_blocks.append(learner_block)
                firsts.append(int(learner_first))
                player = learner
            else:
                player = opponent
            intended = np.asarray(
                player.select_intended(state, cond, hh, SIE, block), dtype=np.float32)
            realized = env_noise.sample_batch(intended[None], 1).reshape(4).astype(np.float32)
            post, _illegal = env_bridge.apply_legality(
                state, env_bridge.simulate_one(state, cond, realized)[None], hh, cond)
            state = post[0]
            cond = env_bridge.next_condition(cond, SIE)
            hh -= 1
        print(f"[states] game {game_idx + 1}/{n_games} ({len(xs)} learner states)", flush=True)

    order = np.lexsort((np.asarray(games), np.asarray(hs)))
    payload = dict(
        x=np.stack(xs)[order].astype(np.float32),
        c=np.stack(cs)[order].astype(np.float32),
        horizon=np.asarray(hs, dtype=np.int64)[order],
        game=np.asarray(games, dtype=np.int64)[order],
        learner_block=np.asarray(learner_blocks, dtype=np.int64)[order],
        learner_first=np.asarray(firsts, dtype=np.int64)[order],
    )
    counts = Counter(map(int, payload["horizon"]))
    expected = {h: ARGS.n_per_parity for h in range(1, 11)}
    if dict(sorted(counts.items())) != expected:
        raise RuntimeError(f"state horizon imbalance: got {counts}, expected {expected}")
    tmp = OUT / "states.tmp.npz"
    np.savez_compressed(tmp, **payload)
    tmp.replace(state_path)
    manifest = {
        "phase": "states", "seed": ARGS.seed, "games": n_games,
        "states": len(payload["horizon"]), "states_per_horizon": ARGS.n_per_parity,
        "source": "fresh deployed v25-v26 48x8 ends",
        "v25": ARGS.v25, "v26": ARGS.v26, "config": ARGS.config,
    }
    (OUT / "states.manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[states] wrote {len(payload['horizon'])} balanced states -> {state_path}", flush=True)


def make_actions():
    import torch
    from csas.search import _sample_actions_batch, load_policy

    from world.config import load_config
    from world.eval.head_to_head import WorldPlayer
    from world.search.candidates import generate_candidates
    from world.search.collect import _mc_rollout_terminal_batch
    from world.search.noise import make_noise
    from world.search.vec_tree import VecTree

    state_path = OUT / "states.npz"
    if not state_path.exists():
        raise FileNotFoundError(f"run --phase states first: {state_path}")
    z = np.load(state_path, allow_pickle=True)
    cfg_full = load_config(ARGS.config)
    cfg = cfg_full.search
    noise_path = _noise_path(cfg_full)
    device = torch.device(ARGS.device if torch.cuda.is_available() else "cpu")
    env_bridge.warm_jax()

    policy, amean_t, astd_t = load_policy(ARGS.policy, device)
    v25 = _load_player(ARGS.v25, device, noise_path, ARGS.seed + 101, "v25")
    azv28 = _load_player(ARGS.azv28, device, noise_path, ARGS.seed + 103, "azv28")
    opponent = _load_player(ARGS.v26, device, noise_path, ARGS.seed + 107, "v26")

    out_path = OUT / f"actions_shard{ARGS.shard_id}of{ARGS.num_shards}.jsonl"
    done_rows = _load_jsonl([out_path])
    done = {int(r["sid"]) for r in done_rows}
    if len(done) != len(done_rows):
        raise RuntimeError(f"duplicate action rows in {out_path}")

    for sid in range(len(z["horizon"])):
        h = int(z["horizon"][sid])
        if h not in HORIZONS or sid % ARGS.num_shards != ARGS.shard_id or sid in done:
            continue
        x = np.asarray(z["x"][sid], dtype=np.float32)
        c = np.asarray(z["c"][sid], dtype=np.float32)
        learner_block = int(z["learner_block"][sid])
        if int(round(float(c[2]))) != learner_block:
            raise RuntimeError(f"sid {sid}: saved state is not learner-to-move")

        action_seed = ARGS.seed + 200000 + sid * 1009
        _reset_torch(action_seed)
        _reset_noise(v25.noise, action_seed + 11)
        a_v25 = np.asarray(v25.select_intended(x, c, h, SIE, learner_block), np.float32)
        _reset_torch(action_seed)
        _reset_noise(azv28.noise, action_seed + 11)
        a_azv28 = np.asarray(azv28.select_intended(x, c, h, SIE, learner_block), np.float32)

        search_seed = ARGS.seed + 400000 + sid * 2017
        _reset_torch(search_seed)
        _reset_noise(opponent.noise, search_seed + 23)
        tree_noise = make_noise(noise_path, search_seed + 29)
        rng = np.random.default_rng(search_seed + 31)

        def sample_batch(states, cond2, n):
            states = np.asarray(states, np.float32)
            cb = np.broadcast_to(np.asarray(cond2, np.float32), (len(states), 3)).astype(np.float32)
            return np.asarray(
                _sample_actions_batch(policy, amean_t, astd_t, states, cb, n,
                                      device, 1.1, 1.2, 0.0),
                np.float32).reshape(len(states), n, 4)

        def rollout_batch(states, cond2, h2, perspective):
            def opponent_tail(ss, cb, h3, pb):
                return opponent.sample_intended_batch(ss, cb, 1)[:, 0, :]

            return _mc_rollout_terminal_batch(
                policy, amean_t, astd_t, np.asarray(states, np.float32),
                np.asarray(cond2, np.float32), int(h2), SIE, int(perspective),
                device, rng, tree_noise, cfg.rollout_temp, cfg.std_scale,
                opponent_action_batch_fn=opponent_tail, opponent_block=1 - learner_block)

        def opponent_batch(states, cond2, h2, depth2, n):
            states = np.asarray(states, np.float32)
            cb = np.broadcast_to(np.asarray(cond2, np.float32), (len(states), 3)).astype(np.float32)
            if int(depth2) <= int(getattr(cfg, "opponent_model_deploy_depth", 1)):
                return opponent.select_intended_batch(
                    states, cb, int(h2), SIE, 1 - learner_block, n_actions=int(n),
                    n_candidates=int(getattr(cfg, "opponent_model_candidates", 16)),
                    selection_noise_samples=int(getattr(cfg, "opponent_model_noise_samples", 2)))
            return opponent.sample_intended_batch(states, cb, int(n))

        pool_all = np.asarray(
            generate_candidates(policy, amean_t, astd_t, x, c, cfg, rng, device),
            dtype=np.float32)
        n_pol = min(int(cfg.policy_candidates), len(pool_all))
        rest = pool_all[n_pol:]
        extra = (rest[rng.choice(len(rest), size=min(32, len(rest)), replace=False)]
                 if len(rest) else pool_all[:0])
        pool = np.concatenate([pool_all[:n_pol], extra])[:128]
        prior = np.concatenate([
            np.full(n_pol, 0.8 / max(n_pol, 1)),
            np.full(len(pool) - n_pol, 0.2 / max(len(pool) - n_pol, 1)),
        ])
        tree = VecTree(
            x, c, h, SIE, pool, prior,
            sample_batch_fn=sample_batch, rollout_batch_fn=rollout_batch,
            noise=tree_noise, rng=rng,
            max_depth=int(getattr(cfg, "vectree_depth", 4)), wave=32,
            out_cap=int(getattr(cfg, "vectree_out_cap", 8)),
            root_out_cap=int(getattr(cfg, "vectree_root_out_cap", 0)),
            inner_pool=int(getattr(cfg, "vectree_inner_pool", 8)),
            opponent_batch_fn=opponent_batch, opponent_block=1 - learner_block,
            opponent_samples=int(getattr(cfg, "opponent_model_actions", 1)))
        tree.run([int(ARGS.budget)])
        acts, q, se, visits = tree.root_stats()
        if len(acts) == 0:
            raise RuntimeError(f"sid {sid}: tree returned no root action")
        ranking = np.argsort(q)[::-1]
        best_idx = int(ranking[0])
        a_search = np.asarray(acts[best_idx], np.float32)
        gap = float("nan")
        gap_t = float("nan")
        if len(ranking) >= 2:
            second_idx = int(ranking[1])
            gap = float(q[best_idx] - q[second_idx])
            den = float(np.sqrt(se[best_idx] ** 2 + se[second_idx] ** 2))
            gap_t = gap / den if np.isfinite(den) and den > 0 else float("nan")

        def nearest_stats(action):
            d = np.linalg.norm((acts.astype(np.float64) - action[None]) / ACTION_SCALE[None], axis=1)
            j = int(np.argmin(d))
            return j, float(d[j])

        j25, d25 = nearest_stats(a_v25)
        j28, d28 = nearest_stats(a_azv28)
        row = {
            "sid": sid, "horizon": h, "game": int(z["game"][sid]),
            "learner_block": learner_block, "learner_first": int(z["learner_first"][sid]),
            "budget": int(ARGS.budget), "root_actions_evaluated": int(len(acts)),
            "a_v25": a_v25.tolist(), "a_search": a_search.tolist(),
            "a_azv28": a_azv28.tolist(),
            "search_q": float(q[best_idx]), "search_se": float(se[best_idx]),
            "search_visits": float(visits[best_idx]),
            "search_top_gap": gap, "search_top_gap_t": gap_t,
            "nearest_v25_q": float(q[j25]), "nearest_v25_distance": d25,
            "nearest_azv28_q": float(q[j28]), "nearest_azv28_distance": d28,
            "approx_predicted_delta_nearest_v25": float(q[best_idx] - q[j25]),
        }
        _append_jsonl(out_path, row)
        print(f"[actions] sid={sid} h={h} q={q[best_idx]:+.3f} "
              f"gap_t={gap_t:+.2f} ({len(done) + 1} new)", flush=True)
        done.add(sid)
    print(f"EXP075_ACTIONS_SHARD_DONE shard={ARGS.shard_id}", flush=True)


def evaluate_actions():
    import torch

    from world.config import load_config
    from world.search.noise import make_noise

    state_path = OUT / "states.npz"
    if not state_path.exists():
        raise FileNotFoundError(f"run --phase states first: {state_path}")
    z = np.load(state_path, allow_pickle=True)
    action_rows = _load_jsonl(sorted(OUT.glob("actions_shard*of*.jsonl")))
    by_sid = {int(r["sid"]): r for r in action_rows}
    if len(by_sid) != len(action_rows):
        raise RuntimeError("duplicate action rows across shards")

    cfg_full = load_config(ARGS.config)
    noise_path = _noise_path(cfg_full)
    device = torch.device(ARGS.device if torch.cuda.is_available() else "cpu")
    env_bridge.warm_jax()
    learner = _load_player(ARGS.v25, device, noise_path, ARGS.eval_seed + 11, "v25")
    opponent = _load_player(ARGS.v26, device, noise_path, ARGS.eval_seed + 13, "v26")
    env_noise = make_noise(noise_path, ARGS.eval_seed + 17)

    out_path = OUT / f"eval_shard{ARGS.shard_id}of{ARGS.num_shards}.jsonl"
    existing = _load_jsonl([out_path])
    done = {(int(r["sid"]), int(r["rep"]), str(r["arm"])) for r in existing}
    if len(done) != len(existing):
        raise RuntimeError(f"duplicate eval rows in {out_path}")

    def forced_score(x, c, h, learner_block, action, crn_seed):
        _reset_torch(crn_seed)
        _reset_noise(learner.noise, crn_seed + 11)
        _reset_noise(opponent.noise, crn_seed + 13)
        _reset_noise(env_noise, crn_seed + 17)

        state = np.asarray(x, np.float32).copy()
        cond = np.asarray(c, np.float32).copy()
        intended = np.asarray(action, np.float32)
        realized = env_noise.sample_batch(intended[None], 1).reshape(4).astype(np.float32)
        post, illegal = env_bridge.apply_legality(
            state, env_bridge.simulate_one(state, cond, realized)[None], int(h), cond)
        state = post[0]
        cond = env_bridge.next_condition(cond, SIE)
        hh = int(h) - 1
        while hh >= 1:
            block = int(round(float(cond[2])))
            player = learner if block == int(learner_block) else opponent
            intended = np.asarray(
                player.select_intended(state, cond, hh, SIE, block), dtype=np.float32)
            realized = env_noise.sample_batch(intended[None], 1).reshape(4).astype(np.float32)
            post, _ill = env_bridge.apply_legality(
                state, env_bridge.simulate_one(state, cond, realized)[None], hh, cond)
            state = post[0]
            cond = env_bridge.next_condition(cond, SIE)
            hh -= 1
        return float(env_bridge.score_end(state, int(learner_block))), bool(np.asarray(illegal).any())

    selected_sids = [sid for sid in range(len(z["horizon"]))
                     if int(z["horizon"][sid]) in HORIZONS
                     and sid % ARGS.num_shards == ARGS.shard_id]
    for pos, sid in enumerate(selected_sids, start=1):
        if sid not in by_sid:
            raise RuntimeError(f"missing frozen actions for sid {sid}")
        row = by_sid[sid]
        x = np.asarray(z["x"][sid], np.float32)
        c = np.asarray(z["c"][sid], np.float32)
        h = int(z["horizon"][sid])
        learner_block = int(z["learner_block"][sid])
        for rep in range(ARGS.repeats):
            crn_seed = ARGS.eval_seed + sid * 10007 + rep * 97
            for arm in ARM_NAMES:
                key = (sid, rep, arm)
                if key in done:
                    continue
                score, illegal = forced_score(
                    x, c, h, learner_block, row[f"a_{arm}"], crn_seed)
                _append_jsonl(out_path, {
                    "sid": sid, "horizon": h, "rep": rep, "arm": arm,
                    "score": score, "root_illegal": illegal, "crn_seed": crn_seed,
                })
                done.add(key)
        print(f"[eval] sid={sid} h={h} {pos}/{len(selected_sids)}", flush=True)
    print(f"EXP075_EVAL_SHARD_DONE shard={ARGS.shard_id}", flush=True)


def _summary(values):
    a = np.asarray(values, dtype=np.float64)
    n = len(a)
    mean = float(a.mean()) if n else float("nan")
    se = float(a.std(ddof=1) / math.sqrt(n)) if n > 1 else float("inf")
    t = mean / se if np.isfinite(se) and se > 0 else float("nan")
    return {"n_states": n, "mean": mean, "se": se, "t": t}


def _json_finite(value):
    """Make result.json strict JSON instead of emitting NaN/Infinity."""
    if isinstance(value, dict):
        return {key: _json_finite(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_finite(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def aggregate():
    state_path = OUT / "states.npz"
    if not state_path.exists():
        raise FileNotFoundError(state_path)
    z = np.load(state_path, allow_pickle=True)
    selected = {sid for sid in range(len(z["horizon"])) if int(z["horizon"][sid]) in HORIZONS}
    actions = _load_jsonl(sorted(OUT.glob("actions_shard*of*.jsonl")))
    action_keys = [int(r["sid"]) for r in actions if int(r["sid"]) in selected]
    if len(action_keys) != len(set(action_keys)) or set(action_keys) != selected:
        missing = sorted(selected - set(action_keys))
        raise RuntimeError(f"incomplete actions: {len(set(action_keys))}/{len(selected)}, missing={missing[:20]}")
    eval_rows = _load_jsonl(sorted(OUT.glob("eval_shard*of*.jsonl")))
    actual = [(int(r["sid"]), int(r["rep"]), str(r["arm"])) for r in eval_rows
              if int(r["sid"]) in selected]
    expected = {(sid, rep, arm) for sid in selected for rep in range(ARGS.repeats)
                for arm in ARM_NAMES}
    if len(actual) != len(set(actual)) or set(actual) != expected:
        missing = sorted(expected - set(actual))
        extra = sorted(set(actual) - expected)
        raise RuntimeError(f"incomplete eval: {len(set(actual))}/{len(expected)}, "
                           f"missing={missing[:10]}, extra={extra[:10]}")

    scores = defaultdict(list)
    illegal = Counter()
    for r in eval_rows:
        sid = int(r["sid"])
        if sid not in selected:
            continue
        scores[(sid, str(r["arm"]))].append(float(r["score"]))
        illegal[str(r["arm"])] += int(bool(r.get("root_illegal", False)))
    state_mean = {(sid, arm): float(np.mean(scores[(sid, arm)]))
                  for sid in selected for arm in ARM_NAMES}
    deltas = {
        "search_minus_v25": {sid: state_mean[(sid, "search")] - state_mean[(sid, "v25")]
                              for sid in selected},
        "azv28_minus_v25": {sid: state_mean[(sid, "azv28")] - state_mean[(sid, "v25")]
                             for sid in selected},
        "search_minus_azv28": {sid: state_mean[(sid, "search")] - state_mean[(sid, "azv28")]
                                for sid in selected},
    }

    h_by_sid = {sid: int(z["horizon"][sid]) for sid in selected}
    buckets = {
        "all": lambda h: True,
        "odd": lambda h: h % 2 == 1,
        "even": lambda h: h % 2 == 0,
        "early_h1_h5": lambda h: h <= 5,
        "late_h6_h10": lambda h: h >= 6,
    }
    summaries = {}
    for comparison, by_state in deltas.items():
        summaries[comparison] = {
            bucket: _summary([v for sid, v in by_state.items() if predicate(h_by_sid[sid])])
            for bucket, predicate in buckets.items()
        }
        summaries[comparison]["by_horizon"] = {
            f"h{h:02d}": _summary([v for sid, v in by_state.items() if h_by_sid[sid] == h])
            for h in HORIZONS
        }

    primary = summaries["search_minus_v25"]["all"]
    student = summaries["azv28_minus_v25"]["all"]
    search_certified = bool(primary["mean"] > 0 and primary["t"] >= 2.1)
    student_certified = bool(student["mean"] > 0 and student["t"] >= 2.1)
    if primary["mean"] <= 0:
        diagnosis = "teacher action fails exact-v26 audit"
    elif not search_certified:
        diagnosis = "teacher action exact-v26 advantage is inconclusive"
    elif not student_certified:
        diagnosis = "teacher passes; distillation does not retain a certified advantage"
    else:
        diagnosis = "teacher and student pass; proceed to fixed high-N mixture confirmation"

    action_by_sid = {int(r["sid"]): r for r in actions}
    pred = np.asarray([float(action_by_sid[sid]["approx_predicted_delta_nearest_v25"])
                       for sid in sorted(selected)], dtype=np.float64)
    exact = np.asarray([deltas["search_minus_v25"][sid] for sid in sorted(selected)], dtype=np.float64)
    finite = np.isfinite(pred) & np.isfinite(exact)
    corr = (float(np.corrcoef(pred[finite], exact[finite])[0, 1])
            if finite.sum() > 2 and pred[finite].std() > 0 and exact[finite].std() > 0
            else float("nan"))

    result = {
        "design": {
            "states": len(selected), "states_per_horizon": ARGS.n_per_parity,
            "continuation_repeats": ARGS.repeats, "horizons": list(HORIZONS),
            "teacher_budget": ARGS.budget,
            "exact_continuation": "v25 learner vs v26 opponent, both deployed 48x8",
            "sampling_unit": "mean paired delta per state",
        },
        "completeness": {
            "action_rows": len(action_keys), "eval_rows": len(actual),
            "expected_eval_rows": len(expected), "root_illegal_by_arm": dict(illegal),
        },
        "comparisons": summaries,
        "approx_predicted_vs_exact_correlation": corr,
        "search_certified": search_certified,
        "student_certified": student_certified,
        "gate": "paired mean score advantage > 0 with t >= 2.1 across states",
        "diagnosis": diagnosis,
    }
    out_path = OUT / "result.json"
    out_path.write_text(json.dumps(_json_finite(result), indent=2, allow_nan=False) + "\n")

    print("\n**EXP-075 paired v26 oracle audit:**\n")
    print(f"- completeness: {len(action_keys)}/{len(selected)} action rows; "
          f"{len(actual)}/{len(expected)} paired evaluation rows")
    for comparison in ("search_minus_v25", "azv28_minus_v25", "search_minus_azv28"):
        s = summaries[comparison]["all"]
        print(f"- {comparison}: {s['mean']:+.4f} +/- {s['se']:.4f}/end, "
              f"t={s['t']:+.2f} ({s['n_states']} states)")
    for bucket in ("odd", "even", "early_h1_h5", "late_h6_h10"):
        s = summaries["search_minus_v25"][bucket]
        print(f"- teacher {bucket}: {s['mean']:+.4f} +/- {s['se']:.4f}, t={s['t']:+.2f}")
    print(f"- verdict: **{diagnosis.upper()}**")
    print(f"- machine-readable: `{out_path}`")


if ARGS.phase == "states":
    make_states()
elif ARGS.phase == "actions":
    make_actions()
elif ARGS.phase == "eval":
    evaluate_actions()
else:
    aggregate()
