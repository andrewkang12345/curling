#!/usr/bin/env python3
"""EXP-077: independent audit of a baseline-aware response-oracle gate.

EXP-075 showed that the opponent-model tree's internal top-two confidence was
not calibrated to exact improvement over the incumbent.  This script screens a
frozen tree action against the frozen v25 action with a small exact paired test,
falls back to v25 when the screen rejects it, and evaluates that hybrid target
operator with disjoint exact continuation seeds.

The state and frozen-action phases intentionally reuse EXP-075's implementation
with a fresh output directory and fresh seeds.  This file owns only the paired
screen, independent readout, and aggregation phases.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")
import world  # noqa: F401
from world import env_bridge


SIE = 10


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=("gate", "eval", "aggregate"))
    ap.add_argument("--config", default="configs/exp_074_opp_vt_targets.yaml")
    ap.add_argument("--v25", default="checkpoints/csas_world/az_v25_br/best.pt")
    ap.add_argument("--v26", default="checkpoints/csas_world/az_v26_br2/best.pt")
    ap.add_argument("--out-dir", default="eval_out/exp077_paired_gate")
    ap.add_argument("--horizons", default="1,2,3,4,5,6,7,8,9,10")
    ap.add_argument("--gate-repeats", type=int, default=8)
    ap.add_argument("--eval-repeats", type=int, default=16)
    ap.add_argument("--gate-t", type=float, default=0.5)
    ap.add_argument("--gate-seed", type=int, default=1_077_000)
    ap.add_argument("--eval-seed", type=int, default=20_077_000)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    return ap.parse_args()


ARGS = parse_args()
OUT = Path(ARGS.out_dir)
OUT.mkdir(parents=True, exist_ok=True)
HORIZONS = tuple(sorted({int(h) for h in ARGS.horizons.split(",")}))
if not HORIZONS or min(HORIZONS) < 1 or max(HORIZONS) > 10:
    raise ValueError(f"invalid horizons: {HORIZONS}")
if ARGS.gate_repeats < 2 or ARGS.eval_repeats < 2:
    raise ValueError("gate/eval repeats must both be >= 2")
if ARGS.gate_t < 0:
    raise ValueError("gate-t must be non-negative")
if abs(ARGS.eval_seed - ARGS.gate_seed) < 10_000_000:
    raise ValueError("gate/eval seed namespaces must be separated by at least 10,000,000")
if not (0 <= ARGS.shard_id < ARGS.num_shards):
    raise ValueError("shard-id must be in [0, num-shards)")


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
        fh.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        fh.flush()


def _reset_torch(seed: int):
    import torch

    np.random.seed(int(seed) % (2**32 - 1))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _reset_noise(noise, seed: int):
    if noise is not None:
        noise.rng = np.random.default_rng(int(seed))


def _noise_path(cfg_full):
    return cfg_full.csas_path(cfg_full.search.noise_config).as_posix()


def _load_player(ckpt, device, noise_path, seed, name):
    from world.eval.head_to_head import WorldPlayer
    from world.search.noise import make_noise

    return WorldPlayer(ckpt, device, n_candidates=48, name=name,
                       noise=make_noise(noise_path, seed), sel_noise_samples=8)


def _load_inputs():
    state_path = OUT / "states.npz"
    if not state_path.exists():
        raise FileNotFoundError(f"missing state file: {state_path}")
    z = np.load(state_path, allow_pickle=False)
    actions = _load_jsonl(sorted(OUT.glob("actions_shard*of*.jsonl")))
    by_sid = {int(r["sid"]): r for r in actions}
    if len(by_sid) != len(actions):
        raise RuntimeError("duplicate frozen-action rows across shards")
    selected = [sid for sid in range(len(z["horizon"]))
                if int(z["horizon"][sid]) in HORIZONS]
    missing = sorted(set(selected) - set(by_sid))
    if missing:
        raise RuntimeError(f"missing frozen actions for {len(missing)} states: {missing[:20]}")
    extra = sorted(set(by_sid) - set(selected))
    if extra:
        raise RuntimeError(f"unexpected frozen actions outside selected horizons: {extra[:20]}")
    return z, by_sid, selected


def _paired_stats(values):
    a = np.asarray(values, dtype=np.float64)
    mean = float(a.mean())
    se = float(a.std(ddof=1) / math.sqrt(len(a)))
    if np.isfinite(se) and se > 0:
        t = mean / se
    elif mean > 0:
        t = float("inf")
    elif mean < 0:
        t = float("-inf")
    else:
        t = 0.0
    return mean, se, t


def _finite_or_none(value):
    value = float(value)
    return value if np.isfinite(value) else None


def _screen_accept(deltas):
    mean, se, t = _paired_stats(deltas)
    return bool(mean > 0 and t >= ARGS.gate_t), mean, se, t


def _forced_score(x, c, h, learner_block, action, crn_seed,
                  learner, opponent, env_noise):
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
    score = float(env_bridge.score_end(state, int(learner_block)))
    return score, bool(np.asarray(illegal).any())


def _make_exact_context(seed):
    import torch
    from world.config import load_config
    from world.search.noise import make_noise

    cfg_full = load_config(ARGS.config)
    noise_path = _noise_path(cfg_full)
    device = torch.device(ARGS.device if torch.cuda.is_available() else "cpu")
    env_bridge.warm_jax()
    learner = _load_player(ARGS.v25, device, noise_path, seed + 11, "v25")
    opponent = _load_player(ARGS.v26, device, noise_path, seed + 13, "v26")
    env_noise = make_noise(noise_path, seed + 17)
    return learner, opponent, env_noise


def run_gate():
    z, actions, selected = _load_inputs()
    learner, opponent, env_noise = _make_exact_context(ARGS.gate_seed)
    out_path = OUT / f"gate_shard{ARGS.shard_id}of{ARGS.num_shards}.jsonl"
    existing = _load_jsonl([out_path])
    done = {int(r["sid"]) for r in existing}
    if len(done) != len(existing):
        raise RuntimeError(f"duplicate gate rows in {out_path}")

    shard_sids = [sid for sid in selected if sid % ARGS.num_shards == ARGS.shard_id]
    for pos, sid in enumerate(shard_sids, start=1):
        if sid in done:
            continue
        arow = actions[sid]
        x = np.asarray(z["x"][sid], np.float32)
        c = np.asarray(z["c"][sid], np.float32)
        h = int(z["horizon"][sid])
        learner_block = int(z["learner_block"][sid])
        scores = {"v25": [], "search": []}
        illegal = {"v25": [], "search": []}
        for rep in range(ARGS.gate_repeats):
            crn_seed = ARGS.gate_seed + sid * 10007 + rep * 97
            for arm in ("v25", "search"):
                score, ill = _forced_score(
                    x, c, h, learner_block, arow[f"a_{arm}"], crn_seed,
                    learner, opponent, env_noise)
                scores[arm].append(score)
                illegal[arm].append(ill)
        deltas = np.asarray(scores["search"]) - np.asarray(scores["v25"])
        accepted, mean, se, t = _screen_accept(deltas)
        _append_jsonl(out_path, {
            "sid": sid, "horizon": h, "repeats": ARGS.gate_repeats,
            "gate_t_threshold": ARGS.gate_t, "mean_delta": mean, "se_delta": se,
            "t": _finite_or_none(t), "zero_variance": bool(se == 0),
            "accepted": accepted, "scores_v25": scores["v25"],
            "scores_search": scores["search"], "root_illegal_v25": illegal["v25"],
            "root_illegal_search": illegal["search"],
        })
        done.add(sid)
        print(f"[gate] sid={sid} h={h} d={mean:+.3f} t={t:+.2f} "
              f"accepted={int(accepted)} {pos}/{len(shard_sids)}", flush=True)
    print(f"EXP077_GATE_SHARD_DONE shard={ARGS.shard_id}", flush=True)


def _gate_rows(selected):
    rows = _load_jsonl(sorted(OUT.glob("gate_shard*of*.jsonl")))
    by_sid = {int(r["sid"]): r for r in rows}
    if len(by_sid) != len(rows):
        raise RuntimeError("duplicate gate rows across shards")
    missing = sorted(set(selected) - set(by_sid))
    if missing:
        raise RuntimeError(f"missing gate rows for {len(missing)} states: {missing[:20]}")
    extra = sorted(set(by_sid) - set(selected))
    if extra:
        raise RuntimeError(f"unexpected gate rows outside selected horizons: {extra[:20]}")
    return by_sid


def run_eval():
    z, actions, selected = _load_inputs()
    gates = _gate_rows(selected)
    learner, opponent, env_noise = _make_exact_context(ARGS.eval_seed)
    out_path = OUT / f"eval_shard{ARGS.shard_id}of{ARGS.num_shards}.jsonl"
    existing = _load_jsonl([out_path])
    done = {int(r["sid"]) for r in existing}
    if len(done) != len(existing):
        raise RuntimeError(f"duplicate eval rows in {out_path}")

    shard_sids = [sid for sid in selected if sid % ARGS.num_shards == ARGS.shard_id]
    for pos, sid in enumerate(shard_sids, start=1):
        if sid in done:
            continue
        arow = actions[sid]
        x = np.asarray(z["x"][sid], np.float32)
        c = np.asarray(z["c"][sid], np.float32)
        h = int(z["horizon"][sid])
        learner_block = int(z["learner_block"][sid])
        scores = {"v25": [], "search": []}
        illegal = {"v25": [], "search": []}
        for rep in range(ARGS.eval_repeats):
            crn_seed = ARGS.eval_seed + sid * 10007 + rep * 97
            for arm in ("v25", "search"):
                score, ill = _forced_score(
                    x, c, h, learner_block, arow[f"a_{arm}"], crn_seed,
                    learner, opponent, env_noise)
                scores[arm].append(score)
                illegal[arm].append(ill)
        raw_delta = float(np.mean(np.asarray(scores["search"]) - np.asarray(scores["v25"])))
        accepted = bool(gates[sid]["accepted"])
        hybrid_delta = raw_delta if accepted else 0.0
        _append_jsonl(out_path, {
            "sid": sid, "horizon": h, "repeats": ARGS.eval_repeats,
            "accepted": accepted, "raw_mean_delta": raw_delta,
            "hybrid_mean_delta": hybrid_delta, "scores_v25": scores["v25"],
            "scores_search": scores["search"], "root_illegal_v25": illegal["v25"],
            "root_illegal_search": illegal["search"],
        })
        done.add(sid)
        print(f"[eval] sid={sid} h={h} accepted={int(accepted)} "
              f"hybrid_d={hybrid_delta:+.3f} {pos}/{len(shard_sids)}", flush=True)
    print(f"EXP077_EVAL_SHARD_DONE shard={ARGS.shard_id}", flush=True)


def _summary(values):
    a = np.asarray(values, dtype=np.float64)
    n = len(a)
    mean = float(a.mean()) if n else float("nan")
    se = float(a.std(ddof=1) / math.sqrt(n)) if n > 1 else float("inf")
    if np.isfinite(se) and se > 0:
        t = mean / se
    elif n and mean > 0:
        t = float("inf")
    elif n and mean < 0:
        t = float("-inf")
    else:
        t = 0.0
    return {"n_states": n, "mean": mean, "se": se, "t": t}


def _json_finite(value):
    if isinstance(value, dict):
        return {key: _json_finite(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_finite(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def aggregate():
    z, actions, selected_list = _load_inputs()
    selected = set(selected_list)
    gates = _gate_rows(selected_list)
    eval_rows = _load_jsonl(sorted(OUT.glob("eval_shard*of*.jsonl")))
    evals = {int(r["sid"]): r for r in eval_rows}
    if len(evals) != len(eval_rows) or set(evals) != selected:
        missing = sorted(selected - set(evals))
        raise RuntimeError(f"incomplete/duplicate eval rows: {len(evals)}/{len(selected)}, "
                           f"missing={missing[:20]}")

    h_by_sid = {sid: int(z["horizon"][sid]) for sid in selected}
    for sid in selected:
        grow = gates[sid]
        if int(grow["horizon"]) != h_by_sid[sid]:
            raise RuntimeError(f"sid {sid}: gate horizon mismatch")
        if int(grow["repeats"]) != ARGS.gate_repeats:
            raise RuntimeError(f"sid {sid}: gate repeat count mismatch")
        if not np.isclose(float(grow["gate_t_threshold"]), ARGS.gate_t):
            raise RuntimeError(f"sid {sid}: gate threshold mismatch")
        if len(grow["scores_search"]) != ARGS.gate_repeats or len(
                grow["scores_v25"]) != ARGS.gate_repeats:
            raise RuntimeError(f"sid {sid}: incomplete gate score vectors")
        deltas = np.asarray(grow["scores_search"], dtype=np.float64) - np.asarray(
            grow["scores_v25"], dtype=np.float64)
        accepted, mean, se, _t = _screen_accept(deltas)
        if accepted != bool(grow["accepted"]):
            raise RuntimeError(f"sid {sid}: stored gate decision does not reproduce")
        if not np.isclose(mean, float(grow["mean_delta"])) or not np.isclose(
                se, float(grow["se_delta"])):
            raise RuntimeError(f"sid {sid}: stored gate statistics do not reproduce")

    for sid in selected:
        erow = evals[sid]
        if int(erow["horizon"]) != h_by_sid[sid] or int(
                erow["repeats"]) != ARGS.eval_repeats:
            raise RuntimeError(f"sid {sid}: eval design mismatch")
        if bool(erow["accepted"]) != bool(gates[sid]["accepted"]):
            raise RuntimeError(f"sid {sid}: eval/gate decision mismatch")
        if len(erow["scores_search"]) != ARGS.eval_repeats or len(
                erow["scores_v25"]) != ARGS.eval_repeats:
            raise RuntimeError(f"sid {sid}: incomplete eval score vectors")
        exact_raw = float(np.mean(np.asarray(erow["scores_search"], dtype=np.float64)
                                  - np.asarray(erow["scores_v25"], dtype=np.float64)))
        if not np.isclose(exact_raw, float(erow["raw_mean_delta"])):
            raise RuntimeError(f"sid {sid}: stored eval delta does not reproduce")
    accepted = {sid: bool(gates[sid]["accepted"]) for sid in selected}
    raw = {sid: float(evals[sid]["raw_mean_delta"]) for sid in selected}
    hybrid = {sid: (raw[sid] if accepted[sid] else 0.0) for sid in selected}
    rejected = {sid: (raw[sid] if not accepted[sid] else 0.0) for sid in selected}
    buckets = {
        "all": lambda h: True,
        "odd": lambda h: h % 2 == 1,
        "even": lambda h: h % 2 == 0,
        "early_h1_h5": lambda h: h <= 5,
        "late_h6_h10": lambda h: h >= 6,
    }

    def summaries(by_sid):
        out = {
            name: _summary([v for sid, v in by_sid.items() if pred(h_by_sid[sid])])
            for name, pred in buckets.items()
        }
        out["by_horizon"] = {
            f"h{h:02d}": _summary([v for sid, v in by_sid.items() if h_by_sid[sid] == h])
            for h in HORIZONS
        }
        return out

    primary = summaries(hybrid)
    raw_summary = summaries(raw)
    rejected_summary = summaries(rejected)
    p = primary["all"]
    certified = bool(p["mean"] > 0 and p["t"] >= 2.1)
    accept_by_h = {f"h{h:02d}": sum(accepted[sid] for sid in selected if h_by_sid[sid] == h)
                   for h in HORIZONS}

    ordered = sorted(selected)
    target_actions = np.stack([
        np.asarray(actions[sid]["a_search" if accepted[sid] else "a_v25"], np.float32)
        for sid in ordered
    ])
    np.savez_compressed(
        OUT / "screened_targets.npz",
        sid=np.asarray(ordered, np.int64), x=np.asarray(z["x"])[ordered].astype(np.float32),
        c=np.asarray(z["c"])[ordered].astype(np.float32),
        horizon=np.asarray(z["horizon"])[ordered].astype(np.int64),
        action=target_actions, accepted_search=np.asarray([accepted[s] for s in ordered], bool),
    )

    illegal = Counter()
    for phase_rows in (gates.values(), evals.values()):
        for row in phase_rows:
            illegal["v25"] += sum(bool(v) for v in row["root_illegal_v25"])
            illegal["search"] += sum(bool(v) for v in row["root_illegal_search"])
    result = {
        "design": {
            "states": len(selected), "states_per_horizon": len(selected) // len(HORIZONS),
            "gate_repeats": ARGS.gate_repeats, "eval_repeats": ARGS.eval_repeats,
            "gate_t_threshold": ARGS.gate_t, "gate_seed": ARGS.gate_seed,
            "eval_seed": ARGS.eval_seed,
            "gate": (f"{ARGS.gate_repeats} paired exact-v26 CRN continuations; "
                     f"accept iff mean>0 and t>={ARGS.gate_t}"),
            "fallback": "deployed v25 action; one point target at every learner state",
            "independent_readout": (f"disjoint {ARGS.eval_repeats} paired exact-v26 "
                                    "CRN continuations"),
            "sampling_unit": "independent-readout mean delta per state, including zero fallback deltas",
            "threshold_provenance": "selected exploratorily on split EXP-075; EXP-077 is fresh confirmation",
        },
        "completeness": {
            "action_rows": len(actions), "gate_rows": len(gates), "eval_rows": len(evals),
            "root_illegal_draws_by_arm_across_gate_and_eval": dict(illegal),
        },
        "accepted_search_targets": int(sum(accepted.values())),
        "fallback_v25_targets": int(len(accepted) - sum(accepted.values())),
        "accepted_by_horizon": accept_by_h,
        "comparisons": {
            "hybrid_minus_v25_primary": primary,
            "raw_search_minus_v25": raw_summary,
            "rejected_search_minus_v25_zeroed": rejected_summary,
        },
        "certified": certified,
        "primary_gate": "hybrid-v25 paired mean > 0 with t >= 2.1 across fresh states",
        "decision": ("proceed to full paired-gated mixture collection/training"
                     if certified else "do not train; paired screen did not independently certify"),
    }
    out_path = OUT / "result.json"
    out_path.write_text(json.dumps(_json_finite(result), indent=2, allow_nan=False) + "\n")

    r = raw_summary["all"]
    print("\n**EXP-077 independent paired-gate audit:**\n")
    print(f"- completeness: {len(actions)}/{len(selected)} actions; "
          f"{len(gates)}/{len(selected)} gates; {len(evals)}/{len(selected)} readouts")
    print(f"- accepted tree corrections: {sum(accepted.values())}/{len(selected)}; "
          f"fallback v25 anchors: {len(selected) - sum(accepted.values())}/{len(selected)}")
    print(f"- independent hybrid - v25: {p['mean']:+.4f} +/- {p['se']:.4f}/end, "
          f"t={p['t']:+.2f}")
    print(f"- independent raw search - v25: {r['mean']:+.4f} +/- {r['se']:.4f}/end, "
          f"t={r['t']:+.2f}")
    print(f"- gate: **{'CERTIFIED' if certified else 'NOT CERTIFIED'}**")
    print(f"- decision: **{result['decision'].upper()}**")
    print(f"- machine-readable: `{out_path}`")


if ARGS.phase == "gate":
    run_gate()
elif ARGS.phase == "eval":
    run_eval()
else:
    aggregate()
