#!/usr/bin/env python3
"""Generate terminal-return value targets and weighted continuous policy targets.

This is the corrected MCTS-style target generator:
- root states are pre-throw states;
- root candidate Q is the final simulated end score after search-guided rollout;
- policy targets are weighted candidate throws from kernel-smoothed Q scores;
- value targets are expected terminal returns under the weighted search policy.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from common import FIXED_ROOT, KEY_COLS, NUM_STONES, POS_MAX, STONE_COLS, log, set_seed
from dataset import ValueDataset
from kr_uct_search import KRUctSearcher, evaluate_states, kr_smooth_scores, _sample_actions, _simulate_candidates
from preplaced_value_data import load_preplaced_mcts_roots
from train_holdout_models_cond3 import make_holdout_split

BUTTON_RAW = np.array([750.0, 800.0], dtype=np.float32)
M_PER_RAW = 0.003048
HOUSE_RADIUS_M = 1.8288
HOUSE_RADIUS_RAW = HOUSE_RADIUS_M / M_PER_RAW


def _row_positions(row: pd.Series) -> np.ndarray:
    return row[STONE_COLS].to_numpy(dtype=np.float32).reshape(NUM_STONES, 2)


def _row_condition(row: pd.Series) -> np.ndarray:
    return np.asarray([row["shot_norm"], row["team_order"], row["stone_block"]], dtype=np.float32)


def _previous_row(df: pd.DataFrame, idx: int) -> pd.Series | None:
    row = df.iloc[idx]
    prev = df[
        (df["CompetitionID"] == row["CompetitionID"])
        & (df["SessionID"] == row["SessionID"])
        & (df["GameID"] == row["GameID"])
        & (df["EndID"] == row["EndID"])
        & (df["ShotID"] < row["ShotID"])
    ].sort_values("ShotID")
    if prev.empty:
        return None
    return prev.iloc[-1]


def _in_play(stones_raw: np.ndarray) -> np.ndarray:
    stones = stones_raw.reshape(NUM_STONES, 2)
    return ((stones[:, 0] > 0) | (stones[:, 1] > 0)) & (stones[:, 0] < POS_MAX) & (stones[:, 1] < POS_MAX)


def score_end_value(stones_norm: np.ndarray, perspective_block: int) -> float:
    """Curling-rule end score differential for slots 1-6 vs 7-12."""
    stones = np.asarray(stones_norm, dtype=np.float32).reshape(NUM_STONES, 2) * POS_MAX
    live = _in_play(stones)
    if not np.any(live):
        return 0.0
    dist_raw = np.linalg.norm(stones - BUTTON_RAW[None, :], axis=1)
    in_house = live & (dist_raw <= HOUSE_RADIUS_RAW)
    if not np.any(in_house):
        return 0.0
    teams = np.zeros(NUM_STONES, dtype=np.int32)
    teams[6:] = 1
    best_idx = int(np.argmin(np.where(in_house, dist_raw, np.inf)))
    scoring_team = int(teams[best_idx])
    opponent_best = np.min(np.where(in_house & (teams != scoring_team), dist_raw, np.inf))
    points = int(np.sum(in_house & (teams == scoring_team) & (dist_raw < opponent_best)))
    sign = 1.0 if scoring_team == int(perspective_block) else -1.0
    return sign * float(points)


def _soft_topk(scores: np.ndarray, top_k: int, temperature: float) -> tuple[np.ndarray, np.ndarray]:
    top_k = min(int(top_k), len(scores))
    idx = np.argsort(scores)[-top_k:][::-1]
    z = scores[idx] / max(float(temperature), 1e-6)
    z = z - np.max(z)
    w = np.exp(z)
    w = w / np.maximum(w.sum(), 1e-12)
    return idx, w.astype(np.float32)


class TerminalMctsTargeter:
    def __init__(self, args):
        self.args = args
        self.searcher = KRUctSearcher(
            args.policy,
            args.value,
            device=args.device,
            candidates=args.root_candidates,
            rollout_depth=1,
            kernel_bandwidth=args.kernel_bandwidth,
            uct_c=args.uct_c,
            temperature=args.temperature,
            std_scale=args.std_scale,
            global_frac=args.global_frac,
        )

    def _rollout_to_end(self, state: np.ndarray, cond: np.ndarray, shot_index: int, shots_in_end: int,
                        perspective_block: int) -> float:
        cur_state = state
        cur_cond = cond.copy()
        cur_shot = int(shot_index)
        while cur_shot >= 0 and cur_shot < int(shots_in_end):
            if cur_shot >= int(shots_in_end) - 1:
                return score_end_value(cur_state, perspective_block)
            n = int(self.args.rollout_candidates)
            actions = _sample_actions(
                self.searcher.policy,
                self.searcher.action_mean_t,
                self.searcher.action_std_t,
                cur_state,
                cur_cond,
                n,
                self.searcher.device,
                self.args.temperature,
                self.args.std_scale,
                self.args.global_frac,
            )
            posts = _simulate_candidates(cur_state, cur_cond, actions)
            next_cond = cur_cond.copy()
            denom = max(1.0, float(shots_in_end - 1))
            next_cond[0] = min(1.0, (cur_shot + 1) / denom)
            next_cond[1] = 1.0 - next_cond[1]
            next_cond[2] = 1.0 - next_cond[2]
            if cur_shot + 1 >= int(shots_in_end) - 1:
                vals = np.asarray([score_end_value(p, perspective_block) for p in posts], dtype=np.float32)
            else:
                vals = evaluate_states(
                    self.searcher.value_model,
                    posts,
                    next_cond,
                    self.searcher.device,
                    self.searcher.eval_batch_size,
                )
            maximize = int(round(float(cur_cond[2]))) == int(perspective_block)
            pick = int(np.argmax(vals) if maximize else np.argmin(vals))
            cur_state = posts[pick]
            cur_cond = next_cond
            cur_shot += 1
        return score_end_value(cur_state, perspective_block)

    def root_search(self, pre_state: np.ndarray, cond: np.ndarray, shot_index: int, shots_in_end: int):
        perspective_block = int(round(float(cond[2])))
        actions = _sample_actions(
            self.searcher.policy,
            self.searcher.action_mean_t,
            self.searcher.action_std_t,
            pre_state,
            cond,
            int(self.args.root_candidates),
            self.searcher.device,
            self.args.temperature,
            self.args.std_scale,
            self.args.global_frac,
        )
        posts = _simulate_candidates(pre_state, cond, actions)
        next_cond = cond.copy()
        denom = max(1.0, float(shots_in_end - 1))
        next_cond[0] = min(1.0, (shot_index + 1) / denom)
        next_cond[1] = 1.0 - next_cond[1]
        next_cond[2] = 1.0 - next_cond[2]
        q = np.asarray(
            [
                self._rollout_to_end(posts[i], next_cond, shot_index + 1, shots_in_end, perspective_block)
                for i in range(len(actions))
            ],
            dtype=np.float32,
        )
        smooth = kr_smooth_scores(
            actions,
            q,
            self.searcher.action_mean,
            self.searcher.action_std,
            self.args.kernel_bandwidth,
            self.args.uct_c,
        )
        top_idx, weights = _soft_topk(smooth, self.args.top_k, self.args.policy_temperature)
        value_target = float(np.sum(weights * q[top_idx]))
        return actions, q, smooth, top_idx, weights, value_target


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", type=int, default=0)
    ap.add_argument("--policy", required=True)
    ap.add_argument("--value", required=True)
    ap.add_argument("--out-value", required=True)
    ap.add_argument("--out-policy", required=True)
    ap.add_argument("--split", default="train", choices=["train", "val", "test"])
    ap.add_argument("--include-preplaced-roots", action="store_true")
    ap.add_argument("--preplaced-only", action="store_true")
    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--root-candidates", type=int, default=96)
    ap.add_argument("--rollout-candidates", type=int, default=24)
    ap.add_argument("--top-k", type=int, default=16)
    ap.add_argument("--kernel-bandwidth", type=float, default=0.75)
    ap.add_argument("--uct-c", type=float, default=0.02)
    ap.add_argument("--temperature", type=float, default=1.35)
    ap.add_argument("--std-scale", type=float, default=1.6)
    ap.add_argument("--global-frac", type=float, default=0.20)
    ap.add_argument("--policy-temperature", type=float, default=0.35)
    ap.add_argument("--early-mid-oversample", type=float, default=1.5)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    set_seed(args.seed + args.shard_index)
    value_out = Path(args.out_value)
    policy_out = Path(args.out_policy)
    log_path = value_out.with_suffix(".log")
    ds = ValueDataset(str(FIXED_ROOT / "2026" / "Stones.csv"), str(FIXED_ROOT / "2026" / "Ends.csv"), augment_positions=False, augment_flip=False)
    train_idx, val_idx, test_idx, _ = make_holdout_split(ds.df, args.holdout, 0.10, 123)
    base_idx = {"train": train_idx, "val": val_idx, "test": test_idx}[args.split]
    roots = []
    if not args.preplaced_only:
        rows = []
        for idx in np.asarray(base_idx, dtype=np.int64):
            row = ds.df.iloc[int(idx)]
            if _previous_row(ds.df, int(idx)) is None:
                continue
            rows.append(int(idx))
        rows = np.asarray(rows, dtype=np.int64)
        shot_norm = ds.df.iloc[rows]["shot_norm"].to_numpy(dtype=np.float32)
        if args.early_mid_oversample > 1.0:
            early = rows[shot_norm <= 0.60]
            extra = np.random.choice(early, size=int((args.early_mid_oversample - 1.0) * len(early)), replace=True) if len(early) else np.array([], dtype=np.int64)
            rows = np.concatenate([rows, extra])
        for idx in rows:
            roots.append(("real", int(idx)))
    if args.include_preplaced_roots or args.preplaced_only:
        train_comps = set(int(x) for x in pd.unique(ds.df.iloc[train_idx]["CompetitionID"]).tolist())
        if args.split != "train":
            split_idx = {"val": val_idx, "test": test_idx}.get(args.split, train_idx)
            train_comps = set(int(x) for x in pd.unique(ds.df.iloc[split_idx]["CompetitionID"]).tolist())
        roots.extend(("preplaced", r) for r in load_preplaced_mcts_roots(train_comps, ds.df))
    roots = [r for i, r in enumerate(roots) if i % args.num_shards == args.shard_index]
    if args.max_rows > 0:
        roots = roots[: args.max_rows]

    log(f"mcts roots={len(roots)} shard={args.shard_index}/{args.num_shards} preplaced_only={args.preplaced_only}", log_path)
    targeter = TerminalMctsTargeter(args)
    value_records = []
    policy_records = []
    for n, (root_kind, payload) in enumerate(roots, start=1):
        if root_kind == "real":
            row = ds.df.iloc[int(payload)]
            prev = _previous_row(ds.df, int(payload))
            if prev is None:
                continue
            pre_raw = _row_positions(prev)
            pre_norm = (pre_raw.reshape(-1) / POS_MAX).astype(np.float32)
            cond = _row_condition(row)
            shot_index = int(row["ShotIndex"])
            shots_in_end = int(row["ShotsInEnd"])
            base = {k: int(row[k]) for k in KEY_COLS}
            base["TeamID"] = int(row["TeamID"])
        else:
            root = payload
            pre_norm = root["state_norm"]
            cond = root["cond"]
            shot_index = int(root["ShotIndex"])
            shots_in_end = int(root["ShotsInEnd"])
            base = {k: int(root[k]) for k in KEY_COLS}
            base["TeamID"] = int(root["TeamID"])
            base["root_kind"] = "preplaced"
            base["mode"] = root["mode"]
            base["guard_slot"] = int(root["guard_slot"])
        actions, q, smooth, top_idx, weights, value_target = targeter.root_search(pre_norm, cond, shot_index, shots_in_end)
        base["ShotIndex"] = shot_index
        base["ShotsInEnd"] = shots_in_end
        vrec = dict(base)
        for j, val in enumerate(pre_norm):
            vrec[f"x{j}"] = float(val)
        for j, val in enumerate(cond):
            vrec[f"c{j}"] = float(val)
        vrec["terminal_return"] = value_target
        vrec["best_terminal_return"] = float(q[int(np.argmax(smooth))])
        vrec["mean_terminal_return"] = float(np.mean(q))
        value_records.append(vrec)

        for rank, (aidx, w) in enumerate(zip(top_idx, weights)):
            prec = dict(base)
            for j, val in enumerate(pre_norm):
                prec[f"x{j}"] = float(val)
            for j, val in enumerate(cond):
                prec[f"c{j}"] = float(val)
            prec["rank"] = int(rank)
            prec["weight"] = float(w)
            prec["q_terminal"] = float(q[aidx])
            prec["search_score"] = float(smooth[aidx])
            prec["est_speed"] = float(actions[aidx, 0])
            prec["est_angle"] = float(actions[aidx, 1])
            prec["est_spin"] = float(actions[aidx, 2])
            prec["est_y0"] = float(actions[aidx, 3])
            policy_records.append(prec)
        if n % 10 == 0:
            log(f"finished {n}/{len(rows)} value={value_target:+.3f}", log_path)
            pd.DataFrame(value_records).to_csv(value_out, index=False)
            pd.DataFrame(policy_records).to_csv(policy_out, index=False)
    value_out.parent.mkdir(parents=True, exist_ok=True)
    policy_out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(value_records).to_csv(value_out, index=False)
    pd.DataFrame(policy_records).to_csv(policy_out, index=False)
    log(f"saved value={value_out} rows={len(value_records)} policy={policy_out} rows={len(policy_records)}", log_path)


if __name__ == "__main__":
    main()
