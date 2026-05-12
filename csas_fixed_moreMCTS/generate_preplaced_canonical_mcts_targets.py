#!/usr/bin/env python3
"""Generate MCTS targets for the six canonical preplacement states."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from common import KEY_COLS, log, set_seed
from generate_mcts_iteration_targets import TerminalMctsTargeter, score_end_value, _soft_topk
from kr_uct_search import _sample_actions, _simulate_candidates, evaluate_states, kr_smooth_scores
from preplaced_value_data import canonical_preplacement_cases


class LocalNoise:
    def __init__(self, path: str, seed: int):
        self.cfg = json.loads(Path(path).read_text())
        self.block = self.cfg.get("local", {})
        self.min_std = float(self.block.get("min_std", self.cfg.get("meta", {}).get("min_std", 1e-3)))
        self.rng = np.random.default_rng(seed)

    def sample(self, center: np.ndarray, n: int) -> np.ndarray:
        center = np.asarray(center, dtype=np.float32).reshape(4)
        block = self.block
        std = np.maximum(np.asarray(block.get("std", [0.0123, 0.003, 0.08, 0.015]), dtype=np.float32), self.min_std)
        if str(block.get("distribution", "gaussian")).lower() == "student_t":
            nu = float(block.get("nu", 5.0))
            z = self.rng.standard_t(df=nu, size=(n, 4)).astype(np.float32)
            scales = std * math.sqrt((nu - 2.0) / nu)
            if "speed_scale" in block:
                scales[0] = max(float(block["speed_scale"]), self.min_std)
            if "angle_speed_range" in block and "angle_scale_range" in block:
                speed_range = np.asarray(block["angle_speed_range"], dtype=np.float32)
                scale_range = np.asarray(block["angle_scale_range"], dtype=np.float32)
                speed = float(np.clip(abs(float(center[0])), float(speed_range[0]), float(speed_range[1])))
                frac = 0.0 if speed_range[1] <= speed_range[0] else (speed - speed_range[0]) / (speed_range[1] - speed_range[0])
                angle_scale = float(scale_range[1] + frac * (scale_range[0] - scale_range[1]))
                scales[1] = max(angle_scale, self.min_std)
            return center[None] + z * scales[None]
        return center[None] + self.rng.normal(0.0, std, size=(n, 4)).astype(np.float32)


class NoisyTerminalMctsTargeter(TerminalMctsTargeter):
    def __init__(self, args):
        super().__init__(args)
        self.noise = LocalNoise(args.noise_config, args.seed)
        self.noise_samples = int(args.noise_samples)
        self.rollout_noise_samples = int(args.rollout_noise_samples)

    def _noisy_posts(self, state: np.ndarray, cond: np.ndarray, action: np.ndarray, n: int) -> np.ndarray:
        actual = self.noise.sample(action, n).astype(np.float32)
        actual[:, 0] = np.clip(actual[:, 0], 0.4, 3.0)
        actual[:, 1] = np.clip(actual[:, 1], -1.2, 1.2)
        actual[:, 2] = np.clip(actual[:, 2], -4.0, 4.0)
        actual[:, 3] = np.clip(actual[:, 3], -0.75, 0.75)
        return _simulate_candidates(state, cond, actual)

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
            next_cond = cur_cond.copy()
            denom = max(1.0, float(shots_in_end - 1))
            next_cond[0] = min(1.0, (cur_shot + 1) / denom)
            next_cond[1] = 1.0 - next_cond[1]
            next_cond[2] = 1.0 - next_cond[2]
            vals = []
            representative_posts = []
            for a in actions:
                posts = self._noisy_posts(cur_state, cur_cond, a, self.rollout_noise_samples)
                representative_posts.append(posts[0])
                if cur_shot + 1 >= int(shots_in_end) - 1:
                    vals.append(float(np.mean([score_end_value(p, perspective_block) for p in posts])))
                else:
                    vals.append(float(np.mean(evaluate_states(self.searcher.value_model, posts, next_cond, self.searcher.device, self.searcher.eval_batch_size))))
            vals = np.asarray(vals, dtype=np.float32)
            maximize = int(round(float(cur_cond[2]))) == int(perspective_block)
            pick = int(np.argmax(vals) if maximize else np.argmin(vals))
            cur_state = representative_posts[pick]
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
        next_cond = cond.copy()
        denom = max(1.0, float(shots_in_end - 1))
        next_cond[0] = min(1.0, (shot_index + 1) / denom)
        next_cond[1] = 1.0 - next_cond[1]
        next_cond[2] = 1.0 - next_cond[2]
        q = []
        for a in actions:
            posts = self._noisy_posts(pre_state, cond, a, self.noise_samples)
            vals = [
                self._rollout_to_end(p, next_cond, shot_index + 1, shots_in_end, perspective_block)
                for p in posts
            ]
            q.append(float(np.mean(vals)))
        q = np.asarray(q, dtype=np.float32)
        smooth = kr_smooth_scores(actions, q, self.searcher.action_mean, self.searcher.action_std, self.args.kernel_bandwidth, self.args.uct_c)
        top_idx, weights = _soft_topk(smooth, self.args.top_k, self.args.policy_temperature)
        value_target = float(np.sum(weights * q[top_idx]))
        return actions, q, smooth, top_idx, weights, value_target


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--value", required=True)
    ap.add_argument("--out-value", required=True)
    ap.add_argument("--out-policy", required=True)
    ap.add_argument("--root-candidates", type=int, default=128)
    ap.add_argument("--rollout-candidates", type=int, default=32)
    ap.add_argument("--top-k", type=int, default=16)
    ap.add_argument("--kernel-bandwidth", type=float, default=0.75)
    ap.add_argument("--uct-c", type=float, default=0.02)
    ap.add_argument("--temperature", type=float, default=1.35)
    ap.add_argument("--std-scale", type=float, default=1.6)
    ap.add_argument("--global-frac", type=float, default=0.20)
    ap.add_argument("--policy-temperature", type=float, default=0.35)
    ap.add_argument("--shots-in-end", type=int, default=10)
    ap.add_argument("--noise-config", default="/mnt/data/curling2/csas_fixed/noise_versions/v1_bowling.json")
    ap.add_argument("--noise-samples", type=int, default=4)
    ap.add_argument("--rollout-noise-samples", type=int, default=2)
    ap.add_argument("--use-noise", action="store_true")
    ap.add_argument(
        "--case-indices",
        default="",
        help="Optional comma-separated 1-based canonical case indices to run. Use this to shard across GPUs.",
    )
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    set_seed(args.seed)
    value_out = Path(args.out_value)
    policy_out = Path(args.out_policy)
    log_path = value_out.with_suffix(".log")
    targeter = NoisyTerminalMctsTargeter(args) if args.use_noise else TerminalMctsTargeter(args)

    value_records = []
    policy_records = []
    selected = set()
    if args.case_indices.strip():
        selected = {int(x) for x in args.case_indices.split(",") if x.strip()}
    cases = list(enumerate(canonical_preplacement_cases(), start=1))
    if selected:
        cases = [(n, case) for n, case in cases if n in selected]
        if not cases:
            raise ValueError(f"No canonical cases selected by --case-indices={args.case_indices!r}")
    for n, case in cases:
        pre_norm = (case["stones_raw"].reshape(-1) / 4095.0).astype("float32")
        cond = case["cond"]
        actions, q, smooth, top_idx, weights, value_target = targeter.root_search(
            pre_norm,
            cond,
            shot_index=0,
            shots_in_end=int(args.shots_in_end),
        )
        base = {
            "CompetitionID": -1,
            "SessionID": 0,
            "GameID": 0,
            "EndID": n,
            "ShotID": 7,
            "TeamID": int(case["thrower_block"]),
            "ShotIndex": 0,
            "ShotsInEnd": int(args.shots_in_end),
            "root_kind": "canonical_preplaced",
            "mode": case["mode"],
            "guard_slot": int(case["guard_slot"]),
        }
        vrec = dict(base)
        for j, val in enumerate(pre_norm):
            vrec[f"x{j}"] = float(val)
        for j, val in enumerate(cond):
            vrec[f"c{j}"] = float(val)
        vrec["terminal_return"] = float(value_target)
        vrec["best_terminal_return"] = float(q[int(smooth.argmax())])
        vrec["mean_terminal_return"] = float(q.mean())
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
        log(f"finished canonical {n}/6 {case['mode']} guard={case['guard_slot']} value={value_target:+.3f}", log_path)

    value_out.parent.mkdir(parents=True, exist_ok=True)
    policy_out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(value_records).to_csv(value_out, index=False)
    pd.DataFrame(policy_records).to_csv(policy_out, index=False)
    log(f"saved value={value_out} rows={len(value_records)} policy={policy_out} rows={len(policy_records)}", log_path)


if __name__ == "__main__":
    main()
