from __future__ import annotations

import json
import math
import os
import random
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "inverse"))
sys.path.insert(0, str(ROOT_DIR / "valueModel"))

from inverse.curling_sim_jax import CurlingParams, simulate_from_params  # type: ignore
from score_shots_mc_seq import POS_MAX, evaluate_state_value, load_value_model  # type: ignore
from sim_presets import contact_mild_params  # type: ignore

STATIC_DIR = THIS_DIR / "static"
NOISE_PATH = ROOT_DIR / "noise_versions" / "v1_bowling.json"
DEFAULT_DEVICE = os.environ.get("CURLING_GAME_DEVICE", "cpu")
MAX_SCENARIOS = int(os.environ.get("THROW_QUIZ_MAX_SCENARIOS", "120"))
POOL_SIZE = int(os.environ.get("THROW_QUIZ_POOL_SIZE", "160"))
MIN_CLEAR = 2 * 0.145
SEPARATE_PASSES = 6
SHOT_STAGE_VALUES = [1 / 9, 2 / 9, 3 / 9, 4 / 9, 5 / 9, 6 / 9, 7 / 9, 8 / 9, 1.0]


@dataclass(frozen=True)
class NoiseConfig:
    nu: float
    speed_scale: float
    angle_scale_min: float
    angle_scale_max: float
    angle_speed_min: float
    angle_speed_max: float
    spin_std: float
    y0_std: float


class SelectRequest(BaseModel):
    scenario_id: str
    option_id: str
    seed: int | None = None


def _load_noise_config() -> NoiseConfig:
    data = json.loads(NOISE_PATH.read_text())
    local = data["local"]
    return NoiseConfig(
        nu=float(local["nu"]),
        speed_scale=float(local["speed_scale"]),
        angle_scale_min=float(local["angle_scale_range"][0]),
        angle_scale_max=float(local["angle_scale_range"][1]),
        angle_speed_min=float(local["angle_speed_range"][0]),
        angle_speed_max=float(local["angle_speed_range"][1]),
        spin_std=float(local["std"][2]),
        y0_std=float(local["std"][3]),
    )


def _angle_scale_for_speed(speed: float, cfg: NoiseConfig) -> float:
    speed_clamped = float(np.clip(speed, cfg.angle_speed_min, cfg.angle_speed_max))
    alpha = (speed_clamped - cfg.angle_speed_min) / (cfg.angle_speed_max - cfg.angle_speed_min)
    return float(cfg.angle_scale_max + alpha * (cfg.angle_scale_min - cfg.angle_scale_max))


def _sample_noisy_params(intended: np.ndarray, cfg: NoiseConfig, rng: np.random.Generator) -> np.ndarray:
    speed, angle, spin, y0 = [float(x) for x in intended]
    noisy = np.array(
        [
            speed + rng.standard_t(cfg.nu) * cfg.speed_scale,
            angle + rng.standard_t(cfg.nu) * _angle_scale_for_speed(speed, cfg),
            spin + rng.normal(0.0, cfg.spin_std),
            y0 + rng.normal(0.0, cfg.y0_std),
        ],
        dtype=np.float32,
    )
    noisy[0] = float(np.clip(noisy[0], 0.1, 3.0))
    noisy[1] = float(np.clip(noisy[1], -0.35, 0.35))
    noisy[2] = float(np.clip(noisy[2], -3.0, 3.0))
    noisy[3] = float(np.clip(noisy[3], -0.23, 0.23))
    return noisy


def _settf_gaussian_checkpoint_map() -> dict[int, Path]:
    out: dict[int, Path] = {}
    for split_path in (ROOT_DIR / "holdouts").glob("*/model_settf_gaussian/split_summary.json"):
        try:
            data = json.loads(split_path.read_text())
            holdout_comp = int(data["holdout_competition"])
            ckpt = split_path.parent / "model.pt"
            if ckpt.exists():
                out[holdout_comp] = ckpt
        except Exception:
            continue
    if not out:
        raise FileNotFoundError("No Gaussian SetTransformer holdout checkpoints found.")
    return out


@lru_cache(maxsize=8)
def _load_model_for_competition(competition_id: int):
    ckpt = _settf_gaussian_checkpoint_map().get(int(competition_id))
    if ckpt is None:
        raise FileNotFoundError(f"No Gaussian SetTransformer checkpoint for competition {competition_id}.")
    model_fn, _ = load_value_model(ckpt, device=DEFAULT_DEVICE)
    return model_fn, ckpt


def _stone_cols(prefix: str = "stone") -> list[str]:
    cols: list[str] = []
    for i in range(1, 13):
        cols.extend([f"{prefix}_{i}_x", f"{prefix}_{i}_y"])
    return cols


def _board_from_row(row: pd.Series, prefix: str = "stone") -> np.ndarray:
    board = np.full((12, 2), np.nan, dtype=np.float32)
    for i in range(1, 13):
        x = row.get(f"{prefix}_{i}_x", np.nan)
        y = row.get(f"{prefix}_{i}_y", np.nan)
        if pd.isna(x) or pd.isna(y):
            continue
        x = float(x)
        y = float(y)
        if x in (0.0, POS_MAX) or y in (0.0, POS_MAX):
            continue
        board[i - 1, 0] = (800.0 - y) * 0.003048
        board[i - 1, 1] = (x - 750.0) * 0.003048
    return board


def _board_to_client(board_m: np.ndarray) -> list[dict[str, float | int | str]]:
    out: list[dict[str, float | int | str]] = []
    for slot in range(12):
        if np.isfinite(board_m[slot, 0]) and np.isfinite(board_m[slot, 1]):
            out.append(
                {
                    "slot": int(slot),
                    "team": "A" if slot < 6 else "B",
                    "x": float(board_m[slot, 0]),
                    "y": float(board_m[slot, 1]),
                }
            )
    return out


def _occupied_slots(board_m: np.ndarray) -> list[int]:
    return [i for i in range(12) if np.isfinite(board_m[i, 0]) and np.isfinite(board_m[i, 1])]


def _separate_overlaps(pts: np.ndarray, min_gap: float = MIN_CLEAR, passes: int = SEPARATE_PASSES) -> np.ndarray:
    if pts.size == 0:
        return pts
    out = pts.copy()
    n = out.shape[0]
    for _ in range(passes):
        moved = False
        for i in range(n):
            for j in range(i + 1, n):
                dx = out[j, 0] - out[i, 0]
                dy = out[j, 1] - out[i, 1]
                d = math.hypot(dx, dy)
                if d < 1e-9:
                    dx, dy, d = 1e-6, 0.0, 1e-6
                if d < min_gap:
                    push = 0.5 * (min_gap - d)
                    nx, ny = dx / d, dy / d
                    out[i, 0] -= push * nx
                    out[i, 1] -= push * ny
                    out[j, 0] += push * nx
                    out[j, 1] += push * ny
                    moved = True
        if not moved:
            break
    return out


def _sanitize_board(board_m: np.ndarray) -> np.ndarray:
    out = board_m.copy()
    slots = _occupied_slots(out)
    if slots:
        out[slots] = _separate_overlaps(out[slots].astype(np.float32))
    return out


def _next_slot_for_block(board_m: np.ndarray, stone_block: float) -> int:
    start = 0 if int(round(float(stone_block))) == 0 else 6
    for slot in range(start, start + 6):
        if not (np.isfinite(board_m[slot, 0]) and np.isfinite(board_m[slot, 1])):
            return slot
    return start + 5


def _slotted_board_from_compact(final_compact: np.ndarray, prev_slots: list[int], new_slot: int) -> np.ndarray:
    board = np.full((12, 2), np.nan, dtype=np.float32)
    n_prev = len(prev_slots)
    for idx, slot in enumerate(prev_slots):
        if idx < final_compact.shape[0]:
            xy = final_compact[idx]
            if np.isfinite(xy[0]) and np.isfinite(xy[1]):
                board[slot] = xy
    if final_compact.shape[0] > n_prev:
        xy = final_compact[-1]
        if np.isfinite(xy[0]) and np.isfinite(xy[1]):
            board[new_slot] = xy
    return board


def _frame_to_jsonable(board: np.ndarray) -> list[list[float] | None]:
    out: list[list[float] | None] = []
    for slot in range(12):
        if np.isfinite(board[slot, 0]) and np.isfinite(board[slot, 1]):
            out.append([float(board[slot, 0]), float(board[slot, 1])])
        else:
            out.append(None)
    return out


def _sample_trajectory_frames(traj_compact: np.ndarray, prev_slots: list[int], new_slot: int) -> dict[str, Any]:
    keep = max(1, int(math.ceil(len(traj_compact) / 80)))
    sampled = traj_compact[::keep]
    if not np.array_equal(sampled[-1], traj_compact[-1]):
        sampled = np.concatenate([sampled, traj_compact[-1:]], axis=0)
    return {
        "stone_slot": int(new_slot),
        "frames": [_frame_to_jsonable(_slotted_board_from_compact(frame, prev_slots, new_slot)) for frame in sampled],
    }


def _param_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(
        abs(float(a[0] - b[0])) / 0.12
        + abs(float(a[1] - b[1])) / 0.014
        + abs(float(a[2] - b[2])) / 0.55
        + abs(float(a[3] - b[3])) / 0.045
    )


def _trajectory_signature(candidate: dict[str, Any]) -> tuple[float, float, float, float] | None:
    traj = candidate.get("intended_trajectory", {})
    frames = traj.get("frames") or []
    slot = traj.get("stone_slot")
    if slot is None or len(frames) < 3:
        return None
    slot = int(slot)
    mid = frames[len(frames) // 2][slot]
    end = frames[-1][slot]
    if mid is None or end is None:
        return None
    return float(mid[0]), float(mid[1]), float(end[0]), float(end[1])


def _trajectory_distance(a: dict[str, Any], b: dict[str, Any]) -> float:
    sa = _trajectory_signature(a)
    sb = _trajectory_signature(b)
    if sa is None or sb is None:
        return 0.0
    mid_dist = math.hypot(sa[0] - sb[0], sa[1] - sb[1])
    end_dist = math.hypot(sa[2] - sb[2], sa[3] - sb[3])
    return float(0.45 * mid_dist + end_dist)


def _raw_distinct(a: np.ndarray, b: np.ndarray) -> bool:
    hits = 0
    hits += int(abs(float(a[0] - b[0])) >= 0.12)
    hits += int(abs(float(a[1] - b[1])) >= 0.022)
    hits += int(abs(float(a[2] - b[2])) >= 0.75)
    hits += int(abs(float(a[3] - b[3])) >= 0.065)
    return hits >= 2 or abs(float(a[1] - b[1])) >= 0.040 or abs(float(a[3] - b[3])) >= 0.115


def _thrower_endpoint(candidate: dict[str, Any]) -> tuple[float, float] | None:
    traj = candidate.get("intended_trajectory", {})
    frames = traj.get("frames") or []
    slot = traj.get("stone_slot")
    if slot is None or not frames:
        return None
    xy = frames[-1][int(slot)]
    if xy is None:
        return None
    return float(xy[0]), float(xy[1])


def _endpoint_distance(a: dict[str, Any], b: dict[str, Any]) -> float:
    pa = _thrower_endpoint(a)
    pb = _thrower_endpoint(b)
    if pa is None or pb is None:
        return 0.0
    return float(math.hypot(pa[0] - pb[0], pa[1] - pb[1]))


def _candidate_distinct(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return _endpoint_distance(a, b) >= 0.60


class QuizEngine:
    def __init__(self) -> None:
        self.noise = _load_noise_config()
        self.model_paths = _settf_gaussian_checkpoint_map()
        self.sim_params = contact_mild_params(CurlingParams)
        self.scenario_rows = self._load_real_shot_scenarios()
        self.scenarios = {row["id"]: row for row in self.scenario_rows}
        self.candidate_cache: dict[str, list[dict[str, Any]]] = {}

    def _load_real_shot_scenarios(self) -> list[dict[str, Any]]:
        key_cols = ["CompetitionID", "SessionID", "GameID", "EndID", "ShotID"]
        score_frames = [pd.read_csv(p) for p in sorted((ROOT_DIR / "holdouts").glob("*/scoring_settf_gaussian/shot_scores_local.csv"))]
        if not score_frames:
            raise FileNotFoundError("No Gaussian SetTransformer local score files found.")
        scores = pd.concat(score_frames, ignore_index=True)

        stones = pd.read_csv(ROOT_DIR / "2026" / "Stones.csv")
        stones = stones.sort_values(key_cols).reset_index(drop=True)
        prev_stones = stones.groupby(["CompetitionID", "SessionID", "GameID", "EndID"], dropna=False)[_stone_cols()].shift(1)
        prev_stones.columns = [f"prev_{c}" for c in prev_stones.columns]
        stones = pd.concat([stones, prev_stones], axis=1)
        merged = scores.merge(stones, on=key_cols + ["TeamID", "PlayerID", "Task", "Handle"], how="inner")

        name_frames = []
        for p in sorted((ROOT_DIR / "holdouts").glob("*/reports/coach_report_mc/shot_scores_local_vs_global_merged_settf_gaussian.csv")):
            df = pd.read_csv(p, usecols=["CompetitionID", "TeamID", "PlayerID", "player_name", "player_label", "team_name"])
            name_frames.append(df.drop_duplicates())
        if name_frames:
            name_df = pd.concat(name_frames, ignore_index=True).drop_duplicates(subset=["CompetitionID", "TeamID", "PlayerID"])
            merged = merged.merge(name_df, on=["CompetitionID", "TeamID", "PlayerID"], how="left")
        else:
            merged["player_name"] = np.nan
            merged["player_label"] = np.nan
            merged["team_name"] = np.nan

        teams = pd.read_csv(ROOT_DIR / "2026" / "Teams.csv")
        merged = merged.merge(
            teams.rename(columns={"Name": "team_name_fallback"}),
            on=["CompetitionID", "TeamID"],
            how="left",
        )
        merged["team_name"] = merged["team_name"].fillna(merged["team_name_fallback"]).fillna("Unknown")
        merged["player_name"] = merged["player_name"].fillna("Player " + merged["PlayerID"].astype(str))
        merged["player_label"] = merged["player_label"].fillna(merged["player_name"] + " (" + merged["team_name"] + ")")

        valid = merged[
            np.isfinite(merged["est_speed"])
            & np.isfinite(merged["est_angle"])
            & np.isfinite(merged["est_spin"])
            & np.isfinite(merged["est_y0"])
            & np.isfinite(merged["v_prev"])
        ].copy()
        valid["abs_dv_obs"] = valid["dv_obs"].abs()
        valid["shot_stage"] = valid["shot_norm_next"].apply(lambda x: min(SHOT_STAGE_VALUES, key=lambda y: abs(float(x) - y)))
        per_stage = max(1, MAX_SCENARIOS // len(SHOT_STAGE_VALUES))
        picked = []
        for stage in SHOT_STAGE_VALUES:
            stage_df = valid[valid["shot_stage"] == stage].sort_values(
                ["abs_dv_obs", "CompetitionID", "GameID", "EndID", "ShotID"],
                ascending=[False, True, True, True, True],
            )
            picked.append(stage_df.head(per_stage))
        valid = pd.concat(picked, ignore_index=True).head(MAX_SCENARIOS)

        rows: list[dict[str, Any]] = []
        for _, row in valid.iterrows():
            pre_board = _sanitize_board(_board_from_row(row, prefix="prev_stone"))
            post_board = _board_from_row(row, prefix="stone")
            scenario_id = f"{int(row.CompetitionID)}-{int(row.SessionID)}-{int(row.GameID)}-{int(row.EndID)}-{int(row.ShotID)}"
            rows.append(
                {
                    "id": scenario_id,
                    "competition_id": int(row.CompetitionID),
                    "session_id": int(row.SessionID),
                    "game_id": int(row.GameID),
                    "end_id": int(row.EndID),
                    "shot_id": int(row.ShotID),
                    "athlete_name": str(row.player_name),
                    "athlete_label": str(row.player_label),
                    "team_name": str(row.team_name),
                    "task": int(row.Task),
                    "handle": int(row.Handle),
                    "team_order": float(row.team_order),
                    "stone_block": float(row.stone_block),
                    "shot_norm_next": float(row.shot_norm_next),
                    "v_prev": float(row.v_prev),
                    "v_next_observed": float(row.v_next),
                    "athlete_dv": float(row.dv_obs),
                    "model_path": str(self.model_paths[int(row.CompetitionID)]),
                    "defaults": np.array([float(row.est_speed), float(row.est_angle), float(row.est_spin), float(row.est_y0)], dtype=np.float32),
                    "pre_board_m": pre_board,
                    "post_board_m": post_board,
                }
            )
        if not rows:
            raise RuntimeError("No usable throw quiz scenarios were loaded.")
        return rows

    def _simulate_board(self, scenario: dict[str, Any], params: np.ndarray) -> tuple[np.ndarray, list[int], int, np.ndarray]:
        pre_board = _sanitize_board(scenario["pre_board_m"])
        prev_slots = _occupied_slots(pre_board)
        prev_compact = pre_board[prev_slots].astype(np.float32)
        new_slot = _next_slot_for_block(pre_board, scenario["stone_block"])
        traj = simulate_from_params(
            self.sim_params,
            jax.device_put(prev_compact),
            jax.device_put(params.astype(np.float32)),
            dynamic=True,
        )
        traj_np = np.asarray(jax.device_get(traj), dtype=np.float32)
        final_board = _slotted_board_from_compact(traj_np[-1], prev_slots, new_slot)
        return final_board, prev_slots, new_slot, traj_np

    def _state_value(self, board: np.ndarray, scenario: dict[str, Any]) -> float:
        raw_defaults = np.full((12, 2), POS_MAX, dtype=np.float32)
        c_vec = np.array([scenario["shot_norm_next"], scenario["team_order"], scenario["stone_block"]], dtype=np.float32)
        model_fn, _ = _load_model_for_competition(int(scenario["competition_id"]))
        return float(
            evaluate_state_value(
                model_fn,
                board,
                raw_defaults,
                c_vec,
                float(scenario["stone_block"]),
                float("nan"),
                float("nan"),
                float(scenario["shot_norm_next"]),
                use_rule_based_terminal=True,
            )
        )

    def _make_params(self, base: np.ndarray, rng: np.random.Generator, idx: int) -> np.ndarray:
        presets = np.array(
            [
                [0.00, 0.000, 0.00, 0.000],
                [0.24, 0.045, 1.00, 0.105],
                [-0.20, -0.045, -1.00, -0.105],
                [0.16, -0.060, 1.50, 0.145],
                [-0.16, 0.060, -1.50, -0.145],
                [0.30, 0.020, -1.50, 0.165],
                [-0.26, -0.020, 1.50, -0.165],
                [0.42, 0.085, 2.25, 0.220],
                [-0.34, -0.085, -2.25, -0.220],
                [0.38, -0.095, 2.50, -0.210],
                [-0.32, 0.095, -2.50, 0.210],
                [0.55, 0.000, -2.75, 0.000],
                [-0.45, 0.000, 2.75, 0.000],
            ],
            dtype=np.float32,
        )
        if idx < presets.shape[0]:
            params = base + presets[idx]
            params[0] = float(np.clip(params[0], 0.1, 3.0))
            params[1] = float(np.clip(params[1], -0.35, 0.35))
            params[2] = float(np.clip(params[2], -3.0, 3.0))
            params[3] = float(np.clip(params[3], -0.23, 0.23))
            return params.astype(np.float32)
        wide = idx % 3 == 0
        speed_sd = 0.30 if wide else 0.15
        angle_sd = 0.055 if wide else 0.026
        spin_sd = 1.35 if wide else 0.65
        y0_sd = 0.130 if wide else 0.065
        params = np.array(
            [
                base[0] + rng.normal(0.0, speed_sd),
                base[1] + rng.normal(0.0, angle_sd),
                base[2] + rng.normal(0.0, spin_sd),
                base[3] + rng.normal(0.0, y0_sd),
            ],
            dtype=np.float32,
        )
        if idx % 7 == 0:
            params[2] = rng.choice(np.array([-1.5, -0.75, 0.0, 0.75, 1.5], dtype=np.float32))
        params[0] = float(np.clip(params[0], 0.1, 3.0))
        params[1] = float(np.clip(params[1], -0.35, 0.35))
        params[2] = float(np.clip(params[2], -3.0, 3.0))
        params[3] = float(np.clip(params[3], -0.23, 0.23))
        return params

    def _generate_candidates(self, scenario: dict[str, Any]) -> list[dict[str, Any]]:
        cached = self.candidate_cache.get(scenario["id"])
        if cached is not None:
            return cached

        seed = abs(hash(scenario["id"])) % (2**32)
        rng = np.random.default_rng(seed)
        base = scenario["defaults"].astype(np.float32)
        pool: list[dict[str, Any]] = []
        seen: set[tuple[float, float, float, float]] = set()
        for idx in range(POOL_SIZE):
            params = self._make_params(base, rng, idx)
            key = tuple(round(float(x), 4) for x in params)
            if key in seen:
                continue
            seen.add(key)
            try:
                final_board, prev_slots, new_slot, traj_np = self._simulate_board(scenario, params)
                post_value = self._state_value(final_board, scenario)
            except Exception:
                continue
            if not np.isfinite(post_value):
                continue
            pool.append(
                {
                    "params": params,
                    "post_value": float(post_value),
                    "decision_value": float(post_value - scenario["v_prev"]),
                    "intended_final_board": _board_to_client(final_board),
                    "intended_trajectory": _sample_trajectory_frames(traj_np, prev_slots, new_slot),
                }
            )

        if len(pool) < 3:
            raise HTTPException(status_code=500, detail=f"Only generated {len(pool)} viable throw options.")

        observed_option = self._observed_option(scenario)
        pool.sort(key=lambda c: c["decision_value"], reverse=True)
        best_dv = float(pool[0]["decision_value"])
        near = [c for c in pool if c["decision_value"] >= best_dv - 0.35]
        if len(near) < 8:
            near = [c for c in pool if c["decision_value"] >= best_dv - 0.70]
        near_wide = [c for c in pool if c["decision_value"] >= best_dv - 1.25]
        if len(near) < 3:
            near = pool[: min(len(pool), 20)]

        selected: list[dict[str, Any]] = [near[0]]
        selected_keys = {tuple(round(float(x), 4) for x in near[0]["params"])}

        def choose_diverse_candidate(
            source: list[dict[str, Any]],
            strict: bool,
            min_endpoint_distance: float = 0.0,
        ) -> dict[str, Any] | None:
            best_candidate: dict[str, Any] | None = None
            best_score = -float("inf")
            for cand in source:
                key = tuple(round(float(x), 4) for x in cand["params"])
                if key in selected_keys:
                    continue
                endpoint_distances = [_endpoint_distance(cand, chosen) for chosen in selected]
                trajectory_distances = [_trajectory_distance(cand, chosen) for chosen in selected]
                if strict and not all(_candidate_distinct(cand, chosen) for chosen in selected):
                    continue
                min_endpoint_dist = min(endpoint_distances)
                min_trajectory_dist = min(trajectory_distances)
                if min_endpoint_dist < min_endpoint_distance:
                    continue
                value_penalty = abs(float(cand["decision_value"]) - best_dv)
                score = 14.0 * min_endpoint_dist + 3.0 * min_trajectory_dist - 0.20 * value_penalty
                if score > best_score:
                    best_score = score
                    best_candidate = cand
            return best_candidate

        while len(selected) < 3:
            best_candidate = (
                choose_diverse_candidate(near, strict=True)
                or choose_diverse_candidate(near_wide, strict=True)
                or choose_diverse_candidate(near_wide, strict=False, min_endpoint_distance=0.75)
                or choose_diverse_candidate(pool, strict=True)
                or choose_diverse_candidate(pool, strict=False, min_endpoint_distance=0.75)
                or choose_diverse_candidate(pool, strict=False, min_endpoint_distance=0.45)
            )
            if best_candidate is None:
                break
            selected.append(best_candidate)
            selected_keys.add(tuple(round(float(x), 4) for x in best_candidate["params"]))

        display = selected[:3]
        rng.shuffle(display)
        labels = ["A", "B", "C"]
        options: list[dict[str, Any]] = []
        for label, cand in zip(labels, display, strict=True):
            params = cand["params"]
            options.append(
                {
                    "id": label,
                    "label": label,
                    "speed": float(params[0]),
                    "angle": float(params[1]),
                    "spin": float(params[2]),
                    "y0": float(params[3]),
                    "kind": "generated",
                    "post_value": float(cand["post_value"]),
                    "decision_value": float(cand["decision_value"]),
                    "intended_final_board": cand["intended_final_board"],
                    "intended_trajectory": cand["intended_trajectory"],
                }
            )
        observed_option["id"] = "D"
        observed_option["label"] = "D"
        options.append(observed_option)
        self.candidate_cache[scenario["id"]] = options
        return options

    def _observed_option(self, scenario: dict[str, Any]) -> dict[str, Any]:
        params = scenario["defaults"].astype(np.float32)
        final_board, prev_slots, new_slot, traj_np = self._simulate_board(scenario, params)
        post_value = self._state_value(final_board, scenario)
        return {
            "id": "D",
            "label": "D",
            "speed": float(params[0]),
            "angle": float(params[1]),
            "spin": float(params[2]),
            "y0": float(params[3]),
            "kind": "observed",
            "post_value": float(post_value),
            "decision_value": float(post_value - scenario["v_prev"]),
            "observed_model_decision_value": float(scenario["athlete_dv"]),
            "intended_final_board": _board_to_client(final_board),
            "intended_trajectory": _sample_trajectory_frames(traj_np, prev_slots, new_slot),
        }

    def scenario_payload(self, index: int | None = None, scenario_id: str | None = None) -> dict[str, Any]:
        if scenario_id is not None:
            scenario = self.scenarios.get(scenario_id)
            if scenario is None:
                raise HTTPException(status_code=404, detail="Unknown scenario.")
            index = self.scenario_rows.index(scenario)
        else:
            index = 0 if index is None else int(index) % len(self.scenario_rows)
            scenario = self.scenario_rows[index]
        options = self._generate_candidates(scenario)
        return {
            "index": int(index),
            "count": len(self.scenario_rows),
            "id": scenario["id"],
            "name": f"{scenario['athlete_label']} | End {scenario['end_id']} Shot {scenario['shot_id']}",
            "description": f"Choose from three generated near-optimal throws plus the real observed throw. Task {scenario['task']}, handle {scenario['handle']}.",
            "athlete_name": scenario["athlete_name"],
            "athlete_label": scenario["athlete_label"],
            "team_name": scenario["team_name"],
            "throwing_team": "A" if int(round(float(scenario["stone_block"]))) == 0 else "B",
            "thrower_color": "black",
            "thrower_slot": int(_next_slot_for_block(scenario["pre_board_m"], scenario["stone_block"])),
            "pre_value": float(scenario["v_prev"]),
            "observed_decision_value": float(scenario["athlete_dv"]),
            "pre_board": _board_to_client(scenario["pre_board_m"]),
            "observed_post_board": _board_to_client(scenario["post_board_m"]),
            "options": options,
        }

    def select(self, req: SelectRequest) -> dict[str, Any]:
        scenario = self.scenarios.get(req.scenario_id)
        if scenario is None:
            raise HTTPException(status_code=404, detail="Unknown scenario.")
        options = self._generate_candidates(scenario)
        option = next((o for o in options if o["id"] == req.option_id), None)
        if option is None:
            raise HTTPException(status_code=404, detail="Unknown option.")

        intended = np.array([option["speed"], option["angle"], option["spin"], option["y0"]], dtype=np.float32)
        rng = np.random.default_rng(req.seed if req.seed is not None else random.randrange(1 << 30))
        noisy = _sample_noisy_params(intended, self.noise, rng)
        final_board, prev_slots, new_slot, traj_np = self._simulate_board(scenario, noisy)
        post_value = self._state_value(final_board, scenario)
        executed_decision = float(post_value - scenario["v_prev"])
        intended_decision = float(option["decision_value"])
        sorted_options = sorted(options, key=lambda o: o["decision_value"], reverse=True)
        return {
            "scenario_id": scenario["id"],
            "selected_option_id": option["id"],
            "selected_rank": 1 + [o["id"] for o in sorted_options].index(option["id"]),
            "best_option_id": sorted_options[0]["id"],
            "pre_value": float(scenario["v_prev"]),
            "intended_post_value": float(option["post_value"]),
            "executed_post_value": float(post_value),
            "decision_value": intended_decision,
            "executed_decision_value": executed_decision,
            "execution_value": float(executed_decision - intended_decision),
            "sampled_params": {
                "speed": float(noisy[0]),
                "angle": float(noisy[1]),
                "spin": float(noisy[2]),
                "y0": float(noisy[3]),
            },
            "final_board": _board_to_client(final_board),
            "trajectory": _sample_trajectory_frames(traj_np, prev_slots, new_slot),
            "options": options,
        }


engine = QuizEngine()
app = FastAPI(title="Curling Throw Quiz")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "model_arch": "set_transformer_gaussian",
        "device": DEFAULT_DEVICE,
        "scenario_count": len(engine.scenario_rows),
        "candidate_pool_size": POOL_SIZE,
        "holdout_models": {str(k): str(v) for k, v in sorted(engine.model_paths.items())},
    }


@app.get("/api/scenario")
def scenario(index: int = 0) -> dict[str, Any]:
    return engine.scenario_payload(index=index)


@app.get("/api/random")
def random_scenario() -> dict[str, Any]:
    return engine.scenario_payload(index=random.randrange(len(engine.scenario_rows)))


@app.post("/api/select")
def select(req: SelectRequest) -> dict[str, Any]:
    return engine.select(req)


@app.get("/")
def root() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")
