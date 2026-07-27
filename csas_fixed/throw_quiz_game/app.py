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
os.environ.setdefault("GNN_EDGE_SCALAR_MODE", "button_visible_plus_curl_arc_reach_with_outgoing")
os.environ.setdefault("GNN_NODE_FEATURE_MODE", "none")
os.environ.setdefault("GNN_RELEASE_NODE_MODE", "three_plus_takeout_boundary")
os.environ.setdefault("GNN_EDGE_PRUNE_MODE", "none")

import jax
import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parent
MORE_MCTS_DIR = ROOT_DIR.parent / "csas_fixed_moreMCTS"
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "inverse"))
sys.path.insert(0, str(ROOT_DIR / "valueModel"))
sys.path.insert(0, str(MORE_MCTS_DIR))

from inverse.curling_sim_jax import CurlingParams, simulate_from_params  # type: ignore
from kr_uct_search import _sample_actions, kr_smooth_scores, load_policy  # type: ignore
from score_shots_mc_seq import (  # type: ignore
    POS_MAX,
    SimFnCache,
    evaluate_state_distribution,
    evaluate_state_distribution_batch,
    load_value_model,
    normalize_raw_matrix,
    positions_m_to_raw_matrix,
)
from sim_presets import contact_mild_params  # type: ignore

STATIC_DIR = THIS_DIR / "static"
NOISE_PATH = ROOT_DIR / "noise_versions" / "v1_bowling.json"
DEFAULT_DEVICE = os.environ.get("CURLING_GAME_DEVICE", "cpu")
MAX_SCENARIOS = int(os.environ.get("THROW_QUIZ_MAX_SCENARIOS", "120"))
POOL_SIZE = int(os.environ.get("THROW_QUIZ_POOL_SIZE", "256"))
NOISE_SAMPLES = int(os.environ.get("THROW_QUIZ_NOISE_SAMPLES", "16"))
POLICY_FRACTION = float(os.environ.get("THROW_QUIZ_POLICY_FRACTION", "0.375"))
STRUCTURED_FRACTION = float(os.environ.get("THROW_QUIZ_STRUCTURED_FRACTION", "0.375"))
LOCAL_FRACTION = float(os.environ.get("THROW_QUIZ_LOCAL_FRACTION", "0.125"))
POLICY_TEMPERATURE = float(os.environ.get("THROW_QUIZ_POLICY_TEMPERATURE", "0.75"))
POLICY_STD_SCALE = float(os.environ.get("THROW_QUIZ_POLICY_STD_SCALE", "1.0"))
KR_BANDWIDTH = float(os.environ.get("THROW_QUIZ_KR_BANDWIDTH", "0.75"))
KR_UCT_C = float(os.environ.get("THROW_QUIZ_KR_UCT_C", "0.05"))
SEMANTIC_VALUE_WINDOW = float(os.environ.get("THROW_QUIZ_SEMANTIC_VALUE_WINDOW", "3.50"))
MIN_CLEAR = 2 * 0.145
SEPARATE_PASSES = 6
SHOT_STAGE_VALUES = [1 / 9, 2 / 9, 3 / 9, 4 / 9, 5 / 9, 6 / 9, 7 / 9, 8 / 9, 1.0]
GAUSSIAN_QUARTILE_Z = 0.6744897501960817
WEBAPP_GRAPHTF_CKPT = Path(
    os.environ.get(
        "THROW_QUIZ_VALUE_CKPT_HOLDOUT0",
        str(
            ROOT_DIR
            / "holdouts"
            / "0"
            / "model_graphtf_gaussian_curl_arc_reach_outgoing_plus_takeout_vertices"
            / "model.pt"
        ),
    )
)
POLICY_ROOT = Path(
    os.environ.get(
        "THROW_QUIZ_POLICY_ROOT",
        str(MORE_MCTS_DIR / "checkpoints" / "policy_mcts_graph_prior_frozen_value_single_gpu"),
    )
)
ACTION_MIN = np.array([0.55, -0.25, -20.0, -0.23], dtype=np.float32)
ACTION_MAX = np.array([2.35, 0.25, 20.0, 0.23], dtype=np.float32)
ACTION_SCALE = np.array([0.52, 0.22, 1.05, 0.14], dtype=np.float32)
MIN_DISPLAY_SPIN = 0.55
ALLOWED_SHOT_PURPOSES = {
    "draw",
    "front",
    "guard",
    "raise",
    "wick",
    "freeze",
    "take-out",
    "hit and roll",
    "clearing",
    "double take-out",
    "promotion take-out",
    "through",
}


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


def _distribution_summary(mean: float, std: float) -> dict[str, float]:
    std = max(0.0, float(std))
    quartile_offset = GAUSSIAN_QUARTILE_Z * std
    mean = float(mean)
    return {
        "mean": mean,
        "std": std,
        "p25": mean - quartile_offset,
        "p75": mean + quartile_offset,
        "minus": quartile_offset,
        "plus": quartile_offset,
    }


def _difference_distribution(
    left: dict[str, float],
    right: dict[str, float],
) -> dict[str, float]:
    """Approximate a difference assuming independent Gaussian predictions."""
    mean = float(left["mean"] - right["mean"])
    std = math.sqrt(float(left["std"]) ** 2 + float(right["std"]) ** 2)
    return _distribution_summary(mean, std)


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


def _sample_noisy_action_batch(
    actions: np.ndarray,
    samples_per_action: int,
    cfg: NoiseConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float32)
    repeated = np.repeat(actions[:, None, :], samples_per_action, axis=1)
    repeated[:, :, 0] += rng.standard_t(cfg.nu, size=repeated.shape[:2]).astype(np.float32) * cfg.speed_scale
    speed_for_scale = np.clip(actions[:, 0], cfg.angle_speed_min, cfg.angle_speed_max)
    alpha = (speed_for_scale - cfg.angle_speed_min) / (cfg.angle_speed_max - cfg.angle_speed_min)
    angle_scales = cfg.angle_scale_max + alpha * (cfg.angle_scale_min - cfg.angle_scale_max)
    repeated[:, :, 1] += (
        rng.standard_t(cfg.nu, size=repeated.shape[:2]).astype(np.float32) * angle_scales[:, None]
    )
    repeated[:, :, 2] += rng.normal(0.0, cfg.spin_std, size=repeated.shape[:2]).astype(np.float32)
    repeated[:, :, 3] += rng.normal(0.0, cfg.y0_std, size=repeated.shape[:2]).astype(np.float32)
    return np.clip(repeated, ACTION_MIN, ACTION_MAX).astype(np.float32)


def _unique_actions(actions: np.ndarray) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float32)
    if not len(actions):
        return actions.reshape(0, 4)
    _, idx = np.unique(np.round(actions, decimals=5), axis=0, return_index=True)
    return actions[np.sort(idx)]


def _global_actions(count: int, rng: np.random.Generator) -> np.ndarray:
    if count <= 0:
        return np.zeros((0, 4), dtype=np.float32)
    actions = rng.uniform(ACTION_MIN, ACTION_MAX, size=(count, 4)).astype(np.float32)
    weak_spin = np.abs(actions[:, 2]) < MIN_DISPLAY_SPIN
    if np.any(weak_spin):
        actions[weak_spin, 2] = rng.choice(
            np.array([-1.25, 1.25], dtype=np.float32),
            size=int(np.count_nonzero(weak_spin)),
        )
    return actions


def _structured_actions(state_norm: np.ndarray, count: int) -> np.ndarray:
    """Broad deterministic draws, hits, ticks, and curl directions."""
    if count <= 0:
        return np.zeros((0, 4), dtype=np.float32)
    raw = np.asarray(state_norm, dtype=np.float32).reshape(12, 2) * POS_MAX
    live = ((raw[:, 0] > 0) | (raw[:, 1] > 0)) & (raw[:, 0] < POS_MAX) & (raw[:, 1] < POS_MAX)
    lateral_m = (raw[live, 0] - 750.0) * 0.003048
    lateral_targets = [0.0]
    for lateral in lateral_m:
        lateral_targets.extend(
            float(np.clip(lateral + offset, ACTION_MIN[3], ACTION_MAX[3]))
            for offset in (-0.16, -0.08, 0.0, 0.08, 0.16)
        )
    lateral_targets.extend(np.linspace(ACTION_MIN[3], ACTION_MAX[3], 13).tolist())
    lateral_targets = np.unique(np.round(np.asarray(lateral_targets, dtype=np.float32), 4))
    speeds = np.asarray([0.70, 0.85, 1.00, 1.15, 1.30, 1.50, 1.75, 2.05, 2.30], dtype=np.float32)
    angles = np.linspace(ACTION_MIN[1], ACTION_MAX[1], 9, dtype=np.float32)
    spins = np.asarray([-1.5, -1.05, 1.05, 1.5], dtype=np.float32)
    rows = np.asarray(
        [[speed, angle, spin, lateral] for speed in speeds for lateral in lateral_targets for angle in angles for spin in spins],
        dtype=np.float32,
    )
    if len(rows) <= count:
        return rows
    return rows[np.linspace(0, len(rows) - 1, count, dtype=int)]


def _local_actions(
    seeds: np.ndarray,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if count <= 0 or not len(seeds):
        return np.zeros((0, 4), dtype=np.float32)
    seed_idx = rng.integers(0, len(seeds), size=count)
    std = np.array([0.12, 0.025, 0.35, 0.035], dtype=np.float32)
    actions = seeds[seed_idx] + rng.normal(size=(count, 4)).astype(np.float32) * std
    actions = np.clip(actions, ACTION_MIN, ACTION_MAX).astype(np.float32)
    weak_spin = np.abs(actions[:, 2]) < MIN_DISPLAY_SPIN
    if np.any(weak_spin):
        signs = np.sign(actions[weak_spin, 2])
        random_signs = rng.choice(
            np.array([-1.0, 1.0], dtype=np.float32),
            size=int(np.count_nonzero(weak_spin)),
        )
        signs = np.where(signs == 0.0, random_signs, signs)
        actions[weak_spin, 2] = signs * MIN_DISPLAY_SPIN
    return actions


def _ensure_decision_handles(actions: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Avoid presenting neutral-curl decisions by assigning a real handle."""
    actions = np.asarray(actions, dtype=np.float32).copy()
    if actions.size == 0:
        return actions.reshape(0, 4)
    weak_spin = np.abs(actions[:, 2]) < MIN_DISPLAY_SPIN
    if np.any(weak_spin):
        signal = actions[weak_spin, 2] + 2.5 * actions[weak_spin, 1] + 0.8 * actions[weak_spin, 3]
        signs = np.sign(signal)
        random_signs = rng.choice(
            np.array([-1.0, 1.0], dtype=np.float32),
            size=int(np.count_nonzero(weak_spin)),
        )
        signs = np.where(signs == 0.0, random_signs, signs)
        actions[weak_spin, 2] = signs * MIN_DISPLAY_SPIN
    return np.clip(actions, ACTION_MIN, ACTION_MAX).astype(np.float32)


def _mixture_distribution(means: np.ndarray, stds: np.ndarray) -> dict[str, float]:
    """Moment-match predictive Gaussians across execution-noise outcomes."""
    means = np.asarray(means, dtype=np.float64)
    stds = np.asarray(stds, dtype=np.float64)
    mean = float(np.mean(means))
    variance = float(np.mean(stds**2 + means**2) - mean**2)
    return _distribution_summary(mean, math.sqrt(max(0.0, variance)))


@lru_cache(maxsize=10)
def _load_search_policy(horizon: int):
    horizon = int(np.clip(horizon, 1, 10))
    path = POLICY_ROOT / f"h{horizon:02d}" / "policy" / "best.pt"
    if not path.exists():
        path = POLICY_ROOT / f"h{horizon:02d}" / "policy" / "model.pt"
    if not path.exists():
        raise FileNotFoundError(f"No search policy checkpoint for horizon {horizon}: {path}")
    device = torch.device(DEFAULT_DEVICE)
    policy, action_mean, action_std = load_policy(path, device)
    return policy, action_mean, action_std, path


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
    if int(competition_id) == 0 and WEBAPP_GRAPHTF_CKPT.exists():
        ckpt = WEBAPP_GRAPHTF_CKPT
    else:
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


def _slotted_boards_from_compact_batch(final_compact_batch: np.ndarray, prev_slots: list[int], new_slot: int) -> np.ndarray:
    boards = np.full((final_compact_batch.shape[0], 12, 2), np.nan, dtype=np.float32)
    n_prev = len(prev_slots)
    for idx, slot in enumerate(prev_slots):
        if idx < final_compact_batch.shape[1]:
            xy = final_compact_batch[:, idx, :]
            finite = np.isfinite(xy[:, 0]) & np.isfinite(xy[:, 1])
            boards[finite, slot, :] = xy[finite]
    if final_compact_batch.shape[1] > n_prev:
        xy = final_compact_batch[:, -1, :]
        finite = np.isfinite(xy[:, 0]) & np.isfinite(xy[:, 1])
        boards[finite, new_slot, :] = xy[finite]
    return boards


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
    endpoint = candidate.get("endpoint")
    if endpoint is not None:
        return float(endpoint[0]), float(endpoint[1])
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
    endpoint_distance = _endpoint_distance(a, b)
    board_a = np.asarray(a.get("final_board"), dtype=np.float32)
    board_b = np.asarray(b.get("final_board"), dtype=np.float32)
    live = np.isfinite(board_a[:, 0]) | np.isfinite(board_b[:, 0])
    board_distance = 0.0
    if np.any(live):
        filled_a = np.where(np.isfinite(board_a), board_a, 10.0)
        filled_b = np.where(np.isfinite(board_b), board_b, 10.0)
        board_distance = float(np.sqrt(np.mean(np.sum((filled_a[live] - filled_b[live]) ** 2, axis=1))))
    action_distance = float(np.linalg.norm((a["params"] - b["params"]) / ACTION_SCALE))
    return endpoint_distance >= 0.80 or board_distance >= 0.55 or action_distance >= 3.0 or (
        endpoint_distance >= 0.45 and action_distance >= 1.75
    )


def _candidate_diversity(a: dict[str, Any], b: dict[str, Any]) -> float:
    endpoint_distance = _endpoint_distance(a, b)
    board_a = np.asarray(a["final_board"], dtype=np.float32)
    board_b = np.asarray(b["final_board"], dtype=np.float32)
    live = np.isfinite(board_a[:, 0]) | np.isfinite(board_b[:, 0])
    filled_a = np.where(np.isfinite(board_a), board_a, 10.0)
    filled_b = np.where(np.isfinite(board_b), board_b, 10.0)
    board_distance = (
        float(np.sqrt(np.mean(np.sum((filled_a[live] - filled_b[live]) ** 2, axis=1))))
        if np.any(live)
        else 0.0
    )
    action_distance = float(np.linalg.norm((a["params"] - b["params"]) / ACTION_SCALE))
    return 2.0 * min(endpoint_distance, 2.5) + 1.5 * min(board_distance, 2.5) + 0.25 * min(action_distance, 5.0)


def _live_mask_m(board: np.ndarray) -> np.ndarray:
    return np.isfinite(board[:, 0]) & np.isfinite(board[:, 1])


def _classify_shot_semantics(
    pre_board: np.ndarray,
    final_board: np.ndarray,
    params: np.ndarray,
    new_slot: int,
    stone_block: float,
) -> tuple[str, str]:
    """Classify shot purpose and curl shape from the simulated board effect."""
    pre_live = _live_mask_m(pre_board)
    post_live = _live_mask_m(final_board)
    own = np.zeros((12,), dtype=bool)
    own[:6] = stone_block < 0.5
    own[6:] = stone_block >= 0.5
    opponent = ~own

    removed_opponent_count = int(np.sum(pre_live & opponent & ~post_live))
    removed_opponent = removed_opponent_count > 0
    removed_any = bool(np.any(pre_live & ~post_live))
    common = pre_live & post_live
    displacement = np.zeros((12,), dtype=np.float32)
    displacement[common] = np.linalg.norm(final_board[common] - pre_board[common], axis=1)
    max_displacement = float(np.max(displacement)) if np.any(common) else 0.0
    own_moved = bool(np.any(common & own & (displacement >= 0.16)))
    opponent_moved = bool(np.any(common & opponent & (displacement >= 0.16)))

    endpoint = final_board[new_slot]
    delivered_live = bool(post_live[new_slot])
    endpoint_radius = float(np.linalg.norm(endpoint)) if delivered_live else float("inf")
    nearest_stone = float("inf")
    if delivered_live:
        other_live = post_live.copy()
        other_live[new_slot] = False
        if np.any(other_live):
            nearest_stone = float(np.min(np.linalg.norm(final_board[other_live] - endpoint[None], axis=1)))
    draw_weight = float(params[0]) <= 1.40
    if removed_opponent_count >= 2:
        purpose = "double take-out"
    elif removed_opponent and own_moved:
        purpose = "promotion take-out"
    elif removed_opponent:
        purpose = "take-out"
    elif removed_any:
        purpose = "clearing"
    elif draw_weight and delivered_live and nearest_stone <= 0.34:
        purpose = "freeze"
    elif draw_weight and delivered_live and endpoint_radius <= 1.829:
        purpose = "draw"
    elif draw_weight and delivered_live and float(endpoint[0]) < -1.0:
        purpose = "front" if abs(float(endpoint[1])) <= 0.55 else "guard"
    elif max_displacement >= 0.45:
        purpose = "hit and roll"
    elif max_displacement >= 0.10:
        purpose = "raise" if own_moved and not opponent_moved else "wick"
    elif delivered_live and nearest_stone <= 0.34:
        purpose = "freeze"
    elif delivered_live and endpoint_radius <= 1.829:
        purpose = "draw"
    elif delivered_live and float(endpoint[0]) < -1.0:
        purpose = "front" if abs(float(endpoint[1])) <= 0.55 else "guard"
    elif delivered_live:
        purpose = "guard"
    else:
        purpose = "through"

    # Compare the actual endpoint with the no-curl extension of release angle.
    # A residual toward the center line is an in-curl; away is an out-curl.
    # Do not expose a "neutral" curl option: real choices should have a handle.
    curl = "in-curl" if float(params[2]) >= 0.0 else "out-curl"
    if delivered_live:
        travel = max(0.0, float(endpoint[0]) + 6.40)
        no_curl_lateral = float(params[3]) + math.tan(float(params[1])) * travel
        actual_lateral = float(endpoint[1])
        if abs(actual_lateral) + 0.06 < abs(no_curl_lateral):
            curl = "in-curl"
        elif abs(actual_lateral) > abs(no_curl_lateral) + 0.06:
            curl = "out-curl"
    if purpose not in ALLOWED_SHOT_PURPOSES:
        purpose = "through"
    return purpose, curl


def _is_unproductive_through(
    pre_board: np.ndarray,
    final_board: np.ndarray,
    new_slot: int,
) -> bool:
    """True when the delivered stone exits play without affecting another stone."""
    pre_live = _live_mask_m(pre_board)
    post_live = _live_mask_m(final_board)
    if post_live[new_slot]:
        return False
    if np.any(pre_live & ~post_live):
        return False
    common = pre_live & post_live
    if not np.any(common):
        return True
    max_displacement = float(
        np.max(np.linalg.norm(final_board[common] - pre_board[common], axis=1))
    )
    return max_displacement < 0.08


class QuizEngine:
    def __init__(self) -> None:
        self.noise = _load_noise_config()
        self.model_paths = _settf_gaussian_checkpoint_map()
        if WEBAPP_GRAPHTF_CKPT.exists():
            self.model_paths[0] = WEBAPP_GRAPHTF_CKPT
        self.sim_params = contact_mild_params(CurlingParams)
        self.sim_cache = SimFnCache(self.sim_params)
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
        end_cols = ["CompetitionID", "SessionID", "GameID", "EndID"]
        stones["ShotIndex"] = stones.groupby(end_cols, dropna=False).cumcount()
        stones["ShotsInEnd"] = stones.groupby(end_cols, dropna=False)["ShotID"].transform("size")
        prev_stones = stones.groupby(end_cols, dropna=False)[_stone_cols()].shift(1)
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
            stone_block = float(row.stone_block)
            shot_norm_prev = float(row.shot_norm_prev)
            shot_norm_next = float(row.shot_norm_next)
            team_order = float(row.team_order)
            prev_slots = _occupied_slots(pre_board)
            prev_compact = pre_board[prev_slots].astype(np.float32)
            new_slot = _next_slot_for_block(pre_board, stone_block)
            shot_index = int(row.ShotIndex)
            shots_in_end = int(row.ShotsInEnd)
            horizon = max(1, shots_in_end - shot_index)
            scenario_id = f"{int(row.CompetitionID)}-{int(row.SessionID)}-{int(row.GameID)}-{int(row.EndID)}-{int(row.ShotID)}"
            rows.append(
                {
                    "id": scenario_id,
                    "competition_id": int(row.CompetitionID),
                    "session_id": int(row.SessionID),
                    "game_id": int(row.GameID),
                    "end_id": int(row.EndID),
                    "shot_id": int(row.ShotID),
                    "shot_index": shot_index,
                    "shots_in_end": shots_in_end,
                    "horizon": horizon,
                    "athlete_name": str(row.player_name),
                    "athlete_label": str(row.player_label),
                    "team_name": str(row.team_name),
                    "task": int(row.Task),
                    "handle": int(row.Handle),
                    "team_order": team_order,
                    "stone_block": stone_block,
                    "shot_norm_prev": shot_norm_prev,
                    "shot_norm_next": shot_norm_next,
                    "v_prev": float(row.v_prev),
                    "v_next_observed": float(row.v_next),
                    "athlete_dv": float(row.dv_obs),
                    "model_path": str(self.model_paths[int(row.CompetitionID)]),
                    "defaults": np.array([float(row.est_speed), float(row.est_angle), float(row.est_spin), float(row.est_y0)], dtype=np.float32),
                    "pre_board_m": pre_board,
                    "post_board_m": post_board,
                    "prev_slots": prev_slots,
                    "prev_compact": prev_compact,
                    "new_slot": new_slot,
                    "raw_defaults": np.full((12, 2), POS_MAX, dtype=np.float32),
                    "pre_c_vec": np.array([shot_norm_prev, team_order, stone_block], dtype=np.float32),
                    "c_vec": np.array([shot_norm_next, 1.0 - team_order, 1.0 - stone_block], dtype=np.float32),
                }
            )
        if not rows:
            raise RuntimeError("No usable throw quiz scenarios were loaded.")
        return rows

    def _simulate_board(self, scenario: dict[str, Any], params: np.ndarray) -> tuple[np.ndarray, list[int], int, np.ndarray]:
        prev_slots = scenario["prev_slots"]
        prev_compact = scenario["prev_compact"]
        new_slot = int(scenario["new_slot"])
        traj = simulate_from_params(
            self.sim_params,
            jax.device_put(prev_compact),
            jax.device_put(params.astype(np.float32)),
            dynamic=True,
        )
        traj_np = np.asarray(jax.device_get(traj), dtype=np.float32)
        final_board = _slotted_board_from_compact(traj_np[-1], prev_slots, new_slot)
        return final_board, prev_slots, new_slot, traj_np

    def _simulate_final_boards_batch(self, scenario: dict[str, Any], param_batch: np.ndarray) -> np.ndarray:
        prev_slots = scenario["prev_slots"]
        prev_compact = scenario["prev_compact"]
        new_slot = int(scenario["new_slot"])
        sim_fn = self.sim_cache.get(len(prev_slots), int(param_batch.shape[0]))
        finals = np.asarray(
            sim_fn(
                jax.device_put(prev_compact),
                jax.device_put(param_batch.astype(np.float32, copy=False)),
            ),
            dtype=np.float32,
        )
        return _slotted_boards_from_compact_batch(finals, prev_slots, new_slot)

    def _outcome_targeted_structured_actions(
        self,
        scenario: dict[str, Any],
        count: int,
    ) -> np.ndarray:
        """Simulate a coarse grid and retain balanced shot-purpose/curl outcomes."""
        if count <= 0:
            return np.zeros((0, 4), dtype=np.float32)
        speeds = np.asarray([0.70, 0.90, 1.10, 1.30, 1.55, 1.85, 2.20], dtype=np.float32)
        angles = np.linspace(ACTION_MIN[1], ACTION_MAX[1], 9, dtype=np.float32)
        spins = np.asarray([-1.5, -1.05, 1.05, 1.5], dtype=np.float32)
        y0s = np.linspace(ACTION_MIN[3], ACTION_MAX[3], 7, dtype=np.float32)
        grid = np.asarray(
            np.meshgrid(speeds, angles, spins, y0s, indexing="ij"),
            dtype=np.float32,
        ).reshape(4, -1).T
        boards = self._simulate_final_boards_batch(scenario, grid)
        boards, illegal = self._apply_early_legality(boards, scenario)

        groups: dict[tuple[str, str], list[int]] = {}
        for idx, (action, board) in enumerate(zip(grid, boards, strict=True)):
            if illegal[idx] or _is_unproductive_through(
                scenario["pre_board_m"],
                board,
                int(scenario["new_slot"]),
            ):
                continue
            semantics = _classify_shot_semantics(
                scenario["pre_board_m"],
                board,
                action,
                int(scenario["new_slot"]),
                float(scenario["stone_block"]),
            )
            groups.setdefault(semantics, []).append(idx)
        if not groups:
            return np.zeros((0, 4), dtype=np.float32)

        purpose_order = {
            "draw": 0,
            "front": 1,
            "guard": 2,
            "raise": 3,
            "wick": 4,
            "freeze": 5,
            "take-out": 6,
            "hit and roll": 7,
            "clearing": 8,
            "double take-out": 9,
            "promotion take-out": 10,
            "through": 11,
        }
        curl_order = {"in-curl": 0, "out-curl": 1}
        keys = sorted(
            groups,
            key=lambda key: (
                purpose_order.get(key[0], 99),
                curl_order.get(key[1], 99),
            ),
        )
        group_rows: list[list[np.ndarray]] = []
        per_group = max(1, int(math.ceil(count / len(keys))))
        for key in keys:
            indices = np.asarray(groups[key], dtype=np.int64)
            endpoints = boards[indices, int(scenario["new_slot"])]
            endpoint_angle = np.arctan2(endpoints[:, 1], endpoints[:, 0])
            endpoint_radius = np.linalg.norm(endpoints, axis=1)
            order = np.lexsort((grid[indices, 0], endpoint_radius, endpoint_angle))
            ordered = indices[order]
            take = min(per_group, len(ordered))
            sampled = ordered[np.linspace(0, len(ordered) - 1, take, dtype=int)]
            group_rows.append([grid[int(idx)] for idx in sampled])

        selected: list[np.ndarray] = []
        depth = 0
        while len(selected) < count:
            added = False
            for rows in group_rows:
                if depth < len(rows):
                    selected.append(rows[depth])
                    added = True
                    if len(selected) >= count:
                        break
            if not added:
                break
            depth += 1
        return _unique_actions(np.asarray(selected, dtype=np.float32))[:count]

    def _state_distribution(
        self,
        board: np.ndarray,
        scenario: dict[str, Any],
        c_vec: np.ndarray,
        shot_norm: float,
        perspective_block: float,
        terminal: bool,
    ) -> dict[str, float]:
        model_fn, _ = _load_model_for_competition(int(scenario["competition_id"]))
        mean, std = evaluate_state_distribution(
            model_fn,
            board,
            scenario["raw_defaults"],
            c_vec,
            perspective_block,
            float("nan"),
            float("nan"),
            shot_norm,
            use_rule_based_terminal=terminal,
        )
        return _distribution_summary(mean, std)

    def _state_distribution_batch(
        self,
        boards: np.ndarray,
        scenario: dict[str, Any],
        c_vec: np.ndarray,
        shot_norm: float,
        perspective_block: float,
        terminal: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        model_fn, _ = _load_model_for_competition(int(scenario["competition_id"]))
        c_batch = np.broadcast_to(
            c_vec.reshape(1, -1),
            (boards.shape[0], c_vec.shape[0]),
        ).astype(np.float32, copy=False)
        return evaluate_state_distribution_batch(
            model_fn,
            boards,
            scenario["raw_defaults"],
            c_batch,
            perspective_block,
            float("nan"),
            float("nan"),
            shot_norm,
            use_rule_based_terminal=terminal,
        )

    def _pre_distribution(self, scenario: dict[str, Any]) -> dict[str, float]:
        cached = scenario.get("pre_distribution")
        if cached is None:
            cached = self._state_distribution(
                scenario["pre_board_m"],
                scenario,
                scenario["pre_c_vec"],
                float(scenario["shot_norm_prev"]),
                float(scenario["stone_block"]),
                terminal=False,
            )
            scenario["pre_distribution"] = cached
        return cached

    def _post_q_distribution(self, board: np.ndarray, scenario: dict[str, Any]) -> dict[str, float]:
        terminal = int(scenario["horizon"]) <= 1
        if terminal:
            return self._state_distribution(
                board,
                scenario,
                scenario["c_vec"],
                float(scenario["shot_norm_next"]),
                float(scenario["stone_block"]),
                terminal=True,
            )
        next_distribution = self._state_distribution(
            board,
            scenario,
            scenario["c_vec"],
            float(scenario["shot_norm_next"]),
            1.0 - float(scenario["stone_block"]),
            terminal=False,
        )
        return _distribution_summary(-next_distribution["mean"], next_distribution["std"])

    def _post_q_distribution_batch(
        self,
        boards: np.ndarray,
        scenario: dict[str, Any],
    ) -> tuple[np.ndarray, np.ndarray]:
        terminal = int(scenario["horizon"]) <= 1
        perspective_block = (
            float(scenario["stone_block"])
            if terminal
            else 1.0 - float(scenario["stone_block"])
        )
        means, stds = self._state_distribution_batch(
            boards,
            scenario,
            scenario["c_vec"],
            float(scenario["shot_norm_next"]),
            perspective_block,
            terminal=terminal,
        )
        return (means, stds) if terminal else (-means, stds)

    def _apply_early_legality(
        self,
        boards: np.ndarray,
        scenario: dict[str, Any],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Restore the pre-shot board if throws 1-3 remove an opponent stone."""
        corrected = np.asarray(boards, dtype=np.float32).copy()
        illegal = np.zeros((len(corrected),), dtype=bool)
        if int(scenario["shot_index"]) >= 3:
            return corrected, illegal
        pre_raw = positions_m_to_raw_matrix(
            scenario["pre_board_m"],
            raw_defaults=scenario["raw_defaults"],
        )
        pre_live = ((pre_raw[:, 0] > 0) | (pre_raw[:, 1] > 0)) & (
            (pre_raw[:, 0] < POS_MAX) & (pre_raw[:, 1] < POS_MAX)
        )
        opponent = np.zeros((12,), dtype=bool)
        if float(scenario["stone_block"]) < 0.5:
            opponent[6:] = True
        else:
            opponent[:6] = True
        for idx, board in enumerate(corrected):
            post_raw = positions_m_to_raw_matrix(board, raw_defaults=scenario["raw_defaults"])
            post_live = ((post_raw[:, 0] > 0) | (post_raw[:, 1] > 0)) & (
                (post_raw[:, 0] < POS_MAX) & (post_raw[:, 1] < POS_MAX)
            )
            illegal[idx] = bool(np.any(pre_live & opponent & ~post_live))
        corrected[illegal] = scenario["pre_board_m"]
        return corrected, illegal

    def _materialize_candidate(self, scenario: dict[str, Any], label: str, kind: str, cand: dict[str, Any]) -> dict[str, Any]:
        params = cand["params"]
        final_board, prev_slots, new_slot, traj_np = self._simulate_board(scenario, params)
        return {
            "id": label,
            "label": label,
            "speed": float(params[0]),
            "angle": float(params[1]),
            "spin": float(params[2]),
            "y0": float(params[3]),
            "kind": kind,
            "shot_purpose": cand["shot_purpose"],
            "curl_type": cand["curl_type"],
            "post_value": float(cand["post_value"]),
            "decision_value": float(cand["decision_value"]),
            "post_distribution": cand["post_distribution"],
            "decision_distribution": cand["decision_distribution"],
            "intended_final_board": _board_to_client(final_board),
            "intended_trajectory": _sample_trajectory_frames(traj_np, prev_slots, new_slot),
        }

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
        try:
            state_norm = normalize_raw_matrix(
                positions_m_to_raw_matrix(
                    scenario["pre_board_m"],
                    raw_defaults=scenario["raw_defaults"],
                )
            )
            cond = scenario["pre_c_vec"].astype(np.float32)
            policy_count = max(32, int(round(POOL_SIZE * POLICY_FRACTION)))
            structured_count = max(32, int(round(POOL_SIZE * STRUCTURED_FRACTION)))
            local_count = max(16, int(round(POOL_SIZE * LOCAL_FRACTION)))
            global_count = max(16, POOL_SIZE - policy_count - structured_count - local_count)
            policy, action_mean, action_std, _ = _load_search_policy(int(scenario["horizon"]))
            policy_actions = _sample_actions(
                policy,
                action_mean,
                action_std,
                state_norm,
                cond,
                policy_count,
                torch.device(DEFAULT_DEVICE),
                POLICY_TEMPERATURE,
                POLICY_STD_SCALE,
                0.0,
            )
            policy_actions = _ensure_decision_handles(policy_actions, rng)
            structured = self._outcome_targeted_structured_actions(
                scenario,
                structured_count,
            )
            if len(structured) < structured_count:
                structured = _unique_actions(
                    np.concatenate(
                        [
                            structured,
                            _structured_actions(
                                state_norm,
                                structured_count - len(structured),
                            ),
                        ],
                        axis=0,
                    )
                )[:structured_count]
            local_seeds = np.concatenate(
                [
                    scenario["defaults"].astype(np.float32)[None],
                    policy_actions[: min(12, len(policy_actions))],
                    structured[: min(12, len(structured))],
                ],
                axis=0,
            )
            local = _local_actions(local_seeds, local_count, rng)
            global_actions = _global_actions(global_count, rng)
            param_batch = _unique_actions(
                _ensure_decision_handles(
                    np.concatenate([policy_actions, structured, local, global_actions], axis=0),
                    rng,
                )
            )
            if len(param_batch) < 4:
                raise RuntimeError(f"Only generated {len(param_batch)} unique actions")

            intended_boards = self._simulate_final_boards_batch(scenario, param_batch)
            intended_boards, intended_illegal = self._apply_early_legality(intended_boards, scenario)
            productive = np.asarray(
                [
                    not _is_unproductive_through(
                        scenario["pre_board_m"],
                        board,
                        int(scenario["new_slot"]),
                    )
                    for board in intended_boards
                ],
                dtype=bool,
            )
            if np.count_nonzero(productive) < 4:
                raise RuntimeError("Fewer than four productive intended actions survived")
            param_batch = param_batch[productive]
            intended_boards = intended_boards[productive]
            intended_illegal = intended_illegal[productive]

            noisy_actions = _sample_noisy_action_batch(
                param_batch,
                NOISE_SAMPLES,
                self.noise,
                rng,
            )
            noisy_boards = self._simulate_final_boards_batch(
                scenario,
                noisy_actions.reshape(-1, 4),
            )
            noisy_boards, illegal_samples = self._apply_early_legality(noisy_boards, scenario)
            q_means, q_stds = self._post_q_distribution_batch(noisy_boards, scenario)
            q_means = q_means.reshape(len(param_batch), NOISE_SAMPLES)
            q_stds = q_stds.reshape(len(param_batch), NOISE_SAMPLES)
            illegal_samples = illegal_samples.reshape(len(param_batch), NOISE_SAMPLES)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to generate throw candidates: {exc}") from exc

        pool: list[dict[str, Any]] = []
        new_slot = int(scenario["new_slot"])
        pre_distribution = self._pre_distribution(scenario)
        mean_q = np.mean(q_means, axis=1)
        action_illegal = np.any(illegal_samples, axis=1)
        search_values = mean_q.copy()
        if np.any(~action_illegal):
            search_values[action_illegal] = -1.0e6
        smooth_scores = kr_smooth_scores(
            param_batch,
            search_values,
            np.asarray(action_mean.detach().cpu(), dtype=np.float32),
            np.asarray(action_std.detach().cpu(), dtype=np.float32),
            KR_BANDWIDTH,
            KR_UCT_C,
        )
        for idx, (params, final_board) in enumerate(zip(param_batch, intended_boards, strict=True)):
            if not np.isfinite(mean_q[idx]):
                continue
            endpoint = final_board[new_slot]
            post_distribution = _mixture_distribution(q_means[idx], q_stds[idx])
            decision_distribution = _difference_distribution(post_distribution, pre_distribution)
            shot_purpose, curl_type = _classify_shot_semantics(
                scenario["pre_board_m"],
                final_board,
                params,
                new_slot,
                float(scenario["stone_block"]),
            )
            pool.append(
                {
                    "params": params,
                    "key": tuple(round(float(x), 4) for x in params),
                    "post_value": float(post_distribution["mean"]),
                    "decision_value": float(decision_distribution["mean"]),
                    "post_distribution": post_distribution,
                    "decision_distribution": decision_distribution,
                    "final_board": final_board,
                    "endpoint": endpoint.copy() if np.isfinite(endpoint[0]) and np.isfinite(endpoint[1]) else None,
                    "search_score": float(smooth_scores[idx]),
                    "illegal": bool(action_illegal[idx] or intended_illegal[idx]),
                    "illegal_rate": float(np.mean(illegal_samples[idx])),
                    "shot_purpose": shot_purpose,
                    "curl_type": curl_type,
                }
            )

        if len(pool) < 4:
            raise HTTPException(status_code=500, detail=f"Only generated {len(pool)} viable throw options.")

        legal_pool = [candidate for candidate in pool if not candidate["illegal"]]
        ranked = sorted(legal_pool or pool, key=lambda candidate: candidate["search_score"], reverse=True)
        best_dv = float(max(candidate["decision_value"] for candidate in ranked))
        scenario["semantic_pool_counts"] = {
            purpose: sum(candidate["shot_purpose"] == purpose for candidate in ranked)
            for purpose in sorted({candidate["shot_purpose"] for candidate in ranked})
        }
        selected: list[dict[str, Any]] = [
            max(ranked, key=lambda candidate: candidate["decision_value"])
        ]
        purpose_priority = [
            "draw",
            "front",
            "guard",
            "freeze",
            "raise",
            "wick",
            "take-out",
            "hit and roll",
            "clearing",
            "double take-out",
            "promotion take-out",
        ]
        while len(selected) < 3:
            selected_keys = {candidate["key"] for candidate in selected}
            purposes = {candidate["shot_purpose"] for candidate in selected}
            next_purpose = next(
                (
                    purpose
                    for purpose in purpose_priority
                    if purpose not in purposes
                    and any(
                        candidate["shot_purpose"] == purpose
                        and candidate["key"] not in selected_keys
                        and best_dv - candidate["decision_value"] <= 6.0
                        for candidate in ranked
                    )
                ),
                None,
            )
            if next_purpose is None:
                break
            purpose_alternatives = [
                candidate
                for candidate in ranked
                if candidate["key"] not in selected_keys
                and candidate["shot_purpose"] == next_purpose
                and best_dv - candidate["decision_value"] <= 6.0
            ]
            if not purpose_alternatives:
                break
            selected.append(
                max(
                    purpose_alternatives,
                    key=lambda candidate: (
                        candidate["decision_value"]
                        + 0.30
                        * float(
                            candidate["curl_type"]
                            not in {chosen["curl_type"] for chosen in selected}
                        )
                        + 0.20
                        * min(
                            _candidate_diversity(candidate, chosen)
                            for chosen in selected
                        )
                    ),
                )
            )
        for value_window in (0.75, 1.75, SEMANTIC_VALUE_WINDOW):
            while len(selected) < 4:
                selected_keys = {candidate["key"] for candidate in selected}
                purposes = {candidate["shot_purpose"] for candidate in selected}
                curls = {candidate["curl_type"] for candidate in selected}
                combinations = {
                    (candidate["shot_purpose"], candidate["curl_type"])
                    for candidate in selected
                }
                candidates = [
                    candidate
                    for candidate in ranked
                    if candidate["key"] not in selected_keys
                    and best_dv - candidate["decision_value"] <= value_window
                ]
                if not candidates:
                    break

                def semantic_score(candidate: dict[str, Any]) -> float:
                    purpose = candidate["shot_purpose"]
                    curl = candidate["curl_type"]
                    combination = (purpose, curl)
                    min_geometry = min(
                        _candidate_diversity(candidate, chosen)
                        for chosen in selected
                    )
                    purpose_bonus = 4.0 if len(selected) >= 3 else 10.0
                    curl_bonus = 10.0 if len(selected) >= 3 else 4.0
                    return (
                        purpose_bonus * float(purpose not in purposes)
                        + curl_bonus * float(curl not in curls)
                        + 3.0 * float(combination not in combinations)
                        + min_geometry
                        - 1.5 * max(0.0, best_dv - candidate["decision_value"])
                    )

                selected.append(max(candidates, key=semantic_score))
            if len(selected) >= 4:
                break
        if len(selected) < 4:
            selected_keys = {candidate["key"] for candidate in selected}
            selected.extend(
                candidate
                for candidate in ranked
                if candidate["key"] not in selected_keys
            )

        display = selected[:4]
        rng.shuffle(display)
        labels = ["A", "B", "C", "D"]
        options: list[dict[str, Any]] = []
        for label, cand in zip(labels, display, strict=True):
            option = self._materialize_candidate(scenario, label, "searched", cand)
            option["illegal_rate"] = cand["illegal_rate"]
            options.append(option)
        self.candidate_cache[scenario["id"]] = options
        return options

    def _observed_option(self, scenario: dict[str, Any]) -> dict[str, Any]:
        params = scenario["defaults"].astype(np.float32)
        final_board, prev_slots, new_slot, traj_np = self._simulate_board(scenario, params)
        post_distribution = self._post_q_distribution(final_board, scenario)
        decision_distribution = _difference_distribution(post_distribution, self._pre_distribution(scenario))
        return {
            "id": "D",
            "label": "D",
            "speed": float(params[0]),
            "angle": float(params[1]),
            "spin": float(params[2]),
            "y0": float(params[3]),
            "kind": "observed",
            "post_value": float(post_distribution["mean"]),
            "decision_value": float(decision_distribution["mean"]),
            "post_distribution": post_distribution,
            "decision_distribution": decision_distribution,
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
        pre_distribution = self._pre_distribution(scenario)
        return {
            "index": int(index),
            "count": len(self.scenario_rows),
            "id": scenario["id"],
            "name": f"{scenario['athlete_label']} | End {scenario['end_id']} Shot {scenario['shot_id']}",
            "description": (
                f"Choose from four diverse policy-guided searches, ranked by expected value under execution noise. "
                f"Task {scenario['task']}, handle {scenario['handle']}, {scenario['horizon']} throws remaining."
            ),
            "athlete_name": scenario["athlete_name"],
            "athlete_label": scenario["athlete_label"],
            "team_name": scenario["team_name"],
            "throwing_team": "A" if int(round(float(scenario["stone_block"]))) == 0 else "B",
            "thrower_color": "black",
            "thrower_slot": int(scenario["new_slot"]),
            "pre_value": float(pre_distribution["mean"]),
            "pre_distribution": pre_distribution,
            "observed_decision_value": float(scenario["athlete_dv"]),
            "search_semantics_available": scenario.get("semantic_pool_counts", {}),
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
        corrected_boards, illegal = self._apply_early_legality(final_board[None], scenario)
        final_board = corrected_boards[0]
        pre_distribution = self._pre_distribution(scenario)
        executed_post_distribution = self._post_q_distribution(final_board, scenario)
        intended_post_distribution = option["post_distribution"]
        intended_decision_distribution = option["decision_distribution"]
        executed_decision_distribution = _difference_distribution(executed_post_distribution, pre_distribution)
        post_value = float(executed_post_distribution["mean"])
        executed_decision = float(executed_decision_distribution["mean"])
        intended_decision = float(intended_decision_distribution["mean"])
        execution_value = executed_decision - intended_decision
        execution_distribution = _difference_distribution(
            executed_post_distribution,
            intended_post_distribution,
        )
        sorted_options = sorted(options, key=lambda o: o["decision_value"], reverse=True)
        return {
            "scenario_id": scenario["id"],
            "selected_option_id": option["id"],
            "selected_rank": 1 + [o["id"] for o in sorted_options].index(option["id"]),
            "best_option_id": sorted_options[0]["id"],
            "pre_value": float(pre_distribution["mean"]),
            "pre_distribution": pre_distribution,
            "intended_post_value": float(option["post_value"]),
            "executed_post_value": float(post_value),
            "decision_value": intended_decision,
            "decision_distribution": intended_decision_distribution,
            "executed_decision_value": executed_decision,
            "executed_decision_distribution": executed_decision_distribution,
            "execution_value": float(execution_value),
            "execution_distribution": execution_distribution,
            "illegal_early_takeout": bool(illegal[0]),
            "intended_post_distribution": intended_post_distribution,
            "executed_post_distribution": executed_post_distribution,
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
        "model_arch": "graph_transformer_gaussian@holdout0,set_transformer_gaussian@others",
        "device": DEFAULT_DEVICE,
        "scenario_count": len(engine.scenario_rows),
        "candidate_pool_size": POOL_SIZE,
        "execution_noise_samples": NOISE_SAMPLES,
        "search_policy_root": str(POLICY_ROOT),
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
