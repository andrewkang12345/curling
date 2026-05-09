#!/usr/bin/env python3
"""
score_shots_mc_by_competition.py

Run score_shots_mc "one competition at a time":
- Loads all inverse chunks + Stones.csv once
- Merges once
- Then iterates over CompetitionID in order:
    * runs smoke (optional) for that competition only
    * runs full scoring for that competition only
    * writes per-competition output CSV (and optionally a combined CSV)

Notes:
- This keeps memory reasonable and makes it easy to resume/parallelize at the competition level.
- If you want to run competitions in parallel, you can launch multiple processes with --only-competition / --competition-ids.

It is based on your existing script, with minimal behavior changes.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
import pathlib
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

THIS_DIR = pathlib.Path(__file__).resolve().parent
sys.path.append(str(THIS_DIR / "inverse"))
sys.path.append(str(THIS_DIR / "valueModel"))
sys.path.append(str(THIS_DIR))

POS_MAX = 4095.0

# Must match the inverse pipeline (make_BC_data.py / grid_rescue.py)
PAD_POS_M = np.array([50.0, 50.0], dtype=np.float32)
STONE_RADIUS_M = 0.145
MIN_CLEAR = 2 * STONE_RADIUS_M + 1e-3
SEPARATE_PASSES = 6

# Terminal-end scoring constants. These intentionally match the synthetic
# terminal-label generation in valueModel/synth.py and synth_terminal.py.
RAW_HOGLINE_Y = 2900.0
RAW_BACKLINE_Y = 200.0
RAW_UNITS_PER_FOOT = (RAW_HOGLINE_Y - RAW_BACKLINE_Y) / 27.0
RAW_R_12 = 6.0 * RAW_UNITS_PER_FOOT
RAW_STONE_R = 0.46 * RAW_UNITS_PER_FOOT
RAW_HOUSE_RADIUS = RAW_R_12 + RAW_STONE_R


def _preload_nvidia_cuda_libs():
    if os.environ.get("CSAS_PRELOAD_NVIDIA_LIBS", "0").lower() not in {"1", "true", "yes"}:
        return
    lib_paths = [
        "/opt/pytorch/lib/python3.12/site-packages/nvidia/nvjitlink/lib/libnvJitLink.so.12",
        "/opt/pytorch/lib/python3.12/site-packages/nvidia/cuda_runtime/lib/libcudart.so.12",
        "/opt/pytorch/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib/libnvrtc.so.12",
        "/opt/pytorch/lib/python3.12/site-packages/nvidia/cublas/lib/libcublas.so.12",
        "/opt/pytorch/lib/python3.12/site-packages/nvidia/cublas/lib/libcublasLt.so.12",
        "/opt/pytorch/lib/python3.12/site-packages/nvidia/cusparse/lib/libcusparse.so.12",
        "/opt/pytorch/lib/python3.12/site-packages/nvidia/cusolver/lib/libcusolver.so.11",
        "/opt/pytorch/lib/python3.12/site-packages/nvidia/cufft/lib/libcufft.so.11",
        "/opt/pytorch/lib/python3.12/site-packages/nvidia/cudnn/lib/libcudnn.so.9",
    ]
    for lib in lib_paths:
        if os.path.exists(lib):
            ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)


try:
    _preload_nvidia_cuda_libs()
except Exception:
    # Let the regular JAX import path surface the concrete failure.
    pass

# ----------------------------
# Optional JAX simulation imports
# ----------------------------
CURLING_IMPORT_ERROR = None
try:
    import jax
    import jax.numpy as jnp
    from curling_sim_jax import CurlingParams, simulate_from_params, simulate_from_params_flex  # type: ignore
    from curling_inverse import MIN_X, MAX_X, MIN_Y, MAX_Y, SolveBounds  # type: ignore
except Exception as e:  # noqa: BLE001
    CURLING_IMPORT_ERROR = e
    jax = None  # type: ignore
    jnp = None  # type: ignore
    CurlingParams = None  # type: ignore
    simulate_from_params = None  # type: ignore
    SolveBounds = None  # type: ignore
    MIN_X = MIN_Y = -1e9
    MAX_X = MAX_Y = 1e9

try:
    import xgboost as xgb
except Exception:
    xgb = None  # noqa: N816

from sim_presets import contact_mild_params

CSV_BUTTON_Y = 800.0
CSV_CENTER_X = 750.0
CSV_TO_M = 0.003048

SHOT_KEY = ["CompetitionID", "SessionID", "GameID", "EndID", "ShotID"]
PARAM_COLS = ["est_speed", "est_angle", "est_spin", "est_y0"]


# ----------------------------
# Coordinate helpers
# ----------------------------
def meters_to_raw_xy(x_m: float, y_m: float) -> Tuple[float, float]:
    raw_x = y_m / CSV_TO_M + CSV_CENTER_X
    raw_y = CSV_BUTTON_Y - x_m / CSV_TO_M
    return float(raw_x), float(raw_y)


def positions_m_to_raw_matrix(pos_m: np.ndarray, raw_defaults: np.ndarray | None = None) -> np.ndarray:
    if raw_defaults is None:
        out = np.full_like(pos_m, POS_MAX, dtype=np.float32)
    else:
        out = np.asarray(raw_defaults, dtype=np.float32).copy()
    for i, (xm, ym) in enumerate(pos_m):
        if np.isnan(xm) or np.isnan(ym):
            continue
        inb = (MIN_X < float(xm) < MAX_X) and (MIN_Y < float(ym) < MAX_Y)
        if not inb:
            out[i, 0] = POS_MAX
            out[i, 1] = POS_MAX
            continue
        rx, ry = meters_to_raw_xy(float(xm), float(ym))
        out[i, 0] = rx
        out[i, 1] = ry
    return out


def positions_m_to_raw_matrix_batch(pos_m: np.ndarray, raw_defaults: np.ndarray | None = None) -> np.ndarray:
    pos_m = np.asarray(pos_m, dtype=np.float32)
    if raw_defaults is None:
        out = np.full_like(pos_m, POS_MAX, dtype=np.float32)
    else:
        defaults = np.asarray(raw_defaults, dtype=np.float32)
        out = np.broadcast_to(defaults, pos_m.shape).copy()

    xm = pos_m[..., 0]
    ym = pos_m[..., 1]
    finite = np.isfinite(xm) & np.isfinite(ym)
    inb = finite & (MIN_X < xm) & (xm < MAX_X) & (MIN_Y < ym) & (ym < MAX_Y)
    off = finite & (~inb)

    raw_x = ym / CSV_TO_M + CSV_CENTER_X
    raw_y = CSV_BUTTON_Y - xm / CSV_TO_M

    out[..., 0] = np.where(inb, raw_x, out[..., 0])
    out[..., 1] = np.where(inb, raw_y, out[..., 1])
    out[..., 0] = np.where(off, POS_MAX, out[..., 0])
    out[..., 1] = np.where(off, POS_MAX, out[..., 1])
    return out


def normalize_raw_matrix(pos_raw: np.ndarray) -> np.ndarray:
    arr = np.where(np.isfinite(pos_raw), pos_raw, POS_MAX).astype(np.float32)
    return (arr / POS_MAX).reshape(-1).astype(np.float32)


def normalize_raw_matrix_batch(pos_raw: np.ndarray) -> np.ndarray:
    arr = np.where(np.isfinite(pos_raw), pos_raw, POS_MAX).astype(np.float32)
    return (arr / POS_MAX).reshape(arr.shape[0], -1).astype(np.float32)


def extract_state_from_row(row: pd.Series, prefix: str) -> Tuple[np.ndarray, List[int]]:
    mat = np.full((12, 2), np.nan, dtype=np.float32)
    for i in range(1, 13):
        x = row.get(f"{prefix}_stone_{i}_x_m", np.nan)
        y = row.get(f"{prefix}_stone_{i}_y_m", np.nan)
        if not (pd.isna(x) or pd.isna(y)):
            mat[i - 1, 0] = float(x)
            mat[i - 1, 1] = float(y)
    keep_mask = ~np.isnan(mat).any(axis=1)
    stone_ids = [idx + 1 for idx, keep in enumerate(keep_mask) if keep]
    return mat, stone_ids


def compact_positions(mat: np.ndarray) -> Tuple[np.ndarray, List[int]]:
    keep_mask = ~np.isnan(mat).any(axis=1)
    compact = mat[keep_mask]
    ids = [i + 1 for i, flag in enumerate(keep_mask) if flag]
    return compact.astype(np.float32), ids


def assign_final_to_slots(final_pos: np.ndarray, prev_ids: List[int], new_id: int) -> np.ndarray:
    out = np.full((12, 2), np.nan, dtype=np.float32)
    for idx, sid in enumerate(prev_ids):
        if idx < final_pos.shape[0]:
            out[sid - 1] = final_pos[idx]
    if final_pos.shape[0] > len(prev_ids):
        out[new_id - 1] = final_pos[-1]
    return out


def assign_finals_to_slots_batch(final_pos: np.ndarray, prev_ids: List[int], new_id: int) -> np.ndarray:
    final_pos = np.asarray(final_pos, dtype=np.float32)
    out = np.full((final_pos.shape[0], 12, 2), np.nan, dtype=np.float32)
    prev_slots = np.asarray(prev_ids, dtype=np.int64) - 1
    if prev_slots.size > 0:
        out[:, prev_slots, :] = final_pos[:, : len(prev_ids), :]
    if final_pos.shape[1] > len(prev_ids):
        out[:, new_id - 1, :] = final_pos[:, -1, :]
    return out


def _separate_overlaps(pts: np.ndarray, min_gap: float = MIN_CLEAR, passes: int = SEPARATE_PASSES) -> np.ndarray:
    """Push apart overlapping stones — must match inverse pipeline exactly."""
    if pts.size == 0:
        return pts
    import math as _math
    p = pts.copy()
    n = p.shape[0]
    for _ in range(passes):
        moved = False
        for i in range(n):
            for j in range(i + 1, n):
                dx = p[j, 0] - p[i, 0]
                dy = p[j, 1] - p[i, 1]
                d = _math.hypot(dx, dy)
                if d < 1e-9:
                    dx, dy, d = 1e-6, 0.0, 1e-6
                if d < min_gap:
                    push = 0.5 * (min_gap - d)
                    nx, ny = dx / d, dy / d
                    p[i, 0] -= push * nx
                    p[i, 1] -= push * ny
                    p[j, 0] += push * nx
                    p[j, 1] += push * ny
                    moved = True
        if not moved:
            break
    return p


def state_to_fixed_slot_arrays(mat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Build 12-slot position array + mask from (12,2) matrix with NaN for absent stones."""
    arr = np.tile(PAD_POS_M, (12, 1)).astype(np.float32)
    mask = np.zeros(12, dtype=bool)
    for i in range(12):
        if np.isfinite(mat[i, 0]) and np.isfinite(mat[i, 1]):
            arr[i] = mat[i]
            mask[i] = True
    return arr, mask


def assign_final_12slot(final_13: np.ndarray, prev_slot_mask: np.ndarray, new_id: int) -> np.ndarray:
    """Extract a 12-slot board from 13-element (12 slots + thrown) simulation output."""
    out = np.full((12, 2), np.nan, dtype=np.float32)
    for i in range(12):
        if prev_slot_mask[i]:
            out[i] = final_13[i]
    out[new_id - 1] = final_13[12]
    return out


def assign_finals_12slot_batch(finals_13: np.ndarray, prev_slot_mask: np.ndarray, new_id: int) -> np.ndarray:
    """Batch version of assign_final_12slot. finals_13 is (B, 13, 2)."""
    B = finals_13.shape[0]
    out = np.full((B, 12, 2), np.nan, dtype=np.float32)
    for i in range(12):
        if prev_slot_mask[i]:
            out[:, i, :] = finals_13[:, i, :]
    out[:, new_id - 1, :] = finals_13[:, 12, :]
    return out


def _mode_or_round_mean(vals: pd.Series) -> float:
    mode_vals = vals.mode(dropna=True)
    if not mode_vals.empty:
        return float(mode_vals.iloc[0])
    mean_val = vals.mean()
    if pd.isna(mean_val):
        return np.nan
    return float(np.round(float(mean_val)))


def add_throw_slot_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds per-row throw-slot hints from observed transitions in inverse rows:
      - obs_throw_slot_id: exact added slot id when exactly one slot is added
      - team_slot_block: per-(end,TeamID) mode block (0 => slots 1..6, 1 => 7..12)
    """
    out = df.copy()

    prev_x_cols = [f"prev_stone_{i}_x_m" for i in range(1, 13)]
    prev_y_cols = [f"prev_stone_{i}_y_m" for i in range(1, 13)]
    next_x_cols = [f"next_stone_{i}_x_m" for i in range(1, 13)]
    next_y_cols = [f"next_stone_{i}_y_m" for i in range(1, 13)]

    req_cols = prev_x_cols + prev_y_cols + next_x_cols + next_y_cols
    if any(c not in out.columns for c in req_cols):
        out["obs_throw_slot_id"] = np.nan
        out["team_slot_block"] = np.nan
        return out

    prev_present = np.isfinite(out[prev_x_cols].to_numpy()) & np.isfinite(out[prev_y_cols].to_numpy())
    next_present = np.isfinite(out[next_x_cols].to_numpy()) & np.isfinite(out[next_y_cols].to_numpy())
    added_mask = next_present & (~prev_present)
    added_count = np.sum(added_mask, axis=1)

    slot_ids = np.arange(1, 13, dtype=np.float32)[None, :]
    obs_throw_slot_id = np.where(added_count == 1, np.sum(added_mask.astype(np.float32) * slot_ids, axis=1), np.nan)
    out["obs_throw_slot_id"] = obs_throw_slot_id
    out["obs_throw_block"] = np.where(np.isfinite(obs_throw_slot_id), (obs_throw_slot_id > 6).astype(np.float32), np.nan)

    group_cols = ["CompetitionID", "SessionID", "GameID", "EndID", "TeamID"]
    if all(c in out.columns for c in group_cols):
        valid = out[np.isfinite(out["obs_throw_block"])].copy()
        if not valid.empty:
            mode_df = (
                valid.groupby(group_cols, dropna=False)["obs_throw_block"]
                .agg(_mode_or_round_mean)
                .reset_index()
                .rename(columns={"obs_throw_block": "team_slot_block"})
            )
            out = pd.merge(out, mode_df, on=group_cols, how="left")
        else:
            out["team_slot_block"] = np.nan
    else:
        out["team_slot_block"] = np.nan

    if "obs_throw_block" in out.columns:
        out = out.drop(columns=["obs_throw_block"])
    return out


def infer_thrower_block(
    prev_ids: List[int],
    next_ids: List[int],
    obs_throw_slot_id: float,
    team_slot_block: float,
) -> int:
    if np.isfinite(obs_throw_slot_id):
        return 0 if int(round(float(obs_throw_slot_id))) <= 6 else 1

    added = [sid for sid in next_ids if sid not in prev_ids]
    if len(added) == 1:
        return 0 if added[0] <= 6 else 1

    if np.isfinite(team_slot_block):
        return int(np.clip(round(float(team_slot_block)), 0, 1))
    return 0


def choose_new_slot_id(
    prev_ids: List[int],
    next_ids: List[int],
    thrower_block: int,
    obs_throw_slot_id: float,
) -> int:
    if np.isfinite(obs_throw_slot_id):
        return int(round(float(obs_throw_slot_id)))

    added = [sid for sid in next_ids if sid not in prev_ids]
    if len(added) == 1:
        return int(added[0])

    missing = [sid for sid in range(1, 13) if sid not in prev_ids]
    if not missing:
        return prev_ids[-1] if prev_ids else 12

    if int(thrower_block) == 0:
        block_missing = [sid for sid in missing if sid <= 6]
    else:
        block_missing = [sid for sid in missing if sid >= 7]
    if block_missing:
        return block_missing[0]
    return missing[0]


def _team_order_blocks(team_order: float, thrower_block: int) -> Tuple[int, int]:
    order = 0 if not np.isfinite(team_order) else int(np.clip(round(float(team_order)), 0, 1))
    block = int(np.clip(int(thrower_block), 0, 1))
    if order == 0:
        return block, 1 - block
    return 1 - block, block


def _shots_taken_by_team_order(state_shot_index: float) -> Tuple[int, int]:
    if not np.isfinite(state_shot_index):
        return 0, 0
    state_idx = int(np.floor(float(state_shot_index)))
    if state_idx < 0:
        return 0, 0
    total_shots = state_idx + 1
    return (total_shots + 1) // 2, total_shots // 2


def make_raw_defaults_for_state(
    state_shot_index: float,
    team_order: float,
    thrower_block: int,
) -> np.ndarray:
    """
    Reconstruct the raw CSV sentinel layout for a state:
    - 0 for stones not yet thrown in the end
    - 4095 for stones that have been thrown but are currently absent (dead/off-sheet)
    """
    team0_block, team1_block = _team_order_blocks(team_order, thrower_block)
    shots0, shots1 = _shots_taken_by_team_order(state_shot_index)

    thrown_by_block = np.ones((2,), dtype=np.int64)  # pre-placed stone in each block
    thrown_by_block[team0_block] = min(6, 1 + shots0)
    thrown_by_block[team1_block] = min(6, 1 + shots1)

    out = np.zeros((12, 2), dtype=np.float32)
    for slot_idx in range(12):
        block = 0 if slot_idx < 6 else 1
        local_rank = slot_idx if slot_idx < 6 else slot_idx - 6
        if local_rank < int(thrown_by_block[block]):
            out[slot_idx, 0] = POS_MAX
            out[slot_idx, 1] = POS_MAX
    return out


# ----------------------------
# Shot normalization + hammer/order
# ----------------------------
def compute_shot_norm_and_order(stones_df: pd.DataFrame) -> pd.DataFrame:
    df = stones_df.copy()
    df = df.sort_values(SHOT_KEY).reset_index(drop=True)

    end_group = ["CompetitionID", "SessionID", "GameID", "EndID"]

    df["ShotIndex"] = df.groupby(end_group).cumcount()
    df["ShotsInEnd"] = df.groupby(end_group)["ShotID"].transform("count")
    df["shot_norm"] = 0.0
    mask = df["ShotsInEnd"] > 1
    df.loc[mask, "shot_norm"] = df.loc[mask, "ShotIndex"] / (df.loc[mask, "ShotsInEnd"] - 1.0)

    first_team = df.groupby(end_group)["TeamID"].transform("first")
    df["team_order"] = (df["TeamID"] != first_team).astype(np.float32)  # first=0, other=1
    return df


def _coerce_context_dim(c_full: np.ndarray, expected_dim: int) -> np.ndarray:
    if expected_dim <= 0:
        return np.zeros((0,), dtype=np.float32)
    if c_full.shape[0] == expected_dim:
        return c_full
    if c_full.shape[0] > expected_dim:
        return c_full[:expected_dim]
    out = np.zeros((expected_dim,), dtype=np.float32)
    out[: c_full.shape[0]] = c_full
    return out


def _is_raw_in_play(x: float, y: float) -> bool:
    return ((x > 0.0) or (y > 0.0)) and (x < POS_MAX) and (y < POS_MAX)


def _compute_points_final_board_raw(board_raw: np.ndarray) -> Tuple[int, int]:
    coords = np.asarray(board_raw, dtype=np.float32).reshape(12, 2)
    button = np.array([CSV_CENTER_X, CSV_BUTTON_Y], dtype=np.float32)

    def _dists(side_coords: np.ndarray) -> List[float]:
        out: List[float] = []
        for x, y in side_coords:
            if not _is_raw_in_play(float(x), float(y)):
                continue
            dist = float(np.linalg.norm(np.array([x, y], dtype=np.float32) - button))
            if dist <= RAW_HOUSE_RADIUS:
                out.append(dist)
        return out

    da = _dists(coords[:6])
    db = _dists(coords[6:])

    if not da and not db:
        return 0, 0

    mina = min(da) if da else float("inf")
    minb = min(db) if db else float("inf")

    if np.isclose(mina, minb):
        return 0, 0

    if mina < minb:
        opp = minb
        pts = sum(1 for d in da if d < opp)
        return int(pts), int(-pts)

    opp = mina
    pts = sum(1 for d in db if d < opp)
    return int(-pts), int(pts)


def _terminal_value_from_raw_board(board_raw: np.ndarray, stone_block: float) -> float:
    points_block0, points_block1 = _compute_points_final_board_raw(board_raw)
    block = 0 if not np.isfinite(stone_block) else int(np.clip(round(float(stone_block)), 0, 1))
    return float(points_block0 if block == 0 else points_block1)


def _is_terminal_state(shot_index: float, shots_in_end: float, shot_norm: float) -> bool:
    if np.isfinite(shots_in_end) and np.isfinite(shot_index):
        return int(round(float(shot_index))) >= int(round(float(shots_in_end))) - 1
    return np.isfinite(shot_norm) and float(shot_norm) >= (1.0 - 1e-6)


def evaluate_state_value(
    model_fn,
    board_m: np.ndarray,
    raw_defaults: np.ndarray,
    c_vec: np.ndarray,
    stone_block: float,
    shot_index: float,
    shots_in_end: float,
    shot_norm: float,
    use_rule_based_terminal: bool,
) -> float:
    raw_board = positions_m_to_raw_matrix(board_m, raw_defaults=raw_defaults)
    if use_rule_based_terminal and _is_terminal_state(shot_index, shots_in_end, shot_norm):
        return _terminal_value_from_raw_board(raw_board, stone_block)
    raw_norm = normalize_raw_matrix(raw_board)
    return float(model_fn(raw_norm, c_vec))


def evaluate_state_value_batch(
    model_fn,
    board_batch_m: np.ndarray,
    raw_defaults: np.ndarray,
    c_batch: np.ndarray,
    stone_block: float,
    shot_index: float,
    shots_in_end: float,
    shot_norm: float,
    use_rule_based_terminal: bool,
) -> np.ndarray:
    raw_batch = positions_m_to_raw_matrix_batch(board_batch_m, raw_defaults=raw_defaults)
    if use_rule_based_terminal and _is_terminal_state(shot_index, shots_in_end, shot_norm):
        vals = [_terminal_value_from_raw_board(board_raw, stone_block) for board_raw in raw_batch]
        return np.asarray(vals, dtype=np.float32)
    raw_norm_batch = normalize_raw_matrix_batch(raw_batch)
    return np.asarray(model_fn(raw_norm_batch, c_batch), dtype=np.float32).reshape(-1)


# ----------------------------
# Value model loader
# ----------------------------
def load_value_model(model_path: pathlib.Path, device: str = "cpu"):
    if model_path.suffix.lower() in (".json", ".xgb"):
        if xgb is None:
            raise ImportError("xgboost is not installed but an XGB model path was provided.")
        booster = xgb.Booster()
        booster.load_model(str(model_path))

        def predict(x_flat: np.ndarray, c_vec: np.ndarray):
            x_arr = np.asarray(x_flat, dtype=np.float32)
            c_arr = np.asarray(c_vec, dtype=np.float32)
            single = x_arr.ndim == 1

            if single:
                x_arr = x_arr.reshape(1, -1)
            if c_arr.ndim == 1:
                c_arr = np.broadcast_to(c_arr.reshape(1, -1), (x_arr.shape[0], c_arr.shape[0]))
            elif c_arr.ndim == 2 and c_arr.shape[0] == 1 and x_arr.shape[0] > 1:
                c_arr = np.broadcast_to(c_arr, (x_arr.shape[0], c_arr.shape[1]))
            elif c_arr.ndim != 2 or c_arr.shape[0] != x_arr.shape[0]:
                raise ValueError(
                    f"Context batch shape {c_arr.shape} does not match feature batch shape {x_arr.shape}."
                )

            feat = np.concatenate([x_arr, c_arr], axis=1).astype(np.float32, copy=False)
            pred = booster.predict(xgb.DMatrix(feat)).astype(np.float32, copy=False)
            return float(pred[0]) if single else pred

        return predict, None

    try:
        import torch
    except Exception as e:  # noqa: BLE001
        raise ImportError("torch is required to load PyTorch value-model checkpoints.") from e
    from model import ValueTransformer  # type: ignore

    ckpt = torch.load(model_path, map_location=device)
    input_dim = ckpt["input_dim"]
    cond_dim = int(ckpt["cond_dim"])
    hidden_dim = ckpt["hidden_dim"]
    num_stones = ckpt.get("num_stones", 12)
    args_dict = ckpt.get("args", {})
    n_layers = args_dict.get("n_layers", 4)
    n_heads = args_dict.get("n_heads", 4)
    dropout = args_dict.get("dropout", 0.1)

    arch = str(ckpt.get("arch", "")).strip().lower()

    def _ensure_ablation_on_path():
        ablation_dir = THIS_DIR / "valueModel" / "ablation"
        if str(ablation_dir) not in sys.path:
            sys.path.append(str(ablation_dir))

    # Prefer explicit arch if provided in checkpoint.
    if arch:
        _ensure_ablation_on_path()
        if arch in {"egnn", "graph_transformer", "graph_transformer_gaussian"}:
            from gnn_models import GNN_REGISTRY  # type: ignore

            model = GNN_REGISTRY[arch](
                input_dim=input_dim,
                cond_dim=cond_dim,
                hidden_dim=hidden_dim,
                n_layers=n_layers,
                n_heads=n_heads,
                dropout=dropout,
            ).to(device)
        elif arch in {"set_transformer", "settransformer", "value_set_transformer", "set_transformer_gaussian"}:
            from new_architectures import ValueSetTransformer, ValueSetTransformerGaussian  # type: ignore

            model_cls = ValueSetTransformerGaussian if arch == "set_transformer_gaussian" else ValueSetTransformer

            model = model_cls(
                input_dim=input_dim,
                cond_dim=cond_dim,
                hidden_dim=hidden_dim,
                n_layers=n_layers,
                n_heads=n_heads,
                dropout=dropout,
                min_logvar=float(args_dict.get("min_logvar", -6.0)),
                max_logvar=float(args_dict.get("max_logvar", 3.5)),
            ).to(device)
        elif arch in {"value_transformer", "transformer"}:
            model = ValueTransformer(
                input_dim=input_dim,
                cond_dim=cond_dim,
                hidden_dim=hidden_dim,
                num_stones=num_stones,
                n_layers=n_layers,
                n_heads=n_heads,
                dropout=dropout,
            ).to(device)
        else:
            raise ValueError(f"Unknown checkpoint arch='{arch}' in {model_path}")
    else:
        # Backward-compat: infer architecture from state dict keys.
        state_keys = set(ckpt["model_state_dict"].keys())
        if "team_embed.weight" in state_keys and "stone_index_embed.weight" not in state_keys:
            # SetTransformer (no stone-index embeddings, has team + inplay embeddings)
            _ensure_ablation_on_path()
            from new_architectures import ValueSetTransformer, ValueSetTransformerGaussian  # type: ignore

            is_gaussian = "mean_head.0.weight" in state_keys and "logvar_head.0.weight" in state_keys
            model_cls = ValueSetTransformerGaussian if is_gaussian else ValueSetTransformer
            model = model_cls(
                input_dim=input_dim,
                cond_dim=cond_dim,
                hidden_dim=hidden_dim,
                n_layers=n_layers,
                n_heads=n_heads,
                dropout=dropout,
                min_logvar=float(args_dict.get("min_logvar", -6.0)),
                max_logvar=float(args_dict.get("max_logvar", 3.5)),
            ).to(device)
        else:
            model = ValueTransformer(
                input_dim=input_dim,
                cond_dim=cond_dim,
                hidden_dim=hidden_dim,
                num_stones=num_stones,
                n_layers=n_layers,
                n_heads=n_heads,
                dropout=dropout,
            ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    def predict(x_flat: np.ndarray, c_vec: np.ndarray):
        x_arr = np.asarray(x_flat, dtype=np.float32)
        c_arr = np.asarray(c_vec, dtype=np.float32)
        single = x_arr.ndim == 1

        if single:
            x_arr = x_arr.reshape(1, -1)
        if c_arr.ndim == 1:
            c_arr = np.broadcast_to(c_arr.reshape(1, -1), (x_arr.shape[0], c_arr.shape[0]))
        elif c_arr.ndim == 2 and c_arr.shape[0] == 1 and x_arr.shape[0] > 1:
            c_arr = np.broadcast_to(c_arr, (x_arr.shape[0], c_arr.shape[1]))
        elif c_arr.ndim != 2 or c_arr.shape[0] != x_arr.shape[0]:
            raise ValueError(
                f"Context batch shape {c_arr.shape} does not match feature batch shape {x_arr.shape}."
            )

        c_arr = np.stack([_coerce_context_dim(row, cond_dim) for row in c_arr], axis=0).astype(np.float32)
        x_t = torch.tensor(x_arr, dtype=torch.float32, device=device)
        c_t = torch.tensor(c_arr, dtype=torch.float32, device=device)
        with torch.no_grad():
            out = model(x_t, c_t)
            if isinstance(out, tuple):
                out = out[0]
            val = out.detach().cpu().numpy().reshape(-1).astype(np.float32, copy=False)
        return float(val[0]) if single else val

    return predict, cond_dim


# ----------------------------
# Noise sampler (supports local/grouped/uniform)
# ----------------------------
@dataclass
class NoiseSampler:
    mode: str  # "local" or "grouped" or "uniform"
    default_std: np.ndarray
    task_handle: Dict[str, Dict]
    player_task: Dict[str, Dict]
    local_std: np.ndarray
    local_cfg: Dict
    min_std: float = 1e-3
    uniform_low: np.ndarray | None = None
    uniform_high: np.ndarray | None = None

    @classmethod
    def from_config(cls, cfg: Dict, default_std: Iterable[float]):
        default = np.array(default_std, dtype=np.float32)
        mode = str(cfg.get("mode", "grouped")) if isinstance(cfg, dict) else "grouped"
        task_handle = cfg.get("by_task_handle", {}) if isinstance(cfg, dict) else {}
        player_task = cfg.get("by_player_task", {}) if isinstance(cfg, dict) else {}

        if isinstance(cfg, dict) and "default" in cfg and isinstance(cfg["default"], dict) and "std" in cfg["default"]:
            default = np.array(cfg["default"]["std"], dtype=np.float32)

        local_cfg = cfg.get("local", {}) if isinstance(cfg, dict) and isinstance(cfg.get("local", {}), dict) else {}
        local_std = default.copy()
        if "std" in local_cfg:
            local_std = np.array(local_cfg["std"], dtype=np.float32)

        min_std = float(local_cfg.get("min_std", cfg.get("meta", {}).get("min_std", 1e-3))) if isinstance(cfg, dict) else 1e-3

        uniform_low = None
        uniform_high = None
        if isinstance(cfg, dict) and "uniform" in cfg and isinstance(cfg["uniform"], dict):
            u = cfg["uniform"]
            if "low" in u and "high" in u:
                uniform_low = np.array(u["low"], dtype=np.float32).reshape(4)
                uniform_high = np.array(u["high"], dtype=np.float32).reshape(4)

        return cls(
            mode=mode,
            default_std=default,
            task_handle=task_handle,
            player_task=player_task,
            local_std=local_std,
            local_cfg=local_cfg,
            min_std=min_std,
            uniform_low=uniform_low,
            uniform_high=uniform_high,
        )

    def _select_entry(self, task, handle, player_id=None) -> Dict | None:
        try:
            if player_id is not None and not pd.isna(player_id) and not pd.isna(task):
                key = f"player_{int(player_id)}_task_{int(task)}"
                if key in self.player_task:
                    return self.player_task[key]
            if pd.isna(task) or pd.isna(handle):
                return None
            th_key = f"task_{int(task)}_handle_{int(handle)}"
            return self.task_handle.get(th_key)
        except Exception:
            return None

    def draw(
        self,
        rng: np.random.Generator,
        center: np.ndarray,
        task,
        handle,
        player_id=None,
        cov_from_cfg: bool = False,
        bounds=None,
    ) -> np.ndarray:
        m = self.mode.lower().strip()

        if m in ("uniform", "global_uniform", "uniform_global"):
            if self.uniform_low is not None and self.uniform_high is not None:
                low = self.uniform_low
                high = self.uniform_high
            else:
                if bounds is None:
                    raise ValueError("NoiseSampler(mode='uniform') requires SolveBounds passed as 'bounds'.")
                low = np.array([bounds.speed_min, bounds.angle_min, bounds.spin_min, bounds.y0_min], dtype=np.float32)
                high = np.array([bounds.speed_max, bounds.angle_max, bounds.spin_max, bounds.y0_max], dtype=np.float32)
            return rng.uniform(low=low, high=high).astype(np.float32)

        if m == "local":
            local_cfg = self.local_cfg if isinstance(self.local_cfg, dict) else {}
            dist_name = str(local_cfg.get("distribution", "gaussian")).strip().lower()
            std = np.maximum(self.local_std.astype(np.float32), self.min_std)

            if dist_name == "student_t":
                nu = float(local_cfg.get("nu", 5.0))
                if nu <= 2.0:
                    raise ValueError(f"Student-t local noise requires nu>2, got {nu}")

                z = rng.standard_t(df=nu, size=4).astype(np.float32)
                # Convert requested gaussian-equivalent stds into Student-t scales.
                t_scale = np.sqrt((nu - 2.0) / nu).astype(np.float32)
                scales = std * t_scale

                if "speed_scale" in local_cfg:
                    scales[0] = max(float(local_cfg["speed_scale"]), self.min_std)

                if "angle_speed_range" in local_cfg and np.isfinite(center[0]):
                    speed_range = np.array(local_cfg["angle_speed_range"], dtype=np.float32).reshape(2)
                    lo_speed, hi_speed = float(speed_range[0]), float(speed_range[1])
                    speed = float(np.clip(abs(float(center[0])), lo_speed, hi_speed))
                    frac = 0.0 if hi_speed <= lo_speed else (speed - lo_speed) / (hi_speed - lo_speed)

                    if "angle_scale_range" in local_cfg:
                        scale_range = np.array(local_cfg["angle_scale_range"], dtype=np.float32).reshape(2)
                        # Bowling note: larger speed -> smaller angular error.
                        angle_scale = float(scale_range[1] + frac * (scale_range[0] - scale_range[1]))
                        scales[1] = max(angle_scale, self.min_std)
                    elif "angle_variance_range" in local_cfg:
                        # Backward-compatible handling for the stale configs.
                        var_range = np.array(local_cfg["angle_variance_range"], dtype=np.float32).reshape(2)
                        angle_var = float(var_range[1] + frac * (var_range[0] - var_range[1]))
                        scales[1] = max(math.sqrt(max(angle_var, 0.0)) * float(t_scale), self.min_std)

                return (center + z * scales).astype(np.float32)

            cov = np.diag(std ** 2)
            return rng.multivariate_normal(center, cov).astype(np.float32)

        entry = self._select_entry(task, handle, player_id)
        if entry is None:
            std = np.maximum(self.default_std.astype(np.float32), self.min_std)
            cov = np.diag(std ** 2)
        else:
            std = np.maximum(np.array(entry.get("std", self.default_std), dtype=np.float32), self.min_std)
            if cov_from_cfg and "cov" in entry:
                cov = np.array(entry["cov"], dtype=np.float32)
            else:
                cov = np.diag(std ** 2)

        return rng.multivariate_normal(center, cov).astype(np.float32)


# ----------------------------
# Physics sampler (stochastic per-throw physics)
# ----------------------------
@dataclass
class PhysicsSampler:
    """Sample physics parameter vectors from a fitted multivariate log-normal."""
    mean_log: np.ndarray
    cov_log: np.ndarray
    clip_lo: np.ndarray
    clip_hi: np.ndarray
    phys_keys: List[str]

    @classmethod
    def from_json(cls, path: str | pathlib.Path) -> "PhysicsSampler":
        data = json.loads(pathlib.Path(path).read_text())
        return cls(
            mean_log=np.array(data["mean_log"], dtype=np.float64),
            cov_log=np.array(data["cov_log"], dtype=np.float64),
            clip_lo=np.array(data["clip_lo"], dtype=np.float64),
            clip_hi=np.array(data["clip_hi"], dtype=np.float64),
            phys_keys=data["phys_keys"],
        )

    def draw(self, rng: np.random.Generator) -> np.ndarray:
        log_sample = rng.multivariate_normal(self.mean_log, self.cov_log)
        sample = np.exp(log_sample)
        return np.clip(sample, self.clip_lo, self.clip_hi).astype(np.float32)

    def draw_batch(self, rng: np.random.Generator, n: int) -> np.ndarray:
        log_samples = rng.multivariate_normal(self.mean_log, self.cov_log, size=n)
        samples = np.exp(log_samples)
        return np.clip(samples, self.clip_lo, self.clip_hi).astype(np.float32)


# ----------------------------
# Data loading / merge
# ----------------------------
def load_inverse(glob_pattern: str) -> pd.DataFrame:
    pattern_path = pathlib.Path(glob_pattern)
    if pattern_path.is_absolute():
        paths = sorted(pathlib.Path(p) for p in pattern_path.parent.glob(pattern_path.name))
    else:
        paths = sorted(pathlib.Path(".").glob(glob_pattern))
    if not paths:
        raise FileNotFoundError(f"No inverse files matched: {glob_pattern}")
    frames = [pd.read_csv(p) for p in paths]
    return pd.concat(frames, ignore_index=True)


def prepare_dataframe_all(
    stones_csv: str,
    inverse_glob: str,
    only_solver_ok: bool,
    hard_loss_max: float | None,
) -> pd.DataFrame:
    inv_df = load_inverse(inverse_glob)

    stones_df = pd.read_csv(stones_csv)
    stones_df = compute_shot_norm_and_order(stones_df)
    stones_df = stones_df.sort_values(SHOT_KEY).reset_index(drop=True)

    end_group = ["CompetitionID", "SessionID", "GameID", "EndID"]
    stones_df["shot_norm_next"] = stones_df["shot_norm"].astype(float)
    stones_df["shot_norm_prev"] = stones_df.groupby(end_group)["shot_norm"].shift(1).astype(float)
    stones_df["ShotID_prev"] = stones_df.groupby(end_group)["ShotID"].shift(1)

    meta_cols = SHOT_KEY + [
        "TeamID",
        "PlayerID",
        "Task",
        "Handle",
        "ShotIndex",
        "ShotsInEnd",
        "shot_norm_prev",
        "shot_norm_next",
        "ShotID_prev",
        "team_order",
    ]
    merged = pd.merge(inv_df, stones_df[meta_cols], on=SHOT_KEY, how="left", validate="one_to_one")
    merged = add_throw_slot_features(merged)

    if only_solver_ok and "solver_ok" in merged.columns:
        merged = merged[merged["solver_ok"] == True].copy()  # noqa: E712

    if hard_loss_max is not None and "hard_loss_refine" in merged.columns:
        merged = merged[merged["hard_loss_refine"] <= float(hard_loss_max)].copy()

    return merged.reset_index(drop=True)


# ----------------------------
# JAX simulator (batched, cached by shape)
# ----------------------------
def clip_to_bounds(x: np.ndarray, bounds: SolveBounds) -> np.ndarray:
    lo = np.array([bounds.speed_min, bounds.angle_min, bounds.spin_min, bounds.y0_min], dtype=np.float32)
    hi = np.array([bounds.speed_max, bounds.angle_max, bounds.spin_max, bounds.y0_max], dtype=np.float32)
    return np.clip(x, lo, hi).astype(np.float32)


class SimFnCache:
    def __init__(self, p: CurlingParams):
        self.p = p
        self._cache: Dict[Tuple[int, int], object] = {}

    def get(self, prev_n: int, batch_size: int):
        key = (int(prev_n), int(batch_size))
        if key in self._cache:
            return self._cache[key]

        p = self.p

        def _sim_one(prev_xy, x_params):
            return simulate_from_params(p, prev_xy, x_params, dynamic=False)

        def _sim_batch(prev_xy, x_batch):
            return jax.vmap(lambda x: _sim_one(prev_xy, x))(x_batch)

        sim_fn = jax.jit(_sim_batch)
        self._cache[key] = sim_fn
        return sim_fn


class SimFnCacheFlex:
    """Like SimFnCache but vmaps over both throw params and physics params."""
    def __init__(self, p: CurlingParams):
        self.p = p
        self._cache: Dict[Tuple[int, int], object] = {}

    def get(self, prev_n: int, batch_size: int):
        key = (int(prev_n), int(batch_size))
        if key in self._cache:
            return self._cache[key]

        p = self.p

        def _sim_one(prev_xy, x_params, phys):
            return simulate_from_params_flex(p, prev_xy, x_params, phys)

        def _sim_batch(prev_xy, x_batch, phys_batch):
            return jax.vmap(lambda x, ph: _sim_one(prev_xy, x, ph))(x_batch, phys_batch)

        sim_fn = jax.jit(_sim_batch)
        self._cache[key] = sim_fn
        return sim_fn


# ----------------------------
# Stats helpers
# ----------------------------
def percentile_of_score(samples: np.ndarray, obs: float) -> float:
    if samples.size == 0 or not np.isfinite(obs):
        return math.nan
    return float(np.mean(samples <= obs))


def cvar(values: np.ndarray, alpha: float) -> float:
    if values.size == 0:
        return math.nan
    k = max(1, int(values.size * alpha))
    part = np.partition(values, k - 1)[:k]
    return float(np.mean(part))


# ----------------------------
# Core scoring loop
# ----------------------------
def score_dataframe(
    df: pd.DataFrame,
    model_fn,
    sampler: NoiseSampler,
    curl_params: CurlingParams,
    bounds: SolveBounds,
    num_samples: int,
    seed: int,
    use_cov: bool,
    use_rule_based_terminal: bool = False,
    verbose_every: int = 0,
    desc: str = "",
    physics_sampler: PhysicsSampler | None = None,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    sim_cache = SimFnCache(curl_params)
    sim_cache_flex = SimFnCacheFlex(curl_params) if physics_sampler is not None else None
    B = int(num_samples)

    rows_out: List[dict] = []
    # Cache v_next per ShotID within an end, for zero-sum v_prev computation.
    # Cleared at each new end.
    _v_next_cache: Dict[int, float] = {}
    _prev_end_key = None
    it = df.itertuples(index=False)

    pbar = tqdm(it, total=len(df), desc=desc) if desc else tqdm(it, total=len(df))
    for idx, row in enumerate(pbar):
        row_dict = row._asdict()
        srow = pd.Series(row_dict)

        # Clear v_next cache at end boundaries
        end_key = (row_dict.get("CompetitionID"), row_dict.get("SessionID"),
                   row_dict.get("GameID"), row_dict.get("EndID"))
        if end_key != _prev_end_key:
            _v_next_cache.clear()
            _prev_end_key = end_key

        prev_mat, _ = extract_state_from_row(srow, "prev")
        next_mat, _ = extract_state_from_row(srow, "next")

        _, prev_ids_compact = compact_positions(prev_mat)
        _, next_ids = compact_positions(next_mat)
        prev_n = len(prev_ids_compact)

        # Build 12-slot padded arrays matching inverse pipeline
        prev_slots, prev_slot_mask = state_to_fixed_slot_arrays(prev_mat)
        if int(np.sum(prev_slot_mask)) > 1:
            prev_slots[prev_slot_mask] = _separate_overlaps(prev_slots[prev_slot_mask])

        obs_throw_slot_id = float(row_dict.get("obs_throw_slot_id", np.nan))
        team_slot_block = float(row_dict.get("team_slot_block", np.nan))
        thrower_block = infer_thrower_block(
            prev_ids=prev_ids_compact,
            next_ids=next_ids,
            obs_throw_slot_id=obs_throw_slot_id,
            team_slot_block=team_slot_block,
        )
        new_id = choose_new_slot_id(
            prev_ids=prev_ids_compact,
            next_ids=next_ids,
            thrower_block=thrower_block,
            obs_throw_slot_id=obs_throw_slot_id,
        )

        shot_norm_prev = float(row_dict.get("shot_norm_prev", np.nan))
        shot_norm_next = float(row_dict.get("shot_norm_next", np.nan))
        if not np.isfinite(shot_norm_prev) and np.isfinite(shot_norm_next):
            shot_norm_prev = shot_norm_next
        if not np.isfinite(shot_norm_prev):
            shot_norm_prev = 0.0
        if not np.isfinite(shot_norm_next):
            shot_norm_next = shot_norm_prev

        shot_index_next = float(row_dict.get("ShotIndex", np.nan))
        if not np.isfinite(shot_index_next):
            shot_index_next = 0.0
        shot_index_prev = shot_index_next - 1.0
        shots_in_end = float(row_dict.get("ShotsInEnd", np.nan))

        team_order = float(row_dict.get("team_order", 0.0))
        stone_block = float(thrower_block)

        c_next = np.array([shot_norm_next, team_order, stone_block], dtype=np.float32)

        next_defaults = make_raw_defaults_for_state(shot_index_next, team_order, thrower_block)
        v_next = evaluate_state_value(
            model_fn=model_fn,
            board_m=next_mat,
            raw_defaults=next_defaults,
            c_vec=c_next,
            stone_block=stone_block,
            shot_index=shot_index_next,
            shots_in_end=shots_in_end,
            shot_norm=shot_norm_next,
            use_rule_based_terminal=bool(use_rule_based_terminal),
        )

        # Zero-sum enforcement: v_prev = -v_next of the previous shot (opponent's
        # perspective on the same board state).  For the first scored shot in an
        # end (no predecessor) we fall back to evaluating prev_mat directly.
        shot_id_prev = row_dict.get("ShotID_prev", np.nan)
        if np.isfinite(shot_id_prev) and int(shot_id_prev) in _v_next_cache:
            v_prev = -_v_next_cache[int(shot_id_prev)]
        else:
            # No predecessor v_next to sign-flip from (first scored shot in end).
            # Fall back to evaluating prev_mat from thrower's perspective.
            # This is imperfect (non-zero-sum conditioning) but self-consistent
            # with the same model used for v_next and v_sim.
            c_prev = np.array([shot_norm_prev, team_order, stone_block], dtype=np.float32)
            prev_defaults = make_raw_defaults_for_state(shot_index_prev, team_order, thrower_block)
            v_prev = evaluate_state_value(
                model_fn=model_fn,
                board_m=prev_mat,
                raw_defaults=prev_defaults,
                c_vec=c_prev,
                stone_block=stone_block,
                shot_index=shot_index_prev,
                shots_in_end=shots_in_end,
                shot_norm=shot_norm_prev,
                use_rule_based_terminal=bool(use_rule_based_terminal),
            )

        dv_obs = v_next - v_prev

        # Cache v_next for use as v_prev of the next shot (zero-sum sign flip)
        current_shot_id = row_dict.get("ShotID", np.nan)
        if np.isfinite(current_shot_id):
            _v_next_cache[int(current_shot_id)] = v_next

        est_params = np.array([row_dict.get(c, np.nan) for c in PARAM_COLS], dtype=np.float32)
        dv_samples = np.empty((0,), dtype=np.float32)

        valid_center = np.all(np.isfinite(est_params)) and np.isfinite(dv_obs) and (0 <= prev_n <= 12)
        if valid_center:
            x_batch = np.zeros((B, 4), dtype=np.float32)
            task = row_dict.get("Task", 0)
            handle = row_dict.get("Handle", 0)
            player_id = row_dict.get("PlayerID", None)

            for b in range(B):
                s = sampler.draw(
                    rng,
                    center=est_params,
                    task=task,
                    handle=handle,
                    player_id=player_id,
                    cov_from_cfg=use_cov,
                    bounds=bounds,
                )
                x_batch[b] = clip_to_bounds(s, bounds)

            # Simulate with full 12-slot array (matching inverse pipeline)
            prev_j = jnp.asarray(prev_slots, dtype=jnp.float32)
            x_j = jnp.asarray(x_batch, dtype=jnp.float32)
            if physics_sampler is not None and sim_cache_flex is not None:
                phys_batch = physics_sampler.draw_batch(rng, B)
                phys_j = jnp.asarray(phys_batch, dtype=jnp.float32)
                sim_fn_flex = sim_cache_flex.get(12, B)
                finals = np.asarray(sim_fn_flex(prev_j, x_j, phys_j))  # (B, 13, 2)
            else:
                sim_fn = sim_cache.get(12, B)
                finals = np.asarray(sim_fn(prev_j, x_j))  # (B, 13, 2)

            full_final_batch = assign_finals_12slot_batch(finals, prev_slot_mask, new_id)
            c_next_batch = np.broadcast_to(c_next.reshape(1, -1), (B, c_next.shape[0])).astype(np.float32, copy=False)
            v_sim_batch = evaluate_state_value_batch(
                model_fn=model_fn,
                board_batch_m=full_final_batch,
                raw_defaults=next_defaults,
                c_batch=c_next_batch,
                stone_block=stone_block,
                shot_index=shot_index_next,
                shots_in_end=shots_in_end,
                shot_norm=shot_norm_next,
                use_rule_based_terminal=bool(use_rule_based_terminal),
            )
            dv_samples = v_sim_batch - np.float32(v_prev)

        if dv_samples.size > 0:
            dv_mean = float(np.mean(dv_samples))
            dv_std = float(np.std(dv_samples, ddof=1)) if dv_samples.size > 1 else math.nan
            dv_p10 = float(np.percentile(dv_samples, 10))
            dv_p50 = float(np.percentile(dv_samples, 50))
            dv_p90 = float(np.percentile(dv_samples, 90))
            cvar10 = cvar(dv_samples, 0.10)
            pct_obs = percentile_of_score(dv_samples, float(dv_obs))
            z_obs = (float(dv_obs) - dv_mean) / (float(dv_std) + 1e-8) if np.isfinite(dv_std) else math.nan
            se_mean = float(dv_std) / math.sqrt(dv_samples.size) if dv_samples.size > 1 and np.isfinite(dv_std) else math.nan
        else:
            dv_mean = dv_std = dv_p10 = dv_p50 = dv_p90 = cvar10 = pct_obs = z_obs = se_mean = math.nan

        # Perspective conventions:
        # - dv_* columns are thrower-perspective (same as TeamID on this row).
        # - opponent columns are sign-flipped.
        # - team_order{0,1} columns provide a fixed per-end team-order frame.
        sign_team0 = 1.0 if int(round(float(team_order))) == 0 else -1.0
        dv_obs_thrower = float(dv_obs) if np.isfinite(dv_obs) else math.nan
        dv_mean_thrower = float(dv_mean) if np.isfinite(dv_mean) else math.nan
        dv_obs_opponent = -dv_obs_thrower if np.isfinite(dv_obs_thrower) else math.nan
        dv_mean_opponent = -dv_mean_thrower if np.isfinite(dv_mean_thrower) else math.nan
        dv_obs_team_order0 = dv_obs_thrower * sign_team0 if np.isfinite(dv_obs_thrower) else math.nan
        dv_mean_team_order0 = dv_mean_thrower * sign_team0 if np.isfinite(dv_mean_thrower) else math.nan
        dv_obs_team_order1 = -dv_obs_team_order0 if np.isfinite(dv_obs_team_order0) else math.nan
        dv_mean_team_order1 = -dv_mean_team_order0 if np.isfinite(dv_mean_team_order0) else math.nan

        out_row = {k: row_dict.get(k, np.nan) for k in SHOT_KEY + ["TeamID", "PlayerID", "Task", "Handle"]}
        out_row.update(
            dict(
                shot_norm_prev=float(shot_norm_prev),
                shot_norm_next=float(shot_norm_next),
                stone_block=float(stone_block),
                team_order=float(team_order),
                v_prev=float(v_prev),
                v_next=float(v_next),
                dv_obs=float(dv_obs) if np.isfinite(dv_obs) else math.nan,
                dv_mean=dv_mean,
                dv_std=dv_std,
                dv_p10=dv_p10,
                dv_p50=dv_p50,
                dv_p90=dv_p90,
                cvar_10=cvar10,
                percentile_obs=pct_obs,
                z_obs=z_obs,
                se_mean=se_mean,
                xscore_perspective="thrower",
                dv_obs_thrower=dv_obs_thrower,
                dv_mean_thrower=dv_mean_thrower,
                dv_obs_opponent=dv_obs_opponent,
                dv_mean_opponent=dv_mean_opponent,
                dv_obs_team_order0=dv_obs_team_order0,
                dv_mean_team_order0=dv_mean_team_order0,
                dv_obs_team_order1=dv_obs_team_order1,
                dv_mean_team_order1=dv_mean_team_order1,
                sample_count=int(dv_samples.size),
                prev_N=prev_n,
                solver_ok=bool(row_dict.get("solver_ok", True)),
                hard_loss=float(row_dict.get("hard_loss_refine", math.nan)),
                est_speed=float(row_dict.get("est_speed", math.nan)),
                est_angle=float(row_dict.get("est_angle", math.nan)),
                est_spin=float(row_dict.get("est_spin", math.nan)),
                est_y0=float(row_dict.get("est_y0", math.nan)),
            )
        )
        rows_out.append(out_row)

        if verbose_every and (idx + 1) % int(verbose_every) == 0:
            print(f"[progress] processed {idx+1}/{len(df)} shots", flush=True)

    return pd.DataFrame(rows_out)


def write_csv(df: pd.DataFrame, out_path: pathlib.Path) -> pathlib.Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return out_path


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser(description="Monte Carlo shot-value scorer, per-competition outputs.")

    ap.add_argument("--stones-csv", type=str, default="2026/Stones.csv")
    ap.add_argument("--inverse-glob", type=str, default="inverseDataset/stones_with_estimates.chunk*.csv")
    ap.add_argument("--value-model", type=str, default="valueModel/value_model_synth_v4best.pt")
    ap.add_argument("--noise-config", type=str, default="noise_config_old.json")
    ap.add_argument("--num-samples", type=int, default=128)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--device", type=str, default="cpu")

    ap.add_argument("--only-solver-ok", action="store_true")
    ap.add_argument("--hard-loss-max", type=float, default=0.5)
    ap.add_argument("--use-cov", action="store_true")
    ap.add_argument(
        "--rule-based-terminal",
        action="store_true",
        help="Score terminal end states by curling counting-stone rules instead of the learned value model.",
    )
    ap.add_argument("--verbose-every", type=int, default=0)
    ap.add_argument("--physics-dist", type=str, default=None,
                    help="JSON file with fitted physics parameter distribution (enables stochastic physics per MC variant)")

    # competition controls
    ap.add_argument("--out-dir", type=str, default="shot_scores_by_competition", help="Directory for per-competition CSVs")
    ap.add_argument("--write-combined", action="store_true", help="Also write combined CSV after all competitions")
    ap.add_argument("--combined-name", type=str, default="shot_scores_all.csv")

    ap.add_argument("--competition-ids", type=str, default="", help="Comma-separated CompetitionID list to run (subset)")
    ap.add_argument("--only-competition", type=int, default=None, help="Run only this CompetitionID (single)")
    ap.add_argument("--num-shards", type=int, default=1, help="Split selected rows into this many shards.")
    ap.add_argument("--shard-index", type=int, default=0, help="0-based shard index to run.")

    ap.add_argument("--limit-per-competition", type=int, default=None, help="Optional cap on rows per competition (debug)")

    # Smoke test controls (per competition)
    ap.add_argument("--no-smoke", action="store_true")
    ap.add_argument("--smoke-limit", type=int, default=32)
    ap.add_argument("--smoke-samples", type=int, default=16)
    ap.add_argument("--smoke-prefix", type=str, default="smoke_", help="Prefix for per-competition smoke CSV files")

    args = ap.parse_args()

    if CURLING_IMPORT_ERROR is not None:
        raise SystemExit(
            "Missing simulation dependency (JAX/curling_sim_jax). "
            "Activate the correct environment or install JAX.\n"
            f"Original import error: {CURLING_IMPORT_ERROR}"
        )
    if args.num_samples <= 0:
        raise SystemExit("--num-samples must be > 0")

    # Load + merge once
    full_df = prepare_dataframe_all(
        stones_csv=args.stones_csv,
        inverse_glob=args.inverse_glob,
        only_solver_ok=bool(args.only_solver_ok),
        hard_loss_max=(None if args.hard_loss_max is None or args.hard_loss_max < 0 else float(args.hard_loss_max)),
    )
    if "CompetitionID" not in full_df.columns:
        raise SystemExit("Merged dataframe missing CompetitionID; check your joins / inputs.")

    # Choose competitions to run
    comp_ids = sorted(pd.unique(full_df["CompetitionID"].dropna()).tolist())
    comp_ids = [int(x) for x in comp_ids]

    if args.only_competition is not None:
        comp_ids = [int(args.only_competition)]

    if args.competition_ids.strip():
        wanted = [int(x.strip()) for x in args.competition_ids.split(",") if x.strip()]
        comp_ids = [c for c in comp_ids if c in set(wanted)]

    if not comp_ids:
        raise SystemExit("No competitions selected (after filtering).")

    if args.num_shards < 1:
        raise SystemExit("--num-shards must be >= 1")
    if not (0 <= args.shard_index < args.num_shards):
        raise SystemExit("--shard-index must satisfy 0 <= shard-index < num-shards")

    print(f"[info] merged dataset rows: {len(full_df)}")
    print(f"[info] competitions selected: {comp_ids} (count={len(comp_ids)})")
    if args.hard_loss_max is not None and args.hard_loss_max >= 0:
        print(f"[info] hard_loss_refine <= {float(args.hard_loss_max)}")
    if args.rule_based_terminal:
        print("[info] terminal end states will be scored by curling-rule counting logic", flush=True)
    if args.num_shards > 1:
        print(f"[info] shard {args.shard_index + 1}/{args.num_shards}")

    # Model + noise + sim params loaded once
    model_fn, model_cond_dim = load_value_model(pathlib.Path(args.value_model), device=args.device)
    if model_cond_dim is not None and model_cond_dim != 4:
        print(f"[warn] value model checkpoint cond_dim={model_cond_dim}; scorer will coerce context vector accordingly.", flush=True)

    cfg = {}
    cfg_path = pathlib.Path(args.noise_config)
    if cfg_path.exists():
        try:
            cfg = json.loads(cfg_path.read_text())
        except Exception as e:
            print(f"[warn] failed to parse noise_config ({cfg_path}): {e} ; using defaults", flush=True)
            cfg = {}
    sampler = NoiseSampler.from_config(cfg, default_std=[0.20, 0.05, 0.50, 0.10])

    physics_sampler = None
    if args.physics_dist:
        phys_dist_path = pathlib.Path(args.physics_dist)
        if phys_dist_path.exists():
            physics_sampler = PhysicsSampler.from_json(phys_dist_path)
            print(f"[info] loaded physics distribution from {phys_dist_path} "
                  f"({len(physics_sampler.phys_keys)} params)", flush=True)
        else:
            print(f"[warn] physics-dist file not found: {phys_dist_path}; using fixed physics", flush=True)

    curl_params = contact_mild_params(
        CurlingParams,
        dt=0.02,
        substeps=2,
        k_penalty=2.5e4,
    )
    bounds = SolveBounds()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_parts: List[pd.DataFrame] = []

    # Iterate competitions
    for comp_id in comp_ids:
        comp_df = full_df[full_df["CompetitionID"] == comp_id].copy()
        if args.limit_per_competition is not None:
            comp_df = comp_df.head(int(args.limit_per_competition)).copy()
        if args.num_shards > 1:
            mask = (np.arange(len(comp_df)) % int(args.num_shards)) == int(args.shard_index)
            comp_df = comp_df.iloc[mask].copy()

        print(f"\n[comp {comp_id}] rows={len(comp_df)}")

        if len(comp_df) == 0:
            print(f"[comp {comp_id}] skipping (no rows)")
            continue

        # Smoke test per competition
        if not args.no_smoke:
            smoke_n = min(int(args.smoke_limit), len(comp_df))
            smoke_df = comp_df.head(smoke_n).copy()
            print(f"[comp {comp_id}][smoke] shots={smoke_n}, samples={int(args.smoke_samples)}")

            smoke_scores = score_dataframe(
                smoke_df,
                model_fn=model_fn,
                sampler=sampler,
                curl_params=curl_params,
                bounds=bounds,
                num_samples=int(args.smoke_samples),
                seed=int(args.seed) + 13 * int(comp_id),
                use_cov=bool(args.use_cov),
                use_rule_based_terminal=bool(args.rule_based_terminal),
                verbose_every=0,
                desc=f"smoke comp {comp_id}",
                physics_sampler=physics_sampler,
            )
            smoke_out = out_dir / f"{args.smoke_prefix}{comp_id}.csv"
            write_csv(smoke_scores, smoke_out)
            print(f"[comp {comp_id}][smoke] wrote: {smoke_out}")

            required_cols = ["dv_obs", "dv_mean", "dv_std", "sample_count", "team_order"]
            for c in required_cols:
                if c not in smoke_scores.columns:
                    raise SystemExit(f"[comp {comp_id}][smoke] missing required output column: {c}")
            if smoke_scores["dv_obs"].notna().sum() == 0:
                raise SystemExit(f"[comp {comp_id}][smoke] dv_obs is all NaN; check value model inputs / joins.")
            print(f"[comp {comp_id}][smoke] OK. Proceeding to full.")

        # Full run per competition
        print(f"[comp {comp_id}][full] samples={int(args.num_samples)}")
        comp_scores = score_dataframe(
            comp_df,
            model_fn=model_fn,
            sampler=sampler,
            curl_params=curl_params,
            bounds=bounds,
            num_samples=int(args.num_samples),
            seed=int(args.seed) + 999 + 37 * int(comp_id),
            use_cov=bool(args.use_cov),
            use_rule_based_terminal=bool(args.rule_based_terminal),
            verbose_every=int(args.verbose_every),
            desc=f"full comp {comp_id}",
            physics_sampler=physics_sampler,
        )

        comp_out = out_dir / f"shot_scores_comp_{comp_id}.csv"
        write_csv(comp_scores, comp_out)
        print(f"[comp {comp_id}][done] wrote {len(comp_scores)} rows -> {comp_out}")

        if args.write_combined:
            all_parts.append(comp_scores)

    if args.write_combined and all_parts:
        combined = pd.concat(all_parts, ignore_index=True)
        combined_out = out_dir / args.combined_name
        write_csv(combined, combined_out)
        print(f"\n[all] wrote combined {len(combined)} rows -> {combined_out}")


if __name__ == "__main__":
    main()
