#!/usr/bin/env python3
"""Shared utilities for the MCTS/KR-UCT value-model experiment."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
FIXED_ROOT = REPO_ROOT / "csas_fixed"

for p in (
    FIXED_ROOT,
    FIXED_ROOT / "valueModel",
    FIXED_ROOT / "valueModel" / "ablation",
    FIXED_ROOT / "inverse",
):
    sys.path.insert(0, str(p))

POS_MAX = 4095.0
NUM_STONES = 12
STONE_COLS = [f"stone_{i}_{axis}" for i in range(1, NUM_STONES + 1) for axis in ("x", "y")]
KEY_COLS = ["CompetitionID", "SessionID", "GameID", "EndID", "ShotID"]
END_KEY = ["CompetitionID", "SessionID", "GameID", "EndID"]
ACTION_COLS = ["est_speed", "est_angle", "est_spin", "est_y0"]
FLIP_CENTER_X_NORM = 1500.0 / POS_MAX


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def write_json(obj: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def log(msg: str, log_path: Path | None = None) -> None:
    print(msg, flush=True)
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a") as f:
            f.write(msg + "\n")


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def flip_state_batch(x: torch.Tensor) -> torch.Tensor:
    """Mirror normalized raw stone x-coordinates across sheet centerline."""
    out = x.clone()
    stones = out[:, : NUM_STONES * 2].view(-1, NUM_STONES, 2)
    live = (stones[:, :, 0] < 0.999) | (stones[:, :, 1] < 0.999)
    flipped_x = FLIP_CENTER_X_NORM - stones[:, :, 0]
    stones[:, :, 0] = torch.where(live, flipped_x, stones[:, :, 0])
    return out


def flip_action_batch(a: torch.Tensor) -> torch.Tensor:
    """Mirror raw throw parameters: speed unchanged, angle/spin/y0 change sign."""
    out = a.clone()
    out[:, 1:4] = -out[:, 1:4]
    return out


def team_swap_state_cond_batch(x: torch.Tensor, c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Exchange team stone inventories while preserving relative shot-order context."""
    out_x = x.clone()
    stones = out_x[:, : NUM_STONES * 2].view(-1, NUM_STONES, 2)
    first = stones[:, :6, :].clone()
    stones[:, :6, :] = stones[:, 6:, :]
    stones[:, 6:, :] = first
    out_c = c.clone()
    out_c[:, 2] = 1.0 - out_c[:, 2]
    return out_x, out_c


def random_team_swap_state_cond(
    x: torch.Tensor,
    c: torch.Tensor,
    p: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Randomly swap slot blocks 1-6 and 7-12, flipping only the stone_block condition."""
    if p <= 0.0 or x.numel() == 0:
        return x, c
    do_swap = torch.rand(x.shape[0], device=x.device) < p
    if not bool(do_swap.any()):
        return x, c
    out_x = x.clone()
    out_c = c.clone()
    out_x[do_swap], out_c[do_swap] = team_swap_state_cond_batch(out_x[do_swap], out_c[do_swap])
    return out_x, out_c


def random_flip_state_action_z(
    x: torch.Tensor,
    z: torch.Tensor,
    action_mean: torch.Tensor,
    action_std: torch.Tensor,
    p: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Randomly mirror paired policy states and standardized action targets."""
    if p <= 0.0 or x.numel() == 0:
        return x, z
    do_flip = torch.rand(x.shape[0], device=x.device) < p
    if not bool(do_flip.any()):
        return x, z
    out_x = x.clone()
    out_z = z.clone()
    out_x[do_flip] = flip_state_batch(out_x[do_flip])
    raw_a = out_z[do_flip] * action_std + action_mean
    out_z[do_flip] = (flip_action_batch(raw_a) - action_mean) / action_std
    return out_x, out_z


def random_flip_state_cond(
    x: torch.Tensor,
    c: torch.Tensor,
    p: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Randomly mirror state geometry; conditions are unchanged by horizontal reflection."""
    if p <= 0.0 or x.numel() == 0:
        return x, c
    do_flip = torch.rand(x.shape[0], device=x.device) < p
    if not bool(do_flip.any()):
        return x, c
    out_x = x.clone()
    out_c = c.clone()
    out_x[do_flip] = flip_state_batch(out_x[do_flip])
    return out_x, out_c


def in_play_raw(stones_raw: np.ndarray) -> np.ndarray:
    stones = np.asarray(stones_raw, dtype=np.float32).reshape(-1, 2)
    return ((stones[:, 0] > 0) | (stones[:, 1] > 0)) & (stones[:, 0] < POS_MAX) & (stones[:, 1] < POS_MAX)


def raw_to_compact_m(stones_raw: np.ndarray) -> np.ndarray:
    """Convert raw CSV coordinates to simulator compact meters: [along, lateral]."""
    stones = np.asarray(stones_raw, dtype=np.float32).reshape(NUM_STONES, 2)
    out = np.zeros((NUM_STONES, 2), dtype=np.float32)
    live = in_play_raw(stones)
    out[:, 0] = (800.0 - stones[:, 1]) * 0.003048
    out[:, 1] = (stones[:, 0] - 750.0) * 0.003048
    out[~live] = np.nan
    return out


def compact_m_to_raw(compact: np.ndarray) -> np.ndarray:
    """Convert simulator compact meters [along, lateral] to raw CSV coordinates."""
    compact = np.asarray(compact, dtype=np.float32).reshape(NUM_STONES, 2)
    out = np.full((NUM_STONES, 2), POS_MAX, dtype=np.float32)
    live = np.isfinite(compact).all(axis=1)
    out[live, 0] = compact[live, 1] / 0.003048 + 750.0
    out[live, 1] = 800.0 - compact[live, 0] / 0.003048
    out[~live] = POS_MAX
    return out


def next_condition(c: np.ndarray, shots_in_end: int = 16) -> np.ndarray:
    """Approximate condition for one ply later. Used only inside short search rollouts."""
    c = np.asarray(c, dtype=np.float32).copy()
    denom = max(1.0, float(shots_in_end - 1))
    c[0] = min(1.0, c[0] + 1.0 / denom)
    c[1] = 1.0 - c[1]
    c[2] = 1.0 - c[2]
    return c
