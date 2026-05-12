#!/usr/bin/env python3
"""Build value-model tensors for canonical mixed-doubles preplaced states."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from common import FIXED_ROOT, NUM_STONES, POS_MAX

M_PER_RAW = 0.003048
PREPLACED_DIR = FIXED_ROOT / "preplaced_augment"


def compact_m_to_raw_pair(along: float, lateral: float) -> tuple[float, float]:
    return lateral / M_PER_RAW + 750.0, 800.0 - along / M_PER_RAW


def canonical_board_raw(mode: str, guard_slot: int) -> np.ndarray:
    canon = {
        "standard": {"guard": (-3.4016, 0.0), "inhouse": (0.4572, 0.0)},
        "pp_right": {"guard": (-3.4016, 1.0333), "inhouse": (-0.1524, 1.2192)},
        "pp_left": {"guard": (-3.4016, -1.0333), "inhouse": (-0.1524, -1.2192)},
    }[str(mode)]
    board = np.zeros((NUM_STONES, 2), dtype=np.float32)
    inhouse_slot = 7 if int(guard_slot) == 1 else 1
    for slot, key in ((int(guard_slot), "guard"), (inhouse_slot, "inhouse")):
        board[slot - 1] = compact_m_to_raw_pair(*canon[key])
    return board


def load_preplaced_training_frame() -> pd.DataFrame:
    class_df = pd.read_csv(PREPLACED_DIR / "preplaced_classification.csv")
    ends = pd.read_csv(FIXED_ROOT / "2026" / "Ends.csv")
    inv = pd.read_csv(PREPLACED_DIR / "first_shots_inverse.csv")

    key = ["CompetitionID", "SessionID", "GameID", "EndID"]
    s7 = pd.read_csv(FIXED_ROOT / "2026" / "Stones.csv")
    s7 = s7[s7["ShotID"] == 7][key + ["TeamID"]].rename(columns={"TeamID": "first_team_id"})

    ends = ends.copy()
    ends["TotalResultInEnd"] = ends.groupby(key)["Result"].transform("sum")
    ends["ValueDiff"] = 2.0 * ends["Result"].astype(float) - ends["TotalResultInEnd"].astype(float)
    first = s7.merge(
        ends[key + ["TeamID", "ValueDiff"]],
        left_on=key + ["first_team_id"],
        right_on=key + ["TeamID"],
        how="inner",
    )
    df = class_df.merge(first[key + ["first_team_id", "ValueDiff"]], on=key, how="inner")
    df = df.merge(inv[key + ["thrower_block"]], on=key, how="left")
    df = df[df["mode"].isin(["standard", "pp_left", "pp_right"])].copy()
    df["guard_slot"] = df["guard_slot"].astype(int)
    fallback_block = np.where(df["guard_slot"].to_numpy() == 7, 1.0, 0.0).astype(np.float32)
    df["thrower_block"] = df["thrower_block"].fillna(pd.Series(fallback_block, index=df.index)).astype(np.float32)
    return df


def materialize_preplaced(df: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    xs, cs, ys = [], [], []
    for _, row in df.iterrows():
        board = canonical_board_raw(str(row["mode"]), int(row["guard_slot"]))
        xs.append((board.reshape(-1) / POS_MAX).astype(np.float32))
        # First state of the end, first team to throw, with the actual thrower stone block.
        cs.append(np.asarray([0.0, 0.0, float(row["thrower_block"])], dtype=np.float32))
        ys.append(np.asarray([float(row["ValueDiff"])], dtype=np.float32))
    return torch.tensor(np.stack(xs)), torch.tensor(np.stack(cs)), torch.tensor(np.stack(ys))


def materialize_preplaced_policy(
    train_competitions: set[int],
    max_loss: float | None = 0.5,
    require_solver_ok: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, pd.DataFrame]:
    df = load_preplaced_training_frame()
    inv = pd.read_csv(PREPLACED_DIR / "first_shots_inverse.csv")
    key = ["CompetitionID", "SessionID", "GameID", "EndID"]
    action_cols = ["est_speed", "est_angle", "est_spin", "est_y0"]
    inv = inv[key + action_cols + ["hard_loss_refine", "solver_ok"]].copy()
    df = df.merge(inv, on=key, how="inner")
    df = df[df["CompetitionID"].astype(int).isin(train_competitions)].copy()
    finite = np.isfinite(df[action_cols].to_numpy(dtype=np.float32)).all(axis=1)
    keep = finite
    if require_solver_ok and "solver_ok" in df:
        keep &= df["solver_ok"].astype(bool).to_numpy()
    if max_loss is not None:
        keep &= pd.to_numeric(df["hard_loss_refine"], errors="coerce").fillna(np.inf).to_numpy() <= float(max_loss)
    df = df.loc[keep].copy()

    xs, cs, actions = [], [], []
    for _, row in df.iterrows():
        board = canonical_board_raw(str(row["mode"]), int(row["guard_slot"]))
        xs.append((board.reshape(-1) / POS_MAX).astype(np.float32))
        cs.append(np.asarray([0.0, 0.0, float(row["thrower_block"])], dtype=np.float32))
        actions.append(row[action_cols].to_numpy(dtype=np.float32))
    return torch.tensor(np.stack(xs)), torch.tensor(np.stack(cs)), torch.tensor(np.stack(actions)), df


def load_preplaced_tensors_for_train(train_competitions: set[int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, pd.DataFrame]:
    df = load_preplaced_training_frame()
    df = df[df["CompetitionID"].astype(int).isin(train_competitions)].copy()
    x, c, y = materialize_preplaced(df)
    return x, c, y, df


def load_preplaced_mcts_roots(train_competitions: set[int], value_dataset_df: pd.DataFrame) -> list[dict]:
    df = load_preplaced_training_frame()
    df = df[df["CompetitionID"].astype(int).isin(train_competitions)].copy()
    key = ["CompetitionID", "SessionID", "GameID", "EndID"]
    s7 = value_dataset_df[value_dataset_df["ShotID"].astype(int) == 7][key + ["ShotIndex", "ShotsInEnd"]].copy()
    df = df.merge(s7, on=key, how="inner")
    roots = []
    for _, row in df.iterrows():
        board = canonical_board_raw(str(row["mode"]), int(row["guard_slot"]))
        roots.append(
            {
                "state_norm": (board.reshape(-1) / POS_MAX).astype(np.float32),
                "cond": np.asarray([0.0, 0.0, float(row["thrower_block"])], dtype=np.float32),
                "CompetitionID": int(row["CompetitionID"]),
                "SessionID": int(row["SessionID"]),
                "GameID": int(row["GameID"]),
                "EndID": int(row["EndID"]),
                "ShotID": 7,
                "TeamID": int(row["first_team_id"]),
                "ShotIndex": 0,
                "ShotsInEnd": int(row["ShotsInEnd"]),
                "mode": str(row["mode"]),
                "guard_slot": int(row["guard_slot"]),
            }
        )
    return roots


def canonical_preplacement_cases() -> list[dict]:
    cases = []
    for mode in ("standard", "pp_left", "pp_right"):
        for guard_slot in (1, 7):
            block = 0 if guard_slot == 1 else 1
            board = canonical_board_raw(mode, guard_slot)
            thrown_slot = 1 if block == 0 else 7
            if guard_slot == thrown_slot:
                thrown_slot += 1
            cases.append(
                {
                    "mode": mode,
                    "guard_slot": guard_slot,
                    "thrower_block": block,
                    "thrown_slot": thrown_slot,
                    "stones_raw": board,
                    "cond": np.asarray([0.0, 0.0, float(block)], dtype=np.float32),
                }
            )
    return cases
