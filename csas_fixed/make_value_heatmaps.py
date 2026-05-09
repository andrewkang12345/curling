#!/usr/bin/env python3
"""Render value-model heatmaps around the button for held-out game states."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "valueModel"))
sys.path.insert(0, str(ROOT / "valueModel" / "ablation"))

# The current best GraphTF checkpoint was trained with these edge features.
os.environ.setdefault("GNN_EDGE_SCALAR_MODE", "button_visible_plus_release_reach_with_product")
os.environ.setdefault("GNN_NODE_FEATURE_MODE", "none")
os.environ.setdefault("GNN_RELEASE_NODE_MODE", "three")

from dataset import NUM_STONES, POS_MAX, ValueDataset  # noqa: E402
from gnn_models import GNN_REGISTRY  # noqa: E402
from train_holdout_models_cond3 import make_holdout_split  # noqa: E402

BUTTON_RAW = np.array([750.0, 800.0], dtype=np.float32)
MM_PER_RAW = 3.048
M_PER_RAW = MM_PER_RAW / 1000.0
STONE_RADIUS_M = 0.145
HOUSE_RADII_M = [0.1524, 0.6096, 1.2192, 1.8288]
THROWER_COLOR = "#f2c14e"
END_SENTINEL_RAW = np.array([POS_MAX, POS_MAX], dtype=np.float32)


def _in_play(stones_raw: np.ndarray) -> np.ndarray:
    return (stones_raw.sum(axis=1) > 1e-3) & (stones_raw.max(axis=1) < POS_MAX - 1e-3)


def _row_positions_raw(row: pd.Series) -> np.ndarray:
    vals = []
    for i in range(1, NUM_STONES + 1):
        vals.extend([float(row[f"stone_{i}_x"]), float(row[f"stone_{i}_y"])])
    return np.asarray(vals, dtype=np.float32).reshape(NUM_STONES, 2)


def _row_condition(row: pd.Series) -> np.ndarray:
    return np.asarray(
        [float(row["shot_norm"]), float(row["team_order"]), float(row["stone_block"])],
        dtype=np.float32,
    )


def _stone_team_name(slot: int) -> str:
    return "black" if int(slot) >= 6 else "white"


def _previous_row(df: pd.DataFrame, row_idx: int) -> pd.Series | None:
    row = df.iloc[row_idx]
    if int(row["ShotID"]) <= 1:
        return None

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


def _find_thrown_slot(df: pd.DataFrame, row_idx: int) -> int | None:
    row = df.iloc[row_idx]
    prev_row = _previous_row(df, row_idx)
    if prev_row is None:
        return None

    prev_stones = _row_positions_raw(prev_row)
    curr_stones = _row_positions_raw(row)
    added = np.flatnonzero(_in_play(curr_stones) & ~_in_play(prev_stones))
    if len(added) == 1:
        return int(added[0])

    block = int(round(float(row.get("stone_block", 0.0))))
    block_start = 6 if block else 0
    live = _in_play(curr_stones)
    block_live = np.flatnonzero(live[block_start : block_start + 6]) + block_start
    if len(block_live):
        return int(block_live[-1])
    return None


def _load_model(ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    args = ckpt.get("args", {})
    model = GNN_REGISTRY["graph_transformer"](
        input_dim=int(ckpt.get("input_dim", 24)),
        cond_dim=int(ckpt.get("cond_dim", 3)),
        hidden_dim=int(ckpt.get("hidden_dim", 256)),
        n_layers=int(args.get("n_layers", 4)),
        n_heads=int(args.get("n_heads", 4)),
        dropout=float(args.get("dropout", 0.1)),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def _draw_house(ax):
    for r in HOUSE_RADII_M:
        circle = plt.Circle((0.0, 0.0), r, fill=False, color="0.35", lw=1.0, alpha=0.9)
        ax.add_patch(circle)
    button = plt.Circle((0.0, 0.0), STONE_RADIUS_M, fill=False, color="0.25", lw=1.0)
    ax.add_patch(button)
    ax.axhline(0.0, color="0.85", lw=0.8, zorder=0)
    ax.axvline(0.0, color="0.85", lw=0.8, zorder=0)


def _plot_stones(ax, stones_raw: np.ndarray, thrown_slot: int):
    live = _in_play(stones_raw)
    xy_m = (stones_raw - BUTTON_RAW) * M_PER_RAW
    for i, (x, y) in enumerate(xy_m):
        if not live[i] or i == thrown_slot:
            continue
        face = "black" if i >= 6 else "white"
        txt = "white" if i >= 6 else "black"
        circ = plt.Circle((x, y), STONE_RADIUS_M, facecolor=face, edgecolor="0.15", lw=1.0, zorder=3)
        ax.add_patch(circ)
        ax.text(x, y, str(i + 1), color=txt, ha="center", va="center", fontsize=7, zorder=4)


def _plot_thrower_original(ax, stones_raw: np.ndarray, thrown_slot: int):
    xy_m = (stones_raw - BUTTON_RAW) * M_PER_RAW
    x, y = xy_m[thrown_slot]
    circ = plt.Circle(
        (float(x), float(y)),
        STONE_RADIUS_M,
        facecolor=THROWER_COLOR,
        edgecolor="0.05",
        lw=1.1,
        zorder=5,
    )
    ax.add_patch(circ)
    ax.text(x, y, str(thrown_slot + 1), color="black", ha="center", va="center", fontsize=7, zorder=6)


def _predict_value(model, stones_raw: np.ndarray, cond: np.ndarray, device: torch.device) -> float:
    x = torch.from_numpy((stones_raw.reshape(1, -1) / POS_MAX).astype(np.float32)).to(device)
    c = torch.from_numpy(cond.reshape(1, 3)).to(device)
    with torch.no_grad():
        return float(model(x, c).detach().cpu().numpy().reshape(-1)[0])


def _candidate_heatmap(
    model,
    pre_stones_raw: np.ndarray,
    cond: np.ndarray,
    thrown_slot: int,
    device: torch.device,
    grid_n: int,
    extent_m: float,
    batch_size: int,
):
    xs_m = np.linspace(-extent_m, extent_m, grid_n, dtype=np.float32)
    ys_m = np.linspace(-extent_m, extent_m, grid_n, dtype=np.float32)
    xx_m, yy_m = np.meshgrid(xs_m, ys_m)
    points_raw = BUTTON_RAW + np.stack([xx_m.ravel(), yy_m.ravel()], axis=1) / M_PER_RAW

    pre_value = _predict_value(model, pre_stones_raw, cond, device)
    boards = np.repeat(pre_stones_raw.reshape(1, NUM_STONES, 2), len(points_raw), axis=0)
    boards[:, thrown_slot, :] = points_raw
    x = torch.from_numpy((boards.reshape(len(points_raw), -1) / POS_MAX).astype(np.float32))
    c = torch.from_numpy(np.repeat(cond.reshape(1, 3), len(points_raw), axis=0))

    preds = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = x[start : start + batch_size].to(device)
            cb = c[start : start + batch_size].to(device)
            preds.append(model(xb, cb).detach().cpu().numpy().reshape(-1))
    post_values = np.concatenate(preds).reshape(grid_n, grid_n)
    return xs_m, ys_m, post_values - pre_value, pre_value


def _synthetic_state(rng: np.random.Generator, state_idx: int):
    stones = np.tile(END_SENTINEL_RAW, (NUM_STONES, 1)).astype(np.float32)
    n_live = int(rng.integers(3, 9))
    slots = rng.choice(NUM_STONES, size=n_live, replace=False)
    placed_m: list[np.ndarray] = []
    for slot in slots:
        for _ in range(200):
            radius = float(rng.uniform(0.15, 1.85))
            theta = float(rng.uniform(0.0, 2.0 * np.pi))
            pos_m = np.array([radius * np.cos(theta), radius * np.sin(theta)], dtype=np.float32)
            if all(np.linalg.norm(pos_m - p) > 2.1 * STONE_RADIUS_M for p in placed_m):
                stones[slot] = BUTTON_RAW + pos_m / M_PER_RAW
                placed_m.append(pos_m)
                break

    block = int(rng.integers(0, 2))
    block_start = 6 if block else 0
    candidates = [i for i in range(block_start, block_start + 6) if not _in_play(stones[[i]])[0]]
    thrown_slot = candidates[0] if candidates else block_start
    stones[thrown_slot] = END_SENTINEL_RAW

    cond = np.array(
        [float(rng.uniform(0.05, 0.65)), float(rng.integers(0, 2)), float(block)],
        dtype=np.float32,
    )
    label = f"synthetic_{state_idx:02d}"
    team = _stone_team_name(thrown_slot)
    return {
        "label": label,
        "title": f"Synthetic state {state_idx} | thrower: {team} stone {thrown_slot + 1}",
        "pre_stones": stones,
        "cond": cond,
        "slot": int(thrown_slot),
    }


def _real_case(ds: ValueDataset, idx: int, slot: int):
    row = ds.df.iloc[idx]
    prev_row = _previous_row(ds.df, idx)
    if prev_row is None:
        return None
    team = _stone_team_name(slot)
    return {
        "label": (
            f"early_comp{int(row['CompetitionID'])}_game{int(row['GameID'])}_"
            f"end{int(row['EndID'])}_shot{int(row['ShotID'])}"
        ),
        "title": (
            f"Early test state | comp {int(row['CompetitionID'])} game {int(row['GameID'])} "
            f"end {int(row['EndID'])} shot {int(row['ShotID'])} | thrower: {team} stone {slot + 1}"
        ),
        "pre_stones": _row_positions_raw(prev_row),
        "cond": _row_condition(row),
        "slot": int(slot),
    }


def _select_real_cases(ds: ValueDataset, test_idx: np.ndarray, rng: np.random.Generator, n: int, early_only: bool):
    rows = []
    seen = set()
    shuffled = np.asarray(test_idx, dtype=np.int64).copy()
    rng.shuffle(shuffled)
    for idx in shuffled:
        row = ds.df.iloc[int(idx)]
        key = (
            int(row["CompetitionID"]),
            int(row["GameID"]),
            int(row["EndID"]),
            int(row["ShotID"]),
        )
        if key in seen:
            continue
        if early_only and float(row["shot_norm"]) > 0.45:
            continue
        slot = _find_thrown_slot(ds.df, int(idx))
        if slot is None:
            continue
        case = _real_case(ds, int(idx), slot)
        if case is not None:
            rows.append(case)
            seen.add(key)
        if len(rows) >= n:
            break
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", type=int, default=23240026)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--grid", type=int, default=71)
    ap.add_argument("--extent-m", type=float, default=2.2)
    ap.add_argument("--seed", type=int, default=20260509)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out-dir", default=str(ROOT / "figures" / "value_heatmaps"))
    ap.add_argument(
        "--case-mode",
        choices=["real", "mixed_extra"],
        default="real",
        help="real: held-out test states; mixed_extra: early real states plus synthetic states",
    )
    args = ap.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")

    ds = ValueDataset(
        str(ROOT / "2026" / "Stones.csv"),
        str(ROOT / "2026" / "Ends.csv"),
        augment_positions=False,
        augment_flip=False,
    )
    _, _, test_idx, _ = make_holdout_split(ds.df, args.holdout, 0.10, 123)
    rng = np.random.default_rng(args.seed)
    test_idx = np.asarray(test_idx, dtype=np.int64)
    rng.shuffle(test_idx)

    ckpt = ROOT / "holdouts" / str(args.holdout) / "model_graphtf" / "model.pt"
    model = _load_model(ckpt, device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.case_mode == "mixed_extra":
        cases = _select_real_cases(ds, test_idx, rng, min(3, args.n), early_only=True)
        while len(cases) < args.n:
            cases.append(_synthetic_state(rng, len(cases) + 1))
    else:
        cases = _select_real_cases(ds, test_idx, rng, args.n, early_only=False)
    if len(cases) < args.n:
        raise RuntimeError(f"Only found {len(cases)} plottable states")

    for k, case in enumerate(cases, start=1):
        pre_stones_raw = case["pre_stones"]
        slot = int(case["slot"])
        xs_m, ys_m, value_delta, pre_value = _candidate_heatmap(
            model, pre_stones_raw, case["cond"], slot, device, args.grid, args.extent_m, 4096
        )

        fig, ax = plt.subplots(figsize=(6.2, 6.8), dpi=180)
        lim = float(np.nanmax(np.abs(value_delta)))
        if not np.isfinite(lim) or lim <= 1e-6:
            lim = 1.0
        im = ax.imshow(
            value_delta,
            origin="lower",
            extent=[xs_m.min(), xs_m.max(), ys_m.min(), ys_m.max()],
            cmap="coolwarm",
            vmin=-lim,
            vmax=lim,
            alpha=0.88,
            aspect="equal",
        )
        _draw_house(ax)
        _plot_stones(ax, pre_stones_raw, thrown_slot=-1)
        ax.set_xlim(-args.extent_m, args.extent_m)
        ax.set_ylim(-args.extent_m, args.extent_m)
        ax.set_xlabel("lateral from button (m)")
        ax.set_ylabel("along-sheet from button (m)")
        ax.set_title(f"{case['title']} | Vpre={pre_value:+.2f}", fontsize=9)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("predicted value change: V(post) - V(pre)")
        fig.tight_layout()
        out_path = out_dir / f"value_heatmap_{k:02d}_{case['label']}.png"
        fig.savefig(out_path)
        plt.close(fig)
        print(out_path)


if __name__ == "__main__":
    main()
