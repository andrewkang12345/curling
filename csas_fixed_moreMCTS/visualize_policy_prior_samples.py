#!/usr/bin/env python3
"""Visualize sampled throw priors from the policy model."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import pandas as pd
import torch

from common import FIXED_ROOT, NUM_STONES, POS_MAX, raw_to_compact_m, compact_m_to_raw, in_play_raw
from dataset import ValueDataset
from kr_uct_search import load_policy, _sample_actions, _simulate_candidates
from preplaced_value_data import canonical_preplacement_cases
from train_holdout_models_cond3 import make_holdout_split

BUTTON_RAW = np.array([750.0, 800.0], dtype=np.float32)
M_PER_RAW = 0.003048
STONE_R = 0.145
HOUSE_RINGS = (0.1524, 0.6096, 1.2192, 1.8288)


def _draw_house(ax):
    for r in HOUSE_RINGS:
        ax.add_patch(Circle((0.0, 0.0), r, fill=False, color="0.35", lw=1.1))
    ax.axhline(0.0, color="0.85", lw=0.8, zorder=0)
    ax.axvline(0.0, color="0.85", lw=0.8, zorder=0)


def _plot_stones(ax, stones_raw):
    live = in_play_raw(stones_raw)
    xy = (stones_raw.reshape(NUM_STONES, 2) - BUTTON_RAW[None]) * M_PER_RAW
    for i, (x, y) in enumerate(xy):
        if not live[i]:
            continue
        face = "black" if i >= 6 else "white"
        text = "white" if i >= 6 else "black"
        ax.add_patch(Circle((float(x), float(y)), STONE_R, facecolor=face, edgecolor="0.1", lw=1.0, zorder=4))
        ax.text(float(x), float(y), str(i + 1), ha="center", va="center", fontsize=7, color=text, zorder=5)


def _new_slot(raw_state: np.ndarray, cond: np.ndarray) -> int:
    live = in_play_raw(raw_state)
    block = int(round(float(cond[2])))
    start = 6 if block else 0
    for idx in range(start, start + 6):
        if not live[idx]:
            return idx
    for idx in range(NUM_STONES):
        if not live[idx]:
            return idx
    return NUM_STONES - 1


def _endpoint_m_from_states(states_norm, original_raw, cond):
    pts = []
    slot = _new_slot(original_raw, cond)
    for s in states_norm:
        raw = s.reshape(NUM_STONES, 2) * POS_MAX
        if not in_play_raw(raw[[slot]])[0]:
            continue
        pt = (raw[slot] - BUTTON_RAW) * M_PER_RAW
        if np.isfinite(pt).all() and abs(pt[0]) < 6 and abs(pt[1]) < 8:
            pts.append(pt)
    return np.asarray(pts, dtype=np.float32)


def _row_positions_raw(row: pd.Series) -> np.ndarray:
    vals = []
    for i in range(1, NUM_STONES + 1):
        vals.extend([float(row[f"stone_{i}_x"]), float(row[f"stone_{i}_y"])])
    return np.asarray(vals, dtype=np.float32).reshape(NUM_STONES, 2)


def _row_condition(row: pd.Series) -> np.ndarray:
    return np.asarray([row["shot_norm"], row["team_order"], row["stone_block"]], dtype=np.float32)


def _previous_row(df: pd.DataFrame, idx: int):
    row = df.iloc[idx]
    prev = df[
        (df["CompetitionID"] == row["CompetitionID"])
        & (df["SessionID"] == row["SessionID"])
        & (df["GameID"] == row["GameID"])
        & (df["EndID"] == row["EndID"])
        & (df["ShotID"] < row["ShotID"])
    ].sort_values("ShotID")
    return None if prev.empty else prev.iloc[-1]


def real_cases(n: int, seed: int):
    ds = ValueDataset(str(FIXED_ROOT / "2026" / "Stones.csv"), str(FIXED_ROOT / "2026" / "Ends.csv"), augment_positions=False, augment_flip=False)
    _, val_idx, _, _ = make_holdout_split(ds.df, 0, 0.10, 123)
    rng = np.random.default_rng(seed)
    idxs = np.asarray(val_idx, dtype=np.int64)
    rng.shuffle(idxs)
    out = []
    for idx in idxs:
        row = ds.df.iloc[int(idx)]
        if float(row["shot_norm"]) > 0.35:
            continue
        prev = _previous_row(ds.df, int(idx))
        if prev is None:
            continue
        out.append(
            {
                "label": f"real_comp{int(row['CompetitionID'])}_game{int(row['GameID'])}_end{int(row['EndID'])}_shot{int(row['ShotID'])}",
                "title": f"real early state: comp {int(row['CompetitionID'])}, game {int(row['GameID'])}, end {int(row['EndID'])}, shot {int(row['ShotID'])}",
                "stones_raw": _row_positions_raw(prev),
                "cond": _row_condition(row),
            }
        )
        if len(out) >= n:
            break
    return out


def preplaced_cases():
    out = []
    for c in canonical_preplacement_cases():
        out.append(
            {
                "label": f"preplaced_{c['mode']}_guard{c['guard_slot']}",
                "title": f"preplaced {c['mode']} guard slot {c['guard_slot']}",
                "stones_raw": c["stones_raw"],
                "cond": c["cond"],
            }
        )
    return out


def plot_case(case, policy, mean_t, std_t, device, out_path, n_samples, temperature, std_scale, global_frac):
    x = (case["stones_raw"].reshape(-1) / POS_MAX).astype(np.float32)
    c = case["cond"].astype(np.float32)
    actions = _sample_actions(policy, mean_t, std_t, x, c, n_samples, device, temperature, std_scale, global_frac)
    posts = _simulate_candidates(x, c, actions)
    endpoints = _endpoint_m_from_states(posts, case["stones_raw"], c)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.8), dpi=170)
    ax = axes[0]
    _draw_house(ax)
    _plot_stones(ax, case["stones_raw"])
    if len(endpoints):
        ax.scatter(endpoints[:, 0], endpoints[:, 1], s=8, alpha=0.28, color="tab:blue", edgecolors="none")
    ax.set_xlim(-2.375, 2.375)
    ax.set_ylim(-6.40, 6.40)
    ax.set_aspect("equal")
    ax.set_xlabel("lateral from button (m)")
    ax.set_ylabel("along-sheet from button (m)")
    ax.set_title("simulated endpoints")

    labels = ["speed", "angle", "spin", "y0"]
    for j, lab in enumerate(labels):
        axes[1].hist(actions[:, j], bins=40, alpha=0.55, label=lab)
    axes[1].legend(fontsize=8)
    axes[1].set_title("sampled action parameters")
    axes[1].set_xlabel("raw action value")
    axes[1].set_ylabel("count")

    fig.suptitle(case["title"], fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="checkpoints/policy_prior_preplaced_h0/model.pt")
    ap.add_argument("--out-dir", default="figures/policy_prior_preplaced_samples")
    ap.add_argument("--n-samples", type=int, default=512)
    ap.add_argument("--n-real", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=1.35)
    ap.add_argument("--std-scale", type=float, default=1.6)
    ap.add_argument("--global-frac", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=20260510)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    policy, mean_t, std_t = load_policy(args.policy, device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cases = preplaced_cases() + real_cases(args.n_real, args.seed)
    rows = []
    for i, case in enumerate(cases, start=1):
        out = out_dir / f"policy_prior_{i:02d}_{case['label']}.png"
        plot_case(case, policy, mean_t, std_t, device, out, args.n_samples, args.temperature, args.std_scale, args.global_frac)
        rows.append({"path": str(out), "label": case["label"], "title": case["title"]})
        print(out)
    pd.DataFrame(rows).to_csv(out_dir / "manifest.csv", index=False)
    print(out_dir / "manifest.csv")


if __name__ == "__main__":
    main()
