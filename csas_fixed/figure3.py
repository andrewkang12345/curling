#!/usr/bin/env python3
"""
figure3.py

Figure 3: Two exemplar shots from the dataset:
  - most value gained
  - most value lost

Per exemplar (column):
  - TOP:  previous state (PREV stones)
  - BOT:  next state     (NEXT stones)

No trajectories, no arrows, no explicit "next markers" beyond plotting the next state itself.

Stone styling:
  - Throwing team stones: white fill + black outline
  - Opponent stones:      black fill + black outline

IMPORTANT (sign convention):
  shot_scores dv_obs is from the THROWER perspective (per-row).
  This figure keeps that perspective so sign and colors stay aligned:
    - dv_obs > 0 favors WHITE (thrower)
    - dv_obs < 0 favors BLACK (opponent)

Inputs:
  - shot_scores_old.csv / .parquet: must include dv_obs, team_order, TeamID, and SHOT_KEY
  - inverseDataset stones_with_estimates.chunk*.csv: provides prev/next stone positions in meters

Usage:
  python figure3.py \
    --shot-scores /mnt/data/curling2/csas/shot_scores_old.csv \
    --inverse-glob /mnt/data/curling2/csas/inverseDataset/stones_with_estimates.chunk*.csv \
    --out figures/figure3.png
"""

from __future__ import annotations

import argparse
import glob
import pathlib
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


SHOT_KEY = ["CompetitionID", "SessionID", "GameID", "EndID", "ShotID"]

# House geometry (meters), centered at button (0,0)
X_BUTTON = 0.0
Y_BUTTON = 0.0
R_8 = 1.2192
R_4 = 0.6096
R_BUTTON = 0.1524

# Plot window: reduce unnecessary left whitespace while still indicating left->right throw direction
# (You can tighten further if desired.)
X_MIN = -6.0
X_MAX = 2.4
Y_LIM = 2.6


# ----------------------------
# Robust loaders
# ----------------------------
def _looks_like_parquet(p: pathlib.Path) -> bool:
    try:
        with p.open("rb") as f:
            head = f.read(4)
        return head == b"PAR1"
    except Exception:
        return False


def load_table(path: str) -> pd.DataFrame:
    p = pathlib.Path(path)
    suf = p.suffix.lower()
    if suf == ".csv":
        return pd.read_csv(p)
    if suf == ".parquet":
        if not _looks_like_parquet(p):
            return pd.read_csv(p)
        try:
            return pd.read_parquet(p)
        except Exception:
            return pd.read_csv(p)
    return pd.read_csv(p)


def load_inverse_chunks(glob_pattern: str) -> pd.DataFrame:
    paths = sorted(glob.glob(glob_pattern))
    if not paths:
        raise FileNotFoundError(f"No inverse files matched: {glob_pattern}")
    frames = [pd.read_csv(p) for p in paths]
    return pd.concat(frames, ignore_index=True)


# ----------------------------
# Stone helpers
# ----------------------------
def extract_stones_xy(row: pd.Series, prefix: str, n: int = 12) -> np.ndarray:
    out = np.full((n, 2), np.nan, dtype=np.float32)
    for i in range(1, n + 1):
        x = row.get(f"{prefix}_stone_{i}_x_m", np.nan)
        y = row.get(f"{prefix}_stone_{i}_y_m", np.nan)
        if pd.notna(x) and pd.notna(y):
            out[i - 1, 0] = float(x)
            out[i - 1, 1] = float(y)
    return out


def present_mask(xy: np.ndarray) -> np.ndarray:
    return np.isfinite(xy).all(axis=1)


def infer_thrower_block(prev_xy: np.ndarray, next_xy: np.ndarray) -> int:
    """
    Infer thrower team slot block from the observed transition:
      - block 0 => slots 1..6
      - block 1 => slots 7..12
    """
    prev_m = present_mask(prev_xy)
    next_m = present_mask(next_xy)
    added_slots = np.where(next_m & (~prev_m))[0] + 1
    if added_slots.size == 1:
        return 0 if int(added_slots[0]) <= 6 else 1

    # Fallback: in normal ends, the thrower tends to have thrown no more stones than opponent pre-shot.
    prev_a = int(np.sum(prev_m[:6]))
    prev_b = int(np.sum(prev_m[6:]))
    return 0 if prev_a <= prev_b else 1


def thrower_slot_mask_from_block(thrower_block: int, n_slots: int = 12) -> np.ndarray:
    slot_ids = np.arange(1, n_slots + 1, dtype=int)
    if int(thrower_block) == 0:
        return slot_ids <= 6
    return slot_ids >= 7


# ----------------------------
# Plotting
# ----------------------------
def draw_house(ax: plt.Axes) -> None:
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    for r in [R_8, R_4, R_BUTTON]:
        ax.add_patch(plt.Circle((X_BUTTON, Y_BUTTON), r, fill=False, linewidth=1.25))

    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(-Y_LIM, Y_LIM)


def plot_state(ax: plt.Axes, xy: np.ndarray, is_thrower_slot: np.ndarray) -> None:
    draw_house(ax)
    m = present_mask(xy)
    if not m.any():
        return

    thr = m & is_thrower_slot
    opp = m & (~is_thrower_slot)

    # Opponent: black filled circles + black edge
    if opp.any():
        ax.scatter(
            xy[opp, 0],
            xy[opp, 1],
            s=90,
            marker="o",
            facecolors="black",
            edgecolors="black",
            linewidths=1.4,
            alpha=0.92,
            zorder=3,
        )

    # Thrower: white filled circles + black edge
    if thr.any():
        ax.scatter(
            xy[thr, 0],
            xy[thr, 1],
            s=90,
            marker="o",
            facecolors="white",
            edgecolors="black",
            linewidths=1.8,
            alpha=0.98,
            zorder=4,
        )


def build_info_text(score_row: pd.Series, dv_obs_thrower: float, team_shot_num: int) -> str:
    # Minimal but complete metadata per your request
    team_order = int(round(float(score_row.get("team_order", 0.0)))) if pd.notna(score_row.get("team_order", np.nan)) else 0
    hammer_txt = "Yes" if team_order == 1 else "No"

    sid = (
        f"C{int(score_row['CompetitionID'])}-S{int(score_row['SessionID'])}"
        f"-G{int(score_row['GameID'])}-E{int(score_row['EndID'])}-Shot{int(score_row['ShotID'])}"
    )

    return (
        f"{sid}\n"
        f"dv_obs(thrower)={dv_obs_thrower:+.3f}\n"
        f"Team shot # in end: {int(team_shot_num)}\n"
        f"Hammer: {hammer_txt}\n"
        f"Thrower=White, Opponent=Black\n"
        f"Throw direction: \u2192"
    )


# ----------------------------
# Main
# ----------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Figure 3: best/worst shots as PREV(top)/NEXT(bottom) states (standalone).")
    # ap.add_argument("--shot-scores", type=str, default="/mnt/data/curling2/csas/shot_scores_old.csv")
    ap.add_argument("--shot-scores", type=str, default="/mnt/data/curling2/csas/shot_scores_by_competition/shot_scores_comp_0.csv")
    ap.add_argument("--inverse-glob", type=str, default="/mnt/data/curling2/csas/inverseDataset/stones_with_estimates.chunk*.csv")
    ap.add_argument("--out", type=str, default="figure3.png")
    ap.add_argument("--only-solver-ok", action="store_true", help="Restrict to solver_ok==True (from inverseDataset).")
    args = ap.parse_args()

    # --- Load shot_scores ---
    ss = load_table(args.shot_scores)

    # Validate required columns
    needed_cols = set(SHOT_KEY + ["dv_obs", "TeamID", "team_order"])
    missing = [c for c in needed_cols if c not in ss.columns]
    if missing:
        raise SystemExit(f"[error] shot_scores missing required columns: {missing}")

    ss = ss.copy()
    ss["dv_obs"] = pd.to_numeric(ss["dv_obs"], errors="coerce")
    ss["TeamID"] = pd.to_numeric(ss["TeamID"], errors="coerce")
    ss["ShotID"] = pd.to_numeric(ss["ShotID"], errors="coerce")
    ss = ss.dropna(subset=["dv_obs"] + SHOT_KEY)

    if ss.empty:
        raise SystemExit("[error] shot_scores has no usable rows after cleaning.")

    # --- Compute "kth shot by that team in the end" (1-indexed) ---
    end_group = ["CompetitionID", "SessionID", "GameID", "EndID", "TeamID"]
    ss = ss.sort_values(SHOT_KEY).reset_index(drop=True)
    ss["team_shot_num_in_end"] = ss.groupby(end_group).cumcount() + 1

    # Rank best/worst by thrower-perspective dv.
    best_score = ss.loc[ss["dv_obs"].idxmax()].copy()
    worst_score = ss.loc[ss["dv_obs"].idxmin()].copy()

    # --- Load inverse chunks (prev/next positions) ---
    inv = load_inverse_chunks(args.inverse_glob)

    want_cols: List[str] = SHOT_KEY + ["solver_ok", "prev_N"]
    for pref in ["prev", "next"]:
        for i in range(1, 13):
            want_cols += [f"{pref}_stone_{i}_x_m", f"{pref}_stone_{i}_y_m"]
    want_cols = [c for c in want_cols if c in inv.columns]
    inv = inv[want_cols].copy()

    if args.only_solver_ok and "solver_ok" in inv.columns:  # typo-proofing: keep old name? (see below)
        inv = inv[inv["solver_ok"] == True].copy()  # noqa: E712

    # Correct flag name
    if args.only_solver_ok and "solver_ok" in inv.columns:
        inv = inv[inv["solver_ok"] == True].copy()  # noqa: E712

    def _fetch_one(key_row: pd.Series) -> pd.Series:
        m = np.ones(len(inv), dtype=bool)
        for c in SHOT_KEY:
            m &= (inv[c].astype("Int64") == int(key_row[c]))
        sub = inv[m]
        if sub.empty:
            raise SystemExit(f"[error] could not find inverseDataset row for key={key_row[SHOT_KEY].to_dict()}")
        return sub.iloc[0]

    best_inv = _fetch_one(best_score)
    worst_inv = _fetch_one(worst_score)

    # --- Determine thrower-vs-opponent slot mask for coloring ---
    def _thrower_slot_mask(inv_row: pd.Series) -> np.ndarray:
        prev_xy = extract_stones_xy(inv_row, "prev", n=12)
        next_xy = extract_stones_xy(inv_row, "next", n=12)
        thrower_block = infer_thrower_block(prev_xy, next_xy)
        return thrower_slot_mask_from_block(thrower_block, n_slots=12)

    best_thrower_slots = _thrower_slot_mask(best_inv)
    worst_thrower_slots = _thrower_slot_mask(worst_inv)

    # --- Extract states ---
    best_prev = extract_stones_xy(best_inv, "prev", n=12)
    best_next = extract_stones_xy(best_inv, "next", n=12)
    worst_prev = extract_stones_xy(worst_inv, "prev", n=12)
    worst_next = extract_stones_xy(worst_inv, "next", n=12)

    # --- Render ---
    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Tight layout: reduce excess whitespace between "buttons" without crowding text
    fig, axes = plt.subplots(
        2, 2,
        figsize=(12.2, 7.0),
        gridspec_kw=dict(wspace=0.06, hspace=0.10),
    )

    # Plot states (no titles)
    plot_state(axes[0, 0], best_prev, best_thrower_slots)
    plot_state(axes[1, 0], best_next, best_thrower_slots)
    plot_state(axes[0, 1], worst_prev, worst_thrower_slots)
    plot_state(axes[1, 1], worst_next, worst_thrower_slots)

    # Info boxes: place in TOP axes; avoid overlap by using consistent anchors and modest box size
    best_info = build_info_text(
        best_score,
        dv_obs_thrower=float(best_score["dv_obs"]),
        team_shot_num=int(best_score["team_shot_num_in_end"]),
    )
    worst_info = build_info_text(
        worst_score,
        dv_obs_thrower=float(worst_score["dv_obs"]),
        team_shot_num=int(worst_score["team_shot_num_in_end"]),
    )

    # Put info in top-left corner (inside axes). Use smaller font to avoid covering stones.
    axes[0, 0].text(
        0.02, 0.02, best_info,
        transform=axes[0, 0].transAxes,
        ha="left", va="bottom",
        fontsize=8.8,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.85", alpha=0.92),
        zorder=10,
    )
    axes[0, 1].text(
        0.02, 0.02, worst_info,
        transform=axes[0, 1].transAxes,
        ha="left", va="bottom",
        fontsize=8.8,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.85", alpha=0.92),
        zorder=10,
    )

    fig.savefig(out_path, dpi=240, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"[done] wrote {out_path}")


if __name__ == "__main__":
    main()
