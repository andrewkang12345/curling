#!/usr/bin/env python3

from __future__ import annotations

import argparse
import pathlib
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SHOT_KEY = ["CompetitionID", "SessionID", "GameID", "EndID", "ShotID"]
MIN_X = -6.6
MAX_X = 2.4
MIN_Y = -2.286
MAX_Y = 2.286


def _plot_house(ax):
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(MIN_Y, MAX_Y)
    ax.set_ylim(MIN_X, MAX_X)
    ax.set_xlabel("lateral (m)")
    ax.set_ylabel("along-sheet (m)")
    for r in [1.8288, 1.2192, 0.6096, 0.1524]:
        ax.add_patch(plt.Circle((0.0, 0.0), r, fill=False, linewidth=1.0, color="0.35"))
    ax.axhline(0.0, linewidth=0.6, alpha=0.25, color="0.2")
    ax.axvline(0.0, linewidth=0.6, alpha=0.25, color="0.2")


def _slot_colors(slot_ids: np.ndarray, thrower_block: int):
    if int(thrower_block) == 0:
        thr = slot_ids <= 6
    else:
        thr = slot_ids >= 7
    face = np.where(thr, "white", "black")
    edge = np.full(slot_ids.shape, "black", dtype=object)
    return face, edge


def _draw_state(ax, state_12x2: np.ndarray, thrower_block: int, alpha: float = 1.0, marker: str = "o", s: float = 80.0):
    mask = ~np.isnan(state_12x2).any(axis=1)
    if not mask.any():
        return
    slot_ids = np.nonzero(mask)[0] + 1
    pts = state_12x2[mask]
    face, edge = _slot_colors(slot_ids, thrower_block)
    for i, (slot_id, (x_m, y_m)) in enumerate(zip(slot_ids, pts)):
        ax.scatter(
            y_m,
            x_m,
            s=s,
            marker=marker,
            facecolors=face[i] if marker != "x" else "none",
            edgecolors=("tab:red" if marker == "x" else edge[i]),
            color=("tab:red" if marker == "x" else None),
            linewidths=1.4,
            alpha=alpha,
            zorder=4,
        )
        text_color = "white" if face[i] == "black" else "black"
        ax.text(y_m, x_m, str(int(slot_id)), ha="center", va="center", fontsize=7, color=text_color, alpha=min(1.0, alpha + 0.1), zorder=6)


def _extract_state_from_row(row: pd.Series, prefix: str) -> np.ndarray:
    mat = np.full((12, 2), np.nan, dtype=np.float32)
    for i in range(1, 13):
        x = row.get(f"{prefix}_stone_{i}_x_m", np.nan)
        y = row.get(f"{prefix}_stone_{i}_y_m", np.nan)
        if pd.isna(x) or pd.isna(y):
            continue
        x = float(x)
        y = float(y)
        if abs(x) >= 40 or abs(y) >= 40:
            continue
        mat[i - 1, 0] = x
        mat[i - 1, 1] = y
    return mat


def _slugify(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(text)).strip("_")


def _render_pair(row: pd.Series, out_path: pathlib.Path, after_row: pd.Series | None = None) -> None:
    prev_mat = _extract_state_from_row(row, "prev")
    next_mat = _extract_state_from_row(row, "next")
    thrower_block = int(row.get("team_slot_block", 0)) if np.isfinite(row.get("team_slot_block", np.nan)) else 0

    after_mat = None
    if after_row is not None:
        after_mat = _extract_state_from_row(after_row, "next")

    n_panels = 3 if after_mat is not None else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(5.0 * n_panels, 5.5), dpi=160)
    for ax in axes:
        _plot_house(ax)

    axes[0].set_title(f"Prev state (before shot)\nN={int(row.get('prev_N', -1))}")
    _draw_state(axes[0], prev_mat, thrower_block, alpha=0.96)

    axes[1].set_title(f"Next state (after shot)\nin-bounds={int(row.get('next_in_bounds_N', -1))} total={int(row.get('next_total_N', -1))}")
    _draw_state(axes[1], next_mat, thrower_block, alpha=0.96)

    if after_mat is not None:
        after_shot_id = int(after_row["ShotID"])
        after_n_ib = int(after_row.get("next_in_bounds_N", -1))
        after_n_tot = int(after_row.get("next_total_N", -1))
        axes[2].set_title(f"After next shot (shot {after_shot_id})\nin-bounds={after_n_ib} total={after_n_tot}")
        _draw_state(axes[2], after_mat, thrower_block, alpha=0.96)

    comp, sess, game, end, shot = [int(row[k]) for k in SHOT_KEY]
    fig.suptitle(
        f"Shot {comp},{sess},{game},{end},{shot} | hard_loss={float(row['hard_loss_refine']):.3f}",
        fontsize=12,
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Render excluded bad rows as prev-vs-next PNGs.")
    ap.add_argument("--input-csv", required=True)
    ap.add_argument("--out-dir", default="/mnt/data/curling2/csas_fixed/bad_inverse_viz/excluded_empty_next")
    ap.add_argument("--threshold", type=float, default=0.1)
    ap.add_argument("--all-bad", action="store_true",
                    help="Only filter on threshold (skip next_in_bounds_N==0 and prev_N!=0 conditions).")
    ap.add_argument("--lookup-csv", type=str, default=None,
                    help="Full dataset CSV to look up the following shot in each end. "
                         "If provided, a third panel shows the state after the next shot.")
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input_csv, low_memory=False)
    if args.all_bad:
        mask = df["hard_loss_refine"] > args.threshold
    else:
        mask = (
            (df["hard_loss_refine"] > args.threshold)
            & (df["next_in_bounds_N"] == 0)
            & (df["prev_N"] != 0)
        )
    sub = df.loc[mask].copy().sort_values(["hard_loss_refine"] + SHOT_KEY, ascending=[False, True, True, True, True, True])

    lookup_df = None
    if args.lookup_csv:
        lookup_df = pd.read_csv(args.lookup_csv, low_memory=False)
        for k in SHOT_KEY:
            lookup_df[k] = lookup_df[k].astype(int)

    def _find_following_shot(comp, sess, game, end, shot):
        if lookup_df is None:
            return None
        end_mask = (
            (lookup_df["CompetitionID"] == comp)
            & (lookup_df["SessionID"] == sess)
            & (lookup_df["GameID"] == game)
            & (lookup_df["EndID"] == end)
            & (lookup_df["ShotID"] > shot)
        )
        candidates = lookup_df.loc[end_mask]
        if candidates.empty:
            return None
        return candidates.sort_values("ShotID").iloc[0]

    manifest_rows = []
    for i, (_, row) in enumerate(sub.iterrows(), start=1):
        comp, sess, game, end, shot = [int(row[k]) for k in SHOT_KEY]
        stem = f"{i:03d}_{comp}_{sess}_{game}_{end}_{shot}_loss_{float(row['hard_loss_refine']):.3f}"
        png_name = f"{_slugify(stem)}.png"
        out_path = out_dir / png_name
        after_row = _find_following_shot(comp, sess, game, end, shot)
        _render_pair(row, out_path, after_row=after_row)
        manifest_rows.append(
            {
                "rank": i,
                "CompetitionID": comp,
                "SessionID": sess,
                "GameID": game,
                "EndID": end,
                "ShotID": shot,
                "prev_N": int(row.get("prev_N", -1)),
                "next_in_bounds_N": int(row.get("next_in_bounds_N", -1)),
                "next_total_N": int(row.get("next_total_N", -1)),
                "hard_loss_refine": float(row["hard_loss_refine"]),
                "has_after": after_row is not None,
                "png": png_name,
            }
        )
        print(f"[done] {out_path}")

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(out_dir / "manifest.csv", index=False)
    print(f"[done] wrote {len(manifest)} PNGs and manifest to {out_dir}")


if __name__ == "__main__":
    main()
