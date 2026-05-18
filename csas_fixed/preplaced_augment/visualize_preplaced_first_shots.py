#!/usr/bin/env python3
"""
Visualize mixed-doubles preplaced stones before the first shot and the
observed post-first-shot state.

Inputs:
  preplaced_augment/first_shots_inverse.csv

Outputs:
  preplaced_augment/figures/preplaced_first_shots/*.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import pandas as pd


THIS_DIR = Path(__file__).resolve().parent
STONE_R = 0.145
HOUSE_RINGS = (0.1524, 0.6096, 1.2192, 1.8288)


def _stone_color(slot: int) -> tuple[str, str]:
    if 1 <= slot <= 6:
        return "black", "white"
    return "white", "black"


def _plot_house(ax: plt.Axes) -> None:
    for r in HOUSE_RINGS:
        ax.add_patch(Circle((0.0, 0.0), r, fill=False, lw=1.2, color="0.35", alpha=0.9))
    ax.axhline(0.0, color="0.85", lw=0.8, zorder=0)
    ax.axvline(0.0, color="0.85", lw=0.8, zorder=0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-2.35, 2.35)
    ax.set_ylim(-4.25, 2.35)
    ax.set_xlabel("lateral (m)")
    ax.set_ylabel("along-sheet (m)")
    ax.grid(False)


def _row_points(row: pd.Series, prefix: str) -> dict[int, tuple[float, float]]:
    pts: dict[int, tuple[float, float]] = {}
    for slot in range(1, 13):
        x = row.get(f"{prefix}_stone_{slot}_x_m", np.nan)
        y = row.get(f"{prefix}_stone_{slot}_y_m", np.nan)
        if pd.notna(x) and pd.notna(y):
            pts[slot] = (float(x), float(y))
    return pts


def _plot_stones(
    ax: plt.Axes,
    pts: dict[int, tuple[float, float]],
    *,
    highlight: set[int] | None = None,
) -> None:
    highlight = highlight or set()
    for slot, (along, lateral) in sorted(pts.items()):
        face, text = _stone_color(slot)
        edge = "tab:red" if slot in highlight else "0.15"
        lw = 2.2 if slot in highlight else 1.0
        ax.add_patch(
            Circle((lateral, along), STONE_R, facecolor=face, edgecolor=edge, lw=lw, alpha=0.96, zorder=3)
        )
        ax.text(
            lateral,
            along,
            str(slot),
            ha="center",
            va="center",
            fontsize=7,
            color=text,
            weight="bold",
            zorder=4,
        )


def _new_slots(prev: dict[int, tuple[float, float]], nxt: dict[int, tuple[float, float]]) -> set[int]:
    return set(nxt) - set(prev)


def _plot_example(row: pd.Series, out_path: Path) -> None:
    prev = _row_points(row, "prev")
    nxt = _row_points(row, "next")
    new = _new_slots(prev, nxt)

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 6.3), dpi=180, sharex=True, sharey=True)
    for ax in axes:
        _plot_house(ax)

    _plot_stones(axes[0], prev)
    _plot_stones(axes[1], nxt, highlight=new)

    meta = (
        f"comp={int(row['CompetitionID'])} session={int(row['SessionID'])} "
        f"game={int(row['GameID'])} end={int(row['EndID'])}"
    )
    mode = str(row.get("mode", "unknown"))
    guard_slot = row.get("guard_slot", np.nan)
    thrower_block = row.get("thrower_block", np.nan)
    loss = row.get("hard_loss_refine", np.nan)
    axes[0].set_title(f"pre-first shot: {mode}, guard slot {int(guard_slot) if pd.notna(guard_slot) else '?'}")
    axes[1].set_title(
        f"post-first shot: new {sorted(new) if new else 'not visible'}; "
        f"thrower team {'A' if thrower_block == 0 else 'B'}"
    )
    fig.suptitle(f"{meta}; inverse loss={loss:.4g}", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _select_examples(df: pd.DataFrame, n_per_mode: int) -> pd.DataFrame:
    df = df[df["solver_ok"].fillna(False)].copy()
    rows = []
    for mode in ["standard", "pp_left", "pp_right"]:
        d = df[df["mode"] == mode].copy()
        if d.empty:
            continue
        d["_n_next"] = [
            len(_row_points(r, "next")) for _, r in d.iterrows()
        ]
        d["_n_new"] = [
            len(_new_slots(_row_points(r, "prev"), _row_points(r, "next"))) for _, r in d.iterrows()
        ]
        visible = d[d["_n_new"] > 0].copy()
        if not visible.empty:
            d = visible
        # Mix clean examples and examples where the first shot adds/moves more stones.
        pick = pd.concat(
            [
                d.sort_values("hard_loss_refine", ascending=True).head(max(1, n_per_mode // 2)),
                d.sort_values(["_n_new", "_n_next", "hard_loss_refine"], ascending=[False, False, True]).head(n_per_mode),
            ],
            ignore_index=False,
        ).drop_duplicates(subset=["CompetitionID", "SessionID", "GameID", "EndID"])
        rows.append(pick.head(n_per_mode))
    if not rows:
        return df.head(0)
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(THIS_DIR / "first_shots_inverse.csv"))
    ap.add_argument("--out-dir", default=str(THIS_DIR / "figures" / "preplaced_first_shots"))
    ap.add_argument("--n-per-mode", type=int, default=4)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    examples = _select_examples(df, args.n_per_mode)

    manifest = []
    for idx, row in examples.reset_index(drop=True).iterrows():
        new_slots = sorted(_new_slots(_row_points(row, "prev"), _row_points(row, "next")))
        out = out_dir / f"preplaced_first_shot_{idx:02d}_{row['mode']}_c{int(row['CompetitionID'])}_g{int(row['GameID'])}_e{int(row['EndID'])}.png"
        _plot_example(row, out)
        manifest.append(
            {
                "path": str(out),
                "CompetitionID": row["CompetitionID"],
                "SessionID": row["SessionID"],
                "GameID": row["GameID"],
                "EndID": row["EndID"],
                "mode": row["mode"],
                "guard_slot": row["guard_slot"],
                "thrower_block": row["thrower_block"],
                "new_slots": " ".join(map(str, new_slots)),
                "hard_loss_refine": row["hard_loss_refine"],
            }
        )

    pd.DataFrame(manifest).to_csv(out_dir / "manifest.csv", index=False)
    print(f"wrote {len(manifest)} figures to {out_dir}")
    print(f"wrote manifest to {out_dir / 'manifest.csv'}")


if __name__ == "__main__":
    main()
