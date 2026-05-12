#!/usr/bin/env python3
"""Render value heatmaps over realistic visible sheet bounds."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

FIXED_ROOT = Path("/mnt/data/curling2/csas_fixed")
sys.path.insert(0, str(FIXED_ROOT))

import make_value_heatmaps as hv  # noqa: E402
from dataset import ValueDataset  # noqa: E402
from train_holdout_models_cond3 import make_holdout_split  # noqa: E402
from preplaced_value_data import canonical_preplacement_cases  # noqa: E402


def _plot_graph_source_points(ax, include_release: bool, include_takeout: bool) -> None:
    """Overlay graph landmark source points in sheet-meter coordinates."""
    if include_release:
        release_raw = np.array(
            [
                [350.0, 2900.0],
                [750.0, 2900.0],
                [1150.0, 2900.0],
            ],
            dtype=np.float32,
        )
        release_m = (release_raw - hv.BUTTON_RAW) * hv.M_PER_RAW
        ax.scatter(
            release_m[:, 0],
            release_m[:, 1],
            marker="^",
            s=42,
            c="#2f6f9f",
            edgecolors="white",
            linewidths=0.7,
            zorder=5,
            label="release source nodes",
        )
    if include_takeout:
        stone_radius_norm = 0.012
        button_x = 750.0 / hv.POS_MAX
        button_y = 800.0 / hv.POS_MAX
        house_radius = 600.0 / hv.POS_MAX
        takeout_y_norm = button_y - house_radius - 4.0 * stone_radius_norm
        offsets_raw = np.array([-600.0, -300.0, 0.0, 300.0, 600.0], dtype=np.float32)
        takeout_raw = np.stack(
            [
                np.full_like(offsets_raw, button_x * hv.POS_MAX) + offsets_raw,
                np.full_like(offsets_raw, takeout_y_norm * hv.POS_MAX),
            ],
            axis=1,
        )
        takeout_m = (takeout_raw - hv.BUTTON_RAW) * hv.M_PER_RAW
        ax.scatter(
            takeout_m[:, 0],
            takeout_m[:, 1],
            marker="X",
            s=54,
            c="#f08a24",
            edgecolors="black",
            linewidths=0.6,
            zorder=5,
            label="takeoutability source nodes",
        )


def candidate_heatmap_sheet(model, pre_stones_raw, cond, thrown_slot, device, nx, ny, x_min, x_max, y_min, y_max, batch_size):
    xs_m = np.linspace(x_min, x_max, nx, dtype=np.float32)
    ys_m = np.linspace(y_min, y_max, ny, dtype=np.float32)
    xx_m, yy_m = np.meshgrid(xs_m, ys_m)
    points_raw = hv.BUTTON_RAW + np.stack([xx_m.ravel(), yy_m.ravel()], axis=1) / hv.M_PER_RAW

    pre_value = hv._predict_value(model, pre_stones_raw, cond, device)
    boards = np.repeat(pre_stones_raw.reshape(1, hv.NUM_STONES, 2), len(points_raw), axis=0)
    boards[:, thrown_slot, :] = points_raw
    x = torch.from_numpy((boards.reshape(len(points_raw), -1) / hv.POS_MAX).astype(np.float32))
    c = torch.from_numpy(np.repeat(cond.reshape(1, 3), len(points_raw), axis=0))

    preds = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            out = model(x[start:start + batch_size].to(device), c[start:start + batch_size].to(device))
            if isinstance(out, tuple):
                out = out[0]
            preds.append(out.detach().cpu().numpy().reshape(-1))
    return xs_m, ys_m, np.concatenate(preds).reshape(ny, nx) - pre_value, pre_value


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", type=int, default=0)
    ap.add_argument("--split", choices=["val", "test"], default="test")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--nx", type=int, default=95)
    ap.add_argument("--ny", type=int, default=177)
    ap.add_argument("--x-min", type=float, default=-2.375)
    ap.add_argument("--x-max", type=float, default=2.375)
    ap.add_argument("--y-min", type=float, default=-2.44)
    ap.add_argument("--y-max", type=float, default=6.40)
    ap.add_argument("--seed", type=int, default=20260509)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--model-kind", choices=["settf_gaussian", "graphtf"], default="settf_gaussian")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--overlay-observed-throw", action="store_true")
    ap.add_argument("--show-release-source-points", action="store_true")
    ap.add_argument("--show-takeout-source-points", action="store_true")
    ap.add_argument("--case-mode", choices=["real", "preplaced"], default="real")
    ap.add_argument(
        "--color-lim",
        type=float,
        default=None,
        help="Use a fixed symmetric color limit. By default, one shared limit is computed across all rendered cases.",
    )
    ap.add_argument(
        "--color-percentile",
        type=float,
        default=99.0,
        help="Percentile of absolute deltas used for the shared color limit when --color-lim is omitted.",
    )
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)
    model = hv._load_model(Path(args.checkpoint), device, args.model_kind)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.case_mode == "preplaced":
        cases = []
        for c in canonical_preplacement_cases():
            team = "black" if int(c["thrower_block"]) == 1 else "white"
            cases.append(
                {
                    "label": f"preplaced_{c['mode']}_guard{c['guard_slot']}",
                    "title": f"Preplaced {c['mode']} | guard slot {c['guard_slot']} | first thrower: {team}",
                    "pre_stones": c["stones_raw"],
                    "observed_stones": None,
                    "cond": c["cond"],
                    "slot": int(c["thrown_slot"]) - 1,
                    "mode": c["mode"],
                    "guard_slot": int(c["guard_slot"]),
                    "thrower_block": int(c["thrower_block"]),
                }
            )
        cases = cases[: args.n] if args.n > 0 else cases
    else:
        ds = ValueDataset(str(FIXED_ROOT / "2026" / "Stones.csv"), str(FIXED_ROOT / "2026" / "Ends.csv"), augment_positions=False, augment_flip=False)
        _, val_idx, test_idx, _ = make_holdout_split(ds.df, args.holdout, 0.10, 123)
        split_idx = val_idx if args.split == "val" else test_idx
        cases = hv._select_real_cases(ds, np.asarray(split_idx, dtype=np.int64), rng, args.n, early_only=False)

    rendered = []
    for case in cases:
        xs_m, ys_m, value_delta, pre_value = candidate_heatmap_sheet(
            model,
            case["pre_stones"],
            case["cond"],
            int(case["slot"]),
            device,
            args.nx,
            args.ny,
            args.x_min,
            args.x_max,
            args.y_min,
            args.y_max,
            4096,
        )
        rendered.append((case, xs_m, ys_m, value_delta, pre_value))

    if args.color_lim is not None:
        shared_lim = float(args.color_lim)
    else:
        all_abs = np.concatenate([np.abs(vd).ravel() for _, _, _, vd, _ in rendered])
        shared_lim = float(np.nanpercentile(all_abs, args.color_percentile))
    if not np.isfinite(shared_lim) or shared_lim <= 1e-6:
        shared_lim = 1.0
    print(f"[color] shared symmetric limit: +/-{shared_lim:.4f}")

    state_rows = []
    for k, (case, xs_m, ys_m, value_delta, pre_value) in enumerate(rendered, start=1):
        state_rows.append(
            {
                "label": case["label"],
                "mode": case.get("mode", ""),
                "guard_slot": case.get("guard_slot", ""),
                "thrower_block": case.get("thrower_block", ""),
                "state_value": float(pre_value),
            }
        )
        fig, ax = plt.subplots(figsize=(5.4, 9.4), dpi=180)
        im = ax.imshow(
            value_delta,
            origin="lower",
            extent=[xs_m.min(), xs_m.max(), ys_m.min(), ys_m.max()],
            cmap="coolwarm",
            vmin=-shared_lim,
            vmax=shared_lim,
            alpha=0.88,
            aspect="equal",
        )
        hv._draw_house(ax)
        hv._plot_stones(ax, case["pre_stones"], thrown_slot=-1)
        if args.overlay_observed_throw:
            hv._plot_observed_throw(ax, case.get("observed_stones"), int(case["slot"]))
        if args.show_release_source_points or args.show_takeout_source_points:
            _plot_graph_source_points(
                ax,
                include_release=args.show_release_source_points,
                include_takeout=args.show_takeout_source_points,
            )
            ax.legend(loc="upper right", fontsize=6, framealpha=0.75)
        ax.set_xlim(args.x_min, args.x_max)
        ax.set_ylim(args.y_min, args.y_max)
        ax.set_xlabel("lateral from button (m)")
        ax.set_ylabel("along-sheet from button (m)")
        ax.set_title(f"{case['title']} | Vpre={pre_value:+.2f}", fontsize=8)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("predicted value change: V(post) - V(pre)")
        fig.tight_layout()
        out = out_dir / f"value_heatmap_{k:02d}_{case['label']}.png"
        fig.savefig(out)
        plt.close(fig)
        print(out)
    pd.DataFrame(state_rows).to_csv(out_dir / "state_values.csv", index=False)
    print(out_dir / "state_values.csv")


if __name__ == "__main__":
    main()
