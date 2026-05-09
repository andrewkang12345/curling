#!/usr/bin/env python3
from __future__ import annotations

import math
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import pandas as pd
import torch

THIS_DIR = pathlib.Path(__file__).resolve().parent
VALUE_MODEL_DIR = THIS_DIR.parent
REPO_ROOT = VALUE_MODEL_DIR.parent
sys.path.insert(0, str(VALUE_MODEL_DIR))
sys.path.insert(0, str(THIS_DIR))

from dataset import ValueDataset, POS_MAX
import gnn_models


OUTPUT_DIR = THIS_DIR / "reachability_diagnostics_20260408"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HOUSE_RADIUS_RAW = 600.0
STONE_RADIUS_RAW = gnn_models.STONE_RADIUS_NORM * POS_MAX
BUTTON_RAW = np.array([gnn_models.BUTTON_X * POS_MAX, gnn_models.BUTTON_Y * POS_MAX], dtype=np.float32)
RELEASE_CENTER_NODE = gnn_models.NUM_STONES + 2


def _to_raw_xy(norm_xy: np.ndarray) -> np.ndarray:
    return norm_xy * POS_MAX


def _sample_board_indices(ds: ValueDataset, n_boards: int = 4) -> list[int]:
    df = ds.df.copy()
    board = np.stack([ds[i][0].numpy() for i in range(len(ds))], axis=0).reshape(len(ds), 12, 2)
    in_play = ((board.sum(axis=-1) > 0.001) & (board.max(axis=-1) < 0.999)).sum(axis=1)
    df["n_live"] = in_play
    df = df[(df["n_live"] >= 6) & (df["shot_norm"] >= 0.2) & (df["shot_norm"] <= 0.8)].copy()
    df = df.sort_values(["n_live", "shot_norm", "CompetitionID", "ShotID"], ascending=[False, True, True, True])
    picks = []
    seen = set()
    for idx, row in df.iterrows():
        key = int(row["CompetitionID"])
        if key in seen:
            continue
        picks.append(int(idx))
        seen.add(key)
        if len(picks) >= n_boards:
            break
    if len(picks) < n_boards:
        picks.extend(df.index.tolist()[: max(0, n_boards - len(picks))])
    return picks[:n_boards]


def _ray_first_hit_lambda(source_xy: np.ndarray, center_xy: np.ndarray, radius: float, theta: float) -> float | None:
    u = np.array([math.cos(theta), math.sin(theta)], dtype=np.float64)
    w = center_xy.astype(np.float64) - source_xy.astype(np.float64)
    b = float(np.dot(w, u))
    if b <= 0.0:
        return None
    d2 = float(np.dot(w, w) - b * b)
    r2 = float(radius * radius)
    if d2 > r2:
        return None
    return float(b - math.sqrt(max(r2 - d2, 0.0)))


def _mc_lane_survival(source_xy: np.ndarray, target_xy: np.ndarray, target_radius: float, blocker_xy: np.ndarray, n_samples: int, *, thrower_mask: bool) -> tuple[float, float]:
    vec = target_xy - source_xy
    if thrower_mask and vec[0] < 0.0:
        return 0.0, 0.0
    dist = float(np.linalg.norm(vec))
    if dist <= target_radius + 1e-6:
        return 0.0, 0.0
    alpha = math.atan2(float(vec[1]), float(vec[0]))
    delta = math.asin(min(target_radius / dist, 1.0 - 1e-6))
    total_width = 2.0 * delta
    thetas = np.random.default_rng(0).uniform(alpha - delta, alpha + delta, size=n_samples)

    feasible = 0
    blockers = np.asarray(blocker_xy, dtype=np.float64)
    for theta in thetas:
        lam_t = _ray_first_hit_lambda(source_xy, target_xy, target_radius, float(theta))
        if lam_t is None:
            continue
        ok = True
        for bxy in blockers:
            lam_b = _ray_first_hit_lambda(source_xy, bxy, 2.0 * STONE_RADIUS_RAW, float(theta))
            if lam_b is not None and lam_b <= lam_t + 1e-8:
                ok = False
                break
        feasible += int(ok)
    return feasible / max(1, n_samples), total_width


def _board_feature_rows(ds: ValueDataset, idx: int, n_mc: int = 128) -> tuple[pd.DataFrame, dict]:
    x, c, _ = ds[idx]
    board = x.view(12, 2).numpy()
    c_batch = c.view(1, -1)
    x_batch = x.view(1, -1)
    node_feats, node_coords, node_mask, _ = gnn_models.build_graph_batch_fast(x_batch, torch.device("cpu"))
    scorer = gnn_models._compute_unoccluded_angular_spans(node_coords, node_feats, node_mask)[0, :, 0].numpy()
    pairwise = gnn_models._compute_pairwise_unoccluded_angular_spans(node_coords, node_feats, node_mask)[0, :, :, 0].numpy()
    thrower_only = gnn_models._compute_pairwise_thrower_masked_spans(node_coords, node_feats, node_mask)[0, :, :, 0].numpy()
    button_region_only = gnn_models._compute_pairwise_button_region_spans(node_coords, node_feats, node_mask, c_batch[:, 2])[0, :, :, 0].numpy()
    full = gnn_models._compute_pairwise_thrower_masked_button_region_spans(node_coords, node_feats, node_mask, c_batch[:, 2])[0, :, :, 0].numpy()

    source_xy = _to_raw_xy(node_coords[0, RELEASE_CENTER_NODE].numpy())
    live_mask = ((board.sum(axis=-1) > 0.001) & (board.max(axis=-1) < 0.999))
    raw_stones = _to_raw_xy(board[live_mask])
    all_blockers = raw_stones.copy()

    rows = []
    for stone_idx in np.where(live_mask)[0]:
        target_xy = _to_raw_xy(board[stone_idx])
        blocker_xy = np.array([b for i, b in zip(np.where(live_mask)[0], all_blockers) if i != stone_idx], dtype=np.float64)
        mc_prob, total_width = _mc_lane_survival(
            source_xy,
            target_xy,
            2.0 * STONE_RADIUS_RAW,
            blocker_xy,
            n_mc,
            thrower_mask=True,
        )
        target_dist = float(np.linalg.norm(target_xy - source_xy))
        target_width = 2.0 * math.asin(min((2.0 * STONE_RADIUS_RAW) / max(target_dist, 1e-6), 1.0 - 1e-6))
        rows.append({
            "dataset_idx": idx,
            "target_type": "stone",
            "target_idx": int(stone_idx),
            "scorability_button_span": float(scorer[stone_idx]),
            "pairwise_span": float(pairwise[RELEASE_CENTER_NODE, stone_idx]),
            "thrower_mask_span": float(thrower_only[RELEASE_CENTER_NODE, stone_idx]),
            "button_region_span": float(button_region_only[RELEASE_CENTER_NODE, stone_idx]),
            "full_reach_span": float(full[RELEASE_CENTER_NODE, stone_idx]),
            "full_reach_ratio": float(full[RELEASE_CENTER_NODE, stone_idx] / max(target_width, 1e-8)),
            "mc_survival_prob": float(mc_prob),
            "target_interval_width": float(total_width),
        })

    button_xy = BUTTON_RAW
    shooter_team = float(c.numpy()[2])
    button_radius = float(
        gnn_models._compute_button_region_radii(node_coords, node_feats, node_mask, c_batch[:, 2])[0].item() * POS_MAX
    )
    mc_prob_button, total_width_button = _mc_lane_survival(
        source_xy,
        button_xy,
        button_radius,
        all_blockers,
        n_mc,
        thrower_mask=True,
    )
    button_dist = float(np.linalg.norm(button_xy - source_xy))
    button_width = 2.0 * math.asin(min(button_radius / max(button_dist, 1e-6), 1.0 - 1e-6))
    button_node_idx = gnn_models.NUM_STONES
    rows.append({
        "dataset_idx": idx,
        "target_type": "button_region",
        "target_idx": int(button_node_idx),
        "scorability_button_span": 0.0,
        "pairwise_span": float(pairwise[RELEASE_CENTER_NODE, button_node_idx]),
        "thrower_mask_span": float(thrower_only[RELEASE_CENTER_NODE, button_node_idx]),
        "button_region_span": float(button_region_only[RELEASE_CENTER_NODE, button_node_idx]),
        "full_reach_span": float(full[RELEASE_CENTER_NODE, button_node_idx]),
        "full_reach_ratio": float(full[RELEASE_CENTER_NODE, button_node_idx] / max(button_width, 1e-8)),
        "mc_survival_prob": float(mc_prob_button),
        "target_interval_width": float(total_width_button),
        "button_region_radius_raw": button_radius,
        "shooter_team": shooter_team,
    })

    meta = {
        "idx": idx,
        "comp": int(ds.df.iloc[idx]["CompetitionID"]),
        "shot_id": int(ds.df.iloc[idx]["ShotID"]),
        "shot_norm": float(ds.df.iloc[idx]["shot_norm"]),
        "team_order": float(ds.df.iloc[idx]["team_order"]),
        "stone_block": float(ds.df.iloc[idx]["stone_block"]),
        "board": board,
    }
    return pd.DataFrame(rows), meta


def _draw_board(ax, meta: dict, rows: pd.DataFrame) -> None:
    board = meta["board"]
    raw = _to_raw_xy(board)
    live_mask = ((board.sum(axis=-1) > 0.001) & (board.max(axis=-1) < 0.999))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(0, 1500)
    ax.set_ylim(3200, 500)
    ax.add_patch(Circle((BUTTON_RAW[0], BUTTON_RAW[1]), HOUSE_RADIUS_RAW, fill=False, color="0.75", lw=1.0))
    ax.add_patch(Circle((BUTTON_RAW[0], BUTTON_RAW[1]), 0.5 * HOUSE_RADIUS_RAW, fill=False, color="0.82", lw=0.8))
    ax.scatter([gnn_models.LANDMARKS[2, 0].item() * POS_MAX], [gnn_models.LANDMARKS[2, 1].item() * POS_MAX], c="tab:red", s=20, label="release")
    ax.scatter([BUTTON_RAW[0]], [BUTTON_RAW[1]], c="tab:blue", s=20, label="button")
    for stone_idx in np.where(live_mask)[0]:
        x, y = raw[stone_idx]
        team = 0 if stone_idx < 6 else 1
        color = "tab:orange" if team == 0 else "tab:green"
        ax.add_patch(Circle((x, y), STONE_RADIUS_RAW, facecolor=color, alpha=0.35, edgecolor=color, lw=1.0))
        row = rows[(rows["target_type"] == "stone") & (rows["target_idx"] == stone_idx)].iloc[0]
        label = f"{stone_idx+1}\nsc={row['scorability_button_span']:.2f}\nre={row['full_reach_span']:.2f}\nmc={row['mc_survival_prob']:.2f}"
        ax.text(x + 18, y - 18, label, fontsize=7, color="black")
    b_row = rows[rows["target_type"] == "button_region"].iloc[0]
    ax.text(
        BUTTON_RAW[0] + 25,
        BUTTON_RAW[1] - 25,
        f"button\nre={b_row['full_reach_span']:.2f}\nmc={b_row['mc_survival_prob']:.2f}",
        fontsize=7,
        color="tab:blue",
    )
    ax.set_title(
        f"idx={meta['idx']} comp={meta['comp']} shot={meta['shot_id']}\nshot_norm={meta['shot_norm']:.2f} block={meta['stone_block']:.0f}",
        fontsize=9,
    )


def main() -> None:
    ds = ValueDataset(
        str(REPO_ROOT / "2026" / "Stones.csv"),
        str(REPO_ROOT / "2026" / "Ends.csv"),
        augment_positions=False,
        augment_flip=False,
    )

    sample_indices = _sample_board_indices(ds, n_boards=4)
    all_rows = []
    metas = []
    for idx in sample_indices:
        rows, meta = _board_feature_rows(ds, idx)
        all_rows.append(rows)
        metas.append(meta)
    full_df = pd.concat(all_rows, ignore_index=True)
    full_df.to_csv(OUTPUT_DIR / "reachability_feature_samples.csv", index=False)

    # Sanity board plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 16), dpi=180)
    for ax, meta in zip(axes.flat, metas):
        rows = full_df[full_df["dataset_idx"] == meta["idx"]]
        _draw_board(ax, meta, rows)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "reachability_sanity_boards.png", dpi=220)
    plt.close(fig)

    # Monte Carlo comparison on a broader sample
    probe_indices = np.linspace(0, len(ds) - 1, 64, dtype=int)
    cmp_rows = []
    for idx in probe_indices:
        rows, _ = _board_feature_rows(ds, int(idx), n_mc=96)
        cmp_rows.append(rows)
    cmp_df = pd.concat(cmp_rows, ignore_index=True)
    cmp_df.to_csv(OUTPUT_DIR / "reachability_mc_comparison.csv", index=False)

    stone_df = cmp_df[cmp_df["target_type"] == "stone"].copy()
    corrs = {
        "pearson_full_ratio_vs_mc": float(stone_df["full_reach_ratio"].corr(stone_df["mc_survival_prob"], method="pearson")),
        "spearman_full_ratio_vs_mc": float(stone_df["full_reach_ratio"].corr(stone_df["mc_survival_prob"], method="spearman")),
        "pearson_thrower_mask_span_vs_mc": float(stone_df["thrower_mask_span"].corr(stone_df["mc_survival_prob"], method="pearson")),
        "pearson_pairwise_span_vs_mc": float(stone_df["pairwise_span"].corr(stone_df["mc_survival_prob"], method="pearson")),
    }
    pd.DataFrame([corrs]).to_csv(OUTPUT_DIR / "reachability_mc_correlations.csv", index=False)

    fig, ax = plt.subplots(figsize=(6.5, 6.0), dpi=180)
    ax.scatter(stone_df["full_reach_ratio"], stone_df["mc_survival_prob"], s=10, alpha=0.35, color="tab:blue")
    ax.set_xlabel("Reachability feature ratio (visible width / target interval)")
    ax.set_ylabel("MC no-contact lane survival probability")
    ax.set_title(
        f"release-center to live-stone targets\npearson={corrs['pearson_full_ratio_vs_mc']:.3f}, "
        f"spearman={corrs['spearman_full_ratio_vs_mc']:.3f}"
    )
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "reachability_mc_scatter.png", dpi=220)
    plt.close(fig)

    print(f"[done] wrote {OUTPUT_DIR}")
    print(pd.DataFrame([corrs]).to_string(index=False))


if __name__ == "__main__":
    main()
