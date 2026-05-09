#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import pathlib
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import jax.numpy as jnp

THIS_DIR = pathlib.Path(__file__).resolve().parent
sys.path.append(str(THIS_DIR))
sys.path.append(str(THIS_DIR / "inverse"))

from render_bad_inverse_examples import _draw_state, _plot_house, _raw_state_from_stones_row  # type: ignore
from visualize import (  # type: ignore
    SHOT_KEY,
    assign_final_to_slots,
    choose_new_slot_id,
    compact_positions,
    compute_shot_norm_and_order,
    extract_state_from_row,
    infer_thrower_block,
    prepare_merged,
)
from curling_sim_jax import CurlingParams, simulate_from_params  # type: ignore
from sim_presets import CONTACT_MILD_SIM_KWARGS  # type: ignore

THROW_LOCATION = np.array([-6.401, 0.0], dtype=np.float32)
BUTTON_LOCATION = np.array([0.0, 0.0], dtype=np.float32)


def _pick_random_raw_state(stones_df: pd.DataFrame, rng: np.random.Generator) -> pd.Series:
    live_counts = []
    for _, row in stones_df.iterrows():
        state = _raw_state_from_stones_row(row)
        live_counts.append(int(np.isfinite(state[:, 0]).sum()))
    stones_df = stones_df.copy()
    stones_df["live_count"] = live_counts
    eligible = stones_df[stones_df["live_count"] >= 2].reset_index(drop=True)
    if eligible.empty:
        raise RuntimeError("No eligible raw states found.")
    return eligible.iloc[int(rng.integers(len(eligible)))]


def _pick_random_inverse_row(merged: pd.DataFrame, rng: np.random.Generator) -> pd.Series:
    mask = np.ones(len(merged), dtype=bool)
    if "solver_ok" in merged.columns:
        mask &= merged["solver_ok"].fillna(False).astype(bool).to_numpy()
    for col in ["est_speed", "est_angle", "est_spin", "est_y0", "hard_loss_refine"]:
        if col in merged.columns:
            mask &= np.isfinite(pd.to_numeric(merged[col], errors="coerce").to_numpy())
    eligible = merged.loc[mask].copy().reset_index(drop=True)
    if eligible.empty:
        raise RuntimeError("No eligible inverse rows found.")
    return eligible.iloc[int(rng.integers(len(eligible)))]


def _make_random_state_figure(row: pd.Series, out_path: pathlib.Path) -> None:
    state = _raw_state_from_stones_row(row)
    fig, ax = plt.subplots(figsize=(6.2, 6.8), dpi=180)
    _plot_house(ax)
    live_pts = state[~np.isnan(state).any(axis=1)]
    graph_pts = []
    graph_pts.extend(live_pts.tolist())
    graph_pts.append(BUTTON_LOCATION.tolist())
    graph_pts.append(THROW_LOCATION.tolist())
    graph_pts = np.asarray(graph_pts, dtype=np.float32)
    for i in range(len(graph_pts)):
        for j in range(i + 1, len(graph_pts)):
            p0 = graph_pts[i]
            p1 = graph_pts[j]
            ax.plot(
                [p0[1], p1[1]],
                [p0[0], p1[0]],
                color="0.45",
                linewidth=0.8,
                alpha=0.35,
                zorder=1,
            )
    _draw_state(ax, state, thrower_block=0, alpha=0.96, s=90.0)
    ax.scatter(
        [THROW_LOCATION[1]],
        [THROW_LOCATION[0]],
        s=80,
        marker="x",
        color="tab:red",
        linewidths=2.0,
        zorder=6,
    )
    shot_index = int(row.get("ShotIndex", 0)) if pd.notna(row.get("ShotIndex", np.nan)) else 0
    shots_in_end = int(row.get("ShotsInEnd", 0)) if pd.notna(row.get("ShotsInEnd", np.nan)) else 0
    stones_left = max(0, shots_in_end - shot_index - 1)
    team_order = int(round(float(row.get("team_order", 0.0)))) if pd.notna(row.get("team_order", np.nan)) else 0
    hammer_label = "black (7-12)" if team_order == 1 else "white (1-6)"
    ax.text(
        0.03,
        0.97,
        f"Stones left: {stones_left}\nHammer: {hammer_label}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.28", facecolor="white", edgecolor="0.7", alpha=0.92),
        zorder=10,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _make_consecutive_inverse_figure(row: pd.Series, out_path: pathlib.Path) -> None:
    prev_mat = extract_state_from_row(row, "prev")
    next_mat = extract_state_from_row(row, "next")
    prev_compact, prev_ids = compact_positions(prev_mat)
    _, next_ids = compact_positions(next_mat)

    obs_throw_slot_id = float(row.get("obs_throw_slot_id", np.nan))
    team_slot_block = float(row.get("team_slot_block", np.nan))
    thrower_block = infer_thrower_block(prev_ids, next_ids, obs_throw_slot_id, team_slot_block)
    new_id = choose_new_slot_id(prev_ids, next_ids, thrower_block, obs_throw_slot_id)

    params = np.array(
        [row["est_speed"], row["est_angle"], row["est_spin"], row["est_y0"]],
        dtype=np.float32,
    )
    curl_params = CurlingParams(
        dt=0.02,
        substeps=2,
        max_steps=900,
        k_penalty=2.5e4,
        **CONTACT_MILD_SIM_KWARGS,
    )
    traj_compact = np.asarray(
        simulate_from_params(
            curl_params,
            jnp.asarray(prev_compact, dtype=jnp.float32),
            jnp.asarray(params, dtype=jnp.float32),
            dynamic=True,
        )
    )
    traj_full = np.full((traj_compact.shape[0], 12, 2), np.nan, dtype=np.float32)
    for t in range(traj_compact.shape[0]):
        traj_full[t] = assign_final_to_slots(traj_compact[t], prev_ids, new_id)
    thrown_traj = traj_full[:, new_id - 1, :]
    thrown_mask = np.isfinite(thrown_traj[:, 0]) & np.isfinite(thrown_traj[:, 1])

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.8), dpi=180)
    for ax in axes:
        _plot_house(ax)

    _draw_state(axes[0], prev_mat, thrower_block=thrower_block, alpha=0.96, s=88.0)

    _draw_state(axes[1], next_mat, thrower_block=thrower_block, alpha=0.96, s=88.0)
    if thrown_mask.any():
        axes[1].plot(
            thrown_traj[thrown_mask, 1],
            thrown_traj[thrown_mask, 0],
            linestyle=":",
            linewidth=2.0,
            color="tab:red",
            alpha=0.9,
            zorder=5,
        )
    fig.tight_layout()
    overlay = fig.add_axes([0, 0, 1, 1], frameon=False)
    overlay.set_axis_off()
    overlay.annotate(
        "",
        xy=(0.545, 0.50),
        xytext=(0.455, 0.50),
        xycoords="axes fraction",
        textcoords="axes fraction",
        arrowprops=dict(arrowstyle="->", lw=3.2, color="0.25", mutation_scale=18),
        annotation_clip=False,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Make two example curling-state figures.")
    ap.add_argument("--stones-csv", default=str(THIS_DIR / "2026" / "Stones.csv"))
    ap.add_argument("--inverse-glob", default=str(THIS_DIR / "inverse_current" / "stones_with_estimates.chunk*.csv"))
    ap.add_argument("--out-dir", default=str(THIS_DIR / "figures"))
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    stones_df = compute_shot_norm_and_order(pd.read_csv(args.stones_csv))
    merged = prepare_merged(args.stones_csv, args.inverse_glob)

    random_state_row = _pick_random_raw_state(stones_df, rng)
    inverse_row = _pick_random_inverse_row(merged, rng)

    out_dir = pathlib.Path(args.out_dir)
    _make_random_state_figure(random_state_row, out_dir / "random_curling_state.png")
    _make_consecutive_inverse_figure(inverse_row, out_dir / "consecutive_states_inverse_solution.png")


if __name__ == "__main__":
    main()
