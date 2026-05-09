#!/usr/bin/env python3
"""
Noise-version comparison pipeline.

Creates 3 versions of execution noise, runs MC scoring for each,
fits player rankings, and generates comparison visualizations.

V1: Bowling et al. — Student-t(ν=5), speed scale 9.5mm/s,
    weight-dependent aim noise σ_θ²(w) ∈ [1.16e-3, 3.65e-3]
V2: Data-fitted — Gaussian fitted to (x_inverse - x_best_local) deltas
V3: Expert guess — domain-knowledge priors for elite curlers

Usage:
    cd /mnt/data/curling2/csas_fixed
    python3 noise_versions/run_comparison.py [--num-samples 64] [--fit-samples 128]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from tqdm.auto import tqdm

# ---- Force JAX CPU to avoid GPU init issues ----
os.environ["JAX_PLATFORMS"] = "cpu"

# ---- path setup ----
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "inverse"))
sys.path.insert(0, str(ROOT / "valueModel"))

from score_shots_mc_seq import (
    SHOT_KEY,
    PARAM_COLS,
    NoiseSampler,
    SimFnCache,
    SolveBounds,
    CurlingParams,
    clip_to_bounds,
    compact_positions,
    state_to_fixed_slot_arrays,
    assign_finals_12slot_batch,
    infer_thrower_block,
    choose_new_slot_id,
    make_raw_defaults_for_state,
    positions_m_to_raw_matrix,
    positions_m_to_raw_matrix_batch,
    normalize_raw_matrix,
    normalize_raw_matrix_batch,
    load_value_model,
    prepare_dataframe_all,
    percentile_of_score,
    cvar,
    _separate_overlaps,
)
from sim_presets import contact_mild_params

import jax
import jax.numpy as jnp

OUT_DIR = ROOT / "noise_versions"

# =====================================================================
# Noise model definitions
# =====================================================================

class BowlingNoiseSampler:
    """
    V1: Bowling et al. fitted execution noise.
    - Weight (speed) noise: Student-t(ν=5), scale = 9.5 mm/s
    - Aim (angle) noise: Student-t(ν=5), scale linearly interpolated
      from 3.65e-3 rad (at w=0.5 m/s) to 1.16e-3 rad (at w=3.0 m/s)
    - Spin noise: Gaussian, σ = 0.08 rad/s (not in Bowling; reasonable)
    - Y0 noise: Gaussian, σ = 0.015 m (not in Bowling; reasonable)
    """
    NU = 5
    SPEED_SCALE = 0.0095      # m/s  (Student-t scale parameter)
    ANGLE_SCALE_LOW_SPEED = 3.65e-3   # at speed ~ 0.5 m/s (draw)
    ANGLE_SCALE_HIGH_SPEED = 1.16e-3  # at speed ~ 3.0 m/s (hard takeout)
    SPEED_RANGE = (0.5, 3.0)
    SPIN_STD = 0.08           # rad/s
    Y0_STD = 0.015            # m

    def draw(self, rng: np.random.Generator, center: np.ndarray, **_kw) -> np.ndarray:
        speed_center = float(center[0])
        # Student-t for speed
        d_speed = rng.standard_t(self.NU) * self.SPEED_SCALE
        # Angle scale interpolated by speed
        t = np.clip(
            (speed_center - self.SPEED_RANGE[0]) / (self.SPEED_RANGE[1] - self.SPEED_RANGE[0]),
            0.0, 1.0,
        )
        angle_scale = self.ANGLE_SCALE_LOW_SPEED + t * (self.ANGLE_SCALE_HIGH_SPEED - self.ANGLE_SCALE_LOW_SPEED)
        d_angle = rng.standard_t(self.NU) * angle_scale
        # Gaussian for spin and y0
        d_spin = rng.normal(0.0, self.SPIN_STD)
        d_y0 = rng.normal(0.0, self.Y0_STD)
        return center + np.array([d_speed, d_angle, d_spin, d_y0], dtype=np.float32)


class FittedGaussianSampler:
    """
    V2: Gaussian fitted to (x_inverse - x_best_local) deltas.
    Initialized with pre-fit mean & std; or fitted from data at runtime.
    """
    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)

    def draw(self, rng: np.random.Generator, center: np.ndarray, **_kw) -> np.ndarray:
        delta = rng.normal(self.mean, self.std).astype(np.float32)
        return center + delta


class ExpertGuessSampler:
    """V3: Expert-guess noise for elite curlers."""
    STD = np.array([0.020, 0.014, 0.15, 0.008], dtype=np.float32)

    def draw(self, rng: np.random.Generator, center: np.ndarray, **_kw) -> np.ndarray:
        delta = rng.normal(0.0, self.STD).astype(np.float32)
        return center + delta


# =====================================================================
# V2 fitting: local-best delta estimation
# =====================================================================

def fit_v2_noise(
    df: pd.DataFrame,
    model_fn,
    curl_params,
    bounds: SolveBounds,
    fit_samples: int = 128,
    search_std: np.ndarray = np.array([0.08, 0.03, 0.25, 0.05], dtype=np.float32),
    max_shots: int = 3000,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    For each shot, sample `fit_samples` local perturbations around the inverse
    solution, simulate + evaluate value. The best-value sample is treated as
    the "intended" shot. Collect deltas (x_inverse - x_best) across shots,
    then fit Gaussian (mean, std).

    Returns (mean_delta, std_delta) arrays of shape (4,).
    """
    rng = np.random.default_rng(seed)
    sim_cache = SimFnCache(curl_params)

    # Subsample for speed
    if len(df) > max_shots:
        df = df.sample(n=max_shots, random_state=seed).copy()
    print(f"[v2-fit] fitting noise from {len(df)} shots, {fit_samples} local samples each")

    deltas = []
    n_valid = 0
    n_skipped = 0

    for idx, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc="V2 fitting")):
        row_dict = row.to_dict()
        est_params = np.array([row_dict.get(c, np.nan) for c in PARAM_COLS], dtype=np.float32)
        if not np.all(np.isfinite(est_params)):
            n_skipped += 1
            continue

        # Build board state
        prev_mat = np.full((12, 2), np.nan, dtype=np.float32)
        next_mat = np.full((12, 2), np.nan, dtype=np.float32)
        for s in range(1, 13):
            px = row_dict.get(f"prev_stone_{s}_x_m", np.nan)
            py = row_dict.get(f"prev_stone_{s}_y_m", np.nan)
            prev_mat[s - 1] = [px, py]
            nx = row_dict.get(f"next_stone_{s}_x_m", np.nan)
            ny = row_dict.get(f"next_stone_{s}_y_m", np.nan)
            next_mat[s - 1] = [nx, ny]

        _, prev_ids_compact = compact_positions(prev_mat)
        _, next_ids = compact_positions(next_mat)
        prev_n = len(prev_ids_compact)
        if prev_n > 12:
            n_skipped += 1
            continue

        prev_slots, prev_slot_mask = state_to_fixed_slot_arrays(prev_mat)
        if int(np.sum(prev_slot_mask)) > 1:
            prev_slots[prev_slot_mask] = _separate_overlaps(prev_slots[prev_slot_mask])

        obs_throw_slot_id = float(row_dict.get("obs_throw_slot_id", np.nan))
        team_slot_block = float(row_dict.get("team_slot_block", np.nan))
        thrower_block = infer_thrower_block(
            prev_ids=prev_ids_compact, next_ids=next_ids,
            obs_throw_slot_id=obs_throw_slot_id, team_slot_block=team_slot_block,
        )
        new_id = choose_new_slot_id(
            prev_ids=prev_ids_compact, next_ids=next_ids,
            thrower_block=thrower_block, obs_throw_slot_id=obs_throw_slot_id,
        )

        shot_norm_next = float(row_dict.get("shot_norm_next", row_dict.get("shot_norm_prev", 0.0)))
        shot_index_next = float(row_dict.get("ShotIndex", 0.0))
        team_order = float(row_dict.get("team_order", 0.0))
        stone_block = float(thrower_block)
        c_next = np.array([shot_norm_next, team_order, stone_block], dtype=np.float32)
        next_defaults = make_raw_defaults_for_state(shot_index_next, team_order, thrower_block)

        # Compute v_prev (simple: evaluate prev board from thrower's perspective)
        shot_norm_prev = float(row_dict.get("shot_norm_prev", shot_norm_next))
        shot_index_prev = shot_index_next - 1.0
        c_prev = np.array([shot_norm_prev, team_order, stone_block], dtype=np.float32)
        prev_defaults = make_raw_defaults_for_state(shot_index_prev, team_order, thrower_block)
        prev_raw_norm = normalize_raw_matrix(positions_m_to_raw_matrix(prev_mat, raw_defaults=prev_defaults))
        v_prev = float(model_fn(prev_raw_norm, c_prev))

        # Sample local perturbations
        B = fit_samples
        x_batch = np.zeros((B, 4), dtype=np.float32)
        for b in range(B):
            delta = rng.normal(0.0, search_std).astype(np.float32)
            x_batch[b] = clip_to_bounds(est_params + delta, bounds)

        # Simulate batch
        prev_j = jnp.asarray(prev_slots, dtype=jnp.float32)
        x_j = jnp.asarray(x_batch, dtype=jnp.float32)
        sim_fn = sim_cache.get(12, B)
        finals = np.asarray(sim_fn(prev_j, x_j))

        full_final_batch = assign_finals_12slot_batch(finals, prev_slot_mask, new_id)
        final_raw_norm_batch = normalize_raw_matrix_batch(
            positions_m_to_raw_matrix_batch(full_final_batch, raw_defaults=next_defaults)
        )
        c_next_batch = np.broadcast_to(c_next.reshape(1, -1), (B, c_next.shape[0])).astype(np.float32, copy=False)
        v_sim_batch = np.asarray(model_fn(final_raw_norm_batch, c_next_batch), dtype=np.float32).reshape(-1)
        dv_samples = v_sim_batch - np.float32(v_prev)

        # Find best-value sample
        best_idx = int(np.argmax(dv_samples))
        x_best = x_batch[best_idx]
        delta = est_params - x_best  # actual - intended = execution noise
        deltas.append(delta)
        n_valid += 1

    print(f"[v2-fit] valid: {n_valid}, skipped: {n_skipped}")
    deltas = np.array(deltas, dtype=np.float32)
    mean_delta = np.mean(deltas, axis=0)
    std_delta = np.std(deltas, axis=0, ddof=1)
    # Enforce minimum std
    std_delta = np.maximum(std_delta, 0.001)
    print(f"[v2-fit] mean delta: {mean_delta}")
    print(f"[v2-fit] std delta:  {std_delta}")
    return mean_delta, std_delta


# =====================================================================
# MC scoring (one shot)
# =====================================================================

def score_shot(
    row_dict: dict,
    sampler,
    model_fn,
    sim_cache: SimFnCache,
    bounds: SolveBounds,
    rng: np.random.Generator,
    B: int,
    v_next_cache: dict,
) -> dict | None:
    """Score a single shot with the given noise sampler. Returns output row dict or None."""
    est_params = np.array([row_dict.get(c, np.nan) for c in PARAM_COLS], dtype=np.float32)

    prev_mat = np.full((12, 2), np.nan, dtype=np.float32)
    next_mat = np.full((12, 2), np.nan, dtype=np.float32)
    for s in range(1, 13):
        prev_mat[s - 1] = [row_dict.get(f"prev_stone_{s}_x_m", np.nan),
                           row_dict.get(f"prev_stone_{s}_y_m", np.nan)]
        next_mat[s - 1] = [row_dict.get(f"next_stone_{s}_x_m", np.nan),
                           row_dict.get(f"next_stone_{s}_y_m", np.nan)]

    _, prev_ids_compact = compact_positions(prev_mat)
    _, next_ids = compact_positions(next_mat)
    prev_n = len(prev_ids_compact)

    prev_slots, prev_slot_mask = state_to_fixed_slot_arrays(prev_mat)
    if int(np.sum(prev_slot_mask)) > 1:
        prev_slots[prev_slot_mask] = _separate_overlaps(prev_slots[prev_slot_mask])

    obs_throw_slot_id = float(row_dict.get("obs_throw_slot_id", np.nan))
    team_slot_block = float(row_dict.get("team_slot_block", np.nan))
    thrower_block = infer_thrower_block(
        prev_ids=prev_ids_compact, next_ids=next_ids,
        obs_throw_slot_id=obs_throw_slot_id, team_slot_block=team_slot_block,
    )
    new_id = choose_new_slot_id(
        prev_ids=prev_ids_compact, next_ids=next_ids,
        thrower_block=thrower_block, obs_throw_slot_id=obs_throw_slot_id,
    )

    shot_norm_prev = float(row_dict.get("shot_norm_prev", np.nan))
    shot_norm_next = float(row_dict.get("shot_norm_next", np.nan))
    if not np.isfinite(shot_norm_prev) and np.isfinite(shot_norm_next):
        shot_norm_prev = shot_norm_next
    if not np.isfinite(shot_norm_prev):
        shot_norm_prev = 0.0
    if not np.isfinite(shot_norm_next):
        shot_norm_next = shot_norm_prev

    shot_index_next = float(row_dict.get("ShotIndex", 0.0))
    if not np.isfinite(shot_index_next):
        shot_index_next = 0.0
    shot_index_prev = shot_index_next - 1.0

    team_order = float(row_dict.get("team_order", 0.0))
    stone_block = float(thrower_block)
    c_next = np.array([shot_norm_next, team_order, stone_block], dtype=np.float32)
    next_defaults = make_raw_defaults_for_state(shot_index_next, team_order, thrower_block)
    next_raw_norm = normalize_raw_matrix(positions_m_to_raw_matrix(next_mat, raw_defaults=next_defaults))
    v_next = float(model_fn(next_raw_norm, c_next))

    shot_id_prev = row_dict.get("ShotID_prev", np.nan)
    if np.isfinite(shot_id_prev) and int(shot_id_prev) in v_next_cache:
        v_prev = -v_next_cache[int(shot_id_prev)]
    else:
        c_prev = np.array([shot_norm_prev, team_order, stone_block], dtype=np.float32)
        prev_defaults = make_raw_defaults_for_state(shot_index_prev, team_order, thrower_block)
        prev_raw_norm = normalize_raw_matrix(positions_m_to_raw_matrix(prev_mat, raw_defaults=prev_defaults))
        v_prev = float(model_fn(prev_raw_norm, c_prev))

    dv_obs = v_next - v_prev

    current_shot_id = row_dict.get("ShotID", np.nan)
    if np.isfinite(current_shot_id):
        v_next_cache[int(current_shot_id)] = v_next

    dv_samples = np.empty((0,), dtype=np.float32)
    valid_center = np.all(np.isfinite(est_params)) and np.isfinite(dv_obs) and (0 <= prev_n <= 12)

    if valid_center:
        x_batch = np.zeros((B, 4), dtype=np.float32)
        for b in range(B):
            s = sampler.draw(rng, center=est_params)
            x_batch[b] = clip_to_bounds(s, bounds)

        prev_j = jnp.asarray(prev_slots, dtype=jnp.float32)
        x_j = jnp.asarray(x_batch, dtype=jnp.float32)
        sim_fn = sim_cache.get(12, B)
        finals = np.asarray(sim_fn(prev_j, x_j))

        full_final_batch = assign_finals_12slot_batch(finals, prev_slot_mask, new_id)
        final_raw_norm_batch = normalize_raw_matrix_batch(
            positions_m_to_raw_matrix_batch(full_final_batch, raw_defaults=next_defaults)
        )
        c_next_batch = np.broadcast_to(c_next.reshape(1, -1), (B, c_next.shape[0])).astype(np.float32, copy=False)
        v_sim_batch = np.asarray(model_fn(final_raw_norm_batch, c_next_batch), dtype=np.float32).reshape(-1)
        dv_samples = v_sim_batch - np.float32(v_prev)

    if dv_samples.size > 0:
        dv_mean = float(np.mean(dv_samples))
        dv_std = float(np.std(dv_samples, ddof=1)) if dv_samples.size > 1 else math.nan
    else:
        dv_mean = dv_std = math.nan

    out = {k: row_dict.get(k, np.nan) for k in SHOT_KEY + ["TeamID", "PlayerID", "Task", "Handle"]}
    out.update(dict(
        dv_obs=float(dv_obs) if np.isfinite(dv_obs) else math.nan,
        dv_mean=dv_mean,
        dv_std=dv_std,
        hard_loss=float(row_dict.get("hard_loss_refine", math.nan)),
        est_speed=float(row_dict.get("est_speed", math.nan)),
        est_angle=float(row_dict.get("est_angle", math.nan)),
        est_spin=float(row_dict.get("est_spin", math.nan)),
        est_y0=float(row_dict.get("est_y0", math.nan)),
    ))
    return out


# =====================================================================
# Score all shots with a given sampler
# =====================================================================

def score_all_shots(
    df: pd.DataFrame,
    sampler,
    model_fn,
    curl_params,
    bounds: SolveBounds,
    num_samples: int,
    seed: int,
    label: str,
) -> pd.DataFrame:
    sim_cache = SimFnCache(curl_params)
    rng = np.random.default_rng(seed)
    v_cache: dict = {}
    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Scoring [{label}]"):
        out = score_shot(row.to_dict(), sampler, model_fn, sim_cache, bounds, rng, num_samples, v_cache)
        if out is not None:
            rows.append(out)
    return pd.DataFrame(rows)


# =====================================================================
# Player rankings (simplified from player_skill_model.py)
# =====================================================================

def compute_player_rankings(
    scores_df: pd.DataFrame,
    competitors_csv: str,
    teams_csv: str,
    max_hard_loss: float = 0.5,
    prior_strength: float = 15.0,
    min_shots: int = 8,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (player_task_df, summary_df)."""
    df = scores_df.copy()
    df = df[df["hard_loss"].notna() & (df["hard_loss"] < max_hard_loss)].copy()
    df = df.dropna(subset=["dv_obs", "dv_mean", "Task", "PlayerID", "TeamID"])
    df["metric"] = df["dv_obs"] - df["dv_mean"]

    # Load names
    try:
        comp_df = pd.read_csv(competitors_csv)
        comp_df["player_ord"] = comp_df.groupby("TeamID").cumcount() + 1
        player_names = {(int(r.TeamID), int(r.player_ord)): str(r.Reportingname)
                        for _, r in comp_df.iterrows()}
    except Exception:
        player_names = {}
    try:
        team_names = {int(r.TeamID): str(r.Name) for _, r in pd.read_csv(teams_csv).iterrows()}
    except Exception:
        team_names = {}

    task_stats = df.groupby("Task")["metric"].agg(["mean", "std", "count"]).rename(
        columns={"mean": "task_mean", "std": "task_std", "count": "task_count"}
    )
    global_mean = float(df["metric"].mean()) if len(df) else 0.0
    global_std = float(df["metric"].std(ddof=1)) if len(df) > 1 else 1.0

    rows = []
    for (pid, tid, task), g in df.groupby(["PlayerID", "TeamID", "Task"]):
        n = len(g)
        if n < min_shots:
            continue
        raw_mean = float(g["metric"].mean())
        raw_std = float(g["metric"].std(ddof=1)) if n > 1 else 0.0
        tr = task_stats.loc[task] if task in task_stats.index else None
        prior_mean = float(tr["task_mean"]) if tr is not None else global_mean
        prior_std = float(tr["task_std"]) if tr is not None else global_std
        weight = n / (n + prior_strength)
        effect = weight * raw_mean + (1 - weight) * prior_mean
        se = math.sqrt(max(raw_std, prior_std) ** 2 / max(1.0, n + prior_strength))
        ci = 1.96 * se

        pid_i, tid_i, task_i = int(pid), int(tid), int(task)
        rows.append(dict(
            PlayerID=pid_i, TeamID=tid_i, Task=task_i, shots=n,
            raw_mean=raw_mean, effect_mean=effect,
            effect_lower=effect - ci, effect_upper=effect + ci, se=se,
            player_name=player_names.get((tid_i, pid_i), ""),
            team_name=team_names.get(tid_i, ""),
        ))

    pt_df = pd.DataFrame(rows)
    if not pt_df.empty:
        pt_df = pt_df.sort_values(["Task", "effect_mean"], ascending=[True, False])

    # Summary
    summaries = []
    for (pid, tid), g in pt_df.groupby(["PlayerID", "TeamID"]):
        total = g["shots"].sum()
        w = g["shots"] / total
        avg_effect = float((g["effect_mean"] * w).sum())
        consistency = float(df[(df["PlayerID"] == pid) & (df["TeamID"] == tid)]["metric"].std(ddof=1))
        summaries.append(dict(
            PlayerID=int(pid), TeamID=int(tid),
            player_name=player_names.get((int(tid), int(pid)), ""),
            team_name=team_names.get(int(tid), ""),
            total_shots=int(total), tasks=len(g),
            avg_effect=avg_effect, consistency=consistency,
        ))
    sum_df = pd.DataFrame(summaries)
    if not sum_df.empty:
        sum_df = sum_df.sort_values("avg_effect", ascending=False)
    return pt_df, sum_df


# =====================================================================
# Visualization
# =====================================================================

TASK_NAMES = {
    -1: "Unknown", 0: "Draw", 1: "Blank", 2: "Freeze", 3: "Guard",
    4: "Raise", 5: "Corner Guard", 6: "Hit & Roll", 7: "Hit & Stay",
    8: "Peel", 9: "Takeout", 10: "Double", 11: "Tick",
}


def plot_player_summary_comparison(
    results: Dict[str, pd.DataFrame],
    out_path: pathlib.Path,
):
    """Bar chart comparing avg_effect across noise versions for each player."""
    versions = list(results.keys())
    # Get union of players across all versions
    all_players = set()
    for df in results.values():
        for _, r in df.iterrows():
            all_players.add((int(r["PlayerID"]), int(r["TeamID"]),
                             str(r.get("player_name", "")), str(r.get("team_name", ""))))
    all_players = sorted(all_players, key=lambda x: x[0])
    if not all_players:
        print("[warn] no players to plot")
        return

    fig, ax = plt.subplots(figsize=(max(12, len(all_players) * 1.5), 7))
    n_versions = len(versions)
    bar_width = 0.8 / n_versions
    colors = ["#2196F3", "#FF9800", "#4CAF50", "#E91E63"]

    for vi, vname in enumerate(versions):
        df = results[vname]
        vals = []
        for pid, tid, pname, tname in all_players:
            match = df[(df["PlayerID"] == pid) & (df["TeamID"] == tid)]
            if len(match):
                vals.append(float(match.iloc[0]["avg_effect"]))
            else:
                vals.append(0.0)
        x = np.arange(len(all_players))
        ax.bar(x + vi * bar_width, vals, bar_width, label=vname,
               color=colors[vi % len(colors)], alpha=0.85, edgecolor="white", linewidth=0.5)

    labels = [f"{p[2]}\n({p[3]})" if p[2] else f"P{p[0]}\nT{p[1]}" for p in all_players]
    ax.set_xticks(np.arange(len(all_players)) + 0.4)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Avg Effect (execution above/below expected)", fontsize=11)
    ax.set_title("Player Rankings by Noise Model", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, loc="upper right")
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] saved player summary comparison -> {out_path}")


def plot_player_task_comparison(
    results: Dict[str, pd.DataFrame],
    out_path: pathlib.Path,
):
    """One subplot per task, showing effect_mean ± CI for each player across noise versions."""
    versions = list(results.keys())
    all_tasks = set()
    for df in results.values():
        all_tasks.update(df["Task"].dropna().unique().tolist())
    all_tasks = sorted(all_tasks)
    # Only keep tasks with data in at least one version
    all_tasks = [t for t in all_tasks if any(
        len(results[v][results[v]["Task"] == t]) > 0 for v in versions
    )]
    if not all_tasks:
        print("[warn] no task data to plot")
        return

    n_tasks = len(all_tasks)
    n_cols = min(3, n_tasks)
    n_rows = math.ceil(n_tasks / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4.5 * n_rows), squeeze=False)

    colors = ["#2196F3", "#FF9800", "#4CAF50", "#E91E63"]

    for ti, task in enumerate(all_tasks):
        ax = axes[ti // n_cols][ti % n_cols]
        task_name = TASK_NAMES.get(int(task), f"Task {int(task)}")
        ax.set_title(f"{task_name} (Task {int(task)})", fontsize=11, fontweight="bold")

        # Collect all players for this task
        players_for_task = set()
        for df in results.values():
            sub = df[df["Task"] == task]
            for _, r in sub.iterrows():
                players_for_task.add((int(r["PlayerID"]), int(r["TeamID"]),
                                      str(r.get("player_name", ""))))
        players_for_task = sorted(players_for_task, key=lambda x: x[0])

        n_versions = len(versions)
        for vi, vname in enumerate(versions):
            df = results[vname]
            sub = df[df["Task"] == task]
            effects = []
            lows = []
            highs = []
            for pid, tid, pname in players_for_task:
                match = sub[(sub["PlayerID"] == pid) & (sub["TeamID"] == tid)]
                if len(match):
                    r = match.iloc[0]
                    effects.append(float(r["effect_mean"]))
                    lows.append(float(r["effect_lower"]))
                    highs.append(float(r["effect_upper"]))
                else:
                    effects.append(np.nan)
                    lows.append(np.nan)
                    highs.append(np.nan)

            x = np.arange(len(players_for_task))
            offset = (vi - n_versions / 2 + 0.5) * 0.15
            valid = [i for i in range(len(effects)) if np.isfinite(effects[i])]
            if valid:
                vx = x[valid] + offset
                ve = [effects[i] for i in valid]
                vl = [effects[i] - lows[i] for i in valid]
                vh = [highs[i] - effects[i] for i in valid]
                ax.errorbar(vx, ve, yerr=[vl, vh], fmt="o", markersize=5,
                            color=colors[vi % len(colors)], label=vname if ti == 0 else None,
                            capsize=3, linewidth=1.2, alpha=0.85)

        labels = [p[2] if p[2] else f"P{p[0]}" for p in players_for_task]
        ax.set_xticks(np.arange(len(players_for_task)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.axhline(0, color="gray", linewidth=0.7, linestyle="--")
        ax.grid(axis="y", alpha=0.25)
        ax.set_ylabel("Effect")

    # Remove empty subplots
    for ti in range(n_tasks, n_rows * n_cols):
        axes[ti // n_cols][ti % n_cols].set_visible(False)

    if n_tasks > 0:
        handles, labels = axes[0][0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=len(versions), fontsize=10,
                   bbox_to_anchor=(0.5, 1.02))

    fig.suptitle("Player Skill by Task and Noise Model", fontsize=14, fontweight="bold", y=1.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] saved player-task comparison -> {out_path}")


def plot_noise_parameter_comparison(
    noise_params: Dict[str, Dict],
    out_path: pathlib.Path,
):
    """Compare noise standard deviations across versions."""
    param_names = ["Speed (m/s)", "Angle (rad)", "Spin (rad/s)", "Y0 (m)"]
    versions = list(noise_params.keys())
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    colors = ["#2196F3", "#FF9800", "#4CAF50"]
    for pi in range(4):
        ax = axes[pi]
        vals = []
        for v in versions:
            vals.append(noise_params[v]["std"][pi])
        x = np.arange(len(versions))
        bars = ax.bar(x, vals, color=[colors[i % len(colors)] for i in range(len(versions))],
                      alpha=0.85, edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels(versions, rotation=20, ha="right", fontsize=9)
        ax.set_title(param_names[pi], fontsize=11, fontweight="bold")
        ax.set_ylabel("Noise Std")
        # Add value labels
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{val:.4f}", ha="center", va="bottom", fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Execution Noise Parameters by Model", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] saved noise parameter comparison -> {out_path}")


def plot_dv_distribution_comparison(
    all_scores: Dict[str, pd.DataFrame],
    out_path: pathlib.Path,
):
    """Overlay histograms of dv_mean and (dv_obs - dv_mean) across noise versions."""
    versions = list(all_scores.keys())
    colors = ["#2196F3", "#FF9800", "#4CAF50"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: dv_mean distribution
    ax = axes[0]
    for vi, v in enumerate(versions):
        df = all_scores[v]
        vals = df["dv_mean"].dropna()
        if len(vals):
            ax.hist(vals, bins=60, alpha=0.45, color=colors[vi % len(colors)],
                    label=f"{v} (μ={vals.mean():.3f})", density=True)
    ax.set_xlabel("dv_mean (expected value added)")
    ax.set_ylabel("Density")
    ax.set_title("MC Expected Value Distribution", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Right: execution excess (dv_obs - dv_mean)
    ax = axes[1]
    for vi, v in enumerate(versions):
        df = all_scores[v]
        excess = (df["dv_obs"] - df["dv_mean"]).dropna()
        if len(excess):
            ax.hist(excess, bins=60, alpha=0.45, color=colors[vi % len(colors)],
                    label=f"{v} (σ={excess.std():.3f})", density=True)
    ax.set_xlabel("Execution Excess (dv_obs − dv_mean)")
    ax.set_ylabel("Density")
    ax.set_title("Execution Quality Distribution", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    fig.suptitle("Shot Value Distributions by Noise Model", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] saved value distribution comparison -> {out_path}")


# =====================================================================
# Main
# =====================================================================

def main():
    ap = argparse.ArgumentParser(description="Compare 3 execution noise models for curling.")
    ap.add_argument("--inverse-glob", type=str,
                    default=str(ROOT / "inverse_current" / "stones_with_estimates.chunk*.csv"))
    ap.add_argument("--stones-csv", type=str, default=str(ROOT / "2026" / "Stones.csv"))
    ap.add_argument("--competitors-csv", type=str, default=str(ROOT / "2026" / "Competitors.csv"))
    ap.add_argument("--teams-csv", type=str, default=str(ROOT / "2026" / "Teams.csv"))
    ap.add_argument("--value-model", type=str,
                    default=str(ROOT / "submission" / "valueModel" / "value_model_synth_v4best.pt"))
    ap.add_argument("--num-samples", type=int, default=32, help="MC samples per shot for scoring")
    ap.add_argument("--fit-samples", type=int, default=64, help="Samples for V2 local search fitting")
    ap.add_argument("--fit-max-shots", type=int, default=500, help="Max shots for V2 fitting (subsample)")
    ap.add_argument("--max-score-shots", type=int, default=3000, help="Max shots for scoring per version")
    ap.add_argument("--hard-loss-max", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--out-dir", type=str, default=str(OUT_DIR))
    ap.add_argument("--skip-fit", action="store_true", help="Skip V2 fitting; use cached v2_fitted.json if exists")
    ap.add_argument("--only-version", type=str, default=None, help="Run only this version (v1/v2/v3)")
    args = ap.parse_args()

    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("[info] loading data...", flush=True)
    full_df = prepare_dataframe_all(
        stones_csv=args.stones_csv,
        inverse_glob=args.inverse_glob,
        only_solver_ok=True,
        hard_loss_max=float(args.hard_loss_max),
    )
    print(f"[info] {len(full_df)} shots after filtering", flush=True)

    model_fn, _ = load_value_model(pathlib.Path(args.value_model), device=args.device)
    curl_params = contact_mild_params(CurlingParams, dt=0.02, substeps=2, k_penalty=2.5e4)
    bounds = SolveBounds()

    # ---- V2 fitting ----
    v2_path = out / "v2_fitted.json"
    if args.skip_fit and v2_path.exists():
        v2_cfg = json.loads(v2_path.read_text())
        v2_mean = np.array(v2_cfg["mean"], dtype=np.float32)
        v2_std = np.array(v2_cfg["std"], dtype=np.float32)
        print(f"[v2] loaded cached fit: mean={v2_mean}, std={v2_std}")
    else:
        v2_mean, v2_std = fit_v2_noise(
            full_df, model_fn, curl_params, bounds,
            fit_samples=args.fit_samples,
            max_shots=args.fit_max_shots,
            seed=args.seed,
        )
        v2_cfg = {"mean": v2_mean.tolist(), "std": v2_std.tolist(),
                   "description": "V2: Gaussian fit to (x_inverse - x_best_local) deltas"}
        v2_path.write_text(json.dumps(v2_cfg, indent=2))
        print(f"[v2] saved fit -> {v2_path}")

    # ---- Build samplers ----
    samplers = {}
    noise_params = {}  # For visualization

    if args.only_version is None or args.only_version == "v1":
        samplers["V1 Bowling"] = BowlingNoiseSampler()
        # Effective stds (Gaussian-equivalent for comparison)
        noise_params["V1 Bowling"] = {"std": [0.0123, 0.050, 0.08, 0.015]}

    if args.only_version is None or args.only_version == "v2":
        samplers["V2 Fitted"] = FittedGaussianSampler(v2_mean, v2_std)
        noise_params["V2 Fitted"] = {"std": v2_std.tolist()}

    if args.only_version is None or args.only_version == "v3":
        samplers["V3 Expert"] = ExpertGuessSampler()
        noise_params["V3 Expert"] = {"std": [0.020, 0.014, 0.15, 0.008]}

    # ---- Also include current baseline for reference ----
    # (the template noise config)
    samplers["Current"] = type("CurrentSampler", (), {
        "draw": lambda self, rng, center, **kw: center + rng.normal(
            0.0, np.array([0.05, 0.008, 0.12, 0.03], dtype=np.float32)
        ).astype(np.float32),
    })()
    noise_params["Current"] = {"std": [0.05, 0.008, 0.12, 0.03]}

    # ---- Score shots for each version ----
    all_scores: Dict[str, pd.DataFrame] = {}
    all_rankings_pt: Dict[str, pd.DataFrame] = {}
    all_rankings_summary: Dict[str, pd.DataFrame] = {}

    # Subsample for scoring speed
    score_df = full_df
    if args.max_score_shots and len(full_df) > args.max_score_shots:
        score_df = full_df.sample(n=args.max_score_shots, random_state=args.seed).copy()
        score_df = score_df.sort_values(SHOT_KEY).reset_index(drop=True)
        print(f"[info] subsampled to {len(score_df)} shots for scoring")

    for vname, sampler in samplers.items():
        safe_name = vname.lower().replace(" ", "_")
        scores_path = out / f"shot_scores_{safe_name}.csv"

        if scores_path.exists():
            print(f"[{vname}] loading cached scores from {scores_path}")
            scores_df = pd.read_csv(scores_path)
        else:
            print(f"\n{'='*60}")
            print(f"  Scoring: {vname}")
            print(f"{'='*60}")
            scores_df = score_all_shots(
                score_df, sampler, model_fn, curl_params, bounds,
                num_samples=args.num_samples, seed=args.seed, label=vname,
            )
            scores_df.to_csv(scores_path, index=False)
            print(f"[{vname}] saved {len(scores_df)} scores -> {scores_path}")

        all_scores[vname] = scores_df

        # Player rankings
        pt_df, sum_df = compute_player_rankings(
            scores_df, args.competitors_csv, args.teams_csv,
        )
        pt_path = out / f"player_task_{safe_name}.csv"
        sum_path = out / f"player_summary_{safe_name}.csv"
        pt_df.to_csv(pt_path, index=False)
        sum_df.to_csv(sum_path, index=False)
        all_rankings_pt[vname] = pt_df
        all_rankings_summary[vname] = sum_df
        print(f"[{vname}] {len(pt_df)} player-task rows, {len(sum_df)} player summaries")

    # ---- Visualizations ----
    print(f"\n{'='*60}")
    print("  Generating visualizations")
    print(f"{'='*60}")

    plot_noise_parameter_comparison(noise_params, out / "noise_params_comparison.png")
    plot_player_summary_comparison(all_rankings_summary, out / "player_summary_comparison.png")
    plot_player_task_comparison(all_rankings_pt, out / "player_task_comparison.png")
    plot_dv_distribution_comparison(all_scores, out / "dv_distribution_comparison.png")

    print(f"\n[done] all outputs in {out}/")


if __name__ == "__main__":
    main()
