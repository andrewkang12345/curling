#!/usr/bin/env python3
"""Visualize sampled throw priors from the policy model."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import pandas as pd
import torch
import jax.numpy as jnp

from common import (
    ACTION_ANGLE_MAX,
    ACTION_ANGLE_MIN,
    ACTION_SPEED_MAX,
    ACTION_SPEED_MIN,
    ACTION_SPIN_MAX,
    ACTION_SPIN_MIN,
    ACTION_Y0_MAX,
    ACTION_Y0_MIN,
    FIXED_ROOT,
    NUM_STONES,
    POS_MAX,
    compact_m_to_raw,
    in_play_raw,
    raw_to_compact_m,
    resolve_stone_overlaps,
)
from dataset import ValueDataset
from kr_uct_search import load_policy, _sample_actions, _simulate_candidates
from curling_sim_jax import CurlingParams, simulate_from_params
from policy_dataset import load_inverse_estimates
from preplaced_value_data import canonical_preplacement_cases
from train_holdout_models_cond3 import make_holdout_split

sys.path.insert(0, str(FIXED_ROOT))
import make_value_heatmaps as hv  # noqa: E402

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


def _plot_highlighted_slot(ax, stones_raw, slot: int, color: str = "#f2c14e"):
    if slot is None:
        return
    live = in_play_raw(stones_raw)
    if not live[int(slot)]:
        return
    xy = (stones_raw.reshape(NUM_STONES, 2) - BUTTON_RAW[None]) * M_PER_RAW
    x, y = xy[int(slot)]
    ax.add_patch(Circle((float(x), float(y)), STONE_R, facecolor=color, edgecolor="0.05", lw=1.2, zorder=6))
    ax.text(float(x), float(y), str(int(slot) + 1), ha="center", va="center", fontsize=7, color="black", zorder=7)


def _exact_observed_throw_slot(pre_stones_raw: np.ndarray, post_stones_raw: np.ndarray) -> int | None:
    pre_live = in_play_raw(pre_stones_raw)
    post_live = in_play_raw(post_stones_raw)
    added = np.flatnonzero(post_live & ~pre_live)
    if len(added) == 1:
        return int(added[0])
    return None


def _team_name_from_block(cond: np.ndarray) -> str:
    return "black" if int(round(float(cond[2]))) == 1 else "white"


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


@torch.no_grad()
def _representative_action(policy, mean_t, std_t, x, c, device):
    xb = torch.as_tensor(x[None], dtype=torch.float32, device=device)
    cb = torch.as_tensor(c[None], dtype=torch.float32, device=device)
    pi_logits, mu, _ = policy(xb, cb)
    mix = int(torch.argmax(pi_logits[0]).item())
    action = mu[0, mix] * std_t.to(device) + mean_t.to(device)
    action = np.asarray(action.detach().cpu().tolist(), dtype=np.float32)
    action[0] = np.clip(action[0], ACTION_SPEED_MIN, ACTION_SPEED_MAX)
    action[1] = np.clip(action[1], ACTION_ANGLE_MIN, ACTION_ANGLE_MAX)
    action[2] = np.clip(action[2], ACTION_SPIN_MIN, ACTION_SPIN_MAX)
    action[3] = np.clip(action[3], ACTION_Y0_MIN, ACTION_Y0_MAX)
    return action


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


def heatmap_real_cases(n: int, seed: int, holdout: int, split: str, horizon: int | None = None):
    if horizon is not None and int(horizon) == 10:
        out = []
        for c in canonical_preplacement_cases():
            team = _team_name_from_block(c["cond"])
            mode = c["mode"]
            guard_slot = int(c["guard_slot"])
            out.append(
                {
                    "label": f"preplaced_{mode}_guard{guard_slot}",
                    "title": f"Preplaced canonical state | {mode} guard {guard_slot} | throws remaining: 10 | thrower team: {team}",
                    "stones_raw": c["stones_raw"],
                    "cond": c["cond"],
                }
            )
            if len(out) >= n:
                break
        return out

    ds = ValueDataset(str(FIXED_ROOT / "2026" / "Stones.csv"), str(FIXED_ROOT / "2026" / "Ends.csv"), augment_positions=False, augment_flip=False)
    _, val_idx, test_idx, _ = make_holdout_split(ds.df, holdout, 0.10, 123)
    split_idx = val_idx if split == "val" else test_idx
    rng = np.random.default_rng(seed)
    inv = load_inverse_estimates(str(FIXED_ROOT / "inverse_current" / "stones_with_estimates.chunk*.csv"), 0.08)
    inv_map = {
        (int(row["CompetitionID"]), int(row["SessionID"]), int(row["GameID"]), int(row["EndID"]), int(row["ShotID"])): np.asarray(
            [row["est_speed"], row["est_angle"], row["est_spin"], row["est_y0"]], dtype=np.float32
        )
        for _, row in inv.iterrows()
    }
    rows = []
    seen = set()
    shuffled = np.asarray(split_idx, dtype=np.int64).copy()
    rng.shuffle(shuffled)
    for idx in shuffled:
        row = ds.df.iloc[int(idx)]
        throws_remaining = int(row["ShotsInEnd"]) - int(row["ShotIndex"])
        if horizon is not None and throws_remaining != int(horizon):
            continue
        if horizon is not None and int(row["ShotsInEnd"]) != 10:
            continue
        key = (
            int(row["CompetitionID"]),
            int(row["SessionID"]),
            int(row["GameID"]),
            int(row["EndID"]),
            int(row["ShotID"]),
        )
        key4 = (key[0], key[2], key[3], key[4])
        if key4 in seen:
            continue
        slot = hv._find_thrown_slot(ds.df, int(idx))
        if slot is None:
            continue
        case = hv._real_case(ds, int(idx), slot)
        if case is None:
            continue
        obs_slot = _exact_observed_throw_slot(case["pre_stones"], case["observed_stones"])
        if obs_slot is None:
            continue
        team = _team_name_from_block(case["post_cond"])
        case["label"] = f"early_comp{key[0]}_sess{key[1]}_game{key[2]}_end{key[3]}_shot{key[4]}"
        case["title"] = (
            f"Early test state | comp {key[0]} sess {key[1]} game {key[2]} "
            f"end {key[3]} shot {key[4]} | throws remaining: {throws_remaining} | thrower team: {team}"
        )
        case["key"] = key
        case["inverse_action"] = inv_map.get(key)
        rows.append(case)
        seen.add(key4)
        if len(rows) >= n:
            break
    return rows


def _trajectory_m_for_action(pre_stones_raw: np.ndarray, cond: np.ndarray, action: np.ndarray) -> np.ndarray | None:
    if action is None:
        return None
    raw = np.asarray(pre_stones_raw, dtype=np.float32).reshape(NUM_STONES, 2)
    live = in_play_raw(raw)
    compact_slots = raw_to_compact_m(raw)
    prev = compact_slots[np.where(live)[0].astype(np.int64)]
    if prev.size == 0:
        prev = np.zeros((0, 2), dtype=np.float32)
    else:
        prev = resolve_stone_overlaps(prev)
    traj = np.asarray(
        simulate_from_params(
            CurlingParams(),
            jnp.asarray(prev, dtype=jnp.float32),
            jnp.asarray(np.asarray(action, dtype=np.float32), dtype=jnp.float32),
            dynamic=True,
        )
    )
    if traj.ndim != 3 or traj.shape[1] == 0:
        return None
    throw_compact = traj[:, -1, :]
    if not np.isfinite(throw_compact).all():
        return None
    # Simulator compact coords are [along_from_button, lateral_from_button] in meters.
    # Plotting uses [lateral, raw_y-button] meters, so y = -along.
    throw_xy_plot = np.stack([throw_compact[:, 1], -throw_compact[:, 0]], axis=1)
    return throw_xy_plot.astype(np.float32)


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
    pre_stones = case.get("pre_stones", case.get("stones_raw"))
    post_stones = case.get("observed_stones")
    cond = case.get("post_cond", case.get("cond"))
    sim_slot = int(case.get("slot", _new_slot(pre_stones, cond)))
    obs_slot = None if post_stones is None else _exact_observed_throw_slot(pre_stones, post_stones)
    team_name = _team_name_from_block(cond)
    traj_m = _trajectory_m_for_action(pre_stones, cond, case.get("inverse_action"))

    x = (pre_stones.reshape(-1) / POS_MAX).astype(np.float32)
    c = cond.astype(np.float32)
    rep_action = _representative_action(policy, mean_t, std_t, x, c, device)
    rep_post = _simulate_candidates(x, c, rep_action[None])[0].reshape(NUM_STONES, 2) * POS_MAX
    rep_traj_m = _trajectory_m_for_action(pre_stones, cond, rep_action)
    actions = _sample_actions(policy, mean_t, std_t, x, c, n_samples, device, temperature, std_scale, global_frac)
    posts = _simulate_candidates(x, c, actions)
    endpoints = _endpoint_m_from_states(posts, pre_stones, c)

    fig, axes = plt.subplots(1, 5, figsize=(20.6, 5.8), dpi=170, gridspec_kw={"width_ratios": [1.0, 1.0, 1.0, 1.0, 1.05]})

    ax = axes[0]
    _draw_house(ax)
    _plot_stones(ax, pre_stones)
    if traj_m is not None and len(traj_m):
        ax.plot(traj_m[:, 0], traj_m[:, 1], linestyle=":", color="#d62728", lw=1.8, alpha=0.95, zorder=2)
    ax.set_xlim(-2.375, 2.375)
    ax.set_ylim(-6.40, 6.40)
    ax.set_aspect("equal")
    ax.set_xlabel("lateral from button (m)")
    ax.set_ylabel("along-sheet from button (m)")
    ax.set_title("pre-throw state")

    ax = axes[1]
    _draw_house(ax)
    if post_stones is not None:
        _plot_stones(ax, post_stones)
        if obs_slot is not None:
            _plot_highlighted_slot(ax, post_stones, obs_slot)
        if traj_m is not None and len(traj_m):
            ax.plot(traj_m[:, 0], traj_m[:, 1], linestyle=":", color="#d62728", lw=1.8, alpha=0.95, zorder=2)
    else:
        _plot_stones(ax, pre_stones)
        ax.text(0.5, 0.5, "no observed post state", transform=ax.transAxes, ha="center", va="center", fontsize=9, color="0.35")
    ax.set_xlim(-2.375, 2.375)
    ax.set_ylim(-6.40, 6.40)
    ax.set_aspect("equal")
    ax.set_xlabel("lateral from button (m)")
    ax.set_ylabel("along-sheet from button (m)")
    ax.set_title("post-throw state")

    ax = axes[2]
    _draw_house(ax)
    _plot_stones(ax, rep_post)
    _plot_highlighted_slot(ax, rep_post, sim_slot, color="#8ecae6")
    if rep_traj_m is not None and len(rep_traj_m):
        ax.plot(rep_traj_m[:, 0], rep_traj_m[:, 1], linestyle=":", color="#1d4ed8", lw=1.8, alpha=0.95, zorder=2)
    ax.set_xlim(-2.375, 2.375)
    ax.set_ylim(-6.40, 6.40)
    ax.set_aspect("equal")
    ax.set_xlabel("lateral from button (m)")
    ax.set_ylabel("along-sheet from button (m)")
    ax.set_title("policy post-throw state")

    ax = axes[3]
    _draw_house(ax)
    _plot_stones(ax, pre_stones)
    if traj_m is not None and len(traj_m):
        ax.plot(traj_m[:, 0], traj_m[:, 1], linestyle=":", color="#d62728", lw=1.2, alpha=0.55, zorder=2)
    if rep_traj_m is not None and len(rep_traj_m):
        ax.plot(rep_traj_m[:, 0], rep_traj_m[:, 1], linestyle=":", color="#1d4ed8", lw=1.2, alpha=0.55, zorder=2)
    if len(endpoints):
        ax.scatter(endpoints[:, 0], endpoints[:, 1], s=8, alpha=0.28, color="tab:blue", edgecolors="none")
    ax.set_xlim(-2.375, 2.375)
    ax.set_ylim(-6.40, 6.40)
    ax.set_aspect("equal")
    ax.set_xlabel("lateral from button (m)")
    ax.set_ylabel("along-sheet from button (m)")
    ax.set_title("policy-sampled endpoints")

    labels = ["speed", "angle", "spin", "y0"]
    for j, lab in enumerate(labels):
        axes[4].hist(actions[:, j], bins=40, alpha=0.55, label=lab)
        axes[4].axvline(float(rep_action[j]), color=["#2563eb", "#f59e0b", "#16a34a", "#dc2626"][j], lw=1.2, linestyle="--")
    axes[4].legend(fontsize=8)
    axes[4].set_title("sampled action parameters")
    axes[4].set_xlabel("raw action value")
    axes[4].set_ylabel("count")

    title = case["title"]
    title = title.replace(f" | thrower: {team_name} stone {sim_slot + 1}", "")
    if obs_slot is not None:
        title = f"{title} | thrower: {team_name} stone {obs_slot + 1}"
    else:
        title = f"{title} | thrower team: {team_name} | observed slot ambiguous"
    fig.suptitle(title, fontsize=10)
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
    ap.add_argument("--case-mode", choices=["mixed", "heatmap_real"], default="mixed")
    ap.add_argument("--holdout", type=int, default=0)
    ap.add_argument("--split", choices=["val", "test"], default="test")
    ap.add_argument("--horizon", type=int, default=None, help="If set with --case-mode heatmap_real, only use states with this many throws remaining.")
    args = ap.parse_args()

    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    policy, mean_t, std_t = load_policy(args.policy, device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.case_mode == "heatmap_real":
        cases = heatmap_real_cases(args.n_real, args.seed, args.holdout, args.split, args.horizon)
    else:
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
