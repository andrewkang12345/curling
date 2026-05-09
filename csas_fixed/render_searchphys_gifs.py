#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
import pathlib
import sys

import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

THIS_DIR = pathlib.Path(__file__).resolve().parent
sys.path.append(str(THIS_DIR))
sys.path.append(str(THIS_DIR / "inverse"))

import jax.numpy as jnp
from curling_inverse import build_batched_hard_loss_by_block_flex  # type: ignore
from curling_sim_jax import CurlingParams, make_initial_state, rollout_positions_until_stop_flex  # type: ignore

SHOT_KEY = ["CompetitionID", "SessionID", "GameID", "EndID", "ShotID"]
MIN_X = -6.6
MAX_X = 2.4
MIN_Y = -2.286
MAX_Y = 2.286
PAD = np.array([50.0, 50.0], dtype=np.float32)
MIN_CLEAR = 2 * 0.145 + 1e-3

SIM_PARAMS = dict(
    dt=0.02,
    substeps=2,
    max_steps=1500,
    k_penalty=2.5e4,
    c_damp=165.0,
    c_damp_sep_frac=1.0,
    c_tangent=20.0,
    mu_tangent=0.05,
    spin_contact=0.08,
    k_curl=0.12,
    a_linear=0.10,
    gamma_spin=0.12,
)

PHYS_DEFAULT = np.array([
    SIM_PARAMS["k_curl"],
    SIM_PARAMS["a_linear"],
    SIM_PARAMS["gamma_spin"],
    SIM_PARAMS["c_damp"],
    SIM_PARAMS["c_tangent"],
    SIM_PARAMS["mu_tangent"],
    SIM_PARAMS["spin_contact"],
], dtype=np.float32)
PHYS_LO = np.array([0.04, 0.03, 0.04, 50.0, 5.0, 0.01, 0.02], dtype=np.float32)
PHYS_HI = np.array([0.40, 0.30, 0.40, 500.0, 80.0, 0.20, 0.30], dtype=np.float32)


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
        if marker == "x":
            ax.scatter(y_m, x_m, s=s, marker=marker, color="tab:red", linewidths=1.7, alpha=alpha, zorder=5)
        else:
            ax.scatter(
                y_m,
                x_m,
                s=s,
                marker=marker,
                facecolors=face[i],
                edgecolors=edge[i],
                linewidths=1.4,
                alpha=alpha,
                zorder=4,
            )
        text_color = "white" if face[i] == "black" else "black"
        ax.text(y_m, x_m, str(int(slot_id)), ha="center", va="center", fontsize=7, color=text_color, alpha=min(1.0, alpha + 0.1), zorder=6)


def _draw_trails(ax, traj_full: np.ndarray, frame_idx: int, thrower_block: int):
    cur_traj = traj_full[: frame_idx + 1]
    for slot_idx in range(traj_full.shape[1]):
        pts = cur_traj[:, slot_idx, :]
        mask = ~np.isnan(pts).any(axis=1)
        if np.count_nonzero(mask) < 2:
            continue
        slot_id = slot_idx + 1
        face, _ = _slot_colors(np.array([slot_id]), thrower_block)
        color = "0.35" if face[0] == "white" else "0.0"
        ax.plot(pts[mask, 1], pts[mask, 0], linewidth=1.0, alpha=0.55, color=color, zorder=2)


def _difference_metrics(pred_12x2: np.ndarray, actual_12x2: np.ndarray):
    pred_mask = ~np.isnan(pred_12x2).any(axis=1)
    actual_mask = ~np.isnan(actual_12x2).any(axis=1)
    both = pred_mask & actual_mask
    pred_only = pred_mask & (~actual_mask)
    actual_only = actual_mask & (~pred_mask)
    if both.any():
        diffs = pred_12x2[both] - actual_12x2[both]
        rmse = float(np.sqrt(np.mean(np.sum(diffs**2, axis=1))))
    else:
        rmse = float("nan")
    return {
        "rmse": rmse,
        "matched": int(np.count_nonzero(both)),
        "pred_only": int(np.count_nonzero(pred_only)),
        "actual_only": int(np.count_nonzero(actual_only)),
        "both_mask": both,
    }


def _fig_to_rgb(fig) -> np.ndarray:
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    rgba = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
    return rgba[..., :3].copy()


def _in_bounds(pt: np.ndarray) -> bool:
    return bool((pt[0] > MIN_X) and (pt[0] < MAX_X) and (pt[1] > MIN_Y) and (pt[1] < MAX_Y))


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


def separate_overlaps(pts: np.ndarray, passes: int = 6) -> np.ndarray:
    p = pts.copy()
    n = p.shape[0]
    for _ in range(passes):
        moved = False
        for i in range(n):
            for j in range(i + 1, n):
                dx, dy = p[j, 0] - p[i, 0], p[j, 1] - p[i, 1]
                d = math.hypot(dx, dy)
                if d < 1e-9:
                    dx, dy, d = 1e-6, 0.0, 1e-6
                if d < MIN_CLEAR:
                    push = 0.5 * (MIN_CLEAR - d)
                    p[i, 0] -= push * dx / d
                    p[i, 1] -= push * dy / d
                    p[j, 0] += push * dx / d
                    p[j, 1] += push * dy / d
                    moved = True
        if not moved:
            break
    return p


def load_shot(row: pd.Series) -> dict:
    prev_s = np.tile(PAD, (12, 1)).astype(np.float32)
    prev_m = np.zeros(12, dtype=bool)
    next_s = np.tile(PAD, (12, 1)).astype(np.float32)
    next_m = np.zeros(12, dtype=bool)

    for i in range(1, 13):
        px, py = row.get(f"prev_stone_{i}_x_m", np.nan), row.get(f"prev_stone_{i}_y_m", np.nan)
        if pd.notna(px) and abs(px) < 40:
            prev_s[i - 1] = [px, py]
            prev_m[i - 1] = True
        nx, ny = row.get(f"next_stone_{i}_x_m", np.nan), row.get(f"next_stone_{i}_y_m", np.nan)
        ib = row.get(f"next_stone_{i}_inbounds", 0)
        if pd.notna(nx) and abs(nx) < 40 and ib:
            next_s[i - 1] = [nx, ny]
            next_m[i - 1] = True

    if int(np.sum(prev_m)) > 1:
        prev_s[prev_m] = separate_overlaps(prev_s[prev_m])

    added = sorted(set(np.where(next_m)[0]) - set(np.where(prev_m)[0]))
    thrower_block = 0
    if len(added) == 1:
        thrower_block = 0 if added[0] < 6 else 1

    return dict(
        prev_slots=prev_s,
        prev_mask=prev_m,
        next_slots=next_s,
        next_mask=next_m,
        thrower_block=thrower_block,
        added_slots=added,
        tgt0=next_s[:6].copy(),
        tgt0m=next_m[:6].copy(),
        tgt1=next_s[6:12].copy(),
        tgt1m=next_m[6:12].copy(),
        est_x=np.array([row["est_speed"], row["est_angle"], row["est_spin"], row["est_y0"]], dtype=np.float32),
    )


def _pred_frame_to_slots(frame_13x2: np.ndarray, shot: dict) -> np.ndarray:
    out = np.full((12, 2), np.nan, dtype=np.float32)
    for i in range(12):
        if not shot["prev_mask"][i]:
            continue
        pt = frame_13x2[i]
        if _in_bounds(pt):
            out[i] = pt
    if len(shot["added_slots"]) == 1:
        throw_pt = frame_13x2[12]
        if _in_bounds(throw_pt):
            out[shot["added_slots"][0]] = throw_pt
    return out


def _recover_best_phys(orig_idx: int, shot: dict, fn_flex, sigma: float, n_phys_samples: int) -> np.ndarray:
    phys_rng = np.random.default_rng(int(orig_idx) + 9999)
    phys_samples = np.clip(
        phys_rng.lognormal(np.log(PHYS_DEFAULT), sigma, (n_phys_samples, 7)).astype(np.float32),
        PHYS_LO,
        PHYS_HI,
    )
    phys_samples = np.concatenate([PHYS_DEFAULT.reshape(1, 7), phys_samples], axis=0)

    x = np.tile(shot["est_x"].reshape(1, 4), (len(phys_samples), 1))
    losses = np.array(
        fn_flex(
            jnp.array(shot["prev_slots"]),
            jnp.array(shot["prev_mask"]),
            jnp.array(shot["thrower_block"]),
            jnp.array(shot["tgt0"]),
            jnp.array(shot["tgt0m"]),
            jnp.array(shot["tgt1"]),
            jnp.array(shot["tgt1m"]),
            jnp.array(x),
            jnp.array(phys_samples),
        ),
        dtype=np.float32,
    )
    return phys_samples[int(np.argmin(losses))]


def main():
    ap = argparse.ArgumentParser(description="Render GIFs for search-physics inverse solutions.")
    ap.add_argument("--input-csv", required=True, help="CSV with improved shots and est_* params")
    ap.add_argument("--reference-csv", required=True, help="Original benchmark CSV used to recover row indices")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--duration", type=float, default=0.08)
    ap.add_argument("--loss-variant", type=str, default="optimal")
    ap.add_argument("--phys-sigma", type=float, default=0.20)
    ap.add_argument("--n-phys-samples", type=int, default=20)
    args = ap.parse_args()

    df = pd.read_csv(args.input_csv, low_memory=False)
    ref = pd.read_csv(args.reference_csv, low_memory=False)
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    curl_params = CurlingParams(**SIM_PARAMS)
    fn_flex = build_batched_hard_loss_by_block_flex(curl_params, loss_variant=args.loss_variant)

    manifest_rows = []
    for rank, (_, row) in enumerate(df.iterrows(), start=1):
        key_mask = np.ones(len(ref), dtype=bool)
        for k in SHOT_KEY:
            key_mask &= (ref[k].astype(int).to_numpy() == int(row[k]))
        if not key_mask.any():
            raise SystemExit(f"Could not find reference row for shot {[int(row[k]) for k in SHOT_KEY]}")
        orig_idx = int(np.nonzero(key_mask)[0][0])

        shot = load_shot(row)
        best_phys = _recover_best_phys(orig_idx, shot, fn_flex, args.phys_sigma, args.n_phys_samples)
        traj = np.asarray(
            rollout_positions_until_stop_flex(
                curl_params,
                make_initial_state(
                    curl_params,
                    jnp.asarray(shot["prev_slots"], dtype=jnp.float32),
                    float(shot["est_x"][1]),
                    float(shot["est_x"][0]),
                    float(shot["est_x"][2]),
                    float(shot["est_x"][3]),
                ),
                jnp.asarray(best_phys, dtype=jnp.float32),
            )[0]
        )

        traj_full = np.full((traj.shape[0], 12, 2), np.nan, dtype=np.float32)
        for t in range(traj.shape[0]):
            traj_full[t] = _pred_frame_to_slots(traj[t], shot)

        next_obs = _extract_state_from_row(row, "next")
        frame_ids = list(range(0, traj_full.shape[0], max(1, int(args.stride))))
        if frame_ids[-1] != traj_full.shape[0] - 1:
            frame_ids.append(traj_full.shape[0] - 1)

        stem = f"{rank:03d}_{int(row['CompetitionID'])}_{int(row['SessionID'])}_{int(row['GameID'])}_{int(row['EndID'])}_{int(row['ShotID'])}_loss_{float(row['hard_loss_refine']):.3f}"
        out_path = out_dir / f"{stem}.gif"
        with imageio.get_writer(out_path, mode="I", duration=float(args.duration)) as writer:
            for frame_idx in frame_ids:
                pred_now = traj_full[frame_idx]
                metrics = _difference_metrics(pred_now, next_obs)

                fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12, 6), dpi=140)
                _plot_house(ax_left)
                _plot_house(ax_right)
                ax_left.set_title(f"Recovered search-physics inverse\nframe {frame_idx + 1}/{traj_full.shape[0]}")
                _draw_trails(ax_left, traj_full, frame_idx, shot["thrower_block"])
                _draw_state(ax_left, pred_now, shot["thrower_block"], alpha=0.95)

                ax_right.set_title(
                    "Predicted vs actual next state\n"
                    f"matched={metrics['matched']} pred-only={metrics['pred_only']} actual-only={metrics['actual_only']} "
                    f"RMSE={metrics['rmse']:.3f}m"
                )
                _draw_state(ax_right, pred_now, shot["thrower_block"], alpha=0.92)
                _draw_state(ax_right, next_obs, shot["thrower_block"], alpha=0.80, marker="x", s=100.0)
                for slot_idx in np.nonzero(metrics["both_mask"])[0]:
                    pred_xy = pred_now[slot_idx]
                    actual_xy = next_obs[slot_idx]
                    ax_right.plot(
                        [pred_xy[1], actual_xy[1]],
                        [pred_xy[0], actual_xy[0]],
                        linestyle="--",
                        linewidth=1.1,
                        color="tab:red",
                        alpha=0.65,
                        zorder=3,
                    )

                fig.suptitle(
                    f"shot {int(row['CompetitionID'])},{int(row['SessionID'])},{int(row['GameID'])},{int(row['EndID'])},{int(row['ShotID'])} | "
                    f"loss={float(row['hard_loss_refine']):.3f}\n"
                    f"k_curl={best_phys[0]:.3f} a_linear={best_phys[1]:.3f} gamma_spin={best_phys[2]:.3f} "
                    f"c_damp={best_phys[3]:.1f} c_tangent={best_phys[4]:.1f} mu_tangent={best_phys[5]:.3f} spin_contact={best_phys[6]:.3f}",
                    y=0.98,
                    fontsize=10,
                )
                fig.tight_layout(rect=[0, 0, 1, 0.93])
                writer.append_data(_fig_to_rgb(fig))
                plt.close(fig)

        manifest_rows.append({
            "rank": rank,
            "CompetitionID": int(row["CompetitionID"]),
            "SessionID": int(row["SessionID"]),
            "GameID": int(row["GameID"]),
            "EndID": int(row["EndID"]),
            "ShotID": int(row["ShotID"]),
            "hard_loss_refine": float(row["hard_loss_refine"]),
            "k_curl": float(best_phys[0]),
            "a_linear": float(best_phys[1]),
            "gamma_spin": float(best_phys[2]),
            "c_damp": float(best_phys[3]),
            "c_tangent": float(best_phys[4]),
            "mu_tangent": float(best_phys[5]),
            "spin_contact": float(best_phys[6]),
            "gif": out_path.name,
        })
        print(f"[done] {out_path}")

    pd.DataFrame(manifest_rows).to_csv(out_dir / "gif_manifest.csv", index=False)
    print(f"[done] wrote {len(manifest_rows)} gifs and manifest to {out_dir}")


if __name__ == "__main__":
    main()
