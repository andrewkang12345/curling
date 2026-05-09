#!/usr/bin/env python3

from __future__ import annotations

import argparse
import pathlib
import sys

import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

THIS_DIR = pathlib.Path(__file__).resolve().parent
sys.path.append(str(THIS_DIR))
sys.path.append(str(THIS_DIR / "inverse"))

import jax.numpy as jnp
from curling_sim_jax import CurlingParams, simulate_from_params  # type: ignore
from visualize import (  # type: ignore
    SHOT_KEY,
    MAX_X,
    MAX_Y,
    MIN_X,
    MIN_Y,
    MetaLookup,
    assign_final_to_slots,
    choose_new_slot_id,
    compact_positions,
    extract_state_from_row,
    infer_thrower_block,
    prepare_merged,
)


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
        fc = face[i]
        ec = edge[i]
        if marker == "x":
            ax.scatter(y_m, x_m, s=s, marker=marker, color="tab:red", linewidths=1.7, alpha=alpha, zorder=5)
        else:
            ax.scatter(
                y_m,
                x_m,
                s=s,
                marker=marker,
                facecolors=fc,
                edgecolors=ec,
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
        d = np.linalg.norm(diffs, axis=1)
        rmse = float(np.sqrt(np.mean(np.sum(diffs**2, axis=1))))
        mean_d = float(np.mean(d))
    else:
        rmse = float("nan")
        mean_d = float("nan")
    return {
        "rmse": rmse,
        "mean_d": mean_d,
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


def main():
    ap = argparse.ArgumentParser(description="Render an animated GIF of an inferred throw and its difference from the actual next state.")
    ap.add_argument("--stones-csv", type=str, default="2026/Stones.csv")
    ap.add_argument("--inverse-glob", type=str, default="inverseDataset/stones_with_estimates.chunk*.csv")
    ap.add_argument("--shot", type=str, required=True, help='Shot key "comp,sess,game,end,shot"')
    ap.add_argument("--out", type=str, default="throw_vs_actual.gif")
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--duration", type=float, default=0.08)
    ap.add_argument("--competition-csv", type=str, default="2026/Competition.csv")
    ap.add_argument("--teams-csv", type=str, default="2026/Teams.csv")
    ap.add_argument("--competitors-csv", type=str, default="2026/Competitors.csv")
    ap.add_argument("--players-csv", type=str, default="")
    args = ap.parse_args()

    parts = [int(x.strip()) for x in args.shot.split(",")]
    if len(parts) != 5:
        raise SystemExit("--shot must be comp,sess,game,end,shot")
    shot_key = dict(zip(SHOT_KEY, parts))

    meta = MetaLookup.from_files(
        competition_csv=args.competition_csv,
        teams_csv=args.teams_csv,
        competitors_csv=args.competitors_csv,
        players_csv=args.players_csv,
    )

    merged = prepare_merged(args.stones_csv, args.inverse_glob)
    mask = np.ones(len(merged), dtype=bool)
    for k, v in shot_key.items():
        mask &= (merged[k].astype(int).to_numpy() == int(v))
    if not mask.any():
        raise SystemExit(f"Shot not found: {args.shot}")
    shot_row = merged.loc[mask].iloc[0]

    prev_mat = extract_state_from_row(shot_row, "prev")
    next_mat_obs = extract_state_from_row(shot_row, "next")
    prev_compact, prev_ids = compact_positions(prev_mat)
    _, next_ids_obs = compact_positions(next_mat_obs)

    obs_throw_slot_id = float(shot_row.get("obs_throw_slot_id", np.nan))
    team_slot_block = float(shot_row.get("team_slot_block", np.nan))
    thrower_block = infer_thrower_block(
        prev_ids=prev_ids,
        next_ids=next_ids_obs,
        obs_throw_slot_id=obs_throw_slot_id,
        team_slot_block=team_slot_block,
    )
    new_id = choose_new_slot_id(
        prev_ids=prev_ids,
        next_ids=next_ids_obs,
        thrower_block=thrower_block,
        obs_throw_slot_id=obs_throw_slot_id,
    )

    est_params = np.array([shot_row.get(c, np.nan) for c in ["est_speed", "est_angle", "est_spin", "est_y0"]], dtype=np.float32)
    if not np.all(np.isfinite(est_params)):
        raise SystemExit("Shot has non-finite inverse parameters.")

    curl_params = CurlingParams(dt=0.02, substeps=2, k_penalty=2.5e4, c_damp=220.0, k_curl=0.10)
    traj_compact = np.asarray(
        simulate_from_params(
            curl_params,
            jnp.asarray(prev_compact, dtype=jnp.float32),
            jnp.asarray(est_params, dtype=jnp.float32),
            dynamic=True,
        )
    )

    traj_full = np.full((traj_compact.shape[0], 12, 2), np.nan, dtype=np.float32)
    for t in range(traj_compact.shape[0]):
        traj_full[t] = assign_final_to_slots(traj_compact[t], prev_ids, new_id)

    frame_ids = list(range(0, traj_full.shape[0], max(1, int(args.stride))))
    if frame_ids[-1] != traj_full.shape[0] - 1:
        frame_ids.append(traj_full.shape[0] - 1)

    comp, sess, game, end, shotid = [int(shot_row[k]) for k in SHOT_KEY]
    team_id = int(shot_row.get("TeamID", -1)) if np.isfinite(shot_row.get("TeamID", np.nan)) else -1
    player_id = shot_row.get("PlayerID", np.nan)
    comp_name = meta.get_comp(comp)
    team_label = meta.get_team(comp, team_id) if team_id >= 0 else f"Team {team_id}"
    player_label = meta.get_player(player_id) if np.isfinite(player_id) else "unknown"

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(out_path, mode="I", duration=float(args.duration)) as writer:
        for frame_idx in frame_ids:
            pred_now = traj_full[frame_idx]
            metrics = _difference_metrics(pred_now, next_mat_obs)

            fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12, 6), dpi=140)
            _plot_house(ax_left)
            _plot_house(ax_right)

            ax_left.set_title(f"Inferred throw\nframe {frame_idx + 1}/{traj_full.shape[0]}")
            _draw_trails(ax_left, traj_full, frame_idx, thrower_block)
            _draw_state(ax_left, pred_now, thrower_block, alpha=0.95)

            ax_right.set_title(
                "Difference vs actual next state\n"
                f"matched={metrics['matched']}  pred-only={metrics['pred_only']}  actual-only={metrics['actual_only']}  "
                f"RMSE={metrics['rmse']:.3f}m"
            )
            _draw_state(ax_right, pred_now, thrower_block, alpha=0.92)
            _draw_state(ax_right, next_mat_obs, thrower_block, alpha=0.80, marker="x", s=100.0)

            both = metrics["both_mask"]
            for slot_idx in np.nonzero(both)[0]:
                pred_xy = pred_now[slot_idx]
                actual_xy = next_mat_obs[slot_idx]
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
                f"{comp_name} | shot {comp},{sess},{game},{end},{shotid}\n"
                f"{team_label} | Player: {player_label} | Task={shot_row.get('Task')} Handle={shot_row.get('Handle')} | "
                f"hard_loss={float(shot_row.get('hard_loss_refine', np.nan)):.6f}",
                y=0.98,
                fontsize=11,
            )
            fig.tight_layout(rect=[0, 0, 1, 0.93])
            writer.append_data(_fig_to_rgb(fig))
            plt.close(fig)

    print(f"[done] wrote gif to: {out_path}")
    print(f"[info] frames={len(frame_ids)} raw_steps={traj_full.shape[0]}")


if __name__ == "__main__":
    main()
