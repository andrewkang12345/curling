#!/usr/bin/env python3

from __future__ import annotations

import argparse
import pathlib
import sys
import re
from typing import List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

THIS_DIR = pathlib.Path(__file__).resolve().parent
sys.path.append(str(THIS_DIR))
sys.path.append(str(THIS_DIR / "inverse"))

import jax.numpy as jnp
from curling_sim_jax import CurlingParams, simulate_from_params  # type: ignore
from sim_presets import CONTACT_MILD_SIM_KWARGS
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


DEFAULT_SHOTS = [
    ("empty_wipeout", "0,8,3,7,22"),
    ("nonempty_worst", "22230015,5,1,3,20"),
    ("nonempty_takeout", "23240026,16,4,2,19"),
    ("survivor_mismatch", "24250026,46,3,1,22"),
]


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


def _draw_trails(ax, traj_full: np.ndarray, thrower_block: int):
    for slot_idx in range(traj_full.shape[1]):
        pts = traj_full[:, slot_idx, :]
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


def _parse_shot(shot: str) -> List[int]:
    vals = [int(x.strip()) for x in shot.split(",")]
    if len(vals) != 5:
        raise ValueError(f"invalid shot key: {shot}")
    return vals


def _raw_state_from_stones_row(row: pd.Series) -> np.ndarray:
    mat = np.full((12, 2), np.nan, dtype=np.float32)
    for i in range(1, 13):
        x = row.get(f"Stone_{i}_x", row.get(f"stone_{i}_x", np.nan))
        y = row.get(f"Stone_{i}_y", row.get(f"stone_{i}_y", np.nan))
        if pd.isna(x) or pd.isna(y):
            continue
        x = float(x)
        y = float(y)
        if x in (0.0, 4095.0) or y in (0.0, 4095.0):
            continue
        xm = (800.0 - y) * 0.003048
        ym = (x - 750.0) * 0.003048
        mat[i - 1, 0] = xm
        mat[i - 1, 1] = ym
    return mat


def _select_row(merged, shot_vals: Sequence[int]):
    mask = np.ones(len(merged), dtype=bool)
    for k, v in zip(SHOT_KEY, shot_vals):
        mask &= (merged[k].astype(int).to_numpy() == int(v))
    if not mask.any():
        raise KeyError(f"shot not found: {shot_vals}")
    return merged.loc[mask].iloc[0]


def _slugify(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(text)).strip("_")


def _build_example_context(shot_row):
    prev_mat = extract_state_from_row(shot_row, "prev")
    next_mat_obs = extract_state_from_row(shot_row, "next")
    _, prev_ids = compact_positions(prev_mat)
    _, next_ids_obs = compact_positions(next_mat_obs)

    obs_throw_slot_id = float(shot_row.get("obs_throw_slot_id", np.nan))
    team_slot_block = float(shot_row.get("team_slot_block", np.nan))
    thrower_block = infer_thrower_block(
        prev_ids=prev_ids,
        next_ids=next_ids_obs,
        obs_throw_slot_id=obs_throw_slot_id,
        team_slot_block=team_slot_block,
    )
    comp, sess, game, end, shotid = [int(shot_row[k]) for k in SHOT_KEY]
    return {
        "shot": f"{comp},{sess},{game},{end},{shotid}",
        "end_key": (comp, sess, game, end),
        "thrower_block": int(thrower_block),
        "hard_loss_refine": float(shot_row.get("hard_loss_refine", np.nan)),
    }


def _make_shot_figure(shot_row, meta: MetaLookup, label: str, out_path: pathlib.Path, curl_params: CurlingParams):
    prev_mat = extract_state_from_row(shot_row, "prev")
    next_mat_obs = extract_state_from_row(shot_row, "next")
    prev_compact, prev_ids = compact_positions(prev_mat)
    _, next_ids_obs = compact_positions(next_mat_obs)
    context = _build_example_context(shot_row)
    thrower_block = context["thrower_block"]
    new_id = choose_new_slot_id(
        prev_ids=prev_ids,
        next_ids=next_ids_obs,
        thrower_block=thrower_block,
        obs_throw_slot_id=obs_throw_slot_id,
    )

    est_params = np.array([shot_row.get(c, np.nan) for c in ["est_speed", "est_angle", "est_spin", "est_y0"]], dtype=np.float32)
    traj_compact = np.asarray(
        simulate_from_params(
            curl_params,
            jnp.asarray(prev_compact, dtype=jnp.float32),
            jnp.asarray(est_params, dtype=jnp.float32),
            dynamic=True,
        )
    )
    pred_final = assign_final_to_slots(traj_compact[-1], prev_ids, new_id)
    traj_full = np.full((traj_compact.shape[0], 12, 2), np.nan, dtype=np.float32)
    for t in range(traj_compact.shape[0]):
        traj_full[t] = assign_final_to_slots(traj_compact[t], prev_ids, new_id)

    metrics = _difference_metrics(pred_final, next_mat_obs)

    comp, sess, game, end, shotid = [int(shot_row[k]) for k in SHOT_KEY]
    team_id = int(shot_row.get("TeamID", -1)) if np.isfinite(shot_row.get("TeamID", np.nan)) else -1
    player_id = shot_row.get("PlayerID", np.nan)
    comp_name = meta.get_comp(comp)
    team_label = meta.get_team(comp, team_id) if team_id >= 0 else f"Team {team_id}"
    player_label = meta.get_player(player_id) if np.isfinite(player_id) else "unknown"

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), dpi=160)
    for ax in axes:
        _plot_house(ax)

    axes[0].set_title("Inferred trajectory + predicted final")
    _draw_state(axes[0], prev_mat, thrower_block, alpha=0.28, s=60.0)
    _draw_trails(axes[0], traj_full, thrower_block)
    _draw_state(axes[0], pred_final, thrower_block, alpha=0.96)

    axes[1].set_title("Actual next state")
    _draw_state(axes[1], next_mat_obs, thrower_block, alpha=0.96)

    axes[2].set_title(
        "Predicted vs actual overlay\n"
        f"matched={metrics['matched']} pred-only={metrics['pred_only']} actual-only={metrics['actual_only']} "
        f"RMSE={metrics['rmse']:.3f}m"
    )
    _draw_state(axes[2], pred_final, thrower_block, alpha=0.92)
    _draw_state(axes[2], next_mat_obs, thrower_block, alpha=0.80, marker="x", s=100.0)
    both = metrics["both_mask"]
    for slot_idx in np.nonzero(both)[0]:
        pred_xy = pred_final[slot_idx]
        actual_xy = next_mat_obs[slot_idx]
        axes[2].plot(
            [pred_xy[1], actual_xy[1]],
            [pred_xy[0], actual_xy[0]],
            linestyle="--",
            linewidth=1.1,
            color="tab:red",
            alpha=0.65,
            zorder=3,
        )

    fig.suptitle(
        f"{label} | {comp_name} | shot {comp},{sess},{game},{end},{shotid}\n"
        f"{team_label} | Player: {player_label} | Task={shot_row.get('Task')} Handle={shot_row.get('Handle')} | "
        f"hard_loss={float(shot_row.get('hard_loss_refine', np.nan)):.6f} | "
        f"v={est_params[0]:.3f} a={est_params[1]:.3f} spin={est_params[2]:.3f} y0={est_params[3]:.3f}",
        y=0.98,
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return {
        **context,
        "label": label,
        "rmse_slots_m": metrics["rmse"],
        "matched": metrics["matched"],
        "pred_only": metrics["pred_only"],
        "actual_only": metrics["actual_only"],
        "png": str(out_path),
    }


def _make_end_timeline(stones_df: pd.DataFrame, ex: dict, out_path: pathlib.Path):
    comp, sess, game, end = ex["end_key"]
    shotid = int(ex["shot"].split(",")[-1])
    end_rows = stones_df[
        (stones_df["CompetitionID"] == comp)
        & (stones_df["SessionID"] == sess)
        & (stones_df["GameID"] == game)
        & (stones_df["EndID"] == end)
    ].sort_values("ShotID")
    if end_rows.empty:
        return

    fig, axes = plt.subplots(2, 5, figsize=(18, 7.5), dpi=150)
    axes = axes.ravel()
    for ax, (_, row) in zip(axes, end_rows.iterrows()):
        _plot_house(ax)
        state = _raw_state_from_stones_row(row)
        _draw_state(ax, state, ex["thrower_block"], alpha=0.96, s=70.0)
        sid = int(row["ShotID"])
        title = f"Shot {sid}"
        if sid == shotid:
            title += "  focal"
            for spine in ax.spines.values():
                spine.set_edgecolor("tab:orange")
                spine.set_linewidth(2.5)
        ax.set_title(title, fontsize=10)
    for ax in axes[len(end_rows):]:
        ax.axis("off")
    fig.suptitle(f"Actual end states | shot {ex['shot']} | {ex['label']}", y=0.98, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _make_contact_sheet(examples: List[dict], out_path: pathlib.Path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=160)
    axes = axes.ravel()
    for ax, ex in zip(axes, examples):
        img = plt.imread(ex["png"])
        ax.imshow(img)
        ax.set_title(
            f"{ex['label']}\nshot {ex['shot']} | hard={ex['hard_loss_refine']:.3f} | rmse={ex['rmse_slots_m']:.3f}m",
            fontsize=10,
        )
        ax.axis("off")
    for ax in axes[len(examples):]:
        ax.axis("off")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Render representative bad inverse solutions.")
    ap.add_argument("--stones-csv", type=str, default="2026/Stones.csv")
    ap.add_argument(
        "--inverse-glob",
        type=str,
        default="inverse_current/stones_with_estimates.chunk*.csv",
    )
    ap.add_argument("--out-dir", type=str, default="bad_inverse_viz")
    ap.add_argument("--competition-csv", type=str, default="2026/Competition.csv")
    ap.add_argument("--teams-csv", type=str, default="2026/Teams.csv")
    ap.add_argument("--competitors-csv", type=str, default="2026/Competitors.csv")
    ap.add_argument("--players-csv", type=str, default="")
    ap.add_argument("--auto-bad-threshold", type=float, default=None, help="If set, render all rows with hard_loss_refine above this threshold.")
    ap.add_argument("--timeline-only", action="store_true", help="Only render actual end timelines, not the per-shot inverse diagnostic panels.")
    ap.add_argument("--sim-refine-dt", type=float, default=0.02)
    ap.add_argument("--sim-refine-substeps", type=int, default=2)
    ap.add_argument("--sim-refine-k-penalty", type=float, default=2.5e4)
    ap.add_argument("--sim-c-damp", type=float, default=CONTACT_MILD_SIM_KWARGS["c_damp"])
    ap.add_argument("--sim-c-damp-sep-frac", type=float, default=CONTACT_MILD_SIM_KWARGS["c_damp_sep_frac"])
    ap.add_argument("--sim-c-tangent", type=float, default=CONTACT_MILD_SIM_KWARGS["c_tangent"])
    ap.add_argument("--sim-mu-tangent", type=float, default=CONTACT_MILD_SIM_KWARGS["mu_tangent"])
    ap.add_argument("--sim-spin-contact", type=float, default=CONTACT_MILD_SIM_KWARGS["spin_contact"])
    ap.add_argument("--sim-k-curl", type=float, default=CONTACT_MILD_SIM_KWARGS["k_curl"])
    ap.add_argument("--sim-a-linear", type=float, default=CONTACT_MILD_SIM_KWARGS["a_linear"])
    ap.add_argument("--sim-gamma-spin", type=float, default=CONTACT_MILD_SIM_KWARGS["gamma_spin"])
    ap.add_argument(
        "--shot",
        action="append",
        default=[],
        help='Optional "label=comp,sess,game,end,shot". Can be passed multiple times.',
    )
    args = ap.parse_args()

    shots = []
    if args.auto_bad_threshold is not None:
        shots = []
    elif args.shot:
        for spec in args.shot:
            if "=" not in spec:
                raise SystemExit(f"--shot must be label=comp,sess,game,end,shot, got: {spec}")
            label, shot = spec.split("=", 1)
            shots.append((label.strip(), shot.strip()))
    else:
        shots = list(DEFAULT_SHOTS)

    meta = MetaLookup.from_files(
        competition_csv=args.competition_csv,
        teams_csv=args.teams_csv,
        competitors_csv=args.competitors_csv,
        players_csv=args.players_csv,
    )
    merged = prepare_merged(args.stones_csv, args.inverse_glob)
    stones_df = pd.read_csv(args.stones_csv)
    curl_params = CurlingParams(
        dt=float(args.sim_refine_dt),
        substeps=int(args.sim_refine_substeps),
        k_penalty=float(args.sim_refine_k_penalty),
        c_damp=float(args.sim_c_damp),
        c_damp_sep_frac=float(args.sim_c_damp_sep_frac),
        c_tangent=float(args.sim_c_tangent),
        mu_tangent=float(args.sim_mu_tangent),
        spin_contact=float(args.sim_spin_contact),
        k_curl=float(args.sim_k_curl),
        a_linear=float(args.sim_a_linear),
        gamma_spin=float(args.sim_gamma_spin),
    )

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.auto_bad_threshold is not None:
        bad = merged[merged["hard_loss_refine"] > float(args.auto_bad_threshold)].copy()
        bad = bad.sort_values(["hard_loss_refine", "CompetitionID", "SessionID", "GameID", "EndID", "ShotID"], ascending=[False, True, True, True, True, True]).reset_index(drop=True)
        shots = []
        manifest_rows = []
        for idx, row in bad.iterrows():
            key = ",".join(str(int(row[k])) for k in SHOT_KEY)
            label = _slugify(
                f"{idx+1:03d}_{int(row['CompetitionID'])}_{int(row['SessionID'])}_{int(row['GameID'])}_{int(row['EndID'])}_{int(row['ShotID'])}_loss_{float(row['hard_loss_refine']):.3f}"
            )
            shots.append((label, key))
            manifest_rows.append(
                {
                    "label": label,
                    "shot": key,
                    "hard_loss_refine": float(row["hard_loss_refine"]),
                    "next_in_bounds_N": int(row.get("next_in_bounds_N", -1)),
                    "next_total_N": int(row.get("next_total_N", -1)),
                }
            )
        pd.DataFrame(manifest_rows).to_csv(out_dir / "manifest.csv", index=False)

    examples = []
    for label, shot in shots:
        shot_vals = _parse_shot(shot)
        shot_row = _select_row(merged, shot_vals)
        ex = _build_example_context(shot_row)
        ex["label"] = label
        if not args.timeline_only:
            out_path = out_dir / f"{label}.png"
            ex = _make_shot_figure(shot_row, meta, label, out_path, curl_params)
            print(f"[done] {label}: {out_path}")
        examples.append(ex)
        timeline_path = out_dir / f"{label}_end_timeline.png"
        _make_end_timeline(stones_df, ex, timeline_path)
        print(f"[done] {label} timeline: {timeline_path}")

    if examples and not args.timeline_only:
        sheet_examples = examples[:4]
        sheet_path = out_dir / "contact_sheet.png"
        _make_contact_sheet(sheet_examples, sheet_path)
        print(f"[done] contact sheet: {sheet_path}")


if __name__ == "__main__":
    main()
