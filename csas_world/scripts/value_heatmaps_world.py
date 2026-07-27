#!/usr/bin/env python3
"""Value heatmaps using OUR trained WorldModel's value head (e.g. EXP-019), not the human prior.

For each example board state, grid candidate post-stone positions around the button, set the
to-be-thrown stone slot to each grid point, and ask the value head V(s, c) what end-margin it
predicts. We plot the value DIFFERENCE relative to the pre-state value, so colour shows how each
candidate landing improves or worsens the position for the throwing team.

    source scripts/setup_gpu.sh && python3 scripts/value_heatmaps_world.py \
        --champion checkpoints/csas_world/exp_019_consolidate/last.pt \
        --out artifacts/figures/value_heatmaps_exp019_world
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import world  # noqa: F401  (GNN env bootstrap)
from world import env_bridge
from world.config import Config

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402

from csas.common import NUM_STONES, POS_MAX, in_play_raw  # noqa: E402
from csas.make_value_heatmaps import BUTTON_RAW, M_PER_RAW, STONE_RADIUS_M, _draw_house  # noqa: E402


def _xy_m(state_norm):
    raw = np.asarray(state_norm, np.float32).reshape(NUM_STONES, 2) * POS_MAX
    return (raw - BUTTON_RAW) * M_PER_RAW, in_play_raw(raw)


def _next_free_slot(state_norm: np.ndarray, block: int) -> int:
    raw = np.asarray(state_norm, np.float32).reshape(NUM_STONES, 2) * POS_MAX
    live = in_play_raw(raw)
    lo, hi = (0, 6) if block == 0 else (6, 12)
    for s in range(lo, hi):
        if not live[s]:
            return s
    return lo                                                # all six placed: overwrite first


def value_heatmap(player, x_norm: np.ndarray, cond: np.ndarray, grid_n: int = 60, extent_m: float = 2.2):
    """Grid candidate landing positions around the button; return (xs, ys, value_diff_grid, base_value).

    Each grid cell = a candidate post-state where the next-thrown stone lands there; cells where
    the stone would land out of the sheet or atop another stone are excluded (NaN), so the colour
    map only shows physically-legal landings."""
    block = int(round(float(cond[2])))
    slot = _next_free_slot(x_norm, block)
    xs_m = np.linspace(-extent_m, extent_m, grid_n, dtype=np.float32)
    ys_m = np.linspace(-extent_m, extent_m, grid_n, dtype=np.float32)
    xx, yy = np.meshgrid(xs_m, ys_m)
    pts_m = np.stack([xx.ravel(), yy.ravel()], axis=1)
    pts_raw = BUTTON_RAW + pts_m / M_PER_RAW                                # (G,2)

    # base board (raw); for each grid cell, copy the board and overwrite slot with the candidate.
    base_raw = np.asarray(x_norm, np.float32).reshape(NUM_STONES, 2) * POS_MAX
    G = len(pts_raw)
    boards = np.repeat(base_raw[None], G, axis=0)                            # (G,12,2)
    boards[:, slot, :] = pts_raw
    boards_norm = (boards.reshape(G, -1) / POS_MAX).astype(np.float32)
    posts_cond = env_bridge.next_condition(cond, 10)                          # value of post-state, eval as opponent-to-move

    # mask: candidate must (a) be on the sheet, (b) not collide with another live stone
    live_other = in_play_raw(base_raw) & (np.arange(NUM_STONES) != slot)
    other_pts = (base_raw[live_other] - BUTTON_RAW) * M_PER_RAW              # in plot metres
    on_sheet = (np.abs(pts_m[:, 0]) < extent_m + 0.5) & (pts_m[:, 1] < 4.0) & (pts_m[:, 1] > -2.5)
    # collision: ANY existing live stone within 2*r
    if len(other_pts):
        d = np.linalg.norm(pts_m[:, None, :] - other_pts[None, :, :], axis=-1)
        no_collide = (d > 2 * STONE_RADIUS_M).all(axis=1)
    else:
        no_collide = np.ones(G, dtype=bool)
    valid = on_sheet & no_collide

    # base value (pre-throw board, to-move team's perspective via cond)
    base_val = float(player._value_fn(x_norm[None], cond)[0])
    # all candidates' post-values from opponent-to-move side, then negate to get throwing-team-perspective
    post_vals = -player._value_fn(boards_norm, posts_cond)                    # (G,)
    diff = post_vals - base_val
    diff_grid = np.full(G, np.nan, dtype=np.float32)
    diff_grid[valid] = diff[valid]
    return xs_m, ys_m, diff_grid.reshape(grid_n, grid_n), base_val, slot


def example_states(cfg):
    """Same example set as the policy heatmaps script: 3 openings + 3 mid-game horizons."""
    from world.preplaced import board_norm
    from world.eval.head_to_head import build_h2h_roots
    out = []
    for mode in ("standard", "pp_left", "pp_right"):
        out.append((f"opening_{mode}", board_norm(mode, 1), np.array([0.0, 0.0, 0.0], np.float32)))
    for h in (6, 4, 2):
        roots = build_h2h_roots(cfg.paths.csas_v3_root, h, 8, split="val", seed=h)
        if roots:
            r = roots[0]
            out.append((f"h{h:02d}_midgame", r.x.copy(), r.c.copy()))
    return out


def render(label: str, x_norm, cond, xs, ys, diff_grid, base_val, slot, out_dir: str):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.2, 6.6))
    _draw_house(ax)
    # heat: value difference vs pre-throw value (positive = better for the throwing team)
    vmax = float(np.nanmax(np.abs(diff_grid))) if np.isfinite(diff_grid).any() else 1.0
    im = ax.imshow(diff_grid, extent=(xs[0], xs[-1], ys[0], ys[-1]), origin="lower",
                   cmap="RdBu_r", vmin=-vmax, vmax=+vmax, alpha=0.85, zorder=1)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(f"value Δ vs pre-throw (= {base_val:+.2f})", fontsize=8)

    xy, live = _xy_m(x_norm)
    block = int(round(float(cond[2])))
    for s_ in range(NUM_STONES):
        if not live[s_]:
            continue
        team_to_move = (s_ // 6) == block
        ax.add_patch(Circle((xy[s_, 0], xy[s_, 1]), STONE_RADIUS_M,
                            facecolor=("#1d4ed8" if team_to_move else "#dc2626"),
                            edgecolor="white", lw=1.4, zorder=5))
    ax.set_xlim(xs[0] - 0.1, xs[-1] + 0.1); ax.set_ylim(ys[0] - 0.1, ys[-1] + 0.1)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(f"{label} -- world value Δ if next stone landed here\n(to-move=blue, opp=red)",
                 fontsize=10)
    ax.text(0.5, -0.02, f"thrown slot: {slot}   button=centre   +y = front", transform=ax.transAxes,
            ha="center", va="top", fontsize=7.5, color="0.3")
    fig.tight_layout(); fig.savefig(out / f"{label}.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    valid = int(np.isfinite(diff_grid).sum())
    print(f"[value-heatmap] {label}: {valid}/{diff_grid.size} valid cells  range=[{np.nanmin(diff_grid):+.3f},{np.nanmax(diff_grid):+.3f}]  -> {out/label}.png",
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--champion", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--grid", type=int, default=60)
    ap.add_argument("--extent", type=float, default=2.2)
    args = ap.parse_args()
    import torch
    from world.eval.head_to_head import WorldPlayer
    cfg = Config()
    dev = torch.device(args.device)
    env_bridge.warm_jax()
    P = WorldPlayer(args.champion, dev, name="exp019_value")
    for label, x, c in example_states(cfg):
        xs, ys, diff_g, base_v, slot = value_heatmap(P, x, c, args.grid, args.extent)
        render(label, x, c, xs, ys, diff_g, base_v, slot, args.out)
    print("VALUE_HEATMAPS_WORLD_DONE", flush=True)


if __name__ == "__main__":
    main()
