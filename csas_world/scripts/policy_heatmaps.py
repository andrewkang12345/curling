#!/usr/bin/env python3
"""Policy proposal heatmaps: for example board states, show WHERE the policy's proposed throws
would land — colored by density — *if the ice were empty* (each plan simulated against an empty
board, isolating its intended target from collisions).

For each example state we sample N actions from the policy (which DOES see the real board), then
simulate each action on an EMPTY board via the authoritative JAX sim and take the thrown stone's
resting point. The 2D density of those landing points = the policy's intended-target distribution.
The actual board stones + house are overlaid so you can see what the policy is reacting to.

    source scripts/setup_gpu.sh && python3 scripts/policy_heatmaps.py \
        --champion checkpoints/csas_world/exp_019_consolidate/last.pt --out artifacts/figures/policy_heatmaps_exp019
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import world  # noqa: F401  (bootstrap: GNN env + csas path)
from world import env_bridge
from world.config import Config

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402

from csas.common import NUM_STONES, POS_MAX, in_play_raw  # noqa: E402
from csas.make_value_heatmaps import BUTTON_RAW, M_PER_RAW, STONE_RADIUS_M, _draw_house  # noqa: E402
EMPTY_NORM = np.ones(NUM_STONES * 2, dtype=np.float32)   # all slots dead (POS_MAX/POS_MAX) = empty ice


def _xy_m(state_norm):
    raw = np.asarray(state_norm, np.float32).reshape(NUM_STONES, 2) * POS_MAX
    return (raw - BUTTON_RAW) * M_PER_RAW, in_play_raw(raw)


def landing_points(player, x, c, n):
    """Sample n policy actions at (x,c); return their EMPTY-BOARD landing points in plot-metres.

    Uses the fast batched final-position simulator (env_bridge.simulate, what collection runs) on an
    EMPTY board, then reads the single resulting stone's resting position — far cheaper than recording
    each throw's full dynamic trajectory (we only need where it lands)."""
    acts = np.asarray(player._sample_fn(x, c, n), dtype=np.float32).reshape(-1, 4)
    posts = env_bridge.simulate(EMPTY_NORM, c, acts)          # (N,24) empty-board post-throw boards
    pts = []
    for ps in posts:
        raw = np.asarray(ps, np.float32).reshape(NUM_STONES, 2) * POS_MAX
        live = in_play_raw(raw)
        if live.any():                                       # the one thrown stone (0 if it ran out of play)
            pts.append((raw[np.where(live)[0][0]] - BUTTON_RAW) * M_PER_RAW)
    return np.array(pts, dtype=np.float32), acts


def example_states(cfg, n_each=1):
    """A spread of example board states (label, x_norm, c)."""
    from world.preplaced import board_norm, PREPLACED_SHOTS_IN_END  # noqa: F401
    from world.eval.head_to_head import build_h2h_roots
    out = []
    # 1) pre-placed openings (h10) — the three modes, thrower on block 0
    for mode in ("standard", "pp_left", "pp_right"):
        out.append((f"opening_{mode}", board_norm(mode, 1), np.array([0.0, 0.0, 0.0], np.float32)))
    # 2) mid-game human states at a few horizons (real boards w/ several stones)
    for h in (6, 4, 2):
        roots = build_h2h_roots(cfg.paths.csas_v3_root, h, 8, split="val", seed=h)
        if roots:
            r = roots[0]
            out.append((f"h{h:02d}_midgame", r.x.copy(), r.c.copy()))
    return out


def render(label, x, c, pts, out_dir, n_sampled):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.2, 6.6))
    _draw_house(ax)
    # heatmap of landing density (empty-ice target of each proposed plan)
    if len(pts):
        hb = ax.hexbin(pts[:, 0], pts[:, 1], gridsize=44, cmap="inferno", mincnt=1,
                       extent=(-2.5, 2.5, -2.5, 4.5), zorder=1, alpha=0.92)
        cb = fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("policy plans landing here (empty ice)", fontsize=8)
    # overlay the ACTUAL board stones the policy is reacting to
    xy, live = _xy_m(x)
    block = int(round(float(c[2])))
    for slot in range(NUM_STONES):
        if not live[slot]:
            continue
        team_to_move = (slot // 6) == block
        ax.add_patch(Circle((xy[slot, 0], xy[slot, 1]), STONE_RADIUS_M,
                            facecolor=("#1d4ed8" if team_to_move else "#dc2626"),
                            edgecolor="white", lw=1.4, zorder=5))
    ax.set_xlim(-2.5, 2.5); ax.set_ylim(-2.5, 4.5); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(f"{label} — policy proposal landing density\n(to-move=blue, opp=red; empty-ice targets)",
                 fontsize=10)
    ax.text(0.5, -0.02, f"{n_sampled} sampled plans   button=center   +y = front (guard zone)",
            transform=ax.transAxes, ha="center", va="top", fontsize=7.5, color="0.3")
    fig.tight_layout(); fig.savefig(out / f"{label}.png", dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"[heatmap] {label}: {len(pts)}/{n_sampled} landings -> {out/label}.png", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--champion", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n-samples", type=int, default=3000)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--std-scale", type=float, default=1.0)
    args = ap.parse_args()

    import torch
    from world.eval.head_to_head import WorldPlayer
    cfg = Config()
    dev = torch.device(args.device)
    env_bridge.warm_jax()
    P = WorldPlayer(args.champion, dev, name="policy",
                    temperature=args.temperature, std_scale=args.std_scale)
    for label, x, c in example_states(cfg):
        pts, _ = landing_points(P, x, c, args.n_samples)
        render(label, x, c, pts, args.out, args.n_samples)
    print("POLICY_HEATMAPS_DONE", flush=True)


if __name__ == "__main__":
    main()
