#!/usr/bin/env python3
"""Visualize champion self-play ends from the canonical pre-placed roots.

az_v14d plays BOTH teams with its deployed selection (48 candidates, k=8 robust);
execution is the INTENDED shot (no noise) so decisions are judged cleanly. One
PNG per throw (thrown stone's authoritative trajectory dotted, champion's value
annotated) + an animated GIF per game, copied into arena/static/selfplay/ so
they're viewable at http://<host>/static/selfplay/.

    python3 scripts/viz_selfplay_v14d.py --modes standard,pp_left,pp_right --games-per-mode 2
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")
import world  # noqa: F401
from world import env_bridge
from world.eval.head_to_head import WorldPlayer
from world.preplaced import board_norm
from world.search.noise import make_noise

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402

from csas.common import NUM_STONES, POS_MAX, in_play_raw  # noqa: E402
from csas.make_value_heatmaps import BUTTON_RAW, M_PER_RAW, STONE_RADIUS_M, _draw_house  # noqa: E402
from csas.search import _new_slot  # noqa: E402
from csas.visualize_policy_multi_action_samples import _trajectory_m_for_actions  # noqa: E402

CKPT = "checkpoints/csas_world/az_v14d/best.pt"
TEAM_C = {0: "#dc2626", 1: "#eab308"}   # block0 red, block1 yellow (arena colors)

ap = argparse.ArgumentParser()
ap.add_argument("--modes", default="standard,pp_left,pp_right")
ap.add_argument("--games-per-mode", type=int, default=2)
ap.add_argument("--sel-noise", type=int, default=8)
ap.add_argument("--out", default="artifacts/figures/selfplay_v14d")
ap.add_argument("--publish", default="arena/static/selfplay")
args = ap.parse_args()


def _xy_m(state_norm):
    raw = np.asarray(state_norm, np.float32).reshape(NUM_STONES, 2) * POS_MAX
    return (raw - BUTTON_RAW) * M_PER_RAW, in_play_raw(raw)


def play(seed, mode, first_block):
    import torch
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = torch.device("cuda:0")
    P = WorldPlayer(CKPT, dev, n_candidates=48, name="az_v14d",
                    noise=make_noise("/mnt/data/curling2/csas_v3/configs/noise/v2_fullsheet.json",
                                     seed=seed), sel_noise_samples=args.sel_noise)
    guard_slot = 1 if first_block == 0 else 7
    x0 = board_norm(mode, guard_slot)
    c0 = np.asarray([0.0, 0.0, float(first_block)], dtype=np.float32)
    state, cond, hh = x0.copy(), c0.copy(), 10
    frames = [{"state": state.copy(), "slot": None, "traj": None, "k": 0, "v0": None,
               "block": None, "illegal": False}]
    k = 0
    while hh >= 1:
        block = int(round(float(cond[2])))
        intended = np.asarray(P.select_intended(state, cond, hh, 10, block), np.float32)
        slot = int(_new_slot(state.reshape(NUM_STONES, 2) * POS_MAX, float(block)))
        traj = _trajectory_m_for_actions(state.reshape(NUM_STONES, 2) * POS_MAX, intended[None])
        post, illegal = env_bridge.apply_legality(
            state, env_bridge.simulate_one(state, cond, intended)[None], hh, cond)
        post = post[0]
        nc = env_bridge.next_condition(cond, 10)
        if hh == 1:
            v0 = float(env_bridge.score_end(post, 0))
        else:
            v = float(P._value_fn(post[None], nc)[0])
            v0 = -v if int(round(nc[2])) != 0 else v   # block-0 perspective
        k += 1
        frames.append({"state": post.copy(), "slot": slot, "traj": traj[0] if traj else None,
                       "k": k, "v0": v0, "block": block, "illegal": bool(illegal[0])})
        state, cond, hh = post, nc, hh - 1
    return frames, float(env_bridge.score_end(state, 0))


def render(frames, score0, mode, first_block, tag, out_dir, publish_dir):
    out = Path(out_dir) / tag
    out.mkdir(parents=True, exist_ok=True)
    xs, ys = [-2.4, 2.4], [-2.3, 2.3]
    for f in frames:
        if f["traj"] is not None and len(f["traj"]):
            xs += list(f["traj"][:, 0]); ys += list(f["traj"][:, 1])
        xy, live = _xy_m(f["state"])
        if live.any():
            xs += list(xy[live, 0]); ys += list(xy[live, 1])
    xlim = (max(min(xs) - 0.3, -3.2), min(max(xs) + 0.3, 3.2))
    ylim = (max(min(ys) - 0.3, -7.5), min(max(ys) + 0.3, 2.6))
    pngs = []
    for f in frames:
        fig, ax = plt.subplots(figsize=(4.6, 6.2))
        _draw_house(ax)
        xy, live = _xy_m(f["state"])
        for slot in range(NUM_STONES):
            if not live[slot]:
                continue
            ax.add_patch(Circle((xy[slot, 0], xy[slot, 1]), STONE_RADIUS_M,
                                facecolor=TEAM_C[slot // 6],
                                edgecolor="#f59e0b" if slot == f["slot"] else "0.15",
                                lw=2.4 if slot == f["slot"] else 1.0,
                                zorder=4 if slot == f["slot"] else 3))
        if f["traj"] is not None and len(f["traj"]):
            ax.plot(f["traj"][:, 0], f["traj"][:, 1], ":", color=TEAM_C[f["block"]],
                    lw=1.8, alpha=0.9, zorder=2)
        ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_aspect("equal"); ax.axis("off")
        hammer_team = "yellow" if first_block == 0 else "red"
        if f["k"] == 0:
            t = f"{mode} — pre-placement (hammer: {hammer_team})"
        else:
            who = "red" if f["block"] == 0 else "yellow"
            t = (f"{mode} throw {f['k']}/10 — {who}"
                 + ("  [FORFEIT]" if f["illegal"] else "")
                 + (f"   V(red)={f['v0']:+.2f}" if f["v0"] is not None else ""))
        ax.set_title(t, fontsize=9.5)
        if f["k"] == 10:
            ax.text(0.5, -0.03, f"FINAL — red {'+' if score0 > 0 else ''}{score0:.0f}",
                    transform=ax.transAxes, ha="center", fontsize=11, color="#111")
        fp = out / f"t{f['k']:02d}.png"
        fig.savefig(fp, dpi=120, bbox_inches="tight")
        plt.close(fig)
        pngs.append(fp)
    # gif
    from PIL import Image
    ims = [Image.open(p).convert("P") for p in pngs]
    gif = Path(out_dir) / f"{tag}.gif"
    ims[0].save(gif, save_all=True, append_images=ims[1:], duration=1400, loop=0)
    pub = Path(publish_dir)
    pub.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copy(gif, pub / gif.name)
    return gif.name


env_bridge.warm_jax()
names = []
for mode in args.modes.split(","):
    for g in range(args.games_per_mode):
        first_block = g % 2            # alternate who has hammer
        seed = 6000 + hash((mode, g)) % 1000
        frames, score0 = play(seed, mode, first_block)
        tag = f"{mode}_g{g}_hammer{'B' if first_block == 0 else 'A'}"
        name = render(frames, score0, mode, first_block, tag, args.out, args.publish)
        names.append((name, score0))
        print(f"[viz] {name}: final red {score0:+.0f}", flush=True)
print("PUBLISHED:", ", ".join(n for n, _ in names))
