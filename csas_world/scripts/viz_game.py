#!/usr/bin/env python3
"""Visualize one full end played by anchor_noisy vs the latest model (az_v3/iter1).

Each model controls one team; throws alternate. The shot is *selected* by each
model's noisy 1-ply decision-value rule (as in head-to-head) and *executed*
deterministically (intended shot, no execution noise) so the trajectories are
clean. One PNG per state, with the just-thrown stone's path drawn as a dotted
line (from release to rest, via the authoritative simulator -- collisions and
curl included).

Run under the GPU-JAX env:
    source scripts/setup_gpu.sh && python3 scripts/viz_game.py --seed 7 --root-idx 0
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import world  # noqa: F401  (bootstrap: sets GNN env + csas path)
from world import env_bridge
from world.eval.head_to_head import WorldPlayer, build_h2h_roots

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402

from csas.common import NUM_STONES, POS_MAX, in_play_raw  # noqa: E402
from csas.make_value_heatmaps import BUTTON_RAW, M_PER_RAW, STONE_RADIUS_M, _draw_house  # noqa: E402
from csas.search import _new_slot  # noqa: E402
from csas.preplaced_value_data import load_preplaced_training_frame, materialize_preplaced  # noqa: E402
from csas.visualize_policy_multi_action_samples import _trajectory_m_for_actions  # noqa: E402

CSAS = "/mnt/data/curling2/csas_v3"
ANCHOR = "checkpoints/csas_world/anchor_noisy/model.pt"
LATEST = "checkpoints/csas_world/az_v3/iter1/model.pt"
ANCHOR_C, LATEST_C = "#dc2626", "#1d4ed8"   # anchor = red, latest = blue
THROWN_EDGE = "#f59e0b"                       # gold outline on the just-thrown stone


def _xy_m(state_norm):
    """Normalised 24-vec -> (12,2) plot-metres (x=lateral, y=-along, button at 0)."""
    raw = np.asarray(state_norm, np.float32).reshape(NUM_STONES, 2) * POS_MAX
    return (raw - BUTTON_RAW) * M_PER_RAW, in_play_raw(raw)


def play_game(seed, root_idx, sel_noise_samples, n_candidates):
    import torch
    torch.manual_seed(seed); np.random.seed(seed)   # reproducible candidate sampling
    dev = torch.device("cuda:0")
    A = WorldPlayer(ANCHOR, dev, n_candidates=n_candidates, name="anchor_noisy", sel_noise_samples=sel_noise_samples)
    B = WorldPlayer(LATEST, dev, n_candidates=n_candidates, name="az_v3/iter1", sel_noise_samples=sel_noise_samples)

    # True start of the end: the canonical mixed-doubles PRE-PLACEMENT (two stones,
    # one per team), shot_norm=0, the first thrower has no hammer (team_order=0).
    dfp = load_preplaced_training_frame()
    dfp = dfp[dfp["mode"] == "standard"].reset_index(drop=True)
    x_all, c_all, _ = materialize_preplaced(dfp)
    i = root_idx % len(dfp)
    x0 = x_all[i].numpy().astype(np.float32)
    c0 = c_all[i].numpy().astype(np.float32)
    shots_in_end, horizon = 10, 10
    persp = int(round(float(c0[2])))      # first thrower's block == player A (anchor, no hammer)

    def vboth(st, eval_cond, sign):
        """Value of state ``st`` from the throwing team's perspective, by BOTH models.
        ``eval_cond``'s block is the team-to-move (the value head's native perspective);
        ``sign``=+1 if the throwing team IS the team-to-move, -1 if it just threw."""
        va = sign * float(A._value_fn(st[None], eval_cond)[0])
        vv = sign * float(B._value_fn(st[None], eval_cond)[0])
        return va, vv

    state, cond, hh = x0.copy(), c0.copy(), horizon
    # pre-placement: the first thrower (== anchor) is on the clock -> no sign flip.
    va0, vv0 = vboth(state, cond, 1.0)
    shots = [{"state": state.copy(), "thrown_slot": None, "traj": None, "thrower": None,
              "a_team": persp, "hammer": None, "illegal": False, "k": 0, "preplacement": True,
              "v_anchor": va0, "v_v3": vv0, "persp_name": "anchor_noisy", "terminal": False}]
    k = 0
    while hh >= 1:
        block = int(round(float(cond[2])))
        plays_a = (block == persp)
        thrower = A if plays_a else B
        name = "anchor_noisy" if plays_a else "az_v3/iter1"
        hammer = int(round(float(cond[1]))) == 1
        intended = np.asarray(thrower.select_intended(state, cond, hh, shots_in_end, block), np.float32)
        slot = int(_new_slot(state.reshape(NUM_STONES, 2) * POS_MAX, float(block)))
        traj_list = _trajectory_m_for_actions(state.reshape(NUM_STONES, 2) * POS_MAX, intended[None])
        traj = traj_list[0] if traj_list else None
        post, illegal = env_bridge.apply_legality(
            state, env_bridge.simulate_one(state, cond, intended)[None], hh, cond)
        post = post[0]
        eval_cond = env_bridge.next_condition(cond, shots_in_end)   # team-to-move = opponent
        terminal = (hh == 1)   # the resulting state has no further throws this end
        if terminal:
            # terminal state -> EXACT rule-based score; the value model is not used
            # here (as in training/search/eval). Same number from the throwing team's view.
            sc = float(env_bridge.score_end(post, block))
            va = vv = sc
        else:
            # value to the team that just threw -> -V(post, eval_cond)
            va, vv = vboth(post, eval_cond, -1.0)
        k += 1
        shots.append({"state": post.copy(), "thrown_slot": slot, "traj": traj, "thrower": name,
                      "block": block, "a_team": persp, "hammer": hammer, "illegal": bool(illegal[0]),
                      "k": k, "preplacement": False, "v_anchor": va, "v_v3": vv, "persp_name": name,
                      "terminal": terminal})
        cond = eval_cond
        state = post
        hh -= 1
    score_a = float(env_bridge.score_end(state, persp))
    return shots, persp, score_a


def render(shots, persp, score_a, out_dir):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    # fixed view across frames: cover all in-play stones + trajectories + the house
    xs, ys = [-2.4, 2.4], [-1.83 - 0.4, 1.83 + 0.4]
    for s in shots:
        xy, live = _xy_m(s["state"])
        if live.any():
            xs += list(xy[live, 0]); ys += list(xy[live, 1])
        if s["traj"] is not None and len(s["traj"]):
            xs += list(s["traj"][:, 0]); ys += list(s["traj"][:, 1])
    xlim = (min(xs) - 0.3, max(xs) + 0.3)
    ylim = (min(ys) - 0.3, max(ys) + 0.3)
    n_throws = max(s["k"] for s in shots)

    for s in shots:
        fig, ax = plt.subplots(figsize=(5.0, 6.4))
        _draw_house(ax)
        xy, live = _xy_m(s["state"])
        for slot in range(NUM_STONES):
            if not live[slot]:
                continue
            team_a = (slot // 6) == persp
            fc = ANCHOR_C if team_a else LATEST_C
            is_thrown = (slot == s["thrown_slot"])
            ax.add_patch(Circle((xy[slot, 0], xy[slot, 1]), STONE_RADIUS_M, facecolor=fc,
                                edgecolor=THROWN_EDGE if is_thrown else "0.15",
                                lw=2.4 if is_thrown else 1.0, zorder=4 if is_thrown else 3))
        if s["traj"] is not None and len(s["traj"]):
            tc = ANCHOR_C if s["thrower"] == "anchor_noisy" else LATEST_C
            ax.plot(s["traj"][:, 0], s["traj"][:, 1], linestyle=":", color=tc, lw=2.0, alpha=0.95, zorder=2)

        ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_aspect("equal"); ax.axis("off")
        if s["k"] == 0:
            title = "Pre-placement — start of end (2 stones)"
            fname = "state_00_preplacement.png"
        else:
            ham = "hammer" if s["hammer"] else "no hammer"
            tag = "  [FORFEIT: early takeout]" if s["illegal"] else ""
            title = f"Throw {s['k']}/{n_throws} — {s['thrower']} ({ham}){tag}"
            fname = f"state_{s['k']:02d}.png"
        ax.set_title(title, fontsize=11)
        if s.get("terminal"):
            vtxt = (f"value to {s['persp_name']} (throwing team):\n"
                    f"  TERMINAL -- rule-based score: {s['v_anchor']:+.0f}\n"
                    f"  (no value model used)")
        else:
            vtxt = (f"value to {s['persp_name']} (throwing team):\n"
                    f"  anchor_noisy model: {s['v_anchor']:+.2f}\n"
                    f"  az_v3/iter1 model:  {s['v_v3']:+.2f}")
        ax.text(0.015, 0.985, vtxt, transform=ax.transAxes, ha="left", va="top",
                fontsize=8.5, family="monospace", zorder=6,
                bbox=dict(boxstyle="round", fc="white", ec="0.75", alpha=0.9))
        # legend / score footer
        foot = (f"anchor_noisy = red   az_v3/iter1 = blue   (dotted = thrown stone path)\n"
                f"final: anchor_noisy {'+' if score_a >= 0 else ''}{score_a:.0f} (anchor perspective)")
        ax.text(0.5, -0.02, foot if s["k"] == n_throws else
                "anchor_noisy = red   az_v3/iter1 = blue   (dotted = thrown stone path)",
                transform=ax.transAxes, ha="center", va="top", fontsize=8, color="0.3")
        fig.tight_layout()
        fig.savefig(out / fname, dpi=130, bbox_inches="tight")
        plt.close(fig)
    print(f"[viz] wrote {len(shots)} PNGs -> {out}  | final anchor score = {score_a:+.0f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--root-idx", type=int, default=0)
    ap.add_argument("--sel-noise-samples", type=int, default=8)
    ap.add_argument("--n-candidates", type=int, default=48)
    ap.add_argument("--out", default="artifacts/figures/game_anchor_vs_v3")
    args = ap.parse_args()
    env_bridge.warm_jax()
    shots, persp, score_a = play_game(args.seed, args.root_idx, args.sel_noise_samples,
                                      args.n_candidates)
    render(shots, persp, score_a, args.out)


if __name__ == "__main__":
    main()
