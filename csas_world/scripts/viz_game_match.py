#!/usr/bin/env python3
"""Visualize one full end (pre-placement + 10 throws) for an arbitrary pair of players.

Generalizes scripts/viz_game.py: each side may be our trained WorldModel or the human prior
(CsasPlayer). Player A throws first (no hammer at the start); they alternate.

Selection / execution can be matched to the winrate eval:
  --noisy-select   : robust 1-ply selection = mean decision value over sel_noise_samples noisy
                     executions (this is what h2h_eval / the winrate uses). Without it, selection is
                     deterministic (the actual behaviour of the original game_anchor_vs_v3 viz).
  --realize-noise  : the EXECUTED throw is one execution-noise sample of the intended shot (as in the
                     noisy winrate games). Without it, execution is deterministic -> clean trajectories.

One PNG per state; the just-thrown stone's realized path is drawn dotted (authoritative simulator,
collisions + curl included).

    source scripts/setup_gpu.sh && python3 scripts/viz_game_match.py \
        --player-a checkpoints/.../h10/r1/model.pt --label-a "ours" \
        --player-b prior --label-b "human_prior" --noisy-select --realize-noise \
        --out artifacts/figures/game_ours_vs_prior
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import world  # noqa: F401  (bootstrap: GNN env + csas path)
from world import env_bridge
from world.config import Config
from world.eval.head_to_head import CsasPlayer, WorldPlayer

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402

from csas.common import NUM_STONES, POS_MAX, in_play_raw  # noqa: E402
from csas.make_value_heatmaps import BUTTON_RAW, M_PER_RAW, STONE_RADIUS_M, _draw_house  # noqa: E402
from csas.search import _new_slot  # noqa: E402
from csas.preplaced_value_data import load_preplaced_training_frame, materialize_preplaced  # noqa: E402
from csas.visualize_policy_multi_action_samples import _trajectory_m_for_actions  # noqa: E402

A_COLOR, B_COLOR = "#1d4ed8", "#dc2626"   # A (ours) = blue, B (opponent) = red
THROWN_EDGE = "#f59e0b"


def _xy_m(state_norm):
    raw = np.asarray(state_norm, np.float32).reshape(NUM_STONES, 2) * POS_MAX
    return (raw - BUTTON_RAW) * M_PER_RAW, in_play_raw(raw)


def _make_player(spec: str, label: str, dev, n_candidates: int, noise, sns: int):
    cfg = Config()
    if str(spec).lower() == "prior":
        pol = cfg.csas_path(cfg.paths.prior_policy_ckpt).as_posix()
        val = cfg.csas_path(cfg.paths.prior_value_ckpt).as_posix()
        return CsasPlayer(pol, val, dev, n_candidates=n_candidates, name=label,
                          noise=noise, sel_noise_samples=sns)
    return WorldPlayer(spec, dev, n_candidates=n_candidates, name=label,
                       noise=noise, sel_noise_samples=sns)


def play_game(args):
    import torch
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = torch.device(args.device)
    from world.search.noise import make_noise
    cfg = Config()
    ncfg = cfg.csas_path(cfg.search.noise_config).as_posix()
    sel_noise = make_noise(ncfg, 1000 + args.seed) if args.noisy_select else None
    exec_noise = make_noise(ncfg, 9000 + args.seed) if args.realize_noise else None

    A = _make_player(args.player_a, args.label_a, dev, args.n_candidates, sel_noise, args.sel_noise_samples)
    B = _make_player(args.player_b, args.label_b, dev, args.n_candidates, sel_noise, args.sel_noise_samples)

    # start of end: canonical pre-placement; player A is the first thrower (no hammer).
    dfp = load_preplaced_training_frame()
    dfp = dfp[dfp["mode"] == args.mode].reset_index(drop=True)
    x_all, c_all, _ = materialize_preplaced(dfp)
    i = args.root_idx % len(dfp)
    x0 = x_all[i].numpy().astype(np.float32)
    c0 = c_all[i].numpy().astype(np.float32)
    shots_in_end, horizon = 10, 10
    persp = int(round(float(c0[2])))     # first thrower's block == player A

    def vboth(st, eval_cond, sign):
        return (sign * float(A._value_fn(st[None], eval_cond)[0]),
                sign * float(B._value_fn(st[None], eval_cond)[0]))

    state, cond, hh = x0.copy(), c0.copy(), horizon
    va0, vv0 = vboth(state, cond, 1.0)
    shots = [{"state": state.copy(), "thrown_slot": None, "traj": None, "thrower": None,
              "hammer": None, "illegal": False, "k": 0, "preplacement": True,
              "v_a": va0, "v_b": vv0, "persp_name": args.label_a, "terminal": False}]
    k = 0
    while hh >= 1:
        block = int(round(float(cond[2])))
        plays_a = (block == persp)
        thrower = A if plays_a else B
        name = args.label_a if plays_a else args.label_b
        hammer = int(round(float(cond[1]))) == 1
        intended = np.asarray(thrower.select_intended(state, cond, hh, shots_in_end, block), np.float32)
        realized = exec_noise.sample_batch(intended[None], 1).reshape(4) if exec_noise is not None else intended
        slot = int(_new_slot(state.reshape(NUM_STONES, 2) * POS_MAX, float(block)))
        traj_list = _trajectory_m_for_actions(state.reshape(NUM_STONES, 2) * POS_MAX, realized[None])
        traj = traj_list[0] if traj_list else None
        post, illegal = env_bridge.apply_legality(
            state, env_bridge.simulate_one(state, cond, realized)[None], hh, cond)
        post = post[0]
        eval_cond = env_bridge.next_condition(cond, shots_in_end)
        terminal = (hh == 1)
        if terminal:
            sc = float(env_bridge.score_end(post, block))
            va = vv = sc
        else:
            va, vv = vboth(post, eval_cond, -1.0)
        k += 1
        shots.append({"state": post.copy(), "thrown_slot": slot, "traj": traj, "thrower": name,
                      "block": block, "hammer": hammer, "illegal": bool(illegal[0]), "k": k,
                      "preplacement": False, "v_a": va, "v_b": vv, "persp_name": name,
                      "terminal": terminal})
        cond = eval_cond
        state = post
        hh -= 1
    score_a = float(env_bridge.score_end(state, persp))
    return shots, persp, score_a


def render(shots, persp, score_a, args):
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    xs, ys = [-2.4, 2.4], [-1.83 - 0.4, 1.83 + 0.4]
    for s in shots:
        xy, live = _xy_m(s["state"])
        if live.any():
            xs += list(xy[live, 0]); ys += list(xy[live, 1])
        if s["traj"] is not None and len(s["traj"]):
            xs += list(s["traj"][:, 0]); ys += list(s["traj"][:, 1])
    xlim = (min(xs) - 0.3, max(xs) + 0.3); ylim = (min(ys) - 0.3, max(ys) + 0.3)
    n_throws = max(s["k"] for s in shots)
    setup = (f"selection={'noisy-robust x%d' % args.sel_noise_samples if args.noisy_select else 'deterministic'}"
             f"   execution={'NOISY (winrate)' if args.realize_noise else 'deterministic'}")

    for s in shots:
        fig, ax = plt.subplots(figsize=(5.0, 6.4))
        _draw_house(ax)
        xy, live = _xy_m(s["state"])
        for slot in range(NUM_STONES):
            if not live[slot]:
                continue
            team_a = (slot // 6) == persp
            fc = A_COLOR if team_a else B_COLOR
            is_thrown = (slot == s["thrown_slot"])
            ax.add_patch(Circle((xy[slot, 0], xy[slot, 1]), STONE_RADIUS_M, facecolor=fc,
                                edgecolor=THROWN_EDGE if is_thrown else "0.15",
                                lw=2.4 if is_thrown else 1.0, zorder=4 if is_thrown else 3))
        if s["traj"] is not None and len(s["traj"]):
            tc = A_COLOR if s["thrower"] == args.label_a else B_COLOR
            ax.plot(s["traj"][:, 0], s["traj"][:, 1], linestyle=":", color=tc, lw=2.0, alpha=0.95, zorder=2)
        ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_aspect("equal"); ax.axis("off")
        if s["k"] == 0:
            title = "Pre-placement — start of end (2 stones)"; fname = "state_00_preplacement.png"
        else:
            ham = "hammer" if s["hammer"] else "no hammer"
            tag = "  [FORFEIT: early takeout]" if s["illegal"] else ""
            title = f"Throw {s['k']}/{n_throws} — {s['thrower']} ({ham}){tag}"; fname = f"state_{s['k']:02d}.png"
        ax.set_title(title, fontsize=11)
        if s.get("terminal"):
            vtxt = (f"value to {s['persp_name']} (throwing team):\n"
                    f"  TERMINAL -- rule score: {s['v_a']:+.0f}  (no value model)")
        else:
            vtxt = (f"value to {s['persp_name']} (throwing team):\n"
                    f"  {args.label_a} model: {s['v_a']:+.2f}\n  {args.label_b} model: {s['v_b']:+.2f}")
        ax.text(0.015, 0.985, vtxt, transform=ax.transAxes, ha="left", va="top", fontsize=8.5,
                family="monospace", zorder=6, bbox=dict(boxstyle="round", fc="white", ec="0.75", alpha=0.9))
        foot = f"{args.label_a} = blue   {args.label_b} = red   (dotted = thrown path)\n{setup}"
        if s["k"] == n_throws:
            foot += f"\nfinal: {args.label_a} {'+' if score_a >= 0 else ''}{score_a:.0f} ({args.label_a} perspective)"
        ax.text(0.5, -0.02, foot, transform=ax.transAxes, ha="center", va="top", fontsize=8, color="0.3")
        fig.tight_layout(); fig.savefig(out / fname, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"[viz] wrote {len(shots)} PNGs -> {out}  | final {args.label_a} score = {score_a:+.0f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--player-a", required=True, help="WorldModel ckpt path, or 'prior'")
    ap.add_argument("--player-b", required=True, help="WorldModel ckpt path, or 'prior'")
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    ap.add_argument("--noisy-select", action="store_true", help="robust selection (winrate decision rule)")
    ap.add_argument("--realize-noise", action="store_true", help="noisy realized execution (winrate games)")
    ap.add_argument("--sel-noise-samples", type=int, default=8)
    ap.add_argument("--n-candidates", type=int, default=48)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--root-idx", type=int, default=0)
    ap.add_argument("--mode", default="standard", choices=["standard", "pp_left", "pp_right"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    env_bridge.warm_jax()
    shots, persp, score_a = play_game(args)
    render(shots, persp, score_a, args)


if __name__ == "__main__":
    main()
