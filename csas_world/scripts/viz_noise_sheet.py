#!/usr/bin/env python3
"""Diagnostic: does our execution noise produce realistic scatter on our TRUNCATED
sheet, given it was fit on a FULL curling sheet?

Execution error lives in action space [speed, angle, spin, y0]. An aim-angle error
deflects the whole path, so the *lateral* miss at the house = angle_error x travel
distance; a speed error -> *depth* miss ~ (v/a) x speed_error. Our simulator
releases the stone only ``hog_to_tee = 6.401 m`` from the button (a truncated
sheet); a real (full) sheet release-to-button is ~28.35 m (near hog -> far tee).
Same realistic ice friction (a_linear = 0.11) on both, so a full-sheet draw is the
real ~2.5 m/s and our truncated draw is ~1.2 m/s.

We aim 10 draws at the button on each sheet, apply the SAME noise model
(v1_bowling), and plot where they come to rest. Wider scatter on the full sheet =>
our truncated-sheet shots are unrealistically accurate (noise mis-scaled for our
geometry).
"""
from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import numpy as np

import world  # noqa: F401  (bootstrap: csas path + JAX platform)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402

import jax.numpy as jnp  # noqa: E402
from csas.curling_sim_jax import (CurlingParams, make_initial_state,  # noqa: E402
                                  rollout_positions_until_stop, simulate_until_stop)

# v1_bowling noise (replicated WITHOUT the policy action-bound clip, so the
# full-sheet ~2.5 m/s draw is not capped at the truncated-sheet max of 2.35).
NU = 5.0
SPEED_SCALE = 0.0095
ANGLE_CR = (0.00116, 0.00365)   # [high-speed, low-speed] aim scale (rad)
ANGLE_SR = (0.5, 3.0)           # speed range the aim scale interpolates over
SPIN_STD, Y0_STD = 0.08, 0.015  # Student-t scales (pre sqrt((nu-2)/nu) factor)
FULL_HOG_TO_TEE = 28.35         # near hog -> far tee on a full sheet (m)
HOUSE_RINGS_M = (1.829, 1.219, 0.610, 0.152)  # 12/8/4-ft + button radii


def _angle_scale(speed: float) -> float:
    frac = (np.clip(speed, ANGLE_SR[0], ANGLE_SR[1]) - ANGLE_SR[0]) / (ANGLE_SR[1] - ANGLE_SR[0])
    return ANGLE_CR[1] + frac * (ANGLE_CR[0] - ANGLE_CR[1])


def sample_noisy(nominal, n, rng, scale_mult=1.0, dims=(1.0, 1.0, 1.0, 1.0), sds=None):
    """nominal=[speed,angle,spin,y0] -> [n,4] Student-t noisy actions (no clip).

    If ``sds`` (Gaussian-equivalent per-dim SDs [speed,angle,spin,y0]) is given, it
    overrides the v1_bowling scales (each dim's realized SD == sds[i]). Otherwise
    ``dims``/``scale_mult`` scale the fitted v1_bowling scales."""
    nominal = np.asarray(nominal, np.float64)
    f = math.sqrt((NU - 2.0) / NU)
    if sds is not None:
        scale = np.asarray(sds, np.float64) * f          # SD -> Student-t scale
    else:
        base = np.array([SPEED_SCALE, _angle_scale(nominal[0]), SPIN_STD * f, Y0_STD * f])
        scale = base * np.asarray(dims, np.float64) * float(scale_mult)
    z = rng.standard_t(NU, size=(n, 4))
    return nominal[None, :] + z * scale[None, :]


def spin_for_curl(p, target_m):
    """Binary-search the spin handle whose uncorrected (aim-down-centre) curl == target_m."""
    lo, hi = 0.0, 20.0
    for _ in range(28):
        m = 0.5 * (lo + hi)
        c = rest_pos(p, [_speed_for_depth(p, 0.0, m), 0.0, m, 0.0])[1]
        lo, hi = (m, hi) if c < target_m else (lo, m)
    return 0.5 * (lo + hi)


def rest_pos(p, action):
    """Resting [along, lateral] (m, button at origin) of a single thrown stone."""
    s0 = make_initial_state(p, jnp.zeros((0, 2), jnp.float32),
                            float(action[1]), float(action[0]), float(action[2]), float(action[3]))
    sf, _ = simulate_until_stop(p, s0)
    return np.asarray(sf.pos, np.float64)[0]


def _speed_for_depth(p, angle, spin):
    """Binary-search the speed that stops the stone at the button (along==0)."""
    lo, hi = 0.4, 3.6
    for _ in range(32):
        mid = 0.5 * (lo + hi)
        if rest_pos(p, [mid, angle, spin, 0.0])[0] < 0.0:   # short of button
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def solve_aim(p, spin):
    """Solve (speed, angle) for a shot with the given spin that rests on the button.
    speed controls depth; the aim angle is offset to cancel the curl's lateral drift."""
    angle = 0.0
    speed = _speed_for_depth(p, angle, spin)
    if spin == 0.0:
        return speed, 0.0
    for _ in range(6):
        # angle for lateral==0 (increasing aim angle pushes the rest point +lateral)
        alo, ahi = -0.25, 0.25
        for _ in range(30):
            amid = 0.5 * (alo + ahi)
            if rest_pos(p, [speed, amid, spin, 0.0])[1] > 0.0:
                ahi = amid
            else:
                alo = amid
        angle = 0.5 * (alo + ahi)
        speed = _speed_for_depth(p, angle, spin)
        if abs(rest_pos(p, [speed, angle, spin, 0.0])[1]) < 0.005:
            break
    return speed, angle


def traj_along_lat(p, action):
    """Thrown stone's full path -> (along[], lateral[]) in metres (button at 0)."""
    s0 = make_initial_state(p, jnp.zeros((0, 2), jnp.float32),
                            float(action[1]), float(action[0]), float(action[2]), float(action[3]))
    traj, T = rollout_positions_until_stop(p, s0)
    tr = np.asarray(traj, np.float64)[:int(T), 0, :]
    return tr[:, 0], tr[:, 1]


def _draw_house(ax):
    for r in HOUSE_RINGS_M:
        ax.add_patch(Circle((0, 0), r, fill=False, color="0.55", lw=1.0, zorder=1))
    ax.plot(0, 0, "+", color="0.3", ms=8, zorder=2)


def _draw_house_xy(ax, cx=0.0):
    """House rings centred at (along=cx, lateral=0) for the down-the-sheet view."""
    for r in HOUSE_RINGS_M:
        ax.add_patch(Circle((cx, 0.0), r, fill=False, color="0.55", lw=1.0, zorder=1))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--spin-sds", default="0.25,0.5,1.0,2.0",
                    help="comma list of spin SDs to sweep (Gaussian-equiv); y0 fixed at original, speed/aim off")
    ap.add_argument("--n", type=int, default=160, help="executions per panel (large -> clean SDs)")
    ap.add_argument("--spin", type=float, default=None,
                    help="handle; default = spin giving ~0.3 m curl on the full sheet")
    ap.add_argument("--out", default="artifacts/figures/noise_sheet_curl_spinsweep.png")
    args = ap.parse_args()
    spin_sds = [float(s) for s in str(args.spin_sds).split(",")]
    y0_sd = Y0_STD          # original v1_bowling y0 noise (Gaussian-equiv SD), held fixed
    out = Path(args.out)
    rng = np.random.default_rng(0)
    n = int(args.n)
    n_traj = min(12, n)     # plot only a subset of trajectories; SD uses all n rests
    full = dataclasses.replace(CurlingParams(), hog_to_tee=FULL_HOG_TO_TEE)
    trunc = CurlingParams()
    sheets = [("Full sheet", full), ("Truncated (ours)", trunc)]

    # spin needed for ~0.30 m curl on the full sheet; truncated can't reach it.
    s03 = spin_for_curl(full, 0.30)
    tmax = rest_pos(trunc, [_speed_for_depth(trunc, 0.0, 20.0), 0.0, 20.0, 0.0])[1]
    print(f"[curl] spin for ~0.30 m curl on FULL sheet: {s03:.1f}")
    print(f"[curl] truncated sheet MAX curl (spin 20): {tmax*100:.1f} cm  (cannot reach 0.30 m)")
    spin = float(args.spin) if args.spin is not None else round(s03)

    nc = len(spin_sds)
    fig, axes = plt.subplots(2, nc, figsize=(3.9 * nc, 9.2), squeeze=False)
    for row, (title, p) in enumerate(sheets):
        v = _speed_for_depth(p, 0.0, spin)
        nominal = [v, 0.0, spin, 0.0]
        na, nl = traj_along_lat(p, nominal)
        D = float(p.hog_to_tee)
        curl = rest_pos(p, nominal)[1]
        for col, ss in enumerate(spin_sds):
            ax = axes[row][col]
            noisy = sample_noisy(nominal, n, rng, sds=[0.0, 0.0, ss, y0_sd])   # spin sweep, y0 fixed
            rests = np.array([rest_pos(p, a) for a in noisy])                  # all n (clean SD)
            npaths = [traj_along_lat(p, a) for a in noisy[:n_traj]]            # subset for plotting
            for r in HOUSE_RINGS_M:
                ax.add_patch(Circle((0, 0), r, fill=False, color="0.55", lw=0.9, zorder=2))
            ax.axvline(0.0, color="0.7", lw=0.9, ls=(0, (5, 4)), zorder=1)   # centre line (aim)
            for al, la in npaths:
                ax.plot(la, -al, color="#60a5fa", lw=0.9, alpha=0.6, zorder=3)
            ax.plot(nl, -na, "-", color="#111827", lw=1.8, zorder=4)
            ax.scatter(rests[:, 1], -rests[:, 0], s=22, c="#1d4ed8", edgecolors="0.1", lw=0.4, zorder=5)
            ax.plot(0, 0, "o", mfc="none", mec="#16a34a", ms=11, mew=1.6, zorder=6)
            lat_sd, dep_sd = float(rests[:, 1].std()), float(rests[:, 0].std())
            ax.set_title(f"{title}  |  spin sd {ss:g}  (y0 sd {y0_sd*100:.1f} cm fixed)\n"
                         f"curl {curl*100:.0f} cm  |  rests: lat sd {lat_sd*100:.1f} cm, "
                         f"depth sd {dep_sd*100:.1f} cm", fontsize=8.2)
            ax.set_xlim(-2.6, 2.6); ax.set_ylim(-2.5, D + 1.5); ax.set_aspect("equal"); ax.axis("off")
    fig.suptitle(f"Curl shot (spin {spin:+.0f}, aimed down centre): SPIN-noise sweep "
                 f"(sd = {', '.join(f'{s:g}' for s in spin_sds)}); y0 noise fixed at original "
                 f"({y0_sd*100:.1f} cm), speed & aim noise OFF. Full (top) vs truncated (bottom).", fontsize=9.5)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] wrote {out}")


if __name__ == "__main__":
    main()
