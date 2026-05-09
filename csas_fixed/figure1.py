#!/usr/bin/env python3
"""
Figure 1: Monte Carlo trajectory “clouds” for curling throws under Gaussian perturbations.

Interpretation (consistent with your earlier “button” usage):
- Each panel shows the curling HOUSE (rings + button at center).
- Many simulated stone trajectories (simple parametric toy model) are drawn to visualize
  how Gaussian execution noise spreads outcomes.
- Left: LOCAL noise (tight perturbations).
- Right: GLOBAL noise (wider perturbations).

This is a visualization tool, not your JAX simulator. It is intentionally lightweight and
does not depend on your inverse/sim code.

Usage:
  python figure1_mc_buttons.py --out figure1.png --n 256 --seed 123

Optional:
  python figure1_mc_buttons.py --out figure1.png --n 512 --seed 0 \
    --local-std 0.08 0.010 0.20 0.06 \
    --global-std 0.20 0.030 0.70 0.12
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt


# ----------------------------
# Curling geometry (meters)
# ----------------------------
# Coordinate convention for this figure:
#   x = along-sheet (0 at release/hog vicinity, increases toward house/button)
#   y = lateral (0 on centerline; +/− sideways)
#
# We draw a simplified “house” centered at (x_button, y=0).
X_BUTTON = 0.0
Y_BUTTON = 0.0

# Typical ring radii (meters)
R_12 = 1.8288
R_8 = 1.2192
R_4 = 0.6096
R_BUTTON = 0.1524

# Typical release-to-button plotting window (meters)
X_START = -18.0   # far “up-ice”
X_END = 3.0       # just beyond house

# ----------------------------
# Toy trajectory model
# ----------------------------
# Params are analogous to your inverse columns:
#   speed: controls how far it travels before stopping (here: just affects curvature scaling)
#   angle: initial direction (radians, small)
#   spin: curl amount (sign indicates curl direction)
#   y0: initial lateral offset (meters)
#
# The model:
#   x(t) = x_start + (x_end - x_start) * t
#   y(t) = y0 + (x - x_start)*tan(angle) + curl(spin, speed, t)
#
# curl term is a smooth S-shaped lateral drift that grows then saturates.
def simulate_toy_path(
    speed: float,
    angle: float,
    spin: float,
    y0: float,
    n_steps: int = 220,
) -> Tuple[np.ndarray, np.ndarray]:
    t = np.linspace(0.0, 1.0, n_steps, dtype=np.float32)
    x = X_START + (X_END - X_START) * t

    # Base lateral drift from aim angle
    y = y0 + (x - X_START) * math.tan(angle)

    # Curl: saturating lateral drift; scale influenced by spin and speed
    # (Higher speed -> slightly less curl accumulation in this toy.)
    speed_scale = 1.0 / max(0.25, float(speed))
    curl_amp = 1.10 * float(spin) * speed_scale  # meters-ish scale
    curl_shape = (1.0 - np.exp(-4.0 * t)) * (1.0 - np.exp(-2.0 * (1.0 - t)))
    y = y + curl_amp * curl_shape
    return x, y


# ----------------------------
# Figure helpers
# ----------------------------
def draw_house(ax: plt.Axes, title: str) -> None:
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    # ax.set_xlabel("along-sheet x (m)")
    # ax.set_ylabel("lateral y (m)")

    # House rings (centered at button)
    # for r in [R_12, R_8, R_4, R_BUTTON]:
    for r in [R_8, R_4, R_BUTTON]:
    # for r in [R_12, R_8, R_4]:
        circ = plt.Circle((X_BUTTON, Y_BUTTON), r, fill=False, linewidth=1.25)
        ax.add_patch(circ)

    # “Button” marker
    # ax.plot([X_BUTTON], [Y_BUTTON], marker="o", markersize=4)

    ax.set_xlim(X_START, X_END)
    ax.set_ylim(-4, 4)
    # ax.grid(True, linewidth=0.5, alpha=0.35)


@dataclass
class NoiseSpec:
    mean: np.ndarray  # (4,)
    std: np.ndarray   # (4,)


def sample_params(rng: np.random.Generator, spec: NoiseSpec, n: int) -> np.ndarray:
    mean = np.asarray(spec.mean, dtype=np.float32).reshape(4)
    std = np.asarray(spec.std, dtype=np.float32).reshape(4)
    cov = np.diag(std ** 2)
    return rng.multivariate_normal(mean, cov, size=int(n)).astype(np.float32)


def plot_mc_panel(ax: plt.Axes, rng: np.random.Generator, spec: NoiseSpec, n: int, title: str) -> None:
    draw_house(ax, title)

    # Draw mean trajectory bold
    m = np.asarray(spec.mean, dtype=np.float32)
    x0, y0 = simulate_toy_path(float(m[0]), float(m[1]), float(m[2]), float(m[3]))
    ax.plot(x0, y0, linewidth=2.5, alpha=0.95)

    # Draw MC trajectories
    params = sample_params(rng, spec, n)
    for i in range(params.shape[0]):
        s, a, sp, y0 = params[i]
        x, y = simulate_toy_path(float(s), float(a), float(sp), float(y0))
        ax.plot(x, y, linewidth=0.8, alpha=0.10)


def main() -> None:
    ap = argparse.ArgumentParser(description="Draw Figure 1: two side-by-side 'buttons' with MC local/global Gaussian perturbations.")
    ap.add_argument("--out", type=str, default="figure1.png", help="Output image path")
    ap.add_argument("--n", type=int, default=256, help="Monte Carlo trajectories per panel")
    ap.add_argument("--seed", type=int, default=123, help="RNG seed")

    # Default mean chosen to produce a reasonable-looking in-house finish.
    ap.add_argument("--mean", type=float, nargs=4, default=[1.35, 0.015, 0.25, 0.0],
                    help="Mean params: speed angle spin y0")

    # Local vs global noise (these are “sensible constants” for visualization).
    ap.add_argument("--local-std", type=float, nargs=4, default=[0.08, 0.010, 0.20, 0.06],
                    help="LOCAL std: speed angle spin y0")
    ap.add_argument("--global-std", type=float, nargs=4, default=[0.20, 0.030, 0.70, 0.12],
                    help="GLOBAL std: speed angle spin y0")

    args = ap.parse_args()

    rng = np.random.default_rng(int(args.seed))
    mean = np.array(args.mean, dtype=np.float32)

    local = NoiseSpec(mean=mean, std=np.array(args.local_std, dtype=np.float32))
    global_ = NoiseSpec(mean=mean, std=np.array(args.global_std, dtype=np.float32))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

    plot_mc_panel(
        axes[0],
        rng=rng,
        spec=local,
        n=int(args.n),
        title=f"LOCAL Search",
    )
    plot_mc_panel(
        axes[1],
        rng=rng,
        spec=global_,
        n=int(args.n),
        title=f"GLOBAL Search",
    )

    # fig.suptitle("Figure 1 — Monte Carlo trajectory clouds toward the button (local vs global execution noise)", fontsize=12)
    out_path = args.out
    fig.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    print(f"[done] wrote {out_path}")


if __name__ == "__main__":
    main()
