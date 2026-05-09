#!/usr/bin/env python3
"""
figure2_infer_toy.py

Figure 2 (standalone): House/button + stones + inferred trajectory using a lightweight toy model.
NO imports from any other project scripts. Only uses: numpy, matplotlib, argparse, json.

What it draws:
  - House centered at the "button" (two rings + button circle)
  - Prev stones (filled)
  - Target stones (outlined) [optional]
  - Inferred trajectory of the thrown stone (line)
  - Predicted final stone position (filled marker)

Inference:
  - Uses a simple parametric trajectory model (same spirit as your Figure 1 toy).
  - Fits params [speed, angle, spin, y0] so the final position is close to a chosen target.
  - Default: targets the closest target stone to the button (or a user-chosen index).
  - Optimization: coarse random search + local refinement (still very fast; no SciPy needed).

Usage:
  python figure2_infer_toy.py --out figure2.png

With custom stones from a JSON file:
  python figure2_infer_toy.py --out figure2.png --scene scene.json

scene.json format:
{
  "prev": [[0.20, 0.10], [0.05, -0.25], [0.15, 0.25], [-0.30, 0.05], [0.30, 0.05]],
  "targets": [[0.05, 0.00], [-0.10, 0.20]],
  "target_index": 0
}

Optional:
  python figure2_infer_toy.py --out figure2.png --seed 0 --search 8000 --refine 2000
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


# ----------------------------
# House / button geometry (meters)
# ----------------------------
X_BUTTON = 0.0
Y_BUTTON = 0.0

# Two rings + button circle (you can swap R_8 <-> R_12 if desired)
R_8 = 1.2192
R_4 = 0.6096
R_BUTTON = 0.1524

# Plot window
X_START = -18.0
X_END = 3.0

# Tight vertical bounds to avoid "thick" whitespace
Y_LIM = 2.6

# Toy model sim steps
N_STEPS = 220


# ----------------------------
# Toy trajectory model
# ----------------------------
# Params:
#   speed: controls curvature scaling (higher speed => slightly less curl)
#   angle: initial direction (radians)
#   spin: curl amount (sign gives curl direction)
#   y0: initial lateral offset (meters)
#
# x(t) = X_START + (X_END - X_START) * t
# y(t) = y0 + (x - X_START)*tan(angle) + curl(spin, speed, t)
def simulate_toy_path(speed: float, angle: float, spin: float, y0: float, n_steps: int = N_STEPS) -> Tuple[np.ndarray, np.ndarray]:
    t = np.linspace(0.0, 1.0, n_steps, dtype=np.float32)
    x = X_START + (X_END - X_START) * t

    y = y0 + (x - X_START) * math.tan(angle)

    speed_scale = 1.0 / max(0.25, float(speed))
    curl_amp = 1.10 * float(spin) * speed_scale
    curl_shape = (1.0 - np.exp(-4.0 * t)) * (1.0 - np.exp(-2.0 * (1.0 - t)))
    y = y + curl_amp * curl_shape
    return x, y


def final_pos_from_params(p: np.ndarray) -> np.ndarray:
    x, y = simulate_toy_path(float(p[0]), float(p[1]), float(p[2]), float(p[3]))
    return np.array([float(x[-1]), float(y[-1])], dtype=np.float32)


# ----------------------------
# Drawing helpers
# ----------------------------
def draw_house(ax: plt.Axes, title: str) -> None:
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")

    # no grid, no numbers
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # TWO rings + button circle
    for r in [R_8, R_4]:
        ax.add_patch(plt.Circle((X_BUTTON, Y_BUTTON), r, fill=False, linewidth=1.25))
    ax.add_patch(plt.Circle((X_BUTTON, Y_BUTTON), R_BUTTON, fill=False, linewidth=1.25))

    ax.set_xlim(X_START, X_END)
    ax.set_ylim(-Y_LIM, Y_LIM)


def plot_stones(ax: plt.Axes, pts: np.ndarray, label: str, filled: bool, alpha: float = 0.95) -> None:
    if pts is None or len(pts) == 0:
        return
    pts = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
    if filled:
        ax.scatter(pts[:, 0], pts[:, 1], s=120, alpha=alpha, edgecolors="black", linewidths=0.8, label=label)
    else:
        ax.scatter(pts[:, 0], pts[:, 1], s=140, alpha=alpha, facecolors="none", edgecolors="black", linewidths=1.2, label=label)


# ----------------------------
# Inference (no SciPy)
# ----------------------------
@dataclass
class Bounds:
    speed: Tuple[float, float] = (0.6, 2.4)
    angle: Tuple[float, float] = (-0.12, 0.12)  # radians
    spin: Tuple[float, float] = (-1.2, 1.2)
    y0: Tuple[float, float] = (-1.0, 1.0)


def choose_target(targets: np.ndarray, target_index: Optional[int] = None) -> np.ndarray:
    targets = np.asarray(targets, dtype=np.float32).reshape(-1, 2)
    if targets.shape[0] == 0:
        # default target: button
        return np.array([X_BUTTON, Y_BUTTON], dtype=np.float32)

    if target_index is not None:
        i = int(target_index)
        i = max(0, min(i, targets.shape[0] - 1))
        return targets[i]

    # default: target closest to button
    d2 = (targets[:, 0] - X_BUTTON) ** 2 + (targets[:, 1] - Y_BUTTON) ** 2
    return targets[int(np.argmin(d2))]


def objective(p: np.ndarray, target_xy: np.ndarray, reg: float = 1e-3) -> float:
    """
    Loss = final_pos distance^2 + small regularizer on params to keep solutions reasonable.
    """
    f = final_pos_from_params(p)
    d2 = float(((f - target_xy) ** 2).sum())
    # mild preference for smaller |angle| and |spin| to avoid extreme curls in the toy visual
    r = reg * float(p[1] ** 2 + 0.3 * p[2] ** 2 + 0.1 * p[3] ** 2)
    return d2 + r


def random_search(rng: np.random.Generator, b: Bounds, target_xy: np.ndarray, n: int) -> Tuple[np.ndarray, float]:
    low = np.array([b.speed[0], b.angle[0], b.spin[0], b.y0[0]], dtype=np.float32)
    high = np.array([b.speed[1], b.angle[1], b.spin[1], b.y0[1]], dtype=np.float32)

    P = rng.uniform(low=low, high=high, size=(int(n), 4)).astype(np.float32)

    best_p = P[0].copy()
    best_l = objective(best_p, target_xy)
    for i in range(P.shape[0]):
        l = objective(P[i], target_xy)
        if l < best_l:
            best_l = l
            best_p = P[i].copy()
    return best_p, float(best_l)


def local_refine(rng: np.random.Generator, b: Bounds, target_xy: np.ndarray, p0: np.ndarray, n: int) -> Tuple[np.ndarray, float]:
    """
    Simple stochastic local search:
      sample around p0 with shrinking Gaussian noise; keep improvements.
    """
    p = p0.astype(np.float32).copy()
    best_l = objective(p, target_xy)

    # scale relative to bounds width
    span = np.array(
        [b.speed[1] - b.speed[0], b.angle[1] - b.angle[0], b.spin[1] - b.spin[0], b.y0[1] - b.y0[0]],
        dtype=np.float32,
    )
    sigma = 0.08 * span  # starting step
    low = np.array([b.speed[0], b.angle[0], b.spin[0], b.y0[0]], dtype=np.float32)
    high = np.array([b.speed[1], b.angle[1], b.spin[1], b.y0[1]], dtype=np.float32)

    for t in range(int(n)):
        # occasionally shrink step size
        if t in (int(n * 0.35), int(n * 0.7)):
            sigma *= 0.45

        cand = (p + rng.normal(0.0, sigma, size=(4,)).astype(np.float32))
        cand = np.clip(cand, low, high)
        l = objective(cand, target_xy)
        if l < best_l:
            best_l = l
            p = cand

    return p, float(best_l)


# ----------------------------
# Scene I/O
# ----------------------------
def load_scene(path: Optional[str]) -> Dict:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ----------------------------
# Main
# ----------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Figure 2: house + stones + inferred toy trajectory (standalone).")
    ap.add_argument("--out", type=str, default="figure2.png")
    ap.add_argument("--scene", type=str, default=None, help="Optional JSON file with prev/targets/target_index.")
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--search", type=int, default=8000, help="Random search samples")
    ap.add_argument("--refine", type=int, default=2000, help="Local refinement iterations")

    # If you want manual override of the mean / initial guess, you can set these
    ap.add_argument("--target-index", type=int, default=None, help="Override target index (0-based)")

    args = ap.parse_args()
    rng = np.random.default_rng(int(args.seed))

    # Defaults (use your example prev stones; targets optional)
    prev_default = np.array(
        [[0.20, 0.10], [0.05, -0.25], [0.15, 0.25], [-0.30, 0.05], [0.30, 0.05]],
        dtype=np.float32,
    )
    targets_default = np.array([[0.05, 0.00]], dtype=np.float32)  # by default aim near button line

    scene = load_scene(args.scene)
    prev = np.array(scene.get("prev", prev_default), dtype=np.float32).reshape(-1, 2)
    targets = np.array(scene.get("targets", targets_default), dtype=np.float32).reshape(-1, 2)

    target_index = args.target_index
    if target_index is None and "target_index" in scene:
        target_index = scene.get("target_index", None)

    target_xy = choose_target(targets, target_index=target_index)

    # Inference
    b = Bounds()
    p_best, _ = random_search(rng, b, target_xy, n=int(args.search))
    p_best, _ = local_refine(rng, b, target_xy, p_best, n=int(args.refine))

    # Simulate inferred trajectory
    x, y = simulate_toy_path(float(p_best[0]), float(p_best[1]), float(p_best[2]), float(p_best[3]))
    final_xy = np.array([x[-1], y[-1]], dtype=np.float32)

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(9.5, 4.2), constrained_layout=True)
    draw_house(ax, "Inferred throw (toy)")

    plot_stones(ax, prev, label="prev", filled=True, alpha=0.95)
    if targets is not None and targets.shape[0] > 0:
        plot_stones(ax, targets, label="target", filled=False, alpha=0.95)

    # Trajectory and predicted final
    ax.plot(x, y, linewidth=2.2, alpha=0.9, label="inferred trajectory")
    ax.scatter([final_xy[0]], [final_xy[1]], s=140, edgecolors="black", linewidths=0.8, alpha=0.85, label="pred final")

    # Optional: show which target was used for inference
    ax.scatter([target_xy[0]], [target_xy[1]], s=170, facecolors="none", edgecolors="black", linewidths=1.8, alpha=0.9)

    # Legend (compact); remove if you want a cleaner export
    ax.legend(loc="upper left", frameon=False, fontsize=10)

    fig.savefig(args.out, dpi=200, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    print(f"[done] wrote {args.out}")
    print(f"[info] target_xy={target_xy.tolist()} | inferred_params=[speed,angle,spin,y0]={p_best.tolist()} | final_xy={final_xy.tolist()}")


if __name__ == "__main__":
    main()
