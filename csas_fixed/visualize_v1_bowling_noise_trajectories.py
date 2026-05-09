#!/usr/bin/env python3
"""
Visualize a local execution-noise model with the actual JAX simulator.

The prior version used a hand-built quadratic path and sampled directly from the
raw `std` vector, which overstated the Bowling spread. This version:
  - uses the same local Student-t sampling rules as scoring,
  - auto-finds a nominal center speed that finishes near the button, and
  - plots the simulated thrown-stone trajectories on an empty sheet.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np

THIS_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR / "inverse"))

import jax.numpy as jnp
from curling_inverse import MAX_X, MAX_Y, MIN_X, MIN_Y  # type: ignore
from curling_sim_jax import CurlingParams, make_initial_state, simulate_from_params, step  # type: ignore


HOUSE_RADII_M = [1.829, 1.219, 0.610, 0.152]
STONE_RADIUS_M = 0.145
NEAR_HOG_TO_TEE_M = 6.401


def _load_cfg(path: pathlib.Path) -> dict:
    return json.loads(path.read_text())


def _resolve_noise_block(cfg: dict, section: str, task: int | None = None, handle: int | None = None) -> tuple[dict, float]:
    meta_min_std = float(cfg.get("meta", {}).get("min_std", 1e-3))
    if section == "local":
        return cfg.get("local", {}), meta_min_std
    if section == "global":
        if task is not None and handle is not None:
            key = f"task_{int(task)}_handle_{int(handle)}"
            entry = cfg.get("by_task_handle", {}).get(key)
            if isinstance(entry, dict):
                return entry, meta_min_std
        return cfg.get("default", {}), meta_min_std
    raise ValueError(f"Unsupported section: {section}")


def _sample_noise(
    cfg: dict,
    rng: np.random.Generator,
    center: np.ndarray,
    *,
    section: str,
    task: int | None = None,
    handle: int | None = None,
) -> np.ndarray:
    block, meta_min_std = _resolve_noise_block(cfg, section, task=task, handle=handle)
    min_std = float(block.get("min_std", meta_min_std))
    default_std = [0.0123, 0.05, 0.08, 0.015] if section == "local" else [0.55, 0.23, 1.1, 0.15]
    std = np.maximum(np.array(block.get("std", default_std), dtype=float).reshape(4), min_std)
    dist = str(block.get("distribution", "gaussian")).strip().lower()

    if dist == "student_t":
        nu = float(block.get("nu", 5.0))
        if nu <= 2.0:
            raise ValueError(f"Student-t noise requires nu>2, got {nu}")

        z = rng.standard_t(df=nu, size=4)
        t_scale = math.sqrt((nu - 2.0) / nu)
        scales = std * t_scale

        if "speed_scale" in block:
            scales[0] = max(float(block["speed_scale"]), min_std)

        if "angle_speed_range" in block and np.isfinite(center[0]):
            speed_range = np.array(block["angle_speed_range"], dtype=float).reshape(2)
            lo_speed, hi_speed = float(speed_range[0]), float(speed_range[1])
            speed = float(np.clip(abs(float(center[0])), lo_speed, hi_speed))
            frac = 0.0 if hi_speed <= lo_speed else (speed - lo_speed) / (hi_speed - lo_speed)
            if "angle_scale_range" in block:
                scale_range = np.array(block["angle_scale_range"], dtype=float).reshape(2)
                angle_scale = float(scale_range[1] + frac * (scale_range[0] - scale_range[1]))
                scales[1] = max(angle_scale, min_std)
            elif "angle_variance_range" in block:
                var_range = np.array(block["angle_variance_range"], dtype=float).reshape(2)
                angle_var = float(var_range[1] + frac * (var_range[0] - var_range[1]))
                scales[1] = max(math.sqrt(max(angle_var, 0.0)) * t_scale, min_std)

        return center + z * scales

    return center + rng.normal(loc=0.0, scale=std, size=4)


def _simulate_throw_traj(curl_params: CurlingParams, x: np.ndarray) -> np.ndarray:
    prev = jnp.zeros((0, 2), dtype=jnp.float32)
    traj = np.asarray(
        simulate_from_params(
            curl_params,
            prev,
            jnp.asarray(x, dtype=jnp.float32),
            dynamic=True,
        )
    )
    return traj[:, -1, :]


def _final_pos(curl_params: CurlingParams, x: np.ndarray) -> np.ndarray:
    prev = jnp.zeros((0, 2), dtype=jnp.float32)
    final = np.asarray(
        simulate_from_params(
            curl_params,
            prev,
            jnp.asarray(x, dtype=jnp.float32),
            dynamic=False,
        )
    )
    return final[-1]


def _auto_center_speed(curl_params: CurlingParams, center: np.ndarray) -> float:
    lo, hi = 0.1, 3.0
    best_speed = lo
    best_err = float("inf")
    for speed in np.linspace(lo, hi, 80):
        x = center.copy()
        x[0] = float(speed)
        final = _final_pos(curl_params, x)
        err = abs(float(final[0]))
        if err < best_err:
            best_err = err
            best_speed = float(speed)
    return best_speed


def _state_at_along_x(curl_params: CurlingParams, x: np.ndarray, target_x: float) -> np.ndarray | None:
    prev = jnp.zeros((0, 2), dtype=jnp.float32)
    s = make_initial_state(
        curl_params,
        prev,
        jnp.asarray(float(x[1]), dtype=jnp.float32),
        jnp.asarray(float(x[0]), dtype=jnp.float32),
        jnp.asarray(float(x[2]), dtype=jnp.float32),
        jnp.asarray(float(x[3]), dtype=jnp.float32),
    )
    s_prev = s
    x_prev = float(np.asarray(s_prev.pos)[-1, 0])
    if x_prev >= target_x:
        vel_prev = np.asarray(s_prev.vel)[-1]
        speed_prev = float(np.linalg.norm(vel_prev))
        angle_prev = float(math.atan2(float(vel_prev[1]), float(vel_prev[0])))
        return np.array([speed_prev, angle_prev, float(np.asarray(s_prev.omega)[-1]), float(np.asarray(s_prev.pos)[-1, 1])], dtype=float)

    for _ in range(int(curl_params.max_steps)):
        s_next = step(curl_params, s_prev)
        pos_prev = np.asarray(s_prev.pos)[-1]
        pos_next = np.asarray(s_next.pos)[-1]
        x0 = float(pos_prev[0])
        x1 = float(pos_next[0])
        if x0 <= target_x <= x1:
            frac = 0.0 if x1 == x0 else (target_x - x0) / (x1 - x0)
            vel_prev = np.asarray(s_prev.vel)[-1]
            vel_next = np.asarray(s_next.vel)[-1]
            omega_prev = float(np.asarray(s_prev.omega)[-1])
            omega_next = float(np.asarray(s_next.omega)[-1])
            pos_interp = pos_prev + frac * (pos_next - pos_prev)
            vel_interp = vel_prev + frac * (vel_next - vel_prev)
            speed = float(np.linalg.norm(vel_interp))
            angle = float(math.atan2(float(vel_interp[1]), float(vel_interp[0])))
            omega = float(omega_prev + frac * (omega_next - omega_prev))
            return np.array([speed, angle, omega, float(pos_interp[1])], dtype=float)
        s_prev = s_next

    return None


def _plot_sheet(ax, *, zoom_house: bool, hog_to_tee: float) -> None:
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("lateral (m)")
    ax.set_ylabel("along-sheet (m)")

    if zoom_house:
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(2.2, -2.5)
    else:
        ax.set_xlim(MIN_Y, MAX_Y)
        ax.set_ylim(max(MAX_X + 0.3, 2.5), -hog_to_tee - 0.4)

    for radius in HOUSE_RADII_M:
        ax.add_patch(Circle((0.0, 0.0), radius, fill=False, linewidth=1.0, color="0.55"))
    ax.axhline(0.0, linewidth=0.7, alpha=0.25, color="0.25")
    ax.axvline(0.0, linewidth=0.7, alpha=0.25, color="0.25")
    ax.grid(True, alpha=0.18)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--noise-json",
        type=str,
        default="/mnt/data/curling2/csas_fixed/noise_versions/v1_bowling.json",
    )
    ap.add_argument(
        "--out",
        type=str,
        default="/mnt/data/curling2/csas_fixed/noise_versions/v1_bowling_trajectories.png",
    )
    ap.add_argument("--title", type=str, default="")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--section", type=str, default="local", choices=["local", "global"])
    ap.add_argument("--task", type=int, default=None)
    ap.add_argument("--handle", type=int, default=None)
    ap.add_argument("--center-speed", type=float, default=float("nan"))
    ap.add_argument("--center-angle", type=float, default=0.0)
    ap.add_argument("--center-spin", type=float, default=0.0)
    ap.add_argument("--center-y0", type=float, default=0.0)
    ap.add_argument("--sim-hog-to-tee", type=float, default=NEAR_HOG_TO_TEE_M)
    ap.add_argument(
        "--report-near-hog-equivalent",
        action="store_true",
        help="Also compute simulator-derived equivalent release states at the near hog line (6.401m).",
    )
    args = ap.parse_args()

    noise_path = pathlib.Path(args.noise_json)
    cfg = _load_cfg(noise_path)
    rng = np.random.default_rng(int(args.seed))
    curl_params = CurlingParams(
        dt=0.02,
        substeps=2,
        k_penalty=2.5e4,
        c_damp=220.0,
        k_curl=0.10,
        hog_to_tee=float(args.sim_hog_to_tee),
    )

    center = np.array(
        [
            args.center_speed,
            args.center_angle,
            args.center_spin,
            args.center_y0,
        ],
        dtype=float,
    )
    if not np.isfinite(center[0]):
        center[0] = _auto_center_speed(curl_params, center)

    center_traj = _simulate_throw_traj(curl_params, center)
    center_final = center_traj[-1]

    xs = []
    trajs = []
    finals = []
    for _ in range(int(args.n)):
        x = _sample_noise(cfg, rng, center, section=args.section, task=args.task, handle=args.handle)
        xs.append(x)
        traj = _simulate_throw_traj(curl_params, x)
        trajs.append(traj)
        finals.append(traj[-1])

    finals_arr = np.asarray(finals, dtype=float)
    lat_std = float(np.std(finals_arr[:, 1], ddof=1)) if len(finals_arr) > 1 else 0.0
    along_std = float(np.std(finals_arr[:, 0], ddof=1)) if len(finals_arr) > 1 else 0.0
    near_equiv = None
    if args.report_near_hog_equivalent and float(args.sim_hog_to_tee) > NEAR_HOG_TO_TEE_M:
        near_rows = []
        near_center = _state_at_along_x(curl_params, center, -NEAR_HOG_TO_TEE_M)
        for x in xs:
            row = _state_at_along_x(curl_params, x, -NEAR_HOG_TO_TEE_M)
            if row is not None:
                near_rows.append(row)
        if near_center is not None and near_rows:
            near_arr = np.asarray(near_rows, dtype=float)
            near_equiv = {
                "center": near_center,
                "std_speed": float(np.std(near_arr[:, 0], ddof=1)) if len(near_arr) > 1 else 0.0,
                "std_angle": float(np.std(near_arr[:, 1], ddof=1)) if len(near_arr) > 1 else 0.0,
                "std_spin": float(np.std(near_arr[:, 2], ddof=1)) if len(near_arr) > 1 else 0.0,
                "std_y0": float(np.std(near_arr[:, 3], ddof=1)) if len(near_arr) > 1 else 0.0,
            }

    fig, (ax_full, ax_house) = plt.subplots(1, 2, figsize=(12.5, 7.2), dpi=180)
    _plot_sheet(ax_full, zoom_house=False, hog_to_tee=float(args.sim_hog_to_tee))
    _plot_sheet(ax_house, zoom_house=True, hog_to_tee=float(args.sim_hog_to_tee))

    for traj in trajs:
        ax_full.plot(traj[:, 1], traj[:, 0], color="0.15", alpha=0.18, linewidth=0.9)
        ax_house.plot(traj[:, 1], traj[:, 0], color="0.15", alpha=0.12, linewidth=0.8)
        ax_house.add_patch(
            Circle(
                (float(traj[-1, 1]), float(traj[-1, 0])),
                STONE_RADIUS_M,
                facecolor="none",
                edgecolor="0.1",
                linewidth=0.55,
                alpha=0.22,
                zorder=3,
            )
        )

    ax_full.plot(center_traj[:, 1], center_traj[:, 0], color="tab:red", linewidth=1.8, alpha=0.9)
    ax_house.add_patch(
        Circle(
            (float(center_final[1]), float(center_final[0])),
            STONE_RADIUS_M,
            facecolor="none",
            edgecolor="tab:red",
            linewidth=1.2,
            alpha=0.95,
            zorder=4,
        )
    )
    ax_full.scatter([center_final[1]], [center_final[0]], s=22, color="tab:red", zorder=5)

    ax_full.set_title("Simulated trajectories")
    ax_house.set_title("Endpoint cloud near house")

    if args.title:
        title = args.title
    else:
        title = f"{noise_path.stem} noise via simulator"
    fig.suptitle(
        f"{title}\nspawn_hog_to_tee={float(args.sim_hog_to_tee):.3f}m  center=[speed={center[0]:.3f}, angle={center[1]:.4f}, spin={center[2]:.3f}, y0={center[3]:.3f}]  "
        f"endpoint std: lateral={lat_std:.3f}m, along={along_std:.3f}m",
        y=0.98,
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=240)
    plt.close(fig)
    print(f"[done] wrote {out}")
    print(f"[info] center={center.tolist()}")
    print(f"[info] center_final={center_final.tolist()}")
    print(f"[info] endpoint_std lateral={lat_std:.6f} along={along_std:.6f}")
    if near_equiv is not None:
        center_near = near_equiv["center"]
        print(
            "[info] equivalent_near_hog_center="
            f"[speed={center_near[0]:.6f}, angle={center_near[1]:.6f}, spin={center_near[2]:.6f}, y0={center_near[3]:.6f}]"
        )
        print(
            "[info] equivalent_near_hog_std="
            f"[speed={near_equiv['std_speed']:.6f}, angle={near_equiv['std_angle']:.6f}, "
            f"spin={near_equiv['std_spin']:.6f}, y0={near_equiv['std_y0']:.6f}]"
        )


if __name__ == "__main__":
    main()
