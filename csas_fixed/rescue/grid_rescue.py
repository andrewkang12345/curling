#!/usr/bin/env python3
"""
Grid-based rescue for rows whose inverse loss is still above a threshold.

This is the stronger second-stage pass:
- read one full inverse CSV
- select only rows with hard_loss_refine > threshold
- run dense 4D grid + local coordinate polish on visible GPUs
- write a single rescued output CSV
"""

from __future__ import annotations

import argparse
import ctypes
import math
import multiprocessing as mp
import os
import pathlib
import sys
import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize as scipy_min


ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "inverse"))


SIM_PARAMS = dict(
    dt=0.02,
    substeps=2,
    max_steps=1500,
    k_penalty=2.5e4,
    c_damp=165.0,
    c_damp_sep_frac=1.0,
    c_tangent=20.0,
    mu_tangent=0.05,
    spin_contact=0.08,
    k_curl=0.12,
    a_linear=0.10,
    gamma_spin=0.12,
)
COARSE_PARAMS = dict(
    dt=0.03,
    substeps=1,
    max_steps=900,
    k_penalty=2.0e4,
    c_damp=165.0,
    c_damp_sep_frac=1.0,
    c_tangent=20.0,
    mu_tangent=0.05,
    spin_contact=0.08,
    k_curl=0.12,
    a_linear=0.10,
    gamma_spin=0.12,
)

LO = np.array([0.1, -0.35, -3.0, -0.23], dtype=np.float32)
HI = np.array([3.0, 0.35, 3.0, 0.23], dtype=np.float32)
SPAN = HI - LO
PAD = np.array([50.0, 50.0], dtype=np.float32)
MIN_CLEAR = 2 * 0.145 + 1e-3

# Searchable physics params: [k_curl, a_linear, gamma_spin, c_damp, c_tangent, mu_tangent, spin_contact]
PHYS_DEFAULT = np.array([
    SIM_PARAMS["k_curl"], SIM_PARAMS["a_linear"], SIM_PARAMS["gamma_spin"],
    SIM_PARAMS["c_damp"], SIM_PARAMS["c_tangent"], SIM_PARAMS["mu_tangent"],
    SIM_PARAMS["spin_contact"],
], dtype=np.float32)
PHYS_LO = np.array([0.04, 0.03, 0.04, 50.0, 5.0, 0.01, 0.02], dtype=np.float32)
PHYS_HI = np.array([0.40, 0.30, 0.40, 500.0, 80.0, 0.20, 0.30], dtype=np.float32)


def separate_overlaps(pts: np.ndarray, passes: int = 6) -> np.ndarray:
    p = pts.copy()
    n = p.shape[0]
    for _ in range(passes):
        moved = False
        for i in range(n):
            for j in range(i + 1, n):
                dx, dy = p[j, 0] - p[i, 0], p[j, 1] - p[i, 1]
                d = math.hypot(dx, dy)
                if d < 1e-9:
                    dx, dy, d = 1e-6, 0.0, 1e-6
                if d < MIN_CLEAR:
                    push = 0.5 * (MIN_CLEAR - d)
                    p[i, 0] -= push * dx / d
                    p[i, 1] -= push * dy / d
                    p[j, 0] += push * dx / d
                    p[j, 1] += push * dy / d
                    moved = True
        if not moved:
            break
    return p


def load_shot(row: pd.Series) -> dict:
    prev_s = np.tile(PAD, (12, 1)).astype(np.float32)
    prev_m = np.zeros(12, dtype=bool)
    next_s = np.tile(PAD, (12, 1)).astype(np.float32)
    next_m = np.zeros(12, dtype=bool)

    for i in range(1, 13):
        px, py = row.get(f"prev_stone_{i}_x_m", np.nan), row.get(f"prev_stone_{i}_y_m", np.nan)
        if pd.notna(px) and abs(px) < 40:
            prev_s[i - 1] = [px, py]
            prev_m[i - 1] = True
        nx, ny = row.get(f"next_stone_{i}_x_m", np.nan), row.get(f"next_stone_{i}_y_m", np.nan)
        ib = row.get(f"next_stone_{i}_inbounds", 0)
        if pd.notna(nx) and abs(nx) < 40 and ib:
            next_s[i - 1] = [nx, ny]
            next_m[i - 1] = True

    if int(np.sum(prev_m)) > 1:
        prev_s[prev_m] = separate_overlaps(prev_s[prev_m])

    added = set(np.where(next_m)[0] + 1) - set(np.where(prev_m)[0] + 1)
    thrower_block = 0
    if len(added) == 1:
        thrower_block = 0 if list(added)[0] <= 6 else 1

    return dict(
        prev_slots=prev_s,
        prev_mask=prev_m,
        thrower_block=thrower_block,
        tgt0=next_s[:6].copy(),
        tgt0m=next_m[:6].copy(),
        tgt1=next_s[6:12].copy(),
        tgt1m=next_m[6:12].copy(),
        est_x=np.array(
            [row["est_speed"], row["est_angle"], row["est_spin"], row["est_y0"]],
            dtype=np.float32,
        ),
        loss=float(row["hard_loss_refine"]),
    )


def make_grid_x01(n_main: int = 25, n_y0: int = 10) -> np.ndarray:
    axes = [np.linspace(0.02, 0.98, n_main) for _ in range(3)]
    axes.append(np.linspace(0.05, 0.95, n_y0))
    grids = np.meshgrid(*axes, indexing="ij")
    uniform = np.stack([g.ravel() for g in grids], axis=1).astype(np.float32)

    primes = [2, 3, 5, 7]
    n_halton = min(n_main * n_main, 4000)
    halton = np.zeros((n_halton, 4), dtype=np.float32)
    for d in range(4):
        base = primes[d]
        for i in range(n_halton):
            idx = i + 1
            denom = 1.0
            val = 0.0
            while idx > 0:
                idx, rem = divmod(idx, base)
                denom /= base
                val += rem * denom
            halton[i, d] = val
    return np.concatenate([uniform, halton], axis=0)


def _preload_nvidia_libs() -> None:
    pyver = f"python{sys.version_info.major}.{sys.version_info.minor}"
    root = pathlib.Path(sys.prefix) / "lib" / pyver / "site-packages" / "nvidia"
    rels = [
        "nvjitlink/lib/libnvJitLink.so.12",
        "cuda_runtime/lib/libcudart.so.12",
        "cuda_nvrtc/lib/libnvrtc.so.12",
        "cublas/lib/libcublasLt.so.12",
        "cublas/lib/libcublas.so.12",
        "cusparse/lib/libcusparse.so.12",
        "cusolver/lib/libcusolver.so.11",
        "cufft/lib/libcufft.so.11",
        "cudnn/lib/libcudnn.so.9",
    ]
    for rel in rels:
        lib = root / rel
        if lib.exists():
            try:
                ctypes.CDLL(str(lib), mode=ctypes.RTLD_GLOBAL)
            except Exception:
                pass


def _generate_perturbed_params(base_cfg: dict, n: int, sigma: float, seed: int = 42) -> list[dict]:
    rng = np.random.default_rng(seed)
    perturbable = ["k_curl", "a_linear", "gamma_spin", "c_damp", "c_tangent", "mu_tangent", "spin_contact"]
    configs = []
    for _ in range(n):
        cfg = dict(base_cfg)
        for key in perturbable:
            cfg[key] = base_cfg[key] * float(np.exp(rng.normal(0, sigma)))
        configs.append(cfg)
    return configs


def worker(
    gpu_id_str: str,
    rows_df: pd.DataFrame,
    grid_n: int,
    grid_y0: int,
    perturb_n: int,
    perturb_sigma: float,
    top_k: int,
    coord_passes: int,
    eval_chunk: int,
    loss_variant: str,
    lbfgs_variant: str,
    lbfgs_maxiter: int,
    result_queue: mp.Queue,
    sim_perturb_n: int = 0,
    sim_perturb_sigma: float = 0.15,
    search_physics: bool = False,
    n_phys_samples: int = 20,
) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id_str
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.90"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

    _preload_nvidia_libs()

    import jax
    import jax.numpy as jnp
    from curling_inverse import build_batched_hard_loss_by_block, build_batched_hard_loss_by_block_flex
    from curling_sim_jax import CurlingParams

    p_refine = CurlingParams(**SIM_PARAMS)
    p_coarse = CurlingParams(**COARSE_PARAMS)
    print(f"[GPU {gpu_id_str}] building loss fns...", flush=True)
    fn_refine = build_batched_hard_loss_by_block(p_refine, loss_variant=loss_variant)
    fn_coarse = build_batched_hard_loss_by_block(p_coarse, loss_variant=loss_variant)
    fn_current = build_batched_hard_loss_by_block(p_refine, loss_variant="current")
    fn_slot_identity = build_batched_hard_loss_by_block(p_refine, loss_variant="slot_identity")

    fn_flex_refine = None
    if search_physics:
        print(f"[GPU {gpu_id_str}] building flex-physics loss fn...", flush=True)
        fn_flex_refine = build_batched_hard_loss_by_block_flex(p_refine, loss_variant=loss_variant)

    perturbed_coarse_fns = []
    perturbed_refine_fns = []
    if sim_perturb_n > 0 and not search_physics:
        perturbed_coarse_cfgs = _generate_perturbed_params(COARSE_PARAMS, sim_perturb_n, sim_perturb_sigma, seed=42)
        perturbed_refine_cfgs = _generate_perturbed_params(SIM_PARAMS, sim_perturb_n, sim_perturb_sigma, seed=42)
        print(f"[GPU {gpu_id_str}] building {sim_perturb_n} perturbed physics configs (coarse+refine)...", flush=True)
        for ci in range(sim_perturb_n):
            p_c = CurlingParams(**perturbed_coarse_cfgs[ci])
            p_r = CurlingParams(**perturbed_refine_cfgs[ci])
            fn_c = build_batched_hard_loss_by_block(p_c, loss_variant=loss_variant)
            fn_r = build_batched_hard_loss_by_block(p_r, loss_variant=loss_variant)
            perturbed_coarse_fns.append(fn_c)
            perturbed_refine_fns.append(fn_r)
            print(f"[GPU {gpu_id_str}]   perturbed config {ci+1}/{sim_perturb_n} compiled", flush=True)

    all_coarse_fns = [fn_coarse] + perturbed_coarse_fns
    all_refine_fns = [fn_refine] + perturbed_refine_fns

    print(f"[GPU {gpu_id_str}] JIT warmup...", flush=True)
    _dp = jnp.tile(jnp.array(PAD), (12, 1))
    _dm = jnp.zeros(12, dtype=jnp.bool_)
    _dt = jnp.tile(jnp.array(PAD), (6, 1))
    _dtm = jnp.zeros(6, dtype=jnp.bool_)
    _x = jnp.ones((eval_chunk, 4), dtype=jnp.float32)
    for fn in all_coarse_fns + all_refine_fns:
        _ = fn(_dp, _dm, jnp.array(0), _dt, _dtm, _dt, _dtm, _x)
    if fn_flex_refine is not None:
        _phys = jnp.ones((eval_chunk, 7), dtype=jnp.float32)
        _ = fn_flex_refine(_dp, _dm, jnp.array(0), _dt, _dtm, _dt, _dtm, _x, _phys)
    print(f"[GPU {gpu_id_str}] JIT done", flush=True)

    grid_x01 = make_grid_x01(grid_n, grid_y0)
    grid_phys_base = (LO + np.clip(grid_x01, 0, 1) * SPAN).astype(np.float32)
    print(f"[GPU {gpu_id_str}] grid: {len(grid_phys_base)} pts, shots: {len(rows_df)}", flush=True)

    def eval_batch(shot: dict, x_batch: np.ndarray, use_coarse: bool = False) -> np.ndarray:
        fn = fn_coarse if use_coarse else fn_refine
        x = jnp.array(x_batch.reshape(-1, 4), dtype=jnp.float32)
        return np.array(
            fn(
                jnp.array(shot["prev_slots"]),
                jnp.array(shot["prev_mask"]),
                jnp.array(shot["thrower_block"]),
                jnp.array(shot["tgt0"]),
                jnp.array(shot["tgt0m"]),
                jnp.array(shot["tgt1"]),
                jnp.array(shot["tgt1m"]),
                x,
            ),
            dtype=np.float32,
        )

    def eval_batch_stochastic_min(shot: dict, x_batch: np.ndarray, use_coarse: bool = False) -> np.ndarray:
        """Evaluate each candidate under all physics configs, return per-candidate minimum."""
        fns = all_coarse_fns if use_coarse else all_refine_fns
        x = jnp.array(x_batch.reshape(-1, 4), dtype=jnp.float32)
        args = (
            jnp.array(shot["prev_slots"]),
            jnp.array(shot["prev_mask"]),
            jnp.array(shot["thrower_block"]),
            jnp.array(shot["tgt0"]),
            jnp.array(shot["tgt0m"]),
            jnp.array(shot["tgt1"]),
            jnp.array(shot["tgt1m"]),
            x,
        )
        best = np.full(len(x_batch), np.inf, dtype=np.float32)
        for fn in fns:
            losses = np.array(fn(*args), dtype=np.float32)
            np.minimum(best, losses, out=best)
        return best

    def eval_batch_flex(shot: dict, x_batch: np.ndarray, phys_batch: np.ndarray) -> np.ndarray:
        """Evaluate with dynamic physics params. x_batch: (B,4), phys_batch: (B,7)."""
        x = jnp.array(x_batch.reshape(-1, 4), dtype=jnp.float32)
        ph = jnp.array(phys_batch.reshape(-1, 7), dtype=jnp.float32)
        return np.array(
            fn_flex_refine(
                jnp.array(shot["prev_slots"]),
                jnp.array(shot["prev_mask"]),
                jnp.array(shot["thrower_block"]),
                jnp.array(shot["tgt0"]),
                jnp.array(shot["tgt0m"]),
                jnp.array(shot["tgt1"]),
                jnp.array(shot["tgt1m"]),
                x,
                ph,
            ),
            dtype=np.float32,
        )

    def eval_batch_with_fn(shot: dict, x_batch: np.ndarray, fn) -> np.ndarray:
        x = jnp.array(x_batch.reshape(-1, 4), dtype=jnp.float32)
        return np.array(
            fn(
                jnp.array(shot["prev_slots"]),
                jnp.array(shot["prev_mask"]),
                jnp.array(shot["thrower_block"]),
                jnp.array(shot["tgt0"]),
                jnp.array(shot["tgt0m"]),
                jnp.array(shot["tgt1"]),
                jnp.array(shot["tgt1m"]),
                x,
            ),
            dtype=np.float32,
        )

    def eval_one_with_fn(shot: dict, x_phys: np.ndarray, fn) -> float:
        return float(eval_batch_with_fn(shot, x_phys.reshape(1, 4), fn)[0])

    def coord_polish(shot: dict, x_init: np.ndarray, stochastic: bool = False) -> tuple[np.ndarray, float]:
        ef = eval_batch_stochastic_min if stochastic else eval_batch
        best = np.clip(x_init.copy(), LO, HI).astype(np.float32)
        best_h = float(ef(shot, best.reshape(1, 4))[0])
        schedules = [
            [0.12, 0.025, 0.40, 0.030],
            [0.06, 0.012, 0.20, 0.015],
            [0.03, 0.006, 0.10, 0.008],
            [0.015, 0.003, 0.05, 0.004],
            [0.008, 0.0015, 0.025, 0.002],
            [0.004, 0.0008, 0.012, 0.001],
            [0.002, 0.0004, 0.006, 0.0005],
        ]
        pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        for steps in schedules:
            steps = np.array(steps, np.float32)
            improved = True
            passes = 0
            while improved and passes < coord_passes:
                improved = False
                passes += 1
                cands = [best.copy()]
                for d in range(4):
                    for s in (-1.0, 1.0):
                        c = best.copy()
                        c[d] = np.clip(c[d] + s * steps[d], LO[d], HI[d])
                        cands.append(c)
                for d0, d1 in pairs:
                    for s0 in (-1.0, 1.0):
                        for s1 in (-1.0, 1.0):
                            c = best.copy()
                            c[d0] = np.clip(c[d0] + s0 * steps[d0], LO[d0], HI[d0])
                            c[d1] = np.clip(c[d1] + s1 * steps[d1], LO[d1], HI[d1])
                            cands.append(c)
                cb = np.unique(np.round(np.array(cands, np.float32), 7), axis=0)
                hb = ef(shot, cb)
                idx = int(np.argmin(hb))
                candidate_h = float(hb[idx])
                if candidate_h + 1e-7 < best_h:
                    best_h = candidate_h
                    best = cb[idx].copy()
                    improved = True
        return best, best_h

    def lbfgs_refine(shot: dict, x_init: np.ndarray, variant: str, maxiter: int) -> tuple[np.ndarray, float]:
        if variant == "none":
            x = np.clip(x_init.copy(), LO, HI).astype(np.float32)
            return x, float(eval_batch(shot, x.reshape(1, 4))[0])
        if variant == "current":
            fn_obj = fn_current
        elif variant == "slot_identity":
            fn_obj = fn_slot_identity
        else:
            raise ValueError(f"unsupported lbfgs variant: {variant}")

        def obj(x64: np.ndarray) -> float:
            x = np.clip(x64.astype(np.float32), LO, HI)
            return eval_one_with_fn(shot, x, fn_obj)

        x0 = np.clip(x_init.copy(), LO, HI).astype(np.float64)
        res = scipy_min(
            obj,
            x0,
            method="L-BFGS-B",
            bounds=list(zip(LO.tolist(), HI.tolist())),
            options=dict(maxiter=maxiter, ftol=1e-9, eps=5e-4),
        )
        x_opt = np.clip(res.x.astype(np.float32), LO, HI)
        # Always score improvements under the active rescue loss.
        return x_opt, float(eval_batch(shot, x_opt.reshape(1, 4))[0])

    results = []
    t_start = time.time()
    for i, (orig_idx, row) in enumerate(rows_df.iterrows(), start=1):
        shot = load_shot(row)
        old_loss = shot["loss"]
        old_x = shot["est_x"]

        curr_x01 = np.clip((old_x - LO) / (SPAN + 1e-8), 0, 1).reshape(1, 4)
        rng = np.random.default_rng(int(orig_idx))
        perturbs = np.clip(
            curr_x01 + rng.normal(0, perturb_sigma, (perturb_n, 4)).astype(np.float32),
            0,
            1,
        )
        grid_phys = np.concatenate(
            [
                grid_phys_base,
                (LO + perturbs * SPAN).astype(np.float32),
                old_x.reshape(1, 4),
            ],
            axis=0,
        )

        # Standard 4D coarse grid search
        grid_losses_c = np.full(len(grid_phys), np.inf, np.float32)
        for s in range(0, len(grid_phys), eval_chunk):
            e = min(len(grid_phys), s + eval_chunk)
            grid_losses_c[s:e] = eval_batch(shot, grid_phys[s:e], use_coarse=True)

        effective_top_k = min(top_k, len(grid_phys))
        top_idx = np.argsort(grid_losses_c)[:effective_top_k]
        top_throw = grid_phys[top_idx]

        if search_physics and fn_flex_refine is not None:
            # For each top-k throw candidate, evaluate under random physics samples
            phys_rng = np.random.default_rng(int(orig_idx) + 9999)
            phys_samples = np.clip(
                phys_rng.lognormal(
                    np.log(PHYS_DEFAULT), sim_perturb_sigma, (n_phys_samples, 7)
                ).astype(np.float32),
                PHYS_LO, PHYS_HI,
            )
            phys_samples = np.concatenate([PHYS_DEFAULT.reshape(1, 7), phys_samples], axis=0)

            # Cross-product: each throw x each physics
            n_throw = len(top_throw)
            n_phys = len(phys_samples)
            throw_cross = np.repeat(top_throw, n_phys, axis=0)
            phys_cross = np.tile(phys_samples, (n_throw, 1))

            cross_losses = np.full(len(throw_cross), np.inf, np.float32)
            for s in range(0, len(throw_cross), eval_chunk):
                e = min(len(throw_cross), s + eval_chunk)
                cross_losses[s:e] = eval_batch_flex(shot, throw_cross[s:e], phys_cross[s:e])

            best_cross_idx = int(np.argmin(cross_losses))
            best_x = throw_cross[best_cross_idx].copy()
            best_phys = phys_cross[best_cross_idx].copy()
            best_loss = float(cross_losses[best_cross_idx])

            # Also try old_x with all physics samples
            old_x_cross = np.tile(old_x.reshape(1, 4), (n_phys, 1))
            old_losses = np.full(n_phys, np.inf, np.float32)
            for s in range(0, n_phys, eval_chunk):
                e = min(n_phys, s + eval_chunk)
                old_losses[s:e] = eval_batch_flex(shot, old_x_cross[s:e], phys_samples[s:e])
            old_best_idx = int(np.argmin(old_losses))
            if old_losses[old_best_idx] < best_loss:
                best_loss = float(old_losses[old_best_idx])
                best_x = old_x.copy()
                best_phys = phys_samples[old_best_idx].copy()

            # Coordinate polish over throw params (with best physics fixed)
            def coord_polish_flex(shot, x_init, phys_vec):
                best_c = np.clip(x_init.copy(), LO, HI).astype(np.float32)
                best_h = float(eval_batch_flex(shot, best_c.reshape(1, 4), phys_vec.reshape(1, 7))[0])
                schedules = [
                    [0.12, 0.025, 0.40, 0.030],
                    [0.06, 0.012, 0.20, 0.015],
                    [0.03, 0.006, 0.10, 0.008],
                    [0.015, 0.003, 0.05, 0.004],
                    [0.008, 0.0015, 0.025, 0.002],
                    [0.004, 0.0008, 0.012, 0.001],
                    [0.002, 0.0004, 0.006, 0.0005],
                ]
                pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
                for steps in schedules:
                    steps = np.array(steps, np.float32)
                    imp = True
                    passes = 0
                    while imp and passes < coord_passes:
                        imp = False
                        passes += 1
                        cands = [best_c.copy()]
                        for d in range(4):
                            for sv in (-1.0, 1.0):
                                c = best_c.copy()
                                c[d] = np.clip(c[d] + sv * steps[d], LO[d], HI[d])
                                cands.append(c)
                        for d0, d1 in pairs:
                            for s0 in (-1.0, 1.0):
                                for s1 in (-1.0, 1.0):
                                    c = best_c.copy()
                                    c[d0] = np.clip(c[d0] + s0 * steps[d0], LO[d0], HI[d0])
                                    c[d1] = np.clip(c[d1] + s1 * steps[d1], LO[d1], HI[d1])
                                    cands.append(c)
                        cb = np.unique(np.round(np.array(cands, np.float32), 7), axis=0)
                        pv = np.tile(phys_vec.reshape(1, 7), (len(cb), 1))
                        hb = eval_batch_flex(shot, cb, pv)
                        idx_b = int(np.argmin(hb))
                        ch = float(hb[idx_b])
                        if ch + 1e-7 < best_h:
                            best_h = ch
                            best_c = cb[idx_b].copy()
                            imp = True
                return best_c, best_h

            polished_x, polished_loss = coord_polish_flex(shot, best_x, best_phys)
            if polished_loss < best_loss:
                best_loss = polished_loss
                best_x = polished_x.copy()
        else:
            top_losses = np.full(effective_top_k, np.inf, np.float32)
            for s in range(0, effective_top_k, eval_chunk):
                e = min(effective_top_k, s + eval_chunk)
                top_losses[s:e] = eval_batch(shot, top_throw[s:e], use_coarse=False)

            best_idx = int(np.argmin(top_losses))
            best_x = top_throw[best_idx].copy()
            best_loss = float(top_losses[best_idx])
            best_phys = PHYS_DEFAULT.copy()

            if old_loss < best_loss:
                best_x = old_x.copy()
                best_loss = old_loss

            polished_x, polished_loss = coord_polish(shot, best_x)
            if polished_loss < best_loss:
                best_loss = polished_loss
                best_x = polished_x.copy()

            if lbfgs_variant != "none":
                try:
                    lbfgs_x, lbfgs_loss = lbfgs_refine(shot, best_x, lbfgs_variant, lbfgs_maxiter)
                    if lbfgs_loss + 1e-7 < best_loss:
                        best_loss = lbfgs_loss
                        best_x = lbfgs_x.copy()
                except Exception:
                    pass

            if best_loss >= old_loss - 1e-5:
                polished_x2, polished_loss2 = coord_polish(shot, old_x)
                if polished_loss2 < best_loss:
                    best_loss = polished_loss2
                    best_x = polished_x2.copy()
                elif lbfgs_variant != "none":
                    try:
                        lbfgs_x2, lbfgs_loss2 = lbfgs_refine(shot, old_x, lbfgs_variant, lbfgs_maxiter)
                        if lbfgs_loss2 + 1e-7 < best_loss:
                            best_loss = lbfgs_loss2
                            best_x = lbfgs_x2.copy()
                    except Exception:
                        pass

        final_orig_loss = float(eval_batch(shot, best_x.reshape(1, 4))[0])
        improved = best_loss < old_loss - 1e-5
        res_dict = dict(
            orig_idx=orig_idx,
            old_loss=old_loss,
            new_loss=best_loss,
            new_loss_orig_physics=final_orig_loss,
            new_speed=float(best_x[0]),
            new_angle=float(best_x[1]),
            new_spin=float(best_x[2]),
            new_y0=float(best_x[3]),
            improved=improved,
        )
        if search_physics:
            for ki, kn in enumerate(["k_curl", "a_linear", "gamma_spin",
                                      "c_damp", "c_tangent", "mu_tangent",
                                      "spin_contact"]):
                res_dict[f"rescue_phys_{kn}"] = float(best_phys[ki])
        results.append(res_dict)

        if i % 50 == 0 or i == len(rows_df):
            n_improved = sum(1 for r in results if r["improved"])
            elapsed = time.time() - t_start
            eta = elapsed / max(i, 1) * (len(rows_df) - i)
            print(
                f"[GPU {gpu_id_str}] {i}/{len(rows_df)} done, {n_improved} improved, ETA {eta/60:.0f}min",
                flush=True,
            )

    result_queue.put((gpu_id_str, results))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--grid-n", type=int, default=25)
    parser.add_argument("--grid-y0", type=int, default=10)
    parser.add_argument("--perturb-n", type=int, default=300)
    parser.add_argument("--perturb-sigma", type=float, default=0.05)
    parser.add_argument("--top-k", type=int, default=500)
    parser.add_argument("--coord-passes", type=int, default=3)
    parser.add_argument("--eval-chunk", type=int, default=8192)
    parser.add_argument("--loss-variant", type=str, default="slot_identity")
    parser.add_argument("--lbfgs-variant", type=str, default="none", choices=["none", "current", "slot_identity"])
    parser.add_argument("--lbfgs-maxiter", type=int, default=60)
    parser.add_argument("--sim-perturb-n", type=int, default=0,
                        help="Number of perturbed physics configs to explore (0=disabled)")
    parser.add_argument("--sim-perturb-sigma", type=float, default=0.15,
                        help="Log-normal std for physics parameter perturbation")
    parser.add_argument("--search-physics", action="store_true",
                        help="Treat sim physics params as searchable per-throw variables")
    parser.add_argument("--n-phys-samples", type=int, default=20,
                        help="Number of random physics samples per throw candidate")
    args = parser.parse_args()

    full = pd.read_csv(args.input_csv, low_memory=False)
    before_counts = {
        ">0.1": int((full["hard_loss_refine"] > 0.1).sum()),
        ">0.25": int((full["hard_loss_refine"] > 0.25).sum()),
        ">0.5": int((full["hard_loss_refine"] > 0.5).sum()),
        ">1.0": int((full["hard_loss_refine"] > 1.0).sum()),
        ">5.0": int((full["hard_loss_refine"] > 5.0).sum()),
    }

    bad_mask = full["hard_loss_refine"] > args.threshold
    solvable_mask = (full["next_in_bounds_N"] > 0) | (full["prev_N"] == 0)
    target = full[bad_mask & solvable_mask].copy()
    print(f"Loaded {len(full)} rows from {args.input_csv}")
    print(f"Target rows > {args.threshold}: {int(bad_mask.sum())}, solvable subset: {len(target)}")

    if len(target) == 0:
        out_csv = pathlib.Path(args.out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        full.to_csv(out_csv, index=False)
        print(f"[done] no rescue needed; wrote {out_csv}")
        return

    gpu_ids = [g.strip() for g in os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3").split(",") if g.strip()]
    print(f"Using {len(gpu_ids)} GPUs: {gpu_ids}")

    chunks = np.array_split(np.arange(len(target)), len(gpu_ids))
    target_indices = target.index.tolist()

    result_queue: mp.Queue = mp.Queue()
    processes = []
    for gi, gpu_id in enumerate(gpu_ids):
        chunk_idx = chunks[gi]
        if len(chunk_idx) == 0:
            continue
        chunk_orig_indices = [target_indices[ci] for ci in chunk_idx]
        chunk_df = target.loc[chunk_orig_indices]
        p = mp.Process(
            target=worker,
            args=(
                gpu_id,
                chunk_df,
                args.grid_n,
                args.grid_y0,
                args.perturb_n,
                args.perturb_sigma,
                args.top_k,
                args.coord_passes,
                args.eval_chunk,
                args.loss_variant,
                args.lbfgs_variant,
                args.lbfgs_maxiter,
                result_queue,
                args.sim_perturb_n,
                args.sim_perturb_sigma,
                args.search_physics,
                args.n_phys_samples,
            ),
        )
        p.start()
        processes.append(p)

    all_results = []
    for _ in processes:
        gpu_id, results = result_queue.get()
        all_results.extend(results)
        print(f"[main] GPU {gpu_id} returned {len(results)} results", flush=True)

    for p in processes:
        p.join()

    result_map = {r["orig_idx"]: r for r in all_results if r["improved"]}
    improved = len(result_map)
    total = len(all_results)
    total_delta = sum(r["old_loss"] - r["new_loss"] for r in all_results if r["improved"])
    print(f"Improved: {improved}/{total} ({100 * improved / max(1, total):.1f}%)")
    print(f"Total loss reduction on rescued subset: {total_delta:.4f}")

    phys_col_names = ["rescue_phys_k_curl", "rescue_phys_a_linear", "rescue_phys_gamma_spin",
                       "rescue_phys_c_damp", "rescue_phys_c_tangent", "rescue_phys_mu_tangent",
                       "rescue_phys_spin_contact"]
    if args.search_physics:
        for col in phys_col_names:
            if col not in full.columns:
                full[col] = np.nan

    for orig_idx, res in result_map.items():
        full.at[orig_idx, "est_speed"] = res["new_speed"]
        full.at[orig_idx, "est_angle"] = res["new_angle"]
        full.at[orig_idx, "est_spin"] = res["new_spin"]
        full.at[orig_idx, "est_y0"] = res["new_y0"]
        full.at[orig_idx, "hard_loss_refine"] = res["new_loss"]
        if args.search_physics:
            for col in phys_col_names:
                full.at[orig_idx, col] = res[col]

    out_csv = pathlib.Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    full.to_csv(out_csv, index=False)

    after_counts = {
        ">0.1": int((full["hard_loss_refine"] > 0.1).sum()),
        ">0.25": int((full["hard_loss_refine"] > 0.25).sum()),
        ">0.5": int((full["hard_loss_refine"] > 0.5).sum()),
        ">1.0": int((full["hard_loss_refine"] > 1.0).sum()),
        ">5.0": int((full["hard_loss_refine"] > 5.0).sum()),
    }
    print(f"[wrote] {out_csv} ({len(full)} rows)")
    print("Before rescue:", before_counts)
    print("After rescue: ", after_counts)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
