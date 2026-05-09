#!/usr/bin/env python3
"""
Fit a smooth multivariate distribution over physics parameters from the
rescued inverse dataset.

Non-rescued shots (~87%) use the default physics.
Rescued shots (~13%) use the per-throw best-fit physics found during rescue.

Outputs a JSON file describing a multivariate log-normal distribution
(mean and covariance in log-space) that can be sampled during MC scoring.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import pandas as pd

PHYS_KEYS = ["k_curl", "a_linear", "gamma_spin", "c_damp", "c_tangent", "mu_tangent", "spin_contact"]
RESCUE_COLS = [f"rescue_phys_{k}" for k in PHYS_KEYS]
DEFAULT_PHYS = np.array([0.12, 0.10, 0.12, 165.0, 20.0, 0.05, 0.08], dtype=np.float64)
PHYS_LO = np.array([0.04, 0.03, 0.04, 50.0, 5.0, 0.01, 0.02], dtype=np.float64)
PHYS_HI = np.array([0.40, 0.30, 0.40, 500.0, 80.0, 0.20, 0.30], dtype=np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-csv", required=True, help="Rescued inverse CSV")
    ap.add_argument("--out-json", required=True, help="Output physics distribution JSON")
    ap.add_argument("--add-noise-to-default", type=float, default=0.01,
                    help="Small jitter std (log-space) added to default-physics points to avoid singular covariance")
    args = ap.parse_args()

    df = pd.read_csv(args.input_csv, low_memory=False)
    n_total = len(df)

    has_rescue = all(c in df.columns for c in RESCUE_COLS)
    if has_rescue:
        rescued_mask = df[RESCUE_COLS[0]].notna()
        n_rescued = int(rescued_mask.sum())
    else:
        rescued_mask = pd.Series(False, index=df.index)
        n_rescued = 0

    n_default = n_total - n_rescued
    print(f"Total rows: {n_total}, rescued: {n_rescued}, default-physics: {n_default}")

    phys_vectors = np.empty((n_total, len(PHYS_KEYS)), dtype=np.float64)

    phys_vectors[~rescued_mask] = DEFAULT_PHYS[np.newaxis, :]
    if n_rescued > 0:
        for i, col in enumerate(RESCUE_COLS):
            phys_vectors[rescued_mask, i] = df.loc[rescued_mask, col].values

    phys_vectors = np.clip(phys_vectors, PHYS_LO, PHYS_HI)
    log_phys = np.log(phys_vectors)

    rng = np.random.default_rng(42)
    noise_std = args.add_noise_to_default
    default_indices = np.where(~rescued_mask.values)[0]
    if noise_std > 0 and len(default_indices) > 0:
        log_phys[default_indices] += rng.normal(0, noise_std, (len(default_indices), len(PHYS_KEYS)))

    mean_log = np.mean(log_phys, axis=0)
    cov_log = np.cov(log_phys, rowvar=False)

    if cov_log.ndim == 0:
        cov_log = np.array([[float(cov_log)]])

    eigvals = np.linalg.eigvalsh(cov_log)
    min_eig = float(np.min(eigvals))
    if min_eig < 1e-10:
        cov_log += np.eye(len(PHYS_KEYS)) * (1e-8 - min(min_eig, 0))

    dist = {
        "type": "multivariate_lognormal",
        "phys_keys": PHYS_KEYS,
        "mean_log": mean_log.tolist(),
        "cov_log": cov_log.tolist(),
        "clip_lo": PHYS_LO.tolist(),
        "clip_hi": PHYS_HI.tolist(),
        "default_phys": DEFAULT_PHYS.tolist(),
        "stats": {
            "n_total": n_total,
            "n_rescued": n_rescued,
            "n_default": n_default,
            "fraction_rescued": round(n_rescued / max(n_total, 1), 4),
        },
    }

    rescued_phys = phys_vectors[rescued_mask] if n_rescued > 0 else np.empty((0, len(PHYS_KEYS)))
    per_key_stats = {}
    for i, k in enumerate(PHYS_KEYS):
        vals = rescued_phys[:, i] if n_rescued > 0 else np.array([])
        per_key_stats[k] = {
            "default": float(DEFAULT_PHYS[i]),
            "rescued_mean": float(np.mean(vals)) if len(vals) > 0 else float(DEFAULT_PHYS[i]),
            "rescued_std": float(np.std(vals)) if len(vals) > 1 else 0.0,
            "rescued_min": float(np.min(vals)) if len(vals) > 0 else float(DEFAULT_PHYS[i]),
            "rescued_max": float(np.max(vals)) if len(vals) > 0 else float(DEFAULT_PHYS[i]),
        }
    dist["per_key_stats"] = per_key_stats

    out_path = pathlib.Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(dist, indent=2))
    print(f"Wrote physics distribution to {out_path}")
    print(f"Fraction rescued: {dist['stats']['fraction_rescued']:.1%}")
    for k in PHYS_KEYS:
        s = per_key_stats[k]
        print(f"  {k}: default={s['default']:.4f}, rescued mean={s['rescued_mean']:.4f} "
              f"std={s['rescued_std']:.4f} range=[{s['rescued_min']:.4f}, {s['rescued_max']:.4f}]")


if __name__ == "__main__":
    main()
