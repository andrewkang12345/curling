"""Local execution noise for throws (Bowling-et-al. fitted model).

Mirrors `csas` LocalNoise (configs/noise/v1_bowling.json): Student-t(nu=5) per-dim
noise, speed scale 9.5mm/s, speed-dependent aim scale, guessed spin/y0 noise.
Used to average each candidate's value over N noisy executions during search /
target collection, exactly as csas_fixed_moreMCTS does (`--noise_samples`).
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from ..actions import ACTION_HIGH, ACTION_LOW


class LocalNoise:
    def __init__(self, path: str, seed: int = 0):
        self.cfg = json.loads(Path(path).read_text())
        self.block = self.cfg.get("local", {})
        self.min_std = float(self.block.get("min_std", 1e-3))
        self.rng = np.random.default_rng(seed)

    def sample_batch(self, centers: np.ndarray, n: int, crn: bool = False) -> np.ndarray:
        """centers [...,4] -> [..., n, 4] noisy executions (clipped to bounds).

        ``crn=True`` (common random numbers): the SAME n underlying standard
        draws are applied to every center (scales stay per-center), so
        candidate comparisons become paired — the Q-gap variance between
        candidates drops sharply at the same n. Use for ranking/screening;
        keep False for realized play.
        """
        centers = np.asarray(centers, dtype=np.float32)
        flat = centers.reshape(-1, 4)
        zn = 1 if crn else len(flat)
        b = self.block
        std = np.maximum(np.asarray(b.get("std", [0.0123, 0.003, 0.08, 0.015]), np.float32), self.min_std)
        if str(b.get("distribution", "gaussian")).lower() == "student_t":
            nu = float(b.get("nu", 5.0))
            z = self.rng.standard_t(df=nu, size=(zn, int(n), 4)).astype(np.float32)
            scales = np.broadcast_to(std[None, :], (len(flat), 4)).copy()
            scales *= math.sqrt((nu - 2.0) / nu)
            if "speed_scale" in b:
                scales[:, 0] = max(float(b["speed_scale"]), self.min_std)
            if "angle_speed_range" in b and "angle_scale_range" in b:
                sr = np.asarray(b["angle_speed_range"], np.float32)
                cr = np.asarray(b["angle_scale_range"], np.float32)
                spd = np.clip(np.abs(flat[:, 0]), float(sr[0]), float(sr[1]))
                denom = float(sr[1] - sr[0])
                frac = np.zeros_like(spd) if denom <= 0 else (spd - sr[0]) / denom
                scales[:, 1] = np.maximum(cr[1] + frac * (cr[0] - cr[1]), self.min_std)
            out = flat[:, None, :] + z * scales[:, None, :]
        else:
            out = flat[:, None, :] + self.rng.normal(0.0, std, size=(zn, int(n), 4)).astype(np.float32)
        out = np.clip(out, ACTION_LOW[None, None, :], ACTION_HIGH[None, None, :])
        return out.reshape(*centers.shape[:-1], int(n), 4).astype(np.float32)


def make_noise(config_path: Optional[str], seed: int = 0) -> Optional[LocalNoise]:
    if not config_path:
        return None
    p = Path(config_path)
    if not p.exists():
        # allow paths relative to csas_v3
        from .._bootstrap import CSAS_V3_ROOT
        p = CSAS_V3_ROOT / config_path
    if not p.exists():
        return None
    return LocalNoise(str(p), seed)


__all__ = ["LocalNoise", "make_noise"]
