"""Action-space constants and conversions.

Curling action = ``[speed, angle, spin, y0]`` (canonical order from
``csas.common.ACTION_COLS``).  Two normalised representations are used:

  * **box**  : each dim mapped to ``[-1, 1]`` via the physical bounds -- fed to
    the latent dynamics / action encoder (scale-free, no checkpoint dependency).
  * **z**    : standardised by the policy prior's ``action_mean/std`` -- the
    space the full-covariance MDN operates in (used for NLL / sampling).

This module imports neither torch nor jax so it is safe to use everywhere.
The arithmetic works element-wise on numpy arrays and torch tensors alike.
"""
from __future__ import annotations

import numpy as np

# Physical bounds (csas.common.ACTION_*_{MIN,MAX}; full-sheet, 2026-06).
ACTION_LOW = np.array([2.20, -0.1038, -7.0, -0.23], dtype=np.float32)
ACTION_HIGH = np.array([3.01, 0.1038, 7.0, 0.23], dtype=np.float32)
ACTION_NAMES = ("speed", "angle", "spin", "y0")
ACTION_DIM = 4


def raw_to_box(a_raw):
    """Physical action -> [-1, 1] box."""
    low, high = ACTION_LOW, ACTION_HIGH
    return 2.0 * (a_raw - low) / (high - low) - 1.0


def box_to_raw(a_box):
    low, high = ACTION_LOW, ACTION_HIGH
    return (a_box + 1.0) * 0.5 * (high - low) + low


def raw_to_z(a_raw, mean, std):
    return (a_raw - mean) / std


def z_to_raw(a_z, mean, std):
    return a_z * std + mean


def clip_raw(a_raw):
    return np.clip(a_raw, ACTION_LOW, ACTION_HIGH)


def assert_matches_csas() -> None:
    """Sanity-check the hardcoded bounds against csas.common (collection only)."""
    from csas import common  # noqa: WPS433 (heavy import, opt-in)

    lo = np.array([common.ACTION_SPEED_MIN, common.ACTION_ANGLE_MIN,
                   common.ACTION_SPIN_MIN, common.ACTION_Y0_MIN], dtype=np.float32)
    hi = np.array([common.ACTION_SPEED_MAX, common.ACTION_ANGLE_MAX,
                   common.ACTION_SPIN_MAX, common.ACTION_Y0_MAX], dtype=np.float32)
    assert np.allclose(lo, ACTION_LOW), (lo, ACTION_LOW)
    assert np.allclose(hi, ACTION_HIGH), (hi, ACTION_HIGH)


__all__ = [
    "ACTION_LOW", "ACTION_HIGH", "ACTION_NAMES", "ACTION_DIM",
    "raw_to_box", "box_to_raw", "raw_to_z", "z_to_raw", "clip_raw",
    "assert_matches_csas",
]
