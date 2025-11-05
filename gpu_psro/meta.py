# gpu_psro/meta.py
from __future__ import annotations
import numpy as np
from dataclasses import dataclass

@dataclass
class MWUConfig:
    iters: int = 2000
    eta: float = 0.2

@dataclass
class MetaSolveResult:
    x: np.ndarray
    y: np.ndarray
    value: float

def solve_zero_sum_minimax_mwu(payoff: np.ndarray, cfg: MWUConfig = MWUConfig()) -> MetaSolveResult:
    m, n = payoff.shape
    x = np.ones(m) / m
    y = np.ones(n) / n
    wx = np.ones(m); wy = np.ones(n)
    for _ in range(cfg.iters):
        u_rows = payoff @ y
        v_cols = -(x @ payoff)
        wx *= np.exp(cfg.eta * u_rows)
        wy *= np.exp(cfg.eta * v_cols)
        x = wx / wx.sum()
        y = wy / wy.sum()
    return MetaSolveResult(x=x, y=y, value=float(x @ payoff @ y))
