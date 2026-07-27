"""Scalar Gaussian value head  V(h) -> (mean, logvar).

Predicts the end-score differential (zero-sum, perspective = thrower block).
Architecture mirrors ``csas.gnn_models.ValueGraphTransformerGaussianFast``'s
``mean_head`` / ``logvar_head`` exactly, so weights can be warm-started from the
canonical Gaussian value checkpoint.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


def _value_mlp(hidden_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
        nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(),
        nn.Linear(hidden_dim // 2, 1),
    )


class GaussianValueHead(nn.Module):
    def __init__(self, hidden_dim: int = 256, dropout: float = 0.1,
                 min_logvar: float = -6.0, max_logvar: float = 3.5):
        super().__init__()
        self.mean_head = _value_mlp(hidden_dim, dropout)
        self.logvar_head = _value_mlp(hidden_dim, dropout)
        self.min_logvar = min_logvar
        self.max_logvar = max_logvar

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = self.mean_head(h)
        logvar = self.logvar_head(h).clamp(self.min_logvar, self.max_logvar)
        return mean.squeeze(-1), logvar.squeeze(-1)

    def value(self, h: torch.Tensor) -> torch.Tensor:
        return self.mean_head(h).squeeze(-1)

    def load_csas_value_head(self, ckpt_path: str) -> dict:
        """Warm-start mean_head/logvar_head from a csas Gaussian value checkpoint."""
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
        sub = {}
        for k, v in sd.items():
            if k.startswith("mean_head.") or k.startswith("logvar_head."):
                sub[k] = v
        missing, unexpected = self.load_state_dict(sub, strict=False)
        return {"loaded": len(sub), "missing": len(missing), "unexpected": len(unexpected)}


__all__ = ["GaussianValueHead"]
