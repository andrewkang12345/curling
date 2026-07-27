"""SimSiam-style projector/predictor for latent consistency (EfficientZero).

The latent-consistency loss pulls the *predicted* next latent  G(E(s), a)
toward a stop-gradient EMA-target encoding  E_target(simulate(s, a)).  Following
EfficientZero we pass both through a projector, run the online branch through an
extra predictor, and minimise the negative cosine similarity (stop-grad on the
target branch).  ``mode='mse'`` falls back to a plain latent MSE.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConsistencyProjector(nn.Module):
    def __init__(self, hidden_dim: int = 256, proj_dim: int = 256, dropout: float = 0.0):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, proj_dim), nn.BatchNorm1d(proj_dim), nn.GELU(),
            nn.Linear(proj_dim, proj_dim), nn.BatchNorm1d(proj_dim),
        )
        self.predictor = nn.Sequential(
            nn.Linear(proj_dim, proj_dim // 2), nn.BatchNorm1d(proj_dim // 2), nn.GELU(),
            nn.Linear(proj_dim // 2, proj_dim),
        )

    def project(self, h: torch.Tensor) -> torch.Tensor:
        return self.projector(h)

    def predict(self, h: torch.Tensor) -> torch.Tensor:
        return self.predictor(self.projector(h))


def consistency_loss(pred_latent: torch.Tensor, target_latent: torch.Tensor,
                     proj: ConsistencyProjector, mode: str = "simsiam") -> torch.Tensor:
    """pred_latent = G(E(s),a) (online); target_latent = E_target(s') (already detached)."""
    target_latent = target_latent.detach()
    if mode == "mse":
        return F.mse_loss(pred_latent, target_latent)
    # simsiam negative cosine
    p = proj.predict(pred_latent)
    z = proj.project(target_latent).detach()
    return -F.cosine_similarity(p, z, dim=-1).mean()


__all__ = ["ConsistencyProjector", "consistency_loss"]
