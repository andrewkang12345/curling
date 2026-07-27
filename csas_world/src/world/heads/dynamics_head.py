"""Action-conditioned latent dynamics  G(h, a) -> h'.

Used for EfficientZero-style K-step unrolling.  The action is the [-1,1]-box
representation (see :mod:`world.actions`).  G is a residual block so that, at
init, ``G(h, a) ~= h`` -- a stable starting point for the consistency loss that
pulls ``G(E(s), a)`` toward ``E_target(simulate(s, a))``.

G is never used to *generate* training targets (the simulator is authoritative);
it is trained to imitate the simulator's encoded next state, and may later be
used for cheap candidate pre-filtering inside search.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ActionEncoder(nn.Module):
    def __init__(self, action_dim: int = 4, hidden_dim: int = 256, embed_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(action_dim, embed_dim), nn.GELU(),
            nn.Linear(embed_dim, hidden_dim),
        )

    def forward(self, a_box: torch.Tensor) -> torch.Tensor:
        return self.net(a_box)


class LatentDynamics(nn.Module):
    def __init__(self, hidden_dim: int = 256, action_dim: int = 4, embed_dim: int = 64,
                 dropout: float = 0.1):
        super().__init__()
        self.action_encoder = ActionEncoder(action_dim, hidden_dim, embed_dim)
        self.net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)
        # zero-init the residual projection so G(h,a) starts ~ h
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, h: torch.Tensor, a_box: torch.Tensor) -> torch.Tensor:
        a_emb = self.action_encoder(a_box)
        delta = self.net(torch.cat([h, a_emb], dim=-1))
        return self.norm(h + delta)


__all__ = ["ActionEncoder", "LatentDynamics"]
