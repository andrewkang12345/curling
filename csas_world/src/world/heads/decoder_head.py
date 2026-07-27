"""Optional physical next-state decoder  D(h) -> stones.

Reconstructs the 24-d normalised board from a latent, plus per-stone "in play"
logits.  This is the optional grounding head: it forces the latent to retain
geometric information and lets us visualise / sanity-check the learned latents
against the simulator's true next state.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class PhysicalStateDecoder(nn.Module):
    def __init__(self, hidden_dim: int = 256, num_stones: int = 12, dropout: float = 0.1):
        super().__init__()
        self.num_stones = num_stones
        self.body = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
        )
        self.pos_head = nn.Linear(hidden_dim, num_stones * 2)
        self.live_head = nn.Linear(hidden_dim, num_stones)

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.body(h)
        pos = torch.sigmoid(self.pos_head(feat))          # normalised coords in [0,1]
        live_logits = self.live_head(feat)                # [B, num_stones]
        return pos, live_logits


__all__ = ["PhysicalStateDecoder"]
