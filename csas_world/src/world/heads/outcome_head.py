"""Tactical end-outcome head.

Curling reward is sparse and *terminal-only*: a throw yields nothing until the end
resolves, when the score equals the end-score differential.  A per-step scalar
reward head is therefore redundant with the value head -- any self-consistent
per-step reward telescopes to the value difference V(s_{t+1}) - V(s_t) and carries
no information the value head doesn't already have.

So this head predicts the *distribution* over the final end-score margin (a
categorical over bins).  That is strictly more information than the value head's
mean (variance, multimodality, P(steal) vs P(blank) vs giving up a big end) and
doubles as a calibrated win/score-probability readout.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class OutcomeHead(nn.Module):
    def __init__(self, hidden_dim: int = 256, outcome_bins: int = 17, dropout: float = 0.1):
        super().__init__()
        self.outcome_bins = outcome_bins
        self.outcome_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, outcome_bins),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.outcome_head(h)


def margin_to_bin(margin: torch.Tensor, outcome_bins: int) -> torch.Tensor:
    """Map a signed score margin in [-(B//2), B//2] to a class index in [0, B)."""
    half = outcome_bins // 2
    return (margin.round().clamp(-half, half) + half).long()


__all__ = ["OutcomeHead", "margin_to_bin"]
