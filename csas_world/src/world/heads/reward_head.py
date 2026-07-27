"""Scalar step-reward head  r(h) -> 2-step bootstrapped return.

EXP-009 (EXP-B): disentangles the *near-term* signal from the terminal value head.
The target is the 2-step return from a state (to-move perspective, zero-sum):
rule-based end margin if the end terminates within two plies, else the value
model's estimate at the state two plies ahead. Architecture mirrors the value
head's mean MLP. Trained as an auxiliary head; off unless ``use_step_reward``.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class StepRewardHead(nn.Module):
    def __init__(self, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h).squeeze(-1)


__all__ = ["StepRewardHead"]
