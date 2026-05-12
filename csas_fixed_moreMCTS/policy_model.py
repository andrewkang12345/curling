#!/usr/bin/env python3
"""Human throw prior policy: SetTransformer encoder with a Gaussian-mixture head."""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

NUM_STONES = 12


class PolicySetTransformerMDN(nn.Module):
    def __init__(
        self,
        input_dim: int = 24,
        cond_dim: int = 3,
        action_dim: int = 4,
        hidden_dim: int = 256,
        n_layers: int = 4,
        n_heads: int = 4,
        dropout: float = 0.1,
        n_mixtures: int = 12,
        min_log_std: float = -4.5,
        max_log_std: float = 1.5,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.n_mixtures = n_mixtures
        self.min_log_std = float(min_log_std)
        self.max_log_std = float(max_log_std)
        d = hidden_dim

        self.stone_proj = nn.Linear(2, d)
        self.cond_proj = nn.Linear(cond_dim, d)
        self.team_embed = nn.Embedding(2, d)
        self.inplay_embed = nn.Embedding(2, d)
        self.global_token = nn.Parameter(torch.randn(1, 1, d) / math.sqrt(d))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=n_heads,
            dim_feedforward=d * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Dropout(dropout))
        self.pi = nn.Linear(d, n_mixtures)
        self.mu = nn.Linear(d, n_mixtures * action_dim)
        self.log_std = nn.Linear(d, n_mixtures * action_dim)

    def forward(self, x: torch.Tensor, c: torch.Tensor):
        bsz = x.size(0)
        stones = x.view(bsz, NUM_STONES, 2)
        tokens = self.stone_proj(stones)
        team_ids = torch.zeros(NUM_STONES, dtype=torch.long, device=x.device)
        team_ids[6:] = 1
        tokens = tokens + self.team_embed(team_ids).unsqueeze(0)
        inplay = ((stones.sum(dim=-1) > 0.001) & (stones.max(dim=-1).values < 0.999)).long()
        tokens = tokens + self.inplay_embed(inplay)
        global_tok = self.cond_proj(c).unsqueeze(1) + self.global_token.expand(bsz, -1, -1)
        h = self.encoder(torch.cat([global_tok, tokens], dim=1))[:, 0]
        h = self.head(h)
        pi_logits = self.pi(h)
        mu = self.mu(h).view(bsz, self.n_mixtures, self.action_dim)
        log_std = self.log_std(h).view(bsz, self.n_mixtures, self.action_dim)
        log_std = log_std.clamp(self.min_log_std, self.max_log_std)
        return pi_logits, mu, log_std

    def nll_per_sample(self, x: torch.Tensor, c: torch.Tensor, action_z: torch.Tensor) -> torch.Tensor:
        pi_logits, mu, log_std = self(x, c)
        target = action_z.unsqueeze(1)
        inv_var = torch.exp(-2.0 * log_std)
        comp_logp = -0.5 * ((target - mu).pow(2) * inv_var).sum(-1)
        comp_logp = comp_logp - log_std.sum(-1) - 0.5 * self.action_dim * math.log(2.0 * math.pi)
        return -torch.logsumexp(F.log_softmax(pi_logits, dim=-1) + comp_logp, dim=-1)

    def nll(self, x: torch.Tensor, c: torch.Tensor, action_z: torch.Tensor) -> torch.Tensor:
        return self.nll_per_sample(x, c, action_z).mean()

    @torch.no_grad()
    def sample_z(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        n_samples: int,
        temperature: float = 1.25,
        std_scale: float = 1.5,
    ) -> torch.Tensor:
        pi_logits, mu, log_std = self(x, c)
        bsz, n_mix, action_dim = mu.shape
        probs = F.softmax(pi_logits / max(temperature, 1e-6), dim=-1)
        comp = torch.multinomial(probs, num_samples=n_samples, replacement=True)
        gather = comp[:, :, None].expand(bsz, n_samples, action_dim)
        chosen_mu = torch.gather(mu, 1, gather)
        chosen_log_std = torch.gather(log_std, 1, gather)
        eps = torch.randn_like(chosen_mu)
        return chosen_mu + eps * torch.exp(chosen_log_std) * std_scale
