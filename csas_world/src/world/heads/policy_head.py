"""Multimodal (full-covariance Gaussian-mixture) policy head.

The canonical policy lives inside the shared trunk (warm-started from the prior);
this module provides the *math* applied to its outputs -- a weight-aware
full-covariance MDN NLL and a sampler -- plus a standalone head module for the
ablation where we do not warm-start from the prior.

All operations are on the policy outputs:
    pi_logits  : [B, K]        mixture logits
    mu         : [B, K, A]     component means (standardised z action space)
    scale_tril : [B, K, A, A]  lower-triangular Cholesky factors (Sigma = L Lᵀ)
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

_LOG2PI = math.log(2.0 * math.pi)


def fullcov_mdn_nll(pi_logits: torch.Tensor, mu: torch.Tensor, scale_tril: torch.Tensor,
                    action_z: torch.Tensor, weights: Optional[torch.Tensor] = None,
                    reduce: bool = True) -> torch.Tensor:
    """Negative log-likelihood of ``action_z`` under a full-covariance MDN.

    action_z : [B, A].  weights : [B] optional per-sample weights (distillation).
    Mirrors ``csas.policy_graph_model.PolicyGraphTransformerFullCovMDN.nll``.
    """
    B, K, A = mu.shape
    delta = action_z.unsqueeze(1) - mu                      # [B,K,A]
    whitened = torch.linalg.solve_triangular(
        scale_tril, delta.unsqueeze(-1), upper=False).squeeze(-1)  # [B,K,A]
    quad = (whitened ** 2).sum(-1)                          # [B,K]
    logdet = torch.log(torch.diagonal(scale_tril, dim1=-2, dim2=-1)).sum(-1)  # [B,K]
    comp_logp = -0.5 * quad - logdet - 0.5 * A * _LOG2PI    # [B,K]
    log_pi = F.log_softmax(pi_logits, dim=-1)               # [B,K]
    logp = torch.logsumexp(log_pi + comp_logp, dim=-1)      # [B]
    nll = -logp
    if weights is not None:
        nll = nll * weights
        if reduce:
            return nll.sum() / weights.sum().clamp_min(1e-6)
    return nll.mean() if reduce else nll


@torch.no_grad()
def sample_actions_z(pi_logits: torch.Tensor, mu: torch.Tensor, scale_tril: torch.Tensor,
                     n_samples: int = 1, temperature: float = 1.0,
                     std_scale: float = 1.0) -> torch.Tensor:
    """Draw correlated samples in z space. Returns [B, n_samples, A]."""
    B, K, A = mu.shape
    probs = F.softmax(pi_logits / max(temperature, 1e-6), dim=-1)        # [B,K]
    comp = torch.multinomial(probs, n_samples, replacement=True)        # [B,n]
    idx = comp.unsqueeze(-1).unsqueeze(-1)                              # [B,n,1,1]
    mu_sel = torch.gather(mu.unsqueeze(1).expand(B, n_samples, K, A), 2,
                          idx.expand(B, n_samples, 1, A)).squeeze(2)    # [B,n,A]
    L_sel = torch.gather(
        scale_tril.unsqueeze(1).expand(B, n_samples, K, A, A), 2,
        idx.unsqueeze(-1).expand(B, n_samples, 1, A, A)).squeeze(2)     # [B,n,A,A]
    eps = torch.randn(B, n_samples, A, device=mu.device, dtype=mu.dtype)
    z = mu_sel + std_scale * torch.matmul(L_sel, eps.unsqueeze(-1)).squeeze(-1)
    return z


def map_action_z(pi_logits: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
    """Mean of the most-likely mixture component. Returns [B, A]."""
    k = pi_logits.argmax(dim=-1)                                       # [B]
    return torch.gather(mu, 1, k[:, None, None].expand(-1, 1, mu.shape[-1])).squeeze(1)


class FullCovMDNPolicyHead(nn.Module):
    """Standalone full-covariance MDN head (ablation: independent of the prior)."""

    def __init__(self, hidden_dim: int, action_dim: int = 4, n_mixtures: int = 16,
                 min_log_std: float = -4.5, max_log_std: float = 1.5, max_offdiag: float = 4.0,
                 dropout: float = 0.1):
        super().__init__()
        self.action_dim = action_dim
        self.n_mixtures = n_mixtures
        self.min_log_std = min_log_std
        self.max_log_std = max_log_std
        self.max_offdiag = max_offdiag
        self.n_tril = action_dim * (action_dim + 1) // 2
        self.head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout))
        self.pi = nn.Linear(hidden_dim, n_mixtures)
        self.mu = nn.Linear(hidden_dim, n_mixtures * action_dim)
        self.raw_tril = nn.Linear(hidden_dim, n_mixtures * self.n_tril)
        tril_idx = torch.tril_indices(action_dim, action_dim)
        self.register_buffer("_tril_rows", tril_idx[0], persistent=False)
        self.register_buffer("_tril_cols", tril_idx[1], persistent=False)

    def _scale_tril(self, raw: torch.Tensor) -> torch.Tensor:
        B = raw.shape[0]
        raw = raw.view(B, self.n_mixtures, self.n_tril)
        L = raw.new_zeros(B, self.n_mixtures, self.action_dim, self.action_dim)
        L[..., self._tril_rows, self._tril_cols] = raw
        diag = torch.arange(self.action_dim, device=raw.device)
        d = torch.diagonal(L, dim1=-2, dim2=-1)
        d = torch.exp(d.clamp(self.min_log_std, self.max_log_std))
        off = L.clone()
        off[..., diag, diag] = 0.0
        off = off.clamp(-self.max_offdiag, self.max_offdiag)
        off[..., diag, diag] = d
        return off

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feat = self.head(h)
        pi = self.pi(feat)
        mu = self.mu(feat).view(-1, self.n_mixtures, self.action_dim)
        scale_tril = self._scale_tril(self.raw_tril(feat))
        return pi, mu, scale_tril


__all__ = ["fullcov_mdn_nll", "sample_actions_z", "map_action_z", "FullCovMDNPolicyHead"]
