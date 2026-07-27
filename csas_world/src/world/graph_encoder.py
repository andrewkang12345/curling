"""Shared GraphTF trunk  E(s, c) -> h  and the (warm-startable) policy head.

We reuse the canonical ``PolicyGraphTransformerFullCovMDN`` from csas as the
representation network: its ``.encode(x, c)`` is the trunk, and its
``head / pi / mu / raw_tril`` submodules are the full-covariance Gaussian-mixture
policy head.  Crucially, ``policy_from_latent(h)`` lets us apply that head to an
arbitrary latent vector (the output of the latent-dynamics head), which is what
MuZero/EfficientZero-style K-step unrolling needs.

Loading the full-covariance prior's ``model_state_dict`` warm-starts the trunk
AND the policy head in one shot.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from .config import ModelCfg


def build_prior(cfg: ModelCfg) -> nn.Module:
    """Instantiate the csas full-covariance MDN (trunk + policy head)."""
    from csas.policy_graph_model import PolicyGraphTransformerFullCovMDN

    return PolicyGraphTransformerFullCovMDN(
        input_dim=cfg.input_dim,
        cond_dim=cfg.cond_dim,
        action_dim=cfg.action_dim,
        hidden_dim=cfg.hidden_dim,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        dropout=cfg.dropout,
        n_mixtures=cfg.n_mixtures,
    )


class SharedTrunk(nn.Module):
    """E(s,c) -> h[B,hidden]; plus the policy head applied to any latent."""

    TRUNK_PREFIXES = ("global_token", "node_proj", "cond_proj", "layers")

    def __init__(self, cfg: ModelCfg):
        super().__init__()
        self.cfg = cfg
        self.prior = build_prior(cfg)
        self.hidden_dim = cfg.hidden_dim
        self.n_mixtures = cfg.n_mixtures
        self.action_dim = cfg.action_dim

    # -- representation --------------------------------------------------- #
    def encode(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """x:[B,24], c:[B,3] -> h:[B,hidden] (global-token readout)."""
        return self.prior.encode(x, c)

    # -- policy head on a latent ----------------------------------------- #
    def policy_from_latent(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """h:[B,hidden] -> (pi_logits[B,K], mu[B,K,4], scale_tril[B,K,4,4])."""
        feat = self.prior.head(h)
        pi = self.prior.pi(feat)
        mu = self.prior.mu(feat).view(-1, self.n_mixtures, self.action_dim)
        raw_tril = self.prior.raw_tril(feat).view(-1, self.n_mixtures, self.prior.n_tril)
        scale_tril = self.prior._scale_tril(raw_tril)
        return pi, mu, scale_tril

    def policy(self, x: torch.Tensor, c: torch.Tensor):
        return self.policy_from_latent(self.encode(x, c))

    # -- warm-start ------------------------------------------------------- #
    def load_prior_checkpoint(self, ckpt_path: str, load_policy_head: bool = True) -> dict:
        """Load the full-covariance prior; warm-starts trunk (+ optionally head).

        Returns a report dict with the matched key counts.
        """
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
        if load_policy_head:
            keep = sd
        else:
            keep = {k: v for k, v in sd.items() if k.split(".")[0] in self.TRUNK_PREFIXES}
        missing, unexpected = self.prior.load_state_dict(keep, strict=False)
        return {
            "loaded": len(keep),
            "missing": len(missing),
            "unexpected": len(unexpected),
            "action_mean": ckpt.get("action_mean"),
            "action_std": ckpt.get("action_std"),
        }

    def trunk_state_dict(self) -> dict:
        return {k: v for k, v in self.prior.state_dict().items()
                if k.split(".")[0] in self.TRUNK_PREFIXES}


__all__ = ["SharedTrunk", "build_prior"]
