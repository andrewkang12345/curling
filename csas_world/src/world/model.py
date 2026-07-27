"""WorldModel -- EfficientZero-style multi-head graph-transformer for curling.

Shared trunk  E(s,c)  +  heads:
    policy   : full-covariance Gaussian mixture over (speed, angle, spin, y0)
    value    : scalar Gaussian V(h)
    dynamics : action-conditioned latent transition G(h, a)
    reward   : per-step reward + tactical end-outcome
    decoder  : (optional) physical next-state reconstruction D(h)
    target   : EMA-updated trunk E_target for the latent-consistency loss

Every head is gated by the config so the model degrades cleanly to
"policy/value only", "+ consistency", "+ decoder", etc.

Inference API (MuZero/EfficientZero):
    initial_inference(x, c)            -> latent h0 + root predictions
    recurrent_inference(h, a_box)      -> next latent + per-step predictions
    unroll(x0, c0, a_box_seq)          -> K-step predictions + predicted latents
"""
from __future__ import annotations

import copy
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from . import actions as A
from .config import ModelCfg
from .graph_encoder import SharedTrunk
from .heads.consistency import ConsistencyProjector
from .heads.decoder_head import PhysicalStateDecoder
from .heads.dynamics_head import LatentDynamics
from .heads.outcome_head import OutcomeHead
from .heads.reward_head import StepRewardHead
from .heads.value_head import GaussianValueHead


# Prior's action standardiser (overwritten by warm-start; sane fallback).
_DEFAULT_ACTION_MEAN = [1.2594, 0.00263, -0.01238, 0.00175]
_DEFAULT_ACTION_STD = [0.38965, 0.10121, 0.83654, 0.16043]


class WorldModel(nn.Module):
    def __init__(self, cfg: ModelCfg):
        super().__init__()
        self.cfg = cfg
        self.trunk = SharedTrunk(cfg)
        h = cfg.hidden_dim

        self.value_head = GaussianValueHead(h, cfg.dropout) if cfg.use_value else None
        self.outcome_head = OutcomeHead(h, cfg.outcome_bins, cfg.dropout) if cfg.use_outcome else None
        self.reward_head = StepRewardHead(h, cfg.dropout) if getattr(cfg, "use_step_reward", False) else None
        self.dynamics = LatentDynamics(h, cfg.action_dim, dropout=cfg.dropout) if cfg.use_dynamics else None
        self.decoder = PhysicalStateDecoder(h, dropout=cfg.dropout) if cfg.use_decoder else None
        self.consistency_proj = (
            ConsistencyProjector(h)
            if (cfg.use_consistency and cfg.consistency_mode == "simsiam") else None
        )

        # EMA target trunk (only the encoder path is used). Built when consistency
        # is on and ema_decay < 1 (otherwise stop-grad on the online encoder).
        self._use_ema = cfg.use_consistency and cfg.ema_decay < 1.0
        if self._use_ema:
            self.target_trunk = copy.deepcopy(self.trunk)
            for p in self.target_trunk.parameters():
                p.requires_grad_(False)
        else:
            self.target_trunk = None

        # action standardiser buffers (z <-> raw)
        self.register_buffer("action_mean", torch.tensor(_DEFAULT_ACTION_MEAN, dtype=torch.float32))
        self.register_buffer("action_std", torch.tensor(_DEFAULT_ACTION_STD, dtype=torch.float32))
        self.register_buffer("action_low", torch.tensor(A.ACTION_LOW, dtype=torch.float32))
        self.register_buffer("action_high", torch.tensor(A.ACTION_HIGH, dtype=torch.float32))

    # ------------------------------------------------------------------ #
    # action conversions
    # ------------------------------------------------------------------ #
    def raw_to_z(self, a_raw: torch.Tensor) -> torch.Tensor:
        return (a_raw - self.action_mean) / self.action_std

    def raw_to_box(self, a_raw: torch.Tensor) -> torch.Tensor:
        return 2.0 * (a_raw - self.action_low) / (self.action_high - self.action_low) - 1.0

    def set_action_normaliser(self, mean, std) -> None:
        self.action_mean.copy_(torch.as_tensor(mean, dtype=torch.float32))
        self.action_std.copy_(torch.as_tensor(std, dtype=torch.float32))

    # ------------------------------------------------------------------ #
    # building blocks
    # ------------------------------------------------------------------ #
    def encode(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        return self.trunk.encode(x, c)

    @torch.no_grad()
    def target_encode(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        if self.target_trunk is not None:
            return self.target_trunk.encode(x, c)
        return self.trunk.encode(x, c)  # stop-grad equivalent (called under no_grad)

    def policy(self, h: torch.Tensor):
        return self.trunk.policy_from_latent(h)

    def value(self, h: torch.Tensor):
        if self.value_head is None:
            raise RuntimeError("value head disabled")
        return self.value_head(h)

    def outcome(self, h: torch.Tensor):
        if self.outcome_head is None:
            raise RuntimeError("outcome head disabled")
        return self.outcome_head(h)

    def reward(self, h: torch.Tensor):
        if self.reward_head is None:
            raise RuntimeError("reward head disabled")
        return self.reward_head(h)

    def step_dynamics(self, h: torch.Tensor, a_box: torch.Tensor) -> torch.Tensor:
        if self.dynamics is None:
            raise RuntimeError("dynamics head disabled")
        return self.dynamics(h, a_box)

    def decode(self, h: torch.Tensor):
        if self.decoder is None:
            raise RuntimeError("decoder head disabled")
        return self.decoder(h)

    # ------------------------------------------------------------------ #
    # MuZero inference
    # ------------------------------------------------------------------ #
    def initial_inference(self, x: torch.Tensor, c: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.encode(x, c)
        out = {"latent": h}
        out["policy"] = self.policy(h)
        if self.value_head is not None:
            out["value_mean"], out["value_logvar"] = self.value(h)
        if self.outcome_head is not None:
            out["outcome"] = self.outcome(h)
        if self.reward_head is not None:
            out["reward"] = self.reward(h)
        return out

    def recurrent_inference(self, h: torch.Tensor, a_box: torch.Tensor) -> Dict[str, torch.Tensor]:
        h_next = self.step_dynamics(h, a_box)
        out = {"latent": h_next}
        if self.outcome_head is not None:
            out["outcome"] = self.outcome(h_next)
        out["policy"] = self.policy(h_next)
        if self.value_head is not None:
            out["value_mean"], out["value_logvar"] = self.value(h_next)
        if self.reward_head is not None:
            out["reward"] = self.reward(h_next)
        return out

    def unroll(self, x0: torch.Tensor, c0: torch.Tensor,
               a_box_seq: torch.Tensor) -> List[Dict[str, torch.Tensor]]:
        """x0:[B,24], c0:[B,3], a_box_seq:[B,K,4] -> list of K+1 prediction dicts.

        steps[0] are root predictions; steps[k] (k>=1) are after applying
        a_box_seq[:, k-1].  Each dict carries the predicted latent so the trainer
        can attach the consistency loss against E_target(s_k).
        """
        steps: List[Dict[str, torch.Tensor]] = [self.initial_inference(x0, c0)]
        if self.dynamics is None:
            return steps
        h = steps[0]["latent"]
        K = a_box_seq.shape[1]
        for k in range(K):
            step = self.recurrent_inference(h, a_box_seq[:, k])
            steps.append(step)
            h = step["latent"]
        return steps

    # ------------------------------------------------------------------ #
    # EMA
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def update_ema(self) -> None:
        if self.target_trunk is None:
            return
        d = self.cfg.ema_decay
        for tp, op in zip(self.target_trunk.parameters(), self.trunk.parameters()):
            tp.mul_(d).add_(op.detach(), alpha=1.0 - d)
        for tb, ob in zip(self.target_trunk.buffers(), self.trunk.buffers()):
            tb.copy_(ob)

    # ------------------------------------------------------------------ #
    # warm-start
    # ------------------------------------------------------------------ #
    def warm_start(self, prior_policy_ckpt: Optional[str], value_ckpt: Optional[str]) -> dict:
        report = {}
        if prior_policy_ckpt and (self.cfg.warm_start_trunk or self.cfg.warm_start_policy_head):
            report["trunk"] = self.trunk.load_prior_checkpoint(
                prior_policy_ckpt, load_policy_head=self.cfg.warm_start_policy_head)
            am, as_ = report["trunk"].get("action_mean"), report["trunk"].get("action_std")
            if am is not None and as_ is not None:
                self.set_action_normaliser(am, as_)
        if value_ckpt and self.cfg.warm_start_value_head and self.value_head is not None:
            report["value"] = self.value_head.load_csas_value_head(value_ckpt)
        if self.target_trunk is not None:
            self.target_trunk.load_state_dict(self.trunk.state_dict())
        return report


__all__ = ["WorldModel"]
