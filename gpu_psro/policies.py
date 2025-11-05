from __future__ import annotations
from dataclasses import dataclass
from typing import List
import torch
import torch.nn as nn
from torch.distributions import Categorical

class Policy:
    """Base: batched act(obs_vec: [B,D]) -> actions [B,3] on same device."""
    def act(self, obs_vec: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        raise NotImplementedError

class MixturePolicy(Policy):
    """
    Samples one subpolicy per batch element according to normalized weights.
    Groups calls by chosen index to minimize overhead.
    """
    def __init__(self, policies: List[Policy], weights: List[float], device: str = "cuda"):
        super().__init__()
        w = torch.tensor(weights, dtype=torch.float32, device=device)
        assert (w >= 0).all() and float(w.sum()) > 0, "Mixture weights must be non-negative and not all zero."
        self.policies = policies
        self.weights = (w / w.sum()).detach()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

    @torch.no_grad()
    def act(self, obs_vec: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        B = obs_vec.shape[0]
        if deterministic:
            # Choose the highest-weight policy deterministically for all rows.
            k = int(torch.argmax(self.weights).item())
            return self.policies[k].act(obs_vec, deterministic=True)

        # Sample which policy to use per row.
        idx = torch.multinomial(self.weights, num_samples=B, replacement=True)
        actions = torch.empty(B, 3, device=obs_vec.device, dtype=torch.float32)
        for k in range(len(self.policies)):
            mask = (idx == k)
            if mask.any():
                a = self.policies[k].act(obs_vec[mask], deterministic=False)
                actions[mask] = a
        return actions

# ---------- Torch Actor-Critic for continuous+discrete hybrid ----------
@dataclass
class ActorCriticCfg:
    obs_dim: int
    hidden: int = 256
    action_std_init: float = 0.4

class ActorCritic(nn.Module, Policy):
    """
    Hybrid policy:
      - speed in [0,3.2] via tanh-affine
      - angle in [-0.2,0.2] via tanh
      - spin categorical over {-1,0,+1}
      - value V(s)
    """
    def __init__(self, cfg: ActorCriticCfg, device: str = "cuda"):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(cfg.obs_dim, cfg.hidden), nn.Tanh(),
            nn.Linear(cfg.hidden, cfg.hidden), nn.Tanh(),
        )
        self.mu_speed = nn.Linear(cfg.hidden, 1)
        self.mu_angle = nn.Linear(cfg.hidden, 1)
        self.log_std_speed = nn.Parameter(torch.log(torch.tensor(cfg.action_std_init)))
        self.log_std_angle = nn.Parameter(torch.log(torch.tensor(cfg.action_std_init)))
        self.spin_logits = nn.Linear(cfg.hidden, 3)
        self.value_head = nn.Linear(cfg.hidden, 1)
        self.to(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")

    def forward(self, obs_vec: torch.Tensor):
        z = self.backbone(obs_vec)
        mu_s = self.mu_speed(z)                 # [B,1]
        mu_a = self.mu_angle(z)                 # [B,1]
        std_s = self.log_std_speed.exp().expand_as(mu_s)
        std_a = self.log_std_angle.exp().expand_as(mu_a)
        spin_logit = self.spin_logits(z)        # [B,3]
        v = self.value_head(z).squeeze(-1)      # [B]
        return mu_s, std_s, mu_a, std_a, spin_logit, v

    @torch.no_grad()
    def act(self, obs_vec: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        mu_s, std_s, mu_a, std_a, spin_logit, _ = self.forward(obs_vec)
        if deterministic:
            z_s = mu_s
            z_a = mu_a
            sp_idx = spin_logit.argmax(dim=-1)
        else:
            z_s = mu_s + std_s * torch.randn_like(mu_s)
            z_a = mu_a + std_a * torch.randn_like(mu_a)
            sp_idx = Categorical(logits=spin_logit).sample()

        # map to env ranges
        speed = 1.6 * (torch.tanh(z_s).squeeze(-1) + 1.0)  # [0,3.2]
        angle = 0.2 * torch.tanh(z_a).squeeze(-1)          # [-0.2,0.2]
        spin = torch.tensor([-1.0, 0.0, 1.0], device=obs_vec.device)[sp_idx]
        return torch.stack([speed, angle, spin], dim=-1)
