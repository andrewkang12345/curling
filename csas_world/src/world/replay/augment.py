"""On-the-fly batch augmentation: horizontal flip + team-slot swap.

The canonical baselines (human prior, Gaussian value model) were trained WITH
these augmentations; matching them is required for a fair head-to-head on
value/policy NLL.  Applied to the training batch only (eval stays deterministic),
consistently across every state / action / cond / live field of the unroll
record.

  * flip : mirror stone x across the centerline; negate angle/spin/y0 of every
           action; conds unchanged; value/outcome invariant.
  * swap : exchange stone blocks 1-6 <-> 7-12 and flip cond[2] (stone_block);
           actions and value/outcome invariant.
"""
from __future__ import annotations

from typing import Dict

import torch

FLIP_CENTER_X_NORM = 1500.0 / 4095.0


def _flip_state(x: torch.Tensor) -> torch.Tensor:
    s = x.reshape(*x.shape[:-1], 12, 2).clone()
    live = (s[..., 0] < 0.999) | (s[..., 1] < 0.999)
    fx = FLIP_CENTER_X_NORM - s[..., 0]
    s[..., 0] = torch.where(live, fx, s[..., 0])
    return s.reshape(x.shape)


def _swap_state(x: torch.Tensor) -> torch.Tensor:
    s = x.reshape(*x.shape[:-1], 12, 2)
    out = torch.cat([s[..., 6:, :], s[..., :6, :]], dim=-2)
    return out.reshape(x.shape)


def _neg_action(a: torch.Tensor) -> torch.Tensor:
    out = a.clone()
    out[..., 1:4] = -out[..., 1:4]
    return out


@torch.no_grad()
def augment_batch(batch: Dict[str, torch.Tensor], p_flip: float = 0.5,
                  p_swap: float = 0.5) -> Dict[str, torch.Tensor]:
    B = batch["x0"].shape[0]
    dev = batch["x0"].device
    flip = torch.rand(B, device=dev) < p_flip
    swap = torch.rand(B, device=dev) < p_swap

    def m(mask, ndim):  # broadcast a [B] bool mask to ndim
        return mask.view(B, *([1] * (ndim - 1)))

    # ---- flip ----
    if bool(flip.any()):
        batch["x0"] = torch.where(m(flip, 2), _flip_state(batch["x0"]), batch["x0"])
        batch["next_states"] = torch.where(m(flip, 3), _flip_state(batch["next_states"]),
                                           batch["next_states"])
        for key in ("bc_action_raw", "a_raw", "dist_actions_raw"):
            a = batch[key]
            batch[key] = torch.where(m(flip, a.dim()), _neg_action(a), a)

    # ---- team-slot swap ----
    if bool(swap.any()):
        batch["x0"] = torch.where(m(swap, 2), _swap_state(batch["x0"]), batch["x0"])
        batch["next_states"] = torch.where(m(swap, 3), _swap_state(batch["next_states"]),
                                           batch["next_states"])
        nl = batch["next_live"]                       # [B,K,12] swap halves
        nl_sw = torch.cat([nl[..., 6:], nl[..., :6]], dim=-1)
        batch["next_live"] = torch.where(m(swap, 3), nl_sw, nl)
        c0 = batch["c0"].clone(); c0[:, 2] = 1.0 - c0[:, 2]
        batch["c0"] = torch.where(m(swap, 2), c0, batch["c0"])
        nc = batch["next_conds"].clone(); nc[..., 2] = 1.0 - nc[..., 2]
        batch["next_conds"] = torch.where(m(swap, 3), nc, batch["next_conds"])
    return batch


__all__ = ["augment_batch"]
