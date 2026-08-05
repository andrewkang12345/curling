"""Multi-head loss for csas_world.

Consumes a fixed-shape batch dict (see :mod:`world.replay.schema`) and the
model's K-step unroll, producing the weighted EfficientZero/MuZero loss:

    policy   : human BC NLL + MCTS weighted-action distillation NLL   (root)
    value    : MSE + gaussian-NLL against search/n-step value targets (every step)
    reward   : Huber against per-step reward                          (unrolled)
    outcome  : cross-entropy against the final end-margin bin         (every step)
    consist. : G(E(s),a) vs stop-grad EMA  E_target(simulate(s,a))        (unrolled)
    decoder  : reconstruct the simulator's next board                 (optional)

Every enabled head is evaluated on every batch (targets that are absent are
masked to zero), so no parameter is ever "unused" -- DDP runs with
``find_unused_parameters=False``.
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import LossCfg
from .heads.consistency import consistency_loss
from .heads.policy_head import fullcov_mdn_nll
from .heads.outcome_head import margin_to_bin
from .model import WorldModel


def _live_mask_from_state(x: torch.Tensor) -> torch.Tensor:
    """x:[...,24] -> [...,12] live mask (matches csas in_play heuristic)."""
    stones = x.view(*x.shape[:-1], 12, 2)
    s = stones.sum(-1)
    mx = stones.max(-1).values
    return ((s > 0.001) & (mx < 0.999)).float()


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    denom = mask.sum().clamp_min(1.0)
    return (values * mask).sum() / denom


def compute_losses(model: WorldModel, batch: Dict[str, torch.Tensor],
                   cfg: LossCfg) -> Tuple[torch.Tensor, Dict[str, float]]:
    x0, c0 = batch["x0"], batch["c0"]
    a_raw = batch["a_raw"]                       # [B,K,4]
    a_box = torch.stack([model.raw_to_box(a_raw[:, k]) for k in range(a_raw.shape[1])], dim=1)
    steps = model.unroll(x0, c0, a_box)          # list of K+1 dicts
    K = a_raw.shape[1]
    device = x0.device
    metrics: Dict[str, float] = {}
    total = x0.new_zeros(())

    # ---------------- policy (root) ---------------------------------- #
    pi0, mu0, tril0 = steps[0]["policy"]
    if cfg.policy_bc > 0:
        bc_z = model.raw_to_z(batch["bc_action_raw"])
        bc_nll = fullcov_mdn_nll(pi0, mu0, tril0, bc_z, reduce=False)   # [B]
        bc_loss = _masked_mean(bc_nll, batch["bc_mask"])
        total = total + cfg.policy_bc * bc_loss
        metrics["policy_bc"] = float(bc_loss.detach())
    if cfg.policy_distill > 0:
        M = batch["dist_actions_raw"].shape[1]
        dist_z = model.raw_to_z(batch["dist_actions_raw"].reshape(-1, 4)).reshape(-1, M, 4)
        # expand root policy over M candidates
        pi_e = pi0.unsqueeze(1).expand(-1, M, -1).reshape(-1, pi0.shape[-1])
        mu_e = mu0.unsqueeze(1).expand(-1, M, -1, -1).reshape(-1, mu0.shape[1], mu0.shape[2])
        tr_e = tril0.unsqueeze(1).expand(-1, M, -1, -1, -1).reshape(-1, *tril0.shape[1:])
        nll = fullcov_mdn_nll(pi_e, mu_e, tr_e, dist_z.reshape(-1, 4), reduce=False).reshape(-1, M)
        w = batch["dist_weights"] * batch["dist_mask"].unsqueeze(1)     # [B,M]
        dist_loss = (nll * w).sum() / w.sum().clamp_min(1e-6)
        total = total + cfg.policy_distill * dist_loss
        metrics["policy_distill"] = float(dist_loss.detach())

    # ---------------- value (every step) ----------------------------- #
    if model.value_head is not None and cfg.value > 0:
        vt = batch["value_target"].clamp(-cfg.value_clip, cfg.value_clip)   # [B,K+1]
        vmask = batch["value_mask"]
        if not cfg.value_from_mcts:
            # train value only on realized-ValueDiff records (the value buffer),
            # matching the dedicated baseline; ignore MCTS/sim search-value targets.
            from .replay.schema import SOURCE_VALUE
            is_value_src = (batch["source"] == SOURCE_VALUE).float().unsqueeze(1)
            vmask = vmask * is_value_src
        v_mse = x0.new_zeros(())
        v_nll = x0.new_zeros(())
        for k in range(min(len(steps), K + 1)):
            vm = steps[k].get("value_mean")
            if vm is None:
                continue
            lv = steps[k]["value_logvar"]
            tgt = vt[:, k]
            mk = vmask[:, k]
            v_mse = v_mse + _masked_mean((vm - tgt) ** 2, mk)
            v_nll = v_nll + _masked_mean(0.5 * (torch.exp(-lv) * (tgt - vm) ** 2 + lv), mk)
        v_loss = (v_mse + cfg.value_nll * v_nll) / max(len(steps), 1)
        total = total + cfg.value * v_loss
        metrics["value_mse"] = float((v_mse / max(len(steps), 1)).detach())
        metrics["value_nll"] = float((v_nll / max(len(steps), 1)).detach())

    # ---------------- decision-relevant value RANK loss (EXP-064) ---------- #
    # Deployed selection uses V only to RANK candidate posts; train that directly:
    # on sig-gated plies (teacher confidently preferred top-1 over top-2), require
    # Q(top1) - Q(top2) >= margin, with Q(a) = -V(post(a), next_cond) — exactly the
    # deployed sign convention. rank_* fields are backfilled; legacy shards have
    # rank_mask = 0 and skip this block.
    if (model.value_head is not None and getattr(cfg, "value_rank", 0.0) > 0
            and "rank_pos" in batch):
        rmask = batch["rank_mask"]
        sel = rmask > 0.5
        if sel.any():
            rp = batch["rank_pos"][sel]      # [b,R,24]
            rn = batch["rank_neg"][sel]
            rc = batch["rank_cond"][sel]     # [b,3]
            b, R, _ = rp.shape
            rc_r = rc.unsqueeze(1).expand(b, R, 3).reshape(b * R, 3)
            vp = model.value_head.value(model.encode(rp.reshape(b * R, 24), rc_r)).view(b, R)
            vn = model.value_head.value(model.encode(rn.reshape(b * R, 24), rc_r)).view(b, R)
            gap = (-vp.mean(dim=1)) - (-vn.mean(dim=1))
            margin = float(getattr(cfg, "rank_margin", 0.25))
            r_loss = torch.relu(margin - gap).mean()
            total = total + cfg.value_rank * r_loss
            metrics["value_rank"] = float(r_loss.detach())
            metrics["rank_acc"] = float((gap > 0).float().mean().detach())

    # ---------------- step reward (2-step return, EXP-009) ----------- #
    # Optional disentanglement of the near-term signal from terminal value: the head
    # regresses the 2-step return (rule margin if the end ends within 2 plies, else the
    # value model 2 plies ahead), Huber, per visited state in its to-move perspective.
    if getattr(model, "reward_head", None) is not None and cfg.step_reward > 0:
        rt = batch["reward_target"]          # [B,K]  2-step returns from each visited state k
        rmask = batch["reward_mask"]         # [B,K]
        sa = getattr(cfg, "reward_action_conditioned", False)
        r_loss = x0.new_zeros(())
        nr = 0
        for k in range(K):
            # action-conditioned r(s_k,a_k): predict from the POST-action latent steps[k+1]=G(s_k,a_k).
            # state-conditioned (default): predict the return-from-state-k from the state latent steps[k].
            idx = k + 1 if sa else k
            if idx >= len(steps):
                continue
            rk = steps[idx].get("reward")
            if rk is None:
                continue
            r_loss = r_loss + _masked_mean(F.smooth_l1_loss(rk, rt[:, k], reduction="none"), rmask[:, k])
            nr += 1
        r_loss = r_loss / max(nr, 1)
        total = total + cfg.step_reward * r_loss
        metrics["step_reward"] = float(r_loss.detach())

    # ---------------- outcome (root + valid unrolled steps) ---------- #
    # Per-step scalar reward is intentionally absent: curling reward is terminal-only,
    # so any consistent per-step reward telescopes to the value difference and is
    # redundant with the value head. The outcome head (full margin distribution) is
    # the non-redundant tactical signal.
    if model.outcome_head is not None and cfg.outcome > 0:
        bins = model.cfg.outcome_bins
        root_margin = batch["outcome_margin"]
        root_perspective = batch["c0"][:, 2].round()
        omask = batch["outcome_mask"]
        o_num = x0.new_zeros(())
        o_den = x0.new_zeros(())
        for k in range(len(steps)):
            ol = steps[k].get("outcome")
            if ol is None:
                continue
            if k == 0:
                margin = root_margin
                mk = omask
            else:
                # Stored outcome_margin is in the root side-to-throw perspective.
                # Recurrent states alternate perspective, so flip the target sign
                # whenever the next state's side-to-throw differs from the root.
                step_perspective = batch["next_conds"][:, k - 1, 2].round()
                sign = torch.where(step_perspective == root_perspective, 1.0, -1.0)
                margin = root_margin * sign
                mk = omask * batch["consistency_mask"][:, k - 1]
            otgt = margin_to_bin(margin, bins)
            ce = F.cross_entropy(ol, otgt, reduction="none")
            o_num = o_num + (ce * mk).sum()
            o_den = o_den + mk.sum()
        o_loss = o_num / o_den.clamp_min(1.0)
        total = total + cfg.outcome * o_loss
        metrics["outcome"] = float(o_loss.detach())

    # ---------------- latent consistency (unrolled) ------------------ #
    if model.cfg.use_consistency and cfg.consistency > 0 and K > 0 and model.dynamics is not None:
        next_states = batch["next_states"]   # [B,K,24]
        next_conds = batch["next_conds"]     # [B,K,3]
        cmask = batch["consistency_mask"]    # [B,K]
        c_loss = x0.new_zeros(())
        for k in range(1, len(steps)):
            pred_latent = steps[k]["latent"]
            with torch.no_grad():
                tgt_latent = model.target_encode(next_states[:, k - 1], next_conds[:, k - 1])
            per = consistency_loss(pred_latent, tgt_latent, model.consistency_proj,
                                   mode=model.cfg.consistency_mode)
            # consistency_loss returns a scalar; weight by fraction of valid rows
            frac = cmask[:, k - 1].mean().clamp_min(1e-6)
            c_loss = c_loss + per * frac
        c_loss = c_loss / max(K, 1)
        total = total + cfg.consistency * c_loss
        metrics["consistency"] = float(c_loss.detach())

    # ---------------- physical decoder (optional) -------------------- #
    if model.decoder is not None and cfg.decoder > 0:
        d_loss = x0.new_zeros(())
        # root reconstruction
        pos0, live_logits0 = model.decode(steps[0]["latent"])
        live0 = _live_mask_from_state(x0)
        d_loss = d_loss + _pos_recon(pos0, x0, live0) + \
            F.binary_cross_entropy_with_logits(live_logits0, live0)
        n = 1
        for k in range(1, len(steps)):
            pos, live_logits = model.decode(steps[k]["latent"])
            tgt = batch["next_states"][:, k - 1]
            live = batch["next_live"][:, k - 1]
            valid = batch["consistency_mask"][:, k - 1]
            recon = _pos_recon(pos, tgt, live) + \
                _masked_mean(F.binary_cross_entropy_with_logits(live_logits, live, reduction="none").mean(-1),
                             valid)
            d_loss = d_loss + recon * valid.float().mean().clamp_min(1e-6)
            n += 1
        d_loss = d_loss / n
        total = total + cfg.decoder * d_loss
        metrics["decoder"] = float(d_loss.detach())

    metrics["total"] = float(total.detach())
    return total, metrics


def _pos_recon(pos: torch.Tensor, target: torch.Tensor, live: torch.Tensor) -> torch.Tensor:
    """MSE on live-stone xy. pos/target:[B,24], live:[B,12]."""
    p = pos.view(-1, 12, 2)
    t = target.view(-1, 12, 2)
    m = live.unsqueeze(-1)
    se = ((p - t) ** 2) * m
    return se.sum() / m.sum().clamp_min(1.0) / 2.0


class WorldLossModule(nn.Module):
    """DDP-wrappable module: forward(batch) -> (total_loss, metrics)."""

    def __init__(self, model: WorldModel, loss_cfg: LossCfg):
        super().__init__()
        self.model = model
        self.loss_cfg = loss_cfg

    def forward(self, batch: Dict[str, torch.Tensor]):
        return compute_losses(self.model, batch, self.loss_cfg)


__all__ = ["compute_losses", "WorldLossModule", "_live_mask_from_state"]
