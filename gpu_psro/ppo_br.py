from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.distributions import Normal, Categorical
from tqdm import tqdm

from .obs_embed import embed_obs
from .policies import ActorCritic, ActorCriticCfg, Policy

@dataclass
class PPOCfg:
    obs_dim: int
    device: str = "cuda"
    total_env_steps: int = 1_000_000
    rollout_horizon: int = 64
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    lr: float = 3e-4
    epochs: int = 3
    minibatch_size: int = 16384
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    hidden: int = 256
    batch_size_envs: int = 8192
    # Reward shaping strength (Δ = opp_sum_dist - our_sum_dist; alive stones only)
    shaping_coef: float = 0.01
    # Checkpointing
    ckpt_dir: str = ""
    ckpt_tag: str = ""
    ckpt_interval: int = 250_000
    export_torchscript: bool = False
    # DDP
    ddp: bool = False
    world_size: int = 1
    rank: int = 0

def _is_dist(cfg: PPOCfg) -> bool:
    return cfg.ddp and dist.is_available() and dist.is_initialized() and cfg.world_size > 1

def _is_main(cfg: PPOCfg) -> bool:
    return (cfg.rank == 0) or (not _is_dist(cfg))

def _allreduce_min_int(x: int, device: torch.device) -> int:
    t = torch.tensor([x], device=device, dtype=torch.int64)
    if dist.is_initialized():
        dist.all_reduce(t, op=dist.ReduceOp.MIN)
    return int(t.item())

def _atanh_clamped(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x = torch.clamp(x, -1.0 + eps, 1.0 - eps)
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))

class PPOBestResponse:
    """
    Best-response PPO with reward shaping:
      shaped_r = env_r + shaping_coef * (sum_dist_opp - sum_dist_ours)
    where distances are to the tee (alive stones only).
    """
    def __init__(self, env_ctor, p: int, opponent: Policy, cfg: PPOCfg):
        self.env_ctor = env_ctor
        self.p = p
        self.opp = opponent
        self.cfg = cfg
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

        base = ActorCritic(ActorCriticCfg(obs_dim=cfg.obs_dim, hidden=cfg.hidden), device=str(self.device))
        if _is_dist(cfg) and self.device.type == "cuda":
            self.net = torch.nn.parallel.DistributedDataParallel(
                base,
                device_ids=[self.device.index],
                output_device=self.device.index,
                find_unused_parameters=False,
                bucket_cap_mb=25,
            )
        else:
            self.net = base

        self.optim = torch.optim.Adam(self.net.parameters(), lr=cfg.lr)

        # local shard env
        self.local_B = self._split_batch_sizes(cfg.batch_size_envs, cfg.world_size, cfg.rank)
        self.env = None if self.local_B <= 0 else self.env_ctor(self.local_B, str(self.device))

        # constants for action transforms (match ActorCritic)
        self.SPEED_SCALE = 1.6
        self.ANGLE_SCALE = 0.2
        self.LOG_SPEED_SCALE = torch.log(torch.tensor(self.SPEED_SCALE, device=self.device))
        self.LOG_ANGLE_SCALE = torch.log(torch.tensor(self.ANGLE_SCALE, device=self.device))
        self.EPS = 1e-6

    @staticmethod
    def _split_batch_sizes(total: int, k: int, r: int) -> int:
        if k <= 1: return total
        base = total // k
        rem = total % k
        return base + (1 if r < rem else 0)

    def _opp_action(self, obs_vec: torch.Tensor):
        with torch.no_grad():
            return self.opp.act(obs_vec, deterministic=False)

    def _maybe_checkpoint(self, total_steps: int):
        if not _is_main(self.cfg): return
        if not self.cfg.ckpt_dir or self.cfg.ckpt_interval <= 0: return
        if total_steps % self.cfg.ckpt_interval != 0: return
        tag = f"{self.cfg.ckpt_tag}_steps{total_steps}"
        path = f"{self.cfg.ckpt_dir}/{tag}.pt"
        actor_to_save = self.net.module if isinstance(self.net, torch.nn.parallel.DistributedDataParallel) else self.net
        torch.save(actor_to_save.state_dict(), path)

    # --- log-probs for squashed Gaussians ---
    def _logp_speed_from_latent(self, z_s, mu_s, std_s):
        normal = Normal(mu_s, std_s)
        t = torch.tanh(z_s)
        log_jac = self.LOG_SPEED_SCALE + torch.log(torch.clamp(1.0 - t * t, min=self.EPS))
        return normal.log_prob(z_s).squeeze(-1) - log_jac.squeeze(-1)

    def _logp_angle_from_latent(self, z_a, mu_a, std_a):
        normal = Normal(mu_a, std_a)
        t = torch.tanh(z_a)
        log_jac = self.LOG_ANGLE_SCALE + torch.log(torch.clamp(1.0 - t * t, min=self.EPS))
        return normal.log_prob(z_a).squeeze(-1) - log_jac.squeeze(-1)

    def _invert_speed_to_latent(self, speed: torch.Tensor) -> torch.Tensor:
        t = torch.clamp(speed / self.SPEED_SCALE - 1.0, -1.0 + self.EPS, 1.0 - self.EPS)
        return _atanh_clamped(t, self.EPS)

    def _invert_angle_to_latent(self, angle: torch.Tensor) -> torch.Tensor:
        t = torch.clamp(angle / self.ANGLE_SCALE, -1.0 + self.EPS, 1.0 - self.EPS)
        return _atanh_clamped(t, self.EPS)

    # --- shaping helper ---
    @torch.no_grad()
    def _distance_delta_reward(self, obs: dict) -> torch.Tensor:
        """
        Δ = (sum of distances for opponent stones) - (sum of distances for our stones)
        alive stones only; distances to the tee.
        """
        pos   = obs["pos"]                   # [B,N,2]
        alive = obs["alive"][..., 0] > 0.0   # [B,N] bool
        team  = obs["team"][..., 0].long()   # [B,N] 0/1

        # Center of the house (tee)
        cx = self.env.sim.sheet.tee_x
        cy = self.env.sim.sheet.tee_y
        center = torch.tensor([cx, cy], device=pos.device, dtype=pos.dtype)

        d = torch.linalg.vector_norm(pos - center.view(1, 1, 2), dim=-1)  # [B,N]
        d = torch.where(alive, d, torch.zeros_like(d))

        ours = (team == self.p)
        opp  = (team == (1 - self.p))

        sum_ours = (d * ours.float()).sum(dim=1)
        sum_opp  = (d * opp.float()).sum(dim=1)

        return (sum_opp - sum_ours)  # [B]

    # --- training ---
    def train(self) -> ActorCritic:
        if self.local_B == 0:
            return self.net.module if isinstance(self.net, torch.nn.parallel.DistributedDataParallel) else self.net

        shard_obs = self.env.reset()
        total_steps = 0

        pbar = tqdm(total=self.cfg.total_env_steps,
                    desc=f"PPO BR (p={self.p}, rank={self.cfg.rank})",
                    unit="env-steps",
                    disable=not _is_main(self.cfg))

        while total_steps < self.cfg.total_env_steps:
            buf_obs, buf_act, buf_logp, buf_val = [], [], [], []
            buf_rew, buf_mask, buf_done = [], [], []

            T_actual = 0
            for _ in range(self.cfg.rollout_horizon):
                obs_vec = embed_obs(shard_obs)

                with torch.no_grad():
                    mu_s_all, std_s_all, mu_a_all, std_a_all, spin_logit_all, v_all = self.net(obs_vec)

                cur_team = shard_obs["aux"][:, 0].long()  # current player to move (still present in sim.aux)
                our_mask = (cur_team == self.p)
                opp_mask = ~our_mask

                actions = torch.zeros(self.local_B, 3, device=self.device)
                logp = torch.zeros(self.local_B, device=self.device)

                # our actions (sample + full logp)
                if our_mask.any():
                    with torch.no_grad():
                        mu_s = mu_s_all[our_mask]
                        std_s = std_s_all[our_mask]
                        mu_a = mu_a_all[our_mask]
                        std_a = std_a_all[our_mask]
                        spin_logit = spin_logit_all[our_mask]

                        z_s = mu_s + std_s * torch.randn_like(mu_s)
                        z_a = mu_a + std_a * torch.randn_like(mu_a)

                        speed = self.SPEED_SCALE * (torch.tanh(z_s).squeeze(-1) + 1.0)
                        angle = self.ANGLE_SCALE * torch.tanh(z_a).squeeze(-1)

                        sp_dist = Categorical(logits=spin_logit)
                        sp_idx = sp_dist.sample()
                        spin = torch.tensor([-1.0, 0.0, 1.0], device=self.device)[sp_idx]

                        logp_spin = sp_dist.log_prob(sp_idx)
                        logp_speed = self._logp_speed_from_latent(z_s, mu_s, std_s)
                        logp_angle = self._logp_angle_from_latent(z_a, mu_a, std_a)
                        logp_total = logp_spin + logp_speed + logp_angle

                        our_actions = torch.stack([speed, angle, spin], dim=-1)

                    actions[our_mask] = our_actions
                    logp[our_mask] = logp_total

                # opponent actions
                if opp_mask.any():
                    with torch.no_grad():
                        opp_act = self._opp_action(obs_vec[opp_mask])
                    actions[opp_mask] = opp_act

                # step env
                shard_obs, env_rew, done = self.env.step(actions)

                # reward shaping
                with torch.no_grad():
                    delta = self._distance_delta_reward(shard_obs)
                    shaped = env_rew + self.cfg.shaping_coef * delta

                # record
                buf_obs.append(obs_vec)
                buf_act.append(actions)
                buf_logp.append(logp)
                buf_val.append(v_all)
                buf_rew.append(shaped)
                buf_mask.append(our_mask.float())
                buf_done.append(done)

                step_increase = self.local_B
                total_steps += step_increase
                T_actual += 1
                if _is_main(self.cfg):
                    pbar.update(step_increase)
                    self._maybe_checkpoint(total_steps)

                if done.any():
                    shard_obs = self.env.reset()

                if total_steps >= self.cfg.total_env_steps:
                    break

            if T_actual == 0:
                continue

            # ======= compute returns with GAE over T_actual =======
            O = torch.stack(buf_obs, dim=0)
            A = torch.stack(buf_act, dim=0)
            LOGP = torch.stack(buf_logp, dim=0)
            V = torch.stack(buf_val, dim=0)
            R = torch.stack(buf_rew, dim=0)
            M = torch.stack(buf_mask, dim=0)
            D = torch.stack(buf_done, dim=0)

            G = torch.zeros_like(R)
            gae = torch.zeros_like(R[0])
            next_v = torch.zeros_like(V[-1])
            for t in reversed(range(T_actual)):
                nonterminal = (D[t] == 0.0).float()
                delta = R[t] + self.cfg.gamma * next_v * nonterminal - V[t]
                gae = delta + self.cfg.gamma * self.cfg.gae_lambda * nonterminal * gae
                G[t] = gae + V[t]
                next_v = V[t]

            T, B = V.shape
            obs_flat = O.reshape(T * B, -1)
            act_flat = A.reshape(T * B, 3)
            logp_old = LOGP.reshape(T * B)
            val_targ = G.reshape(T * B)
            mask_flat = M.reshape(T * B)
            upd_mask = (mask_flat > 0.5)

            obs_upd = obs_flat[upd_mask]
            act_upd = act_flat[upd_mask]
            logp_upd = logp_old[upd_mask]
            val_upd = val_targ[upd_mask]

            N_local = obs_upd.size(0)
            N_sync = _allreduce_min_int(int(N_local), self.device) if dist.is_initialized() else N_local
            if N_sync == 0:
                continue

            obs_upd = obs_upd[:N_sync]
            act_upd = act_upd[:N_sync]
            logp_upd = logp_upd[:N_sync]
            val_upd = val_upd[:N_sync]

            # PPO updates
            N = obs_upd.size(0)
            idx = torch.randperm(N, device=self.device)
            for _ in range(self.cfg.epochs):
                num_steps = max(1, (N + self.cfg.minibatch_size - 1) // self.cfg.minibatch_size)
                for step in range(num_steps):
                    st = step * self.cfg.minibatch_size
                    en = min(st + self.cfg.minibatch_size, N)
                    mb = idx[st:en]
                    if mb.numel() == 0:
                        dummy = sum(p.sum() * 0.0 for p in self.net.parameters())
                        dummy.backward()
                        continue

                    mu_s, std_s, mu_a, std_a, spin_logit, v = self.net(obs_upd[mb])
                    sp_dist = Categorical(logits=spin_logit)

                    spin_vals = act_upd[mb, 2]
                    idxs = (spin_vals == -1.0).long() * 0 + (spin_vals == 0.0).long() * 1 + (spin_vals == 1.0).long() * 2
                    logp_spin_new = sp_dist.log_prob(idxs)

                    speed = act_upd[mb, 0]
                    angle = act_upd[mb, 1]
                    z_s = self._invert_speed_to_latent(speed).unsqueeze(-1)
                    z_a = self._invert_angle_to_latent(angle).unsqueeze(-1)

                    logp_speed_new = self._logp_speed_from_latent(z_s, mu_s, std_s)
                    logp_angle_new = self._logp_angle_from_latent(z_a, mu_a, std_a)

                    logp_new_total = logp_spin_new + logp_speed_new + logp_angle_new
                    ratio = torch.exp(logp_new_total - logp_upd[mb])

                    adv = (val_upd[mb] - v.detach())
                    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                    pg1 = ratio * adv
                    pg2 = torch.clamp(ratio, 1.0 - self.cfg.clip_eps, 1.0 + self.cfg.clip_eps) * adv
                    policy_loss = -torch.min(pg1, pg2).mean()

                    value_loss = F.mse_loss(v, val_upd[mb])
                    ent = sp_dist.entropy().mean()

                    loss = policy_loss + self.cfg.value_coef * value_loss - self.cfg.entropy_coef * ent
                    self.optim.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                    self.optim.step()

        pbar.close()
        return self.net.module if isinstance(self.net, torch.nn.parallel.DistributedDataParallel) else self.net
