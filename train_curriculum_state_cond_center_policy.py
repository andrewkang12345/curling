# train_curriculum_stones_policy.py
from __future__ import annotations
import argparse, os, math, time, random
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from torch_curling_batched import TorchBatchedCurlingEnv, CurlingConfig, Sheet
from gpu_psro.policies import ActorCritic, ActorCriticCfg
from gpu_psro.obs_embed import embed_obs

# --------------------------------------------------------------------------------------
# DDP utils
# --------------------------------------------------------------------------------------
def ddp_is_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()

def ddp_setup():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))

def ddp_rank() -> int:
    return dist.get_rank() if ddp_is_initialized() else 0

def ddp_world() -> int:
    return dist.get_world_size() if ddp_is_initialized() else 1

def ddp_barrier():
    if ddp_is_initialized():
        dist.barrier()

def all_reduce_mean(t: torch.Tensor) -> torch.Tensor:
    if ddp_is_initialized():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        t /= ddp_world()
    return t

def maybe_print(msg: str):
    if ddp_rank() == 0:
        print(msg, flush=True)

# --------------------------------------------------------------------------------------
# Policy/action mapping — must match ActorCritic heads
# --------------------------------------------------------------------------------------
SPEED_SCALE = 1.6   # speed = 1.6 * (tanh(z_s) + 1)   in [0, 3.2]
ANGLE_SCALE = 0.8   # angle = 0.8 * tanh(z_a)         approx [-0.8, 0.8]

def atanh_clamped(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x = torch.clamp(x, -1.0 + eps, 1.0 - eps)
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))

def speed_to_latent(speed: torch.Tensor) -> torch.Tensor:
    t = speed / SPEED_SCALE - 1.0
    return atanh_clamped(t)

def angle_to_latent(angle: torch.Tensor) -> torch.Tensor:
    t = angle / ANGLE_SCALE
    return atanh_clamped(t)

def logits_to_spin(spin_idx: torch.Tensor) -> torch.Tensor:
    """map {0,1,2} -> {-1,0,+1} (float)"""
    return torch.where(spin_idx == 0, -torch.ones_like(spin_idx),
           torch.where(spin_idx == 1, torch.zeros_like(spin_idx),
                       torch.ones_like(spin_idx))).float()

# --------------------------------------------------------------------------------------
# Prior
# --------------------------------------------------------------------------------------
def prior_action(batch: int,
                 device: str,
                 btn_speed: float,
                 btn_angle: float,
                 btn_spin_class: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                                               torch.Tensor, torch.Tensor]:
    speed_prior = torch.full((batch, 1), btn_speed, device=device, dtype=torch.float32)
    angle_prior = torch.full((batch, 1), btn_angle, device=device, dtype=torch.float32)
    spin_val = {-1: -1.0, 0: 0.0, 1: 1.0}[int(np.sign(btn_spin_class)) if btn_spin_class in (-1, 0, 1) else 0]
    spin_prior = torch.full((batch, 1), spin_val, device=device, dtype=torch.float32)

    z_s_prior = speed_to_latent(speed_prior)
    z_a_prior = angle_to_latent(angle_prior)
    return speed_prior, angle_prior, spin_prior, z_s_prior, z_a_prior

# --------------------------------------------------------------------------------------
# Sampling with residuals around the prior in LATENT space
# --------------------------------------------------------------------------------------
def sample_action_and_logprob_residual(mu_s, std_s, mu_a, std_a, spin_logits,
                                       z_s_prior, z_a_prior):
    eps_s = torch.randn_like(mu_s)
    eps_a = torch.randn_like(mu_a)
    z_s = z_s_prior + (mu_s + std_s * eps_s)
    z_a = z_a_prior + (mu_a + std_a * eps_a)   # (bugfix: z_a_prior, not z_s_prior)

    dist_spin = torch.distributions.Categorical(logits=spin_logits)
    spin_idx = dist_spin.sample()  # [B]
    spin_code = logits_to_spin(spin_idx).unsqueeze(-1)

    speed = SPEED_SCALE * (torch.tanh(z_s) + 1.0)
    angle = ANGLE_SCALE * torch.tanh(z_a)
    act = torch.cat([speed, angle, spin_code], dim=-1)

    dist_s = torch.distributions.Normal(z_s_prior + mu_s, std_s)
    dist_a = torch.distributions.Normal(z_a_prior + mu_a, std_a)
    logp_old = dist_s.log_prob(z_s).squeeze(-1) + dist_a.log_prob(z_a).squeeze(-1) + dist_spin.log_prob(spin_idx)

    ent = dist_s.entropy().squeeze(-1) + dist_a.entropy().squeeze(-1) + dist_spin.entropy()
    return act, logp_old, ent, z_s.detach(), z_a.detach(), spin_idx.detach(), speed.detach(), angle.detach()

def logprob_given_latents_residual(mu_s, std_s, mu_a, std_a, spin_logits,
                                   z_s_prior, z_a_prior,
                                   z_s, z_a, spin_idx):
    dist_s = torch.distributions.Normal(z_s_prior + mu_s, std_s)
    dist_a = torch.distributions.Normal(z_a_prior + mu_a, std_a)
    dist_spin = torch.distributions.Categorical(logits=spin_logits)
    lp = dist_s.log_prob(z_s).squeeze(-1) + dist_a.log_prob(z_a).squeeze(-1) + dist_spin.log_prob(spin_idx)
    return lp

# --------------------------------------------------------------------------------------
# Reward: negative distance of thrown stone to tee
# --------------------------------------------------------------------------------------
@torch.no_grad()
def thrown_distance_reward(env: TorchBatchedCurlingEnv,
                           idx_added: torch.Tensor) -> torch.Tensor:
    sheet = Sheet()
    pos = env.sim.pos
    alive = env.sim.alive[..., 0] > 0.0
    B = pos.shape[0]
    b = torch.arange(B, device=env.device)
    idx = idx_added.clamp(min=0, max=pos.shape[1] - 1)
    xy = pos[b, idx, :]
    d = torch.linalg.vector_norm(xy - torch.tensor([sheet.tee_x, sheet.tee_y], device=env.device), dim=-1)
    is_alive = alive[b, idx]
    d0 = float(sheet.house_radii[-1])  # 12-ft ring
    reward = -(d/d0)**2
    reward = torch.where(is_alive, reward, torch.full_like(reward, -10.0))
    return reward

# --------------------------------------------------------------------------------------
# Rollout: one-step episode using residual prior
# --------------------------------------------------------------------------------------
@torch.no_grad()
def rollout_one_throw(net_eval: ActorCritic,
                      env: TorchBatchedCurlingEnv,
                      placement: str,
                      exec_noise_sigma: float,
                      btn_speed: float,
                      btn_angle: float,
                      btn_spin_class: int):
    env.set_execution_noise(exec_noise_sigma)
    env.reset(placements=[placement] * env.B)

    idx_added = env.sim.count_stones.clone()
    s = env.sim._obs()
    obs_vec = embed_obs(s)

    speed_p, angle_p, spin_p, z_s_p, z_a_p = prior_action(env.B, env.device, btn_speed, btn_angle, btn_spin_class)

    mu_s, std_s, mu_a, std_a, spin_logits, value = net_eval(obs_vec)
    act, logp_old, ent, z_s, z_a, spin_idx, speed_out, angle_out = sample_action_and_logprob_residual(
        mu_s, std_s, mu_a, std_a, spin_logits, z_s_p, z_a_p
    )

    _ = env.step(act)

    reward = thrown_distance_reward(env, idx_added)

    alive = env.sim.alive[..., 0] > 0.0
    b = torch.arange(env.B, device=env.device)
    alive_ratio = alive[b, idx_added].float().mean()

    return {
        "obs": obs_vec,
        "value": value.squeeze(-1),
        "logp_old": logp_old,
        "entropy": ent,
        "reward": reward,
        "z_s": z_s, "z_a": z_a,
        "spin_idx": spin_idx,
        "alive_ratio": alive_ratio,
        "z_s_prior": z_s_p, "z_a_prior": z_a_p,
    }

@torch.no_grad()
def _sanity_print_prior(btn_speed: float, btn_angle: float, btn_spin_class: int,
                        device: str, cfg: CurlingConfig):
    """Fire one throw with the fixed prior under the exact training cfg."""
    spin_val = {-1: -1.0, 0: 0.0, 1: 1.0}[int(np.sign(btn_spin_class)) if btn_spin_class in (-1,0,1) else 0]
    env1 = TorchBatchedCurlingEnv(B=1, device=device, cfg=cfg)
    env1.reset(placements=["EMPTY"])
    # index of the stone that will be thrown
    idx = int(env1.sim.count_stones.item())
    a = torch.tensor([[btn_speed, btn_angle, spin_val]], device=device, dtype=torch.float32)
    env1.step(a)
    y_final = float(env1.sim.pos[0, idx, 1].item())
    x_final = float(env1.sim.pos[0, idx, 0].item())
    alive = bool(env1.sim.alive[0, idx, 0].item() > 0.0)
    sheet = Sheet()
    print(f"[prior:test] speed={btn_speed:.3f}, angle={btn_angle:.3f}, spin={spin_val:+.0f}  "
          f"-> y={y_final:.3f}, x={x_final:.3f}, alive={alive}  (tee_y={sheet.tee_y:.3f})", flush=True)

# --------------------------------------------------------------------------------------
# Curriculum sampler
# --------------------------------------------------------------------------------------
def pick_curriculum_name(step: int, total_steps: int) -> str:
    """
    Progressive curriculum up to CURRICULUM_11.
    At each stage, include a mix of previous counts.
    """
    frac = step / max(1, total_steps)
    max_k = min(11, max(1, int(math.ceil(frac * 11))))
    if frac < 0.10 and random.random() < 0.30:
        return "EMPTY"
    if max_k == 1:
        return "CURRICULUM_1"
    k = max_k if random.random() < 0.7 else random.randint(1, max_k - 1)
    return f"CURRICULUM_{k}"

# --------------------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------------------
def train(args):
    # DDP setup
    ddp_setup()
    rank = ddp_rank()
    world = ddp_world()

    # Device
    device = "cuda" if (args.device.startswith("cuda") and torch.cuda.is_available()) else "cpu"
    local_dev = device
    if ddp_is_initialized():
        local_dev = f"cuda:{int(os.environ.get('LOCAL_RANK','0'))}"

    # Prior config
    btn_speed = args.btn_speed
    if rank == 0:
        print(f"[prior] Using fixed prior: speed={btn_speed:.6f}, angle={args.btn_angle:.4f}, spin={args.btn_spin}")
        print(f"[train] Execution noise fixed at: {args.exec_noise_sigma}")

    # Env per-rank
    total_B = args.batch_size
    B_per_rank = (total_B + world - 1) // world
    cfg = CurlingConfig(
        stones_per_team=1,
        team0_hammer=True,
        use_preplaced=False,
        use_powerplay=False,
        noise_speed=args.exec_noise_sigma,
        noise_angle=args.exec_noise_sigma * 0.2,
        noise_spin_flip_prob=args.noise_spin_flip_prob,
        # safer defaults for speed
        max_sim_steps=3000,
        v_stop=0.05,
        dt=0.02,
        # placement cache sizes (affects fix #2)
        curriculum_cache_per_k=128,
        curriculum_cache_seed=args.seed + rank,
    )

    if ddp_rank() == 0:
        _sanity_print_prior(btn_speed=args.btn_speed,
                            btn_angle=args.btn_angle,
                            btn_spin_class=args.btn_spin,
                            device=(f"cuda:{int(os.environ.get('LOCAL_RANK','0'))}" if torch.cuda.is_available() else "cpu"),
                            cfg=cfg)
                            
    env = TorchBatchedCurlingEnv(B=B_per_rank, device=local_dev, cfg=cfg)

    # Detect obs_dim
    env.reset(placements=["EMPTY"] * env.B)
    s_probe = env.sim._obs()
    obs_vec = embed_obs(s_probe)
    obs_dim = int(obs_vec.shape[-1])
    if rank == 0:
        print(f"[init] world={world}  B={total_B}  B_per_rank={B_per_rank}  obs_dim={obs_dim}", flush=True)

    # Model / opt
    net = ActorCritic(ActorCriticCfg(obs_dim=obs_dim, hidden=args.hidden), device=local_dev)
    if ddp_is_initialized():
        net = DDP(net, device_ids=[int(os.environ.get('LOCAL_RANK','0'))],
                       output_device=int(os.environ.get('LOCAL_RANK','0')))
    optim = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Seeding
    seed = args.seed + rank
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    ema_alive = None
    ema_dist = None
    t0 = time.time()

    for step in range(1, args.train_steps + 1):
        current_placement = pick_curriculum_name(step, args.train_steps)

        # Rollout with prior
        net_eval = net.module if isinstance(net, DDP) else net
        rollout = rollout_one_throw(
            net_eval, env,
            placement=current_placement,
            exec_noise_sigma=args.exec_noise_sigma,
            btn_speed=btn_speed,
            btn_angle=args.btn_angle,
            btn_spin_class=args.btn_spin
        )

        obs = rollout["obs"]
        value_old = rollout["value"]
        logp_old = rollout["logp_old"].detach()
        reward = rollout["reward"].detach()
        alive_ratio = rollout["alive_ratio"].detach()
        z_s = rollout["z_s"]; z_a = rollout["z_a"]; spin_idx = rollout["spin_idx"]
        z_s_prior = rollout["z_s_prior"]; z_a_prior = rollout["z_a_prior"]

        # Advantage
        adv = (reward - value_old).detach()

        # Forward for update (call underlying module when DDP to avoid extra autograd wrappers)
        raw_net = net.module if isinstance(net, DDP) else net
        mu_s, std_s, mu_a, std_a, spin_logits, value_pred = raw_net(obs)

        # Logprob of SAME latents
        logp_new = logprob_given_latents_residual(
            mu_s, std_s, mu_a, std_a, spin_logits,
            z_s_prior, z_a_prior, z_s, z_a, spin_idx
        )

        # PPO objective
        ratio = torch.exp(logp_new - logp_old)
        pg_loss = torch.max(-adv * ratio, -adv * torch.clamp(ratio, 1.0 - args.clip, 1.0 + args.clip)).mean()

        # Value loss
        v_loss = torch.mean((value_pred.squeeze(-1) - reward) ** 2)

        # Entropy bonus
        dist_s = torch.distributions.Normal(z_s_prior + mu_s, std_s)
        dist_a = torch.distributions.Normal(z_a_prior + mu_a, std_a)
        dist_spin = torch.distributions.Categorical(logits=spin_logits)
        entropy = (dist_s.entropy().squeeze(-1) + dist_a.entropy().squeeze(-1) + dist_spin.entropy()).mean()
        ent_loss = -entropy

        loss = pg_loss + args.v_coef * v_loss + args.ent_coef * ent_loss

        # Logs (rank-avg)
        with torch.no_grad():
            sheet = Sheet()
            pos = env.sim.pos
            alive_mat = env.sim.alive[..., 0] > 0.0
            idx_added = env.sim.count_stones - 1
            b = torch.arange(env.B, device=local_dev)
            idx = idx_added.clamp(0, pos.shape[1] - 1)
            xy = pos[b, idx, :]
            dist_now = torch.linalg.vector_norm(xy - torch.tensor([sheet.tee_x, sheet.tee_y], device=local_dev), dim=-1)
            mask = alive_mat[b, idx]
            alive_mean_dist = dist_now[mask].mean() if mask.any() else torch.tensor(float("inf"), device=local_dev)

            logs = torch.tensor([
                pg_loss.item(), v_loss.item(), entropy.item(),
                reward.mean().item(), alive_ratio.item(),
            ], device=local_dev)
            logs = all_reduce_mean(logs)
            alive_mean_dist = all_reduce_mean(alive_mean_dist.unsqueeze(0)).squeeze(0)

        # Optimize
        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), args.max_grad_norm)
        optim.step()

        if ddp_rank() == 0 and (step % args.log_every == 0 or step == 1):
            ema_alive = (0.9 * ema_alive + 0.1 * logs[4].item()) if ema_alive is not None else logs[4].item()
            amd = alive_mean_dist.item()
            if not math.isinf(amd):
                ema_dist = (0.9 * ema_dist + 0.1 * amd) if ema_dist is not None else amd
            elapsed = time.time() - t0
            print(
                f"[{step:6d}] loss={loss.item():.4f}  pg={logs[0]:.4f}  v={logs[1]:.4f}  "
                f"ent={logs[2]:.4f}  r_mean={logs[3]:.4f}  alive={logs[4]*100:.1f}%  "
                f"PLACEMENT: {current_placement}  "
                f"ema_alive={(ema_alive*100.0 if ema_alive is not None else float('nan')):.1f}%  "
                f"ema_dist={(ema_dist if ema_dist is not None else float('inf')):.3f} m  "
                f"time={elapsed:.1f}s",
                flush=True
            )

        # Save
        if (step % args.ckpt_every == 0) or (step == args.train_steps):
            if ddp_rank() == 0:
                os.makedirs(os.path.dirname(args.out), exist_ok=True)
                sd = (net.module.state_dict() if isinstance(net, DDP) else net.state_dict())
                torch.save(sd, args.out)
                maybe_print(f"[save] wrote PSRO-compatible policy to: {args.out}")

    ddp_barrier()
    if ddp_is_initialized():
        dist.destroy_process_group()

# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    # I/O
    ap.add_argument("--out", default="checkpoints/curriculum_stones_policy.pt",
                    help="Output checkpoint (PSRO-compatible)")
    ap.add_argument("--device", default="cuda")

    # Env
    ap.add_argument("--use_preplaced", action="store_true")
    ap.add_argument("--use_powerplay", action="store_true")
    ap.add_argument("--noise_spin_flip_prob", type=float, default=0.0)

    # Execution noise
    ap.add_argument("--exec_noise_sigma", type=float, default=0.0,
                    help="Fixed execution noise (speed/angle).")

    # Prior
    ap.add_argument("--btn_speed", type=float, default=2.82,
                    help="Prior draw-to-button release speed (sim units).")
    ap.add_argument("--btn_angle", type=float, default=0.0,
                    help="Prior draw-to-button release angle (radians).")
    ap.add_argument("--btn_spin", type=int, default=0, choices=[-1,0,1],
                    help="Spin class for the draw prior (-1: out, 0: straight, +1: in).")

    # Model/opt
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--ent_coef", type=float, default=0.01)
    ap.add_argument("--v_coef", type=float, default=0.5)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)

    # Train (consider starting smaller than the original)
    ap.add_argument("--batch_size", type=int, default=32768*6,
                    help="Ends per update (sharded across GPUs if DDP).")
    ap.add_argument("--train_steps", type=int, default=2000)
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--ckpt_every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=1234)

    args = ap.parse_args()
    train(args)

if __name__ == "__main__":
    main()