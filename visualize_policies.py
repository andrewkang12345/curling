# visualize_policies.py
from __future__ import annotations
import argparse, os
import torch
import numpy as np

# Headless rendering
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib import animation

from torch_curling_batched import TorchBatchedCurlingEnv, CurlingConfig, STONE_R, Sheet
from gpu_psro.policies import ActorCritic, ActorCriticCfg
from gpu_psro.obs_embed import embed_obs
from gpu_psro.io_utils import load_policy_state_dict

# --- Match training action mapping (residuals in latent space) ---
SPEED_SCALE = 1.6   # speed = 1.6 * (tanh(z_s) + 1) ∈ [0, 3.2]
ANGLE_SCALE = 0.8   # angle = 0.8 * tanh(z_a)       ≈ [-0.8, 0.8]

def atanh_clamped(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x = torch.clamp(x, -1.0 + eps, 1.0 - eps)
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))

def speed_to_latent(speed: torch.Tensor) -> torch.Tensor:
    return atanh_clamped(speed / SPEED_SCALE - 1.0)

def angle_to_latent(angle: torch.Tensor) -> torch.Tensor:
    return atanh_clamped(angle / ANGLE_SCALE)

def logits_to_spin(spin_idx: torch.Tensor) -> torch.Tensor:
    # {0,1,2} -> {-1,0,+1}
    return torch.where(spin_idx == 0, -torch.ones_like(spin_idx),
           torch.where(spin_idx == 1, torch.zeros_like(spin_idx),
                       torch.ones_like(spin_idx))).float()

def _infer_obs_and_hidden_from_ckpt(path: str, device: str) -> tuple[int, int]:
    sd = torch.load(path, map_location=device)
    if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]
    for k in ("backbone.0.weight", "module.backbone.0.weight"):
        if k in sd:
            w = sd[k]; return int(w.shape[1]), int(w.shape[0])
    # Fallback: first linear-looking weight
    for k, w in sd.items():
        if k.endswith(".weight") and getattr(w, "ndim", 0) == 2:
            return int(w.shape[1]), int(w.shape[0])
    raise RuntimeError(f"Could not infer obs_dim/hidden from checkpoint: {path}")

def load_actor(path: str, device: str) -> ActorCritic:
    obs_dim, hidden = _infer_obs_and_hidden_from_ckpt(path, device)
    actor = ActorCritic(ActorCriticCfg(obs_dim=obs_dim, hidden=hidden), device=device)
    load_policy_state_dict(actor, path, map_location=device)
    actor.eval()
    return actor

@torch.no_grad()
def act_with_residual(policy: ActorCritic,
                      obs_vec: torch.Tensor,
                      btn_speed: float, btn_angle: float, btn_spin_class: int,
                      deterministic: bool) -> torch.Tensor:
    """
    Reproduce training's residual-in-latent sampling around the fixed prior.
    """
    B = obs_vec.shape[0]
    device = obs_vec.device
    # prior -> latents
    speed_prior = torch.full((B, 1), btn_speed, device=device)
    angle_prior = torch.full((B, 1), btn_angle, device=device)
    spin_val = {-1: -1.0, 0: 0.0, 1: 1.0}[int(np.sign(btn_spin_class)) if btn_spin_class in (-1, 0, 1) else 0]
    z_s_prior = speed_to_latent(speed_prior)
    z_a_prior = angle_to_latent(angle_prior)

    mu_s, std_s, mu_a, std_a, spin_logits, _ = policy(obs_vec)

    if deterministic:
        z_s = z_s_prior + mu_s
        z_a = z_a_prior + mu_a
        spin_idx = spin_logits.argmax(dim=-1)
    else:
        eps_s = torch.randn_like(mu_s)
        eps_a = torch.randn_like(mu_a)
        z_s = z_s_prior + (mu_s + std_s * eps_s)
        z_a = z_a_prior + (mu_a + std_a * eps_a)
        spin_idx = torch.distributions.Categorical(logits=spin_logits).sample()

    spin_code = logits_to_spin(spin_idx).unsqueeze(-1)
    speed = SPEED_SCALE * (torch.tanh(z_s) + 1.0)
    angle = ANGLE_SCALE * torch.tanh(z_a)
    return torch.cat([speed, angle, spin_code], dim=-1)

# ----------------- Visualization helpers -----------------
def draw_house(ax: plt.Axes, sheet: Sheet):
    colors = ["#add8e6", "#ffffff", "#ff4d4d"]  # light blue, white, red
    for r, c in zip(sheet.house_radii[::-1], colors):
        ax.add_patch(Circle((sheet.tee_x, sheet.tee_y), r, color=c, zorder=0, alpha=0.6))
    ax.plot([-sheet.width / 2, sheet.width / 2], [sheet.hog_y, sheet.hog_y], "--", color="gray", lw=1)
    ax.axhline(sheet.backline_y, color="k", lw=1)

def plot_state(ax: plt.Axes, snapshot: dict, sheet: Sheet, title: str):
    ax.clear()
    draw_house(ax, sheet)
    pos = snapshot["pos"]; alive = snapshot["alive"]; team = snapshot["team"]
    for i in range(pos.shape[0]):
        if alive[i] <= 0.0: continue
        x, y = float(pos[i, 0]), float(pos[i, 1])
        color = "#1f77b4" if team[i] < 0.5 else "#ff7f0e"
        ax.add_patch(Circle((x, y), STONE_R, color=color, ec="k", zorder=3))
    ax.set_xlim(-sheet.width / 2 - 0.5, sheet.width / 2 + 0.5)
    ax.set_ylim(sheet.hog_y - 4.0, sheet.backline_y + 3.0)
    ax.set_aspect("equal", "box")
    ax.set_title(title)

def save_final_and_panel(frames, reward, out_prefix: str, sheet: Sheet):
    # Final frame
    fig, ax = plt.subplots(figsize=(6, 8))
    plot_state(ax, frames[-1], sheet, f"Final (score={reward:+.1f})")
    fig.tight_layout(); fig.savefig(f"{out_prefix}_final.png", dpi=150); plt.close(fig)

    # Pre-throw panels (every even frame: 0,2,4,... are BEFORE throws)
    pre_frames = frames[::2]
    K = min(8, len(pre_frames))
    if K > 0:
        cols, rows = 2, (K + 1) // 2
        fig, axes = plt.subplots(rows, cols, figsize=(8, 4 * rows))
        axes = np.array(axes).reshape(-1)
        for i in range(K):
            plot_state(axes[i], pre_frames[i], sheet, f"Before throw {i + 1}")
        for j in range(K, len(axes)): axes[j].axis("off")
        fig.tight_layout(); fig.savefig(f"{out_prefix}_panels.png", dpi=150); plt.close(fig)

def save_animation(frames, out_path: str, sheet: Sheet, fps: int = 3):
    fig, ax = plt.subplots(figsize=(6, 8))
    def init():
        plot_state(ax, frames[0], sheet, "Start"); return []
    def animate(i):
        plot_state(ax, frames[i], sheet, f"Step {i + 1}/{len(frames)}"); return []
    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(frames),
                                   interval=1000 // fps, blit=False)
    base, _ = os.path.splitext(out_path)
    try:
        anim.save(out_path, fps=fps, dpi=150)
    except Exception:
        anim.save(base + ".gif", fps=fps)
    plt.close(fig)

# ----------------- Multi-throw end simulation -----------------
@torch.no_grad()
def simulate_end_multi(policy0: ActorCritic,
                       policy1: ActorCritic,
                       placement: str,
                       cfg: CurlingConfig,
                       device: str,
                       btn_speed: float, btn_angle: float, btn_spin: int,
                       deterministic: bool,
                       exec_noise_sigma: float):
    """
    Alternate teams until no rocks remain. Collect frames before and after each throw.
    Returns (frames, final_score).
    """
    env = TorchBatchedCurlingEnv(B=1, device=device, cfg=cfg)
    env.set_execution_noise(exec_noise_sigma)
    env.reset(placements=[placement])

    frames = []

    def snapshot(tag: str):
        s = env.sim._obs()
        frames.append({
            "pos":   s["pos"][0].detach().cpu().numpy().copy(),
            "alive": s["alive"][0, :, 0].detach().cpu().numpy().copy(),
            "team":  s["team"][0, :, 0].detach().cpu().numpy().copy(),
            "turn":  int(s["aux"][0, 0].item()),
            "tag":   tag,
        })
        return s

    # initial pre-throw snapshot
    s = snapshot("start")

    final_reward = 0.0
    step_idx = 0
    while True:
        # who throws
        cur = s["aux"][:, 0].long().item()  # 0 -> team0, 1 -> team1
        obs_vec = embed_obs(s)

        a = act_with_residual(
            policy0 if cur == 0 else policy1,
            obs_vec,
            btn_speed=btn_speed, btn_angle=btn_angle, btn_spin_class=btn_spin,
            deterministic=deterministic
        )

        # step
        _, rew, done = env.step(a)
        step_idx += 1

        # post-throw snapshot
        s = snapshot(f"after_{step_idx}")

        if done.item():
            final_reward = float(rew.item())
            break

        # next pre-throw snapshot
        s = snapshot(f"before_{step_idx+1}")

    return frames, final_reward

def main():
    ap = argparse.ArgumentParser()
    # Policies
    ap.add_argument("--p0", required=False, help="Team 0 policy .pt")
    ap.add_argument("--p1", required=False, help="Team 1 policy .pt (defaults to --p0 if omitted)")
    ap.add_argument("--p",  required=False, help="Single policy to use for both teams (shortcut)")
    ap.add_argument("--device", default="cuda")
    # Placement / env
    ap.add_argument("--placement", default="EMPTY",
                    help="One of 'EMPTY','A','B','PP_left','PP_right', or 'CURRICULUM_k' (k in 1..11)")
    ap.add_argument("--stones_per_team", type=int, default=4,
                    help="Number of stones per team (total throws = 2*stones_per_team + any preplaced)")
    ap.add_argument("--use_preplaced", action="store_true")
    ap.add_argument("--use_powerplay", action="store_true")
    # Execution noise and sim tuning
    ap.add_argument("--exec_noise_sigma", type=float, default=0.0)
    ap.add_argument("--max_sim_steps", type=int, default=3000)
    ap.add_argument("--v_stop", type=float, default=0.05)
    ap.add_argument("--dt", type=float, default=0.02)
    # Prior used in residual mapping
    ap.add_argument("--btn_speed", type=float, default=2.82)
    ap.add_argument("--btn_angle", type=float, default=0.0)
    ap.add_argument("--btn_spin",  type=int, default=0, choices=[-1, 0, 1])
    # Determinism
    ap.add_argument("--deterministic", action="store_true",
                    help="Use mean action (default False -> stochastic)")
    # Output
    ap.add_argument("--out_prefix", default="viz/end_multi")
    args = ap.parse_args()

    device = "cuda" if (args.device.startswith("cuda") and torch.cuda.is_available()) else "cpu"
    os.makedirs(os.path.dirname(args.out_prefix), exist_ok=True)

    # Resolve checkpoints
    if args.p:
        p0_path = p1_path = args.p
    else:
        if not args.p0:
            raise SystemExit("Provide --p or --p0 (and optionally --p1).")
        p0_path = args.p0
        p1_path = args.p1 or p0_path

    # Load policies
    pol0 = load_actor(p0_path, device)
    pol1 = load_actor(p1_path, device)

    # Env config for a full end
    cfg = CurlingConfig(
        stones_per_team=args.stones_per_team,
        team0_hammer=True,
        use_preplaced=args.use_preplaced,
        use_powerplay=args.use_powerplay,
        noise_speed=args.exec_noise_sigma,
        noise_angle=args.exec_noise_sigma * 0.2,
        max_sim_steps=args.max_sim_steps,
        v_stop=args.v_stop,
        dt=args.dt,
        curriculum_cache_per_k=256,
    )

    frames, score = simulate_end_multi(
        pol0, pol1,
        placement=args.placement,
        cfg=cfg,
        device=device,
        btn_speed=args.btn_speed,
        btn_angle=args.btn_angle,
        btn_spin=args.btn_spin,
        deterministic=bool(args.deterministic),
        exec_noise_sigma=args.exec_noise_sigma,
    )

    sheet = Sheet()
    save_final_and_panel(frames, score, args.out_prefix, sheet)
    save_animation(frames, args.out_prefix + ".mp4", sheet, fps=3)

    print(f"Saved: {args.out_prefix}_final.png, {args.out_prefix}_panels.png, {args.out_prefix}.mp4 (or .gif fallback)")
    print(f"End score (team0 - team1): {score:+.1f}")

if __name__ == "__main__":
    main()