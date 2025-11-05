from __future__ import annotations
import argparse, os, datetime
import numpy as np
import torch
import torch.distributed as dist

from torch_curling_batched import CurlingConfig
from gpu_psro.psro_gpu import PSRO, PSROCfg, PPOCfg

def init_dist_and_device():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    ranked = world_size > 1

    if ranked and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, timeout=datetime.timedelta(hours=4))

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device_str = f"cuda:{local_rank}"
    else:
        device_str = "cpu"
    return device_str, local_rank, (dist.get_rank() if dist.is_initialized() else 0), (dist.get_world_size() if dist.is_initialized() else 1)

def is_main():
    return (not dist.is_initialized()) or dist.get_rank() == 0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--placement", type=str, default="A", choices=["A","B","PP_left","PP_right"])
    parser.add_argument("--use_preplaced", action="store_true")
    parser.add_argument("--use_powerplay", action="store_true")
    parser.add_argument("--noise_sigma", type=float, default=0.0)
    parser.add_argument("--max_iterations", type=int, default=20)
    parser.add_argument("--batch_eval", type=int, default=65536*3)
    parser.add_argument("--br_envs", type=int, default=131072)
    parser.add_argument("--br_steps", type=int, default=500_000)
    parser.add_argument("--rollout_horizon", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--mb_size", type=int, default=65536*3)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--shape_coef", type=float, default=0.01, help="Distance-delta shaping strength")
    parser.add_argument("--out_dir", type=str, default="checkpoints")
    args = parser.parse_args()

    device_str, local_rank, rank, world = init_dist_and_device()
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    device = device_str if (args.device.lower().startswith("cuda") and torch.cuda.is_available()) else "cpu"

    make_env_cfg = CurlingConfig(
        stones_per_team=5,
        team0_hammer=True,
        use_preplaced=args.use_preplaced,
        use_powerplay=args.use_powerplay,
    )

    psro = PSRO(
        make_env_cfg=make_env_cfg,
        placement=args.placement,
        psro=PSROCfg(
            device=device,
            batch_eval=args.batch_eval,
            exec_noise_sigma=args.noise_sigma,
            max_iterations=args.max_iterations,
            out_dir=args.out_dir,
        ),
        ppo=PPOCfg(
            obs_dim=68,                    # 16*4 + 4 flags
            device=device,
            total_env_steps=args.br_steps,
            rollout_horizon=args.rollout_horizon,
            epochs=args.epochs,
            minibatch_size=args.mb_size,
            batch_size_envs=args.br_envs,
            lr=args.lr,
            ckpt_dir=args.out_dir,
            ddp=(world > 1),
            world_size=world,
            rank=rank,
            shaping_coef=args.shape_coef,  # <— new
        )
    )

    out = psro.run(br_steps=args.br_steps)

    if is_main():
        print("\n=== PSRO (GPU) Results ===")
        print(f"Placement: {args.placement} | preplaced={args.use_preplaced} | powerplay={args.use_powerplay}")
        print(f"Meta value: {out['meta_value']:.4f}")
        print("Mix P0:")
        for lbl, w in out["mix_p0"]:
            print(f"  {lbl:>12s}: {w:.3f}")
        print("Mix P1:")
        for lbl, w in out["mix_p1"]:
            print(f"  {lbl:>12s}: {w:.3f}")
        print("Payoff matrix:")
        with np.printoptions(precision=3, suppress=True):
            print(out["payoff"])
        print("History:")
        for h in out["history"]:
            print(h)

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

if __name__ == "__main__":
    os.environ.setdefault("NCCL_TIMEOUT", "7200")
    os.environ.setdefault("NCCL_BLOCKING_WAIT", "1")
    main()
