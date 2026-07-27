#!/usr/bin/env python3
"""Run the horizon-staged MCTS curriculum with head-to-head convergence.

    JAX_PLATFORMS=cpu PYTHONPATH=src python scripts/run_curriculum.py \
        --config configs/base.yaml --base checkpoints/csas_world/anchor/model.pt \
        --work checkpoints/csas_world/curriculum --sim-dir artifacts/replay/sim \
        --start 1 --max 10 --rounds 2 --roots 200 --gpus 0,1,2,3

Each stage: collect MCTS targets (fixed checkpoint) -> train (4-GPU) -> head-to-head
vs previous best. Convergence is decided by winrate plateau.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import world  # noqa: E402,F401
from world.config import Config, load_config  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--base", default=None, help="WorldModel anchor checkpoint to start from")
    ap.add_argument("--work", default="checkpoints/csas_world/curriculum")
    ap.add_argument("--sim-dir", default="artifacts/replay/sim")
    ap.add_argument("--start", type=int, default=None)
    ap.add_argument("--max", type=int, default=None)
    ap.add_argument("--rounds", type=int, default=None)
    ap.add_argument("--roots", type=int, default=None)
    ap.add_argument("--gpus", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()

    cfg: Config = load_config(args.config) if args.config else Config()
    if args.gpus:
        cfg.train.gpus = [int(g) for g in args.gpus.split(",") if g != ""]
    if args.start is not None:
        cfg.horizon.start_horizon = args.start
    if args.max is not None:
        cfg.horizon.max_horizon = args.max
    if args.rounds is not None:
        cfg.horizon.rounds_per_stage = args.rounds
    if args.roots is not None:
        cfg.horizon.roots_per_stage = args.roots
    if args.epochs is not None:
        cfg.train.epochs = args.epochs

    from world.train.horizon_loop import run_curriculum

    summary = run_curriculum(cfg, base_ckpt=args.base, work_dir=args.work,
                             config_path=args.config, sim_shard_dir=args.sim_dir)
    print("[curriculum] summary stages:", list(summary.keys()))


if __name__ == "__main__":
    main()
