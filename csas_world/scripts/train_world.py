#!/usr/bin/env python3
"""Launch joint multi-head training (single- or multi-GPU DDP).

    PYTHONPATH=src python scripts/train_world.py --config configs/base.yaml \
        --mcts-dir artifacts/replay/mcts/anchor --sim-dir artifacts/replay/sim \
        --out checkpoints/csas_world/anchor --epochs 30 --gpus 0,1,2,3

The ``--ablation`` flag selects a head configuration:
    policy_value_only | plus_consistency | plus_decoder | full
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import world  # noqa: E402,F401  (bootstrap)
from world.config import Config, load_config, save_config  # noqa: E402


def apply_ablation(cfg: Config, name: str) -> Config:
    m, l = cfg.model, cfg.loss
    if name == "policy_value_only":
        m.use_dynamics = m.use_outcome = m.use_decoder = m.use_consistency = False
        l.outcome = l.consistency = l.decoder = 0.0
    elif name == "plus_consistency":
        m.use_dynamics = m.use_consistency = True
        m.use_outcome = True
        m.use_decoder = False
        l.decoder = 0.0
    elif name == "plus_decoder":
        m.use_dynamics = m.use_consistency = m.use_outcome = m.use_decoder = True
    elif name == "full":
        m.use_dynamics = m.use_consistency = m.use_outcome = True
        m.use_decoder = True
    elif name and name != "default":
        raise ValueError(f"unknown ablation '{name}'")
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--mcts-dir", default=None)
    ap.add_argument("--sim-dir", default=None)
    ap.add_argument("--init", default=None, help="warm-start from a WorldModel checkpoint")
    ap.add_argument("--out", default="checkpoints/csas_world/run")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--gpus", default=None, help="comma list, e.g. 0,1,2,3")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--samples-per-epoch", type=int, default=None)
    ap.add_argument("--ablation", default="default")
    ap.add_argument("--run-name", default=None)
    args = ap.parse_args()

    cfg: Config = load_config(args.config) if args.config else Config()
    cfg = apply_ablation(cfg, args.ablation)
    if args.gpus:
        cfg.train.gpus = [int(g) for g in args.gpus.split(",") if g != ""]
    if args.epochs is not None:
        cfg.train.epochs = args.epochs
    if args.batch_size is not None:
        cfg.train.batch_size = args.batch_size
    if args.lr is not None:
        cfg.train.lr = args.lr
    if args.samples_per_epoch is not None:
        cfg.train.samples_per_epoch = args.samples_per_epoch
    if args.run_name:
        cfg.train.run_name = args.run_name

    Path(args.out).mkdir(parents=True, exist_ok=True)
    save_config(cfg, str(Path(args.out) / "config.yaml"))

    from world.train.trainer import launch

    res = launch(cfg, mcts_shard_dir=args.mcts_dir, sim_shard_dir=args.sim_dir,
                 init_ckpt=args.init, out_dir=args.out,
                 results_path=str(Path(args.out) / "results.json"))
    print("[train_world] final:", res.get("metrics"))


if __name__ == "__main__":
    main()
