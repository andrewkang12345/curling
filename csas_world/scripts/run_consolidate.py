#!/usr/bin/env python3
"""Standalone DDP consolidation train: one model on the UNION of all per-horizon MCTS buffers.

No collection, no curriculum -- just trainer.launch on a union shard dir, warm-started from a given
checkpoint with a fresh optimizer. See EXP-018.
"""
import argparse
import sys

sys.path.insert(0, "src")
import world  # noqa: F401  (bootstrap: GNN env + csas path)
from world.config import load_config
from world.train.trainer import launch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--union", required=True, help="dir of union MCTS shards (rglob *.npz)")
    ap.add_argument("--init", default=None, help="warm-start checkpoint (fresh optimizer); omit to train FROM SCRATCH (az_v14 capacity scaling)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--mcts-val", default=None,
                    help="held-out MCTS shard dir for per-loss val (overfitting detection).")
    args = ap.parse_args()
    cfg = load_config(args.config)
    print(f"[consolidate] union={args.union} init={args.init} epochs={args.epochs or cfg.train.epochs} "
          f"gpus={cfg.train.gpus} augment={cfg.train.augment} mcts_val={args.mcts_val}", flush=True)
    launch(cfg, mcts_shard_dir=args.union, mcts_val_shard_dir=args.mcts_val,
           init_ckpt=args.init, out_dir=args.out,
           epochs=args.epochs, results_path=f"{args.out}/results.json")
    print("[consolidate] training done -> " + args.out, flush=True)


if __name__ == "__main__":
    main()
