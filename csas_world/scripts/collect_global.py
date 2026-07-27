#!/usr/bin/env python3
"""Collect replay shards (sim-transition + base MCTS) with a FIXED checkpoint.

Runs parallel collectors (one per GPU; JAX sim on CPU, torch on its GPU) across a
set of horizons.  Writes ``.npz`` shards consumed by training.

    JAX_PLATFORMS=cpu PYTHONPATH=src python scripts/collect_global.py \
        --kind mcts --policy <csas_policy.pt> --value <csas_value.pt> \
        --out artifacts/replay/mcts/anchor --horizons 1,2,3,4,5,6,7,8,9,10 \
        --roots-per-horizon 300 --gpus 0,1,2,3
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import world  # noqa: E402,F401
from world.config import Config, load_config  # noqa: E402
from world.train.horizon_loop import parallel_collect  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--kind", choices=["mcts", "sim"], default="mcts")
    ap.add_argument("--policy", required=True)
    ap.add_argument("--value", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--horizons", default="1,2,3,4,5,6,7,8,9,10")
    ap.add_argument("--roots-per-horizon", type=int, default=300)
    ap.add_argument("--gpus", default=None)
    args = ap.parse_args()

    cfg: Config = load_config(args.config) if args.config else Config()
    if args.gpus:
        cfg.train.gpus = [int(g) for g in args.gpus.split(",") if g != ""]
    horizons = [int(h) for h in args.horizons.split(",") if h]

    for h in horizons:
        out_dir = str(Path(args.out) / f"h{h:02d}")
        t = time.time()
        print(f"=== collect {args.kind} horizon {h} -> {out_dir} ===", flush=True)
        parallel_collect(cfg, h, args.policy, args.value, out_dir,
                         args.roots_per_horizon, kind=args.kind,
                         config_path=args.config, seed=h)
        print(f"=== horizon {h} done in {time.time()-t:.0f}s ===", flush=True)


if __name__ == "__main__":
    main()
