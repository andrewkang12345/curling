#!/usr/bin/env python3
"""AlphaZero-style iterative training to convergence.

Each iteration:
  1. COLLECT terminal-MC targets with the CURRENT policy (value-model-free; rolls the
     policy to terminal and scores by curling rules) -- 4 GPUs, one shard each.
  2. TRAIN the joint model (policy_bc=0, value_from_mcts=true) warm-started from the
     previous iteration -- 4-GPU DDP.
  3. EXPORT the trained policy to csas format (drives the next collection).
  4. HEAD-TO-HEAD vs the previous iteration (noisy realized throws, full rollout to
     terminal, rule scoring). Report win rate AND avg score differential.
Stop when the new iteration no longer beats the previous by > band (winrate plateau).

Collection (JAX/GPU) and training (torch) run as separate subprocesses with the
appropriate environments (see the two helper shells written alongside).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import world  # noqa: E402,F401

CSAS = "/mnt/data/curling2/csas_v3"
PRIOR = f"{CSAS}/checkpoints/policy/human_prior_fullcov/best.pt"
VALUE = f"{CSAS}/checkpoints/value/holdout0/model.pt"   # loaded but unused (terminal scoring)
CFG = "configs/anchor_mcts_v3.yaml"   # KR-UCT (n_sims=120) + value on clean realized-ValueDiff buffer
GPU_SETUP = "source scripts/setup_gpu.sh"


def sh(cmd, env=None):
    return subprocess.run(["bash", "-lc", cmd], cwd=ROOT, env=env).returncode


def collect_iter(policy_ckpt, world_ckpt, out_dir, roots, horizons, seed_base):
    """Collect with the current policy via the real multi-ply KR-UCT tree
    (value-model-free): the policy proposes candidates, leaves are MC-rolled to
    terminal + rule-scored, the search-improved policy = value-weighted soft-top-k
    of the searched root actions, and value targets = realized terminal ValueDiff."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    for h in horizons:
        hh = f"h{h:02d}"
        if len(list(Path(out_dir, hh).glob("*.npz"))) >= 4:
            print(f"[az] skip {hh}: 4 shards already present (resume)", flush=True)
            continue
        procs = []
        for k in range(4):
            out = f"{out_dir}/{hh}/shard{k}.npz"
            cmd = (f"{GPU_SETUP}; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; "
                   f"CUDA_VISIBLE_DEVICES={k} XLA_PYTHON_CLIENT_MEM_FRACTION=0.30 "
                   f"python3 -m world.search.collect --config {CFG} --horizon {h} --max-roots {roots} "
                   f"--kind mcts --policy {policy_ckpt} --value {VALUE} "
                   f"--out {out} --num-shards 4 --shard-id {k} --device cuda:0 --seed {seed_base + h}")
            procs.append(subprocess.Popen(["bash", "-lc", cmd], cwd=ROOT))
        for p in procs:
            p.wait()
        n = len(list(Path(out_dir, hh).glob("*.npz")))
        print(f"[az] collected {hh}: {n} shards", flush=True)


def train_iter(mcts_dir, out_dir, init_ckpt):
    init = f"--init {init_ckpt}" if init_ckpt else ""
    # clear LD_LIBRARY_PATH so torch uses its own cuDNN (not the vendored JAX one)
    cmd = (f"unset LD_LIBRARY_PATH; export PYTHONPATH=src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; "
           f"python3 scripts/train_world.py --config {CFG} --ablation default "
           f"--mcts-dir {mcts_dir} --sim-dir artifacts/replay/sim --out {out_dir} "
           f"--run-name {Path(out_dir).name} {init}")
    return sh(cmd)


def export_policy(world_ckpt, out_path):
    import torch
    from world.config import Config, model_cfg_from_dict
    from world.model import WorldModel
    from world.train.trainer import export_csas_policy, load_world_checkpoint
    ck = torch.load(world_ckpt, map_location="cpu", weights_only=False)
    m = WorldModel(model_cfg_from_dict(ck["model_cfg"]))
    load_world_checkpoint(m, world_ckpt, map_location="cpu")
    export_csas_policy(m, out_path, Config())


def h2h(new_ckpt, prev_ckpt, horizons, n_roots):
    """Noisy realized head-to-head (new vs prev). Returns avg winrate + avg score diff."""
    import torch
    from world.eval.head_to_head import WorldPlayer, build_h2h_roots, head_to_head
    from world.search.noise import make_noise
    dev = torch.device("cuda:0")
    ncfg = f"{CSAS}/configs/noise/v1_bowling.json"
    env_noise = make_noise(ncfg, 999)
    A = WorldPlayer(new_ckpt, dev, n_candidates=48, name="new", noise=make_noise(ncfg, 1), sel_noise_samples=8)
    B = WorldPlayer(prev_ckpt, dev, n_candidates=48, name="prev", noise=make_noise(ncfg, 2), sel_noise_samples=8)
    wrs, mgs, per = [], [], {}
    for h in horizons:
        roots = build_h2h_roots(CSAS, h, n_roots, split="val", seed=h)
        r = head_to_head(A, B, roots, env_noise=env_noise, realize_noise=True)
        per[f"h{h:02d}"] = r; wrs.append(r["winrate"]); mgs.append(r["mean_margin"])
        print(f"[az h2h] new vs prev h{h:02d}: wr={r['winrate']:.3f} dScore={r['mean_margin']:+.3f}", flush=True)
    return sum(wrs) / len(wrs), sum(mgs) / len(mgs), per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-iters", type=int, default=4)
    ap.add_argument("--roots", type=int, default=120)
    ap.add_argument("--horizons", default="1,2,3,4,5,6,7,8,9,10")
    ap.add_argument("--h2h-horizons", default="4,8")
    ap.add_argument("--h2h-roots", type=int, default=30)
    ap.add_argument("--band", type=float, default=0.04)
    ap.add_argument("--work", default="checkpoints/csas_world/az")
    args = ap.parse_args()
    horizons = [int(x) for x in args.horizons.split(",")]
    h2h_h = [int(x) for x in args.h2h_horizons.split(",")]
    Path(args.work).mkdir(parents=True, exist_ok=True)

    # M_0 = the current best model (anchor_noisy): export its policy, use its value head.
    m0 = "checkpoints/csas_world/anchor_noisy/model.pt"
    m0_pol = "checkpoints/csas_world/anchor_noisy/policy_csas.pt"
    if not Path(m0_pol).exists():
        export_policy(m0, m0_pol)
    prev_policy, prev_world = m0_pol, m0
    history = []
    for it in range(1, args.max_iters + 1):
        tag = f"iter{it}"
        mcts_dir = f"artifacts/replay/mcts/{Path(args.work).name}_{tag}"
        out_dir = f"{args.work}/{tag}"
        print(f"\n===== AZ {tag}: collect (policy+value from {Path(prev_world).parent.name}) =====", flush=True)
        collect_iter(prev_policy, prev_world, mcts_dir, args.roots, horizons, seed_base=300 + 50 * it)
        print(f"===== AZ {tag}: train =====", flush=True)
        train_iter(mcts_dir, out_dir, prev_world)
        world_ckpt = f"{out_dir}/model.pt"
        pol_ckpt = f"{out_dir}/policy_csas.pt"
        export_policy(world_ckpt, pol_ckpt)
        rec = {"iter": it, "world": world_ckpt}
        if prev_world is not None:
            wr, mg, per = h2h(world_ckpt, prev_world, h2h_h, args.h2h_roots)
            rec.update({"winrate_vs_prev": wr, "dscore_vs_prev": mg, "per_h": per})
            print(f"===== AZ {tag}: winrate vs prev = {wr:.3f} (dScore {mg:+.3f}) =====", flush=True)
        history.append(rec)
        json.dump(history, open(f"{args.work}/az_history.json", "w"), indent=2)
        if prev_world is not None and rec["winrate_vs_prev"] <= 0.5 + args.band:
            print(f"===== AZ converged at {tag} (winrate {rec['winrate_vs_prev']:.3f} <= {0.5+args.band}) =====", flush=True)
            break
        prev_policy, prev_world = pol_ckpt, world_ckpt
    print("AZ_CONVERGE_DONE", flush=True)


if __name__ == "__main__":
    main()
