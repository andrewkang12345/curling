#!/usr/bin/env python3
"""AlphaZero-style iterative training to convergence — v2 (the three changes).

Differences vs the original `az_converge.py`:

  1. **Match EXP-021's data shape per iter** — 1,200 train + 400 val records.
     Each horizon collects 4 shards of 40 records (`--max-roots 160 --num-shards 4`);
     shards 0,1,2 go to the train dir (1,200 total) and shard 3 to the held-out
     val dir (400 total). 10 horizons × 160 records.
  2. **Train with the EXP-021 recipe** (`configs/exp_021_valuemcts_earlystop.yaml`,
     value_from_mcts=true, early-stop by val_value_mse_mcts on the held-out partition)
     via `scripts/run_consolidate.py` (DDP across 4 GPUs).
  3. **Convergence criterion = no improvement on the proper 10-horizon × N=400 eval
     vs the human prior**, not the original 60-game 2-horizon parent-H2H. Each iter
     runs `scripts/_eval_parallel.py --vs prior --N 400 --noisy`, we read
     MEAN-of-pairs winrate + MEAN-of-pairs dScore from the per-shard JSONs, and
     declare converged when both have failed to improve over the previous iter by
     more than the established eval-noise floor (winrate band 0.03, dScore band 0.10).

Iter 0 = `checkpoints/csas_world/exp_021_valuemcts_earlystop/best.pt` (re-evaluated
deterministically so we compare like-with-like). Subsequent iters use the policy
from the previous iter to collect and the previous world model to warm-start.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import world  # noqa: E402,F401

CSAS = "/mnt/data/curling2/csas_v3"
PRIOR_POLICY = f"{CSAS}/checkpoints/policy/human_prior_fullcov/best.pt"
VALUE_CV = f"{CSAS}/checkpoints/value/holdout0/model.pt"
COLLECT_CFG_DEFAULT = "configs/exp_017_deploy_robust.yaml"   # baseline recipe (1-ply EZ, noise_samples=8)
TRAIN_CFG_DEFAULT = "configs/exp_021_valuemcts_earlystop.yaml"
GPU_SETUP = "source scripts/setup_gpu.sh"

GNN_ENV = {
    "GNN_EDGE_SCALAR_MODE": "button_visible_plus_curl_arc_reach_with_outgoing",
    "GNN_NODE_FEATURE_MODE": "none",
    "GNN_RELEASE_NODE_MODE": "three_plus_takeout_boundary",
    "GNN_EDGE_PRUNE_MODE": "none",
}


def sh(cmd, env=None):
    return subprocess.run(["bash", "-lc", cmd], cwd=ROOT, env=env).returncode


# ---------- COLLECT ----------
def collect_iter(policy_ckpt, out_dir, horizons, max_roots, seed_base, collect_cfg):
    """Collect 4 shards × `max_roots/4` records per horizon, one shard per GPU.
    Writes to `<out_dir>/h{NN}/shard{0..3}.npz`. Idempotent: skips a horizon if
    all 4 shards exist already (so a crashed run can resume)."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    for h in horizons:
        hh = f"h{h:02d}"
        hdir = Path(out_dir, hh)
        hdir.mkdir(parents=True, exist_ok=True)
        if len(list(hdir.glob("shard*.npz"))) >= 4:
            print(f"[az v2] skip {hh}: 4 shards already present (resume)", flush=True)
            continue
        t0 = time.time()
        procs = []
        for k in range(4):
            out = str(hdir / f"shard{k}.npz")
            cmd = (
                f"{GPU_SETUP}; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; "
                f"CUDA_VISIBLE_DEVICES={k} XLA_PYTHON_CLIENT_MEM_FRACTION=0.30 "
                f"python3 -m world.search.collect --config {collect_cfg} --horizon {h} "
                f"--max-roots {max_roots} --num-shards 4 --shard-id {k} "
                f"--kind mcts --split train --seed {seed_base + h} "
                f"--policy {policy_ckpt} --value {VALUE_CV} "
                f"--out {out} --device cuda:0"
            )
            procs.append(subprocess.Popen(["bash", "-lc", cmd], cwd=ROOT))
        for p in procs:
            p.wait()
        n_files = len(list(hdir.glob("shard*.npz")))
        print(f"[az v2] collected {hh}: {n_files} shards in {time.time()-t0:.0f}s", flush=True)


def split_train_val(collect_dir, train_dir, val_dir):
    """Per horizon, symlink shards 0,1,2 -> train, shard 3 -> val. 75/25 split."""
    for d in (train_dir, val_dir):
        Path(d).mkdir(parents=True, exist_ok=True)
        for f in Path(d).glob("*.npz"):
            f.unlink()
    for hdir in sorted(Path(collect_dir).glob("h*")):
        hh = hdir.name
        for k in range(3):
            src = (hdir / f"shard{k}.npz").resolve()
            if src.exists():
                (Path(train_dir) / f"{hh}_shard{k}.npz").symlink_to(src)
        src3 = (hdir / "shard3.npz").resolve()
        if src3.exists():
            (Path(val_dir) / f"{hh}_shard3.npz").symlink_to(src3)
    n_t = len(list(Path(train_dir).glob("*.npz")))
    n_v = len(list(Path(val_dir).glob("*.npz")))
    print(f"[az v2] split: train={n_t} files, val={n_v} files", flush=True)


# ---------- TRAIN ----------
def train_iter(train_dir, val_dir, init_ckpt, out_dir, train_cfg):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    log = f"{out_dir}/train.log"
    env_pre = (
        "unset LD_LIBRARY_PATH; "
        f"export PYTHONPATH=src:{CSAS}/src JAX_PLATFORMS=cpu PYTHONUNBUFFERED=1; "
        "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; "
        + "; ".join(f"export {k}={v}" for k, v in GNN_ENV.items())
    )
    cmd = (
        f"{env_pre}; "
        f"python3 scripts/run_consolidate.py "
        f"  --config {train_cfg} --union {train_dir} --mcts-val {val_dir} "
        f"  --init {init_ckpt} --out {out_dir} "
        f">> {log} 2>&1"
    )
    rc = sh(cmd)
    print(f"[az v2] train rc={rc} -> {out_dir}", flush=True)
    return rc


def export_policy(world_ckpt, out_path):
    """Convert a world-model ckpt to csas-format policy (used for the next iter's collect)."""
    import torch
    from world.config import Config, model_cfg_from_dict
    from world.model import WorldModel
    from world.train.trainer import export_csas_policy, load_world_checkpoint
    ck = torch.load(world_ckpt, map_location="cpu", weights_only=False)
    mcfg = model_cfg_from_dict(ck["model_cfg"])
    m = WorldModel(mcfg)
    load_world_checkpoint(m, world_ckpt, map_location="cpu")
    cfg = Config()
    cfg.model = mcfg   # export must carry the MODEL's arch (e.g. L8 depth-extended), not defaults
    export_csas_policy(m, out_path, cfg)


# ---------- EVAL (proper 10-horizon × N=400 vs human prior) ----------
def eval_iter(world_ckpt, eval_out, eval_n=400):
    Path(eval_out).mkdir(parents=True, exist_ok=True)
    log = f"{eval_out}/eval.log"
    env_pre = (
        "unset LD_LIBRARY_PATH; "
        f"export PYTHONPATH=src:{CSAS}/src JAX_PLATFORMS=cpu PYTHONUNBUFFERED=1; "
        + "; ".join(f"export {k}={v}" for k, v in GNN_ENV.items())
    )
    cmd = (
        f"{env_pre}; "
        f"python3 scripts/_eval_parallel.py --champion {world_ckpt} --vs prior "
        f"  --N {eval_n} --horizons 1,2,3,4,5,6,7,8,9,10 --gpus 0,1,2,3 --shards 4 --noisy "
        f"  --out-dir {eval_out} "
        f">> {log} 2>&1"
    )
    rc = sh(cmd)
    print(f"[az v2] eval rc={rc} -> {eval_out}", flush=True)
    return rc


def mean_of_pairs(eval_out, label="prior"):
    """Re-aggregate the per-shard JSONs into mean-of-pairs winrate + dScore (model-as-to-move,
    odd/even pair averages — the hammer-neutral skill number from _aggregate_proper_eval.py).
    Returns dict(mean_wr=..., mean_ds=..., per_pair=[...])."""
    from collections import defaultdict
    by_h = defaultdict(lambda: dict(n=0.0, w0=0.0, w1=0.0, m0=0.0, m1=0.0))
    for f in sorted(Path(eval_out).glob(f"{label}__h*__s*of*.json")):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        for k, v in d.items():
            if not k.startswith("h"):
                continue
            h = int(k[1:])
            no = v["n_ends"] / 2.0
            by_h[h]["n"] += no
            by_h[h]["w0"] += v["winrate_order0"] * no
            by_h[h]["w1"] += v["winrate_order1"] * no
            by_h[h]["m0"] += v.get("mean_margin_order0", 0.0) * no
            by_h[h]["m1"] += v.get("mean_margin_order1", 0.0) * no
    per_h = {}
    for h, r in by_h.items():
        if r["n"] > 0:
            per_h[h] = dict(wr_o0=r["w0"] / r["n"], wr_o1=r["w1"] / r["n"],
                            m_o0=r["m0"] / r["n"], m_o1=r["m1"] / r["n"], n=int(r["n"]))
    pair_wr, pair_ds = [], []
    for a, b in ((1, 2), (3, 4), (5, 6), (7, 8), (9, 10)):
        if a in per_h and b in per_h:
            pair_wr.append(0.5 * (per_h[a]["wr_o0"] + per_h[b]["wr_o0"]))
            pair_ds.append(0.5 * (per_h[a]["m_o0"] + per_h[b]["m_o0"]))
    if not pair_wr:
        return None
    return dict(
        mean_wr=sum(pair_wr) / len(pair_wr),
        mean_ds=sum(pair_ds) / len(pair_ds),
        per_pair_wr=pair_wr, per_pair_ds=pair_ds,
        per_h=per_h,
    )


# ---------- CONVERGENCE TEST ----------
def converged(prev, curr, wr_band, ds_band):
    """True iff NEITHER winrate nor dScore improved over the previous iter by more than its
    noise band — i.e., no real game-strength gain on either metric."""
    if prev is None or curr is None:
        return False
    d_wr = curr["mean_wr"] - prev["mean_wr"]
    d_ds = curr["mean_ds"] - prev["mean_ds"]
    return d_wr <= wr_band and d_ds <= ds_band


# ---------- MAIN LOOP ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init-world", default="checkpoints/csas_world/exp_021_valuemcts_earlystop/best.pt",
                    help="iter-0 world-model checkpoint (used as collector + warm-start for iter 1)")
    ap.add_argument("--max-iters", type=int, default=5)
    ap.add_argument("--max-roots", type=int, default=160,
                    help="root pool per horizon per call (sharded 4 ways: each shard collects 40 records)")
    ap.add_argument("--horizons", default="1,2,3,4,5,6,7,8,9,10")
    ap.add_argument("--eval-n", type=int, default=400)
    ap.add_argument("--wr-band", type=float, default=0.03, help="winrate noise floor")
    ap.add_argument("--ds-band", type=float, default=0.10, help="dScore noise floor")
    ap.add_argument("--work", default="checkpoints/csas_world/az_v4")
    ap.add_argument("--train-config", default=TRAIN_CFG_DEFAULT,
                    help="training config (e.g. exp_021 with value_from_mcts=true, or exp_022 with value_from_mcts=false)")
    ap.add_argument("--collect-config", default=COLLECT_CFG_DEFAULT,
                    help="collection config (e.g. exp_017_deploy_robust = 1-ply EZ baseline, or exp_026_2ply_valuemcts = 2-ply tree)")
    args = ap.parse_args()
    horizons = [int(x) for x in args.horizons.split(",")]
    Path(args.work).mkdir(parents=True, exist_ok=True)
    work_base = Path(args.work).name   # e.g. "az_v4" or "az_v5_novaluemcts" — drives all data/eval paths

    # ---- iter 0 baseline: ALWAYS re-eval freshly in our own dir ----
    # (We used to symlink the cached proper-3way shards here; but that dir got contaminated by
    # stray eval workers from a prior racing-launch incident, so its numbers drifted from the
    # values originally cached. The fix: don't share state with the cached dir — always do a
    # fresh eval into a dir we own, so iter-0 and iter-N use the same methodology AND the same
    # cluster state with no contention.)
    init_world = args.init_world
    init_pol = f"{args.work}/iter0_policy_csas.pt"
    if not Path(init_pol).exists():
        export_policy(init_world, init_pol)
    iter0_eval = f"eval_out/{work_base}/iter0_vs_prior"
    summary_p = Path(iter0_eval) / "summary.json"
    # Only skip if the dir is a REAL dir (not a symlink to an external dir we don't own) with
    # a clean summary.json. A symlink is treated as missing, forcing a fresh eval.
    if Path(iter0_eval).is_symlink() or not summary_p.exists():
        if Path(iter0_eval).is_symlink():
            Path(iter0_eval).unlink()
        print(f"\n===== AZ v2 iter 0: re-eval baseline ({init_world}) =====", flush=True)
        eval_iter(init_world, iter0_eval, args.eval_n)
    base = mean_of_pairs(iter0_eval)
    print(f"[az v2 iter0] MEAN-of-pairs wr={base['mean_wr']:.3f}  dScore={base['mean_ds']:+.3f}", flush=True)

    history = [dict(iter=0, world=init_world, **base)]
    json.dump(history, open(f"{args.work}/history.json", "w"), indent=2)

    prev_policy = init_pol
    prev_world = init_world
    prev_metrics = base

    for it in range(1, args.max_iters + 1):
        tag = f"iter{it}"
        collect_dir = f"artifacts/replay/mcts/{work_base}_{tag}"
        train_dir = f"artifacts/replay/{work_base}_{tag}_train"
        val_dir = f"artifacts/replay/{work_base}_{tag}_val"
        out_dir = f"{args.work}/{tag}"
        eval_dir = f"eval_out/{work_base}/{tag}_vs_prior"

        print(f"\n===== AZ v2 {tag}: collect (policy from {Path(prev_world).parent.name}, cfg={Path(args.collect_config).name}) =====", flush=True)
        collect_iter(prev_policy, collect_dir, horizons, args.max_roots, seed_base=400 + 50 * it,
                     collect_cfg=args.collect_config)
        split_train_val(collect_dir, train_dir, val_dir)

        print(f"===== AZ v2 {tag}: train (warm-start {Path(prev_world).parent.name}, cfg={Path(args.train_config).name}) =====", flush=True)
        train_iter(train_dir, val_dir, prev_world, out_dir, args.train_config)
        world_ckpt = f"{out_dir}/best.pt"
        if not Path(world_ckpt).exists():
            world_ckpt = f"{out_dir}/model.pt"
        pol_ckpt = f"{out_dir}/policy_csas.pt"
        export_policy(world_ckpt, pol_ckpt)

        print(f"===== AZ v2 {tag}: eval vs human prior (N={args.eval_n}, 10 horizons) =====", flush=True)
        eval_iter(world_ckpt, eval_dir, args.eval_n)
        curr = mean_of_pairs(eval_dir)
        print(f"[az v2 {tag}] MEAN-of-pairs wr={curr['mean_wr']:.3f}  dScore={curr['mean_ds']:+.3f}  "
              f"(Δwr={curr['mean_wr']-prev_metrics['mean_wr']:+.3f}, "
              f"ΔdS={curr['mean_ds']-prev_metrics['mean_ds']:+.3f})", flush=True)

        history.append(dict(
            iter=it, world=world_ckpt,
            mean_wr=curr["mean_wr"], mean_ds=curr["mean_ds"],
            d_wr=curr["mean_wr"] - prev_metrics["mean_wr"],
            d_ds=curr["mean_ds"] - prev_metrics["mean_ds"],
            per_pair_wr=curr["per_pair_wr"], per_pair_ds=curr["per_pair_ds"],
        ))
        json.dump(history, open(f"{args.work}/history.json", "w"), indent=2)

        if converged(prev_metrics, curr, args.wr_band, args.ds_band):
            print(f"===== AZ v2 converged at {tag}: "
                  f"Δwr={curr['mean_wr']-prev_metrics['mean_wr']:+.3f} ≤ {args.wr_band}, "
                  f"ΔdS={curr['mean_ds']-prev_metrics['mean_ds']:+.3f} ≤ {args.ds_band} =====", flush=True)
            break

        prev_policy, prev_world, prev_metrics = pol_ckpt, world_ckpt, curr

    print("AZ_V2_CONVERGE_DONE", flush=True)


if __name__ == "__main__":
    main()
