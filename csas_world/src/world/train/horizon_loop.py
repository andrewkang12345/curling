"""Horizon-staged MCTS training with head-to-head convergence.

For each horizon h = 1..H (throws-remaining), repeatedly:
  1. COLLECT  MCTS targets at horizon h using the FIXED current checkpoint's
     policy (exported to csas format) and a frozen csas value model -- run as
     parallel subprocesses, one per GPU (JAX sim on CPU, torch on its GPU).
  2. TRAIN    the joint WorldModel (4-GPU DDP) from mixed replay (this stage's
     MCTS shards + the global human/value/sim buffers), warm-started from the
     previous stage.
  3. COMPARE  head-to-head winrate (both throwing orders) of the new policy vs
     the previous best and vs csas baselines.
A stage is converged when the winrate vs the previous best plateaus around 0.5
for ``converge_patience`` rounds (no further improvement from more search).

Collection (JAX) and training (torch) run in separate processes, so the two
never share a GPU process and the cuDNN/JAX-GPU incompatibility is avoided.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

from ..config import Config
from .trainer import export_csas_policy, launch


# --------------------------------------------------------------------------- #
# collection (parallel subprocesses)
# --------------------------------------------------------------------------- #
def parallel_collect(cfg: Config, horizon: int, policy_ckpt: str, value_ckpt: str,
                     out_dir: str, total_roots: int, kind: str = "mcts",
                     config_path: Optional[str] = None, seed: int = 0,
                     value_world_ckpt: Optional[str] = None) -> int:
    gpus = cfg.train.gpus
    n = max(1, len(gpus))
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    per = max(1, total_roots // n)
    gpu_sim = os.environ.get("CURRICULUM_GPU_SIM") == "1"
    setup = "/mnt/data/curling2/csas_world/scripts/setup_gpu.sh"
    procs = []
    for k in range(n):
        out = os.path.join(out_dir, f"shard{k}.npz")
        if gpu_sim:
            # GPU JAX sim: pin shard k to one physical GPU (torch policy/value + JAX share it).
            # setup_gpu.sh is sourced INSIDE the subprocess so its vendored JAX-GPU libs never
            # enter the parent env (the parent keeps a clean LD_LIBRARY_PATH for torch DDP).
            cfg_arg = f"--config {config_path} " if config_path else ""
            vw_arg = f"--value-world {value_world_ckpt} " if value_world_ckpt else ""
            inner = (f"source {setup}; "
                     f"export CUDA_VISIBLE_DEVICES={gpus[k]} XLA_PYTHON_CLIENT_MEM_FRACTION=0.35; "
                     f"exec python3 -m world.search.collect {cfg_arg}{vw_arg}"
                     f"--horizon {horizon} --max-roots {per} "
                     f"--policy {policy_ckpt} --value {value_ckpt} --out {out} "
                     f"--kind {kind} --num-shards {n} --shard-id {k} --device cuda:0 --seed {seed}")
            procs.append(subprocess.Popen(["bash", "-lc", inner]))
        else:
            env = dict(os.environ)
            env["JAX_PLATFORMS"] = "cpu"
            env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
            env.setdefault("PYTHONPATH", "src")
            env["OMP_NUM_THREADS"] = str(max(1, (os.cpu_count() or 8) // n))
            cmd = [sys.executable, "-m", "world.search.collect",
                   "--horizon", str(horizon), "--max-roots", str(per),
                   "--policy", policy_ckpt, "--value", value_ckpt, "--out", out,
                   "--kind", kind, "--device", f"cuda:{gpus[k]}",
                   "--num-shards", str(n), "--shard-id", str(k), "--seed", str(seed)]
            if config_path:
                cmd += ["--config", config_path]
            if value_world_ckpt:
                cmd += ["--value-world", value_world_ckpt]
            procs.append(subprocess.Popen(cmd, env=env))
    rc = 0
    for p in procs:
        rc |= p.wait()
    if rc != 0:
        print(f"[collect h{horizon}] WARNING: a collector exited non-zero (rc={rc})", flush=True)
    return rc


# --------------------------------------------------------------------------- #
# head-to-head between two WorldModel / csas checkpoints
# --------------------------------------------------------------------------- #
def h2h_eval(cfg: Config, new_ckpt: str, opp_kind: str, opp_a: str, opp_b: Optional[str],
             horizon: int, device: Optional[str] = None, n_roots: Optional[int] = None,
             noisy: Optional[bool] = None, root_shard: Optional[tuple] = None) -> Dict:
    import torch

    from ..eval.head_to_head import CsasPlayer, WorldPlayer, build_h2h_roots, head_to_head
    from ..search.noise import make_noise

    dev = torch.device(device or f"cuda:{cfg.train.gpus[0]}")  # keep h2h off GPU 0 unless asked
    # noisy: robust selection (mean decision value over K noisy executions) + noisy realized throws.
    if noisy is None:
        noisy = bool(getattr(cfg.horizon, "noisy_h2h", False))
    ncfg = cfg.csas_path(cfg.search.noise_config).as_posix() if noisy else None
    sns = int(cfg.search.noise_samples) if noisy else 0
    nzA = make_noise(ncfg, 1000 * horizon + 1) if noisy else None
    nzB = make_noise(ncfg, 1000 * horizon + 2) if noisy else None
    env_nz = make_noise(ncfg, 1000 * horizon + 9) if noisy else None
    A = WorldPlayer(new_ckpt, dev, name="world_new", noise=nzA, sel_noise_samples=sns)
    if opp_kind == "world":
        B = WorldPlayer(opp_a, dev, name="opp_world", noise=nzB, sel_noise_samples=sns)
    else:
        B = CsasPlayer(opp_a, opp_b or cfg.csas_path(cfg.paths.baseline_value_ckpt).as_posix(),
                       dev, name="opp_csas", noise=nzB, sel_noise_samples=sns)
    nr = n_roots or cfg.horizon.h2h_games_per_order
    from ..preplaced import PREPLACED_HORIZON
    if bool(getattr(cfg.horizon, "include_preplaced", False)) and int(horizon) >= PREPLACED_HORIZON:
        from ..preplaced import build_preplaced_h2h_roots
        roots = build_preplaced_h2h_roots(int(horizon), nr, split="val", seed=horizon)
    else:
        roots = build_h2h_roots(cfg.paths.csas_v3_root, horizon, nr, split="val", seed=horizon)
    if root_shard is not None:
        sid, ns = int(root_shard[0]), int(root_shard[1])
        roots = roots[sid::ns]   # disjoint interleaved subset (within-horizon parallel sharding)
    return head_to_head(A, B, roots, env_noise=env_nz, realize_noise=noisy)


# --------------------------------------------------------------------------- #
# the curriculum
# --------------------------------------------------------------------------- #
def run_curriculum(cfg: Config, base_ckpt: Optional[str], work_dir: str,
                   config_path: Optional[str] = None,
                   sim_shard_dir: Optional[str] = None) -> Dict:
    Path(work_dir).mkdir(parents=True, exist_ok=True)
    frozen_value = cfg.csas_path(cfg.paths.prior_value_ckpt).as_posix()
    prior_policy = cfg.csas_path(cfg.paths.prior_policy_ckpt).as_posix()
    summary: Dict[str, Dict] = {}

    prev_ckpt = base_ckpt                       # warm-start source for next stage
    prev_best_ckpt = prev_ckpt                  # best WorldModel so far (None -> prior)
    history: List[Dict] = []

    for h in range(cfg.horizon.start_horizon, cfg.horizon.max_horizon + 1):
        stage_dir = os.path.join(work_dir, f"h{h:02d}")
        Path(stage_dir).mkdir(parents=True, exist_ok=True)
        band = cfg.horizon.converge_band
        plateau = 0
        stage_ckpt = prev_ckpt
        for rnd in range(cfg.horizon.rounds_per_stage):
            round_dir = os.path.join(stage_dir, f"r{rnd}")
            mcts_dir = os.path.join(round_dir, "mcts")

            # 1. export the FIXED stage policy to csas format for collection
            policy_for_collect = prior_policy
            if stage_ckpt is not None:
                policy_for_collect = os.path.join(round_dir, "policy_csas.pt")
                _export_from_ckpt(cfg, stage_ckpt, policy_for_collect)

            # 2. collect MCTS targets at this horizon with the fixed checkpoint.
            #    Closed value loop (EXP-010): once a trained stage checkpoint exists, feed ITS
            #    value head into collection (--value-world) so the improving value guides search.
            vw = stage_ckpt if ((getattr(cfg.search, "value_leaf_bootstrap", False)
                                 or getattr(cfg.search, "reward_leaf_select", False))
                                and stage_ckpt is not None) else None
            parallel_collect(cfg, h, policy_for_collect, frozen_value, mcts_dir,
                             cfg.horizon.roots_per_stage, kind="mcts",
                             config_path=config_path, seed=1000 * h + rnd, value_world_ckpt=vw)

            # 3. train the joint model from mixed replay (warm-start from stage_ckpt)
            res = launch(cfg, mcts_shard_dir=mcts_dir, sim_shard_dir=sim_shard_dir,
                         init_ckpt=stage_ckpt, out_dir=round_dir,
                         results_path=os.path.join(round_dir, "results.json"))
            new_ckpt = os.path.join(round_dir, "model.pt")

            # 4. head-to-head vs previous best (winrate convergence metric)
            if prev_best_ckpt is None:
                wr = h2h_eval(cfg, new_ckpt, "csas", prior_policy, frozen_value, h)
            else:
                wr = h2h_eval(cfg, new_ckpt, "world", prev_best_ckpt, None, h)
            rec = {"horizon": h, "round": rnd, "winrate_vs_prev_best": wr,
                   "val": res.get("metrics", {})}
            history.append(rec)
            print(f"[curriculum h{h:02d} r{rnd}] winrate_vs_prev_best="
                  f"{wr['winrate']:.3f} (o0={wr['winrate_order0']:.2f} o1={wr['winrate_order1']:.2f}) "
                  f"val_policy_nll={res.get('metrics', {}).get('val_policy_nll')}", flush=True)

            stage_ckpt = new_ckpt
            mband = cfg.horizon.converge_margin_band
            stronger = (wr["winrate"] > 0.5 + band) or (mband > 0 and wr["mean_margin"] > mband)
            if stronger:
                prev_best_ckpt = new_ckpt       # genuinely stronger (winrate or Δscore) -> new champion
                plateau = 0
            else:
                plateau += 1
            if plateau >= cfg.horizon.converge_patience:
                print(f"[curriculum h{h:02d}] converged after round {rnd}", flush=True)
                break

        prev_ckpt = stage_ckpt
        summary[f"h{h:02d}"] = {"final_ckpt": stage_ckpt, "best_ckpt": prev_best_ckpt,
                                "history": [r for r in history if r["horizon"] == h]}
        with open(os.path.join(work_dir, "curriculum_summary.json"), "w") as fh:
            json.dump(summary, fh, indent=2)

    return summary


def _export_from_ckpt(cfg: Config, world_ckpt: str, out_path: str) -> None:
    """Load a WorldModel checkpoint on CPU and export its policy to csas format."""
    import torch

    from ..config import model_cfg_from_dict
    from ..model import WorldModel
    from .trainer import load_world_checkpoint

    ck = torch.load(world_ckpt, map_location="cpu", weights_only=False)
    mcfg = model_cfg_from_dict(ck["model_cfg"]) if "model_cfg" in ck else cfg.model
    model = WorldModel(mcfg)
    load_world_checkpoint(model, world_ckpt, map_location="cpu")
    export_csas_policy(model, out_path, cfg)


__all__ = ["parallel_collect", "h2h_eval", "run_curriculum"]
