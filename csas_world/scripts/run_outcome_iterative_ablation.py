#!/usr/bin/env python3
"""Controlled iterative comparison of outcome-head on vs off.

Each arm starts from anchor_noisy and differs only in the categorical terminal
outcome auxiliary. Training is performed in five-epoch blocks. Between blocks,
fresh noisy search targets are collected with the current policy/value model.
Adam state is resumed from the previous block. Model selection is external to
the trainer and uses fixed-seed, noisy full-end play against anchor_noisy.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

ANCHOR = ROOT / "checkpoints/csas_world/anchor_noisy/model.pt"
SIM_REPLAY = ROOT / "artifacts/replay/sim"
NOISE_CFG = "/mnt/data/curling2/csas_v3/configs/noise/v1_bowling.json"
ARMS = (
    ("outcome_on", ROOT / "configs/ablations/anchor_iterative_outcome_on.yaml"),
    ("outcome_off", ROOT / "configs/ablations/anchor_iterative_outcome_off.yaml"),
)


def _run(cmd: List[str], *, env: Dict[str, str] | None = None) -> None:
    print("[run]", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)


def export_policy(world_ckpt: Path, out_path: Path) -> None:
    import torch

    from world.config import Config, model_cfg_from_dict
    from world.model import WorldModel
    from world.train.trainer import export_csas_policy, load_world_checkpoint

    ck = torch.load(world_ckpt, map_location="cpu", weights_only=False)
    model = WorldModel(model_cfg_from_dict(ck["model_cfg"]))
    load_world_checkpoint(model, str(world_ckpt), map_location="cpu")
    export_csas_policy(model, str(out_path), Config())


def _complete_shards(path: Path, horizons: Iterable[int], n_shards: int = 2) -> bool:
    return all((path / f"h{h:02d}" / f"shard{k}.npz").exists()
               for h in horizons for k in range(n_shards))


def collect_targets(config: Path, world_ckpt: Path, out_dir: Path,
                    horizons: List[int], roots_per_horizon: int, seed: int) -> None:
    if _complete_shards(out_dir, horizons):
        print(f"[collect] reuse complete fresh-round replay {out_dir}", flush=True)
        return
    policy_ckpt = out_dir.parent / "policy_csas.pt"
    policy_ckpt.parent.mkdir(parents=True, exist_ok=True)
    export_policy(world_ckpt, policy_ckpt)
    per_shard = math.ceil(roots_per_horizon / 2)
    for h in horizons:
        hdir = out_dir / f"h{h:02d}"
        hdir.mkdir(parents=True, exist_ok=True)
        procs = []
        for shard, gpu in enumerate((0, 1)):
            out = hdir / f"shard{shard}.npz"
            if out.exists():
                continue
            env = dict(os.environ)
            env["PYTHONPATH"] = str(ROOT / "src")
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            env["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.42"
            env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
            cmd = [
                sys.executable, "-m", "world.search.collect",
                "--config", str(config),
                "--horizon", str(h),
                "--max-roots", str(per_shard),
                "--kind", "mcts",
                "--policy", str(policy_ckpt),
                "--value", "unused.pt",
                "--value-world", str(world_ckpt),
                "--out", str(out),
                "--num-shards", "2",
                "--shard-id", str(shard),
                "--device", "cuda:0",
                "--seed", str(seed + 101 * h),
            ]
            shell_cmd = f"source scripts/setup_gpu.sh; exec {shlex.join(cmd)}"
            procs.append(subprocess.Popen(["bash", "-lc", shell_cmd], cwd=ROOT, env=env))
        for proc in procs:
            if proc.wait() != 0:
                raise RuntimeError(f"target collector failed for h={h}")


def train_block(config: Path, init_ckpt: Path, mcts_dir: Path, out_dir: Path,
                run_name: str) -> Path:
    model_pt = out_dir / "model.pt"
    if model_pt.exists():
        print(f"[train] reuse completed block {model_pt}", flush=True)
        return model_pt
    out_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    env.pop("LD_LIBRARY_PATH", None)
    _run([
        sys.executable, "scripts/train_world.py",
        "--config", str(config),
        "--mcts-dir", str(mcts_dir),
        "--sim-dir", str(SIM_REPLAY),
        "--init", str(init_ckpt),
        "--out", str(out_dir),
        "--epochs", "5",
        "--gpus", "0,1",
        "--run-name", run_name,
    ], env=env)
    return model_pt


def h2h_worker(candidate: Path, horizon: int, roots: int, seed: int,
               out_path: Path, gpu: int, opponent: Path = ANCHOR) -> subprocess.Popen:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.42"
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    cmd = [
        sys.executable, str(Path(__file__).resolve()),
        "--h2h-worker",
        "--candidate", str(candidate),
        "--opponent", str(opponent),
        "--horizon", str(horizon),
        "--h2h-roots", str(roots),
        "--seed", str(seed),
        "--worker-output", str(out_path),
    ]
    shell_cmd = f"source scripts/setup_gpu.sh; exec {shlex.join(cmd)}"
    return subprocess.Popen(["bash", "-lc", shell_cmd], cwd=ROOT, env=env)


def evaluate_strength(candidate: Path, out_dir: Path, horizons: List[int],
                      roots: int, seed: int) -> Dict:
    summary_path = out_dir / "h2h_vs_anchor.json"
    if summary_path.exists():
        return json.loads(summary_path.read_text())
    out_dir.mkdir(parents=True, exist_ok=True)
    pending = []
    for i, h in enumerate(horizons):
        out = out_dir / f"h{h:02d}.json"
        if not out.exists():
            pending.append((h, out))
    for i in range(0, len(pending), 2):
        batch = pending[i:i + 2]
        procs = [
            h2h_worker(candidate, h, roots, seed + h, out, gpu=j)
            for j, (h, out) in enumerate(batch)
        ]
        for proc in procs:
            if proc.wait() != 0:
                raise RuntimeError("head-to-head worker failed")
    per = {f"h{h:02d}": json.loads((out_dir / f"h{h:02d}.json").read_text())
           for h in horizons}
    wr = sum(v["winrate"] for v in per.values()) / len(per)
    margin = sum(v["mean_margin"] for v in per.values()) / len(per)
    result = {"winrate": wr, "mean_margin": margin, "per_horizon": per}
    summary_path.write_text(json.dumps(result, indent=2))
    return result


def _is_better(result: Dict, best: Dict, tolerance: float = 0.01) -> bool:
    wr, bwr = result["winrate"], best["winrate"]
    if wr > bwr + tolerance:
        return True
    return abs(wr - bwr) <= tolerance and result["mean_margin"] > best["mean_margin"]


def run_arm(name: str, config: Path, work: Path, shared_round0: Path,
            horizons: List[int], h2h_horizons: List[int], roots_per_horizon: int,
            h2h_roots: int, max_rounds: int, patience: int, seed: int) -> Dict:
    arm_dir = work / name
    arm_dir.mkdir(parents=True, exist_ok=True)
    history_path = arm_dir / "history.json"
    history = json.loads(history_path.read_text()) if history_path.exists() else []
    current = Path(history[-1]["checkpoint"]) if history else ANCHOR
    best_result = {"winrate": 0.5, "mean_margin": 0.0}
    best_ckpt = ANCHOR
    stale = 0
    for rec in history:
        if rec.get("selected"):
            best_result = rec["h2h"]
            best_ckpt = Path(rec["checkpoint"])
            stale = 0
        else:
            stale += 1

    for rnd in range(len(history), max_rounds):
        round_dir = arm_dir / f"round{rnd:02d}"
        mcts_dir = shared_round0 if rnd == 0 else round_dir / "mcts"
        collect_targets(config, current, mcts_dir, horizons, roots_per_horizon,
                        seed + 10_000 * rnd)
        candidate = train_block(config, current, mcts_dir, round_dir / "train",
                                f"{name}_round{rnd:02d}")
        result = evaluate_strength(candidate, round_dir / "game_eval", h2h_horizons,
                                   h2h_roots, seed)
        selected = _is_better(result, best_result)
        if selected:
            best_result = result
            best_ckpt = candidate
            stale = 0
            shutil.copy2(candidate, arm_dir / "best_game.pt")
        else:
            stale += 1
        rec = {
            "round": rnd,
            "checkpoint": str(candidate),
            "mcts_dir": str(mcts_dir),
            "h2h": result,
            "selected": selected,
            "best_checkpoint": str(best_ckpt),
        }
        history.append(rec)
        history_path.write_text(json.dumps(history, indent=2))
        current = candidate
        print(f"[{name} r{rnd}] wr={result['winrate']:.3f} "
              f"dScore={result['mean_margin']:+.3f} selected={selected}", flush=True)
        if stale >= patience:
            print(f"[{name}] converged: no game-strength improvement for {stale} blocks", flush=True)
            break
    summary = {
        "arm": name,
        "best_checkpoint": str(best_ckpt),
        "best_h2h": best_result,
        "rounds": history,
    }
    (arm_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def run_h2h_worker(args) -> None:
    import numpy as np
    import torch

    from world.eval.head_to_head import WorldPlayer, build_h2h_roots, head_to_head
    from world.search.noise import make_noise

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda:0")
    # Use the same frozen anchor value evaluator for both players. This isolates
    # policy strength rather than rewarding either arm for value-head drift.
    candidate = WorldPlayer(
        args.candidate, device, n_candidates=48, name="candidate",
        noise=make_noise(NOISE_CFG, args.seed + 1), sel_noise_samples=8,
        value_ckpt=str(ANCHOR),
    )
    opponent_path = args.opponent or str(ANCHOR)
    opponent = WorldPlayer(
        opponent_path, device, n_candidates=48, name="opponent",
        noise=make_noise(NOISE_CFG, args.seed + 2), sel_noise_samples=8,
        value_ckpt=str(ANCHOR),
    )
    env_noise = make_noise(NOISE_CFG, args.seed + 3)
    roots = build_h2h_roots(
        "/mnt/data/curling2/csas_v3", args.horizon, args.h2h_roots,
        split="val", seed=args.seed,
    )
    result = head_to_head(candidate, opponent, roots, env_noise=env_noise, realize_noise=True)
    Path(args.worker_output).write_text(json.dumps(result, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="checkpoints/csas_world/ablations/outcome_controlled")
    ap.add_argument("--max-rounds", type=int, default=4)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--roots-per-horizon", type=int, default=400)
    ap.add_argument("--horizons", default="1,2,3,4,5,6,7,8,9,10")
    ap.add_argument("--h2h-horizons", default="2,5,8,10")
    ap.add_argument("--h2h-roots", type=int, default=24)
    ap.add_argument("--seed", type=int, default=260612)
    ap.add_argument("--h2h-worker", action="store_true")
    ap.add_argument("--candidate")
    ap.add_argument("--opponent")
    ap.add_argument("--horizon", type=int)
    ap.add_argument("--worker-output")
    args = ap.parse_args()
    if args.h2h_worker:
        run_h2h_worker(args)
        return

    work = ROOT / args.work
    work.mkdir(parents=True, exist_ok=True)
    horizons = [int(x) for x in args.horizons.split(",")]
    h2h_horizons = [int(x) for x in args.h2h_horizons.split(",")]
    shared_round0 = work / "shared_round00_mcts"
    summaries = {}
    for name, config in ARMS:
        summaries[name] = run_arm(
            name, config, work, shared_round0, horizons, h2h_horizons,
            args.roots_per_horizon, args.h2h_roots, args.max_rounds,
            args.patience, args.seed,
        )

    on_ckpt = Path(summaries["outcome_on"]["best_checkpoint"])
    off_ckpt = Path(summaries["outcome_off"]["best_checkpoint"])
    direct = {}
    if on_ckpt != ANCHOR or off_ckpt != ANCHOR:
        # Each arm's anchor comparison remains the model-selection result. Direct
        # comparison is recorded separately for interpretation.
        direct["outcome_on"] = summaries["outcome_on"]["best_h2h"]
        direct["outcome_off"] = summaries["outcome_off"]["best_h2h"]
    final = {"arms": summaries, "comparison": direct}
    (work / "comparison.json").write_text(json.dumps(final, indent=2))
    print("OUTCOME_CONTROLLED_ABLATION_DONE", flush=True)


if __name__ == "__main__":
    main()
