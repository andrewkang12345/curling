#!/usr/bin/env python3
"""Option 2: retrain on the EXISTING az_iter1 KR-UCT data but with
``value_from_mcts=false`` so the value head trains on the clean realized-ValueDiff
buffer (not the noisy terminal-MC search returns). Train -> export -> head-to-head
vs anchor_noisy. Reuses the collected MCTS data (no re-collection).

Run under the GPU-JAX env (head-to-head uses the GPU simulator); the train
subprocess clears LD_LIBRARY_PATH so torch uses its own cuDNN.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import world  # noqa: E402,F401

CSAS = "/mnt/data/curling2/csas_v3"
CFG = "configs/anchor_mcts_v2.yaml"
MCTS = "artifacts/replay/mcts/az_iter1"          # reuse EXP-002 collection
OUT = "checkpoints/csas_world/az/iter1_valclean"
ANCHOR = "checkpoints/csas_world/anchor_noisy/model.pt"


def main():
    # 1) train (torch DDP; clear the vendored-JAX cuDNN from LD_LIBRARY_PATH)
    cmd = ("unset LD_LIBRARY_PATH; export PYTHONPATH=src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; "
           f"python3 scripts/train_world.py --config {CFG} --ablation default "
           f"--mcts-dir {MCTS} --sim-dir artifacts/replay/sim --out {OUT} "
           f"--run-name iter1_valclean --init {ANCHOR}")
    rc = subprocess.run(["bash", "-lc", cmd], cwd=ROOT).returncode
    if rc != 0:
        print(f"[valclean] TRAIN FAILED rc={rc}", flush=True)
        sys.exit(rc)

    # 2) export the trained policy
    import torch
    from world.config import Config, model_cfg_from_dict
    from world.model import WorldModel
    from world.train.trainer import export_csas_policy, load_world_checkpoint
    wck, pck = f"{OUT}/model.pt", f"{OUT}/policy_csas.pt"
    ck = torch.load(wck, map_location="cpu", weights_only=False)
    m = WorldModel(model_cfg_from_dict(ck["model_cfg"]))
    load_world_checkpoint(m, wck, map_location="cpu")
    export_csas_policy(m, pck, Config())
    print(f"[valclean] train val metrics: {ck.get('metrics')}", flush=True)

    # 3) head-to-head vs anchor_noisy (same protocol as EXP-002)
    from world.eval.head_to_head import WorldPlayer, build_h2h_roots, head_to_head
    from world.search.noise import make_noise
    dev = torch.device("cuda:0")
    ncfg = f"{CSAS}/configs/noise/v1_bowling.json"
    env_noise = make_noise(ncfg, 999)
    A = WorldPlayer(wck, dev, n_candidates=48, name="valclean", noise=make_noise(ncfg, 1), sel_noise_samples=8)
    B = WorldPlayer(ANCHOR, dev, n_candidates=48, name="anchor", noise=make_noise(ncfg, 2), sel_noise_samples=8)
    res, wrs, mgs = {}, [], []
    for h in (4, 8):
        roots = build_h2h_roots(CSAS, h, 30, split="val", seed=h)
        r = head_to_head(A, B, roots, env_noise=env_noise, realize_noise=True)
        res[f"h{h:02d}"] = {"winrate": float(r["winrate"]), "mean_margin": float(r["mean_margin"])}
        wrs.append(r["winrate"]); mgs.append(r["mean_margin"])
        print(f"[valclean h2h] valclean vs anchor_noisy h{h:02d}: "
              f"wr={r['winrate']:.3f} dScore={r['mean_margin']:+.3f}", flush=True)
    avg = {"winrate": sum(wrs) / len(wrs), "mean_margin": sum(mgs) / len(mgs)}
    res["avg"] = avg
    print(f"[valclean h2h] AVG vs anchor_noisy: wr={avg['winrate']:.3f} dScore={avg['mean_margin']:+.3f}", flush=True)
    json.dump(res, open(f"{OUT}/h2h_vs_anchor.json", "w"), indent=2)
    print("VALCLEAN_DONE", flush=True)


if __name__ == "__main__":
    main()
