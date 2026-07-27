#!/bin/bash
# Compare the noise-aware anchor vs prior bests + vs the noise-free anchor.
set -uo pipefail
cd /mnt/data/curling2/csas_world
source scripts/setup_gpu.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0
WORLD=checkpoints/csas_world/anchor_noisy/model.pt

echo "[$(date +%H:%M:%S)] === anchor_noisy vs csas_v3 baselines (NLL + head-to-head) ==="
python3 -m world.eval.baselines --world "$WORLD" --config configs/anchor_noisy.yaml \
  --horizons 2,6,10 --h2h-roots 40 --n-candidates 24 \
  --baselines checkpoints/policy/human_prior_fullcov/model.pt,checkpoints/policy/mcts_horizon/h10/model.pt \
  --device cuda:0 --out artifacts/metrics/anchor_noisy_compare.json

echo "[$(date +%H:%M:%S)] === noise effect: anchor_noisy vs anchor_v3 (world-vs-world h2h) ==="
python3 - <<'PY' 2>&1 | grep -avE 'warnings|register_custom|xla_cuda12|discover_pjrt|ffi_registrations|^\s*\^'
import json, torch, world
from world.eval.head_to_head import WorldPlayer, build_h2h_roots, head_to_head
dev=torch.device("cuda:0")
A=WorldPlayer("checkpoints/csas_world/anchor_noisy/model.pt", dev, n_candidates=24, name="anchor_noisy")
B=WorldPlayer("checkpoints/csas_world/anchor_v3/model.pt",    dev, n_candidates=24, name="anchor_v3")
out={}
for h in (4,8):
    roots=build_h2h_roots("/mnt/data/curling2/csas_v3", h, 40, split="val", seed=h)
    out[f"h{h:02d}"]=head_to_head(A,B,roots)
    print(f"noisy vs v3 h{h:02d}: winrate={out[f'h{h:02d}']['winrate']:.3f} (o0={out[f'h{h:02d}']['winrate_order0']:.2f} o1={out[f'h{h:02d}']['winrate_order1']:.2f})")
json.dump(out, open("artifacts/metrics/noisy_vs_v3_h2h.json","w"), indent=2)
PY
echo "[$(date +%H:%M:%S)] FINALIZE_COMPARE_DONE"
