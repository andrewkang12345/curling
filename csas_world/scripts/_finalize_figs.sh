#!/bin/bash
# Regenerate all figures on the noise-aware anchor (anchor_noisy), GPU.
set -uo pipefail
cd /mnt/data/curling2/csas_world
source scripts/setup_gpu.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=1
WORLD=checkpoints/csas_world/anchor_noisy/model.pt
WP=checkpoints/csas_world/anchor_noisy/policy_csas.pt
PRIOR=/mnt/data/curling2/csas_v3/checkpoints/policy/human_prior_fullcov/model.pt

echo "[$(date +%H:%M:%S)] policy_world (noise-aware policy)"
python3 -m world.eval.policy_figs --policy "$WP" --label csas_world_noisy \
  --out-root artifacts/figures/policy_world --start-horizon 1 --max-horizon 10 \
  --n-real 4 --n-samples 96 --n-trajectories 12 --device cuda:0

echo "[$(date +%H:%M:%S)] policy_prior (human prior, reference)"
python3 -m world.eval.policy_figs --policy "$PRIOR" --label human_prior_fullcov \
  --out-root artifacts/figures/policy_prior --start-horizon 1 --max-horizon 10 \
  --n-real 4 --n-samples 96 --n-trajectories 12 --device cuda:0

echo "[$(date +%H:%M:%S)] value + collision + best_decision (kind=all, shared vlim=3.0)"
python3 -m world.eval.figures --world "$WORLD" --kind all \
  --horizons 1 3 5 8 10 --n-real 4 --n-shots 1500 --noise-samples 16 \
  --vlim 3.0 --device cuda:0
echo "[$(date +%H:%M:%S)] FINALIZE_FIGS_DONE"
