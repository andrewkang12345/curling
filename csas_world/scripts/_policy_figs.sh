#!/bin/bash
cd /mnt/data/curling2/csas_world
source scripts/setup_gpu.sh
WP=checkpoints/csas_world/anchor_v3/policy_csas.pt
PRIOR=/mnt/data/curling2/csas_v3/checkpoints/policy/human_prior_fullcov/model.pt
echo "[figs] world policy multi-action samples (with hammer)"
python3 -m world.eval.policy_figs --policy "$WP" --label csas_world_anchor \
  --out-root artifacts/figures/policy_world \
  --start-horizon 1 --max-horizon 10 --n-real 4 --n-samples 96 --n-trajectories 12 --device cuda:0
echo "[figs] prior policy multi-action samples (with hammer)"
python3 -m world.eval.policy_figs --policy "$PRIOR" --label human_prior_fullcov \
  --out-root artifacts/figures/policy_prior \
  --start-horizon 1 --max-horizon 10 --n-real 4 --n-samples 96 --n-trajectories 12 --device cuda:0
echo "POLICY_FIGS_DONE"
