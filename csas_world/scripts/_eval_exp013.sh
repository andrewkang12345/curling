#!/usr/bin/env bash
set -uo pipefail
cd /mnt/data/curling2/csas_world
unset LD_LIBRARY_PATH
export PYTHONPATH=/mnt/data/curling2/csas_world/src:/mnt/data/curling2/csas_v3/src
export JAX_PLATFORMS=cpu
export GNN_EDGE_SCALAR_MODE=button_visible_plus_curl_arc_reach_with_outgoing GNN_NODE_FEATURE_MODE=none GNN_RELEASE_NODE_MODE=three_plus_takeout_boundary GNN_EDGE_PRUNE_MODE=none
E13=checkpoints/csas_world/exp_013_reward_robust/h05/r1/model.pt
E10=checkpoints/csas_world/exp_c_valueloop/h03/r1/model.pt
echo "===== EXP-013 (robust select) vs prior — NOISY, incl FGZ h8 ====="
python3 scripts/_eval_highN.py --champion "$E13" --vs prior --N 70 --horizons 2,3,4,5,8 --noisy \
  --out checkpoints/csas_world/exp_013_reward_robust/noisy_vs_prior.json
echo "===== EXP-013 vs EXP-010 (non-robust closed loop) — NOISY ====="
python3 scripts/_eval_highN.py --champion "$E13" --vs "$E10" --N 70 --horizons 2,3,4,5 --noisy \
  --out checkpoints/csas_world/exp_013_reward_robust/noisy_vs_exp010.json
echo "EXP013_EVALS_DONE"
