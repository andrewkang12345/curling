#!/usr/bin/env bash
# Noisy (robust-select + realized-noise) high-N head-to-heads to (1) decide state- vs
# action-conditioned reward and (2) re-confirm the winners vs the prior UNDER execution noise.
set -uo pipefail
cd /mnt/data/curling2/csas_world
unset LD_LIBRARY_PATH
export PYTHONPATH=/mnt/data/curling2/csas_world/src:/mnt/data/curling2/csas_v3/src
export JAX_PLATFORMS=cpu
export GNN_EDGE_SCALAR_MODE=button_visible_plus_curl_arc_reach_with_outgoing GNN_NODE_FEATURE_MODE=none GNN_RELEASE_NODE_MODE=three_plus_takeout_boundary GNN_EDGE_PRUNE_MODE=none
E10=checkpoints/csas_world/exp_c_valueloop/h03/r1/model.pt        # EXP-010 state-conditioned champion
E12=checkpoints/csas_world/exp_d_reward_sa/h04/r1/model.pt        # EXP-012 action-conditioned champion
N=100; H=2,3,4,5
echo "===== (1) EXP-012 (action-cond) vs EXP-010 (state-cond) — decides reward head ====="
python3 scripts/_eval_highN.py --champion "$E12" --vs "$E10" --N $N --horizons $H --noisy \
  --out checkpoints/csas_world/exp_d_reward_sa/noisy_vs_exp010.json
echo "===== (2) EXP-012 vs prior (noisy) ====="
python3 scripts/_eval_highN.py --champion "$E12" --vs prior --N $N --horizons $H --noisy \
  --out checkpoints/csas_world/exp_d_reward_sa/noisy_vs_prior.json
echo "===== (3) EXP-010 vs prior (noisy, reference) ====="
python3 scripts/_eval_highN.py --champion "$E10" --vs prior --N $N --horizons $H --noisy \
  --out checkpoints/csas_world/exp_c_valueloop/noisy_vs_prior.json
echo "ALL_NOISY_EVALS_DONE"
