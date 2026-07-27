#!/usr/bin/env bash
# Full-sheet EfficientZero curriculum (run_curriculum.py / horizon_loop) on GPUs 0,1,2,3.
#   collection = GPU JAX sim, one shard pinned per GPU (CURRICULUM_GPU_SIM=1 makes
#                parallel_collect source setup_gpu.sh inside each subprocess, so the
#                vendored JAX-GPU libs never enter this parent env)
#   training   = torch DDP across all 4 GPUs (mp.spawn; inherits this clean env)
#   head2head  = torch + CPU JAX sim in-parent, on cuda:0 (gpus[0])
# This parent keeps a CLEAN LD_LIBRARY_PATH (torch cuDNN) and JAX_PLATFORMS=cpu (the
# in-parent h2h sim); only the collection subprocesses flip to the GPU-JAX env.
# Args: $1=config yaml  $2=work dir  $3=sim-dir (empty dir => sim source skipped)
#       $4+ passed through to run_curriculum.py (e.g. --base <ckpt> --start <h> to resume)
set -uo pipefail
cd /mnt/data/curling2/csas_world
unset LD_LIBRARY_PATH                                    # torch's own cuDNN (parent + training)
export CURRICULUM_GPU_SIM=1                              # collection runs on GPU (vendored JAX)
export PYTHONPATH=/mnt/data/curling2/csas_world/src:/mnt/data/curling2/csas_v3/src
export JAX_PLATFORMS=cpu                                 # parent-side h2h sim only
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export GNN_EDGE_SCALAR_MODE=button_visible_plus_curl_arc_reach_with_outgoing
export GNN_NODE_FEATURE_MODE=none
export GNN_RELEASE_NODE_MODE=three_plus_takeout_boundary
export GNN_EDGE_PRUNE_MODE=none
CONFIG="${1:?usage: _run_curriculum_fullsheet.sh <config.yaml> <work_dir> <sim_dir>}"
WORK="${2:?work dir}"
SIMDIR="${3:?sim dir}"
mkdir -p "$SIMDIR" "$WORK"
# --base omitted => run_curriculum starts from cfg.paths.prior_policy_ckpt (full-sheet prior)
python3 scripts/run_curriculum.py --config "$CONFIG" --work "$WORK" --sim-dir "$SIMDIR" --gpus 0,1,2,3 "${@:4}"
echo "CURRICULUM_EXIT=$?"
