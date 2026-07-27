#!/usr/bin/env bash
# Proper 3-way game-strength eval for the paper. For each of three models we run a noisy
# head-to-head vs the human prior at N=400 per horizon (which caps to the val-pool size for each
# horizon, so this guarantees deterministic full-pool coverage), both throwing orders, all 10
# horizons, 4-GPU within-horizon sharding (4 shards per horizon -> all four GPUs busy).
set -uo pipefail
cd /mnt/data/curling2/csas_world

EVAL_N=400
HORIZONS="1,2,3,4,5,6,7,8,9,10"
GPUS="0,1,2,3"
SHARDS=4

run_one() {
    local CKPT="$1"; local TAG="$2"
    echo "================ $TAG  ($CKPT) ================" | tee -a /tmp/proper_eval.log
    python3 scripts/_eval_parallel.py \
        --champion "$CKPT" --vs prior \
        --N "$EVAL_N" --horizons "$HORIZONS" \
        --gpus "$GPUS" --shards "$SHARDS" --noisy \
        --out-dir "eval_out/proper/${TAG}_vs_prior" 2>&1 | tee -a /tmp/proper_eval.log
}

# clean start
rm -f /tmp/proper_eval.log
rm -rf eval_out/proper
mkdir -p eval_out/proper

# 1) consolidated (the paper's main model)
run_one "checkpoints/csas_world/exp_019_consolidate/last.pt"            "exp019_consolidated"
# 2) per-stage champion (forgetting-free single-stage best, no pre-placed h10 training)
run_one "checkpoints/csas_world/exp_017_deploy_robust/h07/r0/model.pt"  "exp017_perStage_h07r0"
# 3) sequential 1->10 curriculum (the forgetting baseline)
run_one "checkpoints/csas_world/exp_017_deploy_robust/h10/r1/model.pt"  "exp017_sequential_h10r1"

echo "PROPER_3WAY_DONE" | tee -a /tmp/proper_eval.log
