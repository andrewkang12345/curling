#!/usr/bin/env bash
set -uo pipefail
cd /mnt/data/curling2/csas_world
source scripts/setup_gpu.sh
export PYTHONUNBUFFERED=1
ENVV="WORLD_BOUNDARY_REMOVAL=1 PYTHONPATH=src:/mnt/data/curling2/csas_v3/src JAX_PLATFORMS=cpu \
CUDA_VISIBLE_DEVICES=0 VALUE_EVAL_BATCH=64 POLICY_BATCH_CAP=96 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
GNN_EDGE_SCALAR_MODE=button_visible_plus_curl_arc_reach_with_outgoing \
GNN_NODE_FEATURE_MODE=none GNN_RELEASE_NODE_MODE=three_plus_takeout_boundary GNN_EDGE_PRUNE_MODE=none"
T=artifacts/replay/az_rank2x_train; V=artifacts/replay/az_rank2x_val
for ARM in rank ctrl; do
  W=checkpoints/csas_world/az_v24b_$ARM
  mkdir -p "$W"
  if [ ! -f "$W/best.pt" ]; then
    unset LD_LIBRARY_PATH
    env $ENVV python3 scripts/run_consolidate.py --config "configs/exp_064b_$ARM.yaml" \
      --union "$T" --mcts-val "$V" --init checkpoints/csas_world/az_v14d/best.pt \
      --out "$W" >> "$W/train.log" 2>&1
    echo "[064b] $ARM train rc=$?"
    grep -aoE "rank_acc_mcts=[0-9.]+" "$W/train.log" | tail -3
  fi
done
CKA=checkpoints/csas_world/az_v24b_rank/best.pt; [ -f "$CKA" ] || CKA=checkpoints/csas_world/az_v24b_rank/model.pt
CKB=checkpoints/csas_world/az_v24b_ctrl/best.pt; [ -f "$CKB" ] || CKB=checkpoints/csas_world/az_v24b_ctrl/model.pt
O=eval_out/az_v24b/rank_vs_ctrl_k4
if [ ! -f "$O/summary.json" ]; then
  mkdir -p "$O"; unset LD_LIBRARY_PATH
  env $ENVV python3 scripts/_eval_parallel.py --champion "$CKA" --vs "$CKB" \
    --N 250 --horizons 1,2,3,4,5,6,7,8,9,10 --gpus 0,0,0,0 --shards 4 --noisy --sel-noise 4 \
    --out-dir "$O" >> eval_out/az_v24b/eval.log 2>&1
fi
python3 - <<'PYEOF' | tee -a experiments_log.md
import json, glob, math
W = M = N = 0.0; ms = []
for f in glob.glob("eval_out/az_v24b/rank_vs_ctrl_k4/*__h*__s*.json"):
    d = json.load(open(f))
    for k, v in d.items():
        if k.startswith("h") and isinstance(v, dict):
            W += v["winrate"]*v["n_ends"]; M += v["mean_margin"]*v["n_ends"]; N += v["n_ends"]
            ms.append(v["mean_margin"])
if N:
    kk = len(ms); mu = sum(ms)/kk
    se = math.sqrt(sum((x-mu)**2 for x in ms)/max(kk-1,1)/kk)
    print(f"\n**EXP-064b raw (auto):** rank vs ctrl k=4: winrate {W/N:.4f}, dScore {M/N:+.4f} ± {se:.4f}/end (n={int(N)})")
PYEOF
echo "EXP064B_DONE $(date -u +%FT%TZ)"
