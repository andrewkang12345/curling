#!/usr/bin/env bash
# EXP-064: rank-loss A/B — train az_v23_rank, gate k=4 vs the matched az_v19 control.
set -uo pipefail
cd /mnt/data/curling2/csas_world
WORK=checkpoints/csas_world/az_v23_rank
mkdir -p "$WORK"
source scripts/setup_gpu.sh
export PYTHONUNBUFFERED=1
ENVV="WORLD_BOUNDARY_REMOVAL=1 PYTHONPATH=src:/mnt/data/curling2/csas_v3/src JAX_PLATFORMS=cpu \
CUDA_VISIBLE_DEVICES=0 VALUE_EVAL_BATCH=64 POLICY_BATCH_CAP=96 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
GNN_EDGE_SCALAR_MODE=button_visible_plus_curl_arc_reach_with_outgoing \
GNN_NODE_FEATURE_MODE=none GNN_RELEASE_NODE_MODE=three_plus_takeout_boundary GNN_EDGE_PRUNE_MODE=none"
if [ ! -f "$WORK/best.pt" ]; then
  unset LD_LIBRARY_PATH
  env $ENVV python3 scripts/run_consolidate.py \
    --config configs/exp_064_rank.yaml \
    --union artifacts/replay/az_v19_rank_train --mcts-val artifacts/replay/az_v19_rank_val \
    --init checkpoints/csas_world/az_v14d/best.pt --out "$WORK" >> "$WORK/train.log" 2>&1
  echo "[exp064] train rc=$?"
  grep -aE "early-stop" "$WORK/train.log" | tail -1
  grep -aoE "val_rank_acc_mcts=[0-9.]+" "$WORK/train.log" | tail -3
fi
CK="$WORK/best.pt"; [ -f "$CK" ] || CK="$WORK/model.pt"
O=eval_out/az_v23_rank/vsctrl_k4
if [ ! -f "$O/summary.json" ]; then
  mkdir -p "$O"; unset LD_LIBRARY_PATH
  env $ENVV python3 scripts/_eval_parallel.py --champion "$CK" \
    --vs checkpoints/csas_world/az_v19_newrules/best.pt \
    --N 250 --horizons 1,2,3,4,5,6,7,8,9,10 --gpus 0,0,0,0 --shards 4 --noisy --sel-noise 4 \
    --out-dir "$O" >> "$WORK/eval.log" 2>&1
fi
python3 - <<'PYEOF' | tee -a experiments_log.md
import json, glob, math
W = M = N = 0.0; ms = []
for f in glob.glob("eval_out/az_v23_rank/vsctrl_k4/*__h*__s*.json"):
    d = json.load(open(f))
    for k, v in d.items():
        if k.startswith("h") and isinstance(v, dict):
            W += v["winrate"]*v["n_ends"]; M += v["mean_margin"]*v["n_ends"]; N += v["n_ends"]
            ms.append(v["mean_margin"])
if N:
    kk = len(ms); mu = sum(ms)/kk
    se = math.sqrt(sum((x-mu)**2 for x in ms)/max(kk-1,1)/kk)
    print(f"\n**EXP-064 raw (auto):** az_v23_rank vs az_v19 control k=4: winrate {W/N:.4f}, dScore {M/N:+.4f} ± {se:.4f}/end (n={int(N)})")
PYEOF
echo "EXP064_DONE $(date -u +%FT%TZ)"
