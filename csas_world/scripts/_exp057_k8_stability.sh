#!/usr/bin/env bash
# EXP-057: eval-protocol validation — does the k=2 -> k=8 robust-selection change
# reorder certified rankings? One k=8 draw per decided matchup, each replicated at
# its own certified rules:
#   M1: az_v14d vs az_v9 champion, OLD rules  (certified k=2: ds +0.102 ± 0.005)
#   M2: az_v19_newrules vs az_v14d, NEW rules (certified k=2: ds +0.001 ± 0.012)
# Waits for the EXP-056 shard to drain first. Appends aggregates to the log.
set -uo pipefail
cd /mnt/data/curling2/csas_world
LOG=eval_out/exp057_k8/chain.log
mkdir -p eval_out/exp057_k8
while pgrep -f "exp056_rollout_estimator" > /dev/null; do sleep 300; done
echo "[exp057] start $(date -u +%FT%TZ)" | tee -a "$LOG"
ENVBASE="PYTHONPATH=src:/mnt/data/curling2/csas_v3/src JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=0 \
VALUE_EVAL_BATCH=64 POLICY_BATCH_CAP=96 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
GNN_EDGE_SCALAR_MODE=button_visible_plus_curl_arc_reach_with_outgoing \
GNN_NODE_FEATURE_MODE=none GNN_RELEASE_NODE_MODE=three_plus_takeout_boundary GNN_EDGE_PRUNE_MODE=none"

run_draw() {  # name, A, B, rules_flag, outdir
  local NAME="$1" A="$2" B="$3" RULES="$4" O="eval_out/exp057_k8/$5"
  [ -f "$O/done" ] && return 0
  mkdir -p "$O"; unset LD_LIBRARY_PATH
  env $ENVBASE WORLD_BOUNDARY_REMOVAL=$RULES python3 scripts/_eval_parallel.py \
    --champion "$A" --vs "$B" --N 400 --horizons 1,2,3,4,5,6,7,8,9,10 \
    --gpus 0,0,0,0 --shards 4 --noisy --sel-noise 8 --out-dir "$O" \
    >> "$O/run.log" 2>&1
  touch "$O/done"
  echo "[exp057] $NAME done $(date -u +%FT%TZ)" | tee -a "$LOG"
}

run_draw "M1 v14d-vs-v9champ OLDrules k8" \
  checkpoints/csas_world/az_v14d/best.pt checkpoints/csas_world/az_v9_selfplay/iter2/best.pt 0 m1_v14d_v9_old
run_draw "M2 v19-vs-v14d NEWrules k8" \
  checkpoints/csas_world/az_v19_newrules/best.pt checkpoints/csas_world/az_v14d/best.pt 1 m2_v19_v14d_new

python3 - <<'PYEOF' | tee -a "$LOG" | tee -a experiments_log.md
import json, glob, math
print("\n**EXP-057 k=8 aggregates (auto-appended):**\n")
for tag, cert in (("m1_v14d_v9_old", "+0.102 ± 0.005 (k=2, 3 draws)"),
                  ("m2_v19_v14d_new", "+0.001 ± 0.012 (k=2, 3 draws)")):
    W = M = N = 0.0; ms = []
    for f in glob.glob(f"eval_out/exp057_k8/{tag}/*__h*__s*.json"):
        d = json.load(open(f))
        for k, v in d.items():
            if k.startswith("h") and isinstance(v, dict):
                W += v["winrate"]*v["n_ends"]; M += v["mean_margin"]*v["n_ends"]; N += v["n_ends"]
                ms.append(v["mean_margin"])
    if N:
        kk = len(ms); mu = sum(ms)/kk
        se = math.sqrt(sum((x-mu)**2 for x in ms)/max(kk-1,1)/kk)
        print(f"- {tag}: k=8 winrate {W/N:.4f}, dScore {M/N:+.4f} ± {se:.4f}/end (n={int(N)})  [certified k=2: {cert}]")
    else:
        print(f"- {tag}: (no results)")
PYEOF
echo "EXP057_DONE $(date -u +%FT%TZ)" | tee -a "$LOG"
