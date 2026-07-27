#!/usr/bin/env bash
# az_v17 BIG-CORPUS collection (the data-wall experiment, g5.4xlarge edition).
# Collector: global champion az_v14d (policy export from the L8 leg). Operator: the
# certified sig-gated screen_tree (exp_037). Runs rounds of N parallel workers x G games
# until TARGET_GAMES reached or a STOP file appears. Fully resumable (skips existing
# shards); every shard carries a manifest + certification diag.
#   usage: _collect_bigcorpus.sh <N_workers> <games_per_worker_per_round> <target_games>
#   stop:  touch artifacts/replay/mcts/az_v17_bigcorpus/STOP
set -uo pipefail
cd /mnt/data/curling2/csas_world
N=${1:-8}; G=${2:-16}; TARGET=${3:-5000}
OUT=artifacts/replay/mcts/az_v17_bigcorpus
mkdir -p "$OUT"
LOG="$OUT/collect.log"
LOCK="$OUT/launcher.pid"
if [ -f "$LOCK" ]; then p=$(cat "$LOCK"); [ -n "$p" ] && kill -0 "$p" 2>/dev/null && { echo "REFUSING: $p alive" | tee -a "$LOG"; exit 1; }; fi
echo $$ > "$LOCK"; trap 'rm -f "$LOCK"' EXIT
source scripts/setup_gpu.sh
export PYTHONUNBUFFERED=1
POL=checkpoints/csas_world/az_v15_L8/incumbent0_policy_csas.pt
VAL=/mnt/data/curling2/csas_v3/checkpoints/value/holdout0/model.pt
echo "[bigcorpus] N=$N G=$G TARGET=$TARGET games; start $(date -u +%FT%TZ)" | tee -a "$LOG"

total_games() { ls "$OUT"/r*_shard*.npz 2>/dev/null | wc -l | awk -v g=$G '{print $1*g}'; }

round=1
while true; do
  [ -f "$OUT/STOP" ] && { echo "[bigcorpus] STOP file found" | tee -a "$LOG"; break; }
  done_games=$(total_games)
  [ "$done_games" -ge "$TARGET" ] && { echo "[bigcorpus] target reached: $done_games games" | tee -a "$LOG"; break; }
  R=$(printf "%04d" $round)
  if [ ! -f "$OUT/r${R}_shard0.npz" ]; then
    pids=()
    for k in $(seq 0 $((N-1))); do
      [ "$k" -gt 0 ] && sleep 90   # stagger: serialize the JIT-compilation window (LLVM OOM fix)
      unset LD_LIBRARY_PATH
      PYTHONPATH=src:/mnt/data/curling2/csas_v3/src JAX_PLATFORMS=cpu \
      CUDA_VISIBLE_DEVICES=0 POLICY_BATCH_CAP=96 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      GNN_EDGE_SCALAR_MODE=button_visible_plus_curl_arc_reach_with_outgoing \
      GNN_NODE_FEATURE_MODE=none GNN_RELEASE_NODE_MODE=three_plus_takeout_boundary GNN_EDGE_PRUNE_MODE=none \
      timeout 21600 python3 -m world.search.selfplay --config configs/exp_037_sig_screen_tree.yaml \
        --games "$G" --num-shards "$N" --shard-id "$k" --split train \
        --seed $((10000 + round*100 + k)) --scorer screen_tree \
        --policy "$POL" --value "$VAL" \
        --out "$OUT/r${R}_shard$k.npz" --device cuda:0 \
        > "$OUT/r${R}_shard$k.log" 2>&1 &
      pids+=($!)
    done
    wait "${pids[@]}" || true
  fi
  done_games=$(total_games)
  # aggregate sig stats for the round
  python3 - <<PYEOF 2>/dev/null | tee -a "$LOG"
import numpy as np, glob
tot = act = 0
for f in glob.glob("$OUT/r${R}_shard*.npz"):
    d = np.load(f, allow_pickle=True); m = d["dist_mask"]; tot += len(m); act += int((m>0).sum())
print(f"[bigcorpus] round $R done: +{tot} records ({act} sig, {act/max(tot,1):.1%}); cumulative games: $done_games / $TARGET")
PYEOF
  round=$((round+1))
done
echo "BIGCORPUS_DONE $(date -u +%FT%TZ)" | tee -a "$LOG"
