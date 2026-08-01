#!/usr/bin/env bash
# EXP-059: 2x data-scale test of the StT teacher. Continues the az_v20_stt corpus to
# ~1,250 games (3 workers — the safe GPU ceiling for value-model-resident StT workers),
# retrains from az_v14d, gates at k=4 vs the az_v19 screen control (primary) and vs
# az_v20_stt 1x (scale response).
set -uo pipefail
cd /mnt/data/curling2/csas_world
N=3; G=16; TARGET=1250
OUT=artifacts/replay/mcts/az_v20_stt
WORK=checkpoints/csas_world/az_v21_stt2x
INC=checkpoints/csas_world/az_v14d/best.pt
POL=checkpoints/csas_world/az_v15_L8/incumbent0_policy_csas.pt
VAL=/mnt/data/curling2/csas_v3/checkpoints/value/holdout0/model.pt
LOG="$OUT/exp059.log"
LOCK="$OUT/launcher059.pid"
mkdir -p "$OUT" "$WORK"
if [ -f "$LOCK" ]; then p=$(cat "$LOCK"); [ -n "$p" ] && kill -0 "$p" 2>/dev/null && { echo REFUSING | tee -a "$LOG"; exit 1; }; fi
echo $$ > "$LOCK"; trap 'rm -f "$LOCK"' EXIT
source scripts/setup_gpu.sh
export PYTHONUNBUFFERED=1
ENVV="WORLD_BOUNDARY_REMOVAL=1 PYTHONPATH=src:/mnt/data/curling2/csas_v3/src JAX_PLATFORMS=cpu \
CUDA_VISIBLE_DEVICES=0 POLICY_BATCH_CAP=96 VALUE_EVAL_BATCH=64 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
GNN_EDGE_SCALAR_MODE=button_visible_plus_curl_arc_reach_with_outgoing \
GNN_NODE_FEATURE_MODE=none GNN_RELEASE_NODE_MODE=three_plus_takeout_boundary GNN_EDGE_PRUNE_MODE=none"
echo "[exp059] N=$N G=$G TARGET=$TARGET; start $(date -u +%FT%TZ)" | tee -a "$LOG"

total_games() { ls "$OUT"/r*_shard*.npz 2>/dev/null | wc -l | awk -v g=$G '{print $1*g}'; }

round=1
while true; do
  [ -f "$OUT/STOP" ] && { echo "[exp059] STOP" | tee -a "$LOG"; break; }
  done_games=$(total_games)
  [ "$done_games" -ge "$TARGET" ] && { echo "[exp059] target reached: $done_games games" | tee -a "$LOG"; break; }
  R=$(printf "%04d" $round)
  if [ ! -f "$OUT/r${R}_shard0.npz" ]; then
    pids=()
    for k in $(seq 0 $((N-1))); do
      [ "$k" -gt 0 ] && sleep 90
      unset LD_LIBRARY_PATH
      env $ENVV timeout 21600 python3 -m world.search.selfplay \
        --config configs/exp_058_stt_targets.yaml \
        --games "$G" --num-shards "$N" --shard-id "$k" --split train \
        --seed $((590000 + round*100 + k)) --scorer stt \
        --policy "$POL" --value "$VAL" --value-world "$INC" \
        --out "$OUT/r${R}_shard$k.npz" --device cuda:0 \
        > "$OUT/r${R}_shard$k.log" 2>&1 &
      pids+=($!)
    done
    wait "${pids[@]}" || true
    n_new=$(ls "$OUT/r${R}"_shard*.npz 2>/dev/null | wc -l)
    echo "[exp059] round $R shards on disk: $n_new/$N; cumulative games: $(total_games)/$TARGET" | tee -a "$LOG"
  fi
  round=$((round+1))
done

# split: val = shard2 of rounds 1,4,7,... (round % 3 == 1); train = rest
T=artifacts/replay/az_v21_train; V=artifacts/replay/az_v21_val
rm -rf "$T" "$V"; mkdir -p "$T" "$V"
for f in "$OUT"/r*_shard*.npz; do
  b=$(basename "$f"); r=${b#r}; r=${r%%_*}; k=${b##*shard}; k=${k%.npz}
  if [ "$k" = "2" ] && [ $((10#$r % 3)) -eq 1 ]; then ln -s "$(readlink -f "$f")" "$V/$b"; else ln -s "$(readlink -f "$f")" "$T/$b"; fi
done
echo "[exp059] split: $(ls $T | wc -l) train / $(ls $V | wc -l) val shards" | tee -a "$LOG"
python3 - <<PYEOF | tee -a "$LOG"
import numpy as np, glob
tot = act = 0
for f in glob.glob("$OUT/r*_shard*.npz"):
    d = np.load(f, allow_pickle=True); m = d["dist_mask"]; tot += len(m); act += int((m>0).sum())
print(f"[exp059] corpus: {tot} records, {act} sig plies ({act/max(tot,1):.1%})")
PYEOF

if [ ! -f "$WORK/best.pt" ]; then
  echo "[exp059] TRAIN start $(date -u +%H:%M)" | tee -a "$LOG"
  unset LD_LIBRARY_PATH
  env $ENVV python3 scripts/run_consolidate.py \
    --config configs/exp_059_train.yaml --union "$T" --mcts-val "$V" \
    --init "$INC" --out "$WORK" >> "$WORK/train.log" 2>&1
  echo "[exp059] train rc=$?" | tee -a "$LOG"
  grep -aE "early-stop" "$WORK/train.log" | tail -1 | tee -a "$LOG"
fi
CK="$WORK/best.pt"; [ -f "$CK" ] || CK="$WORK/model.pt"

run_eval() {
  local O="eval_out/az_v21_stt2x/$1"; local OPP="$2"
  [ -f "$O/summary.json" ] && return 0
  mkdir -p "$O"; unset LD_LIBRARY_PATH
  env $ENVV python3 scripts/_eval_parallel.py --champion "$CK" --vs "$OPP" \
    --N 250 --horizons 1,2,3,4,5,6,7,8,9,10 --gpus 0,0,0,0 --shards 4 --noisy --sel-noise 4 \
    --out-dir "$O" >> "$WORK/eval.log" 2>&1
  echo "[exp059] eval $1 done $(date -u +%H:%M)" | tee -a "$LOG"
}
run_eval vs19ctrl_k4 checkpoints/csas_world/az_v19_newrules/best.pt
run_eval vs20stt1x_k4 checkpoints/csas_world/az_v20_stt/best.pt

python3 - <<'PYEOF' | tee -a "$LOG" | tee -a experiments_log.md
import json, glob, math
print("\n**EXP-059 raw eval aggregates (auto-appended):**\n")
for tag in ("vs19ctrl_k4", "vs20stt1x_k4"):
    W = M = N = 0.0; ms = []
    for f in glob.glob(f"eval_out/az_v21_stt2x/{tag}/*__h*__s*.json"):
        d = json.load(open(f))
        for k, v in d.items():
            if k.startswith("h") and isinstance(v, dict):
                W += v["winrate"]*v["n_ends"]; M += v["mean_margin"]*v["n_ends"]; N += v["n_ends"]
                ms.append(v["mean_margin"])
    if N:
        kk = len(ms); mu = sum(ms)/kk
        se = math.sqrt(sum((x-mu)**2 for x in ms)/max(kk-1,1)/kk)
        print(f"- {tag}: winrate {W/N:.4f}, dScore {M/N:+.4f} ± {se:.4f}/end (n={int(N)})")
    else:
        print(f"- {tag}: (no results)")
PYEOF
echo "EXP059_DONE $(date -u +%FT%TZ)" | tee -a "$LOG"
