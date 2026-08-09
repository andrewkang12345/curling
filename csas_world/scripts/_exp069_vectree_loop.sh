#!/usr/bin/env bash
# EXP-069: champion training on VECTORISED 4-PLY TREE targets (16k sims/decision).
#   collect self-play with scorer=vectree -> fine-tune az_v25_br -> k=4 gate vs champion.
set -uo pipefail
cd /mnt/data/curling2/csas_world
N=${1:-16}; G=${2:-4}; TARGET=${3:-320}
OUT=artifacts/replay/mcts/az_v27_vectree
WORK=checkpoints/csas_world/az_v27_vectree
INC=checkpoints/csas_world/az_v25_br/best.pt
POL=checkpoints/csas_world/az_v25_br/policy_csas.pt
VAL=/mnt/data/curling2/csas_v3/checkpoints/value/holdout0/model.pt
LOG="$OUT/exp069.log"
LOCK="$OUT/launcher.pid"
mkdir -p "$OUT" "$WORK"
if [ -f "$LOCK" ]; then p=$(cat "$LOCK"); [ -n "$p" ] && kill -0 "$p" 2>/dev/null && { echo REFUSING | tee -a "$LOG"; exit 1; }; fi
echo $$ > "$LOCK"; trap 'rm -f "$LOCK"' EXIT
export PYTHONUNBUFFERED=1
ENVV="WORLD_BOUNDARY_REMOVAL=1 PYTHONPATH=src:/mnt/data/curling2/csas_v3/src JAX_PLATFORMS=cpu \
OMP_NUM_THREADS=2 VALUE_EVAL_BATCH=128 POLICY_BATCH_CAP=256 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
GNN_EDGE_SCALAR_MODE=button_visible_plus_curl_arc_reach_with_outgoing \
GNN_NODE_FEATURE_MODE=none GNN_RELEASE_NODE_MODE=three_plus_takeout_boundary GNN_EDGE_PRUNE_MODE=none"
echo "[exp069] N=$N G=$G TARGET=$TARGET start $(date -u +%FT%TZ)" | tee -a "$LOG"
total() { ls "$OUT"/r*_shard*.npz 2>/dev/null | wc -l | awk -v g=$G '{print $1*g}'; }

round=1
while true; do
  [ -f "$OUT/STOP" ] && { echo "[exp069] STOP" | tee -a "$LOG"; break; }
  [ "$(total)" -ge "$TARGET" ] && { echo "[exp069] target reached: $(total)" | tee -a "$LOG"; break; }
  R=$(printf "%04d" $round)
  if [ ! -f "$OUT/r${R}_shard0.npz" ]; then
    pids=()
    for k in $(seq 0 $((N-1))); do
      env -u LD_LIBRARY_PATH $ENVV CUDA_VISIBLE_DEVICES=$((k % 4)) timeout 21600 \
        python3 -m world.search.selfplay --config configs/exp_069_vectree_targets.yaml \
        --games "$G" --num-shards "$N" --shard-id "$k" --split train \
        --seed $((690000 + round*100 + k)) --scorer vectree \
        --policy "$POL" --value "$VAL" --value-world "$INC" \
        --out "$OUT/r${R}_shard$k.npz" --device cuda:0 \
        > "$OUT/r${R}_shard$k.log" 2>&1 &
      pids+=($!); sleep 4
    done
    wait "${pids[@]}" || true
  fi
  python3 - <<PYEOF 2>/dev/null | tee -a "$LOG"
import numpy as np, glob
tot = act = 0
for f in glob.glob("$OUT/r${R}_shard*.npz"):
    d = np.load(f, allow_pickle=True); m = d["dist_mask"]; tot += len(m); act += int((m>0).sum())
print(f"[exp069] round $R: +{tot} records ({act} sig, {act/max(tot,1):.1%}); games $(total)/$TARGET")
PYEOF
  round=$((round+1))
done

T=artifacts/replay/az_v27_train; V=artifacts/replay/az_v27_val
rm -rf "$T" "$V"; mkdir -p "$T" "$V"
for f in "$OUT"/r*_shard*.npz; do
  b=$(basename "$f"); r=${b#r}; r=${r%%_*}; k=${b##*shard}; k=${k%.npz}
  if [ "$k" = "0" ] && [ $((10#$r % 3)) -eq 1 ]; then ln -s "$(readlink -f "$f")" "$V/$b"; else ln -s "$(readlink -f "$f")" "$T/$b"; fi
done
echo "[exp069] split: $(ls $T|wc -l) train / $(ls $V|wc -l) val" | tee -a "$LOG"

if [ ! -f "$WORK/best.pt" ]; then
  env -u LD_LIBRARY_PATH $ENVV CUDA_VISIBLE_DEVICES=0 python3 scripts/run_consolidate.py \
    --config configs/exp_069_train.yaml --union "$T" --mcts-val "$V" \
    --init "$INC" --out "$WORK" >> "$WORK/train.log" 2>&1
  echo "[exp069] train rc=$?" | tee -a "$LOG"
  grep -aE "early-stop" "$WORK/train.log" | tail -1 | tee -a "$LOG"
fi
CK="$WORK/best.pt"; [ -f "$CK" ] || CK="$WORK/model.pt"
O=eval_out/az_v27_vectree/vsinc_k4; mkdir -p "$O"
env -u LD_LIBRARY_PATH $ENVV python3 scripts/_eval_parallel.py --champion "$CK" --vs "$INC" \
  --N 250 --horizons 1,2,3,4,5,6,7,8,9,10 --gpus 0,1,2,3 --shards 4 --noisy --sel-noise 4 \
  --out-dir "$O" >> "$O/run.log" 2>&1
python3 - <<'PYEOF' | tee -a "$LOG" | tee -a experiments_log.md
import json, glob, math
W=M=N=0.0; ms=[]
for f in glob.glob("eval_out/az_v27_vectree/vsinc_k4/*__h*__s*.json"):
    d=json.load(open(f))
    for k,v in d.items():
        if k.startswith("h") and isinstance(v,dict):
            W+=v["winrate"]*v["n_ends"]; M+=v["mean_margin"]*v["n_ends"]; N+=v["n_ends"]; ms.append(v["mean_margin"])
if N:
    kk=len(ms); mu=sum(ms)/kk; se=math.sqrt(sum((x-mu)**2 for x in ms)/max(kk-1,1)/kk)
    print(f"\n**EXP-069 raw (auto):** az_v27_vectree vs az_v25_br k=4: winrate {W/N:.4f}, dScore {M/N:+.4f} ± {se:.4f}/end (n={int(N)})")
PYEOF
echo "EXP069_DONE $(date -u +%FT%TZ)" | tee -a "$LOG"
