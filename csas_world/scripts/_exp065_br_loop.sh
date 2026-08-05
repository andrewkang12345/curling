#!/usr/bin/env bash
# EXP-065: exploitability probe — asymmetric best response to az_v14d.
#   A. collect ~600 new-rules self-play games with STT targets (az_v14d generator,
#      champion value head via --value-world for in-rollout search + truncated leaf)
#   B. fine-tune az_v14d with the exp_052 recipe (identical to the az_v19 control arm)
#   C. gate: 3 draws vs az_v19_newrules (the matched train-side contrast), 1 draw vs
#      az_v14d, and 1 k=8 confirmation draw vs az_v19 (EXP-057 policy)
#   usage: _exp065_stt_loop.sh [N_workers=5] [games_per_worker_per_round=16] [target=600]
set -uo pipefail
cd /mnt/data/curling2/csas_world
N=${1:-3}; G=${2:-16}; TARGET=${3:-600}
OUT=artifacts/replay/mcts/az_v25_br
WORK=checkpoints/csas_world/az_v25_br
INC=checkpoints/csas_world/az_v14d/best.pt
CTRL=checkpoints/csas_world/az_v14d/best.pt   # gate opponent = THE INCUMBENT
POL=checkpoints/csas_world/az_v15_L8/incumbent0_policy_csas.pt
VAL=/mnt/data/curling2/csas_v3/checkpoints/value/holdout0/model.pt
LOG="$OUT/exp065.log"
LOCK="$OUT/launcher.pid"
mkdir -p "$OUT" "$WORK"
if [ -f "$LOCK" ]; then p=$(cat "$LOCK"); [ -n "$p" ] && kill -0 "$p" 2>/dev/null && { echo "REFUSING: $p alive" | tee -a "$LOG"; exit 1; }; fi
echo $$ > "$LOCK"; trap 'rm -f "$LOCK"' EXIT
source scripts/setup_gpu.sh
export PYTHONUNBUFFERED=1
ENVV="WORLD_BOUNDARY_REMOVAL=1 PYTHONPATH=src:/mnt/data/curling2/csas_v3/src JAX_PLATFORMS=cpu \
CUDA_VISIBLE_DEVICES=0 POLICY_BATCH_CAP=96 VALUE_EVAL_BATCH=64 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
GNN_EDGE_SCALAR_MODE=button_visible_plus_curl_arc_reach_with_outgoing \
GNN_NODE_FEATURE_MODE=none GNN_RELEASE_NODE_MODE=three_plus_takeout_boundary GNN_EDGE_PRUNE_MODE=none"
echo "[exp065] N=$N G=$G TARGET=$TARGET; start $(date -u +%FT%TZ)" | tee -a "$LOG"

total_games() { ls "$OUT"/r*_shard*.npz 2>/dev/null | wc -l | awk -v g=$G '{print $1*g}'; }

round=1
while true; do
  [ -f "$OUT/STOP" ] && { echo "[exp065] STOP file found" | tee -a "$LOG"; break; }
  done_games=$(total_games)
  [ "$done_games" -ge "$TARGET" ] && { echo "[exp065] target reached: $done_games games" | tee -a "$LOG"; break; }
  R=$(printf "%04d" $round)
  if [ ! -f "$OUT/r${R}_shard0.npz" ]; then
    pids=()
    for k in $(seq 0 $((N-1))); do
      [ "$k" -gt 0 ] && sleep 60
      # GPU-SIM workers (2026-08-05): JAX on cuda via the sourced setup_gpu env
      # (LD_LIBRARY_PATH preserved), small XLA fraction so 2 workers + torch + arena fit.
      env $ENVV JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
        XLA_PYTHON_CLIENT_MEM_FRACTION=0.15 VALUE_EVAL_BATCH=128 \
        timeout 21600 python3 -m world.search.selfplay \
        --config configs/exp_065_br_targets.yaml \
        --games "$G" --num-shards "$N" --shard-id "$k" --split train \
        --seed $((650000 + round*100 + k)) --scorer bigsel \
        --policy "$POL" --value "$VAL" --value-world "$INC" --opponent-world "$INC" \
        --out "$OUT/r${R}_shard$k.npz" --device cuda:0 \
        > "$OUT/r${R}_shard$k.log" 2>&1 &
      pids+=($!)
    done
    wait "${pids[@]}" || true
  fi
  python3 - <<PYEOF 2>/dev/null | tee -a "$LOG"
import numpy as np, glob
tot = act = 0
for f in glob.glob("$OUT/r${R}_shard*.npz"):
    d = np.load(f, allow_pickle=True); m = d["dist_mask"]; tot += len(m); act += int((m>0).sum())
print(f"[exp065] round $R done: +{tot} records ({act} sig, {act/max(tot,1):.1%}); cumulative games: $(total_games) / $TARGET")
PYEOF
  round=$((round+1))
done

T=artifacts/replay/az_v25_train; V=artifacts/replay/az_v25_val
rm -rf "$T" "$V"; mkdir -p "$T" "$V"
for f in "$OUT"/r*_shard*.npz; do
  b=$(basename "$f"); r=${b#r}; r=${r%%_*}; k=${b##*shard}; k=${k%.npz}
  if [ "$k" = "0" ] && [ $((10#$r % 3)) -eq 1 ]; then ln -s "$(readlink -f "$f")" "$V/$b"; else ln -s "$(readlink -f "$f")" "$T/$b"; fi
done
echo "[exp065] split: $(ls $T | wc -l) train / $(ls $V | wc -l) val shards" | tee -a "$LOG"

if [ ! -f "$WORK/best.pt" ]; then
  echo "[exp065] TRAIN start $(date -u +%H:%M)" | tee -a "$LOG"
  unset LD_LIBRARY_PATH
  env $ENVV python3 scripts/run_consolidate.py \
    --config configs/exp_065_train.yaml --union "$T" --mcts-val "$V" \
    --init "$INC" --out "$WORK" >> "$WORK/train.log" 2>&1
  echo "[exp065] train rc=$?" | tee -a "$LOG"
  grep -aE "early-stop" "$WORK/train.log" | tail -1 | tee -a "$LOG"
fi
CK="$WORK/best.pt"; [ -f "$CK" ] || CK="$WORK/model.pt"

# SHORTENED GATE (user, 2026-07-31): one k=4 draw sized to ~2h for an early
# conclusion (k=4 halves selection dilution per EXP-057, so fewer ends suffice);
# the full 3-draw k=2 battery + k=8 confirmation runs only if this looks promising.
run_eval() {  # outdir, opponent, extra flags...
  local O="eval_out/az_v25_br/$1"; local OPP="$2"; shift 2
  [ -f "$O/summary.json" ] && return 0
  mkdir -p "$O"; unset LD_LIBRARY_PATH
  env $ENVV python3 scripts/_eval_parallel.py --champion "$CK" --vs "$OPP" \
    --N 250 --horizons 1,2,3,4,5,6,7,8,9,10 --gpus 0,0,0,0 --shards 4 --noisy "$@" \
    --out-dir "$O" >> "$WORK/eval.log" 2>&1
  echo "[exp065] eval $1 done $(date -u +%H:%M)" | tee -a "$LOG"
}
run_eval vsinc_k4 "$CTRL" --sel-noise 4

python3 - <<'PYEOF' | tee -a "$LOG" | tee -a experiments_log.md
import json, glob, math
print("\n**EXP-063 raw eval aggregates (auto-appended):**\n")
for tag in ("vsinc_k4",):
    W = M = N = 0.0; ms = []
    for f in glob.glob(f"eval_out/az_v25_br/{tag}/*__h*__s*.json"):
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
echo "EXP065_DONE $(date -u +%FT%TZ)" | tee -a "$LOG"
