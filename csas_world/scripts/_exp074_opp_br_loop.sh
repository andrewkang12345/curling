#!/usr/bin/env bash
# EXP-074: train an approximate BR to the provisional EXP-070 meta-Nash mixture.
# Usage: scripts/_exp074_opp_br_loop.sh [games_per_shard=4] [target_games=480] [concurrency=12]
set -euo pipefail
cd /mnt/data/curling2/csas_world

G=${1:-4}
TARGET=${2:-480}
CONCURRENCY=${3:-12}
if (( TARGET % G != 0 )); then
  echo "TARGET must be divisible by games_per_shard" >&2
  exit 2
fi
NSHARDS=$((TARGET / G))
if (( NSHARDS != 120 )); then
  echo "EXP-074's exact 312/96/72 allocation is preregistered for 120 shards; got $NSHARDS" >&2
  exit 2
fi

OUT=artifacts/replay/mcts/az_v28_oppbr_meta
WORK=checkpoints/csas_world/az_v28_oppbr_meta
INC=checkpoints/csas_world/az_v25_br/best.pt
POL=checkpoints/csas_world/az_v25_br/policy_csas.pt
VAL=/mnt/data/curling2/csas_v3/checkpoints/value/holdout0/model.pt
CFG=configs/exp_074_opp_vt_targets.yaml
LOG="$OUT/exp074.log"
LOCK="$OUT/launcher.pid"
mkdir -p "$OUT" "$WORK"
if [[ -f "$LOCK" ]]; then
  old_pid=$(<"$LOCK")
  if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "REFUSING: EXP-074 launcher $old_pid is already alive" | tee -a "$LOG"
    exit 1
  fi
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

export PYTHONUNBUFFERED=1
ENVV="WORLD_BOUNDARY_REMOVAL=1 PYTHONPATH=src:/mnt/data/curling2/csas_v3/src JAX_PLATFORMS=cpu \
OMP_NUM_THREADS=2 POLICY_BATCH_CAP=32 VALUE_EVAL_BATCH=48 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
GNN_EDGE_SCALAR_MODE=button_visible_plus_curl_arc_reach_with_outgoing \
GNN_NODE_FEATURE_MODE=none GNN_RELEASE_NODE_MODE=three_plus_takeout_boundary GNN_EDGE_PRUNE_MODE=none"

declare -A OPP=(
  [v26]="checkpoints/csas_world/az_v26_br2/best.pt"
  [v19]="checkpoints/csas_world/az_v19_newrules/best.pt"
  [v14d]="checkpoints/csas_world/az_v14d/best.pt"
)

run_collect() {
  local tag=$1 games=$2 seed=$3 member=$4 gpu=$5 out=$6
  echo "[exp074] launch $tag member=$member games=$games gpu=$gpu" | tee -a "$LOG"
  env -u LD_LIBRARY_PATH $ENVV CUDA_VISIBLE_DEVICES="$gpu" \
    timeout 21600 python3 -m world.search.selfplay \
      --config "$CFG" --games "$games" --num-shards 1 --shard-id 0 --split train \
      --seed "$seed" --scorer opp_vectree --policy "$POL" --value "$VAL" \
      --value-world "$INC" --opponent-world "${OPP[$member]}" \
      --out "$out" --device cuda:0 > "${out%.npz}.log" 2>&1
}

# Full-budget correctness pilot: one complete end against each support member.
# These records are deliberately excluded from training.
if [[ ! -f "$OUT/PILOT_OK" ]]; then
  pilot_pids=()
  pilot_members=(v26 v19 v14d)
  for j in 0 1 2; do
    member=${pilot_members[$j]}
    out="$OUT/pilot_${member}.npz"
    if [[ ! -f "$out" ]]; then
      run_collect "pilot_$member" 1 $((740010 + j)) "$member" $((j + 1)) "$out" &
      pilot_pids+=("$!")
    fi
  done
  for pid in "${pilot_pids[@]}"; do
    if ! wait "$pid"; then
      echo "[exp074] FATAL: full-budget pilot worker $pid failed" | tee -a "$LOG"
      exit 1
    fi
  done
  python3 - <<'PY'
import json, numpy as np
from pathlib import Path
root = Path("artifacts/replay/mcts/az_v28_oppbr_meta")
for name in ("v26", "v19", "v14d"):
    p = root / f"pilot_{name}.npz"
    m = root / f"pilot_{name}.manifest.json"
    d = np.load(p, allow_pickle=True)
    assert len(d["horizon"]) == 10, (name, len(d["horizon"]))
    assert d["horizon"].tolist() == list(range(10, 0, -1)), name
    spec = json.loads(m.read_text())
    assert spec["records"] == 10 and spec["scorer"] == "opp_vectree", spec
print("EXP074_PILOT_COMPLETE")
PY
  touch "$OUT/PILOT_OK"
  echo "[exp074] three-member full-budget pilot passed $(date -u +%FT%TZ)" | tee -a "$LOG"
fi

# A multiplicative permutation spreads the exact totals across the collection:
# 78/24/18 four-game shards = 312/96/72 games.
member_for_shard() {
  local idx=$1 perm=$(( (idx * 37) % 120 ))
  if (( perm < 78 )); then echo v26
  elif (( perm < 102 )); then echo v19
  else echo v14d
  fi
}

echo "[exp074] collection target=$TARGET games ($NSHARDS shards), concurrency=$CONCURRENCY start=$(date -u +%FT%TZ)" | tee -a "$LOG"
for ((base=0; base<NSHARDS; base+=CONCURRENCY)); do
  pids=()
  labels=()
  for ((off=0; off<CONCURRENCY && base+off<NSHARDS; off++)); do
    idx=$((base + off))
    member=$(member_for_shard "$idx")
    shard=$(printf "%04d" "$idx")
    out="$OUT/shard${shard}_${member}.npz"
    [[ -f "$out" ]] && continue
    gpu=$((idx % 3 + 1))
    run_collect "shard$shard" "$G" $((740100 + idx)) "$member" "$gpu" "$out" &
    pids+=("$!")
    labels+=("$shard")
    sleep 2
  done
  for k in "${!pids[@]}"; do
    if ! wait "${pids[$k]}"; then
      echo "[exp074] FATAL: shard ${labels[$k]} failed" | tee -a "$LOG"
      exit 1
    fi
  done
  complete=$(find "$OUT" -maxdepth 1 -name 'shard*.npz' | wc -l)
  echo "[exp074] collection $complete/$NSHARDS shards complete $(date -u +%FT%TZ)" | tee -a "$LOG"
done

python3 - <<'PY'
import json, glob, numpy as np
files = sorted(glob.glob("artifacts/replay/mcts/az_v28_oppbr_meta/shard*.npz"))
assert len(files) == 120, len(files)
games = {"v26": 0, "v19": 0, "v14d": 0}
records = sig = 0
for p in files:
    d = np.load(p, allow_pickle=True)
    assert len(d["horizon"]) == 40, (p, len(d["horizon"]))
    assert sorted(set(d["horizon"].tolist())) == list(range(1, 11)), p
    records += len(d["horizon"]); sig += int((d["dist_mask"] > 0).sum())
    member = next(name for name in games if f"_{name}.npz" in p)
    games[member] += len(d["horizon"]) // 10
assert games == {"v26": 312, "v19": 96, "v14d": 72}, games
print(json.dumps({"games": games, "records": records, "sig_plies": sig,
                  "sig_rate_all_plies": sig / records}, indent=2))
PY

TRAIN=artifacts/replay/az_v28_oppbr_train
VALID=artifacts/replay/az_v28_oppbr_val
mkdir -p "$TRAIN" "$VALID"
for f in "$OUT"/shard*.npz; do
  base=$(basename "$f")
  idx=${base#shard}; idx=${idx%%_*}
  if (( 10#$idx % 5 == 0 )); then
    ln -sfn "$(readlink -f "$f")" "$VALID/$base"
  else
    ln -sfn "$(readlink -f "$f")" "$TRAIN/$base"
  fi
done
echo "[exp074] split $(find "$TRAIN" -type l | wc -l) train / $(find "$VALID" -type l | wc -l) val" | tee -a "$LOG"

if [[ ! -f "$WORK/best.pt" ]]; then
  echo "[exp074] training start $(date -u +%FT%TZ)" | tee -a "$LOG"
  env -u LD_LIBRARY_PATH $ENVV CUDA_VISIBLE_DEVICES=1 python3 scripts/run_consolidate.py \
    --config configs/exp_074_train.yaml --union "$TRAIN" --mcts-val "$VALID" \
    --init "$INC" --out "$WORK" >> "$WORK/train.log" 2>&1
  echo "[exp074] training complete $(date -u +%FT%TZ)" | tee -a "$LOG"
fi
CK="$WORK/best.pt"
[[ -f "$CK" ]] || { echo "[exp074] FATAL: no selected best.pt" | tee -a "$LOG"; exit 1; }

for member in v26 v19 v14d; do
  EVAL="eval_out/az_v28_oppbr_meta/vs_${member}"
  [[ -f "$EVAL/summary.json" ]] && continue
  mkdir -p "$EVAL"
  env -u LD_LIBRARY_PATH $ENVV python3 scripts/_eval_parallel.py \
    --champion "$CK" --vs "${OPP[$member]}" --N 250 \
    --horizons 1,2,3,4,5,6,7,8,9,10 --gpus 1,2,3 --shards 3 \
    --noisy --sel-noise 4 --out-dir "$EVAL" >> "$EVAL/run.log" 2>&1
done

python3 scripts/exp074_analyze.py --root eval_out/az_v28_oppbr_meta \
  | tee -a "$LOG" | tee -a experiments_log.md
echo "EXP074_DONE $(date -u +%FT%TZ)" | tee -a "$LOG"
