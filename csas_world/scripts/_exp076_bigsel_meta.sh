#!/usr/bin/env bash
# EXP-076: exact successful bigsel+matchup-return BR recipe against mu_070.
set -euo pipefail
cd /mnt/data/curling2/csas_world

OUT=artifacts/replay/mcts/az_v29_bigsel_meta
WORK=checkpoints/csas_world/az_v29_bigsel_meta
TRAIN=artifacts/replay/az_v29_bigsel_meta_train
VALID=artifacts/replay/az_v29_bigsel_meta_val
EVAL=eval_out/az_v29_bigsel_meta
INC=checkpoints/csas_world/az_v25_br/best.pt
POL=checkpoints/csas_world/az_v25_br/policy_csas.pt
VAL=/mnt/data/curling2/csas_v3/checkpoints/value/holdout0/model.pt
MIX=configs/opponents/exp070_meta_nash.json
CFG=configs/exp_076_bigsel_meta_targets.yaml
LOG="$OUT/exp076.log"
LOCK="$OUT/launcher.pid"
GPUS=(0 1 2 3)
NSHARDS=20
GAMES_PER_SHARD=24
SEED_BASE=760764

mkdir -p "$OUT" "$WORK" "$EVAL"
if [[ -f "$LOCK" ]]; then
  old_pid=$(<"$LOCK")
  if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "REFUSING: EXP-076 launcher $old_pid is already alive" | tee -a "$LOG"
    exit 1
  fi
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

source scripts/setup_gpu.sh
export PYTHONUNBUFFERED=1
ENVV="WORLD_BOUNDARY_REMOVAL=1 OMP_NUM_THREADS=2 POLICY_BATCH_CAP=32 VALUE_EVAL_BATCH=256 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
GNN_EDGE_SCALAR_MODE=button_visible_plus_curl_arc_reach_with_outgoing \
GNN_NODE_FEATURE_MODE=none GNN_RELEASE_NODE_MODE=three_plus_takeout_boundary GNN_EDGE_PRUNE_MODE=none \
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.20"

say() { echo "[exp076] $* $(date -u +%FT%TZ)" | tee -a "$LOG"; }
run_gpu() {
  local gpu=$1
  shift
  env $ENVV CUDA_VISIBLE_DEVICES="$gpu" "$@"
}

collect_one() {
  local idx=$1 gpu=$2
  local shard out seed
  shard=$(printf "%04d" "$idx")
  out="$OUT/shard${shard}.npz"
  seed=$((SEED_BASE + idx))
  [[ -f "$out" ]] && return 0
  say "launch shard${shard} games=$GAMES_PER_SHARD gpu=$gpu seed=$seed"
  run_gpu "$gpu" timeout 21600 python3 -m world.search.selfplay \
    --config "$CFG" --games "$GAMES_PER_SHARD" --num-shards 1 --shard-id 0 --split train \
    --seed "$seed" --scorer bigsel --policy "$POL" --value "$VAL" \
    --value-world "$INC" --opponent-mixture "$MIX" \
    --out "$out" --device cuda:0 > "${out%.npz}.log" 2>&1
}

# Excluded end-to-end pilot.  It loads all three mixture members, uses the full
# 192x64 bigsel budget, and verifies learner-only target masking/provenance.
if [[ ! -f "$OUT/PILOT_OK" ]]; then
  say "full-budget pilot start"
  run_gpu 1 timeout 21600 python3 -m world.search.selfplay \
    --config "$CFG" --games 1 --num-shards 1 --shard-id 0 --split train \
    --seed 760700 --scorer bigsel --policy "$POL" --value "$VAL" \
    --value-world "$INC" --opponent-mixture "$MIX" \
    --out "$OUT/pilot.npz" --device cuda:0 > "$OUT/pilot.log" 2>&1
  python3 - <<'PY'
import json, numpy as np
from pathlib import Path
root=Path('artifacts/replay/mcts/az_v29_bigsel_meta')
d=np.load(root/'pilot.npz',allow_pickle=True)
m=json.loads((root/'pilot.manifest.json').read_text())
assert len(d['horizon'])==10 and d['horizon'].tolist()==list(range(10,0,-1))
assert m['records']==10 and m['games']==1 and m['scorer']=='bigsel'
assert m['opponent_mixture']=='configs/opponents/exp070_meta_nash.json'
assert sum(m['opponent_sample_counts'].values())==1
assert int((d['dist_mask']>0).sum()) <= 5
print('EXP076_PILOT_COMPLETE')
PY
  touch "$OUT/PILOT_OK"
  say "full-budget pilot passed"
fi

say "collection start: $NSHARDS shards x $GAMES_PER_SHARD games on GPUs ${GPUS[*]}"
for ((base=0; base<NSHARDS; base+=${#GPUS[@]})); do
  pids=()
  labels=()
  for off in "${!GPUS[@]}"; do
    idx=$((base + off))
    (( idx < NSHARDS )) || continue
    collect_one "$idx" "${GPUS[$off]}" &
    pids+=("$!")
    labels+=("$idx")
    sleep 2
  done
  for k in "${!pids[@]}"; do
    if ! wait "${pids[$k]}"; then
      say "FATAL collection shard ${labels[$k]} failed"
      exit 1
    fi
  done
  complete=$(find "$OUT" -maxdepth 1 -name 'shard*.npz' | wc -l)
  say "collection $complete/$NSHARDS shards complete"
done

python3 - <<'PY'
import glob,json,numpy as np
files=sorted(glob.glob('artifacts/replay/mcts/az_v29_bigsel_meta/shard*.npz'))
assert len(files)==20,len(files)
counts={'v26':0,'v19':0,'v14d':0}; records=sig=0
val_counts={'v26':0,'v19':0,'v14d':0}
for i,p in enumerate(files):
 d=np.load(p,allow_pickle=True)
 m=json.load(open(p.replace('.npz','.manifest.json')))
 assert len(d['horizon'])==240 and all((d['horizon']==h).sum()==24 for h in range(1,11)),p
 assert m['scorer']=='bigsel' and m['games']==24 and m['records']==240,m
 records += len(d['horizon']); sig += int((d['dist_mask']>0).sum())
 for name,n in m['opponent_sample_counts'].items():
  counts[name] += int(n)
  if i % 5 == 0: val_counts[name] += int(n)
assert counts=={'v26':312,'v19':96,'v14d':72},counts
assert val_counts=={'v26':63,'v19':19,'v14d':14},val_counts
print(json.dumps({'games':counts,'val_games':val_counts,'records':records,
                  'sig_plies':sig,'sig_rate_all_plies':sig/records},indent=2))
PY

mkdir -p "$TRAIN" "$VALID"
for f in "$OUT"/shard*.npz; do
  base=$(basename "$f")
  idx=${base#shard}; idx=${idx%.npz}
  if (( 10#$idx % 5 == 0 )); then
    ln -sfn "$(readlink -f "$f")" "$VALID/$base"
  else
    ln -sfn "$(readlink -f "$f")" "$TRAIN/$base"
  fi
done
say "split $(find "$TRAIN" -type l | wc -l) train / $(find "$VALID" -type l | wc -l) val shards"

if [[ ! -f "$WORK/best.pt" ]]; then
  say "training start"
  env -u LD_LIBRARY_PATH $ENVV JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=1 \
    python3 scripts/run_consolidate.py --config configs/exp_076_train.yaml \
      --union "$TRAIN" --mcts-val "$VALID" --init "$INC" --out "$WORK" \
      >> "$WORK/train.log" 2>&1
  say "training complete"
fi
CK="$WORK/best.pt"
[[ -f "$CK" ]] || { say "FATAL no selected best.pt"; exit 1; }

declare -A OPP=(
  [v26]="checkpoints/csas_world/az_v26_br2/best.pt"
  [v19]="checkpoints/csas_world/az_v19_newrules/best.pt"
  [v14d]="checkpoints/csas_world/az_v14d/best.pt"
)
for member in v26 v19 v14d; do
  edir="$EVAL/vs_${member}"
  [[ -f "$edir/summary.json" ]] && continue
  mkdir -p "$edir"
  say "evaluation vs $member start"
  env -u LD_LIBRARY_PATH $ENVV JAX_PLATFORMS=cpu python3 scripts/_eval_parallel.py \
    --champion "$CK" --vs "${OPP[$member]}" --N 250 \
    --horizons 1,2,3,4,5,6,7,8,9,10 --gpus 0,1,2,3 --shards 4 \
    --noisy --sel-noise 4 --out-dir "$edir" >> "$edir/run.log" 2>&1
done

python3 scripts/exp076_analyze.py --root "$EVAL" \
  | tee -a "$LOG" | tee -a experiments_log.md
say "EXP076_DONE"
