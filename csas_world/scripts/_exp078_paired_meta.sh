#!/usr/bin/env bash
# EXP-078: train a paired-gated BR to the provisional EXP-070 meta-Nash mixture.
set -euo pipefail
cd /mnt/data/curling2/csas_world

OUT=artifacts/replay/mcts/az_v30_paired_meta
WORK=checkpoints/csas_world/az_v30_paired_meta
TRAIN=artifacts/replay/az_v30_paired_meta_train
VALID=artifacts/replay/az_v30_paired_meta_val
EVAL=eval_out/az_v30_paired_meta
INC=checkpoints/csas_world/az_v25_br/best.pt
POL=checkpoints/csas_world/az_v25_br/policy_csas.pt
VAL=/mnt/data/curling2/csas_v3/checkpoints/value/holdout0/model.pt
CFG=configs/exp_078_paired_meta_targets.yaml
LOG="$OUT/exp078.log"
LOCK="$OUT/launcher.pid"
GPUS=(0 1 2 3)
GAMES_PER_SHARD=4
NSHARDS=120
CONCURRENCY=8

mkdir -p "$OUT" "$WORK" "$EVAL"
if [[ -f "$LOCK" ]]; then
  old_pid=$(<"$LOCK")
  if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "REFUSING: EXP-078 launcher $old_pid is already alive" | tee -a "$LOG"
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

say() { echo "[exp078] $* $(date -u +%FT%TZ)" | tee -a "$LOG"; }
run() { env -u LD_LIBRARY_PATH $ENVV "$@"; }
run_gpu() {
  local gpu=$1
  shift
  env -u LD_LIBRARY_PATH $ENVV CUDA_VISIBLE_DEVICES="$gpu" "$@"
}
run_collect() {
  local tag=$1 games=$2 seed=$3 member=$4 gpu=$5 out=$6
  say "launch $tag member=$member games=$games gpu=$gpu seed=$seed"
  run_gpu "$gpu" timeout 43200 python3 -m world.search.selfplay \
    --config "$CFG" --games "$games" --num-shards 1 --shard-id 0 --split train \
    --seed "$seed" --scorer opp_vectree_paired --policy "$POL" --value "$VAL" \
    --value-world "$INC" --baseline-world "$INC" --opponent-world "${OPP[$member]}" \
    --out "$out" --device cuda:0 > "${out%.npz}.log" 2>&1
}

# Full-budget excluded pilot: one end against each support member in parallel.
if [[ ! -f "$OUT/PILOT_OK" ]]; then
  say "three-member full-budget pilot start"
  pilot_pids=()
  pilot_members=(v26 v19 v14d)
  for j in 0 1 2; do
    member=${pilot_members[$j]}
    pout="$OUT/pilot_${member}.npz"
    if [[ ! -f "$pout" ]]; then
      run_collect "pilot_$member" 1 $((780010 + j)) "$member" $((j + 1)) "$pout" &
      pilot_pids+=("$!")
    fi
  done
  for pid in "${pilot_pids[@]}"; do
    if ! wait "$pid"; then say "FATAL full-budget pilot worker $pid failed"; exit 1; fi
  done
  run python3 - <<'PY'
import json
from pathlib import Path
import numpy as np
root=Path('artifacts/replay/mcts/az_v30_paired_meta')
for member in ('v26','v19','v14d'):
 p=root/f'pilot_{member}.npz'; d=np.load(p,allow_pickle=False)
 m=json.loads((root/f'pilot_{member}.manifest.json').read_text())
 q=json.loads((root/f'pilot_{member}.diag.json').read_text())
 assert len(d['horizon'])==10 and d['horizon'].tolist()==list(range(10,0,-1)),member
 assert m['records']==10 and m['games']==1 and m['scorer']=='opp_vectree_paired',m
 assert m['baseline_world']=='checkpoints/csas_world/az_v25_br/best.pt'
 assert m['paired_gate']['evaluated']==5
 assert m['paired_gate']['accepted']+m['paired_gate']['fallback']==5
 learner=np.asarray(d['dist_mask'])>0
 assert int(learner.sum())==5
 assert np.all(np.count_nonzero(d['dist_weights'][learner],axis=1)==1)
 assert q['all']['n']==5 and q['all']['accepted']+q['all']['fallback']==5
print('EXP078_FULL_BUDGET_PILOT_COMPLETE')
PY
  touch "$OUT/PILOT_OK"
  say "three-member full-budget pilot passed"
fi

# Multiplicative permutation: 78/24/18 four-game shards = 312/96/72 games.
member_for_shard() {
  local idx=$1 perm=$(( (idx * 37) % 120 ))
  if (( perm < 78 )); then echo v26
  elif (( perm < 102 )); then echo v19
  else echo v14d
  fi
}

say "collection start: $NSHARDS x $GAMES_PER_SHARD games, concurrency=$CONCURRENCY, GPUs ${GPUS[*]}"
for ((base=0; base<NSHARDS; base+=CONCURRENCY)); do
  pids=()
  labels=()
  for ((off=0; off<CONCURRENCY && base+off<NSHARDS; off++)); do
    idx=$((base + off))
    member=$(member_for_shard "$idx")
    shard=$(printf "%04d" "$idx")
    out="$OUT/shard${shard}_${member}.npz"
    [[ -f "$out" ]] && continue
    gpu=${GPUS[$((off % ${#GPUS[@]}))]}
    run_collect "shard$shard" "$GAMES_PER_SHARD" $((780100 + idx)) \
      "$member" "$gpu" "$out" &
    pids+=("$!")
    labels+=("$shard")
    sleep 2
  done
  for k in "${!pids[@]}"; do
    if ! wait "${pids[$k]}"; then say "FATAL collection shard ${labels[$k]} failed"; exit 1; fi
  done
  complete=$(find "$OUT" -maxdepth 1 -name 'shard*.npz' | wc -l)
  say "collection $complete/$NSHARDS shards complete"
done

run python3 - <<'PY'
import glob,json
from collections import Counter
import numpy as np
files=sorted(glob.glob('artifacts/replay/mcts/az_v30_paired_meta/shard*.npz'))
assert len(files)==120,len(files)
games=Counter(); accepted=Counter(); fallback=Counter(); targets=Counter()
records=0
for p in files:
 d=np.load(p,allow_pickle=False)
 m=json.load(open(p.replace('.npz','.manifest.json')))
 q=json.load(open(p.replace('.npz','.diag.json')))
 assert len(d['horizon'])==40 and all(int((d['horizon']==h).sum())==4 for h in range(1,11)),p
 assert m['scorer']=='opp_vectree_paired' and m['games']==4 and m['records']==40,m
 assert m['paired_gate']['evaluated']==20
 assert m['paired_gate']['accepted']+m['paired_gate']['fallback']==20
 learner=np.asarray(d['dist_mask'])>0
 assert int(learner.sum())==20,p
 assert np.all(np.count_nonzero(d['dist_weights'][learner],axis=1)==1),p
 assert q['all']['n']==20 and q['all']['accepted']+q['all']['fallback']==20
 member=next(name for name in ('v26','v19','v14d') if f'_{name}.npz' in p)
 games[member]+=4; accepted[member]+=m['paired_gate']['accepted']; fallback[member]+=m['paired_gate']['fallback']
 for h in d['horizon'][learner]: targets[int(h)]+=1
 records+=len(d['horizon'])
assert games==Counter({'v26':312,'v19':96,'v14d':72}),games
assert records==4800 and sum(targets.values())==2400
# The source roots' initial throwing block is not fixed.  Alternating numeric
# opponent_block therefore produced the complete corpus with a one-target parity
# offset: 239 on each odd horizon and 241 on each even horizon.  Validate that
# exact frozen shape rather than incorrectly requiring 240 in every cell.
expected_targets=Counter({h:(241 if h % 2 == 0 else 239) for h in range(1,11)})
assert targets==expected_targets,targets
print(json.dumps({'games':dict(games),'records':records,'targets_by_horizon':dict(sorted(targets.items())),
                  'accepted_by_member':dict(accepted),'fallback_by_member':dict(fallback)},indent=2))
PY

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
say "split $(find "$TRAIN" -type l | wc -l) train / $(find "$VALID" -type l | wc -l) val shards (val games 64/20/12)"

if [[ ! -f "$WORK/best.pt" ]]; then
  say "training start"
  env -u LD_LIBRARY_PATH $ENVV JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=1 \
    python3 scripts/run_consolidate.py --config configs/exp_078_train.yaml \
      --union "$TRAIN" --mcts-val "$VALID" --init "$INC" --out "$WORK" \
      >> "$WORK/train.log" 2>&1
  say "training complete"
fi
CK="$WORK/best.pt"
[[ -f "$CK" ]] || { say "FATAL no selected best.pt"; exit 1; }

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

run python3 scripts/exp078_analyze.py --root "$EVAL" \
  | tee -a "$LOG" | tee -a experiments_log.md
say "EXP078_DONE"
