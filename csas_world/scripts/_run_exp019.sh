#!/usr/bin/env bash
# EXP-019 = EXP-018's JOINT consolidation (anti-forgetting) + a MODE-BALANCED h10 buffer (anti pp
# under-collection). NOT the sequential curriculum. Steps:
#   1) re-collect ONLY the h10 buffer, balanced across the 6 canonical (mode,guard_slot) states (CPU);
#   2) rebuild the union = exp_017 h1-9 buffers (unchanged) + balanced h10;
#   3) joint-consolidate on the union (warm-start h07/r0, fresh optimizer) -- the EXP-018 recipe.
set -uo pipefail
cd /mnt/data/curling2/csas_world
CV=/mnt/data/curling2/csas_v3
E17=checkpoints/csas_world/exp_017_deploy_robust
H10DIR=checkpoints/csas_world/exp_019_h10balanced
UNION=artifacts/replay/exp019_union_mcts
OUT=checkpoints/csas_world/exp_019_consolidate

export PYTHONPATH=src:$CV/src JAX_PLATFORMS=cpu PYTHONUNBUFFERED=1
export GNN_EDGE_SCALAR_MODE=button_visible_plus_curl_arc_reach_with_outgoing
export GNN_NODE_FEATURE_MODE=none GNN_RELEASE_NODE_MODE=three_plus_takeout_boundary GNN_EDGE_PRUNE_MODE=none
POL=$E17/h09/r1/policy_csas.pt
VW=$E17/h08/r1/model.pt
VAL=$CV/checkpoints/value/holdout0/model.pt

echo "=== STEP 1: re-collect h10 BALANCED (CPU, 2 rounds x 4 shards; build_preplaced_roots balance=True) ==="
rm -rf "$H10DIR"; mkdir -p "$H10DIR"
for rnd in 0 1; do
  seed=$((10000 + rnd)); pids=()
  for k in 0 1 2 3; do
    python3 -m world.search.collect --config configs/exp_017_deploy_robust.yaml --value-world "$VW" \
      --horizon 10 --max-roots 20 --policy "$POL" --value "$VAL" \
      --out "$H10DIR/r${rnd}_shard${k}.npz" --kind mcts --num-shards 4 --shard-id "$k" \
      --device cpu --split train --seed "$seed" > "$H10DIR/collect_r${rnd}_s${k}.log" 2>&1 &
    pids+=($!)
  done
  wait "${pids[@]}"
  echo "  round $rnd: $(ls $H10DIR/r${rnd}_shard*.npz 2>/dev/null | wc -l) shards written"
done
python3 - <<PY
import glob,numpy as np
fs=sorted(glob.glob("$H10DIR/*.npz")); X=np.concatenate([np.load(f)["x0"] for f in fs])
u,c=np.unique(np.round(X,4),axis=0,return_counts=True)
print(f"  balanced h10: {len(X)} records, {len(u)} distinct states, counts={sorted(c.tolist(),reverse=True)}")
PY

echo "=== STEP 2: build union (exp_017 h1-9 + balanced h10) ==="
rm -rf "$UNION"; mkdir -p "$UNION"
for h in 01 02 03 04 05 06 07 08 09; do for r in 0 1; do
  for f in "$E17/h$h/r$r/mcts"/shard*.npz; do [ -e "$f" ] && ln -s "$(readlink -f "$f")" "$UNION/h${h}_r${r}_$(basename "$f")"; done
done; done
for f in "$H10DIR"/r*_shard*.npz; do ln -s "$(readlink -f "$f")" "$UNION/h10bal_$(basename "$f")"; done
echo "  union files: $(ls "$UNION"/*.npz | wc -l)"

echo "=== STEP 3: joint consolidation (EXP-018 recipe, warm-start h07/r0, fresh optimizer) ==="
mkdir -p "$OUT"
bash scripts/_run_consolidate.sh configs/exp_018_consolidate.yaml "$UNION" "$E17/h07/r0/model.pt" "$OUT"
echo "EXP019_TRAIN_DONE"
