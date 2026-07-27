#!/usr/bin/env bash
# Wait for EXP-019 (balanced-h10 joint consolidation) to finish, then:
#  Phase 1 (the pp test): per-mode h10 winrate BEFORE (exp018) vs AFTER (exp019), deterministic,
#          parallelized one (model,mode) job per GPU -> did balancing fix pp_left/right?
#  Phase 2 (no-regression): full 1->10 NOISY eval exp019 vs prior -> MEAN-of-pairs vs EXP-018's 0.557.
set -u
cd /mnt/data/curling2/csas_world
LOG=checkpoints/csas_world/exp_019_consolidate/exp019.log
E18=checkpoints/csas_world/exp_018_consolidate/last.pt
E19=checkpoints/csas_world/exp_019_consolidate/last.pt
GNN="GNN_EDGE_SCALAR_MODE=button_visible_plus_curl_arc_reach_with_outgoing GNN_NODE_FEATURE_MODE=none GNN_RELEASE_NODE_MODE=three_plus_takeout_boundary GNN_EDGE_PRUNE_MODE=none"

for i in $(seq 1 240); do
  grep -qa "EXP019_TRAIN_DONE\|CONSOLIDATE_EXIT" "$LOG" 2>/dev/null && break
  sleep 60
done
[ -f "$E19" ] || { echo ">> EXP-019 ckpt missing; abort"; echo "EXP019_EVAL_DONE"; exit 0; }

echo "=== PHASE 1: per-mode h10 winrate (DETERMINISTIC), BEFORE=exp018 vs AFTER=exp019 ==="
mkdir -p eval_out/exp019_modesplit
g=0; pids=()
for tag in exp018:$E18 exp019:$E19; do
  name=${tag%%:*}; ckpt=${tag##*:}
  for mode in standard pp_left pp_right; do
    of="eval_out/exp019_modesplit/${name}_${mode}.log"
    env CUDA_VISIBLE_DEVICES=$g PYTHONPATH=src:/mnt/data/curling2/csas_v3/src JAX_PLATFORMS=cpu PYTHONUNBUFFERED=1 $GNN \
      python3 scripts/_eval_h10_by_mode.py --device cuda:0 --only-mode $mode --sel-noise-samples 0 \
      --max-per-mode 60 --champion "$ckpt" > "$of" 2>&1 &
    pids+=($!); g=$(((g+1)%4))
    [ ${#pids[@]} -ge 4 ] && { wait "${pids[@]}"; pids=(); }
  done
done
wait "${pids[@]}" 2>/dev/null || true
echo "  model    mode       winrate (h10, deterministic)"
for name in exp018 exp019; do for mode in standard pp_left pp_right; do
  line=$(grep -aE "^$mode " "eval_out/exp019_modesplit/${name}_${mode}.log" 2>/dev/null | tail -1)
  echo "  ${name}  ${line:-($mode: no result)}"
done; done

echo "=== PHASE 2: EXP-019 full 1->10 NOISY vs prior (compare MEAN-of-pairs to EXP-018 0.557) ==="
python3 scripts/_eval_parallel.py --champion "$E19" --vs prior --N 700 \
  --horizons 1,2,3,4,5,6,7,8,9,10 --gpus 0,1,2,3 --shards 4 --noisy --out-dir eval_out/exp019_vs_prior
echo "EXP019_EVAL_DONE"
