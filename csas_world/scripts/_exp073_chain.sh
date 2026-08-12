#!/usr/bin/env bash
set -euo pipefail
cd /mnt/data/curling2/csas_world
OUT=eval_out/exp073_h4; OUT2=eval_out/exp073_h4_seedB
LOG="$OUT/chain.log"; mkdir -p "$OUT" "$OUT2"
say() { echo "[exp073] $* $(date -u +%H:%M)" | tee -a "$LOG"; }
run() { env -u LD_LIBRARY_PATH "$@"; }
wait_all() {
  local failed=0 pid
  for pid in "$@"; do
    wait "$pid" || failed=1
  done
  return "$failed"
}
export WORLD_BOUNDARY_REMOVAL=1 PYTHONUNBUFFERED=1
source scripts/setup_gpu.sh
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.11 VALUE_EVAL_BATCH=128 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

say "states"
[ -f "$OUT/states.npz" ] || CUDA_VISIBLE_DEVICES=0 python3 scripts/exp073_h4_finite.py \
  --phase states --n-states 24 --out-dir "$OUT" >> "$OUT/states.log" 2>&1
cp -n "$OUT/states.npz" "$OUT2/" 2>/dev/null || true

say "finite reference (8 shards)"
pids=(); for k in $(seq 0 7); do
  CUDA_VISIBLE_DEVICES=$((k % 4)) python3 scripts/exp073_h4_finite.py --phase ref \
    --shard-id $k --num-shards 8 --out-dir "$OUT" >> "$OUT/ref_shard$k.log" 2>&1 &
  pids+=($!); sleep 4
done
wait_all "${pids[@]}"
say "reference done"

say "yardstick stability: independent CRN derivation on 24 states"
pids=(); for k in $(seq 0 7); do
  CUDA_VISIBLE_DEVICES=$((k % 4)) python3 scripts/exp073_h4_finite.py --phase ref \
    --ref-seed-base 190000 --n-states 24 --shard-id $k --num-shards 8 --out-dir "$OUT2" \
    >> "$OUT2/ref_shard$k.log" 2>&1 &
  pids+=($!); sleep 4
done
wait_all "${pids[@]}"
say "stability reference done"

say "search arms (vt_plain vs vt_confirm, 2 seeds, 12 shards)"
pids=(); for k in $(seq 0 11); do
  CUDA_VISIBLE_DEVICES=$((k % 4)) python3 scripts/exp073_h4_finite.py --phase search \
    --shard-id $k --num-shards 12 --out-dir "$OUT" >> "$OUT/search_shard$k.log" 2>&1 &
  pids+=($!); sleep 3
done
wait_all "${pids[@]}"
say "search done"

python3 - <<'PY' | tee -a "$LOG" | tee -a experiments_log.md
import json, glob
import numpy as np
from scipy.stats import spearmanr
A, Bt = {}, {}
for f in glob.glob("eval_out/exp073_h4/ref_shard*.jsonl"):
    for l in open(f):
        r = json.loads(l); A[r["sid"]] = np.asarray(r["value"])
for f in glob.glob("eval_out/exp073_h4_seedB/ref_shard*.jsonl"):
    for l in open(f):
        r = json.loads(l); Bt[r["sid"]] = np.asarray(r["value"])
both = sorted(set(A) & set(Bt))
if both:
    rho = [spearmanr(A[s], Bt[s]).correlation for s in both]
    top1 = [int(np.argmax(A[s]) == np.argmax(Bt[s])) for s in both]
    d = [float(np.abs(A[s] - Bt[s]).mean()) for s in both]
    reg = [float(A[s].max() - A[s][int(np.argmax(Bt[s]))]) for s in both]
    print(f"\n**EXP-073 YARDSTICK STABILITY** (independent CRN derivation, {len(both)} states): "
          f"Spearman rho {np.mean(rho):.3f}, top-1 agreement {np.mean(top1):.0%}, "
          f"mean |value diff| {np.mean(d):.3f}/end, "
          f"regret of seedB's pick under seedA's table {np.mean(reg):.3f}/end")
PY
run python3 scripts/exp073_h4_finite.py --phase aggregate --out-dir "$OUT" \
  | tee -a "$LOG" | tee -a experiments_log.md
say "EXP073_DONE"
