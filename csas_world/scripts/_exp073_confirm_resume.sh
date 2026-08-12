#!/usr/bin/env bash
# Resume only the EXP-073 confirmation shards that OOMed in the original 12-way
# launch.  Their shard IDs all mapped to two GPUs, so run one worker per GPU in
# the first wave and the remaining two workers in a second wave.
set -euo pipefail
cd /mnt/data/curling2/csas_world

OUT=eval_out/exp073_h4
LOG="$OUT/confirm_resume.log"
source scripts/setup_gpu.sh
export WORLD_BOUNDARY_REMOVAL=1 PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.18
export VALUE_EVAL_BATCH=128 POLICY_BATCH_CAP=64
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

run_wave() {
  local specs=("$@") pids=() labels=() spec gpu shard
  for spec in "${specs[@]}"; do
    gpu="${spec%%:*}"
    shard="${spec##*:}"
    echo "[exp073-resume] starting shard $shard on GPU $gpu $(date -u +%FT%TZ)" | tee -a "$LOG"
    CUDA_VISIBLE_DEVICES="$gpu" python3 scripts/exp073_h4_finite.py --phase search \
      --shard-id "$shard" --num-shards 12 --out-dir "$OUT" \
      >> "$OUT/search_shard$shard.log" 2>&1 &
    pids+=("$!")
    labels+=("$shard")
  done

  local failed=0 i
  for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then
      echo "[exp073-resume] shard ${labels[$i]} done $(date -u +%FT%TZ)" | tee -a "$LOG"
    else
      echo "[exp073-resume] shard ${labels[$i]} FAILED $(date -u +%FT%TZ)" | tee -a "$LOG"
      failed=1
    fi
  done
  return "$failed"
}

run_wave 0:2 1:3 2:6 3:7
run_wave 0:10 1:11

python3 - <<'PY'
import glob, json

keys = []
for path in glob.glob("eval_out/exp073_h4/search_shard*.jsonl"):
    with open(path) as fh:
        keys.extend((row["sid"], row["arm"], row["seed"])
                    for row in map(json.loads, fh) if row)
if len(set(keys)) != 96:
    raise SystemExit(f"expected 96 unique search rows, found {len(set(keys))}")
print("[exp073-resume] all 96 unique search rows complete")
PY

python3 scripts/exp073_h4_finite.py --phase aggregate --out-dir "$OUT" \
  | tee -a "$LOG" | tee -a experiments_log.md
echo "[exp073-resume] EXP073_CONFIRM_RESUME_DONE $(date -u +%FT%TZ)" | tee -a "$LOG"
