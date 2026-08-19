#!/usr/bin/env bash
# EXP-079: complete az_v30's population row and apply the post-EXP-070 robust gate.
set -euo pipefail
cd /mnt/data/curling2/csas_world

OUT=eval_out/exp079_v30_population
LOG="$OUT/exp079.log"
LOCK="$OUT/launcher.pid"
CANDIDATE=checkpoints/csas_world/az_v30_paired_meta/best.pt
EXP078=eval_out/az_v30_paired_meta
LEGACY=eval_out/exp070_meta

mkdir -p "$OUT"
if [[ -f "$LOCK" ]]; then
  old_pid=$(<"$LOCK")
  if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "REFUSING: EXP-079 launcher $old_pid is already alive" | tee -a "$LOG"
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
  [v21]="checkpoints/csas_world/az_v21_stt2x/best.pt"
  [v25]="checkpoints/csas_world/az_v25_br/best.pt"
  [v27]="checkpoints/csas_world/az_v27_vectree/best.pt"
)

say() { echo "[exp079] $* $(date -u +%FT%TZ)" | tee -a "$LOG"; }

validate_eval() {
  local directory=$1
  python3 - "$directory" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
files = sorted(root.glob("*__h??__s*of4.json"))
assert len(files) == 40, (root, len(files))
for horizon in range(1, 11):
    matches = list(root.glob(f"*__h{horizon:02d}__s*of4.json"))
    assert len(matches) == 4, (root, horizon, len(matches))
    for path in matches:
        value = json.loads(path.read_text())[f"h{horizon:02d}"]
        assert all(isinstance(value[key], (int, float)) for key in ("n_ends", "mean_margin", "winrate"))
summary = json.loads((root / "summary.json").read_text())
assert len(summary) == 1, (root, summary.keys())
horizons = next(iter(summary.values()))
assert len(horizons) == 10 and all(horizons[f"h{h:02d}"]["got"] == 4 for h in range(1, 11))
PY
}

[[ -s "$CANDIDATE" ]] || { say "FATAL missing candidate $CANDIDATE"; exit 1; }
for member in v14d v19 v26; do
  validate_eval "$EXP078/vs_${member}"
done
say "reused EXP-078 rows validated: v14d v19 v26"

for member in v21 v25 v27; do
  directory="$OUT/vs_${member}"
  if [[ -f "$directory/summary.json" ]]; then
    validate_eval "$directory"
    say "evaluation vs $member already complete"
    continue
  fi
  mkdir -p "$directory"
  say "evaluation vs $member start"
  env -u LD_LIBRARY_PATH $ENVV python3 scripts/_eval_parallel.py \
    --champion "$CANDIDATE" --vs "${OPP[$member]}" --N 250 \
    --horizons 1,2,3,4,5,6,7,8,9,10 --gpus 0,1,2,3 --shards 4 \
    --noisy --sel-noise 4 --out-dir "$directory" >> "$directory/run.log" 2>&1
  validate_eval "$directory"
  say "evaluation vs $member complete"
done

if [[ ! -f "$OUT/result.json" ]]; then
  python3 scripts/exp079_analyze.py --root "$OUT" --exp078 "$EXP078" --legacy-root "$LEGACY" \
    | tee -a "$LOG" | tee -a experiments_log.md
fi
say "EXP079_DONE"
