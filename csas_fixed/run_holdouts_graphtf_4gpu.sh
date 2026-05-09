#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
COMPS=(0 22230015 23240026 24250026)

pids=()
for i in "${!COMPS[@]}"; do
  comp="${COMPS[$i]}"
  gpu="$i"
  bash "$ROOT/run_holdout_comp_graphtf.sh" "$comp" "$gpu" &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done

exit "$status"

