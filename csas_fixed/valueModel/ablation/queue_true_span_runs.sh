#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 PID [PID ...]" >&2
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] waiting for prior GNN PIDs: $*"
while :; do
  any_alive=0
  for pid in "$@"; do
    if kill -0 "$pid" 2>/dev/null; then
      any_alive=1
      break
    fi
  done
  if [[ "$any_alive" -eq 0 ]]; then
    break
  fi
  sleep 60
done

"${ROOT_DIR}/launch_true_span_runs.sh"
