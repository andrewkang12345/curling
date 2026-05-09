#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/data/curling2/csas_fixed
PYTHON=/opt/pytorch/bin/python
STONES="$ROOT/2026/Stones.csv"
INVERSE_GLOB="${INVERSE_GLOB:-$ROOT/inverse_current/stones_with_estimates.chunk*.csv}"
MANIFEST="$ROOT/xscore_examples/manifest.csv"
KEY_DIR="$ROOT/xscore_examples/keys"
LOG_DIR="$ROOT/xscore_examples/logs"

export CSAS_PRELOAD_NVIDIA_LIBS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH="$ROOT:$ROOT/inverse:$ROOT/valueModel:/mnt/data/curling2/testBrax/testEnv/lib/python3.12/site-packages:/opt/pytorch/lib/python3.12/site-packages"
export LD_LIBRARY_PATH="/opt/pytorch/lib/python3.12/site-packages/nvidia/nvjitlink/lib:/opt/pytorch/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:/opt/pytorch/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib:/opt/pytorch/lib/python3.12/site-packages/nvidia/cublas/lib:/opt/pytorch/lib/python3.12/site-packages/nvidia/cusparse/lib:/opt/pytorch/lib/python3.12/site-packages/nvidia/cusolver/lib:/opt/pytorch/lib/python3.12/site-packages/nvidia/cufft/lib:/opt/pytorch/lib/python3.12/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}"

mapfile -t ROWS < <("$PYTHON" - <<'PY'
import csv
with open("/mnt/data/curling2/csas_fixed/xscore_examples/manifest.csv", newline="") as f:
    r = csv.DictReader(f)
    for row in r:
        shot = ",".join([row["CompetitionID"], row["SessionID"], row["GameID"], row["EndID"], row["ShotID"]])
        print(f'{int(row["example_id"])}\t{row["holdout_competition"]}\t{shot}')
PY
)

mkdir -p "$KEY_DIR" "$LOG_DIR"

running=0
for rec in "${ROWS[@]}"; do
  IFS=$'\t' read -r EX HOLDOUT SHOT <<< "$rec"
  GPU=$(( (EX - 1) % 4 ))
  OUT="$ROOT/xscore_examples/example_$(printf '%02d' "$EX").png"
  NOUT="$ROOT/xscore_examples/example_$(printf '%02d' "$EX")_neighbors.csv"
  KEYS="$KEY_DIR/example_$(printf '%02d' "$EX").csv"
  MODEL="$ROOT/holdouts/$HOLDOUT/model/model.pt"
  NOISE="$ROOT/holdouts/$HOLDOUT/scoring/noise/local.json"
  (
    export CUDA_VISIBLE_DEVICES="$GPU"
    export JAX_PLATFORMS=cuda
    export XLA_PYTHON_CLIENT_MEM_FRACTION=0.22
    "$PYTHON" "$ROOT/visualize.py" \
      --device cuda \
      --stones-csv "$STONES" \
      --inverse-glob "$INVERSE_GLOB" \
      --value-model "$MODEL" \
      --noise-config "$NOISE" \
      --test-keys-csv "$KEYS" \
      --shot "$SHOT" \
      --out "$OUT" \
      --neighbors-out "$NOUT" \
      > "$LOG_DIR/render_$(printf '%02d' "$EX").log" 2>&1
  ) &
  running=$((running+1))
  if [ "$running" -ge 4 ]; then
    wait -n
    running=$((running-1))
  fi
done
wait

"$PYTHON" - <<'PY'
import pandas as pd
from pathlib import Path

base = Path("/mnt/data/curling2/csas_fixed/xscore_examples")
manifest = pd.read_csv(base / "manifest.csv")
for ex in manifest["example_id"].astype(int):
    png = f"example_{ex:02d}.png"
    ncsv = f"example_{ex:02d}_neighbors.csv"
    idx = manifest.index[manifest["example_id"] == ex][0]
    manifest.loc[idx, "png"] = png
    manifest.loc[idx, "neighbors_csv"] = ncsv
    df = pd.read_csv(base / ncsv)
    col = "dv_sim" if "dv_sim" in df.columns else ("dxscore" if "dxscore" in df.columns else None)
    if col is not None:
        manifest.loc[idx, "p10"] = float(df[col].quantile(0.10))
        manifest.loc[idx, "p50"] = float(df[col].quantile(0.50))
        manifest.loc[idx, "p90"] = float(df[col].quantile(0.90))
        manifest.loc[idx, "dv_mean_viz"] = float(df[col].mean())
manifest.to_csv(base / "manifest.csv", index=False)
print(f"[done] refreshed {len(manifest)} xscore examples")
PY
