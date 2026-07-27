#!/usr/bin/env python3
"""Build the az_v13 significance-filtered buffers from EXISTING champion-generation
collections (zero recollection — the COLLECTIONS.md reuse pattern).

For every record: keep ALL targets (value = grounded MC returns, consistency, reward)
but zero `dist_mask` unless the distillation target is CONFIDENT (top soft-topk weight
>= a per-source threshold). Rationale: distilling flat/noise-ranked targets erodes the
policy's sharp proposal distribution (the flat-target-erosion channel); masked records
still anchor the value/dynamics heads. Writes REAL COPIES (originals untouched).

Sources (champion-generation only; az_v11 excluded — noise-starved targets):
  az_v9_selfplay_iter{3..6}   thresh 0.55  (2-ply value-leaf targets)
  az_v10_terminal_iter{1,2}   thresh 0.60  (flat terminal k_ego=4; noise-sharp risk -> stricter)
  az_v12_screentree_iter{1,2} thresh 0.55  (robust screen -> tree; best quality)

Split: shards 0-2 -> train, shard 3 -> val (same 75/25 convention). The val partition is
filtered identically, so `val_policy_distill_mcts` on it measures "matches the search
where the search is significant" — the aligned checkpoint-selection metric (fix 3).
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

SOURCES = [
    ("az_v9_selfplay_iter3", 0.55), ("az_v9_selfplay_iter4", 0.55),
    ("az_v9_selfplay_iter5", 0.55), ("az_v9_selfplay_iter6", 0.55),
    ("az_v10_terminal_iter1", 0.60), ("az_v10_terminal_iter2", 0.60),
    ("az_v12_screentree_iter1", 0.55), ("az_v12_screentree_iter2", 0.55),
]

def main():
    out_train = ROOT / "artifacts/replay/az_v13_conf_train"
    out_val = ROOT / "artifacts/replay/az_v13_conf_val"
    for d in (out_train, out_val):
        d.mkdir(parents=True, exist_ok=True)
        for f in d.glob("*.npz"):
            f.unlink()
    tot = dict(train=[0, 0], val=[0, 0])   # [records, active-distill]
    for src, thresh in SOURCES:
        for k in range(4):
            fp = ROOT / f"artifacts/replay/mcts/{src}/shard{k}.npz"
            if not fp.exists():
                print(f"[skip] {fp}")
                continue
            d = dict(np.load(fp, allow_pickle=True))
            w = d["dist_weights"]
            w = w / np.maximum(w.sum(axis=1, keepdims=True), 1e-9)
            conf = w.max(axis=1) >= thresh
            d["dist_mask"] = (d["dist_mask"].astype(np.float32) * conf.astype(np.float32))
            split = "train" if k < 3 else "val"
            out = (out_train if k < 3 else out_val) / f"{src}_shard{k}.npz"
            np.savez_compressed(out, **d)
            tot[split][0] += len(conf)
            tot[split][1] += int((d["dist_mask"] > 0).sum())
    for split, (n, a) in tot.items():
        print(f"[conf-buffers] {split}: {n} records, {a} active-distill ({a/max(n,1):.1%})")
    print("CONF_BUFFERS_DONE")


if __name__ == "__main__":
    main()
