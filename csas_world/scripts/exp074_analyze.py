#!/usr/bin/env python3
"""Aggregate EXP-074 payoff against the fixed EXP-070 empirical mixture."""
from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path

import numpy as np


def read_cells(path: Path):
    cells = []
    for filename in glob.glob(str(path / "*.json")):
        if filename.endswith("summary.json"):
            continue
        try:
            payload = json.load(open(filename))
        except Exception:
            continue
        for key, value in payload.items():
            if key.startswith("h") and isinstance(value, dict) and "mean_margin" in value:
                cells.append((key, float(value["mean_margin"]), int(value["n_ends"])))
    if not cells:
        raise RuntimeError(f"no evaluation cells in {path}")
    return cells


def summarize(cells):
    values = np.asarray([v for _h, v, _n in cells], dtype=np.float64)
    counts = np.asarray([n for _h, _v, n in cells], dtype=np.float64)
    mean = float(np.average(values, weights=counts))
    # Match the project's standing gate convention: independent horizon/shard
    # cell means are the sampling units.  The end count is reported separately.
    se = float(values.std(ddof=1) / math.sqrt(len(values))) if len(values) > 1 else float("inf")
    return {"mean": mean, "se": se, "cells": len(values), "ends": int(counts.sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="eval_out/az_v28_oppbr_meta")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    root = Path(args.root)
    weights = {"v26": 0.652, "v19": 0.203, "v14d": 0.145}
    components = {name: summarize(read_cells(root / f"vs_{name}")) for name in weights}
    mean = sum(weights[name] * components[name]["mean"] for name in weights)
    se = math.sqrt(sum((weights[name] * components[name]["se"]) ** 2 for name in weights))
    t_stat = mean / se if np.isfinite(se) and se > 0 else float("nan")
    certified = bool(np.isfinite(t_stat) and mean > 0 and t_stat >= 2.1)
    result = {
        "mixture": weights,
        "components": components,
        "expected_dscore": mean,
        "expected_se": se,
        "t_stat": t_stat,
        "certified": certified,
        "gate": "expected dScore > 0 and t >= 2.1",
    }
    out = Path(args.out) if args.out else root / "mixture_result.json"
    out.write_text(json.dumps(result, indent=2) + "\n")

    print("\n**EXP-074 raw mixture gate (auto-appended):**\n")
    for name in ("v26", "v19", "v14d"):
        r = components[name]
        print(f"- vs {name} (w={weights[name]:.3f}): {r['mean']:+.4f} +/- "
              f"{r['se']:.4f}/end ({r['cells']} cells, {r['ends']} ends)")
    verdict = "CERTIFIED PROFITABLE RESPONSE" if certified else "NOT CERTIFIED"
    print(f"- mixture: **{mean:+.4f} +/- {se:.4f}/end, t={t_stat:+.2f} — {verdict}**")
    print(f"- machine-readable: `{out}`")


if __name__ == "__main__":
    main()
