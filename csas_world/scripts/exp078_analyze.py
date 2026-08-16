#!/usr/bin/env python3
"""Aggregate EXP-078 payoff against the fixed EXP-070 empirical mixture."""
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
    return {
        "mean": float(np.average(values, weights=counts)),
        "se": float(values.std(ddof=1) / math.sqrt(len(values))),
        "cells": len(values),
        "ends": int(counts.sum()),
    }


def independent_contrast(mean, se, path):
    prior = json.loads(Path(path).read_text())
    delta = mean - float(prior["expected_dscore"])
    delta_se = math.sqrt(se**2 + float(prior["expected_se"])**2)
    return {
        "prior_result": str(path),
        "prior_expected_dscore": float(prior["expected_dscore"]),
        "delta": delta,
        "se": delta_se,
        "t_stat": delta / delta_se if delta_se > 0 else None,
        "note": "conservative independent-run contrast; not paired and not a promotion gate",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="eval_out/az_v30_paired_meta")
    ap.add_argument("--exp074", default="eval_out/az_v28_oppbr_meta/mixture_result.json")
    ap.add_argument("--exp076", default="eval_out/az_v29_bigsel_meta/mixture_result.json")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    root = Path(args.root)
    weights = {"v26": 0.652, "v19": 0.203, "v14d": 0.145}
    components = {name: summarize(read_cells(root / f"vs_{name}")) for name in weights}
    mean = sum(weights[name] * components[name]["mean"] for name in weights)
    se = math.sqrt(sum((weights[name] * components[name]["se"])**2 for name in weights))
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
        "independent_contrasts": {
            "vs_exp074_vectree": independent_contrast(mean, se, args.exp074),
            "vs_exp076_bigsel": independent_contrast(mean, se, args.exp076),
        },
    }
    out = Path(args.out) if args.out else root / "mixture_result.json"
    out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    print("\n**EXP-078 paired-gated BR mixture gate (auto-appended):**\n")
    for name in ("v26", "v19", "v14d"):
        r = components[name]
        print(f"- vs {name} (w={weights[name]:.3f}): {r['mean']:+.4f} +/- "
              f"{r['se']:.4f}/end ({r['cells']} cells, {r['ends']} ends)")
    verdict = "CERTIFIED PROFITABLE RESPONSE" if certified else "NOT CERTIFIED"
    print(f"- mixture: **{mean:+.4f} +/- {se:.4f}/end, t={t_stat:+.2f} — {verdict}**")
    for label, contrast in result["independent_contrasts"].items():
        print(f"- independent {label}: {contrast['delta']:+.4f} +/- "
              f"{contrast['se']:.4f}, t={contrast['t_stat']:+.2f}")
    print(f"- machine-readable: `{out}`")


if __name__ == "__main__":
    main()
