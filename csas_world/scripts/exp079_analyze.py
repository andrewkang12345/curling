#!/usr/bin/env python3
"""EXP-079: complete az_v30's population row and apply the robust promotion gate."""
from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path

import numpy as np


LEGACY_ORDER = ("v14d", "v19", "v21", "v25", "v26", "v27")
ALL_ORDER = LEGACY_ORDER + ("v30",)


def read_cells(path: Path, expected_shards: int) -> list[dict]:
    cells = []
    for filename in sorted(glob.glob(str(path / "*__h??__s*of*.json"))):
        payload = json.loads(Path(filename).read_text())
        entries = [
            (key, value)
            for key, value in payload.items()
            if key.startswith("h") and isinstance(value, dict) and "mean_margin" in value
        ]
        if len(entries) != 1:
            raise RuntimeError(f"expected one horizon result in {filename}, got {len(entries)}")
        key, value = entries[0]
        cells.append({
            "horizon": int(key[1:]),
            "mean_margin": float(value["mean_margin"]),
            "winrate": float(value["winrate"]),
            "n_ends": int(value["n_ends"]),
        })
    expected = 10 * expected_shards
    by_h = {h: sum(cell["horizon"] == h for cell in cells) for h in range(1, 11)}
    if len(cells) != expected or any(count != expected_shards for count in by_h.values()):
        raise RuntimeError(
            f"incomplete evaluation in {path}: cells={len(cells)}/{expected}, by_h={by_h}"
        )
    return cells


def summarize(cells: list[dict]) -> dict:
    margins = np.asarray([cell["mean_margin"] for cell in cells], dtype=np.float64)
    winrates = np.asarray([cell["winrate"] for cell in cells], dtype=np.float64)
    counts = np.asarray([cell["n_ends"] for cell in cells], dtype=np.float64)
    mean = float(np.average(margins, weights=counts))
    se = float(margins.std(ddof=1) / math.sqrt(len(margins)))
    return {
        "mean": mean,
        "se": se,
        "t_stat": mean / se if se > 0 else None,
        "winrate": float(np.average(winrates, weights=counts)),
        "matrix_mean": float(margins.mean()),
        "cells": len(cells),
        "ends": int(counts.sum()),
    }


def candidate_components(exp078: Path, root: Path) -> dict[str, dict]:
    sources = {
        "v14d": exp078 / "vs_v14d",
        "v19": exp078 / "vs_v19",
        "v21": root / "vs_v21",
        "v25": root / "vs_v25",
        "v26": exp078 / "vs_v26",
        "v27": root / "vs_v27",
    }
    return {name: summarize(read_cells(path, expected_shards=4)) for name, path in sources.items()}


def legacy_matrix(legacy_root: Path) -> np.ndarray:
    matrix = np.zeros((len(ALL_ORDER), len(ALL_ORDER)), dtype=np.float64)
    for i, a in enumerate(LEGACY_ORDER):
        for j, b in enumerate(LEGACY_ORDER):
            if j <= i:
                continue
            cells = read_cells(legacy_root / f"{a}_vs_{b}", expected_shards=2)
            value = summarize(cells)["matrix_mean"]
            matrix[i, j] = value
            matrix[j, i] = -value
    return matrix


def fictitious_play(matrix: np.ndarray, iterations: int = 20_000) -> np.ndarray:
    n = len(matrix)
    mixture = np.ones(n, dtype=np.float64) / n
    for step in range(1, iterations + 1):
        response = int(np.argmax(matrix @ mixture))
        pure = np.zeros(n, dtype=np.float64)
        pure[response] = 1.0
        mixture = (1.0 - 1.0 / (step + 1)) * mixture + pure / (step + 1)
    return mixture


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="eval_out/exp079_v30_population")
    ap.add_argument("--exp078", default="eval_out/az_v30_paired_meta")
    ap.add_argument("--legacy-root", default="eval_out/exp070_meta")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    root = Path(args.root)
    components = candidate_components(Path(args.exp078), root)
    for result in components.values():
        result["passes"] = bool(
            result["mean"] > 0
            and result["t_stat"] is not None
            and result["t_stat"] >= 2.1
        )

    matrix = legacy_matrix(Path(args.legacy_root))
    v30_index = ALL_ORDER.index("v30")
    for opponent, result in components.items():
        opponent_index = ALL_ORDER.index(opponent)
        matrix[v30_index, opponent_index] = result["matrix_mean"]
        matrix[opponent_index, v30_index] = -result["matrix_mean"]

    ranking = []
    for i, name in enumerate(ALL_ORDER):
        opponent_indices = [j for j in range(len(ALL_ORDER)) if j != i]
        row = matrix[i, opponent_indices]
        worst_local = int(np.argmin(row))
        ranking.append({
            "model": name,
            "maximin": float(row[worst_local]),
            "mean": float(row.mean()),
            "worst_vs": ALL_ORDER[opponent_indices[worst_local]],
        })
    ranking.sort(key=lambda entry: (entry["maximin"], entry["mean"]), reverse=True)

    mixture = fictitious_play(matrix)
    meta_nash = {
        ALL_ORDER[i]: float(mixture[i])
        for i in range(len(ALL_ORDER))
        if mixture[i] > 0.0005
    }
    payoffs_vs_mixture = {
        ALL_ORDER[i]: float(matrix[i] @ mixture) for i in range(len(ALL_ORDER))
    }

    worst_opponent = min(components, key=lambda name: components[name]["mean"])
    certified = all(result["passes"] for result in components.values())
    result = {
        "candidate": "az_v30_paired_meta",
        "incumbent": "az_v25_br",
        "population": list(LEGACY_ORDER),
        "components": components,
        "worst_matchup": {"opponent": worst_opponent, **components[worst_opponent]},
        "head_to_head_vs_v25": components["v25"],
        "certified_robust_champion": certified,
        "gate": "every population dScore > 0 with t >= 2.1",
        "gate_note": (
            "v14d/v19/v26 reuse the already-seen independent EXP-078 full-N evaluation; "
            "v21/v25/v27 are fresh EXP-079 holdouts fixed before evaluation"
        ),
        "matrix_order": list(ALL_ORDER),
        "matrix_dscore": matrix.tolist(),
        "maximin_ranking": ranking,
        "meta_nash": meta_nash,
        "payoffs_vs_meta_nash": payoffs_vs_mixture,
        "meta_nash_note": (
            "seven-model empirical update; legacy-vs-legacy cells retain EXP-070 N=150 "
            "screening resolution, while the v30 row uses N=250"
        ),
    }
    out = Path(args.out) if args.out else root / "result.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    print("\n**EXP-079 az_v30 population robustness gate (auto-appended):**\n")
    for name in LEGACY_ORDER:
        component = components[name]
        verdict = "PASS" if component["passes"] else "FAIL"
        print(
            f"- vs {name}: {component['mean']:+.4f} +/- {component['se']:.4f}/end, "
            f"t={component['t_stat']:+.2f}, winrate={component['winrate']:.4f} "
            f"({component['cells']} cells, {component['ends']} ends) — {verdict}"
        )
    verdict = "NEW CERTIFIED SINGLE ROBUST CHAMPION" if certified else "KEEP az_v25_br"
    worst = components[worst_opponent]
    print(
        f"- worst population matchup: v30 vs {worst_opponent} "
        f"{worst['mean']:+.4f} +/- {worst['se']:.4f}, t={worst['t_stat']:+.2f}"
    )
    print(f"- robust promotion verdict: **{verdict}**")
    print("- maximin ranking: " + ", ".join(
        f"{entry['model']} {entry['maximin']:+.3f}" for entry in ranking
    ))
    print("- updated empirical meta-Nash: " + ", ".join(
        f"{name} {weight:.1%}" for name, weight in sorted(
            meta_nash.items(), key=lambda item: item[1], reverse=True
        )
    ))
    print(f"- machine-readable: `{out}`")


if __name__ == "__main__":
    main()
