#!/usr/bin/env python3
"""EXP-070 analysis: population meta-game -> maximin, meta-Nash, cycle detection.

Reads the pairwise cross-play cells written by _exp070_metagame.sh and reports
what our head-to-head promotion rule cannot see:

  * MAXIMIN  — each model's worst-case dScore over the population. This is the
    robustness / near-unexploitability ranking (a Nash-approaching policy has a
    high floor even when it loses individual matchups).
  * META-NASH — the equilibrium mixture of the empirical zero-sum matrix
    (fictitious play), i.e. what a PSRO champion would actually be.
  * CYCLES — 3-cycles A>B>C>A in the sign pattern; the direct evidence for or
    against the intransitivity risk that makes head-to-head ratchets diverge.
"""
import argparse
import glob
import json
import math
from itertools import combinations
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--out-dir", default="eval_out/exp070_meta")
args = ap.parse_args()
OUT = Path(args.out_dir)


def cell(d):
    """dScore (A perspective) + SE from one pair directory."""
    ms, W, N = [], 0.0, 0.0
    for f in glob.glob(f"{d}/*__h*__s*.json"):
        j = json.load(open(f))
        for k, v in j.items():
            if k.startswith("h") and isinstance(v, dict):
                ms.append(v["mean_margin"]); W += v["winrate"] * v["n_ends"]; N += v["n_ends"]
    if not ms:
        return None
    mu = float(np.mean(ms))
    se = float(np.std(ms, ddof=1) / math.sqrt(len(ms)))
    return mu, se, W / N, int(N)


pairs = {}
names = set()
for d in sorted(OUT.glob("*_vs_*")):
    a, b = d.name.split("_vs_")
    c = cell(d)
    if c is None:
        continue
    pairs[(a, b)] = c
    names |= {a, b}
order = [n for n in ("v14d", "v19", "v21", "v25", "v26", "v27") if n in names]
if not order:
    print("no cells yet")
    raise SystemExit(0)
n = len(order)
M = np.full((n, n), np.nan)
for (a, b), (mu, se, wr, N) in pairs.items():
    i, j = order.index(a), order.index(b)
    M[i, j] = mu
    M[j, i] = -mu                      # zero-sum by construction (dScore is symmetric)
np.fill_diagonal(M, 0.0)

print(f"EXP-070 population meta-game ({n} models, {len(pairs)} measured pairs, "
      f"k=4, ~{list(pairs.values())[0][3]} ends/cell)\n")
print("dScore matrix (row perspective, +row wins):")
print("          " + "".join(f"{c:>9s}" for c in order))
for i, r in enumerate(order):
    print(f"  {r:8s}" + "".join("      —  " if i == j else f"{M[i,j]:+9.3f}" for j in range(n)))

print("\nROBUSTNESS (what the head-to-head rule cannot see):")
print(f"  {'model':8s} {'maximin':>9s} {'mean':>9s} {'worst vs':>10s}")
rank = []
for i, r in enumerate(order):
    row = np.array([M[i, j] for j in range(n) if j != i])
    worst = order[[j for j in range(n) if j != i][int(np.argmin(row))]]
    rank.append((float(row.min()), float(row.mean()), r, worst))
for mn, mean, r, worst in sorted(rank, reverse=True):
    print(f"  {r:8s} {mn:+9.3f} {mean:+9.3f} {worst:>10s}")
print("  (maximin = worst-case dScore over the population: higher floor = less exploitable)")

# ---- meta-Nash by fictitious play on the empirical zero-sum matrix ----
p = np.ones(n) / n
for t in range(1, 20001):
    br = int(np.argmax(M @ p))
    e = np.zeros(n); e[br] = 1.0
    p = (1 - 1.0 / (t + 1)) * p + (1.0 / (t + 1)) * e
val = float(p @ M @ p)
print("\nMETA-NASH mixture (fictitious play, 20k iters):")
for i in np.argsort(p)[::-1]:
    if p[i] > 0.005:
        print(f"  {order[i]:8s} {p[i]:6.1%}   (its dScore vs the mixture: {float(M[i] @ p):+.3f})")
print(f"  equilibrium value {val:+.4f} (0 = balanced, as a symmetric zero-sum game should be)")
print("  exploitability of each model vs the mixture = -(its score vs mixture) when negative")

# ---- intransitivity ----
cycles = []
for a, b, c in combinations(range(n), 3):
    for x, y, z in ((a, b, c), (a, c, b)):
        if M[x, y] > 0 and M[y, z] > 0 and M[z, x] > 0:
            cycles.append((order[x], order[y], order[z],
                           min(M[x, y], M[y, z], M[z, x])))
print("\nINTRANSITIVITY (3-cycles in the sign pattern):")
if not cycles:
    print("  none — the population is TRANSITIVE; a head-to-head ratchet cannot cycle here")
else:
    for x, y, z, w in sorted(cycles, key=lambda t: -t[3]):
        print(f"  {x} > {y} > {z} > {x}   (weakest edge {w:+.3f})")
    print("  => head-to-head promotion CAN cycle; PSRO/mixture-BR is the correct loop")
