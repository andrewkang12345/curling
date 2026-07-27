# Collections registry — provenance + reuse notes

Every MCTS/self-play collection is preserved (52 MB total as of 2026-07-11; never delete).
Shards written by `world.search.selfplay` from 2026-07-11 onward carry a
`shardN.manifest.json` sidecar with full provenance. This file documents everything
collected BEFORE that, plus reuse guidance.

## Reuse rules of thumb

- **Train-side tweaks (loss weights, VFM, epochs, buffer windows, replay mixes): reuse freely** —
  no recollect needed. Precedents: az_v5 reused az_v4's collection via symlink (isolated the
  VFM flag); the fresh-window probe reused it5+it6 only.
- **Partial recollects are fine**: the record schema is identical across all operators, so
  unions can keep most iters and replace a few (or patch single horizons). BUT label the mix —
  distillation targets from different operators are different estimands (tree-Q vs flat
  terminal-MC vs screen+tree) and different noise floors.
- **Mind generation staleness**: targets are tied to the policy that generated them. Training a
  much-improved model on old-generation targets pulls it backward (measured: az_v9 iters 3-6).
  Reuse old generations as *anchors* deliberately, not by default.
- **Eval draws are reusable as baselines** (copied, never symlinked — racing-incident lesson).
  The champion (az_v9/iter2) has 7 independent draws: `eval_out/az_v9_selfplay/iter2_run{1..7}`
  (runs 1-3 = selection draws, biased high as a set; runs 4-7 = unbiased confirmation draws).

## Inventory (pre-manifest era)

| dir (`artifacts/replay/mcts/`) | generator policy | operator | records | quality notes |
|---|---|---|---|---|
| `anchor_noisy/` | human prior | 1-ply robust (legacy, pre-fullsheet-era configs) | 7,400 | historical; obsolete configs |
| `az_iter1/`, `az_v3_iter1/` | anchor_noisy model | full-depth KR-UCT trees (n_sims 48/120), terminal leaves | 2,960 each | failed early AZ attempts; root-pool roots |
| `az_v4_iter1/` (= az_v5's data via symlink) | exp_021 best | 1-ply EZ (exp_017 recipe), human root pool | 6,400 | fine; exp_021-generation |
| `az_v6_2ply_unfrozen_iter1/` | exp_021 best | 2-ply KR-UCT, value leaves (frozen csas value!), root pool | 6,400 | leaves used FROZEN value model |
| `az_v6b_iter2_iter1/`, `az_v7_3ply_from_v6_iter1/` | az_v6/iter1 | 2-ply / 3-ply value-leaf trees (frozen value), root pool | 6,400 each | 3-ply was sim-starved |
| `az_v8_ratchet_iter{1..3}/` | exp_021 best (never promoted) | 2-ply sims=120 k_widen=1.5, value leaves (frozen), root pool | 6,400 each | pre-self-play |
| `az_v9_selfplay_iter{1,2}/` | exp_021 best | **self-play**, 2-ply tree, INCUMBENT value leaves | 6,400 each | **trained the champion** |
| `az_v9_selfplay_iter{3..6}/` | az_v9/iter2 (champion) | same | 6,400 each | champion-generation; fresh-window probe used it5+it6 |
| `az_v10_terminal_iter{1,2}/` | champion | self-play, flat dense terminal-MC (k_ego=4), value-free | 5,120 each | has `.diag.json` margins; loose noise floor |
| `az_v11_tree2term_iter{1,2}/` | champion | self-play, dense-root KR-UCT d=2 + terminal leaves, sims=96 | 2,560 each | **noise-starved (~2 evals/candidate) — do NOT reuse for training** |
| `az_v12_screentree_iter{1,2}/` | champion | self-play, robust screen (k_ego=8) → d=2 tree over top-8 | 2,560 each | iter-1 predates the manifest patch (its workers loaded older code); iter-2 onward carries manifests |

Sim/value auxiliary buffers: `sim/`, `sim_none/` (consistency grounding), plus the fixed
csas_v3 value/human buffers referenced by config paths.

Accumulation dirs (`*_accum_train/val`, `*_train/val`) are SYMLINK FARMS into the per-iter
dirs above — deleting a per-iter dir silently breaks them; don't.

## az_v17_bigcorpus (2026-07-20 → in flight, g5.4xlarge)

`artifacts/replay/mcts/az_v17_bigcorpus/` — THE data-wall collection. Generator: az_v14d
(global champion) self-play; operator: sig-gated screen_tree (exp_037, t≥2 masking).
Target 5,000 games / 50,000 records / ~4,000 significant plies. Round-based shards
(`rNNNN_shardK.npz`), all with manifests + diags; usable at any checkpoint. 6 workers on
one A10G (POLICY_BATCH_CAP=96 chunking — see exp log EXP-048). Stop: touch STOP in the dir.
