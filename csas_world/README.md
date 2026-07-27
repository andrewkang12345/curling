# csas_world

An **EfficientZero-style multi-head graph-transformer** for curling. It reuses the
canonical `csas` (csas_v3) graph-transformer encoder as a shared trunk and the JAX
curling **simulator** as the *authoritative* transition generator, and adds
MuZero/EfficientZero machinery: action-conditioned latent dynamics, latent
consistency against an EMA-target encoder, K-step unrolling, and reward/outcome
heads — all trained jointly from mixed replay.

`csas_world` depends on `csas_v3` (it imports the `csas` package and loads the
canonical prior/value checkpoints); it does not duplicate the simulator or
encoder. Set `CSAS_V3_ROOT` to relocate (default `/mnt/data/curling2/csas_v3`).

## Architecture

```
   state s (24-d)  ─┐
   cond  c (3-d)   ─┴─▶  build graph ──▶  E(s,c)  shared GraphTF trunk  ──▶  h (256-d)
                        (21 nodes + global)   (warm-started from human_prior_fullcov)
                                                          │
   ┌──────────────┬───────────────┬──────────────────────┼───────────────┬──────────────┐
   ▼              ▼               ▼                       ▼               ▼              ▼
 policy        value           reward / outcome        dynamics        decoder      EMA target
 full-cov MDN  Gaussian        Huber + 17-bin CE       G(h,a)→h'       D(h)→board   trunk E_target
 (speed,angle, V(h)=(μ,logσ²)  per-step + end-margin   (residual MLP)  (optional)   (consistency)
  spin,y0)
                                                          │
                          latent consistency:  G(E(s),a) ≈ stop-grad E_target(simulate(s,a))
```

- **K-step unroll** (`WorldModel.unroll`): `h0 = E(s0)`; `h_{k+1} = G(h_k, a_k)`;
  policy/value/reward predicted at every latent step (MuZero/EfficientZero).
- **Authoritative simulator**: every training target and MCTS rollout uses the real
  JAX physics (`csas.curling_sim_jax`), never the learned `G`. `G` only *imitates*
  the simulator's encoded next state (latent consistency) and may later prefilter
  candidates (`search.use_learned_model_prefilter`).
- **Mixed-doubles legality** (no takeout before the 4th delivered stone; own-stone
  ticks legal) is enforced on candidate post-states during collection.
- **Local execution noise**: candidate values are averaged over `noise_samples`
  noisy executions (Bowling-fitted Student-t, `configs/noise/v1_bowling.json`), so
  the distilled policy prefers robust, makeable shots (matches `csas_fixed_moreMCTS`).

## Modeling details — what goes into each model

All states are the **24-vector** = 12 stone slots × `(x, y)`, normalised by
`POS_MAX=4095` (dead/unthrown slots are sentinel `0` or `≈1`). All conditions are
the **3-vector** `c = [shot_norm, team_order, stone_block]` (`team_order==1` ⇒ the
thrower's team has the **hammer**; `is_hammer` was dropped as identical to it). All
actions are the **4-vector** `[speed, angle, spin, y0]` (release speed m/s, aim
angle rad, release spin, lateral release offset m), bounded by `csas.common`'s
realism-fixed ranges.

### Shared trunk  E(s, c)  (the graph transformer)
Builds a graph from `(s, c)`, then runs an edge-biased transformer; the global
token read-out is the 256-d latent `h`. Built from the canonical
`button_visible_plus_curl_arc_reach_with_outgoing` / `three_plus_takeout_boundary`
feature env (must match the warm-start checkpoint):

- **Nodes (22 = 21 + global):** 12 stone slots + 9 virtual landmarks (button
  centre, 3 release points across the hog line, 5 behind-house "takeout boundary"
  points) + 1 condition-carrying global token.
- **Node features (6-d):** `[x, y, team_indicator(0/1), is_live, is_landmark, extra]`.
- **Edge features (9-d = 4 + 5):** base `[dx, dy, dist, same_team]` plus 5 geometric
  scalars — *scorability* (target's unoccluded angular span from the button),
  *reach* (curl-arc feasibility of getting a stone there), *score_out* and
  *takeout_out* (outgoing-direction compatibility toward the button / for ejecting
  an opponent), and their interaction `reach·max(score_out, takeout_out)`.
- **Condition** `[shot_norm, team_order, stone_block]` is projected into the global
  token. Trunk: `hidden=256, layers=4, heads=4, dropout=0.1`. Augmentation at train
  time: horizontal flip + team-slot swap (matches the baselines).

### Heads (all read the 256-d latent `h`)
| head | input | output | trained on |
|---|---|---|---|
| **policy** | `h` | full-cov Gaussian mixture (K=16) over `[speed,angle,spin,y0]` (standardised) | human throws (behaviour cloning) + **MCTS** weighted-candidate distillation (noise-averaged value-surplus) |
| **value** `V(h)` | `h` | scalar Gaussian `(mean, logvar)` over the end-score differential `ValueDiff` | **realized** `ValueDiff` (real 2026 ends + synthetic terminal states, augmented) — *not* search values (overfits) |
| **dynamics** `G(h,a)` | `h`, action (box-normalised `[-1,1]`) | next latent `h'` (residual MLP) | **latent consistency**: SimSiam vs stop-grad `E_target(simulate(s,a))` |
| **reward / outcome** | `h` | per-step reward (Huber) + 17-bin categorical over final end-margin | terminal end score (sparse) |
| **decoder** `D(h)` (optional) | `h` | reconstructed board (24-d) + per-stone live logits | the simulator's next board (geometric grounding) |

The dynamics/consistency/decoder learn the **deterministic** simulator map; the
execution noise lives only in the policy/value-surplus targets, where it belongs.

## Modularity / ablations
Every head is gated by `config.ModelCfg` flags + `config.LossCfg` weights:

| ablation (`--ablation`) | heads on |
|---|---|
| `policy_value_only` | policy + value |
| `plus_consistency`  | + latent dynamics + EMA consistency + reward/outcome |
| `plus_decoder` / `full` | + physical next-state decoder |

Configs: `configs/base.yaml`, `configs/anchor.yaml` (tuned), `configs/anchor_noisy.yaml`
(execution-noise targets), `configs/ablations/*.yaml`.

## Layout
```
configs/            base + anchor (+_noisy) + ablation YAMLs ; noise via csas_v3 v1_bowling.json
src/world/
  _bootstrap.py     wires csas onto sys.path, sets GNN_* feature env
  config.py         typed config (model/loss/replay/train/search/horizon/paths)
  actions.py        action <-> box/z conversions (no heavy imports)
  env_bridge.py     ONLY module touching JAX: simulator transition, scoring,
                    legality, candidate generators, csas checkpoint loading
  graph_encoder.py  SharedTrunk: E(s,c)->h + policy head on any latent
  heads/            policy (full-cov MDN), value, reward/outcome, dynamics,
                    decoder, consistency (SimSiam); noise (LocalNoise) in search/
  model.py          WorldModel: trunk + heads + EMA target + K-step unroll
  losses.py         masked multi-head loss; DDP-safe WorldLossModule
  replay/           schema, buffers (npz shards + MixedReplay), episode, augment
  data/             human BC + offline value datasets (reuse csas pipelines)
  search/           candidate pool + MCTS collector (uses the simulator) + noise
  train/            trainer (4-GPU DDP), horizon_loop (curriculum + head-to-head)
  eval/             head_to_head (true alternating play), baselines, figures, policy_figs
  utils/            distributed (gloo), seed, logging
scripts/            collect / train / curriculum / eval_h2h / setup_gpu + helpers
artifacts/figures/  policy_world, policy_prior, value_heatmaps_world,
                    collision_heatmaps_world, best_decision_world
tests/              model+loss, env-bridge, schema
```

## Workflow
```bash
cd /mnt/data/curling2/csas_world
# collect noise-aware MCTS targets (GPU) + train the anchor (4-GPU DDP):
bash scripts/_retrain_noisy.sh
# compare value/policy heads + game strength vs the previous best versions:
scripts/eval_h2h.sh checkpoints/csas_world/anchor_noisy/model.pt 2,6,10 40
# figures (policy, value heatmaps, collision heatmaps, best-decision):
source scripts/setup_gpu.sh
python -m world.eval.policy_figs --policy <csas_policy.pt> --label csas_world --out-root artifacts/figures/policy_world --device cuda:0
python -m world.eval.figures --world checkpoints/csas_world/anchor_noisy/model.pt --kind all --vlim 3.0 --device cuda:0
# horizon-staged MCTS curriculum (head-to-head winrate convergence):
scripts/curriculum.sh checkpoints/csas_world/anchor_noisy/model.pt 1 10
```

## Results — is csas_world better than csas_v3?
Canonical model: **`anchor_noisy`** (execution-noise MCTS targets). Same holdout-0
split, both models measured identically (see `RESULTS.md` for the full table):

- **Value head:** csas_world MSE **2.18 / 2.08** (val/test) vs the dedicated
  `csas_v3` Gaussian value model **2.22 / 2.09** → **on par / marginally better**,
  while sharing the policy trunk and carrying all the extra heads.
- **Policy head:** csas_world human-NLL **4.19 / 4.18** vs `csas_v3`
  `human_prior_fullcov` **4.07 / 4.10** — slightly *less human-like* because it is
  search-distilled toward higher-value shots, **not** degraded.
- **Game strength (true alternating ends, both throwing orders):** csas_world
  **beats the previous-best search policy `csas_v3` `mcts_horizon/h10` at
  59–64%** winrate (h2/h6/h10) and edges the human prior.

**Verdict:** as a *value* model, csas_world ≈ `csas_v3` (slightly better MSE). As a
*policy / decision-maker* it is **better** — it outplays csas_v3's strongest
search-distilled policy head-to-head. Execution noise (`anchor_noisy` vs the
noise-free `anchor_v3`) gives a marginally stronger, more robust policy (~0.51 h2h)
at comparable NLL.

## Environment notes
- **JAX simulator on GPU:** `source scripts/setup_gpu.sh` (uses the vendored
  matching jax/jaxlib/plugin 0.8.0 tree + cuDNN ≥9.8). ~4.5× faster than CPU. The
  installed plugin prints a harmless "0.9.2 not compatible" warning and is ignored.
  Collection (JAX, GPU) and training (torch, 4-GPU **gloo** DDP — NCCL crashes in
  this image) run as **separate** processes.
- `csas` is not pip-installed; everything imports it via `PYTHONPATH=src` +
  `CSAS_V3_ROOT`.
- Train/eval forward at **batch ≤256** (the curl-arc edge features allocate
  O(batch·stones²) — larger batches OOM). `amp` is off (csas `_scale_tril`'s `exp`
  is fp32-only under autocast).
```

## Arena — the standard head-to-head interface

`arena/` is the standard way to play against csas_world models going forward:
humans through a web UI, LLM agents / scripts through a JSON + plain-text API,
all on the **authoritative stack** (default `CurlingParams()` physics via
`env_bridge`, rules scoring, early-takeout legality, v2_fullsheet execution
noise) with the champion at its deployed strength (48-candidate robust
selection, 8 noise realizations, value-head ranked).

```bash
bash arena/run.sh 8020        # web UI at :8020, agent docs at /api/protocol
```

- Full mixed-doubles matches: pre-placed ends, hammer rules (scoring team and
  blanked ends both pass hammer), power plays, extra ends.
- Four shot-input modalities (`arena/AGENTS.md`): raw params, draw-to-rest,
  contact-point + weight, and move-hit-stone-to-target / takeout — the target
  modalities are solved by a path-bank + CEM inverse solver against the real
  board, and every solve reports its achieved error plus the champion's value
  estimate of the predicted outcome.
- Matches persist in `arena/matches/*.json` with intended vs realized actions
  and per-throw champion evals — completed matches double as analysis logs.
- Env knobs: `ARENA_CKPT`, `ARENA_DEVICE`, `ARENA_CHAMPION_CANDIDATES`,
  `ARENA_CHAMPION_NOISE_SAMPLES`, `ARENA_NOISE_CFG`.
