# csas_world — results

EfficientZero-style multi-head graph-transformer (shared GraphTF trunk + policy /
value / latent-dynamics / reward-outcome / decoder heads, EMA-target latent
consistency, K-step unroll, the curling **simulator** as the authoritative
transition). Trained 4-GPU DDP on mixed replay (human BC / offline value /
sim-transition / MCTS). The canonical checkpoint is **`anchor_noisy`** — its MCTS
distillation targets score each candidate as the mean value over **8 local
execution-noise samples** (Bowling Student-t, `v1_bowling.json`), so the policy
prefers robust, makeable shots.

## Value & policy heads vs the previous best (holdout-0, measured identically)

| metric (val / test)        | csas_world `anchor_noisy` | previous best (`csas_v3`)     |
|----------------------------|---------------------------|-------------------------------|
| value MSE                  | **2.180 / 2.084**         | Gaussian value 2.220 / 2.090  |
| value Gaussian NLL         | 0.800 / 0.826             | (baseline logvar miscalib.)   |
| policy NLL (vs human throw)| 4.192 / 4.182             | `human_prior_fullcov` 4.070 / 4.097 |

The unified value head **matches/beats** the dedicated Gaussian value model while
sharing the policy-warm-started trunk and carrying all the extra heads. The
policy's human-NLL is slightly higher because it is search-distilled toward
higher-value shots — not degraded, as the head-to-head confirms.

## Head-to-head (true alternating ends, both throwing orders averaged)

Every matchup reports **win rate (wr)** and **average score differential (dS** =
mean signed end score, csas_world perspective**)**. `>0.5` / `>0` ⇒ csas_world stronger.

**Deterministic** (1-ply value selection; 80 ends/cell):

| opponent                       | h02            | h06            | h10            | average            |
|--------------------------------|----------------|----------------|----------------|--------------------|
| `human_prior_fullcov`          | wr .500 dS +.06| .512 +.24      | .525 −.03      | **wr .513 dS +.09**|
| **`mcts_horizon/h10`** (prev best) | .588 +.31  | **.637 +.51**  | .625 +.44      | **wr .617 dS +.42**|

**Noisy-robust** (48 candidates × 8 noisy executions → robust selection; throws
realized with execution noise; full rollout to terminal + rule scoring; 60 ends/cell):

| opponent                       | h04            | h08            | average            |
|--------------------------------|----------------|----------------|--------------------|
| `mcts_horizon/h10`             | wr .617 dS +.37| .583 +.10      | **wr .600 dS +.23**|
| `anchor_v3` (noise-free world) | .500 +.12      | .467 −.48      | **wr .483 dS −.18**|

`anchor_noisy` **beats the previous-best search-distilled policy** by ~60% / +0.2–0.4
points-per-end both deterministically and under execution noise. Against the
noise-free `anchor_v3` it is ≈ parity (slightly behind under noise) — the
noise-averaged distillation did not, at this configuration (`policy_bc=1.0`,
`policy_distill=0.5`), produce a measurably more robust policy. (±~6.5% wr sampling
noise at 60 ends.)

**Verdict (is csas_world better than csas_v3?):** as a *value* model, on par /
marginally better (2.18 vs 2.22 MSE). As a *policy / decision-maker*, **better** —
it outplays csas_v3's strongest search-distilled policy head-to-head. The
execution-noise retrain is the canonical model.

## Figures (`artifacts/figures/`, rendered on `anchor_noisy`)
- `policy_world/` (40) / `policy_prior/` (40): multi-action-sample trajectories for
  the csas_world policy and the human prior, h01–h10, with thrower-team + **hammer**
  status in each title.
- `value_heatmaps_world/`: value-Δ draw-shot heatmaps from the value head.
- `collision_heatmaps_world/`: collision-shot values — small markers around each
  struck stone's outline (bins with ≥8 samples; p75 over diverse shots), **overlaid
  on the value-Δ draw heatmap** with a **shared −3..+3 color scale**.
- `best_decision_world/`: best policy shot under 16 local-execution-noise samples.

## Reproduce
```bash
bash scripts/_retrain_noisy.sh        # GPU collect (noise=8) + 4-GPU train -> anchor_noisy
scripts/eval_h2h.sh checkpoints/csas_world/anchor_noisy/model.pt 2,6,10 40
source scripts/setup_gpu.sh
python -m world.eval.policy_figs --policy checkpoints/csas_world/anchor_noisy/policy_csas.pt --label csas_world --out-root artifacts/figures/policy_world --device cuda:0
python -m world.eval.figures --world checkpoints/csas_world/anchor_noisy/model.pt --kind all --vlim 3.0 --device cuda:0
```
Canonical checkpoint: `checkpoints/csas_world/anchor_noisy/model.pt`.
Comparison JSON: `artifacts/metrics/anchor_noisy_compare.json`, `artifacts/metrics/noisy_vs_v3_h2h.json`.
