# csas_fixed_moreMCTS

Isolated experiment package for improving curling value learning with a Gaussian SetTransformer value model, a broad human throw-policy prior, and policy-guided MCTS-style search with KR-UCT-style continuous-action sharing.

The code reads simulator, data, split, and architecture modules from `../csas_fixed`, but writes checkpoints, search targets, logs, figures, and evaluations under this directory.

## Current Active Pipeline

These are the non-stale scripts used by the current preplacement + MCTS experiments.

- `train_policy_prior.py`: trains the stochastic MDN human throw prior over `[speed, angle, spin, y0]`. Current usage includes preplaced first-shot data plus horizontal flip and team-swap augmentation.
- `train_value_search_distilled.py`: trains the Gaussian SetTransformer value model. It supports supervised warm-start training, preplaced data, synthetic terminal labels, search-target distillation, horizontal flip augmentation, and team-swap augmentation.
- `generate_mcts_iteration_targets.py`: active iterative MCTS target generator. It samples policy/global candidate throws, simulates candidates, rolls out to terminal end value using curling-rule scoring at terminal states, writes value targets, and writes weighted continuous policy targets.
- `train_policy_search_distilled.py`: distills the search-improved continuous policy targets back into the MDN policy prior after each MCTS iteration.
- `kr_uct_search.py`: active shared search/simulation utility used by `generate_mcts_iteration_targets.py`, preplacement target generation, policy-prior visualization, and evaluation helpers. It loads policy/value models, samples candidate throws, runs JAX simulator batches, evaluates post states with the value model mean, and applies kernel-smoothed candidate scores.
- `generate_preplaced_canonical_mcts_targets.py`: generates canonical preplacement/first-shot MCTS targets for diagnostics and warm-start experiments.
- `preplaced_value_data.py`: loads and canonicalizes preplacement and first-shot augmentation rows.
- `make_value_heatmaps_sheet.py`: generates current value heatmap figures over visible curling-sheet regions, including baseline-vs-MCTS and preplacement examples.
- `visualize_policy_prior_samples.py`: visualizes policy-prior/MCTS candidate samples, including global-fraction and temperature diagnostics.
- `evaluate_value_by_shot.py`: compares old vs new value-model MSE by throw/phase.
- `common.py`: shared constants, logging, split helpers, value-model data helpers, horizontal flip augmentation, and team-swap augmentation.
- `policy_dataset.py` and `policy_model.py`: policy-prior dataset construction and MDN SetTransformer policy model.

## Current Launchers

- `run_mcts_iterations_psc_augmented_wandb_v100.sbatch`: current Bridges PSC launcher for the full augmented warm-start plus iterative MCTS/distillation run on `8x v100-32`. It pins/reinstalls a compatible JAX CUDA stack into `psc_pydeps`, logs to W&B, trains augmented policy/value warm starts, generates MCTS targets over 8 shards, distills policy, retrains value, and repeats until convergence or `MAX_ITERS`.
- `setup_runtime_env.sh`: shared local EC2 runtime setup for JAX/Torch CUDA library paths.
- `run_mcts_iterations_4gpu.sh`: local 4-GPU iterative MCTS launcher.
- `run_mcts_diagnostic_4gpu.sh`: local diagnostic MCTS target-generation/value-training launcher.
- `run_mcts_iteration_smoke.sh`: local one-iteration smoke test for the current MCTS target flow.

## Older/Secondary Launchers

These still exist for reproducibility but are not the main path lately.

- `run_full_4gpu.sh`: older one-shot KR-UCT search-target distillation pipeline using `generate_search_targets.py`.
- `run_resume_search_4gpu.sh`: resume/helper variant of the older one-shot search flow.
- `run_smoke.sh`: older smoke test for `generate_search_targets.py`.
- `generate_search_targets.py`: older one-shot search-improved value target generator. The current iterative MCTS flow uses `generate_mcts_iteration_targets.py` instead.

## Current PSC Job

The latest submitted Bridges job uses:

```bash
sbatch run_mcts_iterations_psc_augmented_wandb_v100.sbatch
```

Important defaults:

- `HOLDOUT=0`
- `MAX_ITERS=8`
- `ROOT_CANDIDATES=96`
- `ROLLOUT_CANDIDATES=24`
- `TOP_K=16`
- `EARLY_MID_OVERSAMPLE=1.5`
- `POLICY_EPOCHS=40`
- `VALUE_EPOCHS=100`
- `SEARCH_WEIGHT=2`

Override these as Slurm environment variables when needed.
