# csas_fixed_moreMCTS

Isolated experiment package for improving curling value learning with a human throw prior plus policy-guided KR-UCT-style search.

The code reads simulator, data, split, and architecture modules from `../csas_fixed` but writes all checkpoints, search targets, logs, and evaluations under this directory.

## Pieces

- `train_policy_prior.py`: trains a broad MDN human throw prior over `[speed, angle, spin, y0]` from inverse-solver estimates.
- `kr_uct_search.py`: samples hundreds of policy-biased/global throws, simulates them in JAX, evaluates post states with the Gaussian value model mean, and applies kernel-smoothed UCT scores in normalized throw-parameter space.
- `generate_search_targets.py`: distills search-improved values for train rows, with early/mid-end oversampling.
- `train_value_search_distilled.py`: trains a Gaussian SetTransformer on human labels plus search targets. Variance is trained by NLL but search uses only the mean.
- `evaluate_value_by_shot.py`: compares old vs distilled value-model MSE by early/mid/late phase and exact `ShotIndex`.

## Smoke test

```bash
bash run_smoke.sh
```

## Full one-fold 4-GPU run

```bash
HOLDOUT=0 bash run_full_4gpu.sh
```

Useful overrides:

```bash
HOLDOUT=0 CANDIDATES=512 ROLLOUT_DEPTH=2 CHILD_CANDIDATES=128 bash run_full_4gpu.sh
```

