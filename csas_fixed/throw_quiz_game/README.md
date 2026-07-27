# Curling Throw Quiz

Second web app for decision-making practice with the value model.

Run from `/mnt/data/curling2/csas_fixed`:

```bash
uvicorn throw_quiz_game.app:app --host 0.0.0.0 --port 8011
```

By default, holdout `0` uses the graph-transformer checkpoint that seeded the latest `csas_fixed_moreMCTS` runs:

`/mnt/data/curling2/csas_fixed/holdouts/0/model_graphtf_gaussian_curl_arc_reach_outgoing_plus_takeout_vertices/model.pt`

Override that checkpoint without editing code:

```bash
THROW_QUIZ_VALUE_CKPT_HOLDOUT0=/path/to/model.pt uvicorn throw_quiz_game.app:app --host 0.0.0.0 --port 8011
```

The app uses real held-out game states and generates four diverse policy-guided decisions. Search combines the horizon-specific MCTS GraphTF policy, structured draws/hits/ticks, local perturbations, and global fallback actions. Candidates are ranked by expected value over local execution noise, with KR smoothing and the mixed-doubles early-takeout restoration rule. The candidate pool defaults to `256` intended throws with `16` noisy executions each:

```bash
THROW_QUIZ_POOL_SIZE=192 THROW_QUIZ_NOISE_SAMPLES=12 uvicorn throw_quiz_game.app:app --host 0.0.0.0 --port 8011
```
