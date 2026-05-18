# Curling Throw Quiz

Second web app for decision-making practice with the Gaussian SetTransformer value model.

Run from `/mnt/data/curling2/csas_fixed`:

```bash
uvicorn throw_quiz_game.app:app --host 0.0.0.0 --port 8011
```

The app uses real held-out game states, generates three near-optimal but parameter-diverse candidate throws plus the real observed throw per scenario, and caches those candidates after first load. The candidate pool size defaults to `160`; lower it for faster iteration or increase it for deeper searches:

```bash
THROW_QUIZ_POOL_SIZE=120 uvicorn throw_quiz_game.app:app --host 0.0.0.0 --port 8011
```
