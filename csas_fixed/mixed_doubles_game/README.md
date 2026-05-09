# Mixed Doubles Curling Game

Mobile-friendly web app that:
- serves real observed mixed-doubles shots from the existing scoring data,
- initializes the shot controls from the exact inferred observed throw,
- shows the full intended trajectory preview,
- samples one noisy execution from `noise_versions/v1_bowling.json`,
- runs the existing JAX curling simulator once from the throw location,
- scores the resulting board with the existing value model, using rule-based terminal scoring for terminal states.

## Run

From `/mnt/data/curling2/csas_fixed`:

```bash
uvicorn mixed_doubles_game.app:app --host 0.0.0.0 --port 8000
```

Then open:

```text
http://<your-machine-ip>:8000
```

## Stable EC2 Deployment

Deployment templates are in [deploy/](/mnt/data/curling2/csas_fixed/mixed_doubles_game/deploy):
- `curling-game.service`
- `curling-game.nginx.conf`

This repo was configured to serve on the instance Elastic IP:

```text
http://18.217.244.11/
```

The app itself runs on `127.0.0.1:8000` behind `nginx` on port `80`.

## Notes

- By default the app loads the best leakage-free SetTransformer holdout checkpoint from `holdouts/*/model/model.pt`.
- Override the model device with:

```bash
export CURLING_GAME_DEVICE=cpu
```

or

```bash
export CURLING_GAME_DEVICE=cuda:0
```

- The current UI is intentionally simple: scenario picker, four shot controls, one-shot noisy rollout, trajectory preview, and animated playback.
