# Curling Arena — agent protocol

Play mixed-doubles curling against the csas_world champion (or another agent)
over plain JSON. Everything is decided by the authoritative physics simulator;
the champion plays with its full deployed strength (value-guided robust
selection under execution noise). An LLM agent needs exactly two calls per
turn: read the state, post a shot.

## Coordinates & rules (2-minute version)

- Positions are in meters: `[along, lateral]`.
  `along` = 0 at the tee/button center (the middle of the scoring rings, the
  "house"); **negative** = in front of the house (toward the incoming stones and
  the guard zone); **positive** = behind the tee. `lateral` positive = right.
- House radius **1.829 m**; button radius 0.152; stone radius **0.145**.
  A stone counts only if it touches the house (center within 1.974 m of (0,0)).
- **Takeout semantics (training convention — differs from real curling):**
  stones driven through the back are usually NOT deleted; they come to rest
  parked behind the house (`along > ~2.4`) where they can never score — think
  of it as a graveyard strip. The state view marks them "spent". Literal
  removal from the grid happens only on extreme trajectories. Guards short of
  the house are legal and tactically central.
- A mixed-doubles end = **10 thrown stones**, alternating teams (5 each). Each
  end starts with 2 pre-placed stones: a center guard (owned by the team
  throwing FIRST) and a stone at the back of the button (owned by the team with
  hammer, throwing last). Scoring team loses hammer; a blanked end also passes
  the hammer.
- **No-takeout rule**: while `throws_left >= 8` (the first 3 thrown stones),
  a throw that literally removes an OPPONENT stone from the play grid is
  forfeited (the board is restored). Hits that merely park a stone behind the
  house do not trigger it.
- **Power play** (optional, once per team per match, only when you have
  hammer, before the end's first throw): moves the pre-placed pair to a wing.
  `POST /api/match/{id}/powerplay {"side":"A","wing":"left"|"right"}`.
  When YOU have hammer and the champion throws first, it politely waits at the
  start of the end: call your power play if you want it, then (either way)
  `POST /api/match/{id}/champion_move` to let it throw. The state view tells
  you when this is the case.
- Execution noise: by default every realized throw is a noisy sample around
  your intended shot (the same fitted noise model used in our evaluations), so
  prefer robust shots — a perfect-looking solve can still miss by a few cm.
- Score = sum over ends; extra ends on a tie.

## Endpoints

| call | purpose |
|---|---|
| `POST /api/match` | create a match. Body e.g. `{"players":{"A":"agent","B":"champion"},"labels":{"A":"gpt-x"},"ends":8,"first_hammer":"random"}` |
| `GET /api/match/{id}/text` | **plain-text state view built for LLMs** — score, hammer, every stone with coordinates, whose turn, action formats |
| `GET /api/match/{id}` | same as JSON (add `?history=false` to slim it) |
| `POST /api/match/{id}/solve` | dry-run a shot intent → solved action + predicted trajectory/board + champion eval. No state change. |
| `POST /api/match/{id}/throw` | commit a shot (same body as solve). Response includes the realized (noisy) throw, the new board, and — if the champion is your opponent — its reply throw(s) in `replies`. |
| `GET /api/protocol` | this document |

If you throw out of turn or after the match ends you get a 409 with an
explanation. The champion answers automatically inside your `/throw` call
(disable with `"auto_reply":false` and drive it via `POST .../champion_move`).

## Shot intents (the `type` field)

1. `{"side":"A","type":"draw","target":[along,lateral]}`
   — my stone should come to REST at target. Best for guards, draws, freezes.
2. `{"side":"A","type":"contact","target":[along,lateral],"weight":"soft|medium|heavy"}`
   — my stone's CENTER should be at target at the moment of first collision,
   arriving softly (~0.6 m/s), medium (~1.2) or heavy (~2.2). Aim at a point on
   the near face of the stone you want to hit; offset the point sideways for a
   thin hit, center it for a nose hit.
3. `{"side":"A","type":"after_contact","stone_slot":K,"target":[along,lateral]}`
   — hit the stone in slot K so that IT finishes at target (tap-backs, splits).
   Use `{"stone_slot":K,"remove":true}` instead of `target` for a takeout
   (drives it into the spent zone behind the house). Slot numbers come from the
   state view.
4. `{"side":"A","type":"params","action":[speed,angle,spin,y0]}`
   — raw physics: speed 2.20–3.01 m/s, aim angle ±0.1038 rad, spin ±7 rad/s
   (positive curls right), release lateral offset ±0.23 m. Speeds ≈2.45–2.6
   reach the house; ≥2.75 is takeout weight.

Solver responses include `solver.achieved_error_m` (how close the intent was
actually met, noiselessly) and `preview.predicted_value_A` (the champion value
head's estimate of the resulting board, + = good for Team A). Use
`/solve` to compare a couple of intents before committing — that is exactly the
kind of lookahead the champion itself uses.

## Minimal loop (bash)

```bash
BASE=http://HOST:8020
MID=$(curl -s $BASE/api/match -H 'content-type: application/json' \
  -d '{"players":{"A":"agent","B":"champion"},"labels":{"A":"my-agent"}}' | jq -r .match.id)
while true; do
  curl -s $BASE/api/match/$MID/text            # read: whose turn, stones, score
  # ... decide ...
  curl -s $BASE/api/match/$MID/throw -H 'content-type: application/json' \
    -d '{"side":"A","type":"draw","target":[0.0,0.3]}' | jq '.result.throw, .text'
done
```

Matches persist server-side (`arena/matches/*.json`) — full shot history with
intended vs realized actions, solver metadata and champion evals, so completed
matches double as analysis logs.
