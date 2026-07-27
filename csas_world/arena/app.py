"""Curling Arena — the standard head-to-head interface for csas_world models.

Humans play through the bundled web UI (``/``); LLM agents and scripts play the
same matches through a small JSON/plain-text API (see ``AGENTS.md`` or
``GET /api/protocol``). All outcomes go through the authoritative simulator and
the deployed champion selection — see ``arena/engine.py``.

Run (from csas_world/):  bash arena/run.sh   [port 8020]
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "src"))
_CSAS_V3_SRC = os.environ.get("CSAS_V3_SRC", "/mnt/data/curling2/csas_v3/src")
if _CSAS_V3_SRC not in sys.path:
    sys.path.insert(0, _CSAS_V3_SRC)

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import engine, solver
from .engine import Champion, Match

app = FastAPI(title="Curling Arena", version="1.0",
              description="Mixed-doubles arena: humans/agents vs the csas_world champion. "
                          "Agents: GET /api/protocol for the how-to.")


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #
class NewMatch(BaseModel):
    players: Dict[str, str] = Field(default={"A": "human", "B": "champion"},
                                    description='side -> "human" | "agent" | "champion"')
    labels: Dict[str, str] = Field(default={}, description="display names per side")
    ends: int = Field(default=8, ge=1, le=20)
    noise: bool = Field(default=True, description="realize throws under execution noise (eval protocol)")
    first_hammer: str = Field(default="random", description='"A" | "B" | "random"')
    seed: Optional[int] = None


class ShotRequest(BaseModel):
    side: Optional[str] = Field(default=None, description='"A" or "B"; defaults to side on turn')
    type: str = Field(default="params", description='"params" | "draw" | "contact" | "after_contact"')
    action: Optional[List[float]] = Field(default=None, description="params: [speed, angle, spin, y0]")
    target: Optional[List[float]] = Field(default=None, description="[along, lateral] meters")
    weight: Optional[str] = Field(default="medium", description='contact: "soft" | "medium" | "heavy"')
    stone_slot: Optional[int] = Field(default=None, description="after_contact: slot of stone to move")
    remove: bool = Field(default=False, description="after_contact: take the stone out of play")
    seed: int = 0
    preview: bool = Field(default=False, description="solve + predict only; do not throw")
    auto_reply: bool = Field(default=True, description="let the champion answer until it is your turn again")


class PowerPlay(BaseModel):
    side: str
    wing: str = Field(description='"left" | "right"')


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _match(mid: str) -> Match:
    try:
        return Match.load(mid)
    except KeyError:
        raise HTTPException(404, f"no match {mid}")


def _solve_intent(m: Match, req: ShotRequest) -> tuple:
    x, c = m.state_c()
    body = req.model_dump()
    if req.type != "params" and req.type not in ("draw", "contact", "after_contact"):
        raise HTTPException(422, f"unknown shot type {req.type!r}")
    try:
        action, info = solver.solve(x, c, body)
    except (KeyError, TypeError) as e:
        raise HTTPException(422, f"missing/invalid field for type={req.type}: {e}")
    except ValueError as e:
        raise HTTPException(422, str(e))
    return action, info


def _preview(m: Match, action: np.ndarray) -> Dict[str, Any]:
    x, c = m.state_c()
    with engine.SIM_LOCK:
        traj = engine.throw_trajectory(x, c, action)
        post = env_post = engine.env_bridge.simulate_one(x, c, action)
        post, illegal = engine.env_bridge.apply_legality(x, env_post[None],
                                                         int(m.cur_end["throws_left"]), c)
    post = post[0]
    nc = engine.env_bridge.next_condition(c, engine.SHOTS_IN_END)
    out = {
        "intended_trajectory": traj,
        "predicted_board": engine.stones_from_state(post),
        "illegal_takeout": bool(np.asarray(illegal).reshape(-1)[0]),
    }
    try:
        ch = Champion.get()
        v = ch.value(post, nc)
        out["predicted_value_A"] = round((1.0 if int(round(nc[2])) == 0 else -1.0) * v, 4)
    except Exception:
        out["predicted_value_A"] = None
    return out


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.get("/api/health")
def health():
    return {"ok": True, "champion_ckpt": engine.DEFAULT_CKPT,
            "champion_loaded": Champion._inst is not None,
            "bank_ready": solver._BANK is not None, "time": time.time()}


@app.post("/api/warmup")
def warmup():
    t0 = time.time()
    engine.env_bridge.warm_jax()
    solver.get_bank()
    Champion.get()
    return {"ok": True, "seconds": round(time.time() - t0, 1)}


@app.get("/api/protocol", response_class=PlainTextResponse)
def protocol():
    return (_HERE / "AGENTS.md").read_text()


@app.post("/api/match")
def create_match(body: NewMatch):
    for side, kind in body.players.items():
        if side not in ("A", "B") or kind not in ("human", "agent", "champion"):
            raise HTTPException(422, f"bad players entry {side}:{kind}")
    m = Match.create(body.players, ends=body.ends, noise=body.noise,
                     first_hammer=body.first_hammer, seed=body.seed, labels=body.labels)
    replies = m.auto_play() if "champion" in m.data["players"].values() else []
    return {"match": m.to_dict(), "text": m.text_state(),
            "champion_opening": replies or None}


@app.get("/api/matches")
def list_matches():
    out = []
    engine.MATCH_DIR.mkdir(exist_ok=True)
    for p in sorted(engine.MATCH_DIR.glob("*.json"), key=lambda q: q.stat().st_mtime,
                    reverse=True)[:50]:
        try:
            m = Match.load(p.stem)
        except Exception:
            continue
        out.append({"id": m.id, "created": m.data["created"], "status": m.data["status"],
                    "players": m.data["players"], "labels": m.data["labels"],
                    "totals": m.data["totals"], "end": m.end_no,
                    "ends_scheduled": m.data["ends_scheduled"], "winner": m.data["winner"]})
    return {"matches": out}


@app.get("/api/match/{mid}")
def get_match(mid: str, history: bool = True):
    m = _match(mid)
    return {"match": m.to_dict(include_history=history), "text": m.text_state()}


@app.get("/api/match/{mid}/text", response_class=PlainTextResponse)
def get_match_text(mid: str):
    return _match(mid).text_state()


@app.post("/api/match/{mid}/powerplay")
def power_play(mid: str, body: PowerPlay):
    m = _match(mid)
    if body.wing not in ("left", "right"):
        raise HTTPException(422, 'wing must be "left" or "right"')
    try:
        m.set_power_play(body.side, body.wing)
    except ValueError as e:
        raise HTTPException(409, str(e))
    return {"match": m.to_dict(include_history=False), "text": m.text_state()}


@app.post("/api/match/{mid}/solve")
def solve_shot(mid: str, body: ShotRequest):
    m = _match(mid)
    if m.data["status"] != "in_progress":
        raise HTTPException(409, "match is over")
    action, info = _solve_intent(m, body)
    return {"intended": [round(float(v), 6) for v in action], "solver": info,
            "preview": _preview(m, action)}


@app.post("/api/match/{mid}/throw")
def throw(mid: str, body: ShotRequest):
    m = _match(mid)
    if m.data["status"] != "in_progress":
        raise HTTPException(409, "match is over")
    side = body.side or m.turn_team()
    if m.data["players"].get(side) == "champion":
        raise HTTPException(409, f"side {side} is played by the champion")
    turn = m.turn_team()
    if side != turn and m.data["players"].get(turn) == "champion":
        raise HTTPException(409, f"it is the champion's turn (Team {turn}) — POST "
                                 f"/api/match/{mid}/champion_move to let it throw "
                                 f"(you may call your power play first)")
    action, info = _solve_intent(m, body)
    if body.preview:
        return {"intended": [round(float(v), 6) for v in action], "solver": info,
                "preview": _preview(m, action)}
    try:
        result = m.apply_throw(action, side, meta={"request": {
            k: v for k, v in body.model_dump().items()
            if v is not None and k not in ("preview", "auto_reply")}, "solver": info})
    except ValueError as e:
        raise HTTPException(409, str(e))
    replies = m.auto_play() if body.auto_reply else []
    return {"intended": [round(float(v), 6) for v in action], "solver": info,
            "result": result, "replies": replies,
            "match": m.to_dict(include_history=False), "text": m.text_state()}


@app.post("/api/match/{mid}/champion_move")
def champion_move(mid: str):
    """Explicitly let the champion throw (used to resume from the power-play
    window at an end's first throw, or to step through champion-vs-champion)."""
    m = _match(mid)
    try:
        replies = m.auto_play(force_open=True)
    except ValueError as e:
        raise HTTPException(409, str(e))
    if not replies:
        raise HTTPException(409, "the champion is not on turn")
    return {"result": replies[0], "replies": replies[1:],
            "match": m.to_dict(include_history=False), "text": m.text_state()}


# --------------------------------------------------------------------------- #
# Static UI
# --------------------------------------------------------------------------- #
app.mount("/static", StaticFiles(directory=str(_HERE / "static")), name="static")


@app.get("/")
def index():
    return FileResponse(str(_HERE / "static" / "index.html"))
