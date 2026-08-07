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

from fastapi.middleware.gzip import GZipMiddleware

app = FastAPI(title="Curling Arena", version="1.0",
              description="Mixed-doubles arena: humans/agents vs the csas_world champion. "
                          "Agents: GET /api/protocol for the how-to.")
app.add_middleware(GZipMiddleware, minimum_size=1024)


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
    mode: Optional[str] = None


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


class ClaimBody(BaseModel):
    token: str = Field(min_length=6, max_length=64)


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
    m = Match.create(body.players, ends=body.ends, noise=body.noise, mode=body.mode,
                     first_hammer=body.first_hammer, seed=body.seed, labels=body.labels)
    replies = m.auto_play() if "champion" in m.data["players"].values() else []
    return {"match": m.to_dict(), "text": m.text_state(),
            "champion_opening": replies or None}


@app.get("/api/matches")
def list_matches():
    out = []
    engine.MATCH_DIR.mkdir(exist_ok=True)
    for p in sorted((q for q in engine.MATCH_DIR.glob("*.json") if ".replay" not in q.name),
                    key=lambda q: q.stat().st_mtime,
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
    return {"match": m.to_dict(), "text": m.text_state()}


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
            "match": m.to_dict(), "text": m.text_state()}


@app.post("/api/match/{mid}/undo")
def undo(mid: str):
    """Roll the current end back to before the last human/agent throw (their
    throw and any champion replies after it are discarded). Undos are recorded
    in the match data; they cannot cross a completed end."""
    m = _match(mid)
    try:
        info = m.undo_last_human()
    except ValueError as e:
        raise HTTPException(409, str(e))
    return {"undo": info, "match": m.to_dict(),
            "text": m.text_state()}


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
            "match": m.to_dict(), "text": m.text_state()}


@app.post("/api/match/{mid}/claim")
def claim_seat(mid: str, body: ClaimBody):
    """Online play: claim a human seat (per-tab token; idempotent). side=None
    means both seats are taken — the caller is a spectator."""
    m = _match(mid)
    side = m.claim_seat(body.token)
    return {"side": side, "spectator": side is None,
            "seats_taken": sorted((m.data.get("claims") or {}).keys())}


@app.get("/api/match/{mid}/replay")
def replay(mid: str):
    """Board-by-board replay: exact per-throw before/after boards (from stored
    snapshots) + display trajectories. Cached for finished matches."""
    return engine.build_replay(_match(mid))


_HEAT_CACHE: Dict[tuple, Any] = {}
_TRAJ_CACHE: Dict[tuple, Any] = {}


@app.get("/api/match/{mid}/heatmap")
def heatmap(mid: str, res: float = 0.15, end: Optional[int] = None, n: Optional[int] = None):
    """Coach heatmap: champion value if the on-turn team's next stone rested at
    each grid cell (thrower's perspective). With end+n: at that historical
    throw's pre-state (replay coaching); cached."""
    m = _match(mid)
    historical = end is not None and n is not None
    if not historical and m.data["status"] != "in_progress":
        raise HTTPException(409, "match is over (pass end+n for replay heatmaps)")
    key = (mid, end, n, round(float(res), 3))
    if historical and key in _HEAT_CACHE:
        return _HEAT_CACHE[key]
    try:
        out = engine.placement_heatmap(m, res=float(np.clip(res, 0.10, 0.40)), end=end, n=n)
    except ValueError as e:
        raise HTTPException(409, str(e))
    if historical:
        if len(_HEAT_CACHE) > 256:
            _HEAT_CACHE.clear()
        _HEAT_CACHE[key] = out
    return out


@app.get("/api/match/{mid}/throw_traj")
def one_throw_traj(mid: str, end: int, n: int):
    """One historical throw's trajectory + post board (opponent-move animation)."""
    key = (mid, end, n)
    if key in _TRAJ_CACHE:
        return _TRAJ_CACHE[key]
    try:
        out = engine.throw_traj(_match(mid), end, n)
    except ValueError as e:
        raise HTTPException(404, str(e))
    if len(_TRAJ_CACHE) > 512:
        _TRAJ_CACHE.clear()
    _TRAJ_CACHE[key] = out
    return out


class ClientLog(BaseModel):
    events: List[Dict[str, Any]] = Field(max_length=50)


@app.post("/api/client_log")
def client_log(body: ClientLog):
    """Anonymous client-side diagnostics (animation telemetry) for debugging
    device-specific rendering issues. Appended to arena/client_log.jsonl."""
    import json as _json
    with open(_HERE / "client_log.jsonl", "a") as fh:
        for ev in body.events[:50]:
            ev["server_time"] = time.time()
            fh.write(_json.dumps(ev) + "\n")
    return {"ok": True}


# --------------------------------------------------------------------------- #
# Static UI
# --------------------------------------------------------------------------- #
app.mount("/static", StaticFiles(directory=str(_HERE / "static")), name="static")


@app.get("/")
def index():
    return FileResponse(str(_HERE / "static" / "index.html"))


@app.get("/join/{mid}")
def join(mid: str):
    return FileResponse(str(_HERE / "static" / "index.html"))


@app.get("/classic")
def classic():
    return FileResponse(str(_HERE / "static" / "classic.html"))


@app.get("/manifest.webmanifest")
def manifest():
    return FileResponse(str(_HERE / "static" / "manifest.webmanifest"),
                        media_type="application/manifest+json")


@app.get("/sw.js")
def service_worker():
    return FileResponse(str(_HERE / "static" / "sw.js"),
                        media_type="application/javascript")


@app.get("/apple-touch-icon.png")
def apple_icon():
    return FileResponse(str(_HERE / "static" / "icons" / "icon-180.png"))
