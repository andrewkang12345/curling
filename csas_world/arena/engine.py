"""Arena match engine: full mixed-doubles matches on the authoritative stack.

Everything that decides an outcome goes through the exact machinery the models
were trained and evaluated with:

  * physics       : default ``CurlingParams()`` via ``world.env_bridge`` (csas_v3)
  * end scoring   : ``env_bridge.score_end`` (curling rules)
  * legality      : ``env_bridge.apply_legality`` (mixed-doubles early-takeout rule)
  * execution     : one ``LocalNoise`` (v2_fullsheet) sample per realized throw,
                    matching the ``--noisy`` eval protocol
  * champion play : ``WorldPlayer`` 1-ply robust selection (candidates x noise
                    realizations, value-head ranked)

The trajectory rollout exists only for display; the post-throw board always
comes from ``env_bridge.simulate`` so match outcomes are byte-identical to the
eval harness.

Coordinates follow csas compact meters: ``along`` (0 at the tee/button, negative
toward the hog line / guards, positive behind the tee), ``lateral`` (positive =
right when looking down-sheet from the delivering end).
"""
from __future__ import annotations

import json
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from world import env_bridge
from world.actions import ACTION_HIGH, ACTION_LOW, ACTION_NAMES, clip_raw
from world.preplaced import PREPLACED_SHOTS_IN_END, board_norm
from world.search.noise import make_noise

ARENA_DIR = Path(__file__).resolve().parent
MATCH_DIR = ARENA_DIR / "matches"

SHOTS_IN_END = PREPLACED_SHOTS_IN_END  # 10 thrown stones per mixed-doubles end
NUM_STONES = env_bridge.NUM_STONES
POS_MAX = 4095.0

DEFAULT_CKPT = os.environ.get(
    "ARENA_CKPT", str(ARENA_DIR.parent / "checkpoints/csas_world/az_v14d/best.pt"))
DEFAULT_NOISE_CFG = os.environ.get("ARENA_NOISE_CFG", "configs/noise/v2_fullsheet.json")
CHAMPION_CANDIDATES = int(os.environ.get("ARENA_CHAMPION_CANDIDATES", "48"))
CHAMPION_NOISE_SAMPLES = int(os.environ.get("ARENA_CHAMPION_NOISE_SAMPLES", "8"))

# One lock serializes every JAX / torch call (FastAPI sync endpoints run in a
# threadpool; the sim + model stack is not re-entrant).
SIM_LOCK = threading.RLock()


# --------------------------------------------------------------------------- #
# Board helpers
# --------------------------------------------------------------------------- #
def stones_from_state(state_norm: np.ndarray) -> List[Dict[str, Any]]:
    """Compact-meter stone list for clients. Slots 0-5 = team A, 6-11 = team B."""
    raw = np.asarray(state_norm, dtype=np.float32).reshape(NUM_STONES, 2) * POS_MAX
    from csas.common import in_play_raw, raw_to_compact_m

    live = in_play_raw(raw)
    compact = raw_to_compact_m(raw)
    out = []
    for slot in range(NUM_STONES):
        if not live[slot]:
            continue
        out.append({
            "slot": slot,
            "team": "A" if slot < 6 else "B",
            "along": round(float(compact[slot, 0]), 4),
            "lateral": round(float(compact[slot, 1]), 4),
        })
    return out


def live_slots(state_norm: np.ndarray) -> np.ndarray:
    raw = np.asarray(state_norm, dtype=np.float32).reshape(NUM_STONES, 2) * POS_MAX
    from csas.common import in_play_raw

    return np.where(in_play_raw(raw))[0]


def throw_trajectory(state_norm: np.ndarray, cond: np.ndarray, action: np.ndarray,
                     max_frames: int = 240) -> Dict[str, Any]:
    """Display trajectory for one throw: frames of per-slot [along, lateral].

    Runs the same physics (default CurlingParams) with dynamic position
    recording. The FINAL board is NOT taken from here -- callers must use
    ``env_bridge.simulate`` for the authoritative post state; we only append the
    caller's authoritative final frame so the animation lands exactly on it.
    """
    import jax.numpy as jnp
    from csas.common import raw_to_compact_m
    from csas.curling_sim_jax import CurlingParams, simulate_from_params
    from csas.search import _new_slot

    raw = np.asarray(state_norm, dtype=np.float32).reshape(NUM_STONES, 2) * POS_MAX
    slots = live_slots(state_norm)
    compact = raw_to_compact_m(raw)
    prev = compact[slots] if len(slots) else np.zeros((0, 2), dtype=np.float32)
    new_slot = int(_new_slot(raw, float(cond[2])))
    with SIM_LOCK:
        traj = np.asarray(simulate_from_params(
            CurlingParams(), jnp.asarray(prev, dtype=jnp.float32),
            jnp.asarray(np.asarray(action, dtype=np.float32)), dynamic=True))

    # first contact = first frame where any pre-existing stone moved > 3 mm
    contact = None
    if len(slots) and traj.shape[0] > 1:
        moved = np.linalg.norm(traj[:, :len(slots), :] - traj[0:1, :len(slots), :], axis=-1)
        hit_t = np.where(moved.max(axis=1) > 0.003)[0]
        if len(hit_t):
            t = int(hit_t[0])
            hit_stone = int(slots[int(np.argmax(moved[t]))])
            contact = {
                "frame": t,
                "thrown_at": [round(float(traj[t, -1, 0]), 4), round(float(traj[t, -1, 1]), 4)],
                "first_stone_hit_slot": hit_stone,
            }

    stride = max(1, int(np.ceil(traj.shape[0] / max_frames)))
    sampled = traj[::stride]
    if not np.array_equal(sampled[-1], traj[-1]):
        sampled = np.concatenate([sampled, traj[-1:]], axis=0)
    frames = []
    for f in sampled:
        frame = [[None, None]] * NUM_STONES
        for i, slot in enumerate(slots):
            frame[int(slot)] = [round(float(f[i, 0]), 4), round(float(f[i, 1]), 4)]
        frame[new_slot] = [round(float(f[-1, 0]), 4), round(float(f[-1, 1]), 4)]
        frames.append(frame)
    if contact is not None:
        contact["frame"] = int(contact["frame"] // stride)
    return {"stone_slot": new_slot, "frames": frames, "contact": contact,
            "dt": 0.02 * stride}


# --------------------------------------------------------------------------- #
# Champion (the world-model player, deployed selection)
# --------------------------------------------------------------------------- #
class Champion:
    """Lazy singleton around WorldPlayer + value head readout."""

    _inst: Optional["Champion"] = None

    def __init__(self):
        import torch

        from world.eval.head_to_head import WorldPlayer

        dev = os.environ.get("ARENA_DEVICE")
        if dev is None:
            dev = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(dev)
        self.ckpt = DEFAULT_CKPT
        self.noise = make_noise(DEFAULT_NOISE_CFG, seed=int(time.time()) % 100000)
        self.player = WorldPlayer(self.ckpt, self.device, n_candidates=CHAMPION_CANDIDATES,
                                  name="champion", noise=self.noise,
                                  sel_noise_samples=CHAMPION_NOISE_SAMPLES)

    @classmethod
    def get(cls) -> "Champion":
        with SIM_LOCK:
            if cls._inst is None:
                cls._inst = Champion()
            return cls._inst

    def select(self, x: np.ndarray, c: np.ndarray, throws_left: int) -> np.ndarray:
        with SIM_LOCK:
            return np.asarray(self.player.select_intended(
                x, c, throws_left, SHOTS_IN_END, int(round(c[2]))), dtype=np.float32)

    def value(self, x: np.ndarray, c: np.ndarray) -> float:
        """Champion value-head estimate of ``x`` from the perspective of the
        thrower encoded in ``c`` (expected end score differential)."""
        with SIM_LOCK:
            return float(self.player._value_fn(x[None], c)[0])


# --------------------------------------------------------------------------- #
# Match
# --------------------------------------------------------------------------- #
def _new_end(hammer: str, mode: str) -> Dict[str, Any]:
    """Start-of-end root: non-hammer team owns the guard and throws first."""
    first_team = "B" if hammer == "A" else "A"
    guard_slot = 1 if first_team == "A" else 7
    first_block = 0 if first_team == "A" else 1
    x = board_norm(mode, guard_slot)
    c = np.asarray([0.0, 0.0, float(first_block)], dtype=np.float32)
    return {
        "mode": mode, "hammer": hammer, "first_team": first_team,
        "state": x.tolist(), "cond": c.tolist(),
        "throws_left": SHOTS_IN_END, "throws": [], "score": None,
    }


class Match:
    def __init__(self, match_id: str, data: Dict[str, Any]):
        self.id = match_id
        self.data = data

    # ------------------------------------------------------------------ #
    @classmethod
    def create(cls, players: Dict[str, str], ends: int = 8, noise: bool = True,
               first_hammer: str = "random", seed: Optional[int] = None,
               labels: Optional[Dict[str, str]] = None) -> "Match":
        mid = uuid.uuid4().hex[:12]
        seed = int(seed) if seed is not None else int.from_bytes(os.urandom(4), "little")
        rng = np.random.default_rng(seed)
        hammer = first_hammer if first_hammer in ("A", "B") else ("A" if rng.random() < 0.5 else "B")
        data = {
            "id": mid, "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "players": {"A": players.get("A", "human"), "B": players.get("B", "champion")},
            "labels": labels or {}, "ends_scheduled": int(ends), "noise": bool(noise),
            "seed": seed, "status": "in_progress", "winner": None,
            "totals": {"A": 0, "B": 0},
            "power_play_used": {"A": False, "B": False},
            "champion": {"ckpt": DEFAULT_CKPT, "n_candidates": CHAMPION_CANDIDATES,
                         "sel_noise_samples": CHAMPION_NOISE_SAMPLES,
                         "noise_cfg": DEFAULT_NOISE_CFG},
            "ends": [_new_end(hammer, "standard")],
        }
        m = cls(mid, data)
        m.save()
        return m

    # ------------------------------------------------------------------ #
    @property
    def cur_end(self) -> Dict[str, Any]:
        return self.data["ends"][-1]

    @property
    def end_no(self) -> int:
        return len(self.data["ends"])

    def turn_team(self) -> Optional[str]:
        if self.data["status"] != "in_progress":
            return None
        block = int(round(self.cur_end["cond"][2]))
        return "A" if block == 0 else "B"

    def state_c(self):
        e = self.cur_end
        return (np.asarray(e["state"], dtype=np.float32),
                np.asarray(e["cond"], dtype=np.float32))

    def _env_noise(self):
        if not self.data["noise"]:
            return None
        n = getattr(self, "_noise_obj", None)
        if n is None:
            shots = sum(len(e["throws"]) for e in self.data["ends"])
            n = make_noise(DEFAULT_NOISE_CFG, seed=self.data["seed"] * 7919 + shots)
            self._noise_obj = n
        return n

    # ------------------------------------------------------------------ #
    def set_power_play(self, team: str, wing: str) -> None:
        e = self.cur_end
        if self.data["status"] != "in_progress":
            raise ValueError("match is over")
        if e["throws"]:
            raise ValueError("power play must be chosen before the first throw of the end")
        if e["hammer"] != team:
            raise ValueError("only the team with hammer may call its power play")
        if self.data["power_play_used"][team]:
            raise ValueError("power play already used")
        if self.end_no > self.data["ends_scheduled"]:
            raise ValueError("no power play in extra ends")
        mode = "pp_left" if wing == "left" else "pp_right"
        self.data["power_play_used"][team] = True
        self.data["ends"][-1] = _new_end(e["hammer"], mode)
        self.save()

    # ------------------------------------------------------------------ #
    def apply_throw(self, intended: np.ndarray, thrower: str,
                    meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute one throw for the side on turn. Returns the throw record."""
        if self.data["status"] != "in_progress":
            raise ValueError("match is over")
        team = self.turn_team()
        if thrower != team:
            raise ValueError(f"it is {team}'s turn, not {thrower}'s")
        e = self.cur_end
        x, c = self.state_c()
        hh = int(e["throws_left"])
        intended = clip_raw(np.asarray(intended, dtype=np.float32).reshape(4))
        env_noise = self._env_noise()
        if env_noise is not None:
            realized = env_noise.sample_batch(intended[None], 1).reshape(4).astype(np.float32)
        else:
            realized = intended

        with SIM_LOCK:
            traj = throw_trajectory(x, c, realized)
            post_raw = env_bridge.simulate_one(x, c, realized)
            post, illegal = env_bridge.apply_legality(x, post_raw[None], hh, c)
        post = post[0]
        illegal = bool(np.asarray(illegal).reshape(-1)[0])
        next_c = env_bridge.next_condition(c, SHOTS_IN_END)

        rec = {
            "end": self.end_no, "n": len(e["throws"]) + 1, "team": team,
            "intended": [round(float(v), 6) for v in intended],
            "realized": [round(float(v), 6) for v in realized],
            "illegal_takeout": illegal,
            "meta": meta or {},
        }
        # champion value readout (team A perspective), if the champion is loaded
        try:
            ch = Champion.get()
            v = ch.value(post, next_c)
            sign = 1.0 if int(round(next_c[2])) == 0 else -1.0
            rec["value_A"] = round(sign * v, 4)
        except Exception:
            rec["value_A"] = None

        e["throws"].append(rec)
        e["state"] = post.astype(np.float32).tolist()
        e["cond"] = next_c.astype(np.float32).tolist()
        e["throws_left"] = hh - 1
        result: Dict[str, Any] = {"throw": rec, "trajectory": traj,
                                  "board": stones_from_state(post)}
        if illegal:
            # forfeited throw: board restored, no trajectory-final board mismatch
            result["trajectory"] = None
        if e["throws_left"] == 0:
            result["end_result"] = self._finish_end()
        self.save()
        return result

    # ------------------------------------------------------------------ #
    def _finish_end(self) -> Dict[str, Any]:
        e = self.cur_end
        x = np.asarray(e["state"], dtype=np.float32)
        with SIM_LOCK:
            sa = float(env_bridge.score_end(x, 0))  # team A perspective
        pts, scorer = (int(abs(sa)), "A" if sa > 0 else "B") if sa != 0 else (0, None)
        e["score"] = {"team": scorer, "points": pts}
        if scorer:
            self.data["totals"][scorer] += pts
        # mixed doubles hammer rule: scoring team loses hammer; a blank end also
        # passes the hammer to the other team
        old_hammer = e["hammer"]
        new_hammer = ("A" if scorer == "B" else "B") if scorer else ("A" if old_hammer == "B" else "B")
        summary = {"end": self.end_no, "score": e["score"],
                   "totals": dict(self.data["totals"])}
        ta, tb = self.data["totals"]["A"], self.data["totals"]["B"]
        done_regulation = self.end_no >= self.data["ends_scheduled"]
        if done_regulation and ta != tb:
            self.data["status"] = "finished"
            self.data["winner"] = "A" if ta > tb else "B"
            summary["match_over"] = True
            summary["winner"] = self.data["winner"]
        else:
            self.data["ends"].append(_new_end(new_hammer, "standard"))
            summary["match_over"] = False
            summary["next_hammer"] = new_hammer
            if done_regulation:
                summary["extra_end"] = self.end_no + 1
        return summary

    # ------------------------------------------------------------------ #
    def champion_move(self) -> Dict[str, Any]:
        team = self.turn_team()
        if team is None:
            raise ValueError("match is over")
        if self.data["players"][team] != "champion":
            raise ValueError(f"side {team} is not the champion")
        x, c = self.state_c()
        ch = Champion.get()
        intended = ch.select(x, c, int(self.cur_end["throws_left"]))
        return self.apply_throw(intended, team, meta={"by": "champion"})

    def power_play_hold(self) -> bool:
        """True while the champion should WAIT at the start of an end: the
        hammer side is a human/agent whose power play is still available, so
        they get the chance to call it before the end's first throw."""
        e = self.cur_end
        return (self.data["status"] == "in_progress" and not e["throws"]
                and self.data["players"].get(e["hammer"]) != "champion"
                and not self.data["power_play_used"][e["hammer"]]
                and self.end_no <= self.data["ends_scheduled"])

    def auto_play(self, max_moves: int = SHOTS_IN_END,
                  force_open: bool = False) -> List[Dict[str, Any]]:
        """Let the champion play until it is no longer on turn (or end/match
        over). Pauses at an end's first throw while ``power_play_hold`` is
        active, unless ``force_open`` (the explicit resume) is set."""
        out = []
        for _ in range(max_moves):
            team = self.turn_team()
            if team is None or self.data["players"][team] != "champion":
                break
            if self.power_play_hold() and not (force_open and not out):
                break
            out.append(self.champion_move())
            if out[-1].get("end_result"):
                break
        return out

    # ------------------------------------------------------------------ #
    def to_dict(self, include_history: bool = True) -> Dict[str, Any]:
        d = json.loads(json.dumps(self.data))  # deep copy of plain data
        e = d["ends"][-1]
        d["board"] = stones_from_state(np.asarray(e["state"], dtype=np.float32))
        d["turn"] = {
            "team": self.turn_team(),
            "player": self.data["players"].get(self.turn_team()) if self.turn_team() else None,
            "end": self.end_no, "throw": len(e["throws"]) + 1,
            "throws_left": e["throws_left"], "hammer": e["hammer"], "mode": e["mode"],
        }
        if not include_history:
            for end in d["ends"]:
                end.pop("throws", None)
                end.pop("state", None)
        return d

    def text_state(self) -> str:
        """Compact plain-text state for LLM agents."""
        d = self.data
        e = self.cur_end
        team = self.turn_team()
        lines = []
        la = d["labels"].get("A") or d["players"]["A"]
        lb = d["labels"].get("B") or d["players"]["B"]
        lines.append(f"MIXED DOUBLES CURLING — match {d['id']}  [{d['status']}]")
        lines.append(f"Team A = {la}   Team B = {lb}   (execution noise: {'ON' if d['noise'] else 'OFF'})")
        lines.append(f"Score  A {d['totals']['A']} : {d['totals']['B']} B   "
                     f"End {self.end_no} of {d['ends_scheduled']}   Hammer (throws last): {e['hammer']}   "
                     f"Setup: {e['mode']}")
        if d["status"] == "finished":
            lines.append(f"MATCH OVER — winner: Team {d['winner']}")
            return "\n".join(lines)
        lines.append(f"Throw {len(e['throws']) + 1} of {SHOTS_IN_END} this end — Team {team} to throw "
                     f"({e['throws_left']} throws left incl. this one).")
        if self.power_play_hold() and self.data["players"].get(team) == "champion":
            lines.append(f"WAITING: the champion throws first this end but Team {e['hammer']} "
                         f"(you) may still call the power play. Either POST "
                         f"/api/match/{{id}}/powerplay now, or POST /api/match/{{id}}/champion_move "
                         f"to let the champion throw.")
        lines.append("")
        lines.append("Stones in play (compact meters; along: 0 = button/tee center, negative = in front of "
                     "the house toward the guards, positive = behind the tee; lateral: positive = right):")
        board = stones_from_state(np.asarray(e["state"], dtype=np.float32))
        if not board:
            lines.append("  (none)")
        for s in board:
            r = float(np.hypot(s["along"], s["lateral"]))
            if r <= 1.829 + 0.145:
                where = f"{r:.2f} m from button"
            elif s["along"] < -1.974:
                where = "guard zone"
            else:
                where = "near the house"
            lines.append(f"  {s['team']}{s['slot'] % 6 + 1} (slot {s['slot']}): "
                         f"along {s['along']:+.2f}, lateral {s['lateral']:+.2f}   [{where}]")
        lines.append("")
        lines.append("Geometry: house radius 1.829 m around (0,0); button radius 0.152 m; stone radius 0.145 m. "
                     "Only stones touching the house score. A stone is REMOVED from play if it finishes past "
                     "the back line (along > +1.97) or touches a side board (|lateral| > 2.23) — real takeout "
                     "rules. Front guards are legal and matter.")
        if int(e["throws_left"]) >= 8:
            lines.append("RULE ACTIVE: no-takeout rule — a throw that removes an OPPONENT stone from play "
                         "is forfeited (the board is restored and your throw is consumed). Moving opponent "
                         "stones without removing them is legal; removing your own stone is legal.")
        lines.append("")
        lines.append("Submit a throw with POST /api/match/{id}/throw, JSON body one of:")
        lines.append('  {"side":"%s","type":"draw","target":[along,lateral]}            — stone comes to rest there' % team)
        lines.append('  {"side":"%s","type":"contact","target":[along,lateral],"weight":"soft|medium|heavy"}' % team)
        lines.append('        — thrown stone CENTER passes/collides at that point with that remaining speed')
        lines.append('  {"side":"%s","type":"after_contact","stone_slot":K,"target":[along,lateral]}' % team)
        lines.append('        — hit stone in slot K so IT ends up at target (use "remove":true instead of')
        lines.append('          target to take it out of play)')
        lines.append('  {"side":"%s","type":"params","action":[speed,angle,spin,y0]}' % team)
        lines.append(f'        — raw physics: speed {ACTION_LOW[0]:.2f}..{ACTION_HIGH[0]:.2f} m/s, aim angle '
                     f'{ACTION_LOW[1]:.4f}..{ACTION_HIGH[1]:.4f} rad, spin {ACTION_LOW[2]:.0f}..{ACTION_HIGH[2]:.0f} rad/s '
                     f'(positive curls right), release lateral offset {ACTION_LOW[3]:.2f}..{ACTION_HIGH[3]:.2f} m')
        lines.append('Add "preview":true to see the solved shot + predicted board WITHOUT throwing.')
        lines.append('GET /api/match/{id}/text refreshes this view after every throw.')
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    def save(self) -> None:
        MATCH_DIR.mkdir(exist_ok=True)
        tmp = MATCH_DIR / f".{self.id}.tmp"
        tmp.write_text(json.dumps(self.data))
        tmp.replace(MATCH_DIR / f"{self.id}.json")

    @classmethod
    def load(cls, match_id: str) -> "Match":
        p = MATCH_DIR / f"{match_id}.json"
        if not p.exists():
            raise KeyError(match_id)
        return cls(match_id, json.loads(p.read_text()))


__all__ = ["Match", "Champion", "stones_from_state", "throw_trajectory",
           "SHOTS_IN_END", "SIM_LOCK", "ACTION_LOW", "ACTION_HIGH", "ACTION_NAMES"]
