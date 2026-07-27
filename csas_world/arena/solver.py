"""Inverse shot solvers: turn human/agent intents into simulator actions.

Modalities
----------
``params``        raw ``[speed, angle, spin, y0]`` (validated + clipped only)
``draw``          "my stone should come to REST here"
``contact``       "my stone's center should be HERE at the moment of (first)
                  collision, arriving soft/medium/heavy"
``after_contact`` "hit the stone in slot K so that IT ends up THERE (or is
                  removed from play)"

Strategy: a pre-computed shooter-only *path bank* (dense action lattice run
through the authoritative physics on an empty sheet, final rest + subsampled
path + speed-along-path) provides instant nearest-neighbour initialisation for
every modality; candidates are then re-simulated on the REAL board with
``env_bridge.simulate`` (the authoritative transition) and refined with a small
CEM loop. Every solve returns the achieved error so callers can see how close
the request was actually met.

The bank is built once (~1-2 min on CPU) and cached in ``arena/cache/``.
"""
from __future__ import annotations

import hashlib
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from world import env_bridge
from world.actions import ACTION_HIGH, ACTION_LOW, clip_raw

from .engine import NUM_STONES, POS_MAX, SIM_LOCK, live_slots

CACHE_DIR = Path(__file__).resolve().parent / "cache"

# bank lattice (speed x angle x spin x y0)
BANK_SPEEDS = 42
BANK_ANGLES = 43
BANK_SPINS = (-7.0, -3.5, 3.5, 7.0)      # near-zero spin is not a real delivery
BANK_Y0S = (-0.20, 0.0, 0.20)
BANK_PATH_STRIDE = 6                      # record every 6th macro step (0.12 s)

WEIGHT_SPEEDS = {"soft": 0.55, "medium": 1.2, "heavy": 2.2}  # m/s at contact

_BANK: Optional[Dict[str, np.ndarray]] = None
_BANK_LOCK = threading.Lock()


# --------------------------------------------------------------------------- #
# Path bank
# --------------------------------------------------------------------------- #
def _bank_actions() -> np.ndarray:
    sp = np.linspace(ACTION_LOW[0], ACTION_HIGH[0], BANK_SPEEDS)
    an = np.linspace(ACTION_LOW[1], ACTION_HIGH[1], BANK_ANGLES)
    grid = np.stack(np.meshgrid(sp, an, np.asarray(BANK_SPINS), np.asarray(BANK_Y0S),
                                indexing="ij"), axis=-1).reshape(-1, 4)
    return grid.astype(np.float32)


def _build_bank() -> Dict[str, np.ndarray]:
    import jax
    import jax.numpy as jnp
    from csas.curling_sim_jax import CurlingParams, make_initial_state, step

    p = CurlingParams()
    actions = _bank_actions()
    n_rec = p.max_steps // BANK_PATH_STRIDE

    @jax.jit
    def run(batch):
        def one(a):
            s0 = make_initial_state(p, jnp.zeros((0, 2)), a[1], a[0], a[2], a[3])

            def body(s, _):
                s1 = step(p, s)
                return s1, (s1.pos[0], jnp.linalg.norm(s1.vel[0]))

            sT, (pos, spd) = jax.lax.scan(body, s0, None, length=p.max_steps)
            return sT.pos[0], pos[::BANK_PATH_STRIDE], spd[::BANK_PATH_STRIDE]

        return jax.vmap(one)(batch)

    rests, paths, speeds = [], [], []
    B = 512
    for i in range(0, len(actions), B):
        r, pth, spd = run(jnp.asarray(actions[i:i + B]))
        rests.append(np.asarray(r))
        paths.append(np.asarray(pth, dtype=np.float16))
        speeds.append(np.asarray(spd, dtype=np.float16))
    return {
        "actions": actions,
        "rest": np.concatenate(rests).astype(np.float32),
        "path": np.concatenate(paths),          # (N, n_rec, 2) f16
        "path_speed": np.concatenate(speeds),   # (N, n_rec) f16
        "n_rec": np.asarray([n_rec]),
    }


def get_bank() -> Dict[str, np.ndarray]:
    global _BANK
    with _BANK_LOCK:
        if _BANK is not None:
            return _BANK
        key = hashlib.md5(repr((BANK_SPEEDS, BANK_ANGLES, BANK_SPINS, BANK_Y0S,
                                BANK_PATH_STRIDE)).encode()).hexdigest()[:10]
        cache = CACHE_DIR / f"path_bank_{key}.npz"
        if cache.exists():
            _BANK = dict(np.load(cache))
        else:
            with SIM_LOCK:
                _BANK = _build_bank()
            CACHE_DIR.mkdir(exist_ok=True)
            np.savez_compressed(cache, **_BANK)
        return _BANK


# --------------------------------------------------------------------------- #
# Board helpers
# --------------------------------------------------------------------------- #
def _thrown_slot(x: np.ndarray, c: np.ndarray) -> int:
    from csas.search import _new_slot

    raw = np.asarray(x, dtype=np.float32).reshape(NUM_STONES, 2) * POS_MAX
    return int(_new_slot(raw, float(c[2])))


def _final_compact(posts: np.ndarray, slot: int) -> np.ndarray:
    """Final [along, lateral] of ``slot`` per post state; NaN if out of play."""
    from csas.common import in_play_raw, raw_to_compact_m

    out = np.full((len(posts), 2), np.nan, dtype=np.float32)
    for i, post in enumerate(posts):
        raw = np.asarray(post, dtype=np.float32).reshape(NUM_STONES, 2) * POS_MAX
        if in_play_raw(raw)[slot]:
            out[i] = raw_to_compact_m(raw)[slot]
    return out


def _simulate_real(x: np.ndarray, c: np.ndarray, actions: np.ndarray) -> np.ndarray:
    with SIM_LOCK:
        return env_bridge.simulate(x, c, actions)


def _cem_refine(x: np.ndarray, c: np.ndarray, seed_action: np.ndarray, loss_fn,
                iters: int = 2, pop: int = 64, elite: int = 8,
                sigma0: Tuple[float, float, float, float] = (0.02, 0.006, 1.2, 0.04),
                rng: Optional[np.random.Generator] = None) -> Tuple[np.ndarray, float]:
    """Small CEM around ``seed_action``; loss evaluated on REAL-board posts."""
    rng = rng or np.random.default_rng(0)
    mu = np.asarray(seed_action, dtype=np.float32).copy()
    sigma = np.asarray(sigma0, dtype=np.float32).copy()
    best_a, best_l = mu.copy(), float(loss_fn(_simulate_real(x, c, mu[None]))[0])
    for _ in range(iters):
        cand = clip_raw(mu[None] + rng.standard_normal((pop, 4)).astype(np.float32) * sigma)
        cand[0] = mu  # keep the incumbent
        losses = np.asarray(loss_fn(_simulate_real(x, c, cand)), dtype=np.float64)
        order = np.argsort(losses)
        if losses[order[0]] < best_l:
            best_l = float(losses[order[0]])
            best_a = cand[order[0]].copy()
        el = cand[order[:elite]]
        mu = el.mean(axis=0)
        sigma = np.maximum(el.std(axis=0), np.asarray([1e-3, 3e-4, 0.05, 2e-3])) * 1.1
    return best_a, best_l


# --------------------------------------------------------------------------- #
# Modality solvers -- each returns (action, info)
# --------------------------------------------------------------------------- #
def solve_draw(x: np.ndarray, c: np.ndarray, target: Tuple[float, float],
               n_init: int = 96, seed: int = 0) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Thrown stone comes to rest at ``target`` [along, lateral]."""
    bank = get_bank()
    tgt = np.asarray(target, dtype=np.float32)
    slot = _thrown_slot(x, c)
    d0 = np.linalg.norm(bank["rest"] - tgt[None], axis=1)
    init = bank["actions"][np.argsort(d0)[:n_init]]
    posts = _simulate_real(x, c, init)
    finals = _final_compact(posts, slot)
    err = np.where(np.isfinite(finals).all(axis=1),
                   np.linalg.norm(np.nan_to_num(finals, nan=1e3) - tgt[None], axis=1), 1e3)
    seed_a = init[int(np.argmin(err))]

    def loss(posts_):
        f = _final_compact(posts_, slot)
        return np.where(np.isfinite(f).all(axis=1),
                        np.linalg.norm(np.nan_to_num(f, nan=1e3) - tgt[None], axis=1), 1e3)

    a, l = _cem_refine(x, c, seed_a, loss, rng=np.random.default_rng(seed))
    return a, {"achieved_error_m": round(float(l), 3), "target": [float(t) for t in tgt]}


def _bank_contact_candidates(target: np.ndarray, want_speed: Optional[float],
                             n: int, obstacles: Optional[np.ndarray] = None,
                             clearance: float = 0.30) -> np.ndarray:
    """Bank actions whose empty-sheet path passes closest to ``target`` (with a
    speed-at-passage preference when ``want_speed`` is given).

    ``obstacles`` [M,2]: other live stones. Candidates whose path would collide
    with an obstacle BEFORE reaching the target point are heavily penalised, so
    the initialisation prefers paths that curl around guards. (Deliberate raise
    paths are still reachable -- callers can mix in unfiltered candidates.)
    """
    bank = get_bank()
    path = bank["path"].astype(np.float32)           # (N, K, 2)
    N, K, _ = path.shape
    d = np.linalg.norm(path - target[None, None, :], axis=-1)  # (N, K)
    k = np.argmin(d, axis=1)
    dmin = d[np.arange(N), k]
    score = dmin.copy()
    if want_speed is not None:
        spd = bank["path_speed"].astype(np.float32)[np.arange(N), k]
        score = score + 0.35 * np.abs(spd - want_speed)
    if obstacles is not None and len(obstacles):
        obs = np.asarray(obstacles, dtype=np.float32).reshape(-1, 2)
        pre = np.arange(K)[None, :] < k[:, None]                    # (N, K)
        blocked = np.zeros(N, dtype=bool)
        B = 4096
        for i in range(0, N, B):
            dobs = np.linalg.norm(path[i:i + B, :, None, :] - obs[None, None, :, :],
                                  axis=-1).min(axis=2)              # (b, K)
            blocked[i:i + B] = np.any((dobs < clearance) & pre[i:i + B], axis=1)
        score = score + 10.0 * blocked
    return bank["actions"][np.argsort(score)[:n]]


def _other_stones(x: np.ndarray, exclude: Optional[int] = None,
                  near: Optional[np.ndarray] = None, near_r: float = 0.40) -> np.ndarray:
    """Compact positions of live stones, minus ``exclude`` slot and anything
    within ``near_r`` of ``near`` (the intended contact area)."""
    from csas.common import raw_to_compact_m

    raw = np.asarray(x, dtype=np.float32).reshape(NUM_STONES, 2) * POS_MAX
    out = []
    for slot in live_slots(x):
        if exclude is not None and int(slot) == int(exclude):
            continue
        p = raw_to_compact_m(raw)[slot]
        if near is not None and float(np.linalg.norm(p - near)) < near_r:
            continue
        out.append(p)
    return np.asarray(out, dtype=np.float32).reshape(-1, 2)


def solve_contact(x: np.ndarray, c: np.ndarray, target: Tuple[float, float],
                  weight: str = "medium", n_init: int = 48,
                  seed: int = 0) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Thrown stone's center is at ``target`` at the moment of first collision
    (or closest passage), arriving with the requested weight."""
    from .engine import throw_trajectory

    tgt = np.asarray(target, dtype=np.float32)
    want_speed = WEIGHT_SPEEDS.get(str(weight), WEIGHT_SPEEDS["medium"])
    cands = _bank_contact_candidates(tgt, want_speed, n_init,
                                     obstacles=_other_stones(x, near=tgt))

    # score on the real board via display trajectories (contact point + speed)
    def contact_err(action: np.ndarray) -> Tuple[float, Dict[str, Any]]:
        traj = throw_trajectory(x, c, action, max_frames=400)
        frames = traj["frames"]
        slot = traj["stone_slot"]
        pts = np.asarray([f[slot] for f in frames], dtype=np.float32)
        if traj["contact"] is not None:
            pos = np.asarray(traj["contact"]["thrown_at"], dtype=np.float32)
        else:  # no collision: closest passage
            pos = pts[int(np.argmin(np.linalg.norm(pts - tgt[None], axis=1)))]
        e = float(np.linalg.norm(pos - tgt))
        return e, {"contact": traj["contact"] is not None,
                   "at": [round(float(pos[0]), 3), round(float(pos[1]), 3)]}

    rng = np.random.default_rng(seed)
    best_a, best_e, best_info = None, np.inf, {}
    for a in cands[:14]:
        e, info = contact_err(a)
        if e < best_e:
            best_a, best_e, best_info = a, e, info
        if best_e < 0.03:
            break
    # local refine on the real board (small, trajectory calls are per-candidate)
    sigma = np.asarray([0.015, 0.004, 0.8, 0.03], dtype=np.float32)
    for _ in range(2):
        trial = clip_raw(best_a[None] + rng.standard_normal((6, 4)).astype(np.float32) * sigma)
        for a in trial:
            e, info = contact_err(a)
            if e < best_e:
                best_a, best_e, best_info = a, e, info
        sigma *= 0.5
    return best_a, {"achieved_error_m": round(best_e, 3), "target": [float(t) for t in tgt],
                    "weight": weight, **best_info}


def solve_after_contact(x: np.ndarray, c: np.ndarray, stone_slot: int,
                        target: Optional[Tuple[float, float]] = None,
                        remove: bool = False, n_init: int = 128,
                        seed: int = 0) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Hit the stone in ``stone_slot`` so it finishes at ``target`` -- or, with
    ``remove=True``, so it is taken out of play (and the thrown stone stays)."""
    from csas.common import raw_to_compact_m

    if stone_slot not in live_slots(x):
        raise ValueError(f"stone slot {stone_slot} is not in play")
    raw = np.asarray(x, dtype=np.float32).reshape(NUM_STONES, 2) * POS_MAX
    hit_pos = raw_to_compact_m(raw)[stone_slot]
    thrown = _thrown_slot(x, c)

    # init: bank shots whose empty-sheet path passes through the target stone,
    # spread across weights (heavier bias when removing). Mix clear-path
    # candidates (curl around guards) with unfiltered ones (deliberate raises).
    obstacles = _other_stones(x, exclude=int(stone_slot), near=hit_pos)
    speeds = [2.2, 1.5] if not remove else [2.7, 3.3]
    cands = np.concatenate(
        [_bank_contact_candidates(hit_pos, s, (3 * n_init) // 8, obstacles=obstacles)
         for s in speeds] +
        [_bank_contact_candidates(hit_pos, s, n_init // 8) for s in speeds])

    tgt = None if target is None else np.asarray(target, dtype=np.float32)

    # Real takeout rules (boundary removal ON): a removed stone simply
    # disappears from the post state, so "removed" == gone. The shaping term
    # rewards pushing the victim toward the back line when removal fails.
    BACK_LINE = 1.974

    def loss(posts_):
        f_hit = _final_compact(posts_, int(stone_slot))
        gone = ~np.isfinite(f_hit).all(axis=1)
        f_safe = np.nan_to_num(f_hit, nan=0.0)
        moved = np.linalg.norm(f_safe - hit_pos[None], axis=1)
        if remove:
            l = np.where(gone, 0.0, 1.0 + np.maximum(0.0, BACK_LINE - f_safe[:, 0]))
            l = l + np.where(~gone & (moved < 0.05), 2.0, 0.0)
            # tie-break among removals: prefer the shooter to STAY in play
            f_thr = _final_compact(posts_, thrown)
            l = l + np.where(gone & ~np.isfinite(f_thr).all(axis=1), 0.25, 0.0)
            return l
        # target mode: distance of the struck stone to its target; losing the
        # stone or never touching it are both bad
        l = np.where(gone, 8.0, np.linalg.norm(f_safe - tgt[None], axis=1))
        return l + np.where(~gone & (moved < 0.05), 2.0, 0.0)

    posts = _simulate_real(x, c, cands)
    losses = loss(posts)
    seed_a = cands[int(np.argmin(losses))]
    a, l = _cem_refine(x, c, seed_a, loss, iters=3, pop=96,
                       rng=np.random.default_rng(seed))
    final = _final_compact(_simulate_real(x, c, a[None]), int(stone_slot))[0]
    final_desc = "removed" if not np.isfinite(final).all() else \
        [round(float(final[0]), 3), round(float(final[1]), 3)]
    info: Dict[str, Any] = {"stone_slot": int(stone_slot), "stone_final": final_desc}
    if remove:
        info.update({"removed": bool(l < 0.9), "loss": round(float(l), 3)})
    else:
        info.update({"achieved_error_m": round(float(l), 3),
                     "target": [float(t) for t in tgt]})
    return a, info


# --------------------------------------------------------------------------- #
# Dispatch
# --------------------------------------------------------------------------- #
def solve(x: np.ndarray, c: np.ndarray, req: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
    kind = str(req.get("type", "params"))
    seed = int(req.get("seed", 0))
    if kind == "params":
        a = np.asarray(req["action"], dtype=np.float32).reshape(4)
        clipped = clip_raw(a)
        info = {"clipped": bool(np.any(np.abs(clipped - a) > 1e-9))}
        return clipped, info
    if kind == "draw":
        return solve_draw(x, c, req["target"], seed=seed)
    if kind == "contact":
        return solve_contact(x, c, req["target"], weight=req.get("weight", "medium"), seed=seed)
    if kind == "after_contact":
        return solve_after_contact(x, c, int(req["stone_slot"]),
                                   target=req.get("target"),
                                   remove=bool(req.get("remove", False)), seed=seed)
    raise ValueError(f"unknown shot type '{kind}'")


__all__ = ["solve", "solve_draw", "solve_contact", "solve_after_contact", "get_bank"]
