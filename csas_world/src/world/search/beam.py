"""Depth-3 screen-beam operator + the collection d2 operator as callables (EXP-053).

Both are value-free (terminal-MC leaves) and noise-robust, per the az_v10-12
lessons. The d3 beam applies the az_v12 pattern RECURSIVELY: at each ply a
noise-robust flat screen keeps a small beam, and only survivors are expanded.
Interior nodes live on the deterministic spine (intended posts); execution
noise is averaged inside every screen. Illegal throws need no masking at
interior nodes — ``apply_legality`` substitutes forfeit semantics (restored
board, throw consumed), so an illegal candidate simply scores as a wasted
throw. At the root we avoid choosing one, matching the collection operator.

CRN (``noise.sample_batch(..., crn=True)``) pairs the execution-noise draws
across candidates within a screen, sharpening rankings at the same k_ego.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from .. import env_bridge
from .collect import _rollout_terminal_score, score_candidates_terminal, search_state
from .kr_uct_tree import mcts_search


# --------------------------------------------------------------------------- #
# d2: the certified collection operator (az_v12 screen_tree), as a chooser
# --------------------------------------------------------------------------- #
def screen_tree_choose(policy, amean_t, astd_t, amean_np, astd_np, x, c, horizon, sie,
                       cfg, rng, device, noise) -> Optional[Dict]:
    """Exactly the collection screen_tree decision (selfplay.py block), minus
    the distillation bookkeeping. Returns {action, q, diag}."""
    persp = int(round(c[2]))
    res1 = search_state(policy, amean_t, astd_t, amean_np, astd_np, None,
                        x, c, horizon, sie, persp, cfg, rng, device, noise=noise)
    cands1, q1 = res1["cands"], res1["q"]
    legal1 = q1 > -1e8
    k_surv = min(int(getattr(cfg, "screen_topk", 8)), int(legal1.sum()))
    if k_surv < 1:
        return None
    order = np.argsort(np.where(legal1, q1, -1e18))[::-1][:k_surv]
    survivors = cands1[order]

    def rollout_value_fn(s, c2, h2, rp):
        if h2 <= 0:
            return float(env_bridge.score_end(s, rp))
        return _rollout_terminal_score(policy, amean_t, astd_t, s, c2, h2, sie, rp,
                                       device, noise, cfg.rollout_temp, cfg.std_scale)

    def sample_fn(s, c2, n):
        from csas.search import _sample_actions
        return _sample_actions(policy, amean_t, astd_t, s, c2, n, device,
                               cfg.temperature, cfg.std_scale, 0.0)

    res2 = mcts_search(x, c, horizon, sie, persp,
                       sample_fn=sample_fn, rollout_value_fn=rollout_value_fn,
                       action_mean=amean_np, action_std=astd_np,
                       n_sims=int(cfg.mcts_sims),
                       k_widen=cfg.mcts_k_widen, alpha_widen=cfg.mcts_alpha_widen,
                       kernel_bw=cfg.kernel_bandwidth, uct_c=cfg.mcts_uct_c,
                       noise=noise, rng=rng,
                       max_depth=int(getattr(cfg, "mcts_max_depth", 2)),
                       max_children=k_surv, root_candidates=survivors)
    acts, q2, nvis = res2["actions"], res2["q"], res2["n"]
    visited = nvis > 0
    if not visited.any():
        return None
    w = int(np.argmax(np.where(visited, q2, -1e9)))
    return {"action": np.asarray(acts[w], np.float32),
            "q": float(q2[w]), "screen_q_top": float(q1[order[0]])}


# --------------------------------------------------------------------------- #
# d3: recursive screen-beam with minimax backup
# --------------------------------------------------------------------------- #
def _screen(policy, amean_t, astd_t, x, c, horizon, sie, root_persp, n_cands,
            k_ego, cfg, rng, device, noise, crn=True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample n candidates at (x, c), score each by k_ego noisy executions +
    terminal-MC rollouts (root perspective). Returns (cands, q, q_se)."""
    from csas.search import _sample_actions
    cands = np.asarray(_sample_actions(policy, amean_t, astd_t, x, c, int(n_cands), device,
                                       cfg.temperature, cfg.std_scale, 0.0), np.float32)
    q, _posts, _ill, q_se = score_candidates_terminal(
        policy, amean_t, astd_t, x, c, cands, horizon, sie, root_persp,
        device, rng, noise, cfg.rollout_temp, cfg.std_scale,
        value_model=None, n_search=1, k_ego=int(k_ego), return_std=True, crn=crn)
    return cands, np.asarray(q, np.float64), np.asarray(q_se, np.float64)


def screen_beam_choose(policy, amean_t, astd_t, amean_np, astd_np, x, c, horizon, sie,
                       cfg, rng, device, noise,
                       beam_root: int = 6, opp_cands: int = 16, beam_opp: int = 3,
                       my_cands: int = 12, k_ego: Optional[int] = None,
                       crn: bool = True) -> Optional[Dict]:
    """Depth-3 minimax on the deterministic spine.

    ply 1 (root, us):    stage-1 robust screen (same dense proposal as d2) -> top beam_root
    ply 2 (opponent):    per survivor, screen opp_cands replies (root persp; opponent
                         MINIMISES) -> keep beam_opp most dangerous replies
    ply 3 (us):          per reply, screen my_cands follow-ups -> our best (MAX)
    backup:              opponent picks its best reply under the ply-3 re-evaluation
                         (MIN), root picks argmax over survivors.
    Screens at plies 2-3 roll to terminal, so leaves stay value-free.
    """
    persp = int(round(c[2]))
    ke = int(k_ego if k_ego is not None else cfg.noise_samples)

    # ---- ply 1: identical dense proposal + robust screen as d2's stage 1 ----
    res1 = search_state(policy, amean_t, astd_t, amean_np, astd_np, None,
                        x, c, horizon, sie, persp, cfg, rng, device, noise=noise)
    cands1, q1 = res1["cands"], res1["q"]
    legal1 = q1 > -1e8
    kb = min(int(beam_root), int(legal1.sum()))
    if kb < 1:
        return None
    order = np.argsort(np.where(legal1, q1, -1e18))[::-1][:kb]
    if horizon <= 1:                      # last throw: the screen IS the answer
        w = int(order[0])
        return {"action": np.asarray(cands1[w], np.float32), "q": float(q1[w]),
                "depth_used": 1, "screen_q_top": float(q1[order[0]])}

    c1 = env_bridge.next_condition(c, sie)
    n_chains = 0
    q3 = np.full(kb, -np.inf, dtype=np.float64)
    for i, ridx in enumerate(order):
        a = cands1[int(ridx)]
        post1, _ = env_bridge.apply_legality(x, env_bridge.simulate_one(x, c, a)[None], horizon, c)
        s1 = post1[0]
        # ---- ply 2: opponent replies (opponent minimises root-persp q) ----
        oc, oq, _ose = _screen(policy, amean_t, astd_t, s1, c1, horizon - 1, sie, persp,
                               opp_cands, ke, cfg, rng, device, noise, crn=crn)
        n_chains += opp_cands * ke
        if horizon - 1 <= 1:
            q3[i] = float(oq.min())       # opponent's last throw: its best reply directly
            continue
        ko = min(int(beam_opp), len(oq))
        opp_order = np.argsort(oq)[:ko]   # most dangerous first
        c2 = env_bridge.next_condition(c1, sie)
        vals = np.empty(ko, dtype=np.float64)
        for j, oidx in enumerate(opp_order):
            b = oc[int(oidx)]
            post2, _ = env_bridge.apply_legality(s1, env_bridge.simulate_one(s1, c1, b)[None],
                                                 horizon - 1, c1)
            s2 = post2[0]
            # ---- ply 3: our follow-up (we maximise) ----
            _mc, mq, _mse = _screen(policy, amean_t, astd_t, s2, c2, horizon - 2, sie, persp,
                                    my_cands, ke, cfg, rng, device, noise, crn=crn)
            n_chains += my_cands * ke
            vals[j] = float(mq.max())
        q3[i] = float(vals.min())         # opponent picks its best (worst for us)

    w = int(np.argmax(q3))
    return {"action": np.asarray(cands1[int(order[w])], np.float32), "q": float(q3[w]),
            "depth_used": 3 if horizon >= 3 else 2, "screen_q_top": float(q1[order[0]]),
            "extra_chains": int(n_chains)}


__all__ = ["screen_tree_choose", "screen_beam_choose"]
