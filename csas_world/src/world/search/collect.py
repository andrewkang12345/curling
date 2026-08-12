"""MCTS target collection.

Uses the AUTHORITATIVE simulator (never the learned dynamics) to score candidate
throws, kernel-smooths value-surplus, soft-top-k's the winners into a weighted
policy-distillation target, and greedily rolls forward to harvest K-step unroll
targets (next-states for consistency, per-step search values, terminal outcome).

Run as a script to write one shard:
    JAX_PLATFORMS=cpu PYTHONPATH=src python -m world.search.collect \
        --horizon 5 --max-roots 1500 --policy <p.pt> --value <v.pt> \
        --out artifacts/replay/mcts/h05_shard0.npz --device cuda:0
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from .. import env_bridge
from ..config import Config, SearchCfg, load_config
from ..losses import _live_mask_from_state
from ..replay.episode import build_unroll_record
from ..replay.schema import SOURCE_MCTS, SOURCE_SIM, empty_record
from .candidates import generate_candidates


# --------------------------------------------------------------------------- #
# Kernel-effective visit counts (KR-UCT, Yee/Lisý/Bowling 2016)
# --------------------------------------------------------------------------- #
def _kernel_matrix(acts: np.ndarray, amean: np.ndarray, astd: np.ndarray, bw: float) -> np.ndarray:
    """Gaussian kernel K(a,b)=exp(-||z_a-z_b||^2 / 2bw^2) over z-normalised actions [M,M]."""
    z = (np.asarray(acts, np.float64) - amean[None]) / np.maximum(astd[None], 1e-4)
    d2 = ((z[:, None, :] - z[None, :, :]) ** 2).sum(axis=-1)
    return np.exp(-0.5 * d2 / max(float(bw), 1e-6) ** 2)


def kernel_effective_counts(acts: np.ndarray, n_b: np.ndarray, amean: np.ndarray,
                            astd: np.ndarray, bw: float) -> np.ndarray:
    """Effective visit count W(a) = Σ_b K(a,b) n_b (kernel density of data coverage).

    The continuous/stochastic-action analog of a discrete visit count: an action in a
    region densely (and robustly) visited by the search accrues high kernel mass."""
    K = _kernel_matrix(acts, amean, astd, bw)
    return K @ np.asarray(n_b, np.float64)


def kernel_regressed_values(acts: np.ndarray, q: np.ndarray, n_b: np.ndarray, amean: np.ndarray,
                            astd: np.ndarray, bw: float) -> np.ndarray:
    """KR value V̂(a) = Σ_b K(a,b) n_b q_b / Σ_b K(a,b) n_b (count-weighted kernel regression)."""
    K = _kernel_matrix(acts, amean, astd, bw)
    wn = K * np.asarray(n_b, np.float64)[None]
    num = (wn * np.asarray(q, np.float64)[None]).sum(axis=1)
    den = wn.sum(axis=1)
    return num / np.maximum(den, 1e-9)


def _live_np(state_norm: np.ndarray) -> np.ndarray:
    import torch

    return _live_mask_from_state(torch.from_numpy(state_norm.astype(np.float32))).numpy()


# --------------------------------------------------------------------------- #
# Roots
# --------------------------------------------------------------------------- #
@dataclass
class Root:
    x: np.ndarray
    c: np.ndarray
    shots_in_end: int
    perspective_block: int
    horizon: int


def build_roots(csas_v3_root: str, horizon: int, max_roots: int, split: str = "train",
                seed: int = 0, num_shards: int = 1, shard_id: int = 0,
                include_preplaced: bool = False) -> List[Root]:
    from ..data.human import load_human_policy_tensors

    # The first thrown stone of the end (throws_remaining==10) is NOT in the human data -- those are
    # the pre-placed start-of-end states (standard/pp_left/pp_right) the annotators skipped. Serve
    # them from the canonical pre-placed generator instead of falling back to the nearest horizon.
    from ..preplaced import PREPLACED_HORIZON
    if include_preplaced and int(horizon) >= PREPLACED_HORIZON:
        from ..preplaced import build_preplaced_roots
        return build_preplaced_roots(int(horizon), max_roots, split, seed, num_shards, shard_id)

    x, c, _a, sie, si = load_human_policy_tensors(csas_v3_root, holdout=0, split=split)
    throws_remaining = np.clip(sie - si, 1, 10).astype(np.int64)
    mask = throws_remaining == int(horizon)
    idx = np.where(mask)[0]
    if len(idx) == 0:  # fall back to nearest horizon with data
        order = np.argsort(np.abs(throws_remaining - horizon))
        idx = order[: max_roots]
    # deterministic shuffle, then shard across collectors, then cap
    rng = np.random.default_rng(seed)
    idx = rng.permutation(idx)
    if num_shards > 1:
        idx = idx[shard_id::num_shards]
    if len(idx) > max_roots:
        idx = idx[:max_roots]
    roots = []
    for i in idx:
        roots.append(Root(x=x[i].copy(), c=c[i].copy(), shots_in_end=int(round(sie[i])),
                          perspective_block=int(round(c[i, 2])), horizon=int(horizon)))
    return roots


# --------------------------------------------------------------------------- #
# Search at one state
# --------------------------------------------------------------------------- #
def _raw_q(posts: np.ndarray, horizon: int, cond: np.ndarray, perspective_block: int,
           shots_in_end: int, value_model, device, reward_model=None) -> np.ndarray:
    """Unmasked candidate value (our perspective): terminal score at h<=1, else -V(post).
    EXP-013: if ``reward_model`` is given, use -r̂₂(post) (the 2-step reward head, opponent to-move
    at post) instead of -V(post) -- post is one ply ahead, so the 2-step head already grounds to the
    rule score within ~2 plies of the end (so no separate near-terminal branch is needed)."""
    if horizon <= 1:
        return np.array([env_bridge.score_end(p, perspective_block) for p in posts], dtype=np.float64)
    nc = env_bridge.next_condition(cond, shots_in_end)
    model = reward_model if reward_model is not None else value_model
    return -env_bridge.evaluate_value(model, posts, nc, device).astype(np.float64)


def score_posts(posts: np.ndarray, illegal: np.ndarray, horizon: int, cond: np.ndarray,
                perspective_block: int, shots_in_end: int, value_model, device, reward_model=None) -> np.ndarray:
    q = _raw_q(posts, horizon, cond, perspective_block, shots_in_end, value_model, device, reward_model)
    return env_bridge.mask_illegal_scores(q, illegal)


def _two_step_rewards(states_vis, conds_vis, persps_vis, terminal_margin, root_persp,
                      value_model, device) -> np.ndarray:
    """EXP-009 2-step return from each visited state k, in state-k's to-move perspective:
    rule-based end margin if the end ends within 2 plies of k, else the value model's
    estimate 2 plies ahead (state k+2 is the same team to throw, so no extra sign flip)."""
    L = len(states_vis) - 1                       # terminal index (rollout runs to terminal)
    out = np.zeros(max(L, 0), dtype=np.float32)
    for k in range(max(L, 0)):
        if k + 2 >= L:                            # terminal within 2 plies (<=2 ahead) -> rule score
            out[k] = terminal_margin if persps_vis[k] == root_persp else -terminal_margin
        else:                                     # bootstrap from the value model 2 plies ahead
            out[k] = float(env_bridge.evaluate_value(
                value_model, states_vis[k + 2][None], conds_vis[k + 2], device)[0])
    return out


# --------------------------------------------------------------------------- #
# Value-model-FREE Monte-Carlo scoring: roll the policy to terminal, score the
# realized end by curling rules. The conds are identical across all rolled
# trajectories at a given depth (they depend only on the starting cond), so all
# trajectories step forward in lockstep with batched policy + batched simulation.
# --------------------------------------------------------------------------- #
def _legality_batch(pre: np.ndarray, posts: np.ndarray, horizon: int, cond: np.ndarray) -> np.ndarray:
    out = posts.copy()
    for i in range(len(posts)):
        corrected, _ = env_bridge.apply_legality(pre[i], posts[i][None], horizon, cond)
        out[i] = corrected[0]
    return out


def _mc_rollout_terminal_batch(policy, amean_t, astd_t, states, cond, h, sie, root_persp,
                               device, rng, noise, temp, std_scale,
                               value_model=None, n_search=1,
                               max_steps=0, leaf_value_model=None) -> np.ndarray:
    """Roll from each state to terminal; return realized end margins (root persp). With
    ``value_model`` + ``n_search>1`` (EXP-014 "search-based rollout"): at each ply the to-move
    player plays VALUE-GREEDY -- sample n_search policy candidates, and pick the one that minimises
    the opponent's value at the post-state (1-ply search, single noisy realization = noise-naive
    downstream, per the agreed design). Otherwise a single policy sample (the old MC rollout)."""
    from csas.search import _sample_actions_batch

    st = np.asarray(states, dtype=np.float32).reshape(-1, 24).copy()
    B = st.shape[0]
    cc = np.asarray(cond, dtype=np.float32).copy()
    hh = int(h)
    steps_left = int(max_steps) if (max_steps and leaf_value_model is not None) else -1
    while hh >= 1 and B > 0:
        if steps_left == 0:
            # EXP-056 truncated leaf: V(frontier) from the root's perspective (the value
            # head is conditioned on the to-move player encoded in cc)
            v = env_bridge.evaluate_value(leaf_value_model, st, cc, device).astype(np.float64)
            sign = 1.0 if int(round(cc[2])) == int(root_persp) else -1.0
            return sign * v
        steps_left -= 1
        cb = np.broadcast_to(cc, (B, 3)).astype(np.float32)
        if value_model is not None and n_search > 1:
            # The policy GraphTF builds memory-heavy curl-arc edge features.  Confirmation
            # rollouts can contain >1k trajectories, so expanding every trajectory to
            # ``n_search`` candidates in one call exceeds a 24 GiB GPU before the already-
            # chunked value evaluation below is reached.  Honour the same sharing cap as
            # the single-candidate branch and concatenate on CPU.
            import os as _os
            _cap = int(_os.environ.get("POLICY_BATCH_CAP", "0") or 0)
            if _cap > 0 and B > _cap:
                _parts = []
                for _s in range(0, B, _cap):
                    _parts.append(np.asarray(
                        _sample_actions_batch(policy, amean_t, astd_t,
                                              st[_s:_s + _cap], cb[_s:_s + _cap],
                                              n_search, device, temp, std_scale, 0.0),
                        dtype=np.float32).reshape(-1, n_search, 4))
                cands = np.concatenate(_parts, axis=0)
            else:
                cands = np.asarray(
                    _sample_actions_batch(policy, amean_t, astd_t, st, cb, n_search, device,
                                          temp, std_scale, 0.0),
                    dtype=np.float32).reshape(B, n_search, 4)
            nc = env_bridge.next_condition(cc, sie)
            # Chunk over candidates so the value-eval batch (B*n_search states) never blows the
            # GNN curl-arc edge-feature memory (an unchunked B*n_search ~= 9k states OOMs at h>=3).
            best = np.empty(B, dtype=np.int64)
            chunk = max(1, 2048 // max(int(n_search), 1))
            for s0 in range(0, B, chunk):
                e0 = min(B, s0 + chunk)
                bsz = e0 - s0
                st_rep = np.repeat(st[s0:e0], n_search, axis=0)
                cb_rep = np.repeat(cb[s0:e0], n_search, axis=0)
                posts_c = env_bridge.simulate_batched(st_rep, cb_rep, cands[s0:e0].reshape(bsz * n_search, 4))
                vals_c = env_bridge.evaluate_value(value_model, posts_c, nc, device).reshape(bsz, n_search)
                best[s0:e0] = np.argmin(vals_c, axis=1)  # to-move minimises the next-mover's value
            acts = cands[np.arange(B), best]            # (B,4)
        else:
            # POLICY_BATCH_CAP (env): chunk the batched GNN policy inference — the curl-arc
            # edge features scale ~O(B) in GPU memory and spike to ~7-10GB at B≈380, which
            # OOMs when several collection workers share one GPU (g5.4xlarge, az_v17).
            import os as _os
            _cap = int(_os.environ.get("POLICY_BATCH_CAP", "0") or 0)
            if _cap > 0 and B > _cap:
                _parts = []
                for _s in range(0, B, _cap):
                    _parts.append(np.asarray(
                        _sample_actions_batch(policy, amean_t, astd_t, st[_s:_s + _cap],
                                              cb[_s:_s + _cap], 1, device, temp, std_scale, 0.0),
                        dtype=np.float32).reshape(-1, 4))
                acts = np.concatenate(_parts, axis=0)
            else:
                acts = np.asarray(_sample_actions_batch(policy, amean_t, astd_t, st, cb, 1, device,
                                                        temp, std_scale, 0.0), dtype=np.float32).reshape(B, 4)
        if noise is not None:
            acts = noise.sample_batch(acts, 1).reshape(B, 4)
        posts = env_bridge.simulate_batched(st, cb, acts)
        st = _legality_batch(st, posts, hh, cc)
        cc = env_bridge.next_condition(cc, sie)
        hh -= 1
    return np.array([env_bridge.score_end(st[i], root_persp) for i in range(B)], dtype=np.float64)


def score_candidates_terminal(policy, amean_t, astd_t, x, c, cands, horizon, sie,
                              perspective_block, device, rng, noise, temp, std_scale,
                              value_model=None, n_search=1, k_ego=1, return_std=False,
                              crn=False, max_steps=0, leaf_value_model=None):
    """Q[i] = realized terminal end-margin (root persp) of playing candidate i then rolling to
    terminal. EXP-014: ``n_search>1`` + ``value_model`` -> value-greedy (searched) rollout; ``k_ego>1``
    + noise -> 1-ply-robust (mean over k_ego noisy executions of each candidate). Det posts returned
    for the rollout record."""
    cands = np.asarray(cands, dtype=np.float32)
    C = len(cands)
    nc = env_bridge.next_condition(c, sie)
    ke = int(k_ego) if (noise is not None and k_ego > 1) else 1
    realized = noise.sample_batch(cands, ke, crn=crn).reshape(-1, 4) if noise is not None else cands
    posts_all, illegal_all = env_bridge.apply_legality(x, env_bridge.simulate(x, c, realized), horizon, c)
    q_all = _mc_rollout_terminal_batch(policy, amean_t, astd_t, posts_all, nc, horizon - 1, sie,
                                       perspective_block, device, rng, noise, temp, std_scale,
                                       value_model=value_model, n_search=n_search,
                                       max_steps=max_steps, leaf_value_model=leaf_value_model)
    if ke > 1:
        q2 = q_all.reshape(C, ke)
        q = q2.mean(axis=1)
        illegal = illegal_all.reshape(C, ke).any(axis=1)
        det_posts, det_illegal = env_bridge.apply_legality(x, env_bridge.simulate(x, c, cands), horizon, c)
        if return_std:
            q_se = q2.std(axis=1, ddof=1) / np.sqrt(ke)   # MC standard error of each mean
            return q, det_posts, det_illegal, q_se
        return q, det_posts, det_illegal
    if return_std:
        return q_all, posts_all, illegal_all, np.full(C, np.inf, dtype=np.float64)
    return q_all, posts_all, illegal_all


def _record_rollout(policy, amean_t, astd_t, x, c, h, sie, root_persp, device, rng, noise,
                    temp, std_scale, first_action, max_steps=10 ** 9):
    """Single cheap policy rollout recording visited states. Plays ``first_action``
    from the root, then the policy (MAP-ish, 1 action/step -- NO per-step search) for
    up to ``max_steps`` plies or until the end terminates. Returns the visited states/
    conds/perspectives, the actions taken, the terminal margin (root persp; only
    meaningful if terminal reached), and whether the end actually terminated."""
    from csas.search import _sample_actions

    states = [x.copy().astype(np.float32)]
    conds = [c.copy().astype(np.float32)]
    persps = [int(round(c[2]))]
    actions = []
    st, cc, hh = x.copy().astype(np.float32), c.copy().astype(np.float32), int(h)
    a = np.asarray(first_action, dtype=np.float32)
    while hh >= 1 and len(actions) < int(max_steps):
        if noise is not None:
            a = noise.sample_batch(a[None], 1).reshape(4)
        post, _ = env_bridge.apply_legality(st, env_bridge.simulate_one(st, cc, a)[None], hh, cc)
        actions.append(a.copy())
        st = post[0]
        cc = env_bridge.next_condition(cc, sie)
        hh -= 1
        states.append(st.copy()); conds.append(cc.copy()); persps.append(int(round(cc[2])))
        if hh >= 1:
            a = _sample_actions(policy, amean_t, astd_t, st, cc, 1, device, temp, std_scale, 0.0)[0]
    reached_terminal = hh <= 0
    terminal_margin = float(env_bridge.score_end(st, root_persp))
    return states, conds, persps, actions, terminal_margin, reached_terminal


def _rollout_terminal_score(policy, amean_t, astd_t, x, c, h, sie, root_persp, device, noise,
                            temp, std_scale) -> float:
    """On-policy MC rollout to terminal; return the rule-based end margin (root perspective).
    Used as the KR-UCT tree's leaf evaluation."""
    from csas.search import _sample_actions

    st, cc, hh = np.asarray(x, np.float32).copy(), np.asarray(c, np.float32).copy(), int(h)
    while hh >= 1:
        a = _sample_actions(policy, amean_t, astd_t, st, cc, 1, device, temp, std_scale, 0.0)[0]
        if noise is not None:
            a = noise.sample_batch(a[None], 1).reshape(4)
        post, _ = env_bridge.apply_legality(st, env_bridge.simulate_one(st, cc, a)[None], hh, cc)
        st = post[0]
        cc = env_bridge.next_condition(cc, sie)
        hh -= 1
    return float(env_bridge.score_end(st, root_persp))


def search_state(policy, amean_t, astd_t, amean_np, astd_np, value_model, x, c, horizon,
                 shots_in_end, perspective_block, cfg: SearchCfg, rng, device,
                 world_model=None, noise=None, reward_model=None) -> Dict:
    cands = generate_candidates(policy, amean_t, astd_t, x, c, cfg, rng, device, world_model)
    # deterministic (intended) posts -- used as the rollout's next state
    det_posts, det_illegal = env_bridge.apply_legality(x, env_bridge.simulate(x, c, cands), horizon, c)
    q_se = None
    if getattr(cfg, "terminal_rollout_scoring", False):
        # value-model-FREE: score each candidate by rolling the policy to terminal and
        # scoring the realized end with the rules (Monte-Carlo).
        q, posts, illegal, q_se = score_candidates_terminal(
            policy, amean_t, astd_t, x, c, cands, horizon, shots_in_end, perspective_block,
            device, rng, noise, cfg.rollout_temp, cfg.std_scale,
            value_model=value_model, n_search=int(getattr(cfg, "search_rollout_n", 1)),
            k_ego=int(cfg.noise_samples), return_std=True)
    elif noise is not None and cfg.noise_samples > 0:
        # noisy decision value: average each candidate's value over N noisy executions
        ns = int(cfg.noise_samples)
        noisy = noise.sample_batch(cands, ns).reshape(-1, 4)                 # [N*ns,4]
        nposts, nillegal = env_bridge.apply_legality(x, env_bridge.simulate(x, c, noisy), horizon, c)
        q_flat = _raw_q(nposts, horizon, c, perspective_block, shots_in_end, value_model, device, reward_model)
        q2 = q_flat.reshape(len(cands), ns)
        ill2 = nillegal.reshape(len(cands), ns)
        q = q2.mean(axis=1)
        q = env_bridge.mask_illegal_scores(q, ill2.any(axis=1))
        posts, illegal = det_posts, det_illegal
    else:
        posts, illegal = det_posts, det_illegal
        q = score_posts(posts, illegal, horizon, c, perspective_block, shots_in_end, value_model, device, reward_model)
    smoothed = env_bridge.kr_smooth(cands, q, amean_np, astd_np, cfg.kernel_bandwidth, cfg.uct_c)
    smoothed = env_bridge.mask_illegal_scores(smoothed, illegal)
    top_idx, weights = env_bridge.soft_topk(smoothed, cfg.soft_topk, cfg.policy_temperature)
    value_root = float(np.sum(weights * q[top_idx]))
    best_idx = int(top_idx[0])
    return {
        "cands": cands, "posts": posts, "q": q, "smoothed": smoothed,
        "top_idx": top_idx, "weights": weights, "best_idx": best_idx,
        "best_post": posts[best_idx], "value_root": value_root, "q_se": q_se,
    }


# --------------------------------------------------------------------------- #
# Greedy rollout -> one unroll record
# --------------------------------------------------------------------------- #
def collect_root_record(root: Root, policy, amean_t, astd_t, amean_np, astd_np, value_model,
                        cfg: SearchCfg, K: int, M: int, rng, device,
                        world_model=None, noise=None, value_leaf: bool = False,
                        reward_model=None) -> Optional[Dict[str, np.ndarray]]:
    state, cond = root.x.copy(), root.c.copy()
    persp, h = root.perspective_block, root.horizon
    sie = root.shots_in_end

    if getattr(cfg, "policy_target_kernel_visits", False):
        # EXP-015/016: KR-UCT *kernel-effective visit count* W(a)=Σ_b K(a,b)n_b as the
        # policy-distillation target (the continuous/stochastic analog of an AlphaZero
        # visit target), from a depth-1 root bandit with a value-head leaf. Selection +
        # distillation use W; the value target is the kernel-regressed root value V̂_root
        # ONLY when value_target_kernel_root (EXP-016) -- otherwise the value head trains on
        # the real value buffer untouched (EXP-015, value_from_mcts=false).
        from csas.search import _sample_actions

        from .kr_uct_tree import mcts_search

        def sample_fn(s, cc, n):
            return _sample_actions(policy, amean_t, astd_t, s, cc, n, device,
                                   cfg.temperature, cfg.std_scale, 0.0)

        def rollout_value_fn(s, cc, hh, rp):     # value-head leaf (cheap, AlphaZero-style)
            if hh <= 0:
                return float(env_bridge.score_end(s, rp))
            vp = int(round(cc[2]))
            v = float(env_bridge.evaluate_value(value_model, s[None], cc, device)[0])
            return v if vp == rp else -v

        res = mcts_search(root.x, root.c, root.horizon, sie, root.perspective_block,
                          sample_fn=sample_fn, rollout_value_fn=rollout_value_fn,
                          action_mean=amean_np, action_std=astd_np, n_sims=int(cfg.mcts_sims),
                          k_widen=cfg.mcts_k_widen, alpha_widen=cfg.mcts_alpha_widen,
                          kernel_bw=cfg.kernel_bandwidth, uct_c=cfg.mcts_uct_c, noise=noise, rng=rng,
                          root_only=bool(getattr(cfg, "search_root_only", True)))
        acts, q, nvis = res["actions"], res["q"], res["n"].astype(np.float64)
        if len(acts) == 0:
            return None
        W = kernel_effective_counts(acts, nvis, amean_np, astd_np, cfg.kernel_bandwidth)
        Wsum = float(W.sum())
        # policy target ∝ W^{1/τ}: soft_topk on log W (== softmax(log W / τ))
        top_idx, weights = env_bridge.soft_topk(np.log(W + 1e-9), min(M, len(W)), cfg.policy_temperature)
        best_action = acts[int(np.argmax(W))]    # commit to the most kernel-visited action
        Vhat = kernel_regressed_values(acts, q, nvis, amean_np, astd_np, cfg.kernel_bandwidth)
        v_root = float((W * Vhat).sum() / Wsum) if Wsum > 0 else float(res["root_value"])

        states_vis, conds_vis, persps_vis, acts_taken, terminal_margin, _t = _record_rollout(
            policy, amean_t, astd_t, root.x, root.c, h, sie, root.perspective_block,
            device, rng, noise, cfg.rollout_temp, cfg.std_scale, first_action=best_action)
        base_val = v_root if bool(getattr(cfg, "value_target_kernel_root", False)) else terminal_margin
        step_values = np.array(
            [base_val if p == root.perspective_block else -base_val for p in persps_vis],
            dtype=np.float32)
        k_eff = min(K, len(acts_taken))
        return build_unroll_record(
            K, M, x0=root.x, c0=root.c,
            actions_raw=np.array(acts_taken[:k_eff], np.float32) if k_eff else np.zeros((0, 4), np.float32),
            next_states=np.array(states_vis[1:k_eff + 1], np.float32) if k_eff else np.zeros((0, 24), np.float32),
            next_conds=np.array(conds_vis[1:k_eff + 1], np.float32) if k_eff else np.zeros((0, 3), np.float32),
            value_targets=step_values, rewards=np.zeros((k_eff,), np.float32),
            outcome_margin=float(terminal_margin), source=SOURCE_MCTS, horizon=root.horizon,
            dist_actions_raw=acts[top_idx], dist_weights=weights, live_mask_fn=_live_np)

    if getattr(cfg, "use_mcts_tree", False):
        # Real multi-ply KR-UCT tree (value-model-free): policy proposes candidates,
        # leaves evaluated by on-policy MC rollout to terminal + rule scoring. The
        # search-improved policy = value-weighted soft-top-k over the root's searched
        # continuous actions; value targets = realized terminal ValueDiff (no bootstrap).
        from csas.search import _sample_actions

        from .kr_uct_tree import mcts_search

        def sample_fn(s, cc, n):
            return _sample_actions(policy, amean_t, astd_t, s, cc, n, device,
                                   cfg.temperature, cfg.std_scale, 0.0)

        if value_leaf:
            # EXP-010: AlphaZero-style closed loop — bootstrap each leaf with the (world)
            # value head instead of rolling to terminal; exact rule score at terminal leaves.
            def rollout_value_fn(s, cc, hh, rp):
                if hh <= 0:
                    return float(env_bridge.score_end(s, rp))
                vp = int(round(cc[2]))
                v = float(env_bridge.evaluate_value(value_model, s[None], cc, device)[0])
                return v if vp == rp else -v
        else:
            def rollout_value_fn(s, cc, hh, rp):
                return _rollout_terminal_score(policy, amean_t, astd_t, s, cc, hh, sie, rp,
                                               device, noise, cfg.rollout_temp, cfg.std_scale)

        res = mcts_search(root.x, root.c, root.horizon, sie, root.perspective_block,
                          sample_fn=sample_fn, rollout_value_fn=rollout_value_fn,
                          action_mean=amean_np, action_std=astd_np, n_sims=int(cfg.mcts_sims),
                          k_widen=cfg.mcts_k_widen, alpha_widen=cfg.mcts_alpha_widen,
                          kernel_bw=cfg.kernel_bandwidth, uct_c=cfg.mcts_uct_c, noise=noise, rng=rng,
                          max_depth=int(getattr(cfg, "mcts_max_depth", 0)))
        acts, q = res["actions"], res["q"]
        top_idx, weights = env_bridge.soft_topk(q, min(M, len(q)), cfg.policy_temperature)
        best_action = acts[int(np.argmax(q))]
        states_vis, conds_vis, persps_vis, acts_taken, terminal_margin, _t = _record_rollout(
            policy, amean_t, astd_t, root.x, root.c, h, sie, root.perspective_block,
            device, rng, noise, cfg.rollout_temp, cfg.std_scale, first_action=best_action)
        step_values = np.array(
            [terminal_margin if p == root.perspective_block else -terminal_margin for p in persps_vis],
            dtype=np.float32)
        k_eff = min(K, len(acts_taken))
        if getattr(cfg, "collect_step_reward", False):
            rewards_full = _two_step_rewards(states_vis, conds_vis, persps_vis, terminal_margin,
                                             root.perspective_block, value_model, device)
            rewards_k = rewards_full[:k_eff] if k_eff else np.zeros((0,), np.float32)
        else:
            rewards_k = np.zeros((k_eff,), np.float32)
        return build_unroll_record(
            K, M, x0=root.x, c0=root.c,
            actions_raw=np.array(acts_taken[:k_eff], np.float32) if k_eff else np.zeros((0, 4), np.float32),
            next_states=np.array(states_vis[1:k_eff + 1], np.float32) if k_eff else np.zeros((0, 24), np.float32),
            next_conds=np.array(conds_vis[1:k_eff + 1], np.float32) if k_eff else np.zeros((0, 3), np.float32),
            value_targets=step_values, rewards=rewards_k,
            outcome_margin=float(terminal_margin), source=SOURCE_MCTS, horizon=root.horizon,
            dist_actions_raw=acts[top_idx], dist_weights=weights, live_mask_fn=_live_np)

    if getattr(cfg, "terminal_rollout_scoring", False):
        # Efficient value-model-free path: ONE Monte-Carlo root search (each candidate
        # rolled to terminal + rule-scored) for the policy-distillation target, then a
        # single recorded policy rollout to terminal for the K-step unroll + MC value
        # targets. No value model used anywhere in collection.
        root_res = search_state(policy, amean_t, astd_t, amean_np, astd_np, value_model,
                                state, cond, h, sie, persp, cfg, rng, device, world_model, noise=noise,
                                reward_model=reward_model)
        best_action = root_res["cands"][int(root_res["best_idx"])]
        states_vis, conds_vis, persps_vis, acts_taken, terminal_margin, _term = _record_rollout(
            policy, amean_t, astd_t, root.x, root.c, h, sie, root.perspective_block,
            device, rng, noise, cfg.rollout_temp, cfg.std_scale, first_action=best_action)
        step_values = np.array(
            [terminal_margin if p == root.perspective_block else -terminal_margin
             for p in persps_vis], dtype=np.float32)
        k_eff = min(K, len(acts_taken))
        rec = build_unroll_record(
            K, M, x0=root.x, c0=root.c,
            actions_raw=np.array(acts_taken[:k_eff], np.float32) if k_eff else np.zeros((0, 4), np.float32),
            next_states=np.array(states_vis[1:k_eff + 1], np.float32) if k_eff else np.zeros((0, 24), np.float32),
            next_conds=np.array(conds_vis[1:k_eff + 1], np.float32) if k_eff else np.zeros((0, 3), np.float32),
            value_targets=step_values, rewards=np.zeros((k_eff,), np.float32),
            outcome_margin=terminal_margin, source=SOURCE_MCTS, horizon=root.horizon,
            dist_actions_raw=root_res["cands"][root_res["top_idx"]], dist_weights=root_res["weights"],
            live_mask_fn=_live_np,
        )
        return rec

    # EfficientZero n-step path: ONE full search at the root (policy-distillation target
    # + best action), then a single CHEAP policy rollout (1 action/ply, no per-step
    # search) for K=rollout_greedy_steps plies. The value target is the n-step bootstrap:
    # the value model's estimate at the rollout's end (or the realized rule-based end
    # score if the end finishes within K). rewards are 0 mid-end, gamma=1.
    root_res = search_state(policy, amean_t, astd_t, amean_np, astd_np, value_model,
                            state, cond, h, sie, persp, cfg, rng, device, world_model, noise=noise)
    best_action = root_res["cands"][int(root_res["best_idx"])]
    states_vis, conds_vis, persps_vis, acts_taken, _tm, reached_terminal = _record_rollout(
        policy, amean_t, astd_t, root.x, root.c, h, sie, root.perspective_block,
        device, rng, noise, cfg.rollout_temp, cfg.std_scale, first_action=best_action,
        max_steps=int(cfg.rollout_greedy_steps))

    # bootstrap value at the last rolled state, expressed in the ROOT team's perspective
    s_L, cond_L, persp_L = states_vis[-1], conds_vis[-1], persps_vis[-1]
    if reached_terminal:
        b_root = float(env_bridge.score_end(s_L, root.perspective_block))
    else:
        v_pL = float(env_bridge.evaluate_value(value_model, s_L[None], cond_L, device)[0])
        b_root = v_pL if persp_L == root.perspective_block else -v_pL
    step_values = np.array(
        [b_root if p == root.perspective_block else -b_root for p in persps_vis], dtype=np.float32)

    k_eff = min(K, len(acts_taken))
    rec = build_unroll_record(
        K, M, x0=root.x, c0=root.c,
        actions_raw=np.array(acts_taken[:k_eff], np.float32) if k_eff else np.zeros((0, 4), np.float32),
        next_states=np.array(states_vis[1:k_eff + 1], np.float32) if k_eff else np.zeros((0, 24), np.float32),
        next_conds=np.array(conds_vis[1:k_eff + 1], np.float32) if k_eff else np.zeros((0, 3), np.float32),
        value_targets=step_values, rewards=np.zeros((k_eff,), np.float32),
        outcome_margin=float(b_root), source=SOURCE_MCTS, horizon=root.horizon,
        dist_actions_raw=root_res["cands"][root_res["top_idx"]], dist_weights=root_res["weights"],
        live_mask_fn=_live_np,
    )
    return rec


# --------------------------------------------------------------------------- #
# Sim-transition records (consistency / decoder grounding, cheap)
# --------------------------------------------------------------------------- #
def collect_sim_records(roots: List[Root], policy, amean_t, astd_t, cfg: SearchCfg,
                        K: int, M: int, rng, device, n_per_root: int = 1) -> List[Dict[str, np.ndarray]]:
    """Random policy-sampled action sequences simulated forward (grounds G/D)."""
    from csas.search import _sample_actions

    recs = []
    for root in roots:
        for _ in range(n_per_root):
            state, cond = root.x.copy(), root.c.copy()
            h, sie = root.horizon, root.shots_in_end
            acts, nxts, ncs = [], [], []
            for k in range(K):
                if h <= 0:
                    break
                a = _sample_actions(policy, amean_t, astd_t, state, cond, 1, device,
                                    cfg.temperature, cfg.std_scale, global_frac=0.3)[0]
                post = env_bridge.simulate_one(state, cond, a)
                post, _ = env_bridge.apply_legality(state, post[None], h, cond)
                post = post[0]
                nc = env_bridge.next_condition(cond, sie)
                acts.append(a); nxts.append(post); ncs.append(nc)
                state, cond, h = post, nc, h - 1
            if not acts:
                continue
            rec = build_unroll_record(
                K, M, x0=root.x, c0=root.c,
                actions_raw=np.array(acts, np.float32), next_states=np.array(nxts, np.float32),
                next_conds=np.array(ncs, np.float32),
                value_targets=np.zeros((0,), np.float32), rewards=np.zeros((len(acts),), np.float32),
                outcome_margin=0.0, source=SOURCE_SIM, horizon=root.horizon, live_mask_fn=_live_np,
            )
            rec["value_mask"][:] = 0.0       # no value target for sim transitions
            rec["outcome_mask"] = np.float32(0.0)
            rec["reward_mask"][:] = 0.0
            recs.append(rec)
    return recs


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--horizon", type=int, required=True)
    ap.add_argument("--max-roots", type=int, default=1500)
    ap.add_argument("--policy", required=True)
    ap.add_argument("--value", required=True)
    ap.add_argument("--value-world", default=None,
                    help="WorldModel checkpoint: use its value head for 1-ply scoring + "
                         "n-step bootstrap (EfficientZero). Overrides --value.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--kind", choices=["mcts", "sim"], default="mcts")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--split", default="train")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-id", type=int, default=0)
    args = ap.parse_args()

    import torch
    from csas.search import load_policy

    cfg: Config = load_config(args.config) if args.config else Config()
    device = torch.device(args.device)
    env_bridge.warm_jax()
    policy, amean_t, astd_t = load_policy(args.policy, device)
    amean_np = amean_t.detach().cpu().numpy().astype(np.float64)
    astd_np = astd_t.detach().cpu().numpy().astype(np.float64)
    reward_model = None  # EXP-013: -r2 candidate value (built below if reward_leaf_select + value_world)
    if args.value_world:
        # use the current world model's value head (EZ bootstrap), via a thin adapter
        import torch.nn as nn
        from ..config import model_cfg_from_dict
        from ..model import WorldModel
        from ..train.trainer import load_world_checkpoint

        ck = torch.load(args.value_world, map_location=device, weights_only=False)
        _wm = WorldModel(model_cfg_from_dict(ck["model_cfg"])).to(device)
        load_world_checkpoint(_wm, args.value_world, map_location=device)
        _wm.eval()

        class _WorldValue(nn.Module):
            def __init__(self, m): super().__init__(); self.m = m
            def forward(self, x, c): return self.m.value(self.m.encode(x, c))

        value_model = _WorldValue(_wm)
        print(f"[collect] value bootstrap = world value head ({args.value_world})", flush=True)
        if getattr(cfg.search, "reward_leaf_select", False) and getattr(_wm, "reward_head", None) is not None:
            class _WorldReward(nn.Module):  # EXP-013: candidate value uses the 2-step reward head
                def __init__(self, m): super().__init__(); self.m = m
                def forward(self, x, c): return self.m.reward(self.m.encode(x, c))
            reward_model = _WorldReward(_wm)
            print(f"[collect] candidate value = -r2(post) (2-step reward head; EXP-013 reward-leaf select)", flush=True)
    else:
        value_model = env_bridge.load_csas_value(args.value, device)

    roots = build_roots(cfg.paths.csas_v3_root, args.horizon, args.max_roots, args.split,
                        args.seed, num_shards=args.num_shards, shard_id=args.shard_id,
                        include_preplaced=bool(getattr(cfg.horizon, "include_preplaced", False)))
    rng = np.random.default_rng(args.seed + 7919)
    K, M = cfg.replay.unroll_steps, cfg.search.soft_topk

    from .noise import make_noise
    noise = make_noise(cfg.search.noise_config, args.seed + 31) if cfg.search.noise_samples > 0 else None
    if noise is not None:
        print(f"[collect] local execution noise ON: {cfg.search.noise_config} x{cfg.search.noise_samples}", flush=True)

    value_leaf = bool(getattr(cfg.search, "value_leaf_bootstrap", False))
    if value_leaf:
        print(f"[collect] tree leaves bootstrapped by the value head (value_leaf_bootstrap=on)", flush=True)
    recs: List[Dict[str, np.ndarray]] = []
    if args.kind == "sim":
        recs = collect_sim_records(roots, policy, amean_t, astd_t, cfg.search, K, M, rng, device)
    else:
        for i, root in enumerate(roots):
            rec = collect_root_record(root, policy, amean_t, astd_t, amean_np, astd_np,
                                      value_model, cfg.search, K, M, rng, device, noise=noise,
                                      value_leaf=value_leaf, reward_model=reward_model)
            if rec is not None:
                recs.append(rec)
            if (i + 1) % 100 == 0:
                print(f"[collect h{args.horizon}] {i + 1}/{len(roots)} roots", flush=True)

    from ..replay.buffers import save_shard

    n = save_shard(args.out, recs)
    print(f"[collect] wrote {n} {args.kind} records -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
