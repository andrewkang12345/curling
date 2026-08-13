"""True head-to-head play between two policies.

Unlike the csas "head_to_head" script (which compares best-Q on shared roots and
never actually pits the policies against each other), this plays a full
alternating end: each player throws its own block's stones, transitions go
through the AUTHORITATIVE simulator, and the end is scored by curling rules.

Both throwing orders are evaluated (``swap`` False/True) so a checkpoint is
judged with and without hammer.  Winrate (ties = 0.5) is the convergence metric
for horizon-staged MCTS training.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .. import env_bridge


# --------------------------------------------------------------------------- #
# Players
# --------------------------------------------------------------------------- #
class Player:
    """Selects an INTENDED throw (1-ply value-guided search). Uses the simulator.

    If ``noise``/``sel_noise_samples`` are set, each candidate's value is the MEAN
    decision value over that many local-execution-noise samples (robust selection);
    otherwise it is the deterministic single-execution value.
    """

    name = "player"
    noise = None
    sel_noise_samples = 0

    def select_intended(self, x, c, horizon, shots_in_end, perspective_block):  # pragma: no cover
        raise NotImplementedError


def _decision_values(sample_fn, value_fn, x, c, horizon, shots_in_end, perspective_block,
                     n_candidates, noise=None, sel_noise_samples=0):
    """Return (candidate_actions[N,4], mean_decision_value[N]). 1-ply lookahead.

    With noise: each candidate is simulated under ``sel_noise_samples`` noisy
    executions and its value is the mean (illegal if ANY execution is illegal).
    """
    cands = sample_fn(x, c, n_candidates)
    n = len(cands)

    def _score(posts, illegal):
        if horizon <= 1 or value_fn is None:
            q = np.array([env_bridge.score_end(p, perspective_block) for p in posts], dtype=np.float64)
        else:
            nc = env_bridge.next_condition(c, shots_in_end)
            q = -value_fn(posts, nc).astype(np.float64)
        return q

    if noise is not None and int(sel_noise_samples) > 0:
        ns = int(sel_noise_samples)
        noisy = noise.sample_batch(cands, ns).reshape(-1, 4)
        posts, illegal = env_bridge.apply_legality(x, env_bridge.simulate(x, c, noisy), horizon, c)
        qf = _score(posts, illegal)
        q = qf.reshape(n, ns).mean(axis=1)
        q = env_bridge.mask_illegal_scores(q, illegal.reshape(n, ns).any(axis=1))
    else:
        posts, illegal = env_bridge.apply_legality(x, env_bridge.simulate(x, c, cands), horizon, c)
        q = env_bridge.mask_illegal_scores(_score(posts, illegal), illegal)
    return cands, q


class CsasPlayer(Player):
    def __init__(self, policy_ckpt: str, value_ckpt: str, device, n_candidates: int = 48,
                 temperature: float = 1.1, std_scale: float = 1.2, name: Optional[str] = None,
                 noise=None, sel_noise_samples: int = 0):
        from csas.search import _sample_actions, load_policy

        self.device = device
        self.policy, self.amean, self.astd = load_policy(policy_ckpt, device)
        self.value = env_bridge.load_csas_value(value_ckpt, device)
        self._sample = _sample_actions
        self.n = n_candidates
        self.temp = temperature
        self.std = std_scale
        self.name = name or "csas"
        self.noise = noise
        self.sel_noise_samples = sel_noise_samples

    def _sample_fn(self, x, c, n):
        return self._sample(self.policy, self.amean, self.astd, x, c, n, self.device,
                            self.temp, self.std, global_frac=0.0)

    def _value_fn(self, states, cond):
        return env_bridge.evaluate_value(self.value, states, cond, self.device)

    def select_intended(self, x, c, horizon, shots_in_end, perspective_block):
        cands, q = _decision_values(self._sample_fn, self._value_fn, x, c, horizon,
                                    shots_in_end, perspective_block, self.n,
                                    self.noise, self.sel_noise_samples)
        return cands[int(np.argmax(q))]


class WorldPlayer(Player):
    def __init__(self, ckpt: str, device, n_candidates: int = 48, temperature: float = 1.1,
                 std_scale: float = 1.2, name: Optional[str] = None,
                 noise=None, sel_noise_samples: int = 0,
                 value_ckpt: Optional[str] = None):
        import torch

        from ..config import ModelCfg, model_cfg_from_dict
        from ..heads.policy_head import sample_actions_z
        from ..model import WorldModel
        from ..train.trainer import load_world_checkpoint

        ck = torch.load(ckpt, map_location=device, weights_only=False)
        mcfg = model_cfg_from_dict(ck["model_cfg"]) if "model_cfg" in ck else ModelCfg()
        self.model = WorldModel(mcfg).to(device)
        load_world_checkpoint(self.model, ckpt, map_location=device)
        self.model.eval()
        self.value_model = self.model
        if value_ckpt is not None and value_ckpt != ckpt:
            vck = torch.load(value_ckpt, map_location=device, weights_only=False)
            vcfg = model_cfg_from_dict(vck["model_cfg"]) if "model_cfg" in vck else ModelCfg()
            self.value_model = WorldModel(vcfg).to(device)
            load_world_checkpoint(self.value_model, value_ckpt, map_location=device)
            self.value_model.eval()
        self.device = device
        self.n = n_candidates
        self.temp = temperature
        self.std = std_scale
        self._sample_z = sample_actions_z
        self._torch = torch
        self.name = name or "world"
        self.noise = noise
        self.sel_noise_samples = sel_noise_samples

    def _sample_fn(self, x, c, n):
        torch = self._torch
        with torch.no_grad():
            xt = torch.as_tensor(x[None], dtype=torch.float32, device=self.device)
            ct = torch.as_tensor(c[None], dtype=torch.float32, device=self.device)
            pi, mu, tril = self.model.policy(self.model.encode(xt, ct))
            z = self._sample_z(pi, mu, tril, n_samples=n, temperature=self.temp,
                               std_scale=self.std)[0]
            a = z * self.model.action_std + self.model.action_mean
        a = a.cpu().numpy().astype(np.float32)
        from ..actions import clip_raw
        return clip_raw(a)

    def sample_intended_batch(self, states, conds, n_samples: int = 1):
        """Sample policy intentions for a batch of states.

        This is the cheap behaviour-policy model used beyond an explicit search
        depth.  It deliberately does not apply the deployed value selector; use
        :meth:`select_intended_batch` for opponent nodes inside the tree.
        """
        import os

        from ..actions import clip_raw

        torch = self._torch
        states = np.asarray(states, dtype=np.float32).reshape(-1, 24)
        conds = np.asarray(conds, dtype=np.float32)
        if conds.ndim == 1:
            conds = np.broadcast_to(conds, (len(states), 3)).copy()
        else:
            conds = conds.reshape(len(states), 3)
        n_samples = max(1, int(n_samples))
        cap = max(1, int(os.environ.get("POLICY_BATCH_CAP", "0") or len(states)))
        parts = []
        with torch.no_grad():
            for start in range(0, len(states), cap):
                stop = min(len(states), start + cap)
                xt = torch.as_tensor(states[start:stop], dtype=torch.float32, device=self.device)
                ct = torch.as_tensor(conds[start:stop], dtype=torch.float32, device=self.device)
                pi, mu, tril = self.model.policy(self.model.encode(xt, ct))
                z = self._sample_z(pi, mu, tril, n_samples=n_samples,
                                   temperature=self.temp, std_scale=self.std)
                a = z * self.model.action_std + self.model.action_mean
                parts.append(a.float().cpu().numpy())
        return clip_raw(np.concatenate(parts, axis=0).astype(np.float32))

    def select_intended_batch(self, states, conds, horizon, shots_in_end,
                              perspective_block, n_actions: int = 1,
                              n_candidates: Optional[int] = None,
                              selection_noise_samples: Optional[int] = None):
        """Batched deployed 48xK selection for opponent-model search.

        Returns ``[B, n_actions, 4]``.  Each requested action gets an independent
        policy candidate set and is ranked with the same value/noise semantics as
        :meth:`select_intended`.  Multiple actions therefore form Monte-Carlo
        samples from the deployed opponent strategy, not candidates to minimise.
        """
        states = np.asarray(states, dtype=np.float32).reshape(-1, 24)
        conds = np.asarray(conds, dtype=np.float32)
        if conds.ndim == 1:
            conds = np.broadcast_to(conds, (len(states), 3)).copy()
        else:
            conds = conds.reshape(len(states), 3)
        B = len(states)
        n_actions = max(1, int(n_actions))
        n_candidates = self.n if n_candidates is None else max(1, int(n_candidates))
        rep_states = np.repeat(states, n_actions, axis=0)
        rep_conds = np.repeat(conds, n_actions, axis=0)
        R = len(rep_states)

        cands = self.sample_intended_batch(rep_states, rep_conds, n_candidates)
        requested_ns = (self.sel_noise_samples if selection_noise_samples is None
                        else selection_noise_samples)
        ns = int(requested_ns) if self.noise is not None else 0
        if ns > 0:
            executed = self.noise.sample_batch(cands, ns).reshape(R, n_candidates, ns, 4)
        else:
            ns = 1
            executed = cands[:, :, None, :]

        flat_actions = executed.reshape(-1, 4)
        repeats = n_candidates * ns
        flat_states = np.repeat(rep_states, repeats, axis=0)
        flat_conds = np.repeat(rep_conds, repeats, axis=0)
        posts = env_bridge.simulate_batched(flat_states, flat_conds, flat_actions)
        illegal = np.zeros(len(posts), dtype=bool)
        for r in range(R):
            sl = slice(r * repeats, (r + 1) * repeats)
            posts[sl], illegal[sl] = env_bridge.apply_legality(
                rep_states[r], posts[sl], int(horizon), rep_conds[r])

        if int(horizon) <= 1:
            persp = np.asarray(perspective_block)
            if persp.ndim == 0:
                persp = np.full(R, int(persp), dtype=np.int64)
            else:
                persp = np.repeat(persp.reshape(B), n_actions)
            q_flat = np.empty(len(posts), dtype=np.float64)
            for r in range(R):
                sl = slice(r * repeats, (r + 1) * repeats)
                q_flat[sl] = [env_bridge.score_end(p, int(persp[r])) for p in posts[sl]]
        else:
            next_conds = np.stack([
                env_bridge.next_condition(rep_conds[r], int(shots_in_end))
                for r in range(R)
            ])
            q_flat = -self._value_fn(posts, np.repeat(next_conds, repeats, axis=0)).astype(np.float64)

        q_exec = q_flat.reshape(R, n_candidates, ns)
        q = q_exec.mean(axis=2)
        q = np.where(illegal.reshape(R, n_candidates, ns).any(axis=2), -1.0e6, q)
        best = np.argmax(q, axis=1)
        chosen = cands[np.arange(R), best]
        return chosen.reshape(B, n_actions, 4).astype(np.float32)

    def _value_fn(self, states, cond, batch_size: Optional[int] = None):
        # chunked: the GraphTF curl-arc edge features allocate O(batch·stones²·arcs);
        # noise-expanded candidate sets (e.g. 48x8=384 posts) spike several GB unchunked
        import os

        if batch_size is None:
            batch_size = int(os.environ.get("VALUE_EVAL_BATCH", "128") or 128)
        torch = self._torch
        states = np.asarray(states, dtype=np.float32)
        c = np.broadcast_to(cond, (len(states), 3)).astype(np.float32)
        out = np.empty(len(states), dtype=np.float32)
        with torch.no_grad():
            for i in range(0, len(states), batch_size):
                xt = torch.as_tensor(states[i:i + batch_size], dtype=torch.float32, device=self.device)
                ct = torch.as_tensor(c[i:i + batch_size], dtype=torch.float32, device=self.device)
                mean = self.value_model.value_head.value(self.value_model.encode(xt, ct))
                out[i:i + batch_size] = mean.squeeze(-1).float().cpu().numpy()
        return out

    def select_intended(self, x, c, horizon, shots_in_end, perspective_block):
        cands, q = _decision_values(self._sample_fn, self._value_fn, x, c, horizon,
                                    shots_in_end, perspective_block, self.n,
                                    self.noise, self.sel_noise_samples)
        return cands[int(np.argmax(q))]


# --------------------------------------------------------------------------- #
# Game play
# --------------------------------------------------------------------------- #
@dataclass
class H2HRoot:
    x: np.ndarray
    c: np.ndarray
    shots_in_end: int
    horizon: int


def build_h2h_roots(csas_v3_root: str, horizon: int, n_roots: int, split: str = "val",
                    seed: int = 0) -> List[H2HRoot]:
    from ..data.human import load_human_policy_tensors

    x, c, _a, sie, si = load_human_policy_tensors(csas_v3_root, holdout=0, split=split)
    tr = np.clip(sie - si, 1, 10).astype(np.int64)
    idx = np.where(tr == horizon)[0]
    if len(idx) == 0:
        idx = np.argsort(np.abs(tr - horizon))[:n_roots]
    rng = np.random.default_rng(seed)
    if len(idx) > n_roots:
        idx = rng.choice(idx, size=n_roots, replace=False)
    return [H2HRoot(x[i].copy(), c[i].copy(), int(round(sie[i])), int(horizon)) for i in idx]


def play_end(player_a: Player, player_b: Player, root: H2HRoot, swap: bool,
             env_noise=None, realize_noise: bool = False) -> float:
    """Play one full end to terminal; return A's rule-based final score (A perspective).

    Each player picks an INTENDED throw via 1-ply (optionally noisy) selection. If
    ``realize_noise`` and ``env_noise`` are set, the *executed* throw is one
    local-execution-noise sample of the intended action -- so the end tests whether
    the chosen shots survive imperfect execution. The end is rolled out to the last
    stone, then scored by curling rules (``score_end``).
    """
    persp_root = int(round(root.c[2]))
    a_block = persp_root if not swap else (1 - persp_root)
    state, cond, hh = root.x.copy(), root.c.copy(), root.horizon
    while hh >= 1:
        block = int(round(cond[2]))
        is_root_block = (block == persp_root)
        plays_a = (is_root_block and not swap) or ((not is_root_block) and swap)
        player = player_a if plays_a else player_b
        intended = np.asarray(player.select_intended(state, cond, hh, root.shots_in_end, block),
                              dtype=np.float32)
        if realize_noise and env_noise is not None:
            realized = env_noise.sample_batch(intended[None], 1).reshape(4)
        else:
            realized = intended
        post, _ = env_bridge.apply_legality(state, env_bridge.simulate_one(state, cond, realized)[None],
                                            hh, cond)
        state = post[0]
        cond = env_bridge.next_condition(cond, root.shots_in_end)
        hh -= 1
    return env_bridge.score_end(state, a_block)


def _winrate(scores: np.ndarray) -> float:
    wins = (scores > 0).sum() + 0.5 * (scores == 0).sum()
    return float(wins / max(len(scores), 1))


def head_to_head(player_a: Player, player_b: Player, roots: List[H2HRoot],
                 verbose: bool = False, env_noise=None, realize_noise: bool = False) -> Dict[str, float]:
    """Returns winrate of A vs B, per throwing order and overall."""
    res: Dict[str, List[float]] = {"order0": [], "order1": []}
    for ri, root in enumerate(roots):
        res["order0"].append(play_end(player_a, player_b, root, swap=False,
                                      env_noise=env_noise, realize_noise=realize_noise))
        res["order1"].append(play_end(player_a, player_b, root, swap=True,
                                      env_noise=env_noise, realize_noise=realize_noise))
        if verbose and (ri + 1) % 25 == 0:
            print(f"  [h2h] {ri+1}/{len(roots)} ends/order", flush=True)
    s0 = np.array(res["order0"]); s1 = np.array(res["order1"])
    allscores = np.concatenate([s0, s1])
    return {
        "winrate": _winrate(allscores),
        "winrate_order0": _winrate(s0),
        "winrate_order1": _winrate(s1),
        "mean_margin": float(allscores.mean()),
        "mean_margin_order0": float(s0.mean()) if len(s0) else 0.0,
        "mean_margin_order1": float(s1.mean()) if len(s1) else 0.0,
        "n_ends": int(len(allscores)),
    }


__all__ = ["Player", "CsasPlayer", "WorldPlayer", "H2HRoot", "build_h2h_roots",
           "play_end", "head_to_head"]
