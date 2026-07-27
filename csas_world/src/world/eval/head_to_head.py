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

    def _value_fn(self, states, cond):
        torch = self._torch
        with torch.no_grad():
            c = np.broadcast_to(cond, (len(states), 3)).astype(np.float32)
            xt = torch.as_tensor(states, dtype=torch.float32, device=self.device)
            ct = torch.as_tensor(c, dtype=torch.float32, device=self.device)
            mean = self.value_model.value_head.value(self.value_model.encode(xt, ct))
        return mean.cpu().numpy()

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
