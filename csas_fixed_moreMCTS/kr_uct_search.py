#!/usr/bin/env python3
"""Policy-guided batched search with KR-UCT-style continuous-action sharing."""

from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import torch

from common import FIXED_ROOT, NUM_STONES, POS_MAX, compact_m_to_raw, in_play_raw, next_condition, raw_to_compact_m
from curling_sim_jax import CurlingParams, simulate_from_params
from new_architectures import ValueSetTransformerGaussian
from policy_model import PolicySetTransformerMDN


def load_policy(path: str | Path, device: torch.device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    args = ckpt.get("args", {})
    model = PolicySetTransformerMDN(
        input_dim=ckpt.get("input_dim", 24),
        cond_dim=ckpt.get("cond_dim", 3),
        action_dim=ckpt.get("action_dim", 4),
        hidden_dim=args.get("hidden_dim", 256),
        n_layers=args.get("n_layers", 4),
        n_heads=args.get("n_heads", 4),
        dropout=0.0,
        n_mixtures=args.get("n_mixtures", 16),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    mean = torch.tensor(ckpt["action_mean"], dtype=torch.float32, device=device)
    std = torch.tensor(ckpt["action_std"], dtype=torch.float32, device=device)
    return model, mean, std


def load_value_model(path: str | Path, device: torch.device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    args = ckpt.get("args", {})
    model = ValueSetTransformerGaussian(
        input_dim=ckpt.get("input_dim", 24),
        cond_dim=ckpt.get("cond_dim", 3),
        hidden_dim=ckpt.get("hidden_dim", args.get("hidden_dim", 256)),
        n_layers=args.get("n_layers", 4),
        n_heads=args.get("n_heads", 4),
        dropout=0.0,
        min_logvar=args.get("min_logvar", -6.0),
        max_logvar=args.get("max_logvar", 3.5),
    ).to(device)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model.eval()
    return model


@lru_cache(maxsize=16)
def _sim_vmap_for_n(n_prev: int):
    p = CurlingParams()

    @jax.jit
    def run(prev: jnp.ndarray, actions: jnp.ndarray):
        return jax.vmap(lambda a: simulate_from_params(p, prev, a, dynamic=False))(actions)

    return run


def _new_slot(raw_state: np.ndarray, stone_block: float) -> int:
    live = in_play_raw(raw_state)
    start = 6 if stone_block >= 0.5 else 0
    for idx in range(start, start + 6):
        if not live[idx]:
            return idx
    for idx in range(NUM_STONES):
        if not live[idx]:
            return idx
    return NUM_STONES - 1


def _simulate_candidates(raw_state_norm: np.ndarray, cond: np.ndarray, actions: np.ndarray) -> np.ndarray:
    raw = np.asarray(raw_state_norm, dtype=np.float32).reshape(NUM_STONES, 2) * POS_MAX
    live = in_play_raw(raw)
    compact_slots = raw_to_compact_m(raw)
    live_slots = np.where(live)[0].astype(np.int64)
    prev = compact_slots[live_slots]
    if prev.size == 0:
        prev = np.zeros((0, 2), dtype=np.float32)
    n_prev = int(prev.shape[0])
    final = np.asarray(_sim_vmap_for_n(n_prev)(jnp.asarray(prev, dtype=jnp.float32), jnp.asarray(actions, dtype=jnp.float32)))
    new_slot = _new_slot(raw, float(cond[2]))
    states = np.full((len(actions), NUM_STONES, 2), POS_MAX, dtype=np.float32)
    for k in range(len(actions)):
        compact_out = np.full((NUM_STONES, 2), np.nan, dtype=np.float32)
        compact_out[live_slots] = final[k, :n_prev]
        compact_out[new_slot] = final[k, n_prev]
        states[k] = compact_m_to_raw(compact_out)
    return (states.reshape(len(actions), -1) / POS_MAX).astype(np.float32)


@torch.no_grad()
def _sample_actions(policy, action_mean, action_std, x, c, n, device, temperature, std_scale, global_frac):
    z = policy.sample_z(
        torch.as_tensor(x[None], dtype=torch.float32, device=device),
        torch.as_tensor(c[None], dtype=torch.float32, device=device),
        n_samples=n,
        temperature=temperature,
        std_scale=std_scale,
    )[0]
    actions = z * action_std + action_mean
    if global_frac > 0.0:
        m = int(round(n * global_frac))
        if m > 0:
            global_actions = actions[:m].clone()
            global_actions[:, 0] = torch.empty(m, device=device).uniform_(0.75, 2.25)
            global_actions[:, 1] = torch.empty(m, device=device).uniform_(-0.65, 0.65)
            global_actions[:, 2] = torch.empty(m, device=device).uniform_(-2.5, 2.5)
            global_actions[:, 3] = torch.empty(m, device=device).uniform_(-0.45, 0.45)
            actions[:m] = global_actions
    actions[:, 0] = actions[:, 0].clamp(0.4, 3.0)
    actions[:, 1] = actions[:, 1].clamp(-1.2, 1.2)
    actions[:, 2] = actions[:, 2].clamp(-4.0, 4.0)
    actions[:, 3] = actions[:, 3].clamp(-0.75, 0.75)
    return actions.cpu().numpy().astype(np.float32)


def kr_smooth_scores(actions: np.ndarray, values: np.ndarray, action_mean: np.ndarray, action_std: np.ndarray,
                     bandwidth: float, uct_c: float) -> np.ndarray:
    z = (actions - action_mean[None]) / np.maximum(action_std[None], 1e-4)
    d2 = ((z[:, None, :] - z[None, :, :]) ** 2).sum(axis=-1)
    w = np.exp(-0.5 * d2 / max(bandwidth, 1e-6) ** 2)
    eff_n = w.sum(axis=1)
    q = (w @ values) / np.maximum(eff_n, 1e-6)
    bonus = uct_c * np.sqrt(math.log(len(values) + 1.0) / (eff_n + 1.0))
    return q + bonus


@torch.no_grad()
def evaluate_states(value_model, states_norm: np.ndarray, cond: np.ndarray, device: torch.device, batch_size: int = 2048):
    out = []
    c = torch.as_tensor(cond[None].repeat(len(states_norm), axis=0), dtype=torch.float32, device=device)
    for i in range(0, len(states_norm), batch_size):
        x = torch.as_tensor(states_norm[i:i + batch_size], dtype=torch.float32, device=device)
        mean, _ = value_model(x, c[i:i + batch_size])
        out.append(mean.squeeze(-1).detach().cpu().numpy())
    return np.concatenate(out, axis=0)


class KRUctSearcher:
    def __init__(
        self,
        policy_path: str | Path,
        value_path: str | Path,
        device: str = "auto",
        candidates: int = 256,
        rollout_depth: int = 1,
        child_candidates: int = 64,
        kernel_bandwidth: float = 0.75,
        uct_c: float = 0.05,
        temperature: float = 1.35,
        std_scale: float = 1.6,
        global_frac: float = 0.20,
        eval_batch_size: int = 2048,
    ):
        self.device = torch.device(device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.policy, self.action_mean_t, self.action_std_t = load_policy(policy_path, self.device)
        self.value_model = load_value_model(value_path, self.device)
        self.action_mean = self.action_mean_t.detach().cpu().numpy()
        self.action_std = self.action_std_t.detach().cpu().numpy()
        self.candidates = int(candidates)
        self.rollout_depth = int(rollout_depth)
        self.child_candidates = int(child_candidates)
        self.kernel_bandwidth = float(kernel_bandwidth)
        self.uct_c = float(uct_c)
        self.temperature = float(temperature)
        self.std_scale = float(std_scale)
        self.global_frac = float(global_frac)
        self.eval_batch_size = int(eval_batch_size)

    def _one_ply(self, x: np.ndarray, c: np.ndarray, n_candidates: int, maximize: bool = True):
        actions = _sample_actions(
            self.policy, self.action_mean_t, self.action_std_t, x, c, n_candidates, self.device,
            self.temperature, self.std_scale, self.global_frac,
        )
        post = _simulate_candidates(x, c, actions)
        vals = evaluate_states(self.value_model, post, c, self.device, self.eval_batch_size)
        scores = kr_smooth_scores(actions, vals, self.action_mean, self.action_std, self.kernel_bandwidth, self.uct_c)
        idx = int(np.argmax(scores) if maximize else np.argmin(scores))
        return {
            "best_idx": idx,
            "best_action": actions[idx],
            "best_state": post[idx],
            "best_value": float(vals[idx]),
            "best_score": float(scores[idx]),
            "mean_value": float(np.mean(vals)),
            "p90_value": float(np.quantile(vals, 0.90)),
            "values": vals,
            "actions": actions,
        }

    def search(self, x: np.ndarray, c: np.ndarray):
        root = self._one_ply(x, c, self.candidates, maximize=True)
        if self.rollout_depth <= 1:
            return root
        state = root["best_state"]
        cond = next_condition(c)
        value = root["best_value"]
        # Approximate alternating play: opponent minimizes current player's value, then us maximizes it.
        for depth in range(2, self.rollout_depth + 1):
            maximize = (depth % 2 == 1)
            child = self._one_ply(state, cond, self.child_candidates, maximize=maximize)
            value = child["best_value"]
            state = child["best_state"]
            cond = next_condition(cond)
        root["rollout_value"] = float(value)
        return root
