#!/usr/bin/env python3
"""Policy-guided batched search with KR-UCT-style continuous-action sharing."""

from __future__ import annotations

import math
import os
from functools import lru_cache
from pathlib import Path
import sys
from collections import OrderedDict

import jax
import jax.numpy as jnp
import numpy as np
import torch

from common import (
    ACTION_ANGLE_MAX,
    ACTION_ANGLE_MIN,
    ACTION_SPEED_MAX,
    ACTION_SPEED_MIN,
    ACTION_SPIN_MAX,
    ACTION_SPIN_MIN,
    ACTION_Y0_MAX,
    ACTION_Y0_MIN,
    FIXED_ROOT,
    NUM_STONES,
    POS_MAX,
    compact_m_to_raw,
    in_play_raw,
    next_condition,
    raw_to_compact_m,
)
from curling_sim_jax import CurlingParams, simulate_from_params
from new_architectures import ValueSetTransformerGaussian
from policy_model import PolicySetTransformerMDN


def load_policy(path: str | Path, device: torch.device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    args = ckpt.get("args", {})
    arch = str(ckpt.get("arch", "policy_set_transformer_mdn"))
    if arch in {"policy_graph_transformer_mdn", "policy_graph_transformer_fullcov_mdn"}:
        graph_env = ckpt.get("graph_feature_env") or {}
        for key, value in graph_env.items():
            if value is not None:
                os.environ[str(key)] = str(value)
        sys.path.insert(0, str(FIXED_ROOT / "valueModel"))
        sys.path.insert(0, str(FIXED_ROOT / "valueModel" / "ablation"))
        from policy_graph_model import PolicyGraphTransformerFullCovMDN, PolicyGraphTransformerMDN

        model_cls = (
            PolicyGraphTransformerFullCovMDN
            if arch == "policy_graph_transformer_fullcov_mdn"
            else PolicyGraphTransformerMDN
        )
        model = model_cls(
            input_dim=ckpt.get("input_dim", 24),
            cond_dim=ckpt.get("cond_dim", 3),
            action_dim=ckpt.get("action_dim", 4),
            hidden_dim=args.get("hidden_dim", 256),
            n_layers=args.get("n_layers", 4),
            n_heads=args.get("n_heads", 4),
            dropout=0.0,
            n_mixtures=args.get("n_mixtures", 16),
        ).to(device)
    else:
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
    arch = str(ckpt.get("arch", ""))
    if arch == "graph_transformer_gaussian":
        graph_env = ckpt.get("graph_feature_env") or {}
        for key, value in graph_env.items():
            if value is not None:
                os.environ[str(key)] = str(value)
        sys.path.insert(0, str(FIXED_ROOT / "valueModel"))
        sys.path.insert(0, str(FIXED_ROOT / "valueModel" / "ablation"))
        from gnn_models import GNN_REGISTRY  # type: ignore

        args = ckpt.get("args", {})
        model = GNN_REGISTRY["graph_transformer_gaussian"](
            input_dim=ckpt.get("input_dim", 24),
            cond_dim=ckpt.get("cond_dim", 3),
            hidden_dim=ckpt.get("hidden_dim", args.get("hidden_dim", 256)),
            n_layers=args.get("n_layers", 4),
            n_heads=args.get("n_heads", 4),
            dropout=args.get("dropout", 0.0),
            min_logvar=args.get("min_logvar", -6.0),
            max_logvar=args.get("max_logvar", 3.5),
        ).to(device)
        state = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state)
        model.eval()
        return model
    if arch == "graph_transformer_gaussian_precomputed":
        raise NotImplementedError(
            "Search-time evaluation of graph_transformer_gaussian_precomputed checkpoints is not supported; "
            "use the non-precomputed graph checkpoint for horizon-curriculum generation."
        )
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


@lru_cache(maxsize=16)
def _sim_vmap_for_n_batched(n_prev: int):
    p = CurlingParams()

    @jax.jit
    def run(prev_batch: jnp.ndarray, actions_batch: jnp.ndarray):
        return jax.vmap(
            lambda prev, actions: jax.vmap(
                lambda a: simulate_from_params(p, prev, a, dynamic=False)
            )(actions)
        )(prev_batch, actions_batch)

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


def _simulate_candidates_batched(raw_state_norm_batch: np.ndarray, cond_batch: np.ndarray, actions_batch: np.ndarray) -> np.ndarray:
    raw_batch = np.asarray(raw_state_norm_batch, dtype=np.float32).reshape(-1, NUM_STONES, 2) * POS_MAX
    cond_batch = np.asarray(cond_batch, dtype=np.float32).reshape(-1, 3)
    actions_batch = np.asarray(actions_batch, dtype=np.float32)
    batch_size, n_actions = actions_batch.shape[:2]
    live_masks = np.asarray([in_play_raw(raw) for raw in raw_batch], dtype=bool)
    live_slots_list = [np.where(mask)[0].astype(np.int64) for mask in live_masks]
    n_prev_set = {int(len(slots)) for slots in live_slots_list}
    if len(n_prev_set) != 1:
        raise ValueError(f"_simulate_candidates_batched requires uniform n_prev; got {sorted(n_prev_set)}")
    n_prev = int(next(iter(n_prev_set)))
    prev_batch = np.zeros((batch_size, n_prev, 2), dtype=np.float32)
    new_slots = np.zeros((batch_size,), dtype=np.int64)
    for i, raw in enumerate(raw_batch):
        compact_slots = raw_to_compact_m(raw)
        live_slots = live_slots_list[i]
        if n_prev > 0:
            prev_batch[i, :n_prev] = compact_slots[live_slots]
        new_slots[i] = _new_slot(raw, float(cond_batch[i, 2]))
    final = np.asarray(
        _sim_vmap_for_n_batched(n_prev)(
            jnp.asarray(prev_batch, dtype=jnp.float32),
            jnp.asarray(actions_batch, dtype=jnp.float32),
        )
    )
    states = np.full((batch_size, n_actions, NUM_STONES, 2), POS_MAX, dtype=np.float32)
    for i in range(batch_size):
        live_slots = live_slots_list[i]
        new_slot = int(new_slots[i])
        for k in range(n_actions):
            compact_out = np.full((NUM_STONES, 2), np.nan, dtype=np.float32)
            if n_prev > 0:
                compact_out[live_slots] = final[i, k, :n_prev]
            compact_out[new_slot] = final[i, k, n_prev]
            states[i, k] = compact_m_to_raw(compact_out)
    return (states.reshape(batch_size, n_actions, -1) / POS_MAX).astype(np.float32)


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
            global_actions[:, 0] = torch.empty(m, device=device).uniform_(ACTION_SPEED_MIN, ACTION_SPEED_MAX)
            global_actions[:, 1] = torch.empty(m, device=device).uniform_(ACTION_ANGLE_MIN, ACTION_ANGLE_MAX)
            global_actions[:, 2] = torch.empty(m, device=device).uniform_(ACTION_SPIN_MIN, ACTION_SPIN_MAX)
            global_actions[:, 3] = torch.empty(m, device=device).uniform_(ACTION_Y0_MIN, ACTION_Y0_MAX)
            actions[:m] = global_actions
    actions[:, 0] = actions[:, 0].clamp(ACTION_SPEED_MIN, ACTION_SPEED_MAX)
    actions[:, 1] = actions[:, 1].clamp(ACTION_ANGLE_MIN, ACTION_ANGLE_MAX)
    actions[:, 2] = actions[:, 2].clamp(ACTION_SPIN_MIN, ACTION_SPIN_MAX)
    actions[:, 3] = actions[:, 3].clamp(ACTION_Y0_MIN, ACTION_Y0_MAX)
    return np.asarray(actions.detach().cpu().tolist(), dtype=np.float32)


@torch.no_grad()
def _sample_actions_batch(policy, action_mean, action_std, x_batch, c_batch, n, device, temperature, std_scale, global_frac):
    x_t = torch.as_tensor(x_batch, dtype=torch.float32, device=device)
    c_t = torch.as_tensor(c_batch, dtype=torch.float32, device=device)
    z = policy.sample_z(
        x_t,
        c_t,
        n_samples=n,
        temperature=temperature,
        std_scale=std_scale,
    )
    actions = z * action_std.view(1, 1, -1) + action_mean.view(1, 1, -1)
    if global_frac > 0.0:
        m = int(round(n * global_frac))
        if m > 0:
            actions[:, :m, 0] = torch.empty(actions.size(0), m, device=device).uniform_(ACTION_SPEED_MIN, ACTION_SPEED_MAX)
            actions[:, :m, 1] = torch.empty(actions.size(0), m, device=device).uniform_(ACTION_ANGLE_MIN, ACTION_ANGLE_MAX)
            actions[:, :m, 2] = torch.empty(actions.size(0), m, device=device).uniform_(ACTION_SPIN_MIN, ACTION_SPIN_MAX)
            actions[:, :m, 3] = torch.empty(actions.size(0), m, device=device).uniform_(ACTION_Y0_MIN, ACTION_Y0_MAX)
    actions[:, :, 0] = actions[:, :, 0].clamp(ACTION_SPEED_MIN, ACTION_SPEED_MAX)
    actions[:, :, 1] = actions[:, :, 1].clamp(ACTION_ANGLE_MIN, ACTION_ANGLE_MAX)
    actions[:, :, 2] = actions[:, :, 2].clamp(ACTION_SPIN_MIN, ACTION_SPIN_MAX)
    actions[:, :, 3] = actions[:, :, 3].clamp(ACTION_Y0_MIN, ACTION_Y0_MAX)
    return np.asarray(actions.detach().cpu().tolist(), dtype=np.float32)


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
    states_arr = np.asarray(states_norm, dtype=np.float32)
    cond_arr = np.asarray(cond, dtype=np.float32)
    if cond_arr.ndim == 1:
        cond_arr = np.repeat(cond_arr[None], len(states_arr), axis=0)

    combo = np.concatenate([states_arr, cond_arr], axis=1)
    uniq_combo, inverse = np.unique(combo, axis=0, return_inverse=True)
    uniq_states = uniq_combo[:, : states_arr.shape[1]].astype(np.float32, copy=False)
    uniq_conds = uniq_combo[:, states_arr.shape[1] :].astype(np.float32, copy=False)

    out = []
    model_module = sys.modules.get(value_model.__class__.__module__)
    build_graph_batch_fast = getattr(model_module, "build_graph_batch_fast", None)
    compute_edge_features_fast = getattr(model_module, "compute_edge_features_fast", None)
    has_precomputed = callable(getattr(value_model, "forward_precomputed", None)) and callable(build_graph_batch_fast) and callable(compute_edge_features_fast)

    for i in range(0, len(uniq_states), batch_size):
        x = torch.as_tensor(uniq_states[i:i + batch_size], dtype=torch.float32, device=device)
        c = torch.as_tensor(uniq_conds[i:i + batch_size], dtype=torch.float32, device=device)
        if has_precomputed:
            node_feats, node_coords, node_mask, _ = build_graph_batch_fast(x, device)
            edge_feats = compute_edge_features_fast(node_coords, node_feats, node_mask, c=c)
            mean, _ = value_model.forward_precomputed(node_feats, edge_feats, node_mask, c)
        else:
            mean, _ = value_model(x, c)
        out.append(np.asarray(mean.squeeze(-1).detach().cpu().tolist(), dtype=np.float32))
    uniq_values = np.concatenate(out, axis=0)
    return uniq_values[inverse]


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
        self.action_mean = np.asarray(self.action_mean_t.detach().cpu().tolist(), dtype=np.float32)
        self.action_std = np.asarray(self.action_std_t.detach().cpu().tolist(), dtype=np.float32)
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
