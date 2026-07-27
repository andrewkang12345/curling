"""Model + multi-head loss shape/grad tests (CPU)."""
import numpy as np
import pytest
import torch
from types import SimpleNamespace

import world  # noqa: F401  (bootstrap)
from world.config import Config
from world.model import WorldModel
from world.losses import compute_losses
from world.replay.schema import empty_record, collate, SOURCE_HUMAN, SOURCE_MCTS


def _synthetic_batch(K, M, n=4):
    recs = []
    for i in range(n):
        r = empty_record(K, M)
        r["x0"][0:2] = [0.18, 0.20]; r["x0"][12:14] = [0.20, 0.18]
        r["c0"][:] = [0.1 * i, float(i % 2), float(i % 2)]
        r["a_raw"][:] = np.array([1.4, 0.0, 0.5, 0.0], np.float32)[None].repeat(K, 0)
        r["next_states"][:] = r["x0"][None].repeat(K, 0)
        r["next_conds"][:] = r["c0"][None].repeat(K, 0)
        r["next_live"][:, [0, 6]] = 1.0
        r["value_target"][:] = np.linspace(0.5, -0.5, K + 1); r["value_mask"][:] = 1.0
        r["reward_mask"][-1] = 1.0
        r["outcome_margin"] = np.float32(1.0); r["outcome_mask"] = np.float32(1.0)
        r["consistency_mask"][:] = 1.0
        r["bc_action_raw"][:] = [1.5, 0.01, 0.3, 0.0]; r["bc_mask"] = np.float32(i % 2 == 0)
        r["dist_actions_raw"][:] = np.array([1.3, 0.0, 0.4, 0.0], np.float32)[None].repeat(M, 0)
        r["dist_weights"][:] = 1.0 / M; r["dist_mask"] = np.float32(i % 2 == 1)
        r["horizon"] = np.int64(5); r["source"] = np.int64(SOURCE_HUMAN if i % 2 else SOURCE_MCTS)
        recs.append(r)
    return collate(recs)


def test_warm_start_and_forward():
    cfg = Config()
    m = WorldModel(cfg.model)
    rep = m.warm_start(cfg.csas_path(cfg.paths.prior_policy_ckpt).as_posix(),
                       cfg.csas_path(cfg.paths.prior_value_ckpt).as_posix())
    assert rep["trunk"]["missing"] == 0 and rep["trunk"]["unexpected"] == 0
    x = torch.zeros(3, 24); c = torch.zeros(3, 3)
    out = m.initial_inference(x, c)
    assert out["latent"].shape == (3, 256)
    pi, mu, tril = out["policy"]
    assert pi.shape == (3, 16) and mu.shape == (3, 16, 4) and tril.shape == (3, 16, 4, 4)
    assert out["value_mean"].shape == (3,)
    assert "outcome" not in out


def test_full_loss_backward():
    cfg = Config(); cfg.model.use_decoder = True
    cfg.model.use_outcome = True; cfg.loss.outcome = 0.5
    m = WorldModel(cfg.model); m.train()
    batch = _synthetic_batch(cfg.replay.unroll_steps, cfg.search.soft_topk)
    total, metrics = compute_losses(m, batch, cfg.loss)
    assert torch.isfinite(total)
    for k in ["policy_bc", "policy_distill", "value_mse", "outcome",
              "consistency", "decoder"]:
        assert k in metrics
    total.backward()
    g = torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad], 1e9)
    assert torch.isfinite(g)


def test_ablation_policy_value_only():
    cfg = Config()
    cfg.model.use_dynamics = cfg.model.use_outcome = False
    cfg.model.use_consistency = cfg.model.use_decoder = False
    cfg.loss.outcome = cfg.loss.consistency = cfg.loss.decoder = 0.0
    m = WorldModel(cfg.model); m.train()
    batch = _synthetic_batch(cfg.replay.unroll_steps, cfg.search.soft_topk)
    total, metrics = compute_losses(m, batch, cfg.loss)
    assert "consistency" not in metrics and "outcome" not in metrics
    assert "policy_bc" in metrics and "value_mse" in metrics
    total.backward()


def test_unroll_length():
    cfg = Config()
    m = WorldModel(cfg.model)
    K = cfg.replay.unroll_steps
    x = torch.zeros(2, 24); c = torch.zeros(2, 3)
    a = torch.zeros(2, K, 4)
    steps = m.unroll(x, c, a)
    assert len(steps) == K + 1


def test_outcome_targets_follow_recurrent_perspective_and_mask():
    cfg = Config()
    cfg.loss.policy_bc = cfg.loss.policy_distill = cfg.loss.value = 0.0
    cfg.loss.consistency = cfg.loss.decoder = 0.0
    cfg.loss.outcome = 1.0
    batch = _synthetic_batch(1, cfg.search.soft_topk, n=2)
    batch["c0"][:, 2] = 0.0
    batch["next_conds"][:, 0, 2] = 1.0
    batch["outcome_margin"][:] = 1.0
    batch["outcome_mask"][:] = 1.0
    batch["consistency_mask"][:] = 1.0

    root_logits = torch.full((2, 17), -20.0)
    next_logits = torch.full((2, 17), -20.0)
    root_logits[:, 9] = 20.0   # +1 from the root perspective
    next_logits[:, 7] = 20.0   # -1 after side-to-throw switches
    dummy_policy = (
        torch.zeros(2, 1),
        torch.zeros(2, 1, 4),
        torch.eye(4).reshape(1, 1, 4, 4).repeat(2, 1, 1, 1),
    )

    class OutcomeOnlyModel:
        cfg = SimpleNamespace(outcome_bins=17, use_consistency=False)
        outcome_head = object()
        value_head = None
        decoder = None

        @staticmethod
        def raw_to_box(action):
            return action

        @staticmethod
        def unroll(x0, c0, action):
            return [{"policy": dummy_policy, "outcome": root_logits}, {"outcome": next_logits}]

    total, metrics = compute_losses(OutcomeOnlyModel(), batch, cfg.loss)
    assert total < 1e-6
    assert metrics["outcome"] < 1e-6

    # An invalid recurrent transition must not affect the valid root loss.
    batch["consistency_mask"][:] = 0.0
    next_logits[:, 7] = -20.0
    next_logits[:, 9] = 20.0
    total, metrics = compute_losses(OutcomeOnlyModel(), batch, cfg.loss)
    assert total < 1e-6
    assert metrics["outcome"] < 1e-6
