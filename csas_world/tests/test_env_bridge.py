"""Simulator-bridge tests (JAX on CPU)."""
import numpy as np

import world  # noqa: F401
from world import env_bridge
from world.replay.schema import empty_record, collate, field_shapes


def test_simulate_score_nextcond():
    x = np.zeros(24, np.float32)
    x[0:2] = [0.18, 0.20]; x[12:14] = [0.20, 0.18]
    c = np.array([0.0, 1.0, 0.0], np.float32)
    actions = np.array([[1.4, 0.0, 0.5, 0.0], [2.0, 0.05, -0.5, 0.1]], np.float32)
    posts = env_bridge.simulate(x, c, actions)
    assert posts.shape == (2, 24)
    v = env_bridge.score_end(posts[0], int(c[2]))
    assert isinstance(v, float)
    nc = env_bridge.next_condition(c, 8)
    assert nc.shape == (3,)
    assert abs(nc[1] - (1 - c[1])) < 1e-6  # team_order flips


def test_legality_shapes():
    x = np.zeros(24, np.float32)
    x[0:2] = [0.18, 0.20]; x[12:14] = [0.20, 0.18]
    c = np.array([0.0, 1.0, 0.0], np.float32)
    actions = np.array([[2.3, 0.0, 0.0, 0.0]] * 5, np.float32)
    posts = env_bridge.simulate(x, c, actions)
    corrected, illegal = env_bridge.apply_legality(x, posts, horizon=10, cond=c)
    assert corrected.shape == posts.shape
    assert illegal.shape == (5,)
    assert illegal.dtype == bool


def test_schema_collate():
    K, M = 5, 24
    recs = [empty_record(K, M) for _ in range(3)]
    batch = collate(recs)
    shapes = field_shapes(K, M)
    for k, s in shapes.items():
        assert tuple(batch[k].shape) == (3, *s)
