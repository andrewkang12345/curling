import numpy as np

from world.search.selfplay import _paired_gate_stats


def test_paired_gate_accepts_positive_enriched_screen():
    accepted, mean, se, t_stat = _paired_gate_stats(
        np.array([1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0]), 0.5)
    assert accepted
    assert mean > 0 and se > 0 and t_stat >= 0.5


def test_paired_gate_rejects_zero_and_negative_screens():
    assert not _paired_gate_stats(np.zeros(8), 0.5)[0]
    assert not _paired_gate_stats(-np.ones(8), 0.5)[0]


def test_paired_gate_accepts_constant_positive_delta():
    accepted, mean, se, t_stat = _paired_gate_stats(np.ones(8), 0.5)
    assert accepted and mean == 1.0 and se == 0.0 and np.isposinf(t_stat)
