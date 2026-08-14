from __future__ import annotations

import json

import numpy as np

from arena import engine, solver
from world.search.noise import LocalNoise


def test_arena_raw_box_expands_curl_and_release_without_changing_model_box():
    action = np.asarray([2.50, 0.0, engine.ARENA_CURL_MAX, engine.FOUR_FEET_M],
                        dtype=np.float32)
    solved, info = solver.solve(np.ones(24, dtype=np.float32),
                                np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
                                {"type": "params", "action": action})
    assert not info["clipped"]
    np.testing.assert_allclose(solved, action)
    assert engine.ACTION_HIGH[2] == 7.0
    assert engine.ACTION_HIGH[3] == 0.23


def test_sandbox_scenario_encodes_requested_throw_and_turn():
    sc = engine.normalize_scenario({
        "name": "late end",
        "hammer": "B",
        "throw": 6,
        "end": 3,
        "totals": {"A": 2, "B": 1},
        "stones": [
            {"team": "A", "slot": 0, "along": -1.0, "lateral": 0.2},
            {"team": "B", "slot": 6, "along": 0.1, "lateral": -0.3},
        ],
    })
    state, cond = engine.scenario_state(sc)
    assert sc["turn"] == "B"
    assert cond[0] == np.float32(5 / 9)
    assert cond[1] == 1.0
    assert cond[2] == 1.0
    assert {s["slot"] for s in engine.stones_from_state(state)} == {0, 6}


def test_sandbox_match_keeps_end_throw_score_and_noise(tmp_path, monkeypatch):
    monkeypatch.setattr(engine, "MATCH_DIR", tmp_path)
    m = engine.Match.create_from_scenario(
        {"id": "abc", "name": "test", "hammer": "A", "throw": 8, "end": 4,
         "totals": {"A": 3, "B": 5}, "stones": []},
        {"A": "human", "B": "human"}, ends=6, noise=True,
        noise_scales=[0.0, 0.5, 1.5, 2.0], seed=123,
    )
    d = m.to_dict()
    assert d["turn"]["end"] == 4
    assert d["turn"]["throw"] == 8
    assert d["turn"]["team"] == "A"
    assert d["totals"] == {"A": 3, "B": 5}
    assert d["noise_scales"] == [0.0, 0.5, 1.5, 2.0]


def test_noise_scales_are_per_parameter_and_use_custom_bounds(tmp_path):
    cfg = tmp_path / "noise.json"
    cfg.write_text(json.dumps({"local": {"distribution": "gaussian",
                                          "std": [1.0, 1.0, 1.0, 1.0],
                                          "min_std": 0.001}}))
    noise = LocalNoise(str(cfg), seed=7, scale_multipliers=np.asarray([0, 0, 1, 0]),
                       action_low=np.asarray([-10, -10, -50, -10]),
                       action_high=np.asarray([10, 10, 50, 10]))
    center = np.zeros((1, 4), dtype=np.float32)
    samples = noise.sample_batch(center, 256)[0]
    np.testing.assert_array_equal(samples[:, [0, 1, 3]], 0.0)
    assert samples[:, 2].std() > 0.5

