from __future__ import annotations

SCENARIOS = [
    {
        "id": "md_draw_around_guard",
        "name": "Draw Around Center Guard",
        "description": "Mixed doubles mid-end. You have hammer and need to outdraw a partial center guard.",
        "shots_in_end": 10,
        "shot_index": 6,
        "team_order": 1.0,
        "stone_block": 0.0,
        "stones": [
            {"slot": 0, "x": 0.55, "y": 0.18},
            {"slot": 1, "x": -1.55, "y": 0.02},
            {"slot": 6, "x": 0.22, "y": -0.08},
            {"slot": 7, "x": -1.15, "y": 0.62},
        ],
        "defaults": {"speed": 1.55, "angle": 0.04, "spin": 1.2, "y0": 0.02},
    },
    {
        "id": "md_hit_roll_open",
        "name": "Hit And Roll To The Four Foot",
        "description": "Open hit with a roll under cover. Opponent shot stone is biting the button.",
        "shots_in_end": 10,
        "shot_index": 7,
        "team_order": 0.0,
        "stone_block": 1.0,
        "stones": [
            {"slot": 0, "x": -1.75, "y": -0.55},
            {"slot": 6, "x": 0.06, "y": 0.05},
            {"slot": 7, "x": -1.30, "y": -0.05},
            {"slot": 8, "x": 1.05, "y": 0.68},
        ],
        "defaults": {"speed": 2.05, "angle": -0.015, "spin": -1.6, "y0": -0.03},
    },
    {
        "id": "md_last_stone_score_two",
        "name": "Last Stone For Two",
        "description": "Final shot. One counter already scores. Can you draw for a second point?",
        "shots_in_end": 10,
        "shot_index": 9,
        "team_order": 1.0,
        "stone_block": 0.0,
        "stones": [
            {"slot": 0, "x": 0.32, "y": -0.22},
            {"slot": 1, "x": -1.05, "y": -0.12},
            {"slot": 6, "x": 0.64, "y": 0.18},
            {"slot": 7, "x": -1.62, "y": 0.52},
        ],
        "defaults": {"speed": 1.48, "angle": 0.022, "spin": 1.0, "y0": -0.015},
    },
]
