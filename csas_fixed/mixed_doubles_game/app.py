from __future__ import annotations

import json
import math
import os
import random
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "inverse"))
sys.path.insert(0, str(ROOT_DIR / "valueModel"))

from inverse.curling_sim_jax import CurlingParams, simulate_from_params  # type: ignore
from score_shots_mc_seq import POS_MAX, evaluate_state_value, load_value_model  # type: ignore
from sim_presets import contact_mild_params  # type: ignore

STATIC_DIR = THIS_DIR / "static"
NOISE_PATH = ROOT_DIR / "noise_versions" / "v1_bowling.json"
DEFAULT_DEVICE = os.environ.get("CURLING_GAME_DEVICE", "cpu")
MAX_SCENARIOS = 180
MIN_CLEAR = 2 * 0.145
SEPARATE_PASSES = 6
SHOT_STAGE_VALUES = [1 / 9, 2 / 9, 3 / 9, 4 / 9, 5 / 9, 6 / 9, 7 / 9, 8 / 9, 1.0]


@dataclass
class NoiseConfig:
    nu: float
    speed_scale: float
    angle_scale_min: float
    angle_scale_max: float
    angle_speed_min: float
    angle_speed_max: float
    spin_std: float
    y0_std: float


def _load_noise_config() -> NoiseConfig:
    data = json.loads(NOISE_PATH.read_text())
    local = data["local"]
    return NoiseConfig(
        nu=float(local["nu"]),
        speed_scale=float(local["speed_scale"]),
        angle_scale_min=float(local["angle_scale_range"][0]),
        angle_scale_max=float(local["angle_scale_range"][1]),
        angle_speed_min=float(local["angle_speed_range"][0]),
        angle_speed_max=float(local["angle_speed_range"][1]),
        spin_std=float(local["std"][2]),
        y0_std=float(local["std"][3]),
    )


def _angle_scale_for_speed(speed: float, cfg: NoiseConfig) -> float:
    speed_clamped = float(np.clip(speed, cfg.angle_speed_min, cfg.angle_speed_max))
    alpha = (speed_clamped - cfg.angle_speed_min) / (cfg.angle_speed_max - cfg.angle_speed_min)
    return float(cfg.angle_scale_max + alpha * (cfg.angle_scale_min - cfg.angle_scale_max))


def _sample_noisy_params(intended: np.ndarray, cfg: NoiseConfig, rng: np.random.Generator) -> np.ndarray:
    speed, angle, spin, y0 = [float(x) for x in intended]
    noisy = np.array(
        [
            speed + rng.standard_t(cfg.nu) * cfg.speed_scale,
            angle + rng.standard_t(cfg.nu) * _angle_scale_for_speed(speed, cfg),
            spin + rng.normal(0.0, cfg.spin_std),
            y0 + rng.normal(0.0, cfg.y0_std),
        ],
        dtype=np.float32,
    )
    noisy[0] = float(np.clip(noisy[0], 0.1, 3.0))
    noisy[1] = float(np.clip(noisy[1], -0.35, 0.35))
    noisy[2] = float(np.clip(noisy[2], -3.0, 3.0))
    noisy[3] = float(np.clip(noisy[3], -0.23, 0.23))
    return noisy


def _graphtf_checkpoint_map() -> dict[int, Path]:
    out: dict[int, Path] = {}
    for split_path in (ROOT_DIR / "holdouts").glob("*/model_graphtf/split_summary.json"):
        try:
            data = json.loads(split_path.read_text())
            holdout_comp = int(data["holdout_competition"])
            ckpt = split_path.parent / "model.pt"
            if ckpt.exists():
                out[holdout_comp] = ckpt
        except Exception:
            continue
    if not out:
        raise FileNotFoundError("No GraphTF holdout checkpoints found.")
    return out


@lru_cache(maxsize=8)
def _load_model_for_competition(competition_id: int):
    ckpt = _graphtf_checkpoint_map().get(int(competition_id))
    if ckpt is None:
        raise FileNotFoundError(f"No GraphTF checkpoint for competition {competition_id}.")
    model_fn, _ = load_value_model(ckpt, device=DEFAULT_DEVICE)
    return model_fn, ckpt


def _stone_cols(prefix: str = "stone") -> list[str]:
    cols: list[str] = []
    for i in range(1, 13):
        cols.extend([f"{prefix}_{i}_x", f"{prefix}_{i}_y"])
    return cols


def _board_from_row(row: pd.Series, prefix: str = "stone") -> np.ndarray:
    board = np.full((12, 2), np.nan, dtype=np.float32)
    for i in range(1, 13):
        x = row.get(f"{prefix}_{i}_x", np.nan)
        y = row.get(f"{prefix}_{i}_y", np.nan)
        if pd.isna(x) or pd.isna(y):
            continue
        x = float(x)
        y = float(y)
        if x in (0.0, POS_MAX) or y in (0.0, POS_MAX):
            continue
        board[i - 1, 0] = (800.0 - y) * 0.003048
        board[i - 1, 1] = (x - 750.0) * 0.003048
    return board


def _board_to_client(board_m: np.ndarray) -> list[dict[str, float | int | str]]:
    out: list[dict[str, float | int | str]] = []
    for slot in range(12):
        if np.isfinite(board_m[slot, 0]) and np.isfinite(board_m[slot, 1]):
            out.append(
                {
                    "slot": int(slot),
                    "team": "A" if slot < 6 else "B",
                    "x": float(board_m[slot, 0]),
                    "y": float(board_m[slot, 1]),
                }
            )
    return out


def _occupied_slots(board_m: np.ndarray) -> list[int]:
    return [i for i in range(12) if np.isfinite(board_m[i, 0]) and np.isfinite(board_m[i, 1])]


def _separate_overlaps(pts: np.ndarray, min_gap: float = MIN_CLEAR, passes: int = SEPARATE_PASSES) -> np.ndarray:
    if pts.size == 0:
        return pts
    out = pts.copy()
    n = out.shape[0]
    for _ in range(passes):
        moved = False
        for i in range(n):
            for j in range(i + 1, n):
                dx = out[j, 0] - out[i, 0]
                dy = out[j, 1] - out[i, 1]
                d = math.hypot(dx, dy)
                if d < 1e-9:
                    dx, dy, d = 1e-6, 0.0, 1e-6
                if d < min_gap:
                    push = 0.5 * (min_gap - d)
                    nx, ny = dx / d, dy / d
                    out[i, 0] -= push * nx
                    out[i, 1] -= push * ny
                    out[j, 0] += push * nx
                    out[j, 1] += push * ny
                    moved = True
        if not moved:
            break
    return out


def _sanitize_board(board_m: np.ndarray) -> np.ndarray:
    out = board_m.copy()
    slots = _occupied_slots(out)
    if slots:
        compact = out[slots].astype(np.float32)
        out[slots] = _separate_overlaps(compact)
    return out


def _next_slot_for_block(board_m: np.ndarray, stone_block: float) -> int:
    start = 0 if int(round(float(stone_block))) == 0 else 6
    for slot in range(start, start + 6):
        if not (np.isfinite(board_m[slot, 0]) and np.isfinite(board_m[slot, 1])):
            return slot
    return start + 5


def _slotted_board_from_compact(final_compact: np.ndarray, prev_slots: list[int], new_slot: int) -> np.ndarray:
    board = np.full((12, 2), np.nan, dtype=np.float32)
    n_prev = len(prev_slots)
    for idx, slot in enumerate(prev_slots):
        if idx < final_compact.shape[0]:
            xy = final_compact[idx]
            if np.isfinite(xy[0]) and np.isfinite(xy[1]):
                board[slot] = xy
    if final_compact.shape[0] > n_prev:
        xy = final_compact[-1]
        if np.isfinite(xy[0]) and np.isfinite(xy[1]):
            board[new_slot] = xy
    return board


def _sample_trajectory_frames(traj_compact: np.ndarray, prev_slots: list[int], new_slot: int) -> dict[str, Any]:
    keep = max(1, int(math.ceil(len(traj_compact) / 80)))
    sampled = traj_compact[::keep]
    if not np.array_equal(sampled[-1], traj_compact[-1]):
        sampled = np.concatenate([sampled, traj_compact[-1:]], axis=0)
    return {
        "stone_slot": int(new_slot),
        "frames": [_slotted_board_from_compact(frame, prev_slots, new_slot).tolist() for frame in sampled],
    }


class ShotParamsRequest(BaseModel):
    scenario_id: str
    speed: float = Field(ge=0.1, le=3.0)
    angle: float = Field(ge=-0.35, le=0.35)
    spin: float = Field(ge=-3.0, le=3.0)
    y0: float = Field(ge=-0.23, le=0.23)


class PlayShotRequest(ShotParamsRequest):
    seed: int | None = None


class GameEngine:
    def __init__(self) -> None:
        self.noise = _load_noise_config()
        self.model_paths = _graphtf_checkpoint_map()
        self.sim_params = contact_mild_params(CurlingParams)
        self.scenario_rows = self._load_real_shot_scenarios()
        self.scenarios = {row["id"]: row for row in self.scenario_rows}

    def _load_real_shot_scenarios(self) -> list[dict[str, Any]]:
        key_cols = ["CompetitionID", "SessionID", "GameID", "EndID", "ShotID"]

        score_frames = []
        for p in sorted((ROOT_DIR / "holdouts").glob("*/scoring_graphtf/shot_scores_local.csv")):
            df = pd.read_csv(p)
            score_frames.append(df)
        if not score_frames:
            raise FileNotFoundError("No GraphTF local score files found.")
        scores = pd.concat(score_frames, ignore_index=True)

        stones = pd.read_csv(ROOT_DIR / "2026" / "Stones.csv")
        stones = stones.sort_values(["CompetitionID", "SessionID", "GameID", "EndID", "ShotID"]).reset_index(drop=True)
        prev_stones = stones.groupby(["CompetitionID", "SessionID", "GameID", "EndID"], dropna=False)[_stone_cols()].shift(1)
        prev_stones.columns = [f"prev_{c}" for c in prev_stones.columns]
        stones = pd.concat([stones, prev_stones], axis=1)

        merged = scores.merge(stones, on=key_cols + ["TeamID", "PlayerID", "Task", "Handle"], how="inner")

        name_frames = []
        for p in sorted((ROOT_DIR / "holdouts").glob("*/reports/coach_report_mc/shot_scores_local_vs_global_merged_graphtf.csv")):
            df = pd.read_csv(p, usecols=["CompetitionID", "TeamID", "PlayerID", "player_name", "player_label", "team_name"])
            name_frames.append(df.drop_duplicates())
        name_df = pd.concat(name_frames, ignore_index=True).drop_duplicates(
            subset=["CompetitionID", "TeamID", "PlayerID"]
        )
        merged = merged.merge(name_df, on=["CompetitionID", "TeamID", "PlayerID"], how="left")

        teams = pd.read_csv(ROOT_DIR / "2026" / "Teams.csv")
        merged = merged.merge(
            teams.rename(columns={"Name": "team_name_fallback"}),
            on=["CompetitionID", "TeamID"],
            how="left",
        )
        merged["team_name"] = merged["team_name"].fillna(merged["team_name_fallback"])
        merged["player_name"] = merged["player_name"].fillna("Player " + merged["PlayerID"].astype(str))
        merged["player_label"] = merged["player_label"].fillna(
            merged["player_name"] + " (" + merged["team_name"].fillna("Unknown") + ")"
        )

        valid = merged[
            np.isfinite(merged["est_speed"])
            & np.isfinite(merged["est_angle"])
            & np.isfinite(merged["est_spin"])
            & np.isfinite(merged["est_y0"])
        ].copy()
        valid["abs_dv_obs"] = valid["dv_obs"].abs()
        valid["shot_stage"] = valid["shot_norm_next"].apply(
            lambda x: min(SHOT_STAGE_VALUES, key=lambda y: abs(float(x) - y))
        )
        per_stage = max(1, MAX_SCENARIOS // len(SHOT_STAGE_VALUES))
        picked = []
        for stage in SHOT_STAGE_VALUES:
            stage_df = valid[valid["shot_stage"] == stage].copy()
            if stage_df.empty:
                continue
            stage_df = stage_df.sort_values(
                ["abs_dv_obs", "CompetitionID", "GameID", "EndID", "ShotID"],
                ascending=[False, True, True, True, True],
            )
            picked.append(stage_df.head(per_stage))
        valid = pd.concat(picked, ignore_index=True) if picked else valid.head(MAX_SCENARIOS).copy()
        if len(valid) < MAX_SCENARIOS:
            chosen_ids = set(
                zip(
                    valid["CompetitionID"],
                    valid["SessionID"],
                    valid["GameID"],
                    valid["EndID"],
                    valid["ShotID"],
                )
            )
            remainder = valid.iloc[0:0].copy()
            full = merged[
                np.isfinite(merged["est_speed"])
                & np.isfinite(merged["est_angle"])
                & np.isfinite(merged["est_spin"])
                & np.isfinite(merged["est_y0"])
            ].copy()
            full["abs_dv_obs"] = full["dv_obs"].abs()
            full = full.sort_values(
                ["abs_dv_obs", "CompetitionID", "GameID", "EndID", "ShotID"],
                ascending=[False, True, True, True, True],
            )
            mask = ~full.apply(
                lambda r: (
                    r["CompetitionID"],
                    r["SessionID"],
                    r["GameID"],
                    r["EndID"],
                    r["ShotID"],
                )
                in chosen_ids,
                axis=1,
            )
            remainder = full[mask].head(MAX_SCENARIOS - len(valid))
            valid = pd.concat([valid, remainder], ignore_index=True)
        valid = valid.head(MAX_SCENARIOS).copy()

        rows: list[dict[str, Any]] = []
        for _, row in valid.iterrows():
            pre_board = _sanitize_board(_board_from_row(row, prefix="prev_stone"))
            post_board = _board_from_row(row, prefix="stone")
            scenario_id = f"{int(row.CompetitionID)}-{int(row.SessionID)}-{int(row.GameID)}-{int(row.EndID)}-{int(row.ShotID)}"
            rows.append(
                {
                    "id": scenario_id,
                    "competition_id": int(row.CompetitionID),
                    "session_id": int(row.SessionID),
                    "game_id": int(row.GameID),
                    "end_id": int(row.EndID),
                    "shot_id": int(row.ShotID),
                    "athlete_name": str(row.player_name),
                    "athlete_label": str(row.player_label),
                    "team_name": str(row.team_name),
                    "task": int(row.Task),
                    "handle": int(row.Handle),
                    "team_order": float(row.team_order),
                    "stone_block": float(row.stone_block),
                    "shot_norm_prev": float(row.shot_norm_prev),
                    "shot_norm_next": float(row.shot_norm_next),
                    "v_prev": float(row.v_prev),
                    "v_next_observed": float(row.v_next),
                    "athlete_dv": float(row.dv_obs),
                    "model_path": str(self.model_paths[int(row.CompetitionID)]),
                    "defaults": {
                        "speed": float(row.est_speed),
                        "angle": float(row.est_angle),
                        "spin": float(row.est_spin),
                        "y0": float(row.est_y0),
                    },
                    "pre_board_m": pre_board,
                    "post_board_m": post_board,
                }
            )
        return rows

    def scenario_payload(self, scenario: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": scenario["id"],
            "name": f"{scenario['athlete_label']} | End {scenario['end_id']} Shot {scenario['shot_id']}",
            "description": f"Observed {scenario['athlete_name']} shot. Task {scenario['task']}, handle {scenario['handle']}.",
            "athlete_name": scenario["athlete_name"],
            "athlete_label": scenario["athlete_label"],
            "athlete_value_diff": scenario["athlete_dv"],
            "pre_value": scenario["v_prev"],
            "observed_post_value": scenario["v_next_observed"],
            "team_order": scenario["team_order"],
            "stone_block": scenario["stone_block"],
            "shot_norm_prev": scenario["shot_norm_prev"],
            "shot_norm_next": scenario["shot_norm_next"],
            "defaults": scenario["defaults"],
            "pre_board": _board_to_client(scenario["pre_board_m"]),
            "post_board_observed": _board_to_client(scenario["post_board_m"]),
        }

    def _simulate_board(self, scenario: dict[str, Any], params: np.ndarray) -> tuple[np.ndarray, list[int], int, np.ndarray]:
        pre_board = _sanitize_board(scenario["pre_board_m"])
        prev_slots = _occupied_slots(pre_board)
        prev_compact = pre_board[prev_slots].astype(np.float32)
        new_slot = _next_slot_for_block(pre_board, scenario["stone_block"])
        traj = simulate_from_params(
            self.sim_params,
            jax.device_put(prev_compact),
            jax.device_put(params.astype(np.float32)),
            dynamic=True,
        )
        traj_np = np.asarray(jax.device_get(traj), dtype=np.float32)
        final_board = _slotted_board_from_compact(traj_np[-1], prev_slots, new_slot)
        return final_board, prev_slots, new_slot, traj_np

    def _state_value(self, board: np.ndarray, competition_id: int, team_order: float, stone_block: float, shot_norm: float) -> float:
        raw_defaults = np.full((12, 2), POS_MAX, dtype=np.float32)
        c_vec = np.array([shot_norm, team_order, stone_block], dtype=np.float32)
        model_fn, _ = _load_model_for_competition(int(competition_id))
        return float(
            evaluate_state_value(
                model_fn,
                board,
                raw_defaults,
                c_vec,
                float(stone_block),
                float("nan"),
                float("nan"),
                float(shot_norm),
                use_rule_based_terminal=True,
            )
        )

    def preview(self, req: ShotParamsRequest) -> dict[str, Any]:
        scenario = self.scenarios.get(req.scenario_id)
        if scenario is None:
            raise HTTPException(status_code=404, detail="Unknown scenario.")
        intended = np.array([req.speed, req.angle, req.spin, req.y0], dtype=np.float32)
        final_board, prev_slots, new_slot, traj_np = self._simulate_board(scenario, intended)
        post_value = self._state_value(
            final_board,
            scenario["competition_id"],
            scenario["team_order"],
            scenario["stone_block"],
            scenario["shot_norm_next"],
        )
        return {
            "scenario_id": scenario["id"],
            "intended_params": {
                "speed": float(intended[0]),
                "angle": float(intended[1]),
                "spin": float(intended[2]),
                "y0": float(intended[3]),
            },
            "pre_value": float(scenario["v_prev"]),
            "intended_post_value": float(post_value),
            "intended_value_diff": float(post_value - scenario["v_prev"]),
            "intended_trajectory": _sample_trajectory_frames(traj_np, prev_slots, new_slot),
            "intended_final_board": _board_to_client(final_board),
        }

    def play(self, req: PlayShotRequest) -> dict[str, Any]:
        scenario = self.scenarios.get(req.scenario_id)
        if scenario is None:
            raise HTTPException(status_code=404, detail="Unknown scenario.")

        intended = np.array([req.speed, req.angle, req.spin, req.y0], dtype=np.float32)
        rng = np.random.default_rng(req.seed if req.seed is not None else random.randrange(1 << 30))
        noisy = _sample_noisy_params(intended, self.noise, rng)

        intended_final, prev_slots, new_slot, intended_traj = self._simulate_board(scenario, intended)
        final_board, _, _, traj_np = self._simulate_board(scenario, noisy)

        post_value = self._state_value(
            final_board,
            scenario["competition_id"],
            scenario["team_order"],
            scenario["stone_block"],
            scenario["shot_norm_next"],
        )
        pre_value = float(scenario["v_prev"])
        your_dv = float(post_value - pre_value)
        terminal = bool(scenario["shot_norm_next"] >= 1.0 - 1e-6)

        return {
            "scenario_id": scenario["id"],
            "athlete_name": scenario["athlete_name"],
            "athlete_label": scenario["athlete_label"],
            "athlete_value_diff": float(scenario["athlete_dv"]),
            "pre_value": pre_value,
            "your_post_value": float(post_value),
            "your_value_diff": your_dv,
            "observed_post_value": float(scenario["v_next_observed"]),
            "observed_value_diff": float(scenario["athlete_dv"]),
            "terminal": terminal,
            "intended_params": {
                "speed": float(intended[0]),
                "angle": float(intended[1]),
                "spin": float(intended[2]),
                "y0": float(intended[3]),
            },
            "sampled_params": {
                "speed": float(noisy[0]),
                "angle": float(noisy[1]),
                "spin": float(noisy[2]),
                "y0": float(noisy[3]),
            },
            "final_board": _board_to_client(final_board),
            "intended_final_board": _board_to_client(intended_final),
            "trajectory": _sample_trajectory_frames(traj_np, prev_slots, new_slot),
            "intended_trajectory": _sample_trajectory_frames(intended_traj, prev_slots, new_slot),
            "model_path": str(scenario["model_path"]),
        }


engine = GameEngine()
app = FastAPI(title="Mixed Doubles Curling Game")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "model_arch": "graph_transformer",
        "device": DEFAULT_DEVICE,
        "scenario_count": len(engine.scenario_rows),
        "holdout_models": {str(k): str(v) for k, v in sorted(engine.model_paths.items())},
    }


@app.get("/api/scenarios")
def scenarios() -> dict[str, Any]:
    return {"scenarios": [engine.scenario_payload(s) for s in engine.scenario_rows]}


@app.post("/api/preview")
def preview(req: ShotParamsRequest) -> dict[str, Any]:
    return engine.preview(req)


@app.post("/api/play")
def play(req: PlayShotRequest) -> dict[str, Any]:
    return engine.play(req)


@app.get("/")
def root() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")
