import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

# Constants
POS_MAX = 4095.0  # sentinel and upper bound
MAX_ENDS = 8
NUM_STONES = 12
# Horizontal flip center: CSV x ranges 0..1500, center at 750
FLIP_CENTER_X = 1500.0


def _compute_end_context(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds:
      - ShotIndex, ShotsInEnd, shot_norm
      - team_order (0=throws first in end, 1=other team)
    All computed within (CompetitionID,SessionID,GameID,EndID).

    Note: is_hammer was removed because it is always identical to team_order
    (the team that throws second/last has hammer). Keeping both allowed the
    value model to learn non-zero-sum predictions, which is incorrect for a
    zero-sum game.
    """
    group_cols = ["CompetitionID", "SessionID", "GameID", "EndID"]
    df = df.sort_values(group_cols + ["ShotID"]).reset_index(drop=True)

    # ShotIndex / shot_norm
    df["ShotIndex"] = df.groupby(group_cols).cumcount()
    df["ShotsInEnd"] = df.groupby(group_cols)["ShotID"].transform("count")
    df["shot_norm"] = 0.0
    mask = df["ShotsInEnd"] > 1
    df.loc[mask, "shot_norm"] = df.loc[mask, "ShotIndex"] / (df.loc[mask, "ShotsInEnd"] - 1.0)

    # First team per end (based on ShotID ordering)
    first_team = df.groupby(group_cols)["TeamID"].transform("first")
    df["team_order"] = (df["TeamID"] != first_team).astype(np.float32)  # first=0, other=1

    return df


def _has_complete_precomputed_context(df: pd.DataFrame) -> bool:
    cols = ["shot_norm", "team_order"]
    return all(c in df.columns for c in cols) and df[cols].notna().all().all()


def _has_complete_precomputed_block(df: pd.DataFrame) -> bool:
    return "stone_block" in df.columns and df["stone_block"].notna().all()


def _is_in_play(x: float, y: float) -> bool:
    """A stone is in play if it has been thrown and is not dead."""
    return (x > 0 or y > 0) and (x < POS_MAX) and (y < POS_MAX)


def _detect_team_block(df: pd.DataFrame, stone_cols: list) -> pd.DataFrame:
    """
    Detect which stone block (0=slots 1-6, 1=slots 7-12) each TeamID owns
    within each end by observing which slots get newly populated between
    consecutive shots.

    Adds column 'stone_block' (float32): 0.0 or 1.0.
    """
    group_cols = ["CompetitionID", "SessionID", "GameID", "EndID"]
    df = df.sort_values(group_cols + ["ShotID"]).reset_index(drop=True)

    # Build (N, 12) boolean: is stone_i in play?
    present = np.zeros((len(df), NUM_STONES), dtype=bool)
    for i in range(NUM_STONES):
        x = df[f"stone_{i+1}_x"].values.astype(float)
        y = df[f"stone_{i+1}_y"].values.astype(float)
        present[:, i] = ((x > 0) | (y > 0)) & (x < POS_MAX) & (y < POS_MAX)

    # Compute previous-row presence (within same end group)
    prev_present = np.zeros_like(present)
    if len(df) > 1:
        prev_present[1:] = present[:-1]
    # Reset at end-group boundaries
    end_vals = df[group_cols].values
    if len(df) > 1:
        boundaries = np.any(end_vals[1:] != end_vals[:-1], axis=1)
        boundary_idx = np.where(boundaries)[0] + 1
        prev_present[boundary_idx] = False

    # Newly added stones per row
    newly_added = present & ~prev_present  # (N, 12)
    new_count = newly_added.sum(axis=1)

    # Where exactly one stone was added, record its slot (0-indexed)
    single_mask = new_count == 1
    new_slot = np.full(len(df), np.nan)
    if single_mask.any():
        new_slot[single_mask] = np.argmax(newly_added[single_mask], axis=1).astype(float)

    # Block: 0 if slot 0-5 (stones 1-6), 1 if slot 6-11 (stones 7-12)
    df["_throw_block"] = np.where(
        np.isfinite(new_slot), (new_slot >= 6).astype(float), np.nan
    )

    # Take mode per (end, TeamID)
    merge_keys = group_cols + ["TeamID"]
    valid = df[np.isfinite(df["_throw_block"])].copy()
    if not valid.empty:
        block_mode = (
            valid.groupby(merge_keys, dropna=False)["_throw_block"]
            .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan)
            .reset_index()
            .rename(columns={"_throw_block": "stone_block"})
        )
        df = pd.merge(df, block_mode, on=merge_keys, how="left")
    else:
        df["stone_block"] = np.nan

    df = df.drop(columns=["_throw_block"], errors="ignore")
    df["stone_block"] = df["stone_block"].fillna(0.0).astype(np.float32)

    return df


class ValueDataset(Dataset):
    """
    Builds (state, condition) -> value samples from Stones.csv and Ends.csv.

    Label:
      y = Result_team - Result_opponent in that end
        (implemented as 2*Result - sum(Result) over both teams)

    Condition c (cond_dim=3):
      c = [shot_norm, team_order, stone_block]
        - shot_norm: 0..1 within the end (by ShotID order)
        - team_order: 0 if this TeamID throws first in end, else 1 (= has hammer)
        - stone_block: 0 if this TeamID owns stones 1-6, 1 if 7-12

    Augmentation:
      - shuffles stones within each team block (1-6 vs 7-12),
        among stones that are actually in play (not at 0,0 or 4095,4095).
      - optional horizontal (x-wise) flip with 50% probability.
    """

    def __init__(
        self,
        stones_csv_path,
        ends_csv_path,
        normalize=True,
        max_ends=MAX_ENDS,
        min_shots_per_end=1,
        augment_positions=True,
        augment_flip=False,
    ):
        self.stones_csv_path = stones_csv_path
        self.ends_csv_path = ends_csv_path
        self.normalize = normalize
        self.max_ends = max_ends
        self.augment_positions = augment_positions
        self.augment_flip = augment_flip

        # -------- Load Stones --------
        df_s = pd.read_csv(stones_csv_path)

        # Stone position columns
        self.stone_cols = []
        for i in range(1, NUM_STONES + 1):
            self.stone_cols.append(f"stone_{i}_x")
            self.stone_cols.append(f"stone_{i}_y")

        stones_critical = [
            "CompetitionID",
            "SessionID",
            "GameID",
            "EndID",
            "ShotID",
            "TeamID",
            "Task",
            "Handle",
        ] + self.stone_cols
        missing_stones_crit = [c for c in stones_critical if c not in df_s.columns]
        if missing_stones_crit:
            raise ValueError(f"Stones CSV is missing columns: {missing_stones_crit}")

        # Drop rows with NaNs in critical columns (zeros and 4095 are fine)
        df_s = df_s.dropna(subset=stones_critical).reset_index(drop=True)

        # Compute per-end context features from Stones ordering unless already provided.
        if _has_complete_precomputed_context(df_s):
            for col in ("shot_norm", "team_order"):
                df_s[col] = df_s[col].astype(np.float32)
        else:
            df_s = _compute_end_context(df_s)

        # Detect stone block ownership per (end, team) unless already provided.
        if _has_complete_precomputed_block(df_s):
            df_s["stone_block"] = df_s["stone_block"].astype(np.float32)
        else:
            df_s = _detect_team_block(df_s, self.stone_cols)

        # -------- Load Ends --------
        df_e = pd.read_csv(ends_csv_path)

        ends_critical = [
            "CompetitionID",
            "SessionID",
            "GameID",
            "TeamID",
            "EndID",
            "Result",
            "PowerPlay",
        ]
        missing_ends_crit = [c for c in ends_critical if c not in df_e.columns]
        if missing_ends_crit:
            raise ValueError(f"Ends CSV is missing columns: {missing_ends_crit}")

        df_e = df_e.dropna(subset=["Result"]).reset_index(drop=True)
        df_e["Result"] = df_e["Result"].astype(float)

        # -------- Convert Result -> score differential per team --------
        merge_keys = ["CompetitionID", "SessionID", "GameID", "EndID", "TeamID"]
        end_keys_no_team = ["CompetitionID", "SessionID", "GameID", "EndID"]

        df_e["TotalResultInEnd"] = df_e.groupby(end_keys_no_team)["Result"].transform("sum")
        df_e["ValueDiff"] = 2.0 * df_e["Result"] - df_e["TotalResultInEnd"]

        # -------- Merge Stones with Ends (attach differential value) --------
        df = pd.merge(
            df_s,
            df_e[merge_keys + ["ValueDiff"]],
            on=merge_keys,
            how="inner",
        )

        df = df.sort_values(
            ["CompetitionID", "SessionID", "GameID", "EndID", "ShotID"]
        ).reset_index(drop=True)

        # Determine number of tasks (from Stones)
        self.num_tasks = int(df["Task"].max()) + 1 if len(df) else 0

        # Regression target
        df["value_target"] = df["ValueDiff"].astype(float)

        self.df = df.reset_index(drop=True)
        self.pos_dim = NUM_STONES * 2

        # Condition is [shot_norm, team_order, stone_block]
        self.cond_dim = 3
        self.input_dim = self.pos_dim
        self.output_dim = 1  # scalar value

    def __len__(self):
        return len(self.df)

    def _augment_shuffle(self, raw_vals: np.ndarray) -> np.ndarray:
        """Shuffle stones within each team block among in-play slots only."""
        mat = raw_vals.reshape(NUM_STONES, 2).copy()

        # Team A: stones 1-6; Team B: stones 7-12 (dataset convention)
        for start in (0, 6):
            idxs = np.arange(start, start + 6)
            coords = mat[idxs]

            # "in play" if at least one coord > 0 AND both < POS_MAX
            in_play = np.any(coords > 0, axis=1) & np.all(coords < POS_MAX, axis=1)
            play_idxs = idxs[in_play]

            if len(play_idxs) > 1:
                shuffled_local = np.random.permutation(len(play_idxs))
                original_vals = mat[play_idxs].copy()
                mat[play_idxs] = original_vals[shuffled_local]

        return mat.reshape(-1)

    def _augment_flip_x(self, raw_vals: np.ndarray) -> np.ndarray:
        """Flip all in-play stones horizontally (x -> FLIP_CENTER_X - x)."""
        mat = raw_vals.reshape(NUM_STONES, 2).copy()
        for i in range(NUM_STONES):
            x, y = mat[i]
            if _is_in_play(float(x), float(y)):
                mat[i, 0] = FLIP_CENTER_X - x
        return mat.reshape(-1)

    # Keep old name as alias for backwards compatibility
    _augment_positions = _augment_shuffle

    def _extract_positions(self, row: pd.Series) -> np.ndarray:
        raw_vals = row[self.stone_cols].to_numpy(dtype=np.float32)

        if self.augment_positions:
            raw_vals = self._augment_shuffle(raw_vals)

        if self.augment_flip and np.random.random() < 0.5:
            raw_vals = self._augment_flip_x(raw_vals)

        if self.normalize:
            return (raw_vals / POS_MAX).astype(np.float32)
        return raw_vals.astype(np.float32)

    def _make_condition(self, row: pd.Series) -> np.ndarray:
        shot_norm = float(row["shot_norm"])
        team_order = float(row["team_order"])
        stone_block = float(row.get("stone_block", 0.0))
        return np.array([shot_norm, team_order, stone_block], dtype=np.float32)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        x = self._extract_positions(row)       # (24,)
        c = self._make_condition(row)          # (4,)
        y = np.array([row["value_target"]], dtype=np.float32)

        return (
            torch.from_numpy(x).float(),
            torch.from_numpy(c).float(),
            torch.from_numpy(y).float(),
        )


def denormalize_positions(pos_vec, normalize=True):
    arr = np.asarray(pos_vec, dtype=np.float32)
    if normalize:
        arr = arr * POS_MAX
    return arr


def positions_to_matrix(pos_vec):
    arr = np.asarray(pos_vec, dtype=np.float32)
    assert arr.size == NUM_STONES * 2
    return arr.reshape(NUM_STONES, 2)
