from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

SHOT_KEY = ["CompetitionID", "SessionID", "GameID", "EndID", "ShotID"]
END_KEY = ["CompetitionID", "SessionID", "GameID", "EndID"]
COMPETITION_KEY = ["CompetitionID"]


def make_train_val_test_indices(
    n: int,
    val_split: float,
    test_split: float,
    seed: int,
    group_keys: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split indices into train / val / test.

    If *group_keys* is provided (shape ``(n, K)``), the split is done at the
    group level so that all rows belonging to the same group land in the same
    split.  This prevents data leakage when rows within a group share labels
    (e.g. all shots in the same end have the same ValueDiff).

    When *group_keys* is ``None``, falls back to a plain row-level random
    permutation (legacy behaviour).
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    if not (0.0 <= float(val_split) < 1.0):
        raise ValueError(f"val_split must be in [0,1); got {val_split}")
    if not (0.0 <= float(test_split) < 1.0):
        raise ValueError(f"test_split must be in [0,1); got {test_split}")

    g = torch.Generator().manual_seed(int(seed))

    if group_keys is not None:
        # ------ group-level split ------
        group_keys = np.asarray(group_keys)
        assert group_keys.shape[0] == n, (
            f"group_keys rows ({group_keys.shape[0]}) must match n ({n})"
        )
        # Map each row to a unique group id
        _, inverse = np.unique(group_keys, axis=0, return_inverse=True)
        n_groups = int(inverse.max() + 1) if n > 0 else 0

        perm = torch.randperm(n_groups, generator=g).cpu().numpy().astype(np.int64)

        test_size = int(n_groups * float(test_split))
        test_groups = set(perm[:test_size].tolist())
        remaining = perm[test_size:]

        val_size = int(len(remaining) * float(val_split))
        val_groups = set(remaining[:val_size].tolist())
        train_groups = set(remaining[val_size:].tolist())

        train_idx = np.where(np.isin(inverse, list(train_groups)))[0].astype(np.int64)
        val_idx = np.where(np.isin(inverse, list(val_groups)))[0].astype(np.int64)
        test_idx = np.where(np.isin(inverse, list(test_groups)))[0].astype(np.int64)
    else:
        # ------ row-level split (legacy) ------
        perm = torch.randperm(n, generator=g).cpu().numpy().astype(np.int64)

        test_size = int(n * float(test_split))
        test_idx = perm[:test_size]
        train_val_idx = perm[test_size:]

        val_size = int(len(train_val_idx) * float(val_split))
        val_idx = train_val_idx[:val_size]
        train_idx = train_val_idx[val_size:]

    if len(train_idx) == 0:
        raise ValueError(
            f"Empty training split with n={n}, val_split={val_split}, test_split={test_split}. "
            "Reduce val/test fractions."
        )

    return train_idx, val_idx, test_idx


def write_test_shot_keys(
    dataset_df: pd.DataFrame,
    test_indices: np.ndarray,
    out_path: str,
) -> tuple[Path | None, int]:
    if not out_path:
        return None, 0
    if len(test_indices) == 0:
        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=SHOT_KEY).to_csv(path, index=False)
        return path.resolve(), 0

    missing = [c for c in SHOT_KEY if c not in dataset_df.columns]
    if missing:
        raise ValueError(f"Cannot write test shot keys; dataset is missing columns: {missing}")

    keys = dataset_df.iloc[np.asarray(test_indices, dtype=np.int64)][SHOT_KEY].copy()
    for c in SHOT_KEY:
        keys[c] = pd.to_numeric(keys[c], errors="coerce").astype("Int64")
    keys = keys.dropna(subset=SHOT_KEY).astype({c: "int64" for c in SHOT_KEY})
    keys = keys.drop_duplicates(subset=SHOT_KEY).sort_values(SHOT_KEY).reset_index(drop=True)

    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys.to_csv(path, index=False)
    return path.resolve(), int(len(keys))


def write_split_competition_ids(
    dataset_df: pd.DataFrame,
    split_indices: np.ndarray,
    out_path: str,
) -> tuple[Path | None, int]:
    if not out_path:
        return None, 0

    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if len(split_indices) == 0:
        pd.DataFrame(columns=COMPETITION_KEY).to_csv(path, index=False)
        return path.resolve(), 0

    missing = [c for c in COMPETITION_KEY if c not in dataset_df.columns]
    if missing:
        raise ValueError(f"Cannot write competition ids; dataset is missing columns: {missing}")

    comps = dataset_df.iloc[np.asarray(split_indices, dtype=np.int64)][COMPETITION_KEY].copy()
    for c in COMPETITION_KEY:
        comps[c] = pd.to_numeric(comps[c], errors="coerce").astype("Int64")
    comps = comps.dropna(subset=COMPETITION_KEY).astype({c: "int64" for c in COMPETITION_KEY})
    comps = comps.drop_duplicates(subset=COMPETITION_KEY).sort_values(COMPETITION_KEY).reset_index(drop=True)
    comps.to_csv(path, index=False)
    return path.resolve(), int(len(comps))
