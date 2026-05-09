#!/usr/bin/env python3
"""
coach_report_mc.py

Coach-facing Monte Carlo (MC) decision + execution analysis using two precomputed scorings:
  - shot_scores_local.csv  : "local" neighborhood around the inverse solution (Gaussian local perturbations)
  - shot_scores.csv        : "global" neighborhood across feasible throws (uniform/global sampling)

This script produces:
  - summary.md
  - figures/*.png (seaborn; professional styling)

Core concepts (computed per shot):
  1) Decision Value:
       How good the chosen local neighborhood is compared to global alternatives.
       decision_value = dv_mean_local - dv_mean_global
     Interpretation:
       + positive => the planned/selected solution neighborhood looks better than global random feasible throws
       + negative => global search suggests better options exist than the local neighborhood around the chosen solution

  2) Decision Risk:
       How variable the local neighborhood is (uncertainty/sensitivity around the plan).
       decision_risk = dv_std_local
     (Optionally you can compare against global variability; we also report dv_std_global.)

  3) Execution Value:
       How good the executed outcome was vs the local neighborhood distribution.
       execution_value = dv_obs - dv_mean_local   (a.k.a. "excess_local")
       execution_percentile_local = percentile_obs_local (already provided by scorer)
       execution_z_local = z_obs_local (already provided by scorer)

Inputs:
  - shot_scores_local.csv (from score_shots_mc.py using mode=local or gaussian local)
  - shot_scores.csv       (from score_shots_mc.py using mode=uniform/global sampling)
  - Stones.csv            (for Points, Task, Handle; also usable for IDs)
  - Competitors.csv / Teams.csv / competitions.csv / games.csv (optional; for names)

Usage:
  python coach_report_mc.py \
    --scores_local shot_scores_local.csv \
    --scores_global shot_scores.csv \
    --stones-csv 2026/Stones.csv \
    --competitors-csv 2026/Competitors.csv \
    --teams-csv 2026/Teams.csv \
    --competitions-csv 2026/competitions.csv \
    --games-csv 2026/games.csv \
    --out-dir coach_report_mc

Notes:
  - Assumes both score files share SHOT_KEY columns:
      CompetitionID, SessionID, GameID, EndID, ShotID
  - Assumes both include dv_obs, dv_mean, dv_std, percentile_obs, z_obs (as produced by your scorer).
  - If your local/global files have different names for hard_loss (e.g. hard_loss or hard_loss_refine),
    this script uses the "hard_loss" column if present.

"""

from __future__ import annotations

import argparse
import pathlib
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


SHOT_KEY = ["CompetitionID", "SessionID", "GameID", "EndID", "ShotID"]

TASK_NAME: Dict[int, str] = {
    0: "Draw",
    1: "Front",
    2: "Guard",
    3: "Raise / Tap-back",
    4: "Wick / Soft Peeling",
    5: "Freeze",
    6: "Take-out",
    7: "Hit and Roll",
    8: "Clearing",
    9: "Double Take-out",
    10: "Promotion Take-out",
    11: "Through",
    13: "No statistics",
}

HANDLE_NAME: Dict[int, str] = {
    0: "Handle 0",
    1: "Handle 1",
    -1: "Unknown",
}


# ----------------------------
# I/O + naming helpers
# ----------------------------
def _ensure_dir(path: pathlib.Path):
    path.mkdir(parents=True, exist_ok=True)


def _load_csv(path: str) -> pd.DataFrame:
    p = pathlib.Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    return pd.read_csv(p)


def _task_label(x) -> str:
    try:
        t = int(x)
        return TASK_NAME.get(t, f"Task {t}")
    except Exception:
        return "Task (unknown)"


def _handle_label(x) -> str:
    try:
        h = int(x)
        return HANDLE_NAME.get(h, f"Handle {h}")
    except Exception:
        return "Handle (unknown)"


def _load_player_names(competitors_csv: str) -> Dict[Tuple[int, int, int], str]:
    """
    Competitors.csv lacks PlayerID; infer PlayerID within each competition/team by
    order of appearance.
    Returns: (CompetitionID, TeamID, PlayerID) -> Reportingname
    """
    p = pathlib.Path(competitors_csv)
    if not p.exists():
        return {}
    df = pd.read_csv(p)
    if (
        "CompetitionID" not in df.columns
        or "TeamID" not in df.columns
        or "Reportingname" not in df.columns
    ):
        return {}
    df = df.copy()
    df["player_ord"] = df.groupby(["CompetitionID", "TeamID"]).cumcount() + 1
    return {
        (int(r.CompetitionID), int(r.TeamID), int(r.player_ord)): str(r.Reportingname)
        for _, r in df.iterrows()
        if pd.notna(r.CompetitionID) and pd.notna(r.TeamID) and pd.notna(r.player_ord)
    }


def _load_team_names(teams_csv: str) -> Dict[Tuple[int, int], str]:
    p = pathlib.Path(teams_csv)
    if not p.exists():
        return {}
    df = pd.read_csv(p)
    if "CompetitionID" not in df.columns or "TeamID" not in df.columns:
        return {}
    name_col = "Name" if "Name" in df.columns else None
    if name_col is None:
        return {
            (int(r.CompetitionID), int(r.TeamID)): f"Team {int(r.TeamID)}"
            for _, r in df.iterrows()
            if pd.notna(r.CompetitionID) and pd.notna(r.TeamID)
        }
    return {
        (int(r.CompetitionID), int(r.TeamID)): str(r[name_col])
        for _, r in df.iterrows()
        if pd.notna(r.CompetitionID) and pd.notna(r.TeamID)
    }


def _load_competition_names(competitions_csv: str) -> Dict[int, str]:
    p = pathlib.Path(competitions_csv)
    if not p.exists():
        return {}
    df = pd.read_csv(p)
    if "CompetitionID" not in df.columns:
        return {}
    name_col = "CompetitionName" if "CompetitionName" in df.columns else None
    if name_col is None:
        return {int(r.CompetitionID): f"Competition {int(r.CompetitionID)}" for _, r in df.iterrows() if pd.notna(r.CompetitionID)}
    return {int(r.CompetitionID): str(r[name_col]) for _, r in df.iterrows() if pd.notna(r.CompetitionID)}


def _load_game_labels(games_csv: str) -> Dict[Tuple[int, int, int], str]:
    p = pathlib.Path(games_csv)
    if not p.exists():
        return {}
    df = pd.read_csv(p)
    needed = {"CompetitionID", "SessionID", "GameID"}
    if not needed.issubset(set(df.columns)):
        return {}
    noc1 = "NOC1" if "NOC1" in df.columns else None
    noc2 = "NOC2" if "NOC2" in df.columns else None
    sheet = "Sheet" if "Sheet" in df.columns else None

    out: Dict[Tuple[int, int, int], str] = {}
    for _, r in df.iterrows():
        if pd.isna(r.CompetitionID) or pd.isna(r.SessionID) or pd.isna(r.GameID):
            continue
        c, s, g = int(r.CompetitionID), int(r.SessionID), int(r.GameID)
        a = str(r[noc1]) if noc1 else "NOC1"
        b = str(r[noc2]) if noc2 else "NOC2"
        lbl = f"{a} vs {b}"
        if sheet and pd.notna(r[sheet]):
            lbl += f" (Sheet {str(r[sheet])})"
        out[(c, s, g)] = lbl
    return out


def _set_plot_style():
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


# ----------------------------
# Data preparation
# ----------------------------
def _require_cols(df: pd.DataFrame, cols: list[str], name: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise SystemExit(f"[error] {name} missing columns: {missing}")


def _standardize_hard_loss(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure there's a 'hard_loss' column if any plausible source exists.
    """
    out = df.copy()
    if "hard_loss" in out.columns:
        return out
    # fallback to common variants
    for alt in ["hard_loss_refine", "HardLoss", "hardLoss"]:
        if alt in out.columns:
            out["hard_loss"] = out[alt]
            return out
    out["hard_loss"] = np.nan
    return out


def load_and_merge(
    local_path: str,
    global_path: str,
    stones_csv: str,
) -> pd.DataFrame:
    local = _standardize_hard_loss(_load_csv(local_path))
    glob = _standardize_hard_loss(_load_csv(global_path))

    # required columns from scorer
    base_cols = SHOT_KEY + ["dv_obs", "dv_mean", "dv_std"]
    _require_cols(local, base_cols, "local scores")
    _require_cols(glob, base_cols, "global scores")

    # Optional columns if present
    opt_cols = [
        "dv_p10",
        "dv_p50",
        "dv_p90",
        "percentile_obs",
        "z_obs",
        "sample_count",
        "prev_N",
        "Task",
        "Handle",
        "TeamID",
        "PlayerID",
        "hard_loss",
    ]
    local_keep = [c for c in (SHOT_KEY + base_cols[5:] + opt_cols) if c in local.columns]
    glob_keep = [c for c in (SHOT_KEY + base_cols[5:] + opt_cols) if c in glob.columns]

    local = local[sorted(set(local_keep), key=local_keep.index)].copy()
    glob = glob[sorted(set(glob_keep), key=glob_keep.index)].copy()

    # suffix and merge
    local = local.add_suffix("_local")
    glob = glob.add_suffix("_global")

    # restore key names for merge
    for k in SHOT_KEY:
        local.rename(columns={f"{k}_local": k}, inplace=True)
        glob.rename(columns={f"{k}_global": k}, inplace=True)

    df = local.merge(glob, on=SHOT_KEY, how="inner", validate="one_to_one")

    # Bring in Stones columns (Points + Task/Handle/IDs as canonical)
    stones_p = pathlib.Path(stones_csv)
    if not stones_p.exists():
        raise SystemExit(f"[error] stones-csv not found: {stones_p}")
    stones = pd.read_csv(stones_p)

    # minimal enrichment columns (only those present)
    enrich = [c for c in (SHOT_KEY + ["CompetitionID", "SessionID", "GameID", "EndID", "ShotID", "TeamID", "PlayerID", "Task", "Handle", "Points"]) if c in stones.columns]
    enrich = list(dict.fromkeys(enrich))  # keep order, dedupe

    stones_small = stones[enrich].copy()
    df = df.merge(stones_small, on=SHOT_KEY, how="left")

    # Enforce human Points domain: only keep 0–4, drop everything else
    if "Points" in df.columns:
        df["Points"] = pd.to_numeric(df["Points"], errors="coerce")
        df.loc[~df["Points"].between(0, 4, inclusive="both"), "Points"] = np.nan


    # Fill missing ids/task/handle from local/global if Stones missing those values
    for c in ["TeamID", "PlayerID", "Task", "Handle"]:
        if c not in df.columns:
            # if Stones is missing column entirely, prefer local then global if present
            if f"{c}_local" in df.columns:
                df[c] = df[f"{c}_local"]
            elif f"{c}_global" in df.columns:
                df[c] = df[f"{c}_global"]
        else:
            # if column exists but is null, fill from local/global
            if f"{c}_local" in df.columns:
                df[c] = df[c].where(df[c].notna(), df[f"{c}_local"])
            if f"{c}_global" in df.columns:
                df[c] = df[c].where(df[c].notna(), df[f"{c}_global"])

    # Derived metrics
    df["decision_value"] = df["dv_mean_local"].astype(float) - df["dv_mean_global"].astype(float)
    df["decision_risk"] = df["dv_std_local"].astype(float)
    df["decision_risk_global"] = df["dv_std_global"].astype(float)

    df["execution_value_local"] = df["dv_obs_local"].astype(float) - df["dv_mean_local"].astype(float)
    df["execution_value_global"] = df["dv_obs_global"].astype(float) - df["dv_mean_global"].astype(float)
    if "dv_p50_local" in df.columns:
        df["execution_value_p50_local"] = df["dv_obs_local"].astype(float) - df["dv_p50_local"].astype(float)
    if "dv_p90_local" in df.columns:
        df["execution_value_p90_local"] = df["dv_obs_local"].astype(float) - df["dv_p90_local"].astype(float)

    # Prefer the local dv_obs as "observed"; they should be identical if scoring used same value model
    df["dv_obs"] = df["dv_obs_local"].astype(float)

    # If percentiles exist, keep them
    if "percentile_obs_local" in df.columns:
        df["execution_percentile_local"] = df["percentile_obs_local"].astype(float)
    if "z_obs_local" in df.columns:
        df["execution_z_local"] = df["z_obs_local"].astype(float)

    return df


# ----------------------------
# Label enrichment
# ----------------------------
def add_labels(
    df: pd.DataFrame,
    competitors_csv: str,
    teams_csv: str,
    competitions_csv: str,
    games_csv: str,
) -> pd.DataFrame:
    out = df.copy()
    player_names = _load_player_names(competitors_csv)
    team_names = _load_team_names(teams_csv)
    comp_names = _load_competition_names(competitions_csv)
    game_labels = _load_game_labels(games_csv)
    default_competition_id = None
    if "CompetitionID" in out.columns:
        comp_ids = out["CompetitionID"].dropna().astype(int).unique().tolist()
        if len(comp_ids) == 1:
            default_competition_id = comp_ids[0]

    out["task_name"] = out["Task"].apply(_task_label) if "Task" in out.columns else "Task (missing)"
    out["handle_name"] = out["Handle"].apply(_handle_label) if "Handle" in out.columns else "Handle (missing)"

    if "TeamID" in out.columns:
        def _tname(row) -> str:
            try:
                tid = int(row["TeamID"])
                comp_id = int(row["CompetitionID"]) if pd.notna(row.get("CompetitionID")) else default_competition_id
                if comp_id is None:
                    return f"Team {tid}"
                return team_names.get((comp_id, tid), f"Team {tid}")
            except Exception:
                return ""
        out["team_name"] = out.apply(_tname, axis=1)
    else:
        out["team_name"] = ""

    if "TeamID" in out.columns and "PlayerID" in out.columns:
        def _pname(row) -> str:
            try:
                comp_id = int(row["CompetitionID"]) if pd.notna(row.get("CompetitionID")) else default_competition_id
                tid = int(row["TeamID"])
                pid = int(row["PlayerID"])
                if comp_id is None:
                    return f"Player {pid}"
                nm = player_names.get((comp_id, tid, pid), "")
                return nm if nm else f"Player {pid}"
            except Exception:
                return ""
        out["player_name"] = out.apply(_pname, axis=1)
    else:
        out["player_name"] = ""

    if "CompetitionID" in out.columns:
        out["competition_name"] = out["CompetitionID"].apply(
            lambda x: comp_names.get(int(x), f"Competition {int(x)}") if pd.notna(x) else ""
        )
    else:
        out["competition_name"] = ""

    if {"CompetitionID", "SessionID", "GameID"}.issubset(set(out.columns)):
        def _glabel(row) -> str:
            try:
                k = (int(row["CompetitionID"]), int(row["SessionID"]), int(row["GameID"]))
                return game_labels.get(k, f"Game {k[1]}-{k[2]}")
            except Exception:
                return ""
        out["game_label"] = out.apply(_glabel, axis=1)
    else:
        out["game_label"] = ""

    # Common compact label for plots
    out["player_label"] = out.apply(
        lambda r: (
            f"{r['player_name']} ({r['team_name']})".strip()
            if isinstance(r.get("player_name", ""), str) and r.get("player_name", "")
            else f"Player {int(r['PlayerID'])} ({r.get('team_name','')})".strip()
        ),
        axis=1,
    )

    return out


# ----------------------------
# Plots
# ----------------------------
def plot_decision_value_distribution(df: pd.DataFrame, out_path: pathlib.Path):
    """
    Distribution of dv_mean_local - dv_mean_global.
    """
    d = df.dropna(subset=["decision_value"]).copy()
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.histplot(d, x="decision_value", kde=True, ax=ax)
    ax.axvline(0.0, linewidth=1.2, alpha=0.35)
    ax.set_title("Decision Value: local neighborhood mean vs global mean")
    ax.set_xlabel("decision_value = dv_mean_local − dv_mean_global")
    ax.set_ylabel("Shot count")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_decision_risk_vs_value(df: pd.DataFrame, out_path: pathlib.Path):
    """
    Scatter of risk (dv_std_local) vs decision_value; helps see high-upside/low-risk vs volatile plans.
    """
    d = df.dropna(subset=["decision_risk", "decision_value"]).copy()
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.scatterplot(
        data=d,
        x="decision_risk",
        y="decision_value",
        hue="task_name" if "task_name" in d.columns else None,
        alpha=0.7,
        ax=ax,
        legend=False,  # too busy; task-based is better as facets elsewhere
    )
    ax.axhline(0.0, linewidth=1.0, alpha=0.35)
    ax.set_title("Decision Value vs Decision Risk (local variability)")
    ax.set_xlabel("decision_risk = dv_std_local")
    ax.set_ylabel("decision_value = dv_mean_local − dv_mean_global")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_execution_value_differential_by_task(df: pd.DataFrame, out_path: pathlib.Path):
    """
    Execution value distribution per task:
      execution_value_local = dv_obs - dv_mean_local
    """
    d = df.dropna(subset=["execution_value_local", "task_name"]).copy()
    # keep tasks with enough data
    counts = d["task_name"].value_counts()
    keep = counts[counts >= 30].index
    d = d[d["task_name"].isin(keep)].copy()

    fig, ax = plt.subplots(figsize=(14, 7))
    order = d.groupby("task_name")["execution_value_local"].median().sort_values(ascending=False).index
    sns.boxplot(data=d, x="task_name", y="execution_value_local", order=order, ax=ax)
    ax.tick_params(axis="x", rotation=25, labelsize=13)
    ax.axhline(0.0, linewidth=1.0, alpha=0.35)
    ax.set_title("Execution Value Differential by Task (dv_obs vs local neighborhood)")
    ax.set_xlabel("Task")
    ax.set_ylabel("execution_value_local")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_execution_value_by_task(df: pd.DataFrame, out_path: pathlib.Path):
    """
    Observed execution value distribution per task:
      dv_obs
    """
    d = df.dropna(subset=["dv_obs", "task_name"]).copy()
    counts = d["task_name"].value_counts()
    keep = counts[counts >= 30].index
    d = d[d["task_name"].isin(keep)].copy()

    fig, ax = plt.subplots(figsize=(14, 7))
    order = d.groupby("task_name")["dv_obs"].median().sort_values(ascending=False).index
    sns.boxplot(data=d, x="task_name", y="dv_obs", order=order, ax=ax)
    ax.tick_params(axis="x", rotation=25, labelsize=13)
    ax.axhline(0.0, linewidth=1.0, alpha=0.35)
    ax.set_title("Observed state value differential by Task")
    ax.set_xlabel("Task")
    ax.set_ylabel("dv_obs")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_decision_value_by_task(df: pd.DataFrame, out_path: pathlib.Path):
    d = df.dropna(subset=["decision_value", "task_name"]).copy()
    counts = d["task_name"].value_counts()
    keep = counts[counts >= 30].index
    d = d[d["task_name"].isin(keep)].copy()

    fig, ax = plt.subplots(figsize=(14, 7))
    order = d.groupby("task_name")["decision_value"].median().sort_values(ascending=False).index
    sns.boxplot(data=d, x="task_name", y="decision_value", order=order, ax=ax)
    ax.tick_params(axis="x", rotation=25, labelsize=13)
    ax.axhline(0.0, linewidth=1.0, alpha=0.35)
    ax.set_title("Decision Value by Task (local mean - global mean)")
    ax.set_xlabel("Task")
    # ax.set_ylabel("decision_value = dv_mean_local − dv_mean_global")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_points_vs_mc_metrics(df: pd.DataFrame, out_path: pathlib.Path):
    """
    Points correlation with:
      - dv_obs (same across local/global typically)
      - execution_value_local
      - decision_value
      - decision_risk
    """
    if "Points" not in df.columns:
        print("[warn] Points missing; skipping points-vs-mc plots.")
        return
    d = df.dropna(subset=["Points"]).copy()
    if d.empty:
        print("[warn] no Points rows; skipping points-vs-mc plots.")
        return

    d["Points"] = d["Points"].round().astype(int)

    metrics = [
        ("dv_obs", "State value"),
        ("execution_value_local", "Execution value"),
        ("decision_value", "Decision value"),
        ("decision_risk", "Decision risk"),
    ]

    # compute correlations and plot as small multiples
    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 5.8), sharex=False)
    if n == 1:
        axes = [axes]

    for ax, (col, title) in zip(axes, metrics):
        sub = d.dropna(subset=[col]).copy()
        if sub.empty:
            ax.set_axis_off()
            continue
        pear = float(sub["Points"].corr(sub[col], method="pearson"))
        spear = float(sub["Points"].corr(sub[col], method="spearman"))

        sns.boxplot(data=sub, x="Points", y=col, ax=ax, showfliers=False)
        ax.set_title(title, fontsize=16, pad=10)
        ax.set_xlabel("Points (0–4)")
        ax.set_ylabel(col)
        ax.text(
            0.02, 0.98,
            f"Pearson={pear:.3f}\nSpearman={spear:.3f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.8", alpha=0.9),
        )
        ax.axhline(0.0, linewidth=1.0, alpha=0.25)
        ax.tick_params(axis="both", labelsize=10)

    fig.suptitle("Human points vs MC-derived decision / execution metrics", y=0.98, fontsize=18)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path)
    plt.close(fig)


def _plot_player_ranking_metric(
    df: pd.DataFrame,
    out_path: pathlib.Path,
    value_col: str,
    shots_col: str,
    value_label: str,
    title: str,
    annotation_col: str | None = None,
    annotation_label: str | None = None,
    top_n: int | None = None,
):
    d = df.dropna(subset=["PlayerID", "TeamID", value_col]).copy()
    if d.empty:
        return

    agg_map: dict[str, tuple[str, str]] = {
        "shots": (shots_col, "count"),
        "value_mean": (value_col, "mean"),
    }
    if annotation_col is not None and annotation_col in d.columns:
        agg_map["annotation_mean"] = (annotation_col, "mean")

    agg = (
        d.groupby(["PlayerID", "TeamID", "player_label"], dropna=True)
        .agg(**agg_map)
        .reset_index()
    )
    agg = agg[agg["shots"] >= 30].copy()
    if agg.empty:
        return

    agg = agg.sort_values("value_mean", ascending=False)
    if top_n is not None:
        agg = agg.head(int(top_n)).copy()

    fig_h = max(8, 0.35 * len(agg) + 1.5)
    fig, ax = plt.subplots(figsize=(14, fig_h))
    sns.barplot(data=agg, x="value_mean", y="player_label", ax=ax, orient="h", alpha=0.95)
    ax.axvline(0.0, linewidth=1.0, alpha=0.35)
    ax.set_title(title)
    ax.set_xlabel(value_label)
    ax.set_ylabel("")

    for i, r in enumerate(agg.itertuples(index=False)):
        note = f"  n={int(r.shots)}"
        if annotation_col is not None and "annotation_mean" in agg.columns and annotation_label is not None:
            note += f" | {annotation_label}={float(r.annotation_mean):+.3f}"
        ax.text(float(r.value_mean), i, note, va="center", fontsize=10)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_player_ranking_mc(df: pd.DataFrame, out_path: pathlib.Path, top_n: int | None = None):
    """
    Overall player ranking using:
      - execution_value_local (ability)
      - decision_value (strategy/selection quality)
      - decision_risk (volatility)

    We report a composite view: players sorted by mean execution_value_local,
    with annotations for decision_value and risk.
    """
    d = df.dropna(subset=["PlayerID", "TeamID", "execution_value_local", "decision_value", "decision_risk"]).copy()
    if d.empty:
        return

    agg = (
        d.groupby(["PlayerID", "TeamID", "player_label"], dropna=True)
        .agg(
            shots=("execution_value_local", "count"),
            exec_mean=("execution_value_local", "mean"),
            decision_mean=("decision_value", "mean"),
            risk_mean=("decision_risk", "mean"),
        )
        .reset_index()
    )
    agg = agg[agg["shots"] >= 30].copy()
    if agg.empty:
        return

    agg = agg.sort_values("exec_mean", ascending=False)
    if top_n is not None:
        agg = agg.head(int(top_n)).copy()

    fig_h = max(8, 0.35 * len(agg) + 1.5)
    fig, ax = plt.subplots(figsize=(14, fig_h))
    sns.barplot(data=agg, x="exec_mean", y="player_label", ax=ax, orient="h", alpha=0.95)
    ax.axvline(0.0, linewidth=1.0, alpha=0.35)
    title_suffix = f"top {int(top_n)}" if top_n is not None else "all eligible players"
    ax.set_title(f"Player ranking ({title_suffix}) by execution value vs local neighborhood")
    ax.set_xlabel("Mean execution_value_local = dv_obs − dv_mean_local")
    ax.set_ylabel("")

    # annotate with decision_mean and risk_mean
    for i, r in enumerate(agg.itertuples(index=False)):
        ax.text(
            float(r.exec_mean),
            i,
            f"  n={int(r.shots)} | decision={float(r.decision_mean):+.3f} | risk={float(r.risk_mean):.3f}",
            va="center",
            fontsize=10,
        )

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_player_ranking_mc_percentile(df: pd.DataFrame, out_path: pathlib.Path, top_n: int | None = None):
    """
    Player ranking by mean local execution percentile.

    This uses the scorer-provided percentile of the observed execution value
    within the sampled local neighborhood, rather than the raw dv differential.
    """
    d = df.dropna(subset=["PlayerID", "TeamID", "execution_percentile_local"]).copy()
    if d.empty:
        return

    agg = (
        d.groupby(["PlayerID", "TeamID", "player_label"], dropna=True)
        .agg(
            shots=("execution_percentile_local", "count"),
            percentile_mean=("execution_percentile_local", "mean"),
            percentile_median=("execution_percentile_local", "median"),
            exec_mean=("execution_value_local", "mean"),
        )
        .reset_index()
    )
    agg = agg[agg["shots"] >= 30].copy()
    if agg.empty:
        return

    agg["percentile_mean"] = agg["percentile_mean"].astype(float) * 100.0
    agg["percentile_median"] = agg["percentile_median"].astype(float) * 100.0
    agg = agg.sort_values("percentile_mean", ascending=False)
    if top_n is not None:
        agg = agg.head(int(top_n)).copy()

    fig_h = max(8, 0.35 * len(agg) + 1.5)
    fig, ax = plt.subplots(figsize=(14, fig_h))
    sns.barplot(data=agg, x="percentile_mean", y="player_label", ax=ax, orient="h", alpha=0.95)
    ax.axvline(50.0, linewidth=1.0, alpha=0.35)
    title_suffix = f"top {int(top_n)}" if top_n is not None else "all eligible players"
    ax.set_title(f"Player ranking ({title_suffix}) by local execution percentile")
    ax.set_xlabel("Mean execution_percentile_local")
    ax.set_ylabel("")
    ax.set_xlim(0.0, 100.0)

    for i, r in enumerate(agg.itertuples(index=False)):
        ax.text(
            float(r.percentile_mean),
            i,
            f"  n={int(r.shots)} | median={float(r.percentile_median):.1f}% | exec={float(r.exec_mean):+.3f}",
            va="center",
            fontsize=10,
        )

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def infer_player_role_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Infer mixed-doubles thrower role from shot order within each team/end.

    The "lead" here is the player who tends to throw the later stones in the
    team sequence (first and final), while "skip" is the player who tends to
    occupy the middle-shot role.
    """
    needed = ["CompetitionID", "SessionID", "GameID", "EndID", "ShotID", "TeamID", "PlayerID", "player_label"]
    d = df.dropna(subset=[c for c in needed if c in df.columns]).copy()
    if any(c not in d.columns for c in needed):
        out = df.copy()
        out["player_role"] = np.nan
        return out

    order_df = (
        d[needed]
        .drop_duplicates()
        .sort_values(["CompetitionID", "SessionID", "GameID", "EndID", "TeamID", "ShotID"])
        .copy()
    )
    order_df["team_shot_idx"] = (
        order_df.groupby(["CompetitionID", "SessionID", "GameID", "EndID", "TeamID"]).cumcount() + 1
    )

    player_pos = (
        order_df.groupby(["CompetitionID", "TeamID", "PlayerID", "player_label"], dropna=False)
        .agg(
            mean_idx=("team_shot_idx", "mean"),
            max_idx=("team_shot_idx", "max"),
            count=("team_shot_idx", "count"),
        )
        .reset_index()
    )
    player_pos["player_role"] = pd.Series([None] * len(player_pos), dtype="object")

    for (comp_id, team_id), g in player_pos.groupby(["CompetitionID", "TeamID"], dropna=False):
        if len(g) < 2:
            continue
        g = g.sort_values(["mean_idx", "max_idx", "count"], ascending=[False, False, False]).reset_index()
        player_pos.loc[g.loc[0, "index"], "player_role"] = "skip"
        player_pos.loc[g.loc[1:, "index"], "player_role"] = "lead"

    out = df.merge(
        player_pos[["CompetitionID", "TeamID", "PlayerID", "player_role"]],
        on=["CompetitionID", "TeamID", "PlayerID"],
        how="left",
    )
    return out


def plot_player_ranking_mc_percentile_role(
    df: pd.DataFrame,
    out_path: pathlib.Path,
    role: str,
    top_n: int | None = None,
):
    d = infer_player_role_labels(df)
    d = d[d["player_role"] == role].copy()
    if d.empty:
        return
    plot_player_ranking_mc_percentile(d, out_path, top_n=top_n)


def _plot_player_ranking_role(
    df: pd.DataFrame,
    out_path: pathlib.Path,
    role: str,
    plot_fn,
    top_n: int | None = None,
):
    d = infer_player_role_labels(df)
    d = d[d["player_role"] == role].copy()
    if d.empty:
        return
    plot_fn(d, out_path, top_n=top_n)


def plot_player_ranking_decision_mc(df: pd.DataFrame, out_path: pathlib.Path, top_n: int | None = None):
    """
    Player ranking by mean decision_value (= dv_mean_local - dv_mean_global),
    measuring strategic shot-selection quality.

    Annotates each bar with n=shots, exec_mean, and risk_mean.
    """
    d = df.dropna(subset=["PlayerID", "TeamID", "decision_value", "execution_value_local", "decision_risk"]).copy()
    if d.empty:
        return

    agg = (
        d.groupby(["PlayerID", "TeamID", "player_label"], dropna=True)
        .agg(
            shots=("decision_value", "count"),
            decision_mean=("decision_value", "mean"),
            exec_mean=("execution_value_local", "mean"),
            risk_mean=("decision_risk", "mean"),
        )
        .reset_index()
    )
    agg = agg[agg["shots"] >= 30].copy()
    if agg.empty:
        return

    agg = agg.sort_values("decision_mean", ascending=False)
    if top_n is not None:
        agg = agg.head(int(top_n)).copy()

    fig_h = max(8, 0.35 * len(agg) + 1.5)
    fig, ax = plt.subplots(figsize=(14, fig_h))
    sns.barplot(data=agg, x="decision_mean", y="player_label", ax=ax, orient="h", alpha=0.95)
    ax.axvline(0.0, linewidth=1.0, alpha=0.35)
    title_suffix = f"top {int(top_n)}" if top_n is not None else "all eligible players"
    ax.set_title(f"Player ranking ({title_suffix}) by decision value surplus")
    ax.set_xlabel("Mean decision_value = dv_mean_local − dv_mean_global")
    ax.set_ylabel("")

    # annotate with shots, exec_mean, and risk_mean
    for i, r in enumerate(agg.itertuples(index=False)):
        ax.text(
            float(r.decision_mean),
            i,
            f"  n={int(r.shots)} | exec={float(r.exec_mean):+.3f} | risk={float(r.risk_mean):.3f}",
            va="center",
            fontsize=10,
        )

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_player_ranking_mc_mean_diff(df: pd.DataFrame, out_path: pathlib.Path, top_n: int | None = None):
    title_suffix = f"top {int(top_n)}" if top_n is not None else "all eligible players"
    _plot_player_ranking_metric(
        df,
        out_path,
        value_col="execution_value_local",
        shots_col="execution_value_local",
        value_label="Mean execution_value_local = dv_obs − dv_mean_local",
        title=f"Player ranking ({title_suffix}) by local mean differential",
        annotation_col="decision_value",
        annotation_label="decision",
        top_n=top_n,
    )


def plot_player_ranking_mc_p90_diff(df: pd.DataFrame, out_path: pathlib.Path, top_n: int | None = None):
    if "execution_value_p90_local" not in df.columns:
        return
    title_suffix = f"top {int(top_n)}" if top_n is not None else "all eligible players"
    _plot_player_ranking_metric(
        df,
        out_path,
        value_col="execution_value_p90_local",
        shots_col="execution_value_p90_local",
        value_label="Mean execution_value_p90_local = dv_obs − dv_p90_local",
        title=f"Player ranking ({title_suffix}) by local p90 differential",
        annotation_col="execution_value_local",
        annotation_label="mean_diff",
        top_n=top_n,
    )


# ----------------------------
# Summary markdown
# ----------------------------
def build_summary_md(out_dir: pathlib.Path, df: pd.DataFrame) -> None:
    def _fmt(x) -> str:
        try:
            if not np.isfinite(x):
                return "NA"
            return f"{float(x):.3f}"
        except Exception:
            return "NA"

    lines: list[str] = []
    lines.append("# Coach Report — MC Decision + Execution")
    lines.append("")
    lines.append("## Definitions")
    lines.append("- **Perspective**: all dv terms are thrower-perspective (positive favors the shot-taking team).")
    lines.append("- **Decision value**: dv_mean_local − dv_mean_global (plan neighborhood vs global feasible throws)")
    lines.append("- **Decision risk**: dv_std_local (variability around the plan)")
    lines.append("- **Execution value differential**: dv_obs − dv_mean_local (executed outcome vs plan neighborhood)")
    lines.append("- **Observed execution value**: dv_obs")
    lines.append("")

    lines.append("## Overall summary")
    lines.append(f"- Shots analyzed: {len(df)}")
    lines.append(f"- Mean decision value: {_fmt(df['decision_value'].mean())}")
    lines.append(f"- Mean decision risk:  {_fmt(df['decision_risk'].mean())}")
    lines.append(f"- Mean execution value differential (local): {_fmt(df['execution_value_local'].mean())}")
    lines.append(f"- Mean observed execution value: {_fmt(df['dv_obs'].mean())}")
    if "execution_percentile_local" in df.columns:
        lines.append(f"- Mean execution percentile (local): {_fmt(df['execution_percentile_local'].mean())}")
    if "Points" in df.columns and df["Points"].notna().any():
        lines.append(f"- Points coverage: {int(df['Points'].notna().sum())}/{len(df)} ({df['Points'].notna().mean():.1%})")
    lines.append("")

    # Task-level table (top volume)
    if "task_name" in df.columns:
        t = (
            df.dropna(subset=["task_name", "decision_value", "decision_risk", "execution_value_local"])
            .groupby("task_name")
            .agg(
                shots=("decision_value", "count"),
                decision_value_mean=("decision_value", "mean"),
                decision_risk_mean=("decision_risk", "mean"),
                execution_value_differential_mean=("execution_value_local", "mean"),
                observed_execution_value_mean=("dv_obs", "mean"),
            )
            .reset_index()
            .sort_values("shots", ascending=False)
        )
        lines.append("## By task (top by volume)")
        try:
            lines.append(t.head(12).to_markdown(index=False))
        except Exception:
            lines.append(t.head(12).to_string(index=False))
        lines.append("")

    # Handle-level table
    if "handle_name" in df.columns:
        h = (
            df.dropna(subset=["handle_name", "decision_value", "decision_risk", "execution_value_local"])
            .groupby("handle_name")
            .agg(
                shots=("decision_value", "count"),
                decision_value_mean=("decision_value", "mean"),
                decision_risk_mean=("decision_risk", "mean"),
                execution_value_differential_mean=("execution_value_local", "mean"),
                observed_execution_value_mean=("dv_obs", "mean"),
            )
            .reset_index()
            .sort_values("shots", ascending=False)
        )
        lines.append("## By handle (spin)")
        try:
            lines.append(h.to_markdown(index=False))
        except Exception:
            lines.append(h.to_string(index=False))
        lines.append("")

    lines.append("## Figures")
    lines.append("- figures/decision_value_distribution.png")
    lines.append("- figures/decision_risk_vs_value.png")
    lines.append("- figures/decision_value_by_task.png")
    lines.append("- figures/execution_value_by_task.png")
    lines.append("- figures/execution_value_differential_by_task.png")
    lines.append("- figures/points_vs_mc_metrics.png")
    lines.append("- figures/player_ranking_mc.png")
    lines.append("- figures/player_ranking_mc_mean.png")
    lines.append("- figures/player_ranking_mc_percentile.png")
    lines.append("- figures/player_ranking_mc_p90.png")
    lines.append("- figures/player_ranking_decision_mc.png")
    lines.append("")

    (out_dir / "summary.md").write_text("\n".join(lines))


# ----------------------------
# Main
# ----------------------------
def main():
    _set_plot_style()

    ap = argparse.ArgumentParser(description="Coach MC report from local vs global shot_scores runs.")
    ap.add_argument("--scores_local", type=str, default="shot_scores_local_old.csv", help="Local-neighborhood shot scores (CSV)")
    ap.add_argument("--scores_global", type=str, default="shot_scores_old.csv", help="Global/uniform shot scores (CSV)")
    ap.add_argument("--stones-csv", type=str, default="2026/Stones.csv", help="Stones.csv for Points/Task/Handle IDs")

    # Optional label sources
    ap.add_argument("--competitors-csv", type=str, default="2026/Competitors.csv")
    ap.add_argument("--teams-csv", type=str, default="2026/Teams.csv")
    ap.add_argument("--competitions-csv", type=str, default="2026/competitions.csv")
    ap.add_argument("--games-csv", type=str, default="2026/games.csv")

    ap.add_argument("--out-dir", type=str, default="coach_report_mc")
    ap.add_argument("--min-shots-player", type=int, default=30, help="Min shots to include in player ranking")
    ap.add_argument("--top-n-players", type=int, default=30, help="Top N players to plot")
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    fig_dir = out_dir / "figures"
    _ensure_dir(out_dir)
    _ensure_dir(fig_dir)

    df = load_and_merge(args.scores_local, args.scores_global, args.stones_csv)
    df = add_labels(df, args.competitors_csv, args.teams_csv, args.competitions_csv, args.games_csv)

    # Basic sanity: dv_obs should match (usually). If not, keep but warn.
    if np.nanmean(np.abs(df["dv_obs_local"].astype(float) - df["dv_obs_global"].astype(float))) > 1e-6:
        print("[warn] dv_obs differs between local/global runs (value model or conditioning mismatch). Using dv_obs_local as dv_obs.")

    # Figures
    plot_decision_value_distribution(df, fig_dir / "decision_value_distribution.png")
    plot_decision_risk_vs_value(df, fig_dir / "decision_risk_vs_value.png")
    plot_decision_value_by_task(df, fig_dir / "decision_value_by_task.png")
    plot_execution_value_by_task(df, fig_dir / "execution_value_by_task.png")
    plot_execution_value_differential_by_task(df, fig_dir / "execution_value_differential_by_task.png")
    plot_points_vs_mc_metrics(df, fig_dir / "points_vs_mc_metrics.png")
    plot_player_ranking_mc(df, fig_dir / "player_ranking_mc.png", top_n=int(args.top_n_players))
    _plot_player_ranking_role(df, fig_dir / "player_ranking_mc_lead.png", "lead", plot_player_ranking_mc, top_n=int(args.top_n_players))
    _plot_player_ranking_role(df, fig_dir / "player_ranking_mc_skip.png", "skip", plot_player_ranking_mc, top_n=int(args.top_n_players))
    plot_player_ranking_mc_mean_diff(df, fig_dir / "player_ranking_mc_mean.png", top_n=int(args.top_n_players))
    _plot_player_ranking_role(df, fig_dir / "player_ranking_mc_mean_lead.png", "lead", plot_player_ranking_mc_mean_diff, top_n=int(args.top_n_players))
    _plot_player_ranking_role(df, fig_dir / "player_ranking_mc_mean_skip.png", "skip", plot_player_ranking_mc_mean_diff, top_n=int(args.top_n_players))
    plot_player_ranking_mc_percentile(df, fig_dir / "player_ranking_mc_percentile.png", top_n=int(args.top_n_players))
    _plot_player_ranking_role(df, fig_dir / "player_ranking_mc_percentile_lead.png", "lead", plot_player_ranking_mc_percentile, top_n=int(args.top_n_players))
    _plot_player_ranking_role(df, fig_dir / "player_ranking_mc_percentile_skip.png", "skip", plot_player_ranking_mc_percentile, top_n=int(args.top_n_players))
    plot_player_ranking_mc_p90_diff(df, fig_dir / "player_ranking_mc_p90.png", top_n=int(args.top_n_players))
    _plot_player_ranking_role(df, fig_dir / "player_ranking_mc_p90_lead.png", "lead", plot_player_ranking_mc_p90_diff, top_n=int(args.top_n_players))
    _plot_player_ranking_role(df, fig_dir / "player_ranking_mc_p90_skip.png", "skip", plot_player_ranking_mc_p90_diff, top_n=int(args.top_n_players))
    plot_player_ranking_decision_mc(df, fig_dir / "player_ranking_decision_mc.png", top_n=int(args.top_n_players))
    _plot_player_ranking_role(df, fig_dir / "player_ranking_decision_mc_lead.png", "lead", plot_player_ranking_decision_mc, top_n=int(args.top_n_players))
    _plot_player_ranking_role(df, fig_dir / "player_ranking_decision_mc_skip.png", "skip", plot_player_ranking_decision_mc, top_n=int(args.top_n_players))

    # Summary
    build_summary_md(out_dir, df)

    # Also write a merged per-shot table for debugging / downstream analysis
    merged_csv = out_dir / "shot_scores_local_vs_global_merged.csv"
    df.to_csv(merged_csv, index=False)

    print(f"[done] wrote report to {out_dir}")
    print(f"[done] wrote figures to {fig_dir}")
    print(f"[done] wrote merged per-shot table to {merged_csv}")


if __name__ == "__main__":
    main()
