#!/usr/bin/env python3
"""
Generate comparison figures for the value model ablation study.

Produces in holdouts/0/reports/coach_report/figures/:
1. Violin plots: dv_obs vs Points and excess ΔV vs Points (using existing scored data)
2. Player ranking by average human-labeled Points
3. Player ranking by execution value surplus (from existing MC data)
"""

from __future__ import annotations

import pathlib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats


SHOT_KEY = ["CompetitionID", "SessionID", "GameID", "EndID", "ShotID"]

TASK_NAME = {
    0: "Draw", 1: "Front", 2: "Guard", 3: "Raise/Tap-back",
    4: "Wick/Soft Peel", 5: "Freeze", 6: "Take-out", 7: "Hit and Roll",
    8: "Clearing", 9: "Double Take-out", 10: "Promotion Take-out",
    11: "Through", 13: "No statistics",
}


def _set_style():
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update({"figure.dpi": 140, "savefig.dpi": 300,
                          "axes.spines.top": False, "axes.spines.right": False})


def _load_player_names(competitors_csv):
    p = pathlib.Path(competitors_csv)
    if not p.exists():
        return {}
    df = pd.read_csv(p)
    if not {"CompetitionID", "TeamID", "Reportingname"}.issubset(df.columns):
        return {}
    df["player_ord"] = df.groupby(["CompetitionID", "TeamID"]).cumcount() + 1
    return {(int(r.CompetitionID), int(r.TeamID), int(r.player_ord)): str(r.Reportingname)
            for _, r in df.iterrows()
            if pd.notna(r.CompetitionID) and pd.notna(r.TeamID)}


def _load_team_names(teams_csv):
    p = pathlib.Path(teams_csv)
    if not p.exists():
        return {}
    df = pd.read_csv(p)
    name_col = "Name" if "Name" in df.columns else None
    if name_col is None or "CompetitionID" not in df.columns or "TeamID" not in df.columns:
        return {}
    return {(int(r.CompetitionID), int(r.TeamID)): str(r[name_col])
            for _, r in df.iterrows()
            if pd.notna(r.CompetitionID) and pd.notna(r.TeamID)}


def _add_labels(df, base_dir):
    player_names = _load_player_names(base_dir / "2026" / "Competitors.csv")
    team_names = _load_team_names(base_dir / "2026" / "Teams.csv")

    comp_id = int(df["CompetitionID"].mode().iloc[0]) if "CompetitionID" in df.columns else None

    def get_pname(row):
        try:
            c = int(row["CompetitionID"]) if pd.notna(row.get("CompetitionID")) else comp_id
            return player_names.get((c, int(row["TeamID"]), int(row["PlayerID"])), f"Player {int(row['PlayerID'])}")
        except:
            return ""

    def get_tname(row):
        try:
            c = int(row["CompetitionID"]) if pd.notna(row.get("CompetitionID")) else comp_id
            return team_names.get((c, int(row["TeamID"])), f"Team {int(row['TeamID'])}")
        except:
            return ""

    df["player_name"] = df.apply(get_pname, axis=1)
    df["team_name"] = df.apply(get_tname, axis=1)
    df["player_label"] = df.apply(
        lambda r: f"{r['player_name']} ({r['team_name']})" if r.get("player_name") else f"Player {r.get('PlayerID','')}",
        axis=1)
    return df


def load_holdout_data(holdout_dir, base_dir):
    """Load and merge local+global shot scores with Stones data."""
    hd = pathlib.Path(holdout_dir)
    local = pd.read_csv(hd / "scoring" / "shot_scores_local.csv")
    glob = pd.read_csv(hd / "scoring" / "shot_scores_global.csv")
    stones = pd.read_csv(base_dir / "2026" / "Stones.csv")

    # Merge local+global
    local = local.add_suffix("_local")
    glob = glob.add_suffix("_global")
    for k in SHOT_KEY:
        local.rename(columns={f"{k}_local": k}, inplace=True)
        glob.rename(columns={f"{k}_global": k}, inplace=True)

    df = local.merge(glob, on=SHOT_KEY, how="inner")

    # Bring in Points, Task, Handle, PlayerID, TeamID from Stones
    enrich = [c for c in SHOT_KEY + ["TeamID", "PlayerID", "Task", "Handle", "Points"] if c in stones.columns]
    enrich = list(dict.fromkeys(enrich))
    df = df.merge(stones[enrich], on=SHOT_KEY, how="left")

    # Fill missing from local/global
    for c in ["TeamID", "PlayerID", "Task", "Handle"]:
        if c not in df.columns:
            for suffix in ["_local", "_global"]:
                if f"{c}{suffix}" in df.columns:
                    df[c] = df[f"{c}{suffix}"]
                    break
        elif f"{c}_local" in df.columns:
            df[c] = df[c].where(df[c].notna(), df.get(f"{c}_local"))

    # Derived metrics
    df["dv_obs"] = df["dv_obs_local"].astype(float)
    df["dv_mean_local"] = df["dv_mean_local"].astype(float)
    df["dv_mean_global"] = df["dv_mean_global"].astype(float)
    df["decision_value"] = df["dv_mean_local"] - df["dv_mean_global"]
    df["execution_value"] = df["dv_obs"] - df["dv_mean_local"]
    df["excess_global"] = df["dv_obs"] - df["dv_mean_global"]

    if "Points" in df.columns:
        df["Points"] = pd.to_numeric(df["Points"], errors="coerce")
        df.loc[~df["Points"].between(0, 4, inclusive="both"), "Points"] = np.nan

    df = _add_labels(df, base_dir)
    return df


# ─────────────────────────────────────────────────────────────
# Image #1: Violin plots — dv_obs vs Points and Excess vs Points
# ─────────────────────────────────────────────────────────────

def plot_dv_vs_points_violin(df, out_path, metric="dv_obs", metric_label="dv_obs"):
    """Violin + box plot of a metric grouped by human Points (0-4)."""
    d = df.dropna(subset=["Points", metric]).copy()
    if d.empty:
        return
    d["Points"] = d["Points"].round().astype(int)

    pear = float(d["Points"].corr(d[metric], method="pearson"))
    spear = float(d["Points"].corr(d[metric], method="spearman"))

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.violinplot(data=d, x="Points", y=metric, inner=None, cut=0, ax=ax)
    sns.boxplot(data=d, x="Points", y=metric, width=0.28, showfliers=False, ax=ax)
    ax.set_title(f"{metric_label} vs human Points (0–4)")
    ax.set_xlabel("Points (human label)")
    ax.set_ylabel(metric_label)
    ax.text(0.02, 0.98, f"Pearson={pear:.3f}\nSpearman={spear:.3f}",
            transform=ax.transAxes, ha="left", va="top", fontsize=11,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.8", alpha=0.9))
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_points_violin_pair(df, out_path):
    """Two-panel violin: dv_obs vs Points (left) and excess vs Points (right)."""
    d = df.dropna(subset=["Points", "dv_obs", "dv_mean_local"]).copy()
    if d.empty:
        return
    d["Points"] = d["Points"].round().astype(int)
    d["excess"] = d["dv_obs"] - d["dv_mean_local"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: dv_obs
    pear1 = float(d["Points"].corr(d["dv_obs"], method="pearson"))
    spear1 = float(d["Points"].corr(d["dv_obs"], method="spearman"))
    sns.violinplot(data=d, x="Points", y="dv_obs", inner=None, cut=0, ax=axes[0])
    sns.boxplot(data=d, x="Points", y="dv_obs", width=0.28, showfliers=False, ax=axes[0])
    axes[0].set_title("dv_obs vs human Points (0–4)")
    axes[0].set_xlabel("Points (human label)")
    axes[0].set_ylabel("dv_obs")
    axes[0].text(0.02, 0.98, f"Pearson={pear1:.3f}\nSpearman={spear1:.3f}",
                 transform=axes[0].transAxes, ha="left", va="top", fontsize=11,
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.8", alpha=0.9))

    # Right: excess
    pear2 = float(d["Points"].corr(d["excess"], method="pearson"))
    spear2 = float(d["Points"].corr(d["excess"], method="spearman"))
    sns.violinplot(data=d, x="Points", y="excess", inner=None, cut=0, ax=axes[1])
    sns.boxplot(data=d, x="Points", y="excess", width=0.28, showfliers=False, ax=axes[1])
    means = d.groupby("Points")["excess"].mean().reset_index()
    sns.pointplot(data=means, x="Points", y="excess", ax=axes[1], color="black",
                  markers="D", linestyles="-", errorbar=None)
    axes[1].set_title("Excess (dv_obs - dv_mean) vs Points")
    axes[1].set_xlabel("Points (human label)")
    axes[1].set_ylabel("excess ΔV")
    axes[1].text(0.02, 0.98, f"Pearson={pear2:.3f}\nSpearman={spear2:.3f}",
                 transform=axes[1].transAxes, ha="left", va="top", fontsize=11,
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.8", alpha=0.9))

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────
# Image #2: Player ranking by execution value surplus
# ─────────────────────────────────────────────────────────────

def plot_player_ranking_exec(df, out_path, top_n=30):
    """Rank players by mean execution value surplus (dv_obs - dv_mean_local)."""
    d = df.dropna(subset=["PlayerID", "TeamID", "execution_value"]).copy()
    agg = d.groupby(["PlayerID", "TeamID", "player_label"]).agg(
        shots=("execution_value", "count"),
        exec_mean=("execution_value", "mean"),
    ).reset_index()
    agg = agg[agg["shots"] >= 30].sort_values("exec_mean", ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(14, 8))
    sns.barplot(data=agg, x="exec_mean", y="player_label", ax=ax, orient="h", alpha=0.95)
    ax.axvline(0.0, linewidth=1.0, alpha=0.35)
    ax.set_title(f"Overall player ranking (top {top_n})")
    ax.set_xlabel("Overall estimated skill (excess ΔV per shot)")
    ax.set_ylabel("")
    for i, r in enumerate(agg.itertuples(index=False)):
        ax.text(float(r.exec_mean), i, f"  n={int(r.shots)}", va="center", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────
# Image #3: Player ranking by human-labeled Points
# ─────────────────────────────────────────────────────────────

def plot_player_ranking_points(df, out_path, top_n=30):
    """Rank players by average human-labeled Points (0-4)."""
    d = df.dropna(subset=["PlayerID", "TeamID", "Points"]).copy()
    d["Points"] = d["Points"].astype(float)
    agg = d.groupby(["PlayerID", "TeamID", "player_label"]).agg(
        shots=("Points", "count"),
        mean_points=("Points", "mean"),
    ).reset_index()
    agg = agg[agg["shots"] >= 30].sort_values("mean_points", ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(14, 8))
    sns.barplot(data=agg, x="mean_points", y="player_label", ax=ax, orient="h", alpha=0.95)
    ax.set_title(f"Player ranking by human-labeled Points (top {top_n})")
    ax.set_xlabel("Average Points per shot (human label, 0–4)")
    ax.set_ylabel("")
    for i, r in enumerate(agg.itertuples(index=False)):
        ax.text(float(r.mean_points), i, f"  n={int(r.shots)}", va="center", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    _set_style()

    base_dir = pathlib.Path("/mnt/data/curling2/csas_fixed")
    holdout_dir = base_dir / "holdouts" / "0"
    fig_dir = holdout_dir / "reports" / "coach_report" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("Loading holdout 0 data...", flush=True)
    df = load_holdout_data(holdout_dir, base_dir)
    print(f"  {len(df)} shots loaded, Points coverage: {df['Points'].notna().sum()}/{len(df)}", flush=True)

    # Image #1: Violin pair (dv_obs + excess vs Points)
    print("Generating violin plots...", flush=True)
    plot_points_violin_pair(df, fig_dir / "points_vs_dv_violin.png")

    # Also generate individual violins for decision_value and execution_value
    plot_dv_vs_points_violin(df, fig_dir / "dv_obs_vs_points_violin.png",
                             metric="dv_obs", metric_label="dv_obs")
    plot_dv_vs_points_violin(df, fig_dir / "execution_value_vs_points_violin.png",
                             metric="execution_value", metric_label="execution value (dv_obs − dv_mean_local)")
    plot_dv_vs_points_violin(df, fig_dir / "decision_value_vs_points_violin.png",
                             metric="decision_value", metric_label="decision value (dv_mean_local − dv_mean_global)")

    # Image #2: Player ranking by execution value
    print("Generating player rankings...", flush=True)
    plot_player_ranking_exec(df, fig_dir / "player_ranking_execution.png")

    # Image #3: Player ranking by human Points
    plot_player_ranking_points(df, fig_dir / "player_ranking_points.png")

    print(f"Done! Figures saved to {fig_dir}", flush=True)
    print(f"Files generated:", flush=True)
    for f in sorted(fig_dir.glob("*.png")):
        print(f"  {f.name}", flush=True)


if __name__ == "__main__":
    main()
