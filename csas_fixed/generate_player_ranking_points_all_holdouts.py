#!/usr/bin/env python3
"""
Generate a Points-based player ranking figure for each held-out competition.

This mirrors the existing figure:
  holdouts/0/reports/coach_report/figures/player_ranking_points.png

For each holdout directory under holdouts/*, it loads the already-scored
shot scores (local+global) and Stones Points, then writes:
  holdouts/<comp>/reports/coach_report/figures/player_ranking_points.png

Usage:
  python generate_player_ranking_points_all_holdouts.py
  python generate_player_ranking_points_all_holdouts.py --only 0 22230015
"""

from __future__ import annotations

import argparse
import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


SHOT_KEY = ["CompetitionID", "SessionID", "GameID", "EndID", "ShotID"]


def _set_style() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _load_player_names(competitors_csv: pathlib.Path) -> dict[tuple[int, int, int], str]:
    if not competitors_csv.exists():
        return {}
    df = pd.read_csv(competitors_csv)
    if not {"CompetitionID", "TeamID", "Reportingname"}.issubset(df.columns):
        return {}
    df = df.copy()
    df["player_ord"] = df.groupby(["CompetitionID", "TeamID"]).cumcount() + 1
    out: dict[tuple[int, int, int], str] = {}
    for _, r in df.iterrows():
        if pd.isna(r.CompetitionID) or pd.isna(r.TeamID) or pd.isna(r.player_ord):
            continue
        out[(int(r.CompetitionID), int(r.TeamID), int(r.player_ord))] = str(r.Reportingname)
    return out


def _load_team_names(teams_csv: pathlib.Path) -> dict[tuple[int, int], str]:
    if not teams_csv.exists():
        return {}
    df = pd.read_csv(teams_csv)
    if not {"CompetitionID", "TeamID"}.issubset(df.columns):
        return {}
    name_col = "Name" if "Name" in df.columns else None
    if name_col is None:
        return {}
    out: dict[tuple[int, int], str] = {}
    for _, r in df.iterrows():
        if pd.isna(r.CompetitionID) or pd.isna(r.TeamID):
            continue
        out[(int(r.CompetitionID), int(r.TeamID))] = str(r[name_col])
    return out


def _add_labels(df: pd.DataFrame, base_dir: pathlib.Path) -> pd.DataFrame:
    player_names = _load_player_names(base_dir / "2026" / "Competitors.csv")
    team_names = _load_team_names(base_dir / "2026" / "Teams.csv")
    comp_id = int(df["CompetitionID"].mode().iloc[0]) if "CompetitionID" in df.columns else None

    def get_pname(row) -> str:
        try:
            c = int(row["CompetitionID"]) if pd.notna(row.get("CompetitionID")) else comp_id
            return player_names.get(
                (c, int(row["TeamID"]), int(row["PlayerID"])),
                f"Player {int(row['PlayerID'])}",
            )
        except Exception:
            return ""

    def get_tname(row) -> str:
        try:
            c = int(row["CompetitionID"]) if pd.notna(row.get("CompetitionID")) else comp_id
            return team_names.get((c, int(row["TeamID"])), f"Team {int(row['TeamID'])}")
        except Exception:
            return ""

    df = df.copy()
    df["player_name"] = df.apply(get_pname, axis=1)
    df["team_name"] = df.apply(get_tname, axis=1)
    df["player_label"] = df.apply(
        lambda r: f"{r['player_name']} ({r['team_name']})"
        if isinstance(r.get("player_name", ""), str) and r.get("player_name", "")
        else f"Player {r.get('PlayerID','')}",
        axis=1,
    )
    return df


def infer_player_role_labels(df: pd.DataFrame) -> pd.DataFrame:
    needed = ["CompetitionID", "SessionID", "GameID", "EndID", "ShotID", "TeamID", "PlayerID"]
    d = df.dropna(subset=[c for c in needed if c in df.columns]).copy()
    if any(c not in d.columns for c in needed):
        out = df.copy()
        out["player_role"] = pd.NA
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
        order_df.groupby(["CompetitionID", "TeamID", "PlayerID"], dropna=False)
        .agg(mean_idx=("team_shot_idx", "mean"), max_idx=("team_shot_idx", "max"), count=("team_shot_idx", "count"))
        .reset_index()
    )
    player_pos["player_role"] = pd.Series([None] * len(player_pos), dtype="object")
    for (_, _), g in player_pos.groupby(["CompetitionID", "TeamID"], dropna=False):
        if len(g) < 2:
            continue
        g = g.sort_values(["mean_idx", "max_idx", "count"], ascending=[False, False, False]).reset_index()
        player_pos.loc[g.loc[0, "index"], "player_role"] = "skip"
        player_pos.loc[g.loc[1:, "index"], "player_role"] = "lead"
    return df.merge(
        player_pos[["CompetitionID", "TeamID", "PlayerID", "player_role"]],
        on=["CompetitionID", "TeamID", "PlayerID"],
        how="left",
    )


def load_holdout_data(holdout_dir: pathlib.Path, base_dir: pathlib.Path) -> pd.DataFrame:
    local_path = holdout_dir / "scoring" / "shot_scores_local.csv"
    global_path = holdout_dir / "scoring" / "shot_scores_global.csv"
    if not local_path.exists() or not global_path.exists():
        raise FileNotFoundError(f"Missing scored CSV(s) under {holdout_dir}/scoring/")

    local = pd.read_csv(local_path).add_suffix("_local")
    glob = pd.read_csv(global_path).add_suffix("_global")
    for k in SHOT_KEY:
        local.rename(columns={f"{k}_local": k}, inplace=True)
        glob.rename(columns={f"{k}_global": k}, inplace=True)
    df = local.merge(glob, on=SHOT_KEY, how="inner")

    stones = pd.read_csv(base_dir / "2026" / "Stones.csv")
    enrich = [c for c in (SHOT_KEY + ["TeamID", "PlayerID", "Points"]) if c in stones.columns]
    df = df.merge(stones[enrich], on=SHOT_KEY, how="left")

    for c in ["TeamID", "PlayerID"]:
        if c not in df.columns:
            for suffix in ["_local", "_global"]:
                if f"{c}{suffix}" in df.columns:
                    df[c] = df[f"{c}{suffix}"]
                    break
        elif f"{c}_local" in df.columns:
            df[c] = df[c].where(df[c].notna(), df.get(f"{c}_local"))

    if "Points" in df.columns:
        df["Points"] = pd.to_numeric(df["Points"], errors="coerce")
        df.loc[~df["Points"].between(0, 4, inclusive="both"), "Points"] = pd.NA

    df = _add_labels(df, base_dir)
    return df


def plot_player_ranking_points(df: pd.DataFrame, out_path: pathlib.Path, top_n: int | None = None) -> None:
    d = df.dropna(subset=["PlayerID", "TeamID", "Points"]).copy()
    if d.empty:
        return
    d["Points"] = d["Points"].astype(float)
    agg = (
        d.groupby(["PlayerID", "TeamID", "player_label"])
        .agg(shots=("Points", "count"), mean_points=("Points", "mean"))
        .reset_index()
    )
    agg = agg[agg["shots"] >= 30].sort_values("mean_points", ascending=False)
    if top_n is not None:
        agg = agg.head(int(top_n)).copy()

    fig_h = max(8, 0.35 * len(agg) + 1.5)
    fig, ax = plt.subplots(figsize=(14, fig_h))
    sns.barplot(data=agg, x="mean_points", y="player_label", ax=ax, orient="h", alpha=0.95)
    title_suffix = f"top {int(top_n)}" if top_n is not None else "all eligible players"
    ax.set_title(f"Player ranking by human-labeled Points ({title_suffix})")
    ax.set_xlabel("Average Points per shot (human label, 0–4)")
    ax.set_ylabel("")
    for i, r in enumerate(agg.itertuples(index=False)):
        ax.text(float(r.mean_points), i, f"  n={int(r.shots)}", va="center", fontsize=10)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def plot_player_ranking_points_role(
    df: pd.DataFrame,
    out_path: pathlib.Path,
    role: str,
    top_n: int | None = None,
) -> None:
    if "player_role" not in df.columns:
        return
    d = df[df["player_role"] == role].copy()
    if d.empty:
        return
    plot_player_ranking_points(d, out_path, top_n=top_n)


def main() -> None:
    _set_style()
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", type=str, default="/mnt/data/curling2/csas_fixed")
    ap.add_argument("--only", nargs="*", type=str, default=None, help="Optional list of holdout competition IDs to process")
    ap.add_argument("--top-n", type=int, default=None)
    args = ap.parse_args()

    base_dir = pathlib.Path(args.base_dir)
    holdouts_dir = base_dir / "holdouts"
    only = set(args.only) if args.only else None

    for hd in sorted([p for p in holdouts_dir.iterdir() if p.is_dir()], key=lambda p: p.name):
        if only is not None and hd.name not in only:
            continue
        out_path = hd / "reports" / "coach_report" / "figures" / "player_ranking_points.png"
        df = infer_player_role_labels(load_holdout_data(hd, base_dir))
        top_n = int(args.top_n) if args.top_n is not None else None
        plot_player_ranking_points(df, out_path, top_n=top_n)
        plot_player_ranking_points_role(df, hd / "reports" / "coach_report" / "figures" / "player_ranking_points_lead.png", role="lead", top_n=top_n)
        plot_player_ranking_points_role(df, hd / "reports" / "coach_report" / "figures" / "player_ranking_points_skip.png", role="skip", top_n=top_n)
        print(f"[done] wrote {out_path}")


if __name__ == "__main__":
    main()
