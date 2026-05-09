#!/usr/bin/env python3
"""
Generate extra figures for holdout 0:
1. Bar chart of unscored (failed inverse) shots by player
2. End progression visualizations showing stone states + values for Bruce Mouat ends
"""

import pathlib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches

NUM_STONES = 12
POS_MAX = 4095.0
SHOT_KEY = ["CompetitionID", "SessionID", "GameID", "EndID", "ShotID"]

# Curling sheet geometry (raw coords)
BUTTON_X, BUTTON_Y = 750, 800
HOUSE_RADII = [200, 400, 600]  # 4ft, 8ft, 12ft rings
SHEET_X_MIN, SHEET_X_MAX = 0, 1500
SHEET_Y_MIN, SHEET_Y_MAX = 0, 1600  # focus on house area


def _load_player_names(base_dir):
    comps = pd.read_csv(base_dir / "2026" / "Competitors.csv")
    comps["player_ord"] = comps.groupby(["CompetitionID", "TeamID"]).cumcount() + 1
    return {(int(r.CompetitionID), int(r.TeamID), int(r.player_ord)): str(r.Reportingname)
            for _, r in comps.iterrows()
            if pd.notna(r.CompetitionID) and pd.notna(r.TeamID)}


def _load_team_names(base_dir):
    teams = pd.read_csv(base_dir / "2026" / "Teams.csv")
    if "Name" not in teams.columns:
        return {}
    return {(int(r.CompetitionID), int(r.TeamID)): str(r.Name)
            for _, r in teams.iterrows()
            if pd.notna(r.CompetitionID) and pd.notna(r.TeamID)}


def _set_style():
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update({"figure.dpi": 140, "savefig.dpi": 300,
                          "axes.spines.top": False, "axes.spines.right": False})


# ─────────────────────────────────────────────────────────────
# 1. Unscored shots bar chart
# ─────────────────────────────────────────────────────────────

def plot_unscored_by_player(base_dir, holdout_dir, out_path):
    stones = pd.read_csv(base_dir / "2026" / "Stones.csv")
    comp0 = stones[stones["CompetitionID"] == 0].copy()
    scored = pd.read_csv(holdout_dir / "scoring" / "shot_scores_local.csv")

    # Find unscored
    merged = comp0.merge(scored[SHOT_KEY].drop_duplicates(), on=SHOT_KEY, how="left", indicator=True)
    unscored = merged[merged["_merge"] == "left_only"].copy()
    total_by_player = comp0.groupby(["TeamID", "PlayerID"]).size().reset_index(name="total_shots")

    player_names = _load_player_names(base_dir)
    team_names = _load_team_names(base_dir)

    # Count unscored per player
    un_counts = unscored.groupby(["TeamID", "PlayerID"]).size().reset_index(name="unscored")
    un_counts = un_counts.merge(total_by_player, on=["TeamID", "PlayerID"], how="left")
    un_counts["pct"] = (un_counts["unscored"] / un_counts["total_shots"] * 100).round(1)

    # Add names
    def get_label(row):
        pname = player_names.get((0, int(row["TeamID"]), int(row["PlayerID"])), f"Player {int(row['PlayerID'])}")
        tname = team_names.get((0, int(row["TeamID"])), f"Team {int(row['TeamID'])}")
        return f"{pname} ({tname})"

    un_counts["player_label"] = un_counts.apply(get_label, axis=1)
    un_counts = un_counts.sort_values("unscored", ascending=False)

    fig, ax = plt.subplots(figsize=(14, 8))
    bars = sns.barplot(data=un_counts, x="unscored", y="player_label", ax=ax, orient="h", alpha=0.95)
    ax.set_title("Unscored shots by player (failed inverse solution)")
    ax.set_xlabel("Number of unscored shots")
    ax.set_ylabel("")

    for i, r in enumerate(un_counts.itertuples(index=False)):
        ax.text(float(r.unscored), i,
                f"  {int(r.unscored)}/{int(r.total_shots)} ({r.pct}%)",
                va="center", fontsize=10)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


# ─────────────────────────────────────────────────────────────
# 2. End progression visualization
# ─────────────────────────────────────────────────────────────

def _draw_curling_house(ax):
    """Draw the curling house (rings + button)."""
    for r in HOUSE_RADII:
        circle = plt.Circle((BUTTON_X, BUTTON_Y), r, fill=False, color="gray", linewidth=1, linestyle="--")
        ax.add_patch(circle)
    ax.plot(BUTTON_X, BUTTON_Y, "k+", markersize=8, markeredgewidth=1.5)
    # Centerline
    ax.axvline(BUTTON_X, color="gray", linewidth=0.5, alpha=0.5)
    ax.set_xlim(SHEET_X_MIN - 50, SHEET_X_MAX + 50)
    ax.set_ylim(SHEET_Y_MIN - 50, SHEET_Y_MAX + 50)
    ax.set_aspect("equal")
    ax.invert_yaxis()  # y increases downward (toward hog line)


def _draw_stones(ax, row, team_a_id, team_b_id):
    """Draw stones on the sheet, colored by team."""
    colors = {team_a_id: "#E74C3C", team_b_id: "#3498DB"}  # red vs blue
    stone_r = 15

    for i in range(1, NUM_STONES + 1):
        x = float(row.get(f"stone_{i}_x", 0))
        y = float(row.get(f"stone_{i}_y", 0))
        if (x <= 0 and y <= 0) or (x >= POS_MAX and y >= POS_MAX):
            continue  # not in play or dead
        team_id = team_a_id if i <= 6 else team_b_id
        color = colors.get(team_id, "gray")
        circle = plt.Circle((x, y), stone_r, color=color, alpha=0.7, zorder=5)
        ax.add_patch(circle)
        ax.text(x, y, str(i), ha="center", va="center", fontsize=7, color="white", fontweight="bold", zorder=6)


def plot_end_progression(end_shots, scored_end, team_a_id, team_b_id, player_names, team_names, out_path):
    """
    Visualize progression of an end: one panel per shot showing board state + value annotations.
    """
    n_shots = len(end_shots)
    cols = min(5, n_shots)
    rows = (n_shots + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 5 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)

    team_a_name = team_names.get((0, team_a_id), f"Team {team_a_id}")
    team_b_name = team_names.get((0, team_b_id), f"Team {team_b_id}")

    # Build a lookup for scored data
    scored_lookup = {}
    if scored_end is not None and len(scored_end) > 0:
        for _, sr in scored_end.iterrows():
            scored_lookup[int(sr["ShotID"])] = sr

    for idx, (_, shot) in enumerate(end_shots.iterrows()):
        r, c = idx // cols, idx % cols
        ax = axes[r][c]
        _draw_curling_house(ax)
        _draw_stones(ax, shot, team_a_id, team_b_id)

        shot_id = int(shot["ShotID"])
        team_id = int(shot["TeamID"])
        player_id = int(shot["PlayerID"])
        task = int(shot.get("Task", -1))
        points = shot.get("Points", "")

        pname = player_names.get((0, team_id, player_id), f"P{player_id}")
        team_color = "#E74C3C" if team_id == team_a_id else "#3498DB"

        # Title with shot info
        title = f"Shot {shot_id}\n{pname}"
        ax.set_title(title, fontsize=9, color=team_color, fontweight="bold")

        # Value annotation
        if shot_id in scored_lookup:
            sd = scored_lookup[shot_id]
            v_next = float(sd["v_next"])
            dv_obs = float(sd["dv_obs"])
            dv_sign = "+" if dv_obs >= 0 else ""
            ax.text(0.02, 0.02, f"V={v_next:.2f}\nΔV={dv_sign}{dv_obs:.2f}\nPts={points}",
                    transform=ax.transAxes, fontsize=8, va="bottom",
                    bbox=dict(boxstyle="round,pad=0.2", fc="lightyellow", ec="gray", alpha=0.9))
        else:
            ax.text(0.02, 0.02, f"(not scored)\nPts={points}",
                    transform=ax.transAxes, fontsize=8, va="bottom",
                    bbox=dict(boxstyle="round,pad=0.2", fc="mistyrose", ec="gray", alpha=0.9))

        ax.set_xticks([])
        ax.set_yticks([])

    # Hide unused axes
    for idx in range(n_shots, rows * cols):
        r, c = idx // cols, idx % cols
        axes[r][c].set_visible(False)

    session = int(end_shots.iloc[0]["SessionID"])
    game = int(end_shots.iloc[0]["GameID"])
    end_id = int(end_shots.iloc[0]["EndID"])
    fig.suptitle(f"End progression — Session {session}, Game {game}, End {end_id}\n"
                 f"{team_a_name} (red) vs {team_b_name} (blue)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def main():
    _set_style()
    base_dir = pathlib.Path("/mnt/data/curling2/csas_fixed")
    holdout_dir = base_dir / "holdouts" / "0"
    fig_dir = holdout_dir / "reports" / "coach_report" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("Generating unscored shots bar chart...", flush=True)
    plot_unscored_by_player(base_dir, holdout_dir, fig_dir / "unscored_shots_by_player.png")

    # Load data for end progression
    print("\nGenerating end progression visualizations...", flush=True)
    stones = pd.read_csv(base_dir / "2026" / "Stones.csv")
    comp0 = stones[stones["CompetitionID"] == 0].copy()
    scored = pd.read_csv(holdout_dir / "scoring" / "shot_scores_local.csv")
    player_names = _load_player_names(base_dir)
    team_names = _load_team_names(base_dir)

    # Bruce Mouat is TeamID=27. Pick 3 diverse ends from different games.
    # Team 27 opponents vary by game. Find the opponent team for each end.
    team27_ends = comp0[comp0["TeamID"] == 27][["SessionID", "GameID", "EndID"]].drop_duplicates()

    # Pick 3 ends from different games with high scoring coverage
    end_candidates = []
    for _, erow in team27_ends.iterrows():
        mask = ((comp0["SessionID"] == erow["SessionID"]) &
                (comp0["GameID"] == erow["GameID"]) &
                (comp0["EndID"] == erow["EndID"]))
        end_shots = comp0[mask].sort_values("ShotID")
        n_scored = end_shots.merge(scored[SHOT_KEY].drop_duplicates(), on=SHOT_KEY, how="inner").shape[0]
        teams = end_shots["TeamID"].unique().tolist()
        opp_team = [t for t in teams if t != 27]
        opp_id = opp_team[0] if opp_team else 0
        end_candidates.append({
            "SessionID": erow["SessionID"], "GameID": erow["GameID"],
            "EndID": erow["EndID"], "n_scored": n_scored, "total": len(end_shots),
            "opp_team": opp_id
        })

    cands = pd.DataFrame(end_candidates).sort_values("n_scored", ascending=False)
    # Pick from different games
    seen_games = set()
    selected = []
    for _, c in cands.iterrows():
        gkey = (c["SessionID"], c["GameID"])
        if gkey not in seen_games and len(selected) < 3:
            selected.append(c)
            seen_games.add(gkey)

    for i, sel in enumerate(selected):
        mask = ((comp0["SessionID"] == sel["SessionID"]) &
                (comp0["GameID"] == sel["GameID"]) &
                (comp0["EndID"] == sel["EndID"]))
        end_shots = comp0[mask].sort_values("ShotID")

        scored_end = scored.merge(end_shots[SHOT_KEY], on=SHOT_KEY, how="inner")

        opp_team = int(sel["opp_team"])
        # Determine which team throws first in this end (first ShotID's TeamID)
        first_team = int(end_shots.iloc[0]["TeamID"])
        team_a_id = first_team
        team_b_id = 27 if first_team != 27 else opp_team

        fname = f"end_progression_mouat_{i+1}_s{int(sel['SessionID'])}_g{int(sel['GameID'])}_e{int(sel['EndID'])}.png"
        plot_end_progression(end_shots, scored_end, team_a_id, team_b_id,
                             player_names, team_names, fig_dir / fname)

    print(f"\nDone! All figures in {fig_dir}", flush=True)


if __name__ == "__main__":
    main()
