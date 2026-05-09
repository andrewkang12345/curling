#!/usr/bin/env python3
"""Visualize top 10 execution value surplus shots as prev→next board state pairs."""

import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

NUM_STONES = 12
POS_MAX = 4095.0
BUTTON_X, BUTTON_Y = 750, 800
HOUSE_RADII = [200, 400, 600]
SHOT_KEY = ["CompetitionID", "SessionID", "GameID", "EndID", "ShotID"]

BASE = pathlib.Path("/mnt/data/curling2/csas_fixed")


def load_player_names():
    comps = pd.read_csv(BASE / "2026" / "Competitors.csv")
    comps["player_ord"] = comps.groupby(["CompetitionID", "TeamID"]).cumcount() + 1
    return {(int(r.CompetitionID), int(r.TeamID), int(r.player_ord)): str(r.Reportingname)
            for _, r in comps.iterrows() if pd.notna(r.CompetitionID)}


def load_team_names():
    teams = pd.read_csv(BASE / "2026" / "Teams.csv")
    if "Name" not in teams.columns:
        return {}
    return {(int(r.CompetitionID), int(r.TeamID)): str(r.Name)
            for _, r in teams.iterrows() if pd.notna(r.CompetitionID)}


TASK_NAME = {0: "Draw", 1: "Front", 2: "Guard", 3: "Raise/Tap", 4: "Wick/Peel",
             5: "Freeze", 6: "Take-out", 7: "Hit&Roll", 8: "Clearing",
             9: "Dbl Take-out", 10: "Promo Take-out", 11: "Through"}


def draw_house(ax):
    for r in HOUSE_RADII:
        ax.add_patch(plt.Circle((BUTTON_X, BUTTON_Y), r, fill=False, color="gray", lw=0.8, ls="--"))
    ax.plot(BUTTON_X, BUTTON_Y, "k+", ms=6, mew=1)
    ax.axvline(BUTTON_X, color="gray", lw=0.3, alpha=0.5)
    ax.set_xlim(-50, 1550)
    ax.set_ylim(-50, 1650)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])


def draw_stones(ax, inv_row, prefix, team_a, team_b):
    colors = {team_a: "#E74C3C", team_b: "#3498DB"}
    for i in range(1, 13):
        xc, yc = f"{prefix}_stone_{i}_x_m", f"{prefix}_stone_{i}_y_m"
        if xc not in inv_row or pd.isna(inv_row[xc]):
            continue
        # Convert metres to raw coords (316 raw units per metre, button at 750,800)
        x_m, y_m = float(inv_row[xc]), float(inv_row[yc])
        M2RAW = 316.0
        x_raw = BUTTON_X + x_m * M2RAW
        y_raw = BUTTON_Y - y_m * M2RAW  # y inverted
        team = team_a if i <= 6 else team_b
        ax.add_patch(plt.Circle((x_raw, y_raw), 46, color=colors.get(team, "gray"), alpha=0.7, zorder=5))
        ax.text(x_raw, y_raw, str(i), ha="center", va="center", fontsize=6, color="white", fontweight="bold", zorder=6)


def main():
    plt.rcParams.update({"figure.dpi": 140, "savefig.dpi": 300})

    # Load scores
    local = pd.read_csv(BASE / "holdouts" / "0" / "scoring" / "shot_scores_local.csv")
    local["execution_value"] = local["dv_obs"] - local["dv_mean"]
    top10 = local.dropna(subset=["execution_value"]).sort_values("execution_value", ascending=False).head(10)

    # Load inverse data for stone positions
    import glob
    inv_paths = sorted(glob.glob(str(BASE / "inverse_current" / "stones_with_estimates.chunk*.csv")))
    inv_df = pd.concat([pd.read_csv(p) for p in inv_paths], ignore_index=True)

    # Load names
    player_names = load_player_names()
    team_names = load_team_names()

    # Get opponent teams per end from Stones.csv
    stones = pd.read_csv(BASE / "2026" / "Stones.csv")
    comp0 = stones[stones["CompetitionID"] == 0]

    fig, axes = plt.subplots(10, 2, figsize=(8, 40))

    for row_idx, (_, shot) in enumerate(top10.iterrows()):
        # Find inverse row
        mask = True
        for k in SHOT_KEY:
            mask = mask & (inv_df[k] == shot[k])
        inv_row = inv_df[mask]
        if inv_row.empty:
            continue
        inv_row = inv_row.iloc[0]

        sid = int(shot["ShotID"])
        tid = int(shot["TeamID"])
        pid = int(shot["PlayerID"])
        task = int(shot.get("Task", 0))
        exec_val = float(shot["execution_value"])
        dv_obs = float(shot["dv_obs"])

        pname = player_names.get((0, tid, pid), f"Player {pid}")
        tname = team_names.get((0, tid), f"Team {tid}")

        # Find opponent team in this end
        end_shots = comp0[(comp0["SessionID"] == shot["SessionID"]) &
                          (comp0["GameID"] == shot["GameID"]) &
                          (comp0["EndID"] == shot["EndID"])]
        teams = end_shots["TeamID"].unique()
        opp = [t for t in teams if t != tid]
        opp_tid = opp[0] if opp else 0
        first_team = int(end_shots.sort_values("ShotID").iloc[0]["TeamID"])

        # Determine team A (stones 1-6) and team B (stones 7-12)
        team_a = first_team
        team_b = tid if first_team != tid else opp_tid

        # Draw prev state
        ax_prev = axes[row_idx][0]
        draw_house(ax_prev)
        draw_stones(ax_prev, inv_row, "prev", team_a, team_b)
        ax_prev.set_title(f"Before shot {sid}", fontsize=8)
        if row_idx == 0:
            ax_prev.text(0.5, 1.08, "PREV state", transform=ax_prev.transAxes, ha="center", fontsize=9, fontweight="bold")

        # Draw next state
        ax_next = axes[row_idx][1]
        draw_house(ax_next)
        draw_stones(ax_next, inv_row, "next", team_a, team_b)
        ax_next.set_title(f"After shot {sid}", fontsize=8)
        if row_idx == 0:
            ax_next.text(0.5, 1.08, "NEXT state", transform=ax_next.transAxes, ha="center", fontsize=9, fontweight="bold")

        # Label
        task_str = TASK_NAME.get(task, f"Task {task}")
        hard_loss = float(inv_row.get("hard_loss_refine", float("nan")))
        hl_str = f"{hard_loss:.4f}" if np.isfinite(hard_loss) else "N/A"
        label = (f"#{row_idx+1}: {pname} ({tname})\n"
                 f"S{int(shot['SessionID'])}G{int(shot['GameID'])}E{int(shot['EndID'])} Shot{sid} | {task_str}\n"
                 f"exec={exec_val:+.2f} | ΔV={dv_obs:+.2f} | loss={hl_str}")
        ax_prev.text(-0.05, 0.5, label, transform=ax_prev.transAxes, ha="right", va="center",
                     fontsize=7, bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="gray", alpha=0.9))

    fig.suptitle("Top 10 Execution Value Surplus Shots — Beijing 2022", fontsize=14, fontweight="bold", y=1.001)
    fig.tight_layout()
    out_path = BASE / "holdouts" / "0" / "reports" / "coach_report" / "figures" / "top10_execution_surplus.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
