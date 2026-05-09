#!/usr/bin/env python3
"""
Augment curling throw data with pre-placed mixed doubles stone states.

For each end, creates a synthetic ShotID=7 row with PRISTINE pre-placed
stone positions (before any shot is thrown).

IMPORTANT: Stones.csv ShotID=7 and inverse ShotID=8 prev_stone_* both
contain the POST-first-throw state. The first shot disturbs the in-house
stone ~22% of the time. This script uses canonical rule-defined positions
instead, determined by:
  - Ends.csv PowerPlay column (standard vs power play)
  - Which slot (1 vs 7) owns the guard (inferred from post-shot-7 guard
    position, which is disturbed only ~4% of the time)

Canonical positions (meters, tee = origin):
  Standard:   guard (-3.4016, 0.0), in-house (+0.4572, 0.0)
  PP right:   guard (-3.4016, +1.0333), in-house (-0.1524, +1.2192)
  PP left:    guard (-3.4016, -1.0333), in-house (-0.1524, -1.2192)
"""

import os
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INVERSE_CSV = os.path.join(BASE_DIR, "inverse_current", "stones_with_estimates.chunk0000.csv")
STONES_CSV = os.path.join(BASE_DIR, "2026", "Stones.csv")
ENDS_CSV = os.path.join(BASE_DIR, "2026", "Ends.csv")
OUT_DIR = os.path.join(BASE_DIR, "preplaced_augment")
os.makedirs(OUT_DIR, exist_ok=True)

# Canonical pre-placed positions (meters, tee = origin)
CANON = {
    "standard": {
        "guard": (-3.4016, 0.0),
        "inhouse": (0.4572, 0.0),
    },
    "pp_right": {
        "guard": (-3.4016, 1.0333),
        "inhouse": (-0.1524, 1.2192),
    },
    "pp_left": {
        "guard": (-3.4016, -1.0333),
        "inhouse": (-0.1524, -1.2192),
    },
}

GUARD_X_THRESHOLD = -2.0  # stones with x_m < this are guards


def csv_to_meters(x_csv, y_csv):
    return (800.0 - y_csv) * 0.003048, (x_csv - 750.0) * 0.003048


def main():
    print("Loading data...")
    inverse = pd.read_csv(INVERSE_CSV)
    stones_raw = pd.read_csv(STONES_CSV)
    ends_raw = pd.read_csv(ENDS_CSV)

    EK = ["CompetitionID", "SessionID", "GameID", "EndID"]

    # Power play info from Ends.csv
    pp_map = {}
    for _, r in ends_raw.iterrows():
        key = (r["CompetitionID"], r["SessionID"], r["GameID"], r["EndID"])
        pp_val = r.get("PowerPlay", np.nan)
        if pd.notna(pp_val):
            pp_map[key] = int(pp_val)  # 1=right, 2=left

    # Get post-shot-7 state from Stones.csv to determine guard ownership
    s7 = stones_raw[stones_raw["ShotID"] == 7].copy()
    s7["s1_xm"], s7["s1_ym"] = csv_to_meters(s7["stone_1_x"].values, s7["stone_1_y"].values)
    s7["s7_xm"], s7["s7_ym"] = csv_to_meters(s7["stone_7_x"].values, s7["stone_7_y"].values)

    # Build lookup: end_key -> {guard_slot, mode, pp_type}
    end_info = {}
    for _, row in s7.iterrows():
        key = (row["CompetitionID"], row["SessionID"], row["GameID"], row["EndID"])
        s1_x, s7_x = row["s1_xm"], row["s7_xm"]

        # Skip off-sheet sentinels
        if s1_x < -9.0 or s7_x < -9.0:
            end_info[key] = {"mode": "offsheet", "guard_slot": None, "pp_type": None}
            continue

        pp_type = pp_map.get(key, None)

        # Determine guard slot: the stone with lower x_m (further from house)
        # Guard is rarely disturbed by first shot (~4%), so this is reliable
        s1_is_guard = s1_x < s7_x
        if s1_x < GUARD_X_THRESHOLD and s7_x >= GUARD_X_THRESHOLD:
            guard_slot = 1
        elif s7_x < GUARD_X_THRESHOLD and s1_x >= GUARD_X_THRESHOLD:
            guard_slot = 7
        elif s1_is_guard:
            guard_slot = 1  # fallback: whichever is further
        else:
            guard_slot = 7

        # Detect mode using PP flag OR lateral stone positions.
        # 227 PP ends have missing PowerPlay flag in Ends.csv, so we must
        # also check geometry: off-center guard OR in-house at tee = power play.
        guard_y = row["s1_ym"] if guard_slot == 1 else row["s7_ym"]
        inhouse_y = row["s7_ym"] if guard_slot == 1 else row["s1_ym"]
        inhouse_x = row["s7_xm"] if guard_slot == 1 else row["s1_xm"]

        if pp_type is not None:
            # Explicit PP flag in Ends.csv
            mode = "pp_right" if pp_type == 1 else "pp_left"
        elif abs(guard_y) > 0.5 or abs(inhouse_y) > 0.5:
            # Off-center stones → power play with missing flag
            side_y = guard_y if abs(guard_y) > abs(inhouse_y) else inhouse_y
            mode = "pp_right" if side_y > 0 else "pp_left"
        elif inhouse_x < 0.0 and abs(inhouse_y) < 0.3:
            # In-house at tee (x≈-0.15) but on-center — likely PP with missing
            # flag where first shot also moved guard toward center, OR a true
            # anomaly. Use standard as fallback since guard is on center.
            mode = "standard"
        else:
            mode = "standard"

        end_info[key] = {"mode": mode, "guard_slot": guard_slot, "pp_type": pp_type}

    # Build synthetic rows using CANONICAL positions
    end_keys_inv = inverse[inverse["ShotID"] == 8][EK].drop_duplicates()

    synth_rows = []
    classifications = []
    n_matched = 0
    n_skipped = 0
    n_offsheet = 0

    for _, ek in end_keys_inv.iterrows():
        key = (ek["CompetitionID"], ek["SessionID"], ek["GameID"], ek["EndID"])
        info = end_info.get(key)
        if info is None:
            n_skipped += 1
            continue
        if info["mode"] == "offsheet":
            n_offsheet += 1
            classifications.append({
                **dict(zip(EK, key)),
                "mode": "offsheet", "guard_slot": np.nan,
                "guard_x_m": np.nan, "guard_y_m": np.nan,
                "inhouse_x_m": np.nan, "inhouse_y_m": np.nan,
                "pp_type": np.nan, "notes": "off-sheet sentinel",
            })
            continue

        n_matched += 1
        mode = info["mode"]
        guard_slot = info["guard_slot"]
        inhouse_slot = 7 if guard_slot == 1 else 1

        canon = CANON[mode]
        gx, gy = canon["guard"]
        ix, iy = canon["inhouse"]

        # Classification record
        classifications.append({
            **dict(zip(EK, key)),
            "mode": mode,
            "guard_slot": guard_slot,
            "guard_x_m": gx, "guard_y_m": gy,
            "inhouse_x_m": ix, "inhouse_y_m": iy,
            "pp_type": info["pp_type"],
            "notes": "",
        })

        # Synthetic inverse row
        synth = {}
        synth["CompetitionID"] = ek["CompetitionID"]
        synth["SessionID"] = ek["SessionID"]
        synth["GameID"] = ek["GameID"]
        synth["EndID"] = ek["EndID"]
        synth["ShotID"] = 7
        synth["prev_N"] = 0
        synth["next_total_N"] = 2
        synth["next_in_bounds_N"] = 2

        for col in ["est_speed", "est_angle", "est_spin", "est_y0",
                     "hard_loss_coarse", "hard_loss_refine",
                     "objective_hard_loss_coarse", "objective_hard_loss_refine",
                     "warm_start_init_hard"]:
            synth[col] = np.nan
        synth["used_coarse_fallback"] = np.nan
        synth["solver_method"] = "preplaced"
        synth["loss_variant"] = np.nan
        synth["solver_ok"] = True
        synth["solver_error"] = np.nan

        # All prev_stone = NaN (nothing before pre-placement)
        for i in range(1, 13):
            synth[f"prev_stone_{i}_x_m"] = np.nan
            synth[f"prev_stone_{i}_y_m"] = np.nan

        # Next stones: only guard and in-house at canonical positions
        for i in range(1, 13):
            synth[f"next_stone_{i}_x_m"] = np.nan
            synth[f"next_stone_{i}_y_m"] = np.nan
            synth[f"next_stone_{i}_inbounds"] = 0

        synth[f"next_stone_{guard_slot}_x_m"] = gx
        synth[f"next_stone_{guard_slot}_y_m"] = gy
        synth[f"next_stone_{guard_slot}_inbounds"] = 1
        synth[f"next_stone_{inhouse_slot}_x_m"] = ix
        synth[f"next_stone_{inhouse_slot}_y_m"] = iy
        synth[f"next_stone_{inhouse_slot}_inbounds"] = 1

        synth_rows.append(synth)

    synth_df = pd.DataFrame(synth_rows)
    for col in inverse.columns:
        if col not in synth_df.columns:
            synth_df[col] = np.nan
    synth_df = synth_df[inverse.columns]

    augmented = pd.concat([synth_df, inverse], ignore_index=True)
    augmented = augmented.sort_values(
        ["CompetitionID", "SessionID", "GameID", "EndID", "ShotID"]
    ).reset_index(drop=True)

    class_df = pd.DataFrame(classifications)

    # Save
    aug_path = os.path.join(OUT_DIR, "augmented_throws.csv")
    augmented.to_csv(aug_path, index=False)
    print(f"Saved: {aug_path} ({len(synth_df)} synthetic + {len(inverse)} original = {len(augmented)} total)")

    class_path = os.path.join(OUT_DIR, "preplaced_classification.csv")
    class_df.to_csv(class_path, index=False)
    print(f"Saved: {class_path} ({len(class_df)} rows)")

    # Report
    lines = []
    lines.append("=" * 70)
    lines.append("PRE-PLACED STONE CLASSIFICATION REPORT (CANONICAL POSITIONS)")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"Total ends matched:  {n_matched}")
    lines.append(f"Skipped (no match):  {n_skipped}")
    lines.append(f"Off-sheet:           {n_offsheet}")
    lines.append(f"Synthetic rows:      {len(synth_df)}")
    lines.append(f"Original rows:       {len(inverse)}")
    lines.append(f"Augmented total:     {len(augmented)}")
    lines.append("")
    lines.append("MODE BREAKDOWN")
    lines.append("-" * 40)
    for mode, cnt in class_df["mode"].value_counts().items():
        lines.append(f"  {mode:20s}: {cnt:5d}")
    lines.append("")
    lines.append("GUARD SLOT OWNERSHIP")
    lines.append("-" * 40)
    valid = class_df[class_df["mode"] != "offsheet"]
    for slot, cnt in sorted(valid["guard_slot"].value_counts().items()):
        team = "A (stones 1-6)" if slot == 1 else "B (stones 7-12)"
        lines.append(f"  Slot {int(slot)} — team {team}: {cnt}")
    lines.append("")
    lines.append("CANONICAL POSITIONS USED (meters, tee = origin)")
    lines.append("-" * 40)
    for mode, pos in CANON.items():
        lines.append(f"  {mode}:")
        lines.append(f"    guard:   ({pos['guard'][0]:.4f}, {pos['guard'][1]:.4f})")
        lines.append(f"    inhouse: ({pos['inhouse'][0]:.4f}, {pos['inhouse'][1]:.4f})")
    lines.append("")
    lines.append("NOTE: These are rule-defined positions, NOT the post-shot-7")
    lines.append("observed positions. The first thrown shot disturbs the in-house")
    lines.append("stone ~22% of the time, so using post-shot positions would be wrong.")
    lines.append("")
    lines.append("LIMITATIONS")
    lines.append("-" * 40)
    lines.append("- Rules allow 3 reference points for Position A (guard), each with")
    lines.append("  front/back = 6 standard layouts. This dataset shows only 1 guard")
    lines.append("  position (x_m=-3.40). Either teams always chose the same ref point,")
    lines.append("  or pixel resolution masks the front/back distinction (~0.29m apart).")
    lines.append("- ~227 PP ends have missing PowerPlay flag in Ends.csv; detected via")
    lines.append("  lateral stone positions (off-center guard or in-house).")
    lines.append("- ~24 ends have center guard + at-tee in-house with no PP flag;")
    lines.append("  classified as standard (guard was likely disturbed by first shot).")
    lines.append("- ~68 ends have guard disturbed far from canonical position.")
    lines.append("")
    lines.append("=" * 70)

    report = "\n".join(lines)
    report_path = os.path.join(OUT_DIR, "preplaced_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Saved: {report_path}")
    print()
    print(report)


if __name__ == "__main__":
    main()
