import argparse
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import animation

from model import ValueTransformer  # assumes model.py is in same dir
from dataset import ValueDataset    # adjust module name if needed

# ---------------- Geometry constants (tied to your CSV + POS_MAX) ---------------- #

POS_MAX = 4095.0

# From your raw CSV description:
#   Centerline: x = 750
#   Backline:   y = 200
#   Hogline:    y = 2900
#   Button:     (x = 750, y = 800)
CENTERLINE_X_RAW = 750.0
BACKLINE_Y_RAW = 200.0
HOGLINE_Y_RAW = 2900.0
BUTTON_X_RAW = 750.0
BUTTON_Y_RAW = 800.0

# Approximate physical scale:
# backline -> hogline is ~27 ft in real curling
UNITS_PER_FOOT = (HOGLINE_Y_RAW - BACKLINE_Y_RAW) / 27.0  # ~100

# Use plausible sheet width: ~15 ft (half-width 7.5 ft) around centerline
HALF_WIDTH_FT = 7.5
SHEET_X_MIN = CENTERLINE_X_RAW - HALF_WIDTH_FT * UNITS_PER_FOOT
SHEET_X_MAX = CENTERLINE_X_RAW + HALF_WIDTH_FT * UNITS_PER_FOOT

# Use y-range from 2 ft behind backline to 4 ft beyond hogline
SHEET_Y_MIN = max(0.0, BACKLINE_Y_RAW - 2.0 * UNITS_PER_FOOT)
SHEET_Y_MAX = HOGLINE_Y_RAW + 4.0 * UNITS_PER_FOOT

# House radii in raw units (feet -> raw units)
R_12_RAW = 6.0 * UNITS_PER_FOOT   # 12-ft circle radius
R_8_RAW  = 4.0 * UNITS_PER_FOOT   # 8-ft circle radius
R_4_RAW  = 2.0 * UNITS_PER_FOOT   # 4-ft circle radius
R_BUTTON_RAW = 0.5 * UNITS_PER_FOOT  # button radius

# Stone radius (~11" diameter => radius ~0.46 ft)
STONE_R_RAW = 0.46 * UNITS_PER_FOOT

TEAM_A_COLOR = "#d62728"  # slots 1..6 (dataset convention)
TEAM_B_COLOR = "#1f77b4"  # slots 7..12 (dataset convention)


# ---------------- Model & dataset wiring ---------------- #

def build_model_from_ckpt(ckpt_path, device):
    """
    Reconstruct ValueTransformer using metadata stored in the checkpoint.
    """
    ckpt = torch.load(ckpt_path, map_location=device)

    input_dim = ckpt["input_dim"]
    cond_dim = ckpt["cond_dim"]
    hidden_dim = ckpt["hidden_dim"]
    num_stones = ckpt.get("num_stones", 12)

    # Pull architecture args if they exist, else fall back to defaults
    args_dict = ckpt.get("args", {})
    n_layers = args_dict.get("n_layers", 4)
    n_heads = args_dict.get("n_heads", 4)
    dropout = args_dict.get("dropout", 0.1)

    model = ValueTransformer(
        input_dim=input_dim,
        cond_dim=cond_dim,
        hidden_dim=hidden_dim,
        num_stones=num_stones,
        n_layers=n_layers,
        n_heads=n_heads,
        dropout=dropout,
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    num_tasks = ckpt.get("num_tasks", args_dict.get("num_tasks", 1))

    return model, num_stones, input_dim, cond_dim, num_tasks


def create_states_from_dataset(model, device, dataset, num_examples):
    """
    Sample states from a ValueDataset and run the model on them.

    Each state dict contains:
      - x:           (24,) stone positions (normalized 0..1)
      - c:           (cond_dim,) condition vector = [shot_norm, team_order, stone_block]
      - shot_norm:   float
      - value_pred:  model's predicted value
      - value_true:  ground-truth value_target from dataset
      - dataset_idx: index in the dataset (for reference)
    """
    num_examples = min(num_examples, len(dataset))
    indices = np.random.choice(len(dataset), size=num_examples, replace=False)

    states = []
    for idx in indices:
        # Dataset returns tensors: (x, c, y)
        x_tensor, c_tensor, y_tensor = dataset[idx]

        x_np = x_tensor.numpy()                 # (24,) normalized
        c_np = c_tensor.numpy()                 # (cond_dim,)
        y_true = float(y_tensor.numpy()[0])     # scalar

        x = x_tensor.unsqueeze(0).to(device)    # (1, 24)
        c = c_tensor.unsqueeze(0).to(device)    # (1, cond_dim)

        with torch.no_grad():
            value_pred = model(x, c).item()

        states.append(
            {
                "x": x_np,
                "c": c_np,
                "shot_norm": float(c_np[0]),
                "value_pred": value_pred,
                "value_true": y_true,
                "dataset_idx": int(idx),
            }
        )

    return states


# ---------------- Helpers ---------------- #

def stones_norm_to_raw(stones_norm):
    """
    stones_norm: (num_stones, 2) in [0,1]
    Returns: (num_stones, 2) in raw sheet coordinates (0..POS_MAX scale).
    """
    return stones_norm * POS_MAX


def is_dead_or_unthrown(x_raw, y_raw):
    """
    Returns True if the stone is not currently in play.
    In-play coordinates satisfy: 0 < x < POS_MAX and 0 < y < POS_MAX.
    """
    return not ((0.0 < x_raw < POS_MAX) and (0.0 < y_raw < POS_MAX))


def is_unthrown(x_raw, y_raw):
    # Dataset convention: untouched slot is (0, 0).
    return bool(np.isclose(x_raw, 0.0) and np.isclose(y_raw, 0.0))


def team_color_for_slot(slot_idx_0_based, num_stones):
    """
    Color by team block, not odd/even parity:
      - first half of slots => Team A color
      - second half => Team B color
    """
    split = num_stones // 2
    return TEAM_A_COLOR if slot_idx_0_based < split else TEAM_B_COLOR


def team_stone_counts(stones_raw, num_stones):
    """
    Returns per-team counts:
      left    = stones currently in play
      thrown  = stones already thrown (in-play + out-of-play)
    """
    split = num_stones // 2
    team_a = stones_raw[:split]
    team_b = stones_raw[split:]

    def _counts(team_xy):
        left = 0
        thrown = 0
        for x_raw, y_raw in team_xy:
            if not is_unthrown(float(x_raw), float(y_raw)):
                thrown += 1
            if not is_dead_or_unthrown(float(x_raw), float(y_raw)):
                left += 1
        return left, thrown

    a_left, a_thrown = _counts(team_a)
    b_left, b_thrown = _counts(team_b)
    return a_left, a_thrown, b_left, b_thrown


def format_bottom_counts(stones_raw, num_stones):
    a_left, a_thrown, b_left, b_thrown = team_stone_counts(stones_raw, num_stones)
    team_cap = num_stones // 2
    return (
        f"Red left/thrown: {a_left}/{a_thrown} (remaining {max(0, team_cap - a_thrown)})"
        f"  |  "
        f"Blue left/thrown: {b_left}/{b_thrown} (remaining {max(0, team_cap - b_thrown)})"
    )


# ---------------- Visualization helpers ---------------- #

def setup_rink_ax():
    """
    Set up a top-down curling sheet in RAW model coordinates.

    - x, y are in same units as your CSV.
    - Axis limits chosen to show a sheet segment around one house with realistic aspect.
    - Sheet border, house, lines drawn in raw coordinates.
    - No labels on any lines.
    """
    sheet_width = SHEET_X_MAX - SHEET_X_MIN
    sheet_height = SHEET_Y_MAX - SHEET_Y_MIN
    aspect = sheet_height / sheet_width

    # Make figure roughly match aspect ratio (taller than wide)
    fig_width = 4.0
    fig_height = fig_width * aspect
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    ax.set_aspect("equal", "box")
    ax.set_xlim(SHEET_X_MIN, SHEET_X_MAX)
    ax.set_ylim(SHEET_Y_MIN, SHEET_Y_MAX)
    ax.set_xticks([])
    ax.set_yticks([])

    # --- Sheet border ---
    rect = plt.Rectangle(
        (SHEET_X_MIN, SHEET_Y_MIN),
        sheet_width,
        sheet_height,
        linewidth=1.0,
        edgecolor="black",
        facecolor="none",
        zorder=0,
    )
    ax.add_patch(rect)

    # --- House & button at real button position ---
    center = (BUTTON_X_RAW, BUTTON_Y_RAW)

    # Scoring rings: 4 ft, 8 ft, 12 ft
    radii = [R_4_RAW, R_8_RAW, R_12_RAW]
    colors = ["#ccccff", "#eeeeee", "#dddddd"]

    for r, col in zip(radii, colors):
        circ = plt.Circle(
            center,
            r,
            edgecolor="black",
            facecolor=col,
            alpha=0.8,
            zorder=1,
        )
        ax.add_patch(circ)

    # Button (inner circle)
    button = plt.Circle(
        center,
        R_BUTTON_RAW,
        edgecolor="black",
        facecolor="#88ccee",
        zorder=2,
    )
    ax.add_patch(button)

    # --- Lines: centerline, backline, hogline, tee line (no labels) ---

    # Centerline (vertical)
    ax.plot(
        [CENTERLINE_X_RAW, CENTERLINE_X_RAW],
        [SHEET_Y_MIN, SHEET_Y_MAX],
        linestyle="--",
        color="black",
        linewidth=1.0,
        zorder=0.5,
    )

    # Backline
    ax.axhline(
        BACKLINE_Y_RAW,
        linestyle="-",
        color="black",
        linewidth=1.0,
        zorder=0.5,
    )

    # Hogline
    ax.axhline(
        HOGLINE_Y_RAW,
        linestyle="-",
        color="black",
        linewidth=1.0,
        zorder=0.5,
    )

    # Tee line
    ax.axhline(
        BUTTON_Y_RAW,
        linestyle="--",
        color="black",
        linewidth=1.0,
        zorder=0.5,
    )

    return fig, ax


def visualize_states(states, num_stones, out_path, fps=2):
    """
    Create an MP4 where each frame corresponds to an actual validation example:

      - Stones drawn as circular patches with radius STONE_R_RAW in raw coords.
      - Whole sheet (border, lines, house, button) is drawn in raw coords.
      - Stones at (0,0), (0, POS_MAX), (POS_MAX, POS_MAX) are NOT drawn.
      - Top label (outside sheet) shows example index + pred/true.
      - Bottom label shows per-team stones left/thrown.
    """
    fig, ax = setup_rink_ax()

    # Team colors by slot block (dataset convention: 1..6 vs 7..12)
    stone_colors = [team_color_for_slot(i, num_stones) for i in range(num_stones)]

    # Pre-create stone circle patches, to be updated each frame
    stone_patches = []
    for i in range(num_stones):
        patch = plt.Circle(
            (BUTTON_X_RAW, BUTTON_Y_RAW),
            STONE_R_RAW,
            facecolor=stone_colors[i],
            edgecolor="black",
            zorder=5,
        )
        patch.set_visible(False)  # hidden until we place them
        ax.add_patch(patch)
        stone_patches.append(patch)

    # Text outside the sheet, in *figure* coordinates so it never overlaps the sheet
    text_main = fig.text(
        0.5,
        0.96,   # near top of figure
        "",
        ha="center",
        va="top",
        fontsize=11,
    )

    text_cond = fig.text(
        0.5,
        0.04,   # near bottom of figure
        "",
        ha="center",
        va="bottom",
        fontsize=10,
    )

    def init():
        for p in stone_patches:
            p.set_visible(False)
        text_main.set_text("")
        text_cond.set_text("")
        return stone_patches + [text_main, text_cond]

    def update(frame_idx):
        state = states[frame_idx]
        x_flat_norm = state["x"]  # (2 * num_stones,) normalized
        stones_norm = x_flat_norm.reshape(num_stones, 2)
        stones_raw = stones_norm_to_raw(stones_norm)

        value_pred = state["value_pred"]
        value_true = state["value_true"]
        dataset_idx = state["dataset_idx"]

        # Update stone positions (skip dead/unthrown)
        for i, p in enumerate(stone_patches):
            x_raw, y_raw = stones_raw[i, 0], stones_raw[i, 1]
            if is_dead_or_unthrown(x_raw, y_raw):
                p.set_visible(False)
            else:
                p.center = (x_raw, y_raw)
                p.set_visible(True)

        text_main.set_text(
            f"Example {frame_idx + 1}/{len(states)} (idx {dataset_idx})  "
            f"Pred: {value_pred:.2f}, True: {value_true:.2f}"
        )
        text_cond.set_text(format_bottom_counts(stones_raw, num_stones))

        return stone_patches + [text_main, text_cond]

    anim = animation.FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=len(states),
        interval=1000 / fps,
        blit=True,
    )

    print(f"Saving animation to {out_path} ...")
    try:
        Writer = animation.writers["ffmpeg"]
        writer = Writer(fps=fps, metadata={"artist": "viz"})
        anim.save(out_path, writer=writer)
        print("Done.")
    except RuntimeError as e:
        print(f"FFmpeg writer failed: {e}")
        base, _ = os.path.splitext(out_path)
        gif_path = base + ".gif"
        print(f"Falling back to GIF: {gif_path}")
        anim.save(gif_path, writer="pillow", fps=fps)
        print("Done (GIF).")

    plt.close(fig)


def save_states_as_png(states, num_stones, out_dir):
    """
    Save each state as an individual PNG image.

    Stones and full sheet are drawn in raw coordinates with consistent geometry.
    Stones not in play are not drawn.
    Labels are placed outside the sheet (top and bottom) in figure space.
    """
    os.makedirs(out_dir, exist_ok=True)

    for idx, state in enumerate(states):
        fig, ax = setup_rink_ax()

        x_flat_norm = state["x"]
        stones_norm = x_flat_norm.reshape(num_stones, 2)
        stones_raw = stones_norm_to_raw(stones_norm)

        stone_colors = [team_color_for_slot(i, num_stones) for i in range(num_stones)]

        # Draw stones as circles with radius STONE_R_RAW, skipping dead/unthrown
        for i in range(num_stones):
            x_raw, y_raw = stones_raw[i, 0], stones_raw[i, 1]
            if is_dead_or_unthrown(x_raw, y_raw):
                continue
            patch = plt.Circle(
                (x_raw, y_raw),
                STONE_R_RAW,
                facecolor=stone_colors[i],
                edgecolor="black",
                zorder=5,
            )
            ax.add_patch(patch)

        value_pred = state["value_pred"]
        value_true = state["value_true"]
        dataset_idx = state["dataset_idx"]

        # Top label (outside sheet)
        fig.text(
            0.5,
            0.96,
            f"Example {idx + 1}/{len(states)} (idx {dataset_idx})  "
            f"Pred: {value_pred:.2f}, True: {value_true:.2f}",
            ha="center",
            va="top",
            fontsize=11,
        )

        # Bottom label (outside sheet)
        fig.text(
            0.5,
            0.04,
            format_bottom_counts(stones_raw, num_stones),
            ha="center",
            va="bottom",
            fontsize=10,
        )

        out_path = os.path.join(out_dir, f"state_{idx + 1:04d}.png")
        # No bbox_inches="tight" so sheet stays centered with its margins
        fig.savefig(out_path, dpi=150)
        plt.close(fig)

    print(f"Saved {len(states)} PNG files to: {out_dir}")


# ---------------- Main entry point ---------------- #

def main():
    parser = argparse.ArgumentParser(
        description="Visualize value model predictions on validation dataset states as MP4/PNGs."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="value_model.pt",
        help="Path to trained value model checkpoint (.pt)",
    )
    parser.add_argument(
        "--stones_csv",
        type=str,
        default="/mnt/data/curling2/testBrax/brax/2026/Stones.csv",
        help="Path to Stones.csv for the validation set",
    )
    parser.add_argument(
        "--ends_csv",
        type=str,
        default="/mnt/data/curling2/testBrax/brax/2026/Ends.csv",
        help="Path to Ends.csv for the validation set",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="value_viz.mp4",
        help="Output MP4 path",
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=16,
        help="Number of validation examples to visualize",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=2,
        help="Frames per second for the MP4",
    )
    parser.add_argument(
        "--no_cuda",
        action="store_true",
        help="Force CPU even if CUDA is available",
    )

    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    )

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    model, num_stones, input_dim, cond_dim, num_tasks = build_model_from_ckpt(
        args.checkpoint, device
    )

    print(
        f"Loaded model from {args.checkpoint} "
        f"(num_stones={num_stones}, input_dim={input_dim}, cond_dim={cond_dim}, "
        f"num_tasks={num_tasks})"
    )

    # Build validation dataset (config should match how you trained/validated)
    val_dataset = ValueDataset(
        stones_csv_path=args.stones_csv,
        ends_csv_path=args.ends_csv,
        normalize=True,
        max_ends=8,
        min_shots_per_end=1,
        augment_positions=False,  # typically False for validation
    )

    print(f"Validation dataset size: {len(val_dataset)} examples")

    # Sanity check: model and dataset dimensions match
    assert input_dim == val_dataset.input_dim, \
        f"Input dim mismatch: model {input_dim} vs dataset {val_dataset.input_dim}"
    assert cond_dim == val_dataset.cond_dim, \
        f"Cond dim mismatch: model {cond_dim} vs dataset {val_dataset.cond_dim}"

    # Sample states from validation dataset and run the model
    states = create_states_from_dataset(
        model=model,
        device=device,
        dataset=val_dataset,
        num_examples=args.num_examples,
    )

    # Save PNG frames
    png_dir = "value_frames"
    save_states_as_png(states, num_stones, png_dir)

    # Save MP4 (or GIF fallback)
    visualize_states(
        states=states,
        num_stones=num_stones,
        out_path=args.out,
        fps=args.fps,
    )


if __name__ == "__main__":
    main()
