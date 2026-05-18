"""
Graph Neural Network architectures for curling value prediction.

Two main architectures:
1. EGNN (Equivariant Graph Neural Network) - geometric message passing
2. Graph Transformer - attention-based message passing

Both operate on a variable-size graph where:
- Each LIVE stone is a node (skip dead/unthrown at (0,0) or (1,1))
- Key landmark nodes: button center, release positions
- Fully connected edges with rich features

All models follow interface: forward(x, c) -> (B, 1)
where x is (B, 24) normalized stone positions, c is (B, 3) conditions.

Implements message passing from scratch (no torch_geometric dependency).
"""

import math
import os
import torch
from torch import nn
import torch.nn.functional as F

NUM_STONES = 12

# Curling geometry in normalized coords
BUTTON_X = 750.0 / 4095.0
BUTTON_Y = 800.0 / 4095.0
HOUSE_RADIUS = 600.0 / 4095.0

# Landmark nodes: button center + 3 release positions across the sheet width
# Release y is near the far hog line (~0.71 in normalized)
RELEASE_Y = 2900.0 / 4095.0
LANDMARKS = torch.tensor([
    [BUTTON_X, BUTTON_Y],                    # button center
    [350.0 / 4095.0, RELEASE_Y],             # release left
    [750.0 / 4095.0, RELEASE_Y],             # release center
    [1150.0 / 4095.0, RELEASE_Y],            # release right
], dtype=torch.float32)
DEFAULT_N_LANDMARKS = LANDMARKS.shape[0]

# Node feature dimensions
# stone node: (x, y, team_indicator, is_live=1, is_landmark=0, optional_extra_scalar) = 6
# landmark node: (x, y, team_indicator=0, is_live=0, is_landmark=1, optional_extra_scalar=0) = 6
NODE_FEAT_DIM = 6
STONE_RADIUS_NORM = 0.012
INFLATED_STONE_RADIUS_NORM = 2.0 * STONE_RADIUS_NORM
ANGLE_EPS = 1e-4
CLEARANCE_CAP_NORM = math.sqrt(2.0)
ACTIVE_BUTTON_REGION_NUM_VIEWPOINTS = 8
CONTACT_GEOMETRY_SAMPLE_COUNT = 48
CONTACT_GEOMETRY_CURVE_POINTS = 25
CONTACT_GEOMETRY_CURL_MIN = -0.14
CONTACT_GEOMETRY_CURL_MAX = 0.14
CONTACT_GEOMETRY_OPEN_TRAVEL_EPS_NORM = HOUSE_RADIUS * (1.0e-3 / 1.829)
CONTACT_GEOMETRY_X_MIN_NORM = BUTTON_X + HOUSE_RADIUS * (-2.35 / 1.829)
CONTACT_GEOMETRY_X_MAX_NORM = BUTTON_X + HOUSE_RADIUS * (2.35 / 1.829)
CONTACT_GEOMETRY_Y_MIN_NORM = BUTTON_Y + HOUSE_RADIUS * (-2.45 / 1.829)
CONTACT_GEOMETRY_Y_MAX_NORM = BUTTON_Y + HOUSE_RADIUS * (6.8 / 1.829)
def _configured_edge_scalar_dim():
    mode = os.environ.get("GNN_EDGE_SCALAR_MODE", "thrower_masked_button_region_span").strip()
    if mode == "goal_contact_oldbest_plus_chain_2hop":
        return 13
    if mode in {
        "contact_geometry_release_plus_stonepairs_full21",
        "contact_geometry_release_plus_stonepairs_products13",
        "contact_geometry_release_plus_stonepairs_basic9",
        "contact_geometry_release_plus_stonepairs_rac5",
        "contact_geometry_release_plus_stonepairs_score6",
        "contact_geometry_release_plus_stonepairs_scoresemi11",
        "contact_geometry_release_binary_plus_stonepairs_full21",
        "contact_geometry_release_binary_plus_stonepairs_products13",
        "contact_geometry_release_binary_plus_stonepairs_basic9",
        "contact_geometry_release_binary_plus_stonepairs_rac5",
        "contact_geometry_release_binary_plus_stonepairs_scoresemi11",
        "contact_geometry_release_binary_plus_stonepairs_hop1_full21",
        "contact_geometry_release_binary_plus_stonepairs_hop1_products13",
        "contact_geometry_release_binary_plus_stonepairs_hop1_basic9",
        "contact_geometry_release_binary_plus_stonepairs_hop1_rac5",
    }:
        return 21
    if mode == "contact_geometry_release_binary_concat_stonepairs_hop1_products13_allgoals21":
        return 42
    if mode == "contact_geometry_release_binary_concat_stonepairs_hop1_products13_products9":
        return 30
    if mode == "contact_geometry_release_binary_concat_stonepairs_hop1_products13_onehop_no_center14":
        return 35
    if mode == "contact_geometry_release_binary_concat_stonepairs_hop1_products13_onehop_no_takeout_center10":
        return 31
    if mode in {
        "oldbest_plus_stonepairs_full21",
        "oldbest_plus_stonepairs_products13",
        "oldbest_plus_stonepairs_basic9",
        "oldbest_plus_stonepairs_rac5",
        "oldbest_plus_stonepairs_score6",
        "oldbest_plus_stonepairs_scoresemi11",
    }:
        return 26
    if mode == "contact_geometry_all_sources_plus_alignment_plus_clearance_allgoals":
        return 21
    if mode == "contact_geometry_release_binary_reach_products_plus_alignment_plus_clearance_allgoals":
        return 21
    if mode == "contact_geometry_release_products_plus_alignment_plus_clearance_allgoals":
        return 21
    if mode == "contact_geometry_release_products_plus_alignment_plus_onehop_allgoals":
        return 22
    if mode == "contact_geometry_release_binary_reach_products_plus_onehop_allgoals":
        return 18
    if mode == "contact_geometry_release_products_plus_onehop_allgoals":
        return 18
    if mode == "contact_geometry_release_binary_reach_products_plus_onehop_no_takeout_center":
        return 10
    if mode == "contact_geometry_release_products_plus_onehop_no_takeout_center":
        return 10
    if mode == "contact_geometry_release_binary_reach_products_plus_onehop_no_center":
        return 14
    if mode == "contact_geometry_release_products_plus_onehop_no_center":
        return 14
    if mode == "oldbest_plus_contact_geometry_release_products_plus_clearance_plus_onehop":
        return 27
    if mode == "contact_geometry_release_products_plus_clearance_plus_onehop":
        return 22
    if mode == "oldbest_plus_contact_geometry_release_products_plus_clearance":
        return 18
    if mode == "goal_contact_usefulness_2hop":
        return 4
    if mode == "contact_geometry_release_products_plus_clearance":
        return 13
    if mode == "contact_geometry_release_binary_reach_products_plus_clearance":
        return 13
    if mode == "contact_geometry_release_products":
        return 9
    if mode == "contact_geometry_release_binary_reach_products":
        return 9
    if mode in {"goal_contact_chain_1hop", "goal_contact_chain_1hop_topk"}:
        return 6
    if mode in {"goal_contact_chain_2hop", "goal_contact_chain_outside_2hop", "goal_contact_1hop_terms"}:
        return 8
    return 5


EDGE_SCALAR_DIM = _configured_edge_scalar_dim()
TAKEOUT_OFFSET_NORM = 4.0 * STONE_RADIUS_NORM

# Additional landmark sources behind the house, opposite the release side.
# Visibility from these points to a target stone is a proxy for whether that
# target can be moved out through the back/outside corridors after contact.
TAKEOUT_SOURCE_Y = BUTTON_Y - HOUSE_RADIUS - 4.0 * STONE_RADIUS_NORM
TAKEOUT_SOURCE_X_OFFSETS = torch.tensor([-600.0, -300.0, 0.0, 300.0, 600.0], dtype=torch.float32) / 4095.0
TAKEOUT_LANDMARKS = torch.stack(
    [
        torch.full_like(TAKEOUT_SOURCE_X_OFFSETS, BUTTON_X) + TAKEOUT_SOURCE_X_OFFSETS,
        torch.full_like(TAKEOUT_SOURCE_X_OFFSETS, TAKEOUT_SOURCE_Y),
    ],
    dim=-1,
)


def _resolve_edge_scalar_mode():
    mode = os.environ.get("GNN_EDGE_SCALAR_MODE", "thrower_masked_button_region_span").strip()
    allowed = {
        "thrower_masked_button_region_span",
        "button_visible_plus_thrower_masked_span",
        "button_visible_plus_release_reach_span",
        "button_visible_plus_release_reach_with_product",
        "button_visible_plus_source_reach_takeout_edges_with_product",
        "button_visible_plus_curl_arc_reach_with_outgoing",
        "button_visible_plus_curl_arc_reach_clean",
        "button_visible_plus_curl_arc_minkowski_straight_reach_clean",
        "button_visible_plus_contact_arc_full",
        "button_visible_plus_contact_arc_minimal",
        "button_visible_plus_contact_arc_score_product",
        "button_visible_plus_contact_arc_takeout_product",
        "button_visible_plus_contact_arc_old_shape",
        "goal_contact_chain_2hop",
        "goal_contact_chain_outside_2hop",
        "goal_contact_oldbest_plus_chain_2hop",
        "goal_contact_chain_1hop",
        "goal_contact_chain_1hop_topk",
        "goal_contact_1hop_terms",
        "goal_contact_usefulness_2hop",
        "contact_geometry_release_products",
        "contact_geometry_release_products_plus_clearance",
        "contact_geometry_release_plus_stonepairs_full21",
        "contact_geometry_release_plus_stonepairs_products13",
        "contact_geometry_release_plus_stonepairs_basic9",
        "contact_geometry_release_plus_stonepairs_rac5",
        "contact_geometry_release_plus_stonepairs_score6",
        "contact_geometry_release_plus_stonepairs_scoresemi11",
        "contact_geometry_release_binary_reach_products",
        "contact_geometry_release_binary_reach_products_plus_clearance",
        "contact_geometry_release_binary_reach_products_plus_onehop_allgoals",
        "contact_geometry_release_binary_reach_products_plus_onehop_no_takeout_center",
        "contact_geometry_release_binary_reach_products_plus_onehop_no_center",
        "contact_geometry_release_binary_plus_stonepairs_full21",
        "contact_geometry_release_binary_plus_stonepairs_products13",
        "contact_geometry_release_binary_plus_stonepairs_basic9",
        "contact_geometry_release_binary_plus_stonepairs_rac5",
        "contact_geometry_release_binary_plus_stonepairs_scoresemi11",
        "contact_geometry_release_binary_plus_stonepairs_hop1_full21",
        "contact_geometry_release_binary_plus_stonepairs_hop1_products13",
        "contact_geometry_release_binary_plus_stonepairs_hop1_basic9",
        "contact_geometry_release_binary_plus_stonepairs_hop1_rac5",
        "contact_geometry_release_binary_concat_stonepairs_hop1_products13_allgoals21",
        "contact_geometry_release_binary_concat_stonepairs_hop1_products13_products9",
        "contact_geometry_release_binary_concat_stonepairs_hop1_products13_onehop_no_center14",
        "contact_geometry_release_binary_concat_stonepairs_hop1_products13_onehop_no_takeout_center10",
        "oldbest_plus_stonepairs_full21",
        "oldbest_plus_stonepairs_products13",
        "oldbest_plus_stonepairs_basic9",
        "oldbest_plus_stonepairs_rac5",
        "oldbest_plus_stonepairs_score6",
        "oldbest_plus_stonepairs_scoresemi11",
        "contact_geometry_all_sources_plus_alignment_plus_clearance_allgoals",
        "contact_geometry_release_binary_reach_products_plus_alignment_plus_clearance_allgoals",
        "contact_geometry_release_products_plus_alignment_plus_clearance_allgoals",
        "contact_geometry_release_products_plus_clearance_plus_onehop",
        "contact_geometry_release_products_plus_alignment_plus_onehop_allgoals",
        "contact_geometry_release_products_plus_onehop_allgoals",
        "contact_geometry_release_products_plus_onehop_no_takeout_center",
        "contact_geometry_release_products_plus_onehop_no_center",
        "oldbest_plus_contact_geometry_release_products_plus_clearance",
        "oldbest_plus_contact_geometry_release_products_plus_clearance_plus_onehop",
        "button_visible_times_release_reach",
        "active_button_region_visible_plus_release_reach_with_product",
        "button_visible_release_reach_takeout_product",
        "button_visible_release_reach_takeout_only",
        "button_visible_release_reach_takeout_products_only",
        "minkowski_scoring_triplet",
        "minkowski_scoring_takeout_quintet",
        "thrower_masked_pairwise_span",
        "button_region_pairwise_span",
        "exact_line_clearance",
        "pairwise_unoccluded_span",
        "button_visible_span",
    }
    if mode not in allowed:
        raise ValueError(f"Unsupported GNN_EDGE_SCALAR_MODE={mode!r}")
    return mode


def _is_contact_arc_edge_mode(mode):
    return mode in {
        "button_visible_plus_contact_arc_full",
        "button_visible_plus_contact_arc_minimal",
        "button_visible_plus_contact_arc_score_product",
        "button_visible_plus_contact_arc_takeout_product",
        "button_visible_plus_contact_arc_old_shape",
        "goal_contact_chain_2hop",
        "goal_contact_chain_outside_2hop",
        "goal_contact_oldbest_plus_chain_2hop",
        "goal_contact_chain_1hop",
        "goal_contact_chain_1hop_topk",
        "goal_contact_1hop_terms",
        "goal_contact_usefulness_2hop",
    }


def _resolve_node_feature_mode():
    mode = os.environ.get("GNN_NODE_FEATURE_MODE", "none").strip()
    allowed = {
        "none",
        "button_visible_span",
        "release_reach_times_takeout",
        "beats_nearest_opponent_to_button",
    }
    if mode not in allowed:
        raise ValueError(f"Unsupported GNN_NODE_FEATURE_MODE={mode!r}")
    return mode


def _resolve_edge_prune_mode():
    mode = os.environ.get("GNN_EDGE_PRUNE_MODE", "none").strip()
    allowed = {
        "none",
        "stone_pair_zero_pairwise_span",
    }
    if mode not in allowed:
        raise ValueError(f"Unsupported GNN_EDGE_PRUNE_MODE={mode!r}")
    return mode


def _resolve_release_node_mode():
    mode = os.environ.get("GNN_RELEASE_NODE_MODE", "three").strip()
    allowed = {
        "three",
        "single",
        "three_plus_takeout_boundary",
    }
    if mode not in allowed:
        raise ValueError(f"Unsupported GNN_RELEASE_NODE_MODE={mode!r}")
    return mode


def _get_active_landmarks(device=None, dtype=None):
    landmarks = LANDMARKS
    if _resolve_release_node_mode() == "single":
        landmarks = landmarks[[0, 2]]
    elif _resolve_release_node_mode() == "three_plus_takeout_boundary":
        landmarks = torch.cat([LANDMARKS, TAKEOUT_LANDMARKS], dim=0)
    if device is not None or dtype is not None:
        landmarks = landmarks.to(
            device=device if device is not None else landmarks.device,
            dtype=dtype if dtype is not None else landmarks.dtype,
        )
    return landmarks


def _get_release_points(device=None, dtype=None):
    landmarks = _get_active_landmarks(device=device, dtype=dtype)
    if _resolve_release_node_mode() == "three_plus_takeout_boundary":
        return landmarks[1:4, :]
    return landmarks[1:, :]


def _source_landmark_masks(node_coords, node_feats, node_mask):
    """Return masks for release-source and outside/takeout-source landmark nodes."""
    is_landmark = (node_feats[:, :, 4] > 0.5) & node_mask
    release_y = node_coords.new_tensor(RELEASE_Y)
    button_y = node_coords.new_tensor(BUTTON_Y)
    release_sources = is_landmark & ((node_coords[:, :, 1] - release_y).abs() < 1e-6)
    takeout_sources = is_landmark & (node_coords[:, :, 1] < button_y)
    return release_sources, takeout_sources


def _compute_visible_angular_span(stone_xy, all_xy, stone_idx, button_xy):
    """
    Compute the angular span of the button that stone at stone_idx blocks
    from various directions. Simplified: compute the angle subtended by the
    stone as seen from the button, considering occlusion by other stones.

    For each stone, compute the angle it subtends at the button center,
    then subtract the angle blocked by closer stones along similar directions.

    Returns a scalar per stone: unoccluded angular span (0 to 2*pi).
    """
    # Vector from button to this stone
    # stone_xy: (2,), button_xy: (2,)
    # all_xy: (N, 2)
    delta = stone_xy - button_xy  # (2,)
    dist = torch.norm(delta) + 1e-8
    angle = torch.atan2(delta[0], delta[1])  # angle of this stone from button

    # Apparent angular radius of a stone at this distance
    # Curling stone radius ~145mm, sheet width ~4572mm. Normalized: 145/4572 * (1500/4095) ~ 0.012
    STONE_ANGULAR_RADIUS = STONE_RADIUS_NORM
    my_half_angle = STONE_ANGULAR_RADIUS / dist

    # Check other stones that are closer and in similar direction
    other_mask = torch.ones(all_xy.shape[0], dtype=torch.bool, device=all_xy.device)
    other_mask[stone_idx] = False
    others = all_xy[other_mask]  # (N-1, 2)

    if others.shape[0] == 0:
        return 2.0 * my_half_angle

    other_deltas = others - button_xy.unsqueeze(0)  # (N-1, 2)
    other_dists = torch.norm(other_deltas, dim=-1) + 1e-8  # (N-1,)
    other_angles = torch.atan2(other_deltas[:, 0], other_deltas[:, 1])  # (N-1,)

    # Only consider stones closer to button than this one
    closer = other_dists < dist
    if not closer.any():
        return 2.0 * my_half_angle

    # Angular difference
    angle_diff = torch.abs(other_angles[closer] - angle)
    angle_diff = torch.min(angle_diff, 2 * math.pi - angle_diff)

    # Half-angles of blocking stones
    other_half = STONE_ANGULAR_RADIUS / other_dists[closer]

    # Overlap: if angle_diff < my_half_angle + other_half, there is blocking
    overlap = torch.clamp(my_half_angle + other_half - angle_diff, min=0.0)
    total_blocked = overlap.sum()

    return torch.clamp(2.0 * my_half_angle - total_blocked, min=0.0)


def _wrap_angle(delta):
    """Wrap angular differences into [-pi, pi]."""
    return torch.atan2(torch.sin(delta), torch.cos(delta))


def _extract_shooter_team(c, node_coords):
    """Recover the shooting team's stone block (0 or 1) from cond_dim=3 input."""
    B = node_coords.shape[0]
    if c is None:
        return torch.zeros(B, device=node_coords.device, dtype=node_coords.dtype)
    return c[:, 2].to(device=node_coords.device, dtype=node_coords.dtype)


def _button_node_indices(node_coords, node_feats, node_mask):
    """Locate the button landmark node in each batch element."""
    button_xy = node_coords.new_tensor([BUTTON_X, BUTTON_Y]).view(1, 1, 2)
    is_button = (
        node_mask
        & (node_feats[:, :, 4] > 0.5)
        & ((node_coords - button_xy).abs().amax(dim=-1) < 1e-6)
    )
    idx = torch.argmax(is_button.float(), dim=1)
    has_button = is_button.any(dim=1)
    idx = torch.where(has_button, idx, torch.full_like(idx, -1))
    return idx, is_button


def _compute_button_region_radii(node_coords, node_feats, node_mask, shooter_team):
    """
    Radius of the button-region circle: the nearest opponent-stone center to the
    button, capped by the house radius. If no opponent stone is live, fall back
    to the house radius.
    """
    B = node_coords.shape[0]
    button_xy = node_coords.new_tensor([BUTTON_X, BUTTON_Y]).view(1, 1, 2)
    stone_coords = node_coords[:, :NUM_STONES, :]
    stone_live = (node_feats[:, :NUM_STONES, 3] > 0.5) & node_mask[:, :NUM_STONES]
    stone_team = node_feats[:, :NUM_STONES, 2]
    opponent_mask = stone_live & (torch.abs(stone_team - shooter_team.view(B, 1)) > 0.5)
    dist = torch.norm(stone_coords - button_xy, dim=-1)
    capped = torch.where(opponent_mask, dist, torch.full_like(dist, HOUSE_RADIUS))
    radius = capped.min(dim=1).values
    return radius.clamp(min=1e-6, max=HOUSE_RADIUS)


def _compute_beats_nearest_opponent_to_button(node_coords, node_feats, node_mask):
    """
    Per-stone binary feature: 1 if this stone is closer to the button than the
    nearest live opponent stone is to the button. Landmarks and dead stones get 0.
    """
    B, N, _ = node_coords.shape
    button_xy = node_coords.new_tensor([BUTTON_X, BUTTON_Y]).view(1, 1, 2)
    is_live = (node_feats[:, :, 3] > 0.5) & node_mask
    team = node_feats[:, :, 2]
    dist = torch.norm(node_coords - button_xy, dim=-1)
    inf = torch.full_like(dist, float("inf"))
    nearest_opp = torch.full_like(dist, float("inf"))
    for team_value in (0.0, 1.0):
        own = is_live & (torch.abs(team - team_value) < 0.5)
        opp = is_live & (torch.abs(team - (1.0 - team_value)) < 0.5)
        opp_min = torch.where(opp, dist, inf).min(dim=1).values.view(B, 1)
        nearest_opp = torch.where(own, opp_min.expand_as(nearest_opp), nearest_opp)
    beats = is_live & (dist < nearest_opp)
    return beats.to(dtype=node_coords.dtype).unsqueeze(-1)


def _compute_unoccluded_angular_spans(node_coords, node_feats, node_mask):
    """
    Compute exact unoccluded angular span for each live stone as seen from the
    button center.

    For each live stone we form its angular interval at the button, clip the
    intervals from all closer live stones to that target interval, and subtract
    the exact union length of those blocker intervals. Landmarks and dead stones
    receive span 0.

    Returns:
        spans: (B, N, 1)
    """
    B, N, _ = node_coords.shape
    device = node_coords.device
    dtype = node_coords.dtype

    button_xy = node_coords.new_tensor([BUTTON_X, BUTTON_Y])
    is_live = (node_feats[:, :, 3] > 0.5) & node_mask

    delta = node_coords - button_xy.view(1, 1, 2)
    dist = torch.norm(delta, dim=-1).clamp(min=STONE_RADIUS_NORM + 1e-6)  # (B, N)
    angles = torch.atan2(delta[:, :, 0], delta[:, :, 1])  # (B, N)
    half_angles = torch.asin((STONE_RADIUS_NORM / dist).clamp(max=1.0 - 1e-6))  # (B, N)

    spans = torch.zeros(B, N, device=device, dtype=dtype)
    neg_inf_col = torch.full((B, 1), float("-inf"), device=device, dtype=dtype)
    pos_inf = torch.full_like(angles, float("inf"))
    neg_inf = torch.full_like(angles, float("-inf"))

    for target_idx in range(N):
        target_live = is_live[:, target_idx]  # (B,)
        if not bool(target_live.any()):
            continue

        target_angle = angles[:, target_idx:target_idx + 1]  # (B, 1)
        target_half = half_angles[:, target_idx:target_idx + 1]  # (B, 1)

        rel = _wrap_angle(angles - target_angle)  # (B, N)
        low = -target_half.expand_as(rel)
        high = target_half.expand_as(rel)
        starts = torch.maximum(rel - half_angles, low)
        ends = torch.minimum(rel + half_angles, high)

        closer = dist < (dist[:, target_idx:target_idx + 1] - 1e-8)
        blockers = closer & is_live
        blockers[:, target_idx] = False
        valid = blockers & (ends > starts)

        masked_starts = torch.where(valid, starts, pos_inf)
        masked_ends = torch.where(valid, ends, neg_inf)
        order = torch.argsort(masked_starts, dim=1)
        sorted_starts = torch.gather(masked_starts, 1, order)
        sorted_ends = torch.gather(masked_ends, 1, order)

        prev_max_end = torch.cummax(sorted_ends, dim=1).values
        prev_max_end_exclusive = torch.cat([neg_inf_col, prev_max_end[:, :-1]], dim=1)
        newly_covered = (
            sorted_ends - torch.maximum(sorted_starts, prev_max_end_exclusive)
        ).clamp(min=0.0)
        covered = newly_covered.sum(dim=1)

        target_span = (2.0 * half_angles[:, target_idx] - covered).clamp(min=0.0)
        spans[:, target_idx] = torch.where(
            target_live,
            target_span,
            torch.zeros_like(target_span),
        )

    return spans.unsqueeze(-1)


def _compute_target_visible_spans_from_viewpoints(node_coords, node_feats, node_mask, viewpoints):
    """
    Compute exact unoccluded angular span of each live target stone from each
    supplied viewpoint, then return the per-target max over viewpoints.

    Args:
        viewpoints: (B, V, 2)

    Returns:
        spans: (B, N, 1)
    """
    B, N, _ = node_coords.shape
    _, V, _ = viewpoints.shape
    device = node_coords.device
    dtype = node_coords.dtype

    is_live = (node_feats[:, :, 3] > 0.5) & node_mask
    best = torch.zeros(B, N, device=device, dtype=dtype)

    neg_inf_col = torch.full((B, 1), float("-inf"), device=device, dtype=dtype)
    pos_inf = torch.full((B, N), float("inf"), device=device, dtype=dtype)
    neg_inf = torch.full((B, N), float("-inf"), device=device, dtype=dtype)

    for view_idx in range(V):
        viewpoint = viewpoints[:, view_idx:view_idx + 1, :]
        delta = node_coords - viewpoint
        dist = torch.norm(delta, dim=-1).clamp(min=STONE_RADIUS_NORM + 1e-6)
        angles = torch.atan2(delta[:, :, 0], delta[:, :, 1])
        half_angles = torch.asin((STONE_RADIUS_NORM / dist).clamp(max=1.0 - 1e-6))

        spans = torch.zeros(B, N, device=device, dtype=dtype)
        for target_idx in range(N):
            target_live = is_live[:, target_idx]
            if not bool(target_live.any()):
                continue

            target_angle = angles[:, target_idx:target_idx + 1]
            target_half = half_angles[:, target_idx:target_idx + 1]

            rel = _wrap_angle(angles - target_angle)
            low = -target_half.expand_as(rel)
            high = target_half.expand_as(rel)
            starts = torch.maximum(rel - half_angles, low)
            ends = torch.minimum(rel + half_angles, high)

            closer = dist < (dist[:, target_idx:target_idx + 1] - 1e-8)
            blockers = closer & is_live
            blockers[:, target_idx] = False
            valid = blockers & (ends > starts)

            masked_starts = torch.where(valid, starts, pos_inf)
            masked_ends = torch.where(valid, ends, neg_inf)
            order = torch.argsort(masked_starts, dim=1)
            sorted_starts = torch.gather(masked_starts, 1, order)
            sorted_ends = torch.gather(masked_ends, 1, order)

            prev_max_end = torch.cummax(sorted_ends, dim=1).values
            prev_max_end_exclusive = torch.cat([neg_inf_col, prev_max_end[:, :-1]], dim=1)
            newly_covered = (
                sorted_ends - torch.maximum(sorted_starts, prev_max_end_exclusive)
            ).clamp(min=0.0)
            covered = newly_covered.sum(dim=1)

            target_span = (2.0 * half_angles[:, target_idx] - covered).clamp(min=0.0)
            spans[:, target_idx] = torch.where(
                target_live,
                target_span,
                torch.zeros_like(target_span),
            )

        best = torch.maximum(best, spans)

    return best.unsqueeze(-1)


def _compute_target_region_visible_spans_from_viewpoints(
    node_coords,
    node_feats,
    node_mask,
    viewpoints,
    target_centers,
    target_radii,
    blocker_radius,
):
    """
    Generic exact visible angular span of per-target circular regions from a set
    of viewpoints, returning the max span over viewpoints for each target.
    """
    B, N, _ = node_coords.shape
    _, V, _ = viewpoints.shape
    device = node_coords.device
    dtype = node_coords.dtype

    is_live = (node_feats[:, :, 3] > 0.5) & node_mask
    best = torch.zeros(B, N, device=device, dtype=dtype)

    neg_inf_col = torch.full((B, 1), float("-inf"), device=device, dtype=dtype)
    pos_inf = torch.full((B, N), float("inf"), device=device, dtype=dtype)
    neg_inf = torch.full((B, N), float("-inf"), device=device, dtype=dtype)

    for view_idx in range(V):
        viewpoint = viewpoints[:, view_idx:view_idx + 1, :]

        target_delta = target_centers - viewpoint
        target_dist = torch.norm(target_delta, dim=-1).clamp(min=1e-6)
        target_angles = torch.atan2(target_delta[:, :, 0], target_delta[:, :, 1])
        target_half = torch.asin((target_radii / target_dist.clamp(min=target_radii + 1e-6)).clamp(max=1.0 - 1e-6))

        blocker_delta = node_coords - viewpoint
        blocker_dist = torch.norm(blocker_delta, dim=-1).clamp(min=blocker_radius + 1e-6)
        blocker_angles = torch.atan2(blocker_delta[:, :, 0], blocker_delta[:, :, 1])
        blocker_half = torch.asin((blocker_radius / blocker_dist).clamp(max=1.0 - 1e-6))

        spans = torch.zeros(B, N, device=device, dtype=dtype)
        for target_idx in range(N):
            target_live = is_live[:, target_idx]
            if not bool(target_live.any()):
                continue

            angle = target_angles[:, target_idx:target_idx + 1]
            half = target_half[:, target_idx:target_idx + 1]
            rel = _wrap_angle(blocker_angles - angle)
            low = -half.expand_as(rel)
            high = half.expand_as(rel)
            starts = torch.maximum(rel - blocker_half, low)
            ends = torch.minimum(rel + blocker_half, high)

            closer = blocker_dist < (target_dist[:, target_idx:target_idx + 1] - 1e-8)
            blockers = closer & is_live
            blockers[:, target_idx] = False
            valid = blockers & (ends > starts)

            masked_starts = torch.where(valid, starts, pos_inf)
            masked_ends = torch.where(valid, ends, neg_inf)
            order = torch.argsort(masked_starts, dim=1)
            sorted_starts = torch.gather(masked_starts, 1, order)
            sorted_ends = torch.gather(masked_ends, 1, order)

            prev_max_end = torch.cummax(sorted_ends, dim=1).values
            prev_max_end_exclusive = torch.cat([neg_inf_col, prev_max_end[:, :-1]], dim=1)
            newly_covered = (
                sorted_ends - torch.maximum(sorted_starts, prev_max_end_exclusive)
            ).clamp(min=0.0)
            covered = newly_covered.sum(dim=1)

            target_span = (2.0 * target_half[:, target_idx] - covered).clamp(min=0.0)
            spans[:, target_idx] = torch.where(target_live, target_span, torch.zeros_like(target_span))

        best = torch.maximum(best, spans)

    return best.unsqueeze(-1)


def _compute_release_reach_spans(node_coords, node_feats, node_mask):
    """
    Target-stone reachability proxy from the active release landmark set.

    For each live target stone, compute its exact visible angular span from
    each release position and keep the largest span as the target's release-
    reachability scalar.
    """
    release_points = _get_release_points(device=node_coords.device, dtype=node_coords.dtype)
    release_points = release_points.unsqueeze(0).expand(node_coords.shape[0], -1, -1)
    return _compute_target_visible_spans_from_viewpoints(
        node_coords, node_feats, node_mask, release_points
    )


def _compute_minkowski_release_reach_spans(node_coords, node_feats, node_mask):
    release_points = _get_release_points(device=node_coords.device, dtype=node_coords.dtype)
    release_points = release_points.unsqueeze(0).expand(node_coords.shape[0], -1, -1)
    target_centers = node_coords
    target_radii = torch.full(
        node_coords.shape[:2],
        INFLATED_STONE_RADIUS_NORM,
        device=node_coords.device,
        dtype=node_coords.dtype,
    )
    return _compute_target_region_visible_spans_from_viewpoints(
        node_coords, node_feats, node_mask, release_points, target_centers, target_radii, INFLATED_STONE_RADIUS_NORM
    )


def _compute_active_button_region_scorability(node_coords, node_feats, node_mask, shooter_team):
    """
    Target-stone scorability from the active button region rather than the
    button-center point.

    We sample fixed viewpoints on the adaptive button-region circle whose radius
    is the nearest opponent-stone distance to the button center, then take the
    per-target max exact visible span across those circle points.
    """
    B = node_coords.shape[0]
    button_xy = node_coords.new_tensor([BUTTON_X, BUTTON_Y]).view(1, 1, 2)
    radii = _compute_button_region_radii(node_coords, node_feats, node_mask, shooter_team)  # (B,)
    theta = torch.linspace(
        0.0,
        2.0 * math.pi,
        ACTIVE_BUTTON_REGION_NUM_VIEWPOINTS + 1,
        device=node_coords.device,
        dtype=node_coords.dtype,
    )[:-1]
    circle = torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1)  # (V, 2)
    viewpoints = button_xy + radii.view(B, 1, 1) * circle.view(1, ACTIVE_BUTTON_REGION_NUM_VIEWPOINTS, 2)
    return _compute_target_visible_spans_from_viewpoints(
        node_coords, node_feats, node_mask, viewpoints
    )


def _compute_source_button_region_spans_generic(node_coords, node_feats, node_mask, shooter_team, blocker_radius):
    B, N, _ = node_coords.shape
    device = node_coords.device
    dtype = node_coords.dtype

    button_xy = node_coords.new_tensor([BUTTON_X, BUTTON_Y])
    is_live = (node_feats[:, :, 3] > 0.5) & node_mask
    button_region_radius = _compute_button_region_radii(node_coords, node_feats, node_mask, shooter_team)

    spans = torch.zeros(B, N, device=device, dtype=dtype)
    neg_inf_col = torch.full((B, 1), float("-inf"), device=device, dtype=dtype)
    pos_inf = torch.full((B, N), float("inf"), device=device, dtype=dtype)
    neg_inf = torch.full((B, N), float("-inf"), device=device, dtype=dtype)

    for source_idx in range(N):
        source_valid = node_mask[:, source_idx]
        target_live = is_live[:, source_idx] & source_valid
        if not bool(target_live.any()):
            continue

        viewpoint = node_coords[:, source_idx:source_idx + 1, :]
        delta = node_coords - viewpoint
        dist = torch.norm(delta, dim=-1).clamp(min=blocker_radius + 1e-6)
        angles = torch.atan2(delta[:, :, 0], delta[:, :, 1])
        half_angles = torch.asin((blocker_radius / dist).clamp(max=1.0 - 1e-6))

        to_button = button_xy.view(1, 2) - node_coords[:, source_idx, :]
        button_dist = torch.norm(to_button, dim=-1).clamp(min=1e-6)
        button_angle = torch.atan2(to_button[:, 0], to_button[:, 1]).unsqueeze(1)
        target_half = torch.asin((button_region_radius / button_dist).clamp(max=1.0 - 1e-6)).unsqueeze(1)

        rel = _wrap_angle(angles - button_angle)
        low = -target_half.expand_as(rel)
        high = target_half.expand_as(rel)
        starts = torch.maximum(rel - half_angles, low)
        ends = torch.minimum(rel + half_angles, high)

        closer = dist < (button_dist.unsqueeze(1) - 1e-8)
        blockers = closer & is_live
        blockers[:, source_idx] = False
        valid = blockers & (ends > starts)

        masked_starts = torch.where(valid, starts, pos_inf)
        masked_ends = torch.where(valid, ends, neg_inf)
        order = torch.argsort(masked_starts, dim=1)
        sorted_starts = torch.gather(masked_starts, 1, order)
        sorted_ends = torch.gather(masked_ends, 1, order)

        prev_max_end = torch.cummax(sorted_ends, dim=1).values
        prev_max_end_exclusive = torch.cat([neg_inf_col, prev_max_end[:, :-1]], dim=1)
        newly_covered = (
            sorted_ends - torch.maximum(sorted_starts, prev_max_end_exclusive)
        ).clamp(min=0.0)
        covered = newly_covered.sum(dim=1)

        target_span = (2.0 * target_half.squeeze(1) - covered).clamp(min=0.0)
        spans[:, source_idx] = torch.where(target_live, target_span, torch.zeros_like(target_span))

    return spans.unsqueeze(-1)


def _compute_takeoutability_spans(node_coords, node_feats, node_mask):
    B, N, _ = node_coords.shape
    target_centers = node_coords.clone()
    target_centers[:, :, 0] = target_centers[:, :, 0] + TAKEOUT_OFFSET_NORM
    target_radii = torch.full(
        (B, N),
        STONE_RADIUS_NORM,
        device=node_coords.device,
        dtype=node_coords.dtype,
    )
    viewpoints = node_coords
    return _compute_target_region_visible_spans_from_viewpoints(
        node_coords, node_feats, node_mask, viewpoints, target_centers, target_radii, INFLATED_STONE_RADIUS_NORM
    )


def _compute_minkowski_takeoutability_spans(node_coords, node_feats, node_mask):
    B, N, _ = node_coords.shape
    target_centers = node_coords.clone()
    target_centers[:, :, 0] = target_centers[:, :, 0] + TAKEOUT_OFFSET_NORM
    target_radii = torch.full(
        (B, N),
        INFLATED_STONE_RADIUS_NORM,
        device=node_coords.device,
        dtype=node_coords.dtype,
    )
    viewpoints = node_coords
    return _compute_target_region_visible_spans_from_viewpoints(
        node_coords, node_feats, node_mask, viewpoints, target_centers, target_radii, INFLATED_STONE_RADIUS_NORM
    )


def _compute_minkowski_scorability_spans(node_coords, node_feats, node_mask, shooter_team):
    return _compute_source_button_region_spans_generic(
        node_coords, node_feats, node_mask, shooter_team, INFLATED_STONE_RADIUS_NORM
    )


def _compute_minkowski_scoring_reachability_spans(node_coords, node_feats, node_mask, shooter_team):
    B, N, _ = node_coords.shape
    device = node_coords.device
    dtype = node_coords.dtype

    is_live = (node_feats[:, :, 3] > 0.5) & node_mask
    release_points = _get_release_points(device=device, dtype=dtype)
    button_xy = node_coords.new_tensor([BUTTON_X, BUTTON_Y]).view(1, 2)
    button_region_radius = _compute_button_region_radii(node_coords, node_feats, node_mask, shooter_team)

    best = torch.zeros(B, N, device=device, dtype=dtype)
    inf = torch.full((B, N), float("inf"), device=device, dtype=dtype)
    neg_inf = torch.full((B, N), float("-inf"), device=device, dtype=dtype)
    neg_inf_col = torch.full((B, 1), float("-inf"), device=device, dtype=dtype)

    for view_idx in range(release_points.shape[0]):
        viewpoint = release_points[view_idx].view(1, 1, 2).expand(B, 1, 2)
        blocker_delta = node_coords - viewpoint
        blocker_dist = torch.norm(blocker_delta, dim=-1).clamp(min=INFLATED_STONE_RADIUS_NORM + 1e-6)
        blocker_angles = torch.atan2(blocker_delta[:, :, 0], blocker_delta[:, :, 1])
        blocker_half = torch.asin((INFLATED_STONE_RADIUS_NORM / blocker_dist).clamp(max=1.0 - 1e-6))

        target_delta = node_coords - viewpoint
        target_dist = torch.norm(target_delta, dim=-1).clamp(min=INFLATED_STONE_RADIUS_NORM + 1e-6)
        target_angles = torch.atan2(target_delta[:, :, 0], target_delta[:, :, 1])
        target_half = torch.asin((INFLATED_STONE_RADIUS_NORM / target_dist).clamp(max=1.0 - 1e-6))

        to_button = button_xy.unsqueeze(1) - node_coords
        score_angles = torch.atan2(to_button[:, :, 0], to_button[:, :, 1])
        score_dist = torch.norm(to_button, dim=-1).clamp(min=button_region_radius.view(B, 1) + 1e-6)
        score_half = torch.asin((button_region_radius.view(B, 1) / score_dist).clamp(max=1.0 - 1e-6))

        spans = torch.zeros(B, N, device=device, dtype=dtype)
        for target_idx in range(N):
            target_live = is_live[:, target_idx]
            if not bool(target_live.any()):
                continue

            angle = target_angles[:, target_idx:target_idx + 1]
            allowed_low = _wrap_angle(score_angles[:, target_idx:target_idx + 1] - angle) - score_half[:, target_idx:target_idx + 1]
            allowed_high = _wrap_angle(score_angles[:, target_idx:target_idx + 1] - angle) + score_half[:, target_idx:target_idx + 1]

            low = torch.maximum(-target_half[:, target_idx:target_idx + 1], allowed_low)
            high = torch.minimum(target_half[:, target_idx:target_idx + 1], allowed_high)

            rel = _wrap_angle(blocker_angles - angle)
            starts = torch.maximum(rel - blocker_half, low.expand_as(rel))
            ends = torch.minimum(rel + blocker_half, high.expand_as(rel))

            closer = blocker_dist < (target_dist[:, target_idx:target_idx + 1] - 1e-8)
            blockers = closer & is_live
            blockers[:, target_idx] = False
            valid = blockers & (ends > starts)

            masked_starts = torch.where(valid, starts, inf)
            masked_ends = torch.where(valid, ends, neg_inf)
            order = torch.argsort(masked_starts, dim=1)
            sorted_starts = torch.gather(masked_starts, 1, order)
            sorted_ends = torch.gather(masked_ends, 1, order)

            prev_max_end = torch.cummax(sorted_ends, dim=1).values
            prev_max_end_exclusive = torch.cat([neg_inf_col, prev_max_end[:, :-1]], dim=1)
            newly_covered = (
                sorted_ends - torch.maximum(sorted_starts, prev_max_end_exclusive)
            ).clamp(min=0.0)
            covered = newly_covered.sum(dim=1)

            allowed_width = (high.squeeze(1) - low.squeeze(1)).clamp(min=0.0)
            target_span = (allowed_width - covered).clamp(min=0.0)
            spans[:, target_idx] = torch.where(target_live, target_span, torch.zeros_like(target_span))

        best = torch.maximum(best, spans)

    return best.unsqueeze(-1)


def _compute_minkowski_takeout_reachability_spans(node_coords, node_feats, node_mask):
    B, N, _ = node_coords.shape
    device = node_coords.device
    dtype = node_coords.dtype

    is_live = (node_feats[:, :, 3] > 0.5) & node_mask
    release_points = _get_release_points(device=device, dtype=dtype)
    best = torch.zeros(B, N, device=device, dtype=dtype)
    inf = torch.full((B, N), float("inf"), device=device, dtype=dtype)
    neg_inf = torch.full((B, N), float("-inf"), device=device, dtype=dtype)
    neg_inf_col = torch.full((B, 1), float("-inf"), device=device, dtype=dtype)

    target_centers = node_coords.clone()
    target_centers[:, :, 0] = target_centers[:, :, 0] + TAKEOUT_OFFSET_NORM

    for view_idx in range(release_points.shape[0]):
        viewpoint = release_points[view_idx].view(1, 1, 2).expand(B, 1, 2)
        blocker_delta = node_coords - viewpoint
        blocker_dist = torch.norm(blocker_delta, dim=-1).clamp(min=INFLATED_STONE_RADIUS_NORM + 1e-6)
        blocker_angles = torch.atan2(blocker_delta[:, :, 0], blocker_delta[:, :, 1])
        blocker_half = torch.asin((INFLATED_STONE_RADIUS_NORM / blocker_dist).clamp(max=1.0 - 1e-6))

        contact_delta = node_coords - viewpoint
        contact_dist = torch.norm(contact_delta, dim=-1).clamp(min=INFLATED_STONE_RADIUS_NORM + 1e-6)
        contact_angles = torch.atan2(contact_delta[:, :, 0], contact_delta[:, :, 1])
        contact_half = torch.asin((INFLATED_STONE_RADIUS_NORM / contact_dist).clamp(max=1.0 - 1e-6))

        outward_delta = target_centers - node_coords
        outward_angles = torch.atan2(outward_delta[:, :, 0], outward_delta[:, :, 1])
        takeout_half = torch.full((B, N), math.pi / 4.0, device=device, dtype=dtype)

        spans = torch.zeros(B, N, device=device, dtype=dtype)
        for target_idx in range(N):
            target_live = is_live[:, target_idx]
            if not bool(target_live.any()):
                continue

            angle = contact_angles[:, target_idx:target_idx + 1]
            allowed_low = _wrap_angle(outward_angles[:, target_idx:target_idx + 1] - angle) - takeout_half[:, target_idx:target_idx + 1]
            allowed_high = _wrap_angle(outward_angles[:, target_idx:target_idx + 1] - angle) + takeout_half[:, target_idx:target_idx + 1]

            low = torch.maximum(-contact_half[:, target_idx:target_idx + 1], allowed_low)
            high = torch.minimum(contact_half[:, target_idx:target_idx + 1], allowed_high)

            rel = _wrap_angle(blocker_angles - angle)
            starts = torch.maximum(rel - blocker_half, low.expand_as(rel))
            ends = torch.minimum(rel + blocker_half, high.expand_as(rel))

            closer = blocker_dist < (contact_dist[:, target_idx:target_idx + 1] - 1e-8)
            blockers = closer & is_live
            blockers[:, target_idx] = False
            valid = blockers & (ends > starts)

            masked_starts = torch.where(valid, starts, inf)
            masked_ends = torch.where(valid, ends, neg_inf)
            order = torch.argsort(masked_starts, dim=1)
            sorted_starts = torch.gather(masked_starts, 1, order)
            sorted_ends = torch.gather(masked_ends, 1, order)

            prev_max_end = torch.cummax(sorted_ends, dim=1).values
            prev_max_end_exclusive = torch.cat([neg_inf_col, prev_max_end[:, :-1]], dim=1)
            newly_covered = (
                sorted_ends - torch.maximum(sorted_starts, prev_max_end_exclusive)
            ).clamp(min=0.0)
            covered = newly_covered.sum(dim=1)

            allowed_width = (high.squeeze(1) - low.squeeze(1)).clamp(min=0.0)
            target_span = (allowed_width - covered).clamp(min=0.0)
            spans[:, target_idx] = torch.where(target_live, target_span, torch.zeros_like(target_span))

        best = torch.maximum(best, spans)

    return best.unsqueeze(-1)


def _pad_edge_scalars(*tensors):
    edge_scalars = torch.cat(tensors, dim=-1)
    if edge_scalars.shape[-1] < EDGE_SCALAR_DIM:
        pad = torch.zeros(
            *edge_scalars.shape[:-1],
            EDGE_SCALAR_DIM - edge_scalars.shape[-1],
            device=edge_scalars.device,
            dtype=edge_scalars.dtype,
        )
        edge_scalars = torch.cat([edge_scalars, pad], dim=-1)
    return edge_scalars


def _compute_pairwise_unoccluded_angular_spans(node_coords, node_feats, node_mask):
    """
    Compute exact pairwise unoccluded angular spans.

    For each ordered node pair (source i, target j), compute the visible angular
    span of target live stone j as seen from source node i, subtracting overlap
    from any closer live-stone blockers along that line of sight.

    Targets that are not live stones receive span 0. Self-edges receive span 0.

    Returns:
        spans: (B, N, N, 1) where spans[:, i, j, 0] is the visible span of
               target j from source i.
    """
    B, N, _ = node_coords.shape
    device = node_coords.device
    dtype = node_coords.dtype

    is_live = (node_feats[:, :, 3] > 0.5) & node_mask
    spans = torch.zeros(B, N, N, device=device, dtype=dtype)

    neg_inf_col = torch.full((B, 1), float("-inf"), device=device, dtype=dtype)
    pos_inf = torch.full((B, N), float("inf"), device=device, dtype=dtype)
    neg_inf = torch.full((B, N), float("-inf"), device=device, dtype=dtype)

    for source_idx in range(N):
        source_valid = node_mask[:, source_idx]  # (B,)
        if not bool(source_valid.any()):
            continue

        viewpoint = node_coords[:, source_idx:source_idx + 1, :]  # (B, 1, 2)
        delta = node_coords - viewpoint  # (B, N, 2)
        dist = torch.norm(delta, dim=-1).clamp(min=STONE_RADIUS_NORM + 1e-6)  # (B, N)
        angles = torch.atan2(delta[:, :, 0], delta[:, :, 1])  # (B, N)
        half_angles = torch.asin((STONE_RADIUS_NORM / dist).clamp(max=1.0 - 1e-6))  # (B, N)

        for target_idx in range(N):
            target_live = is_live[:, target_idx] & source_valid  # (B,)
            if source_idx == target_idx or not bool(target_live.any()):
                continue

            target_angle = angles[:, target_idx:target_idx + 1]  # (B, 1)
            target_half = half_angles[:, target_idx:target_idx + 1]  # (B, 1)

            rel = _wrap_angle(angles - target_angle)  # (B, N)
            low = -target_half.expand_as(rel)
            high = target_half.expand_as(rel)
            starts = torch.maximum(rel - half_angles, low)
            ends = torch.minimum(rel + half_angles, high)

            closer = dist < (dist[:, target_idx:target_idx + 1] - 1e-8)
            blockers = closer & is_live
            blockers[:, target_idx] = False
            blockers[:, source_idx] = False
            valid = blockers & (ends > starts)

            masked_starts = torch.where(valid, starts, pos_inf)
            masked_ends = torch.where(valid, ends, neg_inf)
            order = torch.argsort(masked_starts, dim=1)
            sorted_starts = torch.gather(masked_starts, 1, order)
            sorted_ends = torch.gather(masked_ends, 1, order)

            prev_max_end = torch.cummax(sorted_ends, dim=1).values
            prev_max_end_exclusive = torch.cat([neg_inf_col, prev_max_end[:, :-1]], dim=1)
            newly_covered = (
                sorted_ends - torch.maximum(sorted_starts, prev_max_end_exclusive)
            ).clamp(min=0.0)
            covered = newly_covered.sum(dim=1)

            target_span = (2.0 * half_angles[:, target_idx] - covered).clamp(min=0.0)
            spans[:, source_idx, target_idx] = torch.where(
                target_live,
                target_span,
                torch.zeros_like(target_span),
            )

    return spans.unsqueeze(-1)


def _compute_pairwise_unoccluded_angular_spans_with_radius(
    node_coords,
    node_feats,
    node_mask,
    target_radius,
    blocker_radius,
):
    """
    Pairwise unoccluded angular span with configurable target/blocker radii.

    Using inflated radii gives a cheap Minkowski-style straight reachability
    feature: the moving stone is reduced to a point, target/blockers are
    expanded, and blocker angular intervals are subtracted from the target
    angular interval.
    """
    B, N, _ = node_coords.shape
    device = node_coords.device
    dtype = node_coords.dtype

    target_radius = node_coords.new_tensor(float(target_radius))
    blocker_radius = node_coords.new_tensor(float(blocker_radius))
    is_live = (node_feats[:, :, 3] > 0.5) & node_mask
    spans = torch.zeros(B, N, N, device=device, dtype=dtype)

    neg_inf_col = torch.full((B, 1), float("-inf"), device=device, dtype=dtype)
    pos_inf = torch.full((B, N), float("inf"), device=device, dtype=dtype)
    neg_inf = torch.full((B, N), float("-inf"), device=device, dtype=dtype)

    for source_idx in range(N):
        source_valid = node_mask[:, source_idx]
        if not bool(source_valid.any()):
            continue

        viewpoint = node_coords[:, source_idx:source_idx + 1, :]
        delta = node_coords - viewpoint
        dist = torch.norm(delta, dim=-1).clamp(min=blocker_radius + 1e-6)
        angles = torch.atan2(delta[:, :, 0], delta[:, :, 1])
        blocker_half = torch.asin((blocker_radius / dist).clamp(max=1.0 - 1e-6))

        for target_idx in range(N):
            target_live = is_live[:, target_idx] & source_valid
            if source_idx == target_idx or not bool(target_live.any()):
                continue

            target_dist = dist[:, target_idx:target_idx + 1].clamp(min=target_radius + 1e-6)
            target_angle = angles[:, target_idx:target_idx + 1]
            target_half = torch.asin((target_radius / target_dist).clamp(max=1.0 - 1e-6))

            rel = _wrap_angle(angles - target_angle)
            low = -target_half.expand_as(rel)
            high = target_half.expand_as(rel)
            starts = torch.maximum(rel - blocker_half, low)
            ends = torch.minimum(rel + blocker_half, high)

            closer = dist < (dist[:, target_idx:target_idx + 1] - 1e-8)
            blockers = closer & is_live
            blockers[:, target_idx] = False
            blockers[:, source_idx] = False
            valid = blockers & (ends > starts)

            masked_starts = torch.where(valid, starts, pos_inf)
            masked_ends = torch.where(valid, ends, neg_inf)
            order = torch.argsort(masked_starts, dim=1)
            sorted_starts = torch.gather(masked_starts, 1, order)
            sorted_ends = torch.gather(masked_ends, 1, order)

            prev_max_end = torch.cummax(sorted_ends, dim=1).values
            prev_max_end_exclusive = torch.cat([neg_inf_col, prev_max_end[:, :-1]], dim=1)
            newly_covered = (
                sorted_ends - torch.maximum(sorted_starts, prev_max_end_exclusive)
            ).clamp(min=0.0)
            covered = newly_covered.sum(dim=1)

            target_span = (2.0 * target_half.squeeze(1) - covered).clamp(min=0.0)
            spans[:, source_idx, target_idx] = torch.where(
                target_live,
                target_span,
                torch.zeros_like(target_span),
            )

    return spans.unsqueeze(-1)


def _arc_points_between(source, target, curvature, num_points=17):
    """
    Quadratic Bezier arc points between source and target.

    Curvature is expressed as a signed fraction of chord length applied in the
    perpendicular direction. This is a cheap curl proxy rather than a simulator.
    """
    t = torch.linspace(0.0, 1.0, num_points, device=source.device, dtype=source.dtype)
    vec = target - source
    chord = torch.norm(vec, dim=-1, keepdim=True).clamp(min=1e-6)
    perp = torch.stack([-vec[..., 1], vec[..., 0]], dim=-1) / chord
    control = 0.5 * (source + target) + curvature * chord * perp
    one_minus = 1.0 - t.view(-1, 1)
    tt = t.view(-1, 1)
    return one_minus.pow(2) * source + 2.0 * one_minus * tt * control + tt.pow(2) * target


def _curve_min_clearance_to_blockers(curve, blocker_xy, blocker_mask, source_idx=None, target_idx=None):
    """
    Minimum swept-disk clearance from a polyline curve to live blocker centers.

    Blockers are inflated by two stone radii, reducing moving-stone collision
    checking to a point curve against forbidden disks.
    """
    dtype = curve.dtype
    device = curve.device
    valid = blocker_mask.clone()
    if source_idx is not None:
        valid[source_idx] = False
    if target_idx is not None:
        valid[target_idx] = False
    if not bool(valid.any()):
        return torch.tensor(CLEARANCE_CAP_NORM, device=device, dtype=dtype)

    blockers = blocker_xy[valid]
    p0 = curve[:-1]
    p1 = curve[1:]
    seg = p1 - p0
    seg_len2 = (seg * seg).sum(dim=-1).clamp(min=1e-12)
    rel = blockers[:, None, :] - p0[None, :, :]
    u = ((rel * seg[None, :, :]).sum(dim=-1) / seg_len2[None, :]).clamp(0.0, 1.0)
    closest = p0[None, :, :] + u[..., None] * seg[None, :, :]
    dist = torch.norm(blockers[:, None, :] - closest, dim=-1)
    return (dist.min() - INFLATED_STONE_RADIUS_NORM).clamp(
        min=-CLEARANCE_CAP_NORM,
        max=CLEARANCE_CAP_NORM,
    )


def _target_outgoing_score_takeout_compat(source, target, shooter_team, target_team, button_xy):
    """
    Approximate whether an impact from source to target sends target toward
    useful scoring or takeout directions.

    The target travel direction after equal-mass contact is approximated by the
    line of centers at contact, here source->target. This is intentionally
    cheap and simulator-free.
    """
    direction = target - source
    direction = direction / torch.norm(direction).clamp(min=1e-6)

    to_button = button_xy - target
    to_button = to_button / torch.norm(to_button).clamp(min=1e-6)
    score = torch.clamp((direction * to_button).sum(), min=0.0)

    # Takeout/exiting is outward from the button, and is most useful for
    # opponent stones. Keep same-team takeout compatibility at zero.
    outward = target - button_xy
    outward = outward / torch.norm(outward).clamp(min=1e-6)
    opp = (torch.abs(target_team - shooter_team) > 0.5).to(dtype=target.dtype)
    takeout = torch.clamp((direction * outward).sum(), min=0.0) * opp
    return score, takeout


def _compute_curl_arc_reachability_and_outgoing(node_coords, node_feats, node_mask, c=None):
    """
    Simulator-free curl-aware source-to-target feature.

    Vectorized version. For release-source landmarks, sample a small bank of
    quadratic arc paths to each target stone and evaluate swept-disk clearance
    against inflated live blocker disks. The returned edge features are nonzero
    only for release-source -> live-stone target edges.
    """
    B, N, _ = node_coords.shape
    device = node_coords.device
    dtype = node_coords.dtype
    is_live = (node_feats[:, :, 3] > 0.5) & node_mask
    release_sources, _ = _source_landmark_masks(node_coords, node_feats, node_mask)
    shooter_team = _extract_shooter_team(c, node_coords)
    button_xy = node_coords.new_tensor([BUTTON_X, BUTTON_Y]).view(1, 1, 2)
    curvatures = node_coords.new_tensor([-0.18, -0.10, -0.04, 0.0, 0.04, 0.10, 0.18])
    t = torch.linspace(0.0, 1.0, 13, device=device, dtype=dtype)

    source = node_coords.unsqueeze(2)  # (B, Nsrc, 1, 2)
    target = node_coords.unsqueeze(1)  # (B, 1, Ntgt, 2)
    vec = target - source
    chord = torch.norm(vec, dim=-1, keepdim=True).clamp(min=1e-6)
    perp = torch.stack([-vec[..., 1], vec[..., 0]], dim=-1) / chord

    control = (
        0.5 * (source.unsqueeze(3) + target.unsqueeze(3))
        + curvatures.view(1, 1, 1, -1, 1) * chord.unsqueeze(3) * perp.unsqueeze(3)
    )  # (B, S, T, C, 2)
    one_minus = (1.0 - t).view(1, 1, 1, 1, -1, 1)
    tt = t.view(1, 1, 1, 1, -1, 1)
    curve = (
        one_minus.pow(2) * source.unsqueeze(3).unsqueeze(4)
        + 2.0 * one_minus * tt * control.unsqueeze(4)
        + tt.pow(2) * target.unsqueeze(3).unsqueeze(4)
    )  # (B, S, T, C, P, 2)

    p0 = curve[..., :-1, :]
    p1 = curve[..., 1:, :]
    seg = p1 - p0
    seg_len2 = (seg * seg).sum(dim=-1).clamp(min=1e-12)  # (B, S, T, C, P-1)

    blockers = node_coords.view(B, 1, 1, 1, 1, N, 2)
    p0e = p0.unsqueeze(5)
    sege = seg.unsqueeze(5)
    rel = blockers - p0e
    u = ((rel * sege).sum(dim=-1) / seg_len2.unsqueeze(5)).clamp(0.0, 1.0)
    closest = p0e + u.unsqueeze(-1) * sege
    dist = torch.norm(blockers - closest, dim=-1)  # (B, S, T, C, P-1, Nblock)

    src_idx = torch.arange(N, device=device).view(1, N, 1, 1, 1, 1)
    tgt_idx = torch.arange(N, device=device).view(1, 1, N, 1, 1, 1)
    block_idx = torch.arange(N, device=device).view(1, 1, 1, 1, 1, N)
    blocker_mask = (
        is_live.view(B, 1, 1, 1, 1, N)
        & (block_idx != src_idx)
        & (block_idx != tgt_idx)
    )
    dist = torch.where(blocker_mask, dist, torch.full_like(dist, float("inf")))
    min_dist = dist.amin(dim=(-1, -2))  # (B, S, T, C)
    no_blockers = ~blocker_mask.any(dim=-1).squeeze(-1).squeeze(-1)  # (B, S, T)
    min_dist = torch.where(no_blockers.unsqueeze(-1), torch.full_like(min_dist, CLEARANCE_CAP_NORM), min_dist)
    clearance = (min_dist - INFLATED_STONE_RADIUS_NORM).clamp(
        min=-CLEARANCE_CAP_NORM,
        max=CLEARANCE_CAP_NORM,
    )

    valid_edge = release_sources.unsqueeze(2) & is_live.unsqueeze(1)
    feasible = (clearance > 0.0) & valid_edge.unsqueeze(-1)
    best_clearance = torch.where(
        valid_edge,
        (clearance.max(dim=-1).values / CLEARANCE_CAP_NORM).clamp(-1.0, 1.0),
        torch.zeros(B, N, N, device=device, dtype=dtype),
    )
    feasible_frac = feasible.to(dtype).mean(dim=-1)

    curv = curvatures.view(1, 1, 1, -1)
    pos_inf = torch.full_like(curv.expand(B, N, N, -1), float("inf"))
    neg_inf = torch.full_like(curv.expand(B, N, N, -1), float("-inf"))
    feasible_curv = feasible
    min_curv = torch.where(feasible_curv, curv, pos_inf).amin(dim=-1)
    max_curv = torch.where(feasible_curv, curv, neg_inf).amax(dim=-1)
    has_feasible = feasible_curv.any(dim=-1)
    curvature_diversity = torch.where(
        has_feasible,
        (max_curv - min_curv).abs() / curvatures.abs().max().clamp(min=1e-6),
        torch.zeros(B, N, N, device=device, dtype=dtype),
    )

    direction = vec / chord
    to_button = button_xy - target
    to_button = to_button / torch.norm(to_button, dim=-1, keepdim=True).clamp(min=1e-6)
    score_base = torch.clamp((direction * to_button).sum(dim=-1), min=0.0)

    outward = target - button_xy
    outward = outward / torch.norm(outward, dim=-1, keepdim=True).clamp(min=1e-6)
    target_team = node_feats[:, :, 2].view(B, 1, N)
    opp = (torch.abs(target_team - shooter_team.view(B, 1, 1)) > 0.5).to(dtype=dtype)
    takeout_base = torch.clamp((direction * outward).sum(dim=-1), min=0.0) * opp

    score_compat = torch.where(has_feasible, score_base, torch.zeros_like(score_base))
    takeout_compat = torch.where(has_feasible, takeout_base, torch.zeros_like(takeout_base))
    zero = torch.zeros(B, N, N, device=device, dtype=dtype)
    best_clearance = torch.where(valid_edge, best_clearance, zero)
    feasible_frac = torch.where(valid_edge, feasible_frac, zero)
    curvature_diversity = torch.where(valid_edge, curvature_diversity, zero)
    score_compat = torch.where(valid_edge, score_compat, zero)
    takeout_compat = torch.where(valid_edge, takeout_compat, zero)

    return (
        best_clearance.unsqueeze(-1),
        feasible_frac.unsqueeze(-1),
        curvature_diversity.unsqueeze(-1),
        score_compat.unsqueeze(-1),
        takeout_compat.unsqueeze(-1),
    )


def _compute_pairwise_outgoing_compatibility(node_coords, node_feats, node_mask, c=None):
    """
    Line-of-centers outgoing compatibility for all source -> live target edges.

    This is used with straight-line reachability for non-throwing sources in
    the curl-aware mode, while release-source edges still use curl-arc
    feasibility to gate these same outgoing semantics.
    """
    B, N, _ = node_coords.shape
    dtype = node_coords.dtype
    button_xy = node_coords.new_tensor([BUTTON_X, BUTTON_Y]).view(1, 1, 2)
    shooter_team = _extract_shooter_team(c, node_coords)

    source = node_coords.unsqueeze(2)
    target = node_coords.unsqueeze(1)
    direction = target - source
    direction = direction / torch.norm(direction, dim=-1, keepdim=True).clamp(min=1e-6)

    to_button = button_xy - target
    to_button = to_button / torch.norm(to_button, dim=-1, keepdim=True).clamp(min=1e-6)
    score = torch.clamp((direction * to_button).sum(dim=-1), min=0.0)

    outward = target - button_xy
    outward = outward / torch.norm(outward, dim=-1, keepdim=True).clamp(min=1e-6)
    target_team = node_feats[:, :, 2].view(B, 1, N)
    opp = (torch.abs(target_team - shooter_team.view(B, 1, 1)) > 0.5).to(dtype=dtype)
    takeout = torch.clamp((direction * outward).sum(dim=-1), min=0.0) * opp

    valid = (node_mask.unsqueeze(2) & ((node_feats[:, :, 3] > 0.5) & node_mask).unsqueeze(1))
    score = torch.where(valid, score, torch.zeros_like(score))
    takeout = torch.where(valid, takeout, torch.zeros_like(takeout))
    return score.unsqueeze(-1), takeout.unsqueeze(-1)


def _compute_clean_curl_reach_edge_scalars(node_coords, node_feats, node_mask, c=None, use_minkowski_straight=False):
    """
    Clean edge stack for curl-aware GraphTF variants:
      [target scorability, source->target reach, product]

    Release-source edges use curl-arc reachability. All other source nodes use
    straight pairwise visibility; the Minkowski option inflates target/blockers
    for the straight visibility calculation.
    """
    B, N, _ = node_coords.shape
    scorability = _compute_unoccluded_angular_spans(node_coords, node_feats, node_mask)
    scorability = scorability.unsqueeze(1).expand(-1, N, -1, -1)
    _, feasible, diversity, _, _ = _compute_curl_arc_reachability_and_outgoing(
        node_coords, node_feats, node_mask, c=c
    )
    if use_minkowski_straight:
        pairwise_reach = _compute_pairwise_unoccluded_angular_spans_with_radius(
            node_coords,
            node_feats,
            node_mask,
            target_radius=INFLATED_STONE_RADIUS_NORM,
            blocker_radius=INFLATED_STONE_RADIUS_NORM,
        )
    else:
        pairwise_reach = _compute_pairwise_unoccluded_angular_spans(node_coords, node_feats, node_mask)
    release_sources, _ = _source_landmark_masks(node_coords, node_feats, node_mask)
    release_edge = release_sources.view(B, N, 1, 1)
    curl_reach = feasible * diversity
    reach = torch.where(release_edge, curl_reach, pairwise_reach)
    return _pad_edge_scalars(scorability, reach, scorability * reach)


def _contact_outgoing_directions(device, dtype):
    angles = torch.linspace(0.0, 2.0 * math.pi, 9, device=device, dtype=dtype)[:-1]
    return torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)


def _straight_contact_reach_strength(node_coords, node_feats, node_mask, contact_centers):
    """
    Straight swept-disk reachability from every source node to each sampled
    target contact-center position.

    Args:
        contact_centers: (B, Ntarget, K, 2)

    Returns:
        strength: (B, Nsource, Ntarget, K) binary 0/1 reachability.
    """
    B, N, _ = node_coords.shape
    K = contact_centers.shape[2]
    is_live = (node_feats[:, :, 3] > 0.5) & node_mask

    source = node_coords.unsqueeze(2).unsqueeze(3)  # (B, S, 1, 1, 2)
    target_contact = contact_centers.unsqueeze(1)  # (B, 1, T, K, 2)
    seg = target_contact - source
    seg_len2 = (seg * seg).sum(dim=-1).clamp(min=1e-12)  # (B, S, T, K)

    blockers = node_coords.view(B, 1, 1, 1, N, 2)
    rel = blockers - source.unsqueeze(4)
    u = ((rel * seg.unsqueeze(4)).sum(dim=-1) / seg_len2.unsqueeze(4)).clamp(0.0, 1.0)
    closest = source.unsqueeze(4) + u.unsqueeze(-1) * seg.unsqueeze(4)
    dist = torch.norm(blockers - closest, dim=-1)  # (B, S, T, K, Nblock)

    src_idx = torch.arange(N, device=node_coords.device).view(1, N, 1, 1, 1)
    tgt_idx = torch.arange(N, device=node_coords.device).view(1, 1, N, 1, 1)
    block_idx = torch.arange(N, device=node_coords.device).view(1, 1, 1, 1, N)
    blocker_mask = (
        is_live.view(B, 1, 1, 1, N)
        & (block_idx != src_idx)
        & (block_idx != tgt_idx)
    )
    dist = torch.where(blocker_mask, dist, torch.full_like(dist, float("inf")))
    min_dist = dist.amin(dim=-1)
    no_blockers = ~blocker_mask.any(dim=-1)
    min_dist = torch.where(no_blockers, torch.full_like(min_dist, CLEARANCE_CAP_NORM), min_dist)
    clear = min_dist - INFLATED_STONE_RADIUS_NORM

    valid = node_mask.unsqueeze(2).unsqueeze(3) & is_live.unsqueeze(1).unsqueeze(3)
    not_self = src_idx.squeeze(-1) != tgt_idx.squeeze(-1)
    valid = valid & not_self
    return ((clear > 0.0) & valid).to(dtype=node_coords.dtype)


def _curl_contact_reach_strength(node_coords, node_feats, node_mask, contact_centers):
    """
    Curl-arc swept-disk reachability from release-source landmarks to sampled
    contact-center positions. Non-release source rows are zero.
    """
    B, N, _ = node_coords.shape
    K = contact_centers.shape[2]
    device = node_coords.device
    dtype = node_coords.dtype
    is_live = (node_feats[:, :, 3] > 0.5) & node_mask
    release_sources, _ = _source_landmark_masks(node_coords, node_feats, node_mask)
    curvatures = node_coords.new_tensor([-0.14, 0.0, 0.14])
    t = torch.linspace(0.0, 1.0, 7, device=device, dtype=dtype)

    source = node_coords.unsqueeze(2).unsqueeze(3)  # (B, S, 1, 1, 2)
    target_contact = contact_centers.unsqueeze(1)  # (B, 1, T, K, 2)
    vec = target_contact - source
    chord = torch.norm(vec, dim=-1, keepdim=True).clamp(min=1e-6)
    perp = torch.stack([-vec[..., 1], vec[..., 0]], dim=-1) / chord

    control = (
        0.5 * (source.unsqueeze(4) + target_contact.unsqueeze(4))
        + curvatures.view(1, 1, 1, 1, -1, 1) * chord.unsqueeze(4) * perp.unsqueeze(4)
    )
    one_minus = (1.0 - t).view(1, 1, 1, 1, 1, -1, 1)
    tt = t.view(1, 1, 1, 1, 1, -1, 1)
    curve = (
        one_minus.pow(2) * source.unsqueeze(4).unsqueeze(5)
        + 2.0 * one_minus * tt * control.unsqueeze(5)
        + tt.pow(2) * target_contact.unsqueeze(4).unsqueeze(5)
    )  # (B, S, T, K, C, P, 2)

    p0 = curve[..., :-1, :]
    p1 = curve[..., 1:, :]
    seg = p1 - p0
    seg_len2 = (seg * seg).sum(dim=-1).clamp(min=1e-12)

    blockers = node_coords.view(B, 1, 1, 1, 1, 1, N, 2)
    p0e = p0.unsqueeze(6)
    sege = seg.unsqueeze(6)
    rel = blockers - p0e
    u = ((rel * sege).sum(dim=-1) / seg_len2.unsqueeze(6)).clamp(0.0, 1.0)
    closest = p0e + u.unsqueeze(-1) * sege
    dist = torch.norm(blockers - closest, dim=-1)

    src_idx = torch.arange(N, device=device).view(1, N, 1, 1, 1, 1, 1)
    tgt_idx = torch.arange(N, device=device).view(1, 1, N, 1, 1, 1, 1)
    block_idx = torch.arange(N, device=device).view(1, 1, 1, 1, 1, 1, N)
    blocker_mask = (
        is_live.view(B, 1, 1, 1, 1, 1, N)
        & (block_idx != src_idx)
        & (block_idx != tgt_idx)
    )
    dist = torch.where(blocker_mask, dist, torch.full_like(dist, float("inf")))
    min_dist = dist.amin(dim=(-1, -2))
    no_blockers = ~blocker_mask.any(dim=-1).squeeze(-1).squeeze(-1)
    min_dist = torch.where(no_blockers.unsqueeze(-1).expand_as(min_dist), torch.full_like(min_dist, CLEARANCE_CAP_NORM), min_dist)
    feasible = min_dist > INFLATED_STONE_RADIUS_NORM

    valid_edge = release_sources.unsqueeze(2).unsqueeze(3) & is_live.unsqueeze(1).unsqueeze(3)
    feasible = feasible & valid_edge.unsqueeze(-1)
    feasible_frac = feasible.to(dtype).mean(dim=-1)
    return feasible_frac


def _compute_contact_arc_components(node_coords, node_feats, node_mask, c=None):
    """
    Contact-aware source->target components:
      contact_reach_frac, max reachable score-out, max reachable takeout-out,
      and center/arc reach used by the old-shape contact-aware stack.
    """
    B, N, _ = node_coords.shape
    device = node_coords.device
    dtype = node_coords.dtype
    is_live = (node_feats[:, :, 3] > 0.5) & node_mask
    shooter_team = _extract_shooter_team(c, node_coords)
    button_xy = node_coords.new_tensor([BUTTON_X, BUTTON_Y]).view(1, 1, 2)
    dirs = _contact_outgoing_directions(device, dtype)  # (K, 2)
    K = dirs.shape[0]

    target = node_coords.unsqueeze(2)  # (B, T, 1, 2)
    contact_centers = target - INFLATED_STONE_RADIUS_NORM * dirs.view(1, 1, K, 2)

    straight_strength = _straight_contact_reach_strength(node_coords, node_feats, node_mask, contact_centers)
    curl_strength = _curl_contact_reach_strength(node_coords, node_feats, node_mask, contact_centers)
    release_sources, _ = _source_landmark_masks(node_coords, node_feats, node_mask)
    release_edge = release_sources.view(B, N, 1, 1)
    contact_strength = torch.where(release_edge, curl_strength, straight_strength)

    valid_target = is_live.view(B, 1, N, 1)
    valid_source = node_mask.view(B, N, 1, 1)
    src_idx = torch.arange(N, device=device).view(1, N, 1, 1)
    tgt_idx = torch.arange(N, device=device).view(1, 1, N, 1)
    contact_strength = contact_strength * (valid_source & valid_target & (src_idx != tgt_idx)).to(dtype)

    target_for_dirs = node_coords.unsqueeze(1).unsqueeze(3)  # (B, 1, T, 1, 2)
    out_dirs = dirs.view(1, 1, 1, K, 2)
    to_button = button_xy.unsqueeze(2) - target_for_dirs
    to_button = to_button / torch.norm(to_button, dim=-1, keepdim=True).clamp(min=1e-6)
    score_by_contact = torch.clamp((out_dirs * to_button).sum(dim=-1), min=0.0)

    outward = target_for_dirs - button_xy.unsqueeze(2)
    outward = outward / torch.norm(outward, dim=-1, keepdim=True).clamp(min=1e-6)
    target_team = node_feats[:, :, 2].view(B, 1, N, 1)
    opp = (torch.abs(target_team - shooter_team.view(B, 1, 1, 1)) > 0.5).to(dtype=dtype)
    takeout_by_contact = torch.clamp((out_dirs * outward).sum(dim=-1), min=0.0) * opp

    reachable = contact_strength > 0.0
    contact_reach = contact_strength.mean(dim=-1, keepdim=True)
    max_score = torch.where(reachable, score_by_contact, torch.zeros_like(score_by_contact)).amax(dim=-1, keepdim=True)
    max_takeout = torch.where(reachable, takeout_by_contact, torch.zeros_like(takeout_by_contact)).amax(dim=-1, keepdim=True)

    _, feasible, diversity, _, _ = _compute_curl_arc_reachability_and_outgoing(
        node_coords, node_feats, node_mask, c=c
    )
    pairwise_reach = _compute_pairwise_unoccluded_angular_spans(node_coords, node_feats, node_mask)
    center_reach = torch.where(release_sources.view(B, N, 1, 1), feasible * diversity, pairwise_reach)
    return contact_reach, max_score, max_takeout, center_reach


def _compute_contact_arc_edge_scalars(node_coords, node_feats, node_mask, c=None, stack="full"):
    B, N, _ = node_coords.shape
    scorability = _compute_unoccluded_angular_spans(node_coords, node_feats, node_mask)
    scorability = scorability.unsqueeze(1).expand(-1, N, -1, -1)
    contact_reach, max_score, max_takeout, center_reach = _compute_contact_arc_components(
        node_coords, node_feats, node_mask, c=c
    )
    useful = torch.maximum(max_score, max_takeout)
    if stack == "full":
        return _pad_edge_scalars(scorability, contact_reach, max_score, max_takeout, contact_reach * useful)
    if stack == "minimal":
        return _pad_edge_scalars(scorability, contact_reach, contact_reach * useful)
    if stack == "score_product":
        return _pad_edge_scalars(scorability, contact_reach, max_score, max_takeout, contact_reach * max_score)
    if stack == "takeout_product":
        return _pad_edge_scalars(scorability, contact_reach, max_score, max_takeout, contact_reach * max_takeout)
    if stack == "old_shape":
        return _pad_edge_scalars(scorability, center_reach, max_score, max_takeout, center_reach * useful)
    raise ValueError(f"Unknown contact arc stack: {stack}")


def _norm_positive_clearance(clearance):
    cap = 3.0 * INFLATED_STONE_RADIUS_NORM
    return (clearance.clamp(min=0.0, max=cap) / cap).clamp(0.0, 1.0)


def _normalized_logsumexp(values, valid, dim=-1, beta=8.0):
    """
    Smooth soft-mass aggregation for values in [0, 1].

    This is normalized by the number of candidate slots, not by the number of
    valid positives, so it increases when either one path is strong or multiple
    candidate paths are useful.
    """
    neg = torch.full_like(values, -1.0e9)
    masked = torch.where(valid, values.clamp(0.0, 1.0) * beta, neg)
    count_slots = values.shape[dim]
    out = (torch.logsumexp(masked, dim=dim, keepdim=True) - math.log(max(1, count_slots))) / beta
    has_any = valid.any(dim=dim, keepdim=True)
    return torch.where(has_any, out.clamp(min=0.0, max=1.0), torch.zeros_like(out))


def _segment_clearance_edges(
    node_coords,
    node_feats,
    node_mask,
    start,
    end,
    exclude_src=True,
    exclude_tgt=True,
    exclude_extra_idx=None,
):
    """
    Swept-disk Minkowski clearance for edge/contact segments.

    start/end: (B, S, T, K, 2)
    returns: (B, S, T, K)
    """
    B, N, _ = node_coords.shape
    device = node_coords.device
    dtype = node_coords.dtype
    is_live = (node_feats[:, :, 3] > 0.5) & node_mask

    seg = end - start
    seg_len2 = (seg * seg).sum(dim=-1).clamp(min=1e-12)
    blockers = node_coords.view(B, 1, 1, 1, N, 2)
    rel = blockers - start.unsqueeze(4)
    u = ((rel * seg.unsqueeze(4)).sum(dim=-1) / seg_len2.unsqueeze(4)).clamp(0.0, 1.0)
    closest = start.unsqueeze(4) + u.unsqueeze(-1) * seg.unsqueeze(4)
    dist = torch.norm(blockers - closest, dim=-1)

    src_idx = torch.arange(N, device=device).view(1, N, 1, 1, 1)
    tgt_idx = torch.arange(N, device=device).view(1, 1, N, 1, 1)
    block_idx = torch.arange(N, device=device).view(1, 1, 1, 1, N)
    blocker_mask = is_live.view(B, 1, 1, 1, N)
    if exclude_src:
        blocker_mask = blocker_mask & (block_idx != src_idx)
    if exclude_tgt:
        blocker_mask = blocker_mask & (block_idx != tgt_idx)
    if exclude_extra_idx is not None:
        blocker_mask = blocker_mask & (block_idx != exclude_extra_idx.unsqueeze(-1))
    dist = torch.where(blocker_mask, dist, torch.full_like(dist, float("inf")))
    min_dist = dist.amin(dim=-1)
    no_blockers = ~blocker_mask.any(dim=-1)
    min_dist = torch.where(no_blockers, torch.full_like(min_dist, CLEARANCE_CAP_NORM), min_dist)
    return min_dist - INFLATED_STONE_RADIUS_NORM


def _curl_clearance_edges(node_coords, node_feats, node_mask, start, end):
    """
    Best clearance across the sampled curl proxies for release-source edges.
    start/end: (B, S, T, K, 2)
    returns: (B, S, T, K)
    """
    B, N, _ = node_coords.shape
    device = node_coords.device
    dtype = node_coords.dtype
    is_live = (node_feats[:, :, 3] > 0.5) & node_mask
    curvatures = node_coords.new_tensor([-0.14, 0.0, 0.14])
    t = torch.linspace(0.0, 1.0, 7, device=device, dtype=dtype)

    vec = end - start
    chord = torch.norm(vec, dim=-1, keepdim=True).clamp(min=1e-6)
    perp = torch.stack([-vec[..., 1], vec[..., 0]], dim=-1) / chord
    control = (
        0.5 * (start.unsqueeze(4) + end.unsqueeze(4))
        + curvatures.view(1, 1, 1, 1, -1, 1) * chord.unsqueeze(4) * perp.unsqueeze(4)
    )
    one_minus = (1.0 - t).view(1, 1, 1, 1, 1, -1, 1)
    tt = t.view(1, 1, 1, 1, 1, -1, 1)
    curve = (
        one_minus.pow(2) * start.unsqueeze(4).unsqueeze(5)
        + 2.0 * one_minus * tt * control.unsqueeze(5)
        + tt.pow(2) * end.unsqueeze(4).unsqueeze(5)
    )

    p0 = curve[..., :-1, :]
    p1 = curve[..., 1:, :]
    seg = p1 - p0
    seg_len2 = (seg * seg).sum(dim=-1).clamp(min=1e-12)
    blockers = node_coords.view(B, 1, 1, 1, 1, 1, N, 2)
    rel = blockers - p0.unsqueeze(6)
    u = ((rel * seg.unsqueeze(6)).sum(dim=-1) / seg_len2.unsqueeze(6)).clamp(0.0, 1.0)
    closest = p0.unsqueeze(6) + u.unsqueeze(-1) * seg.unsqueeze(6)
    dist = torch.norm(blockers - closest, dim=-1)

    src_idx = torch.arange(N, device=device).view(1, N, 1, 1, 1, 1, 1)
    tgt_idx = torch.arange(N, device=device).view(1, 1, N, 1, 1, 1, 1)
    block_idx = torch.arange(N, device=device).view(1, 1, 1, 1, 1, 1, N)
    blocker_mask = (
        is_live.view(B, 1, 1, 1, 1, 1, N)
        & (block_idx != src_idx)
        & (block_idx != tgt_idx)
    )
    dist = torch.where(blocker_mask, dist, torch.full_like(dist, float("inf")))
    min_dist = dist.amin(dim=(-1, -2))
    no_blockers = ~blocker_mask.any(dim=-1).squeeze(-1).squeeze(-1)
    min_dist = torch.where(no_blockers.unsqueeze(-1).expand_as(min_dist), torch.full_like(min_dist, CLEARANCE_CAP_NORM), min_dist)
    return (min_dist - INFLATED_STONE_RADIUS_NORM).amax(dim=-1)


def _opponent_button_distances(node_coords, node_feats, node_mask):
    B, N, _ = node_coords.shape
    is_live = (node_feats[:, :, 3] > 0.5) & node_mask
    team = node_feats[:, :, 2]
    button = node_coords.new_tensor([BUTTON_X, BUTTON_Y]).view(1, 1, 2)
    dist = torch.norm(node_coords - button, dim=-1)
    out = torch.full((B, N), HOUSE_RADIUS, device=node_coords.device, dtype=node_coords.dtype)
    for team_id in (0.0, 1.0):
        this_team = (team == team_id) & is_live
        opp_team = (team != team_id) & is_live
        opp_dist = torch.where(opp_team, dist, torch.full_like(dist, float("inf"))).amin(dim=1)
        opp_dist = torch.where(torch.isfinite(opp_dist), opp_dist, torch.full_like(opp_dist, HOUSE_RADIUS))
        out = torch.where(this_team, opp_dist.unsqueeze(1).expand_as(out), out)
    return out


def _goal_points(target, out_dirs, goal_kind, opp_button_dist=None):
    button = target.new_tensor([BUTTON_X, BUTTON_Y]).view(1, 1, 1, 1, 2)
    rel = target - button
    if goal_kind == "score":
        lam = ((button - target) * out_dirs).sum(dim=-1, keepdim=True).clamp(min=0.0)
        return target + lam * out_dirs
    if goal_kind != "takeout":
        raise ValueError(f"Unknown goal kind: {goal_kind}")
    threshold = (opp_button_dist + STONE_RADIUS_NORM).clamp(min=STONE_RADIUS_NORM, max=HOUSE_RADIUS)
    r0 = torch.norm(rel, dim=-1)
    b = (rel * out_dirs).sum(dim=-1)
    cval = r0.pow(2) - threshold.pow(2)
    disc = (b.pow(2) - cval).clamp(min=0.0)
    lam = -b + torch.sqrt(disc)
    already = r0 >= threshold
    lam = torch.where(already, torch.zeros_like(lam), lam.clamp(min=0.0))
    return target + lam.unsqueeze(-1) * out_dirs


def _contact_geometry_sample_offsets(device, dtype):
    return torch.linspace(
        -0.5 * math.pi,
        0.5 * math.pi,
        CONTACT_GEOMETRY_SAMPLE_COUNT,
        device=device,
        dtype=dtype,
    )


def _contact_geometry_curvature_bank(device, dtype):
    gen = torch.Generator(device="cpu")
    gen.manual_seed(20260515)
    u = torch.rand(CONTACT_GEOMETRY_SAMPLE_COUNT, generator=gen, dtype=torch.float32)
    curv = CONTACT_GEOMETRY_CURL_MIN + (CONTACT_GEOMETRY_CURL_MAX - CONTACT_GEOMETRY_CURL_MIN) * u
    return curv.to(device=device, dtype=dtype)


def _contact_geometry_boundary_lam(start, ray_dir):
    inf = torch.full(start.shape[:-1], float("inf"), device=start.device, dtype=start.dtype)
    eps = start.new_tensor(CONTACT_GEOMETRY_OPEN_TRAVEL_EPS_NORM)

    def x_hit(x_const):
        denom = torch.where(ray_dir[..., 0].abs() > 1e-9, ray_dir[..., 0], torch.ones_like(ray_dir[..., 0]))
        lam = (start[..., 0].new_tensor(x_const) - start[..., 0]) / denom
        y = start[..., 1] + lam * ray_dir[..., 1]
        valid = (
            (ray_dir[..., 0].abs() > 1e-9)
            & (lam > eps)
            & (y >= CONTACT_GEOMETRY_Y_MIN_NORM - 1e-9)
            & (y <= CONTACT_GEOMETRY_Y_MAX_NORM + 1e-9)
        )
        return torch.where(valid, lam, inf)

    def y_hit(y_const):
        denom = torch.where(ray_dir[..., 1].abs() > 1e-9, ray_dir[..., 1], torch.ones_like(ray_dir[..., 1]))
        lam = (start[..., 1].new_tensor(y_const) - start[..., 1]) / denom
        x = start[..., 0] + lam * ray_dir[..., 0]
        valid = (
            (ray_dir[..., 1].abs() > 1e-9)
            & (lam > eps)
            & (x >= CONTACT_GEOMETRY_X_MIN_NORM - 1e-9)
            & (x <= CONTACT_GEOMETRY_X_MAX_NORM + 1e-9)
        )
        return torch.where(valid, lam, inf)

    return torch.minimum(
        torch.minimum(x_hit(CONTACT_GEOMETRY_X_MIN_NORM), x_hit(CONTACT_GEOMETRY_X_MAX_NORM)),
        torch.minimum(y_hit(CONTACT_GEOMETRY_Y_MIN_NORM), y_hit(CONTACT_GEOMETRY_Y_MAX_NORM)),
    )


def _contact_geometry_goal_points(target, out_dirs, goal_kind, opp_button_dist):
    button = target.new_tensor([BUTTON_X, BUTTON_Y]).view(*([1] * (target.dim() - 1)), 2)
    rel = target - button
    r0 = torch.norm(rel, dim=-1)
    boundary_lam = _contact_geometry_boundary_lam(target, out_dirs)
    eps = target.new_tensor(CONTACT_GEOMETRY_OPEN_TRAVEL_EPS_NORM)

    if goal_kind == "score":
        radius = opp_button_dist
        b = (rel * out_dirs).sum(dim=-1)
        cval = r0.pow(2) - radius.pow(2)
        disc = b.pow(2) - cval
        sqrt_disc = torch.sqrt(disc.clamp(min=0.0))
        lam1 = -b - sqrt_disc
        lam2 = -b + sqrt_disc
        inf = torch.full_like(lam1, float("inf"))
        lam = torch.where(
            lam1 > eps,
            lam1,
            torch.where(lam2 > eps, lam2, inf),
        )
        valid = (r0 > radius + 1e-9) & (disc >= 0.0) & torch.isfinite(lam) & (lam <= boundary_lam + 1e-9)
        goal = target + lam.unsqueeze(-1) * out_dirs
        return goal, valid

    if goal_kind == "takeout":
        radius = (opp_button_dist + STONE_RADIUS_NORM).clamp(min=STONE_RADIUS_NORM, max=HOUSE_RADIUS)
        b = (rel * out_dirs).sum(dim=-1)
        cval = r0.pow(2) - radius.pow(2)
        disc = b.pow(2) - cval
        lam = -b + torch.sqrt(disc.clamp(min=0.0))
        valid = (
            (r0 < radius - 1e-9)
            & (disc >= 0.0)
            & (lam > eps)
            & (lam <= boundary_lam + 1e-9)
        )
        goal = target + lam.unsqueeze(-1) * out_dirs
        return goal, valid

    if goal_kind == "center":
        x0 = target[..., 0] - BUTTON_X
        dx = out_dirs[..., 0]
        abs_dx = dx.abs()
        lam = torch.full_like(abs_dx, float("inf"))
        near_center = x0.abs() < 1e-9
        lam = torch.where(near_center, eps / abs_dx.clamp(min=1e-9), lam)
        away = (~near_center) & (dx * torch.sign(x0) > 0.0)
        lam = torch.where(away, target.new_full((), STONE_RADIUS_NORM) / abs_dx.clamp(min=1e-9), lam)
        valid = (abs_dx > 1e-9) & torch.isfinite(lam) & (lam > eps) & (lam <= boundary_lam + 1e-9)
        goal = target + lam.unsqueeze(-1) * out_dirs
        return goal, valid

    if goal_kind == "semi":
        lam_circle = -2.0 * (rel * out_dirs).sum(dim=-1)
        use_circle = (lam_circle > eps) & (lam_circle <= boundary_lam + 1e-9)
        lam = torch.where(use_circle, lam_circle, boundary_lam)
        goal = target + lam.unsqueeze(-1) * out_dirs
        valid = torch.isfinite(boundary_lam)
        return goal, valid

    raise ValueError(f"Unknown contact-geometry goal kind: {goal_kind}")


def _segment_blocked_binary_release_targets(
    node_coords,
    node_feats,
    node_mask,
    start,
    end,
):
    """
    Binary positive-travel hit test for target-motion rays.

    start/end: (B, R, N, K, 2), where the target axis is the full node axis N.
    """
    B, N, _ = node_coords.shape
    device = node_coords.device
    is_live = (node_feats[:, :, 3] > 0.5) & node_mask
    eps = start.new_tensor(CONTACT_GEOMETRY_OPEN_TRAVEL_EPS_NORM)

    seg = end - start
    seg_len = torch.norm(seg, dim=-1).clamp(min=1e-8)
    unit = seg / seg_len.unsqueeze(-1)
    blockers = node_coords.view(B, 1, 1, 1, N, 2)
    rel = blockers - start.unsqueeze(4)
    b = (rel * unit.unsqueeze(4)).sum(dim=-1)
    d2 = (rel * rel).sum(dim=-1) - b.pow(2)
    inside = (INFLATED_STONE_RADIUS_NORM ** 2 - d2).clamp(min=0.0)
    lam = b - torch.sqrt(inside)

    tgt_idx = torch.arange(N, device=device).view(1, 1, N, 1, 1)
    block_idx = torch.arange(N, device=device).view(1, 1, 1, 1, N)
    valid = (
        is_live.view(B, 1, 1, 1, N)
        & (block_idx != tgt_idx)
        & (b > eps)
        & (d2 <= INFLATED_STONE_RADIUS_NORM ** 2)
        & (lam > eps)
        & (lam <= seg_len.unsqueeze(-1))
    )
    return valid.any(dim=-1)


def _segment_blocked_binary_generic(
    node_coords,
    node_feats,
    node_mask,
    start,
    end,
    primary_exclude_idx,
    extra_exclude_idx=None,
):
    """
    Generic binary positive-travel hit test for segments with shape (B, A, N, K, 2).
    """
    B, N, _ = node_coords.shape
    device = node_coords.device
    is_live = (node_feats[:, :, 3] > 0.5) & node_mask
    eps = start.new_tensor(CONTACT_GEOMETRY_OPEN_TRAVEL_EPS_NORM)

    seg = end - start
    seg_len = torch.norm(seg, dim=-1).clamp(min=1e-8)
    unit = seg / seg_len.unsqueeze(-1)
    blockers = node_coords.view(B, 1, 1, 1, N, 2)
    rel = blockers - start.unsqueeze(4)
    b = (rel * unit.unsqueeze(4)).sum(dim=-1)
    d2 = (rel * rel).sum(dim=-1) - b.pow(2)
    inside = (INFLATED_STONE_RADIUS_NORM ** 2 - d2).clamp(min=0.0)
    lam = b - torch.sqrt(inside)

    block_idx = torch.arange(N, device=device).view(1, 1, 1, 1, N)
    blocker_mask = is_live.view(B, 1, 1, 1, N) & (block_idx != primary_exclude_idx.unsqueeze(-1))
    if extra_exclude_idx is not None:
        blocker_mask = blocker_mask & (block_idx != extra_exclude_idx.unsqueeze(-1))
    valid = (
        blocker_mask
        & (b > eps)
        & (d2 <= INFLATED_STONE_RADIUS_NORM ** 2)
        & (lam > eps)
        & (lam <= seg_len.unsqueeze(-1))
    )
    return valid.any(dim=-1)


def _segment_clearance_pairlist(
    node_coords,
    node_feats,
    node_mask,
    start,
    end,
    src_idx,
    tgt_idx,
    extra_exclude_idx=None,
):
    """
    Swept-disk Minkowski clearance for explicit sparse pair lists.

    start/end: (B, P, K, 2)
    src_idx/tgt_idx: (B, P)
    returns: (B, P, K)
    """
    B, N, _ = node_coords.shape
    device = node_coords.device
    is_live = (node_feats[:, :, 3] > 0.5) & node_mask

    seg = end - start
    seg_len2 = (seg * seg).sum(dim=-1).clamp(min=1e-12)
    blockers = node_coords.view(B, 1, 1, N, 2)
    rel = blockers - start.unsqueeze(3)
    u = ((rel * seg.unsqueeze(3)).sum(dim=-1) / seg_len2.unsqueeze(3)).clamp(0.0, 1.0)
    closest = start.unsqueeze(3) + u.unsqueeze(-1) * seg.unsqueeze(3)
    dist = torch.norm(blockers - closest, dim=-1)

    block_idx = torch.arange(N, device=device).view(1, 1, 1, N)
    blocker_mask = (
        is_live.view(B, 1, 1, N)
        & (block_idx != src_idx.unsqueeze(-1).unsqueeze(-1))
        & (block_idx != tgt_idx.unsqueeze(-1).unsqueeze(-1))
    )
    if extra_exclude_idx is not None:
        blocker_mask = blocker_mask & (block_idx != extra_exclude_idx.unsqueeze(-1).unsqueeze(-1))
    dist = torch.where(blocker_mask, dist, torch.full_like(dist, float("inf")))
    min_dist = dist.amin(dim=-1)
    no_blockers = ~blocker_mask.any(dim=-1)
    min_dist = torch.where(no_blockers, torch.full_like(min_dist, CLEARANCE_CAP_NORM), min_dist)
    return min_dist - INFLATED_STONE_RADIUS_NORM


def _segment_blocked_binary_pairlist(
    node_coords,
    node_feats,
    node_mask,
    start,
    end,
    primary_exclude_idx,
    extra_exclude_idx=None,
):
    """
    Binary positive-travel hit test for explicit sparse pair lists.

    start/end: (B, P, K, 2)
    primary_exclude_idx: (B, P)
    """
    B, N, _ = node_coords.shape
    device = node_coords.device
    is_live = (node_feats[:, :, 3] > 0.5) & node_mask
    eps = start.new_tensor(CONTACT_GEOMETRY_OPEN_TRAVEL_EPS_NORM)

    seg = end - start
    seg_len = torch.norm(seg, dim=-1).clamp(min=1e-8)
    unit = seg / seg_len.unsqueeze(-1)
    blockers = node_coords.view(B, 1, 1, N, 2)
    rel = blockers - start.unsqueeze(3)
    b = (rel * unit.unsqueeze(3)).sum(dim=-1)
    d2 = (rel * rel).sum(dim=-1) - b.pow(2)
    inside = (INFLATED_STONE_RADIUS_NORM ** 2 - d2).clamp(min=0.0)
    lam = b - torch.sqrt(inside)

    block_idx = torch.arange(N, device=device).view(1, 1, 1, N)
    valid = (
        is_live.view(B, 1, 1, N)
        & (block_idx != primary_exclude_idx.unsqueeze(-1).unsqueeze(-1))
        & (b > eps)
        & (d2 <= INFLATED_STONE_RADIUS_NORM ** 2)
        & (lam > eps)
        & (lam <= seg_len.unsqueeze(-1))
    )
    if extra_exclude_idx is not None:
        valid = valid & (block_idx != extra_exclude_idx.unsqueeze(-1).unsqueeze(-1))
    return valid.any(dim=-1)


def _first_intersection_from_target_rays(node_coords, node_feats, node_mask, start, end):
    """
    First inflated live-stone hit for segments with shape (B, R, N, K, 2), where
    the excluded primary index is the current target stone on the N axis.
    """
    B, N, _ = node_coords.shape
    device = node_coords.device
    is_live = (node_feats[:, :, 3] > 0.5) & node_mask
    eps = start.new_tensor(CONTACT_GEOMETRY_OPEN_TRAVEL_EPS_NORM)

    seg = end - start
    seg_len = torch.norm(seg, dim=-1).clamp(min=1e-8)
    unit = seg / seg_len.unsqueeze(-1)
    blockers = node_coords.view(B, 1, 1, 1, N, 2)
    rel = blockers - start.unsqueeze(4)
    b = (rel * unit.unsqueeze(4)).sum(dim=-1)
    d2 = (rel * rel).sum(dim=-1) - b.pow(2)
    inside = (INFLATED_STONE_RADIUS_NORM ** 2 - d2).clamp(min=0.0)
    lam = b - torch.sqrt(inside)

    tgt_idx = torch.arange(N, device=device).view(1, 1, N, 1, 1)
    block_idx = torch.arange(N, device=device).view(1, 1, 1, 1, N)
    valid = (
        is_live.view(B, 1, 1, 1, N)
        & (block_idx != tgt_idx)
        & (b > eps)
        & (d2 <= INFLATED_STONE_RADIUS_NORM ** 2)
        & (lam > eps)
        & (lam <= seg_len.unsqueeze(-1))
    )
    lam_masked = torch.where(valid, lam, torch.full_like(lam, float("inf")))
    hit_lam, hit_idx = lam_masked.min(dim=-1)
    hit = torch.isfinite(hit_lam)
    hit_point = start + hit_lam.clamp(max=CLEARANCE_CAP_NORM).unsqueeze(-1) * unit

    k_count = hit_idx.shape[3]
    expand_coords = node_coords.view(B, 1, 1, 1, N, 2).expand(-1, start.shape[1], N, k_count, -1, -1)
    l_coords = torch.gather(
        expand_coords,
        4,
        hit_idx[..., None, None].expand(B, start.shape[1], N, k_count, 1, 2),
    ).squeeze(4)
    second_out = l_coords - hit_point
    second_out = second_out / torch.norm(second_out, dim=-1, keepdim=True).clamp(min=1e-6)

    seg2 = hit_point - start
    seg_len2 = (seg2 * seg2).sum(dim=-1).clamp(min=1e-12)
    rel2 = blockers - start.unsqueeze(4)
    u2 = ((rel2 * seg2.unsqueeze(4)).sum(dim=-1) / seg_len2.unsqueeze(4)).clamp(0.0, 1.0)
    closest = start.unsqueeze(4) + u2.unsqueeze(-1) * seg2.unsqueeze(4)
    dist = torch.norm(blockers - closest, dim=-1)
    blocker_mask = (
        is_live.view(B, 1, 1, 1, N)
        & (block_idx != tgt_idx)
        & (block_idx != hit_idx.unsqueeze(-1))
    )
    dist = torch.where(blocker_mask, dist, torch.full_like(dist, float("inf")))
    min_dist = dist.amin(dim=-1)
    no_blockers = ~blocker_mask.any(dim=-1)
    min_dist = torch.where(no_blockers, torch.full_like(min_dist, CLEARANCE_CAP_NORM), min_dist)
    corridor_clear = min_dist - INFLATED_STONE_RADIUS_NORM
    return hit, hit_idx, hit_point, l_coords, second_out, corridor_clear


def _contact_geometry_release_product_edge_scalars(
    node_coords,
    node_feats,
    node_mask,
    include_clearance=False,
    include_alignment=False,
    include_reach_alignment=False,
    include_onehop=False,
    include_kinds=("score", "takeout", "center", "semi"),
    source_mode="release_only",
    binary_reach=False,
    pad=True,
):
    """
    Release-source -> live-target edge scalars derived from the contact-geometry
    diagnostic features.
    """
    B, N, _ = node_coords.shape
    device = node_coords.device
    dtype = node_coords.dtype
    is_live = (node_feats[:, :, 3] > 0.5) & node_mask
    release_sources, _ = _source_landmark_masks(node_coords, node_feats, node_mask)
    if source_mode not in {"release_only", "all_sources"}:
        raise ValueError(f"Unknown contact-geometry source_mode: {source_mode}")
    if source_mode == "release_only" and not bool(release_sources.any()):
        return torch.zeros(B, N, N, EDGE_SCALAR_DIM, device=device, dtype=dtype)

    if source_mode == "release_only":
        n_source = int(release_sources.sum(dim=1).min().item())
        source_mask = release_sources
        source_order = torch.argsort(source_mask.to(torch.int64), dim=1, descending=True)
        source_idx = source_order[:, :n_source]
        source = torch.gather(node_coords, 1, source_idx.unsqueeze(-1).expand(B, n_source, 2))
        target = node_coords.unsqueeze(1).unsqueeze(3).expand(-1, n_source, -1, -1, -1)
        source_exp = source.unsqueeze(2).unsqueeze(3).expand(-1, -1, N, CONTACT_GEOMETRY_SAMPLE_COUNT, -1)
        source_valid = source_mask[:, :0]  # sentinel, unused in this branch
    else:
        n_source = N
        source_idx = torch.arange(N, device=device).view(1, N).expand(B, -1)
        source = node_coords
        target = node_coords.unsqueeze(1).unsqueeze(3).expand(-1, N, -1, -1, -1)
        source_exp = source.unsqueeze(2).unsqueeze(3).expand(-1, -1, N, CONTACT_GEOMETRY_SAMPLE_COUNT, -1)
        source_valid = is_live | release_sources

    offsets = _contact_geometry_sample_offsets(device, dtype)
    curvatures = _contact_geometry_curvature_bank(device, dtype)
    t_curve = torch.linspace(0.0, 1.0, CONTACT_GEOMETRY_CURVE_POINTS, device=device, dtype=dtype)

    source_vec = source.unsqueeze(2) - node_coords.unsqueeze(1)
    source_dist = torch.norm(source_vec, dim=-1, keepdim=True).clamp(min=1e-6)
    source_dir = source_vec / source_dist
    side_dir = torch.stack([-source_dir[..., 1], source_dir[..., 0]], dim=-1)
    contact_dirs = (
        torch.cos(offsets).view(1, 1, 1, -1, 1) * source_dir.unsqueeze(3)
        + torch.sin(offsets).view(1, 1, 1, -1, 1) * side_dir.unsqueeze(3)
    )
    out_dirs = -contact_dirs
    contacts = target + INFLATED_STONE_RADIUS_NORM * contact_dirs

    incoming_vec = contacts - source_exp
    incoming_unit = incoming_vec / torch.norm(incoming_vec, dim=-1, keepdim=True).clamp(min=1e-6)
    align = torch.clamp((incoming_unit * out_dirs).sum(dim=-1), min=0.0, max=1.0)

    chord = torch.norm(incoming_vec, dim=-1, keepdim=True).clamp(min=1e-6)
    perp = torch.stack([-incoming_vec[..., 1], incoming_vec[..., 0]], dim=-1) / chord
    control = 0.5 * (source_exp + contacts) + curvatures.view(1, 1, 1, -1, 1) * chord * perp
    one_minus = (1.0 - t_curve).view(1, 1, 1, 1, -1, 1)
    tt = t_curve.view(1, 1, 1, 1, -1, 1)
    curve = (
        one_minus.pow(2) * source_exp.unsqueeze(4)
        + 2.0 * one_minus * tt * control.unsqueeze(4)
        + tt.pow(2) * contacts.unsqueeze(4)
    )
    curve_p0 = curve[..., :-1, :]
    curve_p1 = curve[..., 1:, :]
    curve_seg = curve_p1 - curve_p0
    curve_seg_len2 = (curve_seg * curve_seg).sum(dim=-1).clamp(min=1e-12)
    blockers = node_coords.view(B, 1, 1, 1, 1, N, 2)
    rel = blockers - curve_p0.unsqueeze(5)
    u = ((rel * curve_seg.unsqueeze(5)).sum(dim=-1) / curve_seg_len2.unsqueeze(5)).clamp(0.0, 1.0)
    closest = curve_p0.unsqueeze(5) + u.unsqueeze(-1) * curve_seg.unsqueeze(5)
    dist = torch.norm(blockers - closest, dim=-1)
    tgt_idx = torch.arange(N, device=device).view(1, 1, N, 1, 1, 1)
    block_idx = torch.arange(N, device=device).view(1, 1, 1, 1, 1, N)
    blocker_mask = is_live.view(B, 1, 1, 1, 1, N) & (block_idx != tgt_idx)
    dist = torch.where(blocker_mask, dist, torch.full_like(dist, float("inf")))
    min_dist = dist.amin(dim=(-1, -2))
    no_blockers = ~blocker_mask.any(dim=-1).squeeze(-1)
    min_dist = torch.where(
        no_blockers.expand_as(min_dist),
        torch.full_like(min_dist, CLEARANCE_CAP_NORM),
        min_dist,
    )
    reach_clear = min_dist - INFLATED_STONE_RADIUS_NORM
    if binary_reach:
        reach_val = (reach_clear > 0.0).to(dtype)
    else:
        reach_val = _norm_positive_clearance(reach_clear)
    reach_valid = reach_clear > 0.0

    target_team = node_feats[:, :, 2].view(B, 1, N, 1).expand(-1, n_source, -1, CONTACT_GEOMETRY_SAMPLE_COUNT)
    target_button_dist = torch.norm(
        node_coords - node_coords.new_tensor([BUTTON_X, BUTTON_Y]).view(1, 1, 2),
        dim=-1,
    ).view(B, 1, N, 1)
    opp_button_dist = _opponent_button_distances(node_coords, node_feats, node_mask).view(B, 1, N, 1)
    score_radius = opp_button_dist
    takeout_radius = (opp_button_dist + STONE_RADIUS_NORM).clamp(min=STONE_RADIUS_NORM, max=HOUSE_RADIUS)
    score_relevant = (target_button_dist > score_radius + 1e-9).expand(-1, n_source, -1, CONTACT_GEOMETRY_SAMPLE_COUNT)
    takeout_relevant = (target_button_dist <= takeout_radius + 1e-9).expand(-1, n_source, -1, CONTACT_GEOMETRY_SAMPLE_COUNT)
    always_relevant = torch.ones_like(score_relevant)

    out_dirs_exp = out_dirs
    target_exp = target
    opp_exp = opp_button_dist.expand(-1, n_source, -1, CONTACT_GEOMETRY_SAMPLE_COUNT)

    if source_mode == "all_sources":
        source_base = source.unsqueeze(2).unsqueeze(3).expand(-1, -1, N, CONTACT_GEOMETRY_SAMPLE_COUNT, -1)
        contacts_base = contacts
        straight_clear = _segment_clearance_edges(
            node_coords,
            node_feats,
            node_mask,
            source_base,
            contacts_base,
        )
        curl_clear = _curl_clearance_edges(
            node_coords,
            node_feats,
            node_mask,
            source_base,
            contacts_base,
        )
        release_edge = release_sources.view(B, N, 1, 1)
        reach_clear = torch.where(release_edge, curl_clear, straight_clear)
        if binary_reach:
            reach_val = (reach_clear > 0.0).to(dtype)
        else:
            reach_val = _norm_positive_clearance(reach_clear)
        src_idx = torch.arange(N, device=device).view(1, N, 1, 1)
        tgt_idx = torch.arange(N, device=device).view(1, 1, N, 1)
        source_facing = (((contacts - target) * source_vec.unsqueeze(3)).sum(dim=-1) > 0.0)
        edge_valid = (
            source_valid.view(B, N, 1, 1)
            & is_live.view(B, 1, N, 1)
            & (src_idx != tgt_idx)
            & source_facing
        )
        reach_valid = (reach_clear > 0.0) & edge_valid

    include_kinds = tuple(include_kinds)
    kinds = [
        ("score", score_relevant),
        ("takeout", takeout_relevant),
        ("center", always_relevant),
        ("semi", always_relevant),
    ]
    kind_features = {}
    for kind, relevant in kinds:
        goal, goal_valid = _contact_geometry_goal_points(target_exp, out_dirs_exp, kind, opp_exp)
        blocked = _segment_blocked_binary_release_targets(node_coords, node_feats, node_mask, target_exp, goal)
        clear = (~blocked).to(dtype)
        paired_valid = relevant & goal_valid & reach_valid & is_live.view(B, 1, N, 1)
        denom = paired_valid.to(dtype).sum(dim=-1, keepdim=True).clamp(min=1.0)
        kind_features[f"{kind}_clear"] = (
            (clear * paired_valid.to(dtype)).sum(dim=-1, keepdim=True) / denom
        )
        kind_features[f"{kind}_align"] = (
            (align * paired_valid.to(dtype)).sum(dim=-1, keepdim=True) / denom
        )
        kind_features[f"{kind}_reach_x_clear"] = (
            (reach_val * clear * paired_valid.to(dtype)).sum(dim=-1, keepdim=True) / denom
        )
        kind_features[f"{kind}_reach_x_align"] = (
            (reach_val * align * paired_valid.to(dtype)).sum(dim=-1, keepdim=True) / denom
        )
        kind_features[f"{kind}_reach_x_align_x_clear"] = (
            (reach_val * align * clear * paired_valid.to(dtype)).sum(dim=-1, keepdim=True) / denom
        )

    if include_onehop:
        boundary_lam = _contact_geometry_boundary_lam(target_exp, out_dirs_exp)
        far_goal = target_exp + boundary_lam.unsqueeze(-1) * out_dirs_exp
        onehop_hit, hit_idx, hit_point, l_coords, second_out, corridor_clear = _first_intersection_from_target_rays(
            node_coords, node_feats, node_mask, target_exp, far_goal
        )
        onehop_valid = onehop_hit & reach_valid & is_live.view(B, 1, N, 1)
        second_out_safe = torch.where(onehop_hit.unsqueeze(-1), second_out, out_dirs_exp)
        second_align = torch.clamp((out_dirs_exp * second_out_safe).sum(dim=-1), min=0.0, max=1.0)
        reach_ray_count = reach_valid.to(dtype).sum(dim=-1, keepdim=True).clamp(min=1.0)
        kind_features["onehop_reach"] = onehop_valid.to(dtype).sum(dim=-1, keepdim=True) / reach_ray_count

        opp_all = _opponent_button_distances(node_coords, node_feats, node_mask)
        opp_expand = opp_all.view(B, 1, 1, 1, N).expand(-1, target_exp.shape[1], N, hit_idx.shape[3], -1)
        hit_opp = torch.gather(opp_expand, 4, hit_idx.unsqueeze(-1)).squeeze(-1)
        l_dist = torch.norm(
            l_coords - node_coords.new_tensor([BUTTON_X, BUTTON_Y]).view(1, 1, 1, 1, 2),
            dim=-1,
        )
        hit_takeout_radius = (hit_opp + STONE_RADIUS_NORM).clamp(min=STONE_RADIUS_NORM, max=HOUSE_RADIUS)
        hit_kinds = [
            ("score", l_dist > hit_opp + 1e-9),
            ("takeout", l_dist <= hit_takeout_radius + 1e-9),
            ("center", torch.ones_like(onehop_valid)),
            ("semi", torch.ones_like(onehop_valid)),
        ]
        target_idx = torch.arange(N, device=device).view(1, 1, N, 1).expand(B, target_exp.shape[1], -1, target_exp.shape[3])
        for kind, second_relevant in hit_kinds:
            goal2, goal2_valid = _contact_geometry_goal_points(l_coords, second_out_safe, kind, hit_opp)
            goal2_safe = torch.where(goal2_valid.unsqueeze(-1), goal2, l_coords)
            blocked2 = _segment_blocked_binary_generic(
                node_coords,
                node_feats,
                node_mask,
                l_coords,
                goal2_safe,
                primary_exclude_idx=hit_idx,
                extra_exclude_idx=target_idx,
            )
            clear2 = (~blocked2).to(dtype)
            valid2 = onehop_valid & second_relevant & goal2_valid
            denom2 = valid2.to(dtype).sum(dim=-1, keepdim=True).clamp(min=1.0)
            kind_features[f"{kind}_onehop_x_clear"] = (
                (clear2 * valid2.to(dtype)).sum(dim=-1, keepdim=True) / denom2
            )
            kind_features[f"{kind}_onehop_x_align_x_clear"] = (
                (second_align * clear2 * valid2.to(dtype)).sum(dim=-1, keepdim=True) / denom2
            )

    reach_mean = reach_val.mean(dim=-1, keepdim=True)
    if source_mode == "release_only":
        edge_ok = (is_live.view(B, 1, N, 1)).to(dtype)
    else:
        src_idx = torch.arange(N, device=device).view(1, N, 1, 1)
        tgt_idx = torch.arange(N, device=device).view(1, 1, N, 1)
        edge_ok = (
            source_valid.view(B, N, 1, 1)
            & is_live.view(B, 1, N, 1)
            & (src_idx != tgt_idx)
        ).to(dtype)
    feat_list = [
        reach_mean,
    ]
    if include_alignment:
        for kind in include_kinds:
            feat_list.append(kind_features[f"{kind}_align"])
    if include_reach_alignment:
        for kind in include_kinds:
            feat_list.append(kind_features[f"{kind}_reach_x_align"])
    for kind in include_kinds:
        feat_list.append(kind_features[f"{kind}_reach_x_align_x_clear"])
    for kind in include_kinds:
        feat_list.append(kind_features[f"{kind}_reach_x_clear"])
    if include_clearance:
        for kind in include_kinds:
            feat_list.append(kind_features[f"{kind}_clear"])
    if include_onehop:
        feat_list.append(kind_features["onehop_reach"])
        for kind in include_kinds:
            feat_list.append(kind_features[f"{kind}_onehop_x_align_x_clear"])
        for kind in include_kinds:
            feat_list.append(kind_features[f"{kind}_onehop_x_clear"])
    release_feats = torch.cat(feat_list, dim=-1) * edge_ok

    raw_dim = release_feats.shape[-1]
    if source_mode == "release_only":
        edge_scalars = torch.zeros(B, N, N, raw_dim, device=device, dtype=dtype)
        scatter_index = source_idx.unsqueeze(-1).unsqueeze(-1).expand(B, n_source, N, raw_dim)
        edge_scalars.scatter_(1, scatter_index, release_feats)
    else:
        edge_scalars = release_feats
    if pad:
        return _pad_edge_scalars(edge_scalars)
    return edge_scalars


def _first_intersection_on_live_stone(node_coords, node_feats, node_mask, start, end):
    """
    First inflated live-stone disk intersected by each finite segment.
    Returns hit mask, hit index, hit point, and corridor clearance to hit.
    """
    B, N, _ = node_coords.shape
    device = node_coords.device
    dtype = node_coords.dtype
    is_live = (node_feats[:, :, 3] > 0.5) & node_mask

    seg = end - start
    seg_len = torch.norm(seg, dim=-1).clamp(min=1e-8)
    unit = seg / seg_len.unsqueeze(-1)
    blockers = node_coords.view(B, 1, 1, 1, N, 2)
    rel = blockers - start.unsqueeze(4)
    b = (rel * unit.unsqueeze(4)).sum(dim=-1)
    d2 = (rel * rel).sum(dim=-1) - b.pow(2)
    inside = (INFLATED_STONE_RADIUS_NORM ** 2 - d2).clamp(min=0.0)
    lam = b - torch.sqrt(inside)

    src_idx = torch.arange(N, device=device).view(1, N, 1, 1, 1)
    tgt_idx = torch.arange(N, device=device).view(1, 1, N, 1, 1)
    block_idx = torch.arange(N, device=device).view(1, 1, 1, 1, N)
    valid = (
        is_live.view(B, 1, 1, 1, N)
        & (block_idx != src_idx)
        & (block_idx != tgt_idx)
        & (b > 0.0)
        & (d2 <= INFLATED_STONE_RADIUS_NORM ** 2)
        & (lam >= 0.0)
        & (lam <= seg_len.unsqueeze(-1))
    )
    lam_masked = torch.where(valid, lam, torch.full_like(lam, float("inf")))
    hit_lam, hit_idx = lam_masked.min(dim=-1)
    hit = torch.isfinite(hit_lam)
    hit_point = start + hit_lam.clamp(max=CLEARANCE_CAP_NORM).unsqueeze(-1) * unit
    l_coords = torch.gather(
        node_coords.view(B, 1, 1, 1, N, 2).expand(-1, N, N, start.shape[3], -1, -1),
        4,
        hit_idx[..., None, None].expand(hit_idx.shape[0], hit_idx.shape[1], hit_idx.shape[2], hit_idx.shape[3], 1, 2),
    ).squeeze(4)
    second_out = l_coords - hit_point
    second_out = second_out / torch.norm(second_out, dim=-1, keepdim=True).clamp(min=1e-6)

    # Clearance on the first corridor, excluding the hit stone as the intended collision.
    seg2 = hit_point - start
    seg_len2 = (seg2 * seg2).sum(dim=-1).clamp(min=1e-12)
    rel2 = blockers - start.unsqueeze(4)
    u2 = ((rel2 * seg2.unsqueeze(4)).sum(dim=-1) / seg_len2.unsqueeze(4)).clamp(0.0, 1.0)
    closest = start.unsqueeze(4) + u2.unsqueeze(-1) * seg2.unsqueeze(4)
    dist = torch.norm(blockers - closest, dim=-1)
    blocker_mask = (
        is_live.view(B, 1, 1, 1, N)
        & (block_idx != src_idx)
        & (block_idx != tgt_idx)
        & (block_idx != hit_idx.unsqueeze(-1))
    )
    dist = torch.where(blocker_mask, dist, torch.full_like(dist, float("inf")))
    min_dist = dist.amin(dim=-1)
    no_blockers = ~blocker_mask.any(dim=-1)
    min_dist = torch.where(no_blockers, torch.full_like(min_dist, CLEARANCE_CAP_NORM), min_dist)
    corridor_clear = min_dist - INFLATED_STONE_RADIUS_NORM
    return hit, hit_idx, hit_point, l_coords, second_out, corridor_clear


def _oldbest_curl_arc_reach_outgoing_raw_edge_scalars(node_coords, node_feats, node_mask, c=None):
    B, N, _ = node_coords.shape
    scorability = _compute_unoccluded_angular_spans(node_coords, node_feats, node_mask)
    scorability = scorability.unsqueeze(1).expand(-1, N, -1, -1)
    _, feasible, diversity, score_out, takeout_out = _compute_curl_arc_reachability_and_outgoing(
        node_coords, node_feats, node_mask, c=c
    )
    pairwise_reach = _compute_pairwise_unoccluded_angular_spans(node_coords, node_feats, node_mask)
    release_sources, _ = _source_landmark_masks(node_coords, node_feats, node_mask)
    release_edge = release_sources.view(B, N, 1, 1)
    straight_score, straight_takeout = _compute_pairwise_outgoing_compatibility(
        node_coords, node_feats, node_mask, c=c
    )
    curl_reach = feasible * diversity
    reach = torch.where(release_edge, curl_reach, pairwise_reach)
    score_out = torch.where(release_edge, score_out, straight_score)
    takeout_out = torch.where(release_edge, takeout_out, straight_takeout)
    return torch.cat(
        [
            scorability,
            reach,
            score_out,
            takeout_out,
            reach * torch.maximum(score_out, takeout_out),
        ],
        dim=-1,
    )


def _oldbest_plus_contact_geometry_edge_scalars(
    node_coords,
    node_feats,
    node_mask,
    c=None,
    include_clearance=True,
    include_alignment=False,
    include_reach_alignment=False,
    include_onehop=False,
):
    oldbest = _oldbest_curl_arc_reach_outgoing_raw_edge_scalars(
        node_coords, node_feats, node_mask, c=c
    )
    contact = _contact_geometry_release_product_edge_scalars(
        node_coords,
        node_feats,
        node_mask,
        include_clearance=include_clearance,
        include_alignment=include_alignment,
        include_reach_alignment=include_reach_alignment,
        include_onehop=include_onehop,
        pad=False,
    )
    return _pad_edge_scalars(torch.cat([oldbest, contact], dim=-1))


def _contact_geometry_stonepair_channel_mask(device, dtype, stack):
    mask = torch.zeros(21, device=device, dtype=dtype)
    if stack == "full21":
        mask[:] = 1.0
        return mask
    if stack == "products13":
        mask[0] = 1.0
        mask[5:17] = 1.0
        return mask
    if stack == "basic9":
        mask[0] = 1.0
        mask[1:5] = 1.0
        mask[17:21] = 1.0
        return mask
    if stack == "rac5":
        mask[0] = 1.0
        mask[9:13] = 1.0
        return mask
    if stack == "score6":
        mask[0] = 1.0
        mask[1] = 1.0
        mask[5] = 1.0
        mask[9] = 1.0
        mask[13] = 1.0
        mask[17] = 1.0
        return mask
    if stack == "scoresemi11":
        mask[0] = 1.0
        mask[1] = 1.0
        mask[4] = 1.0
        mask[5] = 1.0
        mask[8] = 1.0
        mask[9] = 1.0
        mask[12] = 1.0
        mask[13] = 1.0
        mask[16] = 1.0
        mask[17] = 1.0
        mask[20] = 1.0
        return mask
    raise ValueError(f"Unknown stonepair stack: {stack}")


def _stonepair_feature_scale():
    try:
        return float(os.environ.get("GNN_STONEPAIR_FEATURE_SCALE", "1.0"))
    except ValueError:
        return 1.0


def _stonepair_hop1_pair_chunk():
    try:
        return max(1, int(os.environ.get("GNN_STONEPAIR_HOP1_PAIR_CHUNK", "8")))
    except ValueError:
        return 8


def _release_to_inverse_stonesource_hop1_binary(node_coords, node_feats, node_mask, src_idx, source_contacts):
    """
    For each sampled inverse contact on an intermediate source stone, return whether
    any release landmark can reach that contact under the same sampled curl proxy
    formulation used by release->stone contact reach.

    src_idx: (B, P)
    source_contacts: (B, P, K, 2)
    returns: (B, P, K) bool
    """
    B, N, _ = node_coords.shape
    device = node_coords.device
    dtype = node_coords.dtype
    is_live = (node_feats[:, :, 3] > 0.5) & node_mask
    release_sources, _ = _source_landmark_masks(node_coords, node_feats, node_mask)
    if not bool(release_sources.any()):
        return torch.zeros(source_contacts.shape[:-1], device=device, dtype=torch.bool)

    n_release = int(release_sources.sum(dim=1).min().item())
    release_order = torch.argsort(release_sources.to(torch.int64), dim=1, descending=True)
    release_idx = release_order[:, :n_release]
    release_pts = torch.gather(node_coords, 1, release_idx.unsqueeze(-1).expand(B, n_release, 2))

    P = src_idx.shape[1]
    K = source_contacts.shape[2]
    curvatures = _contact_geometry_curvature_bank(device, dtype)
    t_curve = torch.linspace(0.0, 1.0, CONTACT_GEOMETRY_CURVE_POINTS, device=device, dtype=dtype)
    pair_chunk = _stonepair_hop1_pair_chunk()
    out = torch.zeros(B, P, K, device=device, dtype=torch.bool)
    one_minus = (1.0 - t_curve).view(1, 1, 1, 1, -1, 1)
    tt = t_curve.view(1, 1, 1, 1, -1, 1)
    blockers = node_coords.view(B, 1, 1, 1, 1, N, 2)
    block_idx = torch.arange(N, device=device).view(1, 1, 1, 1, 1, N)
    live_blockers = is_live.view(B, 1, 1, 1, 1, N)
    for p0 in range(0, P, pair_chunk):
        p1 = min(P, p0 + pair_chunk)
        src_idx_chunk = src_idx[:, p0:p1]
        end = source_contacts[:, p0:p1, :].unsqueeze(1).expand(B, n_release, p1 - p0, K, 2)
        start = release_pts.unsqueeze(2).unsqueeze(3).expand(B, n_release, p1 - p0, K, 2)
        incoming_vec = end - start
        chord = torch.norm(incoming_vec, dim=-1, keepdim=True).clamp(min=1e-6)
        perp = torch.stack([-incoming_vec[..., 1], incoming_vec[..., 0]], dim=-1) / chord
        control = 0.5 * (start + end) + curvatures.view(1, 1, 1, K, 1) * chord * perp
        curve = (
            one_minus.pow(2) * start.unsqueeze(4)
            + 2.0 * one_minus * tt * control.unsqueeze(4)
            + tt.pow(2) * end.unsqueeze(4)
        )
        curve_p0 = curve[..., :-1, :]
        curve_p1 = curve[..., 1:, :]
        curve_seg = curve_p1 - curve_p0
        curve_seg_len2 = (curve_seg * curve_seg).sum(dim=-1).clamp(min=1e-12)
        rel = blockers - curve_p0.unsqueeze(5)
        u = ((rel * curve_seg.unsqueeze(5)).sum(dim=-1) / curve_seg_len2.unsqueeze(5)).clamp(0.0, 1.0)
        closest = curve_p0.unsqueeze(5) + u.unsqueeze(-1) * curve_seg.unsqueeze(5)
        dist = torch.norm(blockers - closest, dim=-1)
        blocker_mask = live_blockers & (
            block_idx != src_idx_chunk.view(B, 1, p1 - p0, 1, 1, 1)
        )
        dist = torch.where(blocker_mask, dist, torch.full_like(dist, float("inf")))
        min_dist = dist.amin(dim=(-1, -2))
        any_blockers = blocker_mask.any(dim=-1).squeeze(-1)
        min_dist = torch.where(
            any_blockers.expand_as(min_dist),
            min_dist,
            torch.full_like(min_dist, CLEARANCE_CAP_NORM),
        )
        reach_clear = min_dist - INFLATED_STONE_RADIUS_NORM
        out[:, p0:p1, :] = (reach_clear > 0.0).any(dim=1)
    return out


def _contact_geometry_stonepair_sparse_full21_raw(
    node_coords, node_feats, node_mask, binary_reach=False, use_release_hop1=False
):
    """
    Sparse live-stone-pair contact-geometry scalars with full 21-channel layout.
    These are only computed for live stone->live stone pairs, then scattered back.
    """
    B, N, _ = node_coords.shape
    device = node_coords.device
    dtype = node_coords.dtype
    is_live = (node_feats[:, :, 3] > 0.5) & node_mask
    stone_live = is_live[:, :NUM_STONES]
    pair_mask = stone_live.unsqueeze(2) & stone_live.unsqueeze(1)
    pair_mask = pair_mask & (~torch.eye(NUM_STONES, device=device, dtype=torch.bool).unsqueeze(0))
    stone_y = node_coords[:, :NUM_STONES, 1]
    source_above_target = stone_y.unsqueeze(2) > stone_y.unsqueeze(1)
    pair_mask = pair_mask & source_above_target
    max_pairs = int(pair_mask.sum(dim=(1, 2)).max().item())
    out = torch.zeros(B, N, N, 21, device=device, dtype=dtype)
    if max_pairs == 0:
        return out

    flat_mask = pair_mask.view(B, -1)
    order = torch.argsort(flat_mask.to(torch.int64), dim=1, descending=True)
    flat_idx = order[:, :max_pairs]
    pair_valid = torch.gather(flat_mask, 1, flat_idx)
    src_idx = flat_idx // NUM_STONES
    tgt_idx = flat_idx % NUM_STONES

    stone_coords = node_coords[:, :NUM_STONES, :]
    source = torch.gather(stone_coords, 1, src_idx.unsqueeze(-1).expand(B, max_pairs, 2))
    target = torch.gather(stone_coords, 1, tgt_idx.unsqueeze(-1).expand(B, max_pairs, 2))

    offsets = _contact_geometry_sample_offsets(device, dtype)
    K = offsets.shape[0]
    source_vec = source - target
    source_dist = torch.norm(source_vec, dim=-1, keepdim=True).clamp(min=1e-6)
    source_dir = source_vec / source_dist
    side_dir = torch.stack([-source_dir[..., 1], source_dir[..., 0]], dim=-1)
    contact_dirs = (
        torch.cos(offsets).view(1, 1, K, 1) * source_dir.unsqueeze(2)
        + torch.sin(offsets).view(1, 1, K, 1) * side_dir.unsqueeze(2)
    )
    out_dirs = -contact_dirs
    contacts = target.unsqueeze(2) + INFLATED_STONE_RADIUS_NORM * contact_dirs
    throw_dir = target.new_tensor([0.0, -1.0]).view(1, 1, 1, 2)
    forward_mask = (out_dirs * throw_dir).sum(dim=-1) > 0.0

    incoming_vec = contacts - source.unsqueeze(2)
    incoming_unit = incoming_vec / torch.norm(incoming_vec, dim=-1, keepdim=True).clamp(min=1e-6)
    align = torch.clamp((incoming_unit * out_dirs).sum(dim=-1), min=0.0, max=1.0)

    source_facing = ((contacts - target.unsqueeze(2)) * source_vec.unsqueeze(2)).sum(dim=-1) > 0.0
    sample_valid = source_facing & forward_mask & pair_valid.unsqueeze(-1)
    if use_release_hop1:
        desired_dir = incoming_unit
        inverse_source_contacts = source.unsqueeze(2) - INFLATED_STONE_RADIUS_NORM * desired_dir
        hop1_valid = _release_to_inverse_stonesource_hop1_binary(
            node_coords, node_feats, node_mask, src_idx, inverse_source_contacts
        )
        reach_val = hop1_valid.to(dtype)
        reach_valid = hop1_valid & sample_valid
    else:
        reach_clear = _segment_clearance_pairlist(
            node_coords,
            node_feats,
            node_mask,
            source.unsqueeze(2).expand(-1, -1, K, -1),
            contacts,
            src_idx,
            tgt_idx,
        )
        if binary_reach:
            reach_val = (reach_clear > 0.0).to(dtype)
        else:
            reach_val = _norm_positive_clearance(reach_clear)
        reach_valid = (reach_clear > 0.0) & sample_valid
    reach_denom = sample_valid.to(dtype).sum(dim=-1, keepdim=True).clamp(min=1.0)
    reach_mean = (
        (reach_val * sample_valid.to(dtype)).sum(dim=-1, keepdim=True) / reach_denom
    ) * pair_valid.unsqueeze(-1).to(dtype)

    button_dist = torch.norm(
        node_coords[:, :NUM_STONES, :] - node_coords.new_tensor([BUTTON_X, BUTTON_Y]).view(1, 1, 2),
        dim=-1,
    )
    target_button_dist = torch.gather(button_dist, 1, tgt_idx).unsqueeze(-1).expand(-1, -1, K)
    opp_dist_all = _opponent_button_distances(node_coords, node_feats, node_mask)[:, :NUM_STONES]
    opp_button_dist = torch.gather(opp_dist_all, 1, tgt_idx).unsqueeze(-1).expand(-1, -1, K)
    score_radius = opp_button_dist
    takeout_radius = (opp_button_dist + STONE_RADIUS_NORM).clamp(min=STONE_RADIUS_NORM, max=HOUSE_RADIUS)
    score_relevant = target_button_dist > score_radius + 1e-9
    takeout_relevant = target_button_dist <= takeout_radius + 1e-9
    always_relevant = torch.ones_like(score_relevant)

    kinds = [
        ("score", score_relevant),
        ("takeout", takeout_relevant),
        ("center", always_relevant),
        ("semi", always_relevant),
    ]
    kind_features = {}
    target_exp = target.unsqueeze(2)
    out_dirs_exp = out_dirs
    for kind, relevant in kinds:
        goal, goal_valid = _contact_geometry_goal_points(target_exp, out_dirs_exp, kind, opp_button_dist)
        blocked = _segment_blocked_binary_pairlist(
            node_coords,
            node_feats,
            node_mask,
            target_exp,
            goal,
            primary_exclude_idx=tgt_idx,
            extra_exclude_idx=src_idx,
        )
        clear = (~blocked).to(dtype)
        paired_valid = relevant & goal_valid & reach_valid
        denom = paired_valid.to(dtype).sum(dim=-1, keepdim=True).clamp(min=1.0)
        kind_features[f"{kind}_align"] = (
            (align * paired_valid.to(dtype)).sum(dim=-1, keepdim=True) / denom
        )
        kind_features[f"{kind}_reach_x_align"] = (
            (reach_val * align * paired_valid.to(dtype)).sum(dim=-1, keepdim=True) / denom
        )
        kind_features[f"{kind}_reach_x_align_x_clear"] = (
            (reach_val * align * clear * paired_valid.to(dtype)).sum(dim=-1, keepdim=True) / denom
        )
        kind_features[f"{kind}_reach_x_clear"] = (
            (reach_val * clear * paired_valid.to(dtype)).sum(dim=-1, keepdim=True) / denom
        )
        kind_features[f"{kind}_clear"] = (
            (clear * paired_valid.to(dtype)).sum(dim=-1, keepdim=True) / denom
        )

    pair_feats = torch.cat(
        [
            reach_mean,
            kind_features["score_align"],
            kind_features["takeout_align"],
            kind_features["center_align"],
            kind_features["semi_align"],
            kind_features["score_reach_x_align"],
            kind_features["takeout_reach_x_align"],
            kind_features["center_reach_x_align"],
            kind_features["semi_reach_x_align"],
            kind_features["score_reach_x_align_x_clear"],
            kind_features["takeout_reach_x_align_x_clear"],
            kind_features["center_reach_x_align_x_clear"],
            kind_features["semi_reach_x_align_x_clear"],
            kind_features["score_reach_x_clear"],
            kind_features["takeout_reach_x_clear"],
            kind_features["center_reach_x_clear"],
            kind_features["semi_reach_x_clear"],
            kind_features["score_clear"],
            kind_features["takeout_clear"],
            kind_features["center_clear"],
            kind_features["semi_clear"],
        ],
        dim=-1,
    ) * pair_valid.unsqueeze(-1).to(dtype)

    batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, max_pairs)
    out[batch_idx, src_idx, tgt_idx, :] = pair_feats
    return out


def _contact_geometry_release_plus_stonepairs_edge_scalars(
    node_coords,
    node_feats,
    node_mask,
    stone_stack="full21",
    binary_reach=False,
    stone_hop1=False,
):
    release = _contact_geometry_release_product_edge_scalars(
        node_coords,
        node_feats,
        node_mask,
        include_clearance=True,
        include_alignment=True,
        include_reach_alignment=True,
        include_onehop=False,
        source_mode="release_only",
        binary_reach=binary_reach,
        pad=False,
    )
    stone = _contact_geometry_stonepair_sparse_full21_raw(
        node_coords,
        node_feats,
        node_mask,
        binary_reach=binary_reach,
        use_release_hop1=stone_hop1,
    )
    stone = stone * _contact_geometry_stonepair_channel_mask(node_coords.device, node_coords.dtype, stone_stack).view(1, 1, 1, -1)
    stone = stone * stone.new_tensor(_stonepair_feature_scale())
    return _pad_edge_scalars(release + stone)


def _contact_geometry_release_concat_stonepairs_edge_scalars(
    node_coords,
    node_feats,
    node_mask,
    *,
    release_include_clearance,
    release_include_alignment,
    release_include_reach_alignment,
    release_include_onehop,
    release_include_kinds=("score", "takeout", "center", "semi"),
    stone_stack="products13",
    binary_reach=True,
    stone_hop1=True,
):
    release = _contact_geometry_release_product_edge_scalars(
        node_coords,
        node_feats,
        node_mask,
        include_clearance=release_include_clearance,
        include_alignment=release_include_alignment,
        include_reach_alignment=release_include_reach_alignment,
        include_onehop=release_include_onehop,
        include_kinds=release_include_kinds,
        source_mode="release_only",
        binary_reach=binary_reach,
        pad=False,
    )
    stone = _contact_geometry_stonepair_sparse_full21_raw(
        node_coords,
        node_feats,
        node_mask,
        binary_reach=binary_reach,
        use_release_hop1=stone_hop1,
    )
    stone = stone * _contact_geometry_stonepair_channel_mask(
        node_coords.device, node_coords.dtype, stone_stack
    ).view(1, 1, 1, -1)
    stone = stone * stone.new_tensor(_stonepair_feature_scale())
    return _pad_edge_scalars(torch.cat([release, stone], dim=-1))


def _oldbest_plus_stonepairs_edge_scalars(node_coords, node_feats, node_mask, c=None, stone_stack="full21"):
    oldbest = _oldbest_curl_arc_reach_outgoing_raw_edge_scalars(
        node_coords, node_feats, node_mask, c=c
    )
    stone = _contact_geometry_stonepair_sparse_full21_raw(node_coords, node_feats, node_mask)
    stone = stone * _contact_geometry_stonepair_channel_mask(
        node_coords.device, node_coords.dtype, stone_stack
    ).view(1, 1, 1, -1)
    stone = stone * stone.new_tensor(_stonepair_feature_scale())
    return _pad_edge_scalars(torch.cat([oldbest, stone], dim=-1))


def _goal_contact_chain_2hop_edge_scalars(node_coords, node_feats, node_mask, c=None, pad=True):
    """
    Contact-aware goal features without raw reach mass or product channels.
    Edge scalar order:
      score_quality, takeout_quality, score_width, takeout_width,
      score_open_margin, takeout_open_margin, two_hop_score_quality,
      two_hop_takeout_quality.
    """
    B, N, _ = node_coords.shape
    device = node_coords.device
    dtype = node_coords.dtype
    is_live = (node_feats[:, :, 3] > 0.5) & node_mask
    release_sources, _ = _source_landmark_masks(node_coords, node_feats, node_mask)
    valid_source = is_live | release_sources
    valid_target = is_live

    dirs = _contact_outgoing_directions(device, dtype)
    K = dirs.shape[0]
    source = node_coords.unsqueeze(2).unsqueeze(3)
    target = node_coords.unsqueeze(1).unsqueeze(3)
    out_dirs = dirs.view(1, 1, 1, K, 2)
    contacts = target - INFLATED_STONE_RADIUS_NORM * out_dirs

    source_to_target = source - target
    source_facing = (((contacts - target) * source_to_target).sum(dim=-1) > 0.0)

    straight_clear = _segment_clearance_edges(
        node_coords,
        node_feats,
        node_mask,
        source.expand(-1, -1, N, K, -1),
        contacts.expand(-1, N, -1, -1, -1),
    )
    curl_clear = _curl_clearance_edges(
        node_coords,
        node_feats,
        node_mask,
        source.expand(-1, -1, N, K, -1),
        contacts.expand(-1, N, -1, -1, -1),
    )
    release_edge = release_sources.view(B, N, 1, 1)
    incoming_clear = torch.where(release_edge, curl_clear, straight_clear)

    src_idx = torch.arange(N, device=device).view(1, N, 1, 1)
    tgt_idx = torch.arange(N, device=device).view(1, 1, N, 1)
    edge_valid = (
        valid_source.view(B, N, 1, 1)
        & valid_target.view(B, 1, N, 1)
        & (src_idx != tgt_idx)
        & source_facing
    )
    reachable = edge_valid & (incoming_clear > 0.0)

    button = node_coords.new_tensor([BUTTON_X, BUTTON_Y]).view(1, 1, 1, 1, 2)
    to_button = button - target
    to_button = to_button / torch.norm(to_button, dim=-1, keepdim=True).clamp(min=1e-6)
    outward = target - button
    outward = outward / torch.norm(outward, dim=-1, keepdim=True).clamp(min=1e-6)
    score_align = torch.clamp((out_dirs * to_button).sum(dim=-1), min=0.0).expand(B, N, N, K)
    takeout_align = torch.clamp((out_dirs * outward).sum(dim=-1), min=0.0).expand(B, N, N, K)

    target_exp = target.expand(-1, N, -1, K, -1)
    score_goal = _goal_points(target_exp, out_dirs.expand(B, N, N, -1, -1), "score")
    opp_dist = _opponent_button_distances(node_coords, node_feats, node_mask).view(B, 1, N, 1).expand(-1, N, -1, K)
    takeout_goal = _goal_points(target_exp, out_dirs.expand(B, N, N, -1, -1), "takeout", opp_dist)
    score_open = _segment_clearance_edges(node_coords, node_feats, node_mask, target_exp, score_goal)
    takeout_open = _segment_clearance_edges(node_coords, node_feats, node_mask, target_exp, takeout_goal)

    inc_norm = _norm_positive_clearance(incoming_clear)
    score_open_norm = _norm_positive_clearance(score_open)
    takeout_open_norm = _norm_positive_clearance(takeout_open)

    neg = torch.full_like(incoming_clear, -1.0e9)
    score_useful = reachable & (score_open > 0.0) & (score_align > 0.0)
    takeout_useful = reachable & (takeout_open > 0.0) & (takeout_align > 0.0)
    score_logits = torch.where(score_useful, score_align + inc_norm + score_open_norm, neg)
    takeout_logits = torch.where(takeout_useful, takeout_align + inc_norm + takeout_open_norm, neg)
    score_weight = F.softmax(score_logits, dim=-1) * score_useful.to(dtype)
    takeout_weight = F.softmax(takeout_logits, dim=-1) * takeout_useful.to(dtype)
    score_weight = score_weight / score_weight.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    takeout_weight = takeout_weight / takeout_weight.sum(dim=-1, keepdim=True).clamp(min=1e-8)

    score_quality = (score_weight * score_align).sum(dim=-1, keepdim=True)
    takeout_quality = (takeout_weight * takeout_align).sum(dim=-1, keepdim=True)
    score_width = score_useful.to(dtype).mean(dim=-1, keepdim=True)
    takeout_width = takeout_useful.to(dtype).mean(dim=-1, keepdim=True)
    score_margin = (score_weight * score_open_norm).sum(dim=-1, keepdim=True)
    takeout_margin = (takeout_weight * takeout_open_norm).sum(dim=-1, keepdim=True)

    def two_hop(goal, align_kind):
        hit, hit_idx, hit_point, l_coords, second_out, corridor_clear = _first_intersection_on_live_stone(
            node_coords, node_feats, node_mask, target_exp, goal
        )
        l_team = torch.gather(
            node_feats[:, :, 2].view(B, 1, 1, 1, N).expand(-1, N, N, K, -1),
            4,
            hit_idx.unsqueeze(-1),
        ).squeeze(-1)
        if align_kind == "score":
            l_goal = _goal_points(l_coords, second_out, "score")
            l_to_button = button - l_coords
            l_to_button = l_to_button / torch.norm(l_to_button, dim=-1, keepdim=True).clamp(min=1e-6)
            second_align = torch.clamp((second_out * l_to_button).sum(dim=-1), min=0.0)
        else:
            opp_all = _opponent_button_distances(node_coords, node_feats, node_mask)
            # Recompute for gathered hit stone indices.
            l_opp = torch.gather(
                opp_all.view(B, 1, 1, 1, N).expand(-1, N, N, K, -1),
                4,
                hit_idx.unsqueeze(-1),
            ).squeeze(-1)
            l_goal = _goal_points(l_coords, second_out, "takeout", l_opp)
            l_outward = l_coords - button
            l_outward = l_outward / torch.norm(l_outward, dim=-1, keepdim=True).clamp(min=1e-6)
            second_align = torch.clamp((second_out * l_outward).sum(dim=-1), min=0.0)

        l_open = _segment_clearance_edges(
            node_coords,
            node_feats,
            node_mask,
            l_coords,
            l_goal,
            exclude_src=True,
            exclude_tgt=True,
            exclude_extra_idx=hit_idx,
        )
        useful = reachable & hit & (corridor_clear > 0.0) & (l_open > 0.0) & (second_align > 0.0)
        logits = torch.where(
            useful,
            second_align
            + inc_norm
            + _norm_positive_clearance(corridor_clear)
            + _norm_positive_clearance(l_open),
            neg,
        )
        weight = F.softmax(logits, dim=-1) * useful.to(dtype)
        weight = weight / weight.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        return (weight * second_align).sum(dim=-1, keepdim=True)

    two_hop_score = two_hop(score_goal, "score")
    two_hop_takeout = two_hop(takeout_goal, "takeout")

    edge_ok = (
        valid_source.view(B, N, 1, 1)
        & valid_target.view(B, 1, N, 1)
        & (src_idx != tgt_idx)
    ).to(dtype)
    scalars = torch.cat(
        [
            score_quality * edge_ok,
            takeout_quality * edge_ok,
            score_width * edge_ok,
            takeout_width * edge_ok,
            score_margin * edge_ok,
            takeout_margin * edge_ok,
            two_hop_score * edge_ok,
            two_hop_takeout * edge_ok,
        ],
        dim=-1,
    )
    if not pad:
        return scalars
    return _pad_edge_scalars(scalars)


def _goal_contact_oldbest_plus_chain_2hop_edge_scalars(node_coords, node_feats, node_mask, c=None):
    oldbest = _oldbest_curl_arc_reach_outgoing_raw_edge_scalars(
        node_coords, node_feats, node_mask, c=c
    )
    chain = _goal_contact_chain_2hop_edge_scalars(
        node_coords, node_feats, node_mask, c=c, pad=False
    )
    return _pad_edge_scalars(torch.cat([oldbest, chain], dim=-1))


def _topk_mean(values, logits, useful, k=3):
    masked_logits = torch.where(useful, logits, torch.full_like(logits, -1.0e9))
    kk = min(int(k), masked_logits.shape[-1])
    top_idx = torch.topk(masked_logits, k=kk, dim=-1).indices
    top_values = torch.gather(values, -1, top_idx)
    top_valid = torch.gather(useful, -1, top_idx).to(values.dtype)
    denom = top_valid.sum(dim=-1, keepdim=True).clamp(min=1.0)
    return (top_values * top_valid).sum(dim=-1, keepdim=True) / denom


def _goal_contact_chain_1hop_topk_edge_scalars(node_coords, node_feats, node_mask, c=None):
    d = _goal_contact_1hop_common(node_coords, node_feats, node_mask)
    edge_ok = d["edge_ok"]
    score_logits = d["score_align"] + d["inc_norm"] + d["score_open_norm"]
    takeout_logits = d["takeout_align"] + d["inc_norm"] + d["takeout_open_norm"]
    score_quality = _topk_mean(d["score_align"], score_logits, d["score_useful"])
    takeout_quality = _topk_mean(d["takeout_align"], takeout_logits, d["takeout_useful"])
    score_width = d["score_useful"].to(edge_ok.dtype).mean(dim=-1, keepdim=True)
    takeout_width = d["takeout_useful"].to(edge_ok.dtype).mean(dim=-1, keepdim=True)
    score_margin = _topk_mean(d["score_open_norm"], score_logits, d["score_useful"])
    takeout_margin = _topk_mean(d["takeout_open_norm"], takeout_logits, d["takeout_useful"])
    return _pad_edge_scalars(
        score_quality * edge_ok,
        takeout_quality * edge_ok,
        score_width * edge_ok,
        takeout_width * edge_ok,
        score_margin * edge_ok,
        takeout_margin * edge_ok,
    )


def _goal_contact_1hop_common(node_coords, node_feats, node_mask):
    B, N, _ = node_coords.shape
    device = node_coords.device
    dtype = node_coords.dtype
    is_live = (node_feats[:, :, 3] > 0.5) & node_mask
    release_sources, _ = _source_landmark_masks(node_coords, node_feats, node_mask)
    valid_source = is_live | release_sources
    valid_target = is_live

    dirs = _contact_outgoing_directions(device, dtype)
    K = dirs.shape[0]
    source = node_coords.unsqueeze(2).unsqueeze(3)
    target = node_coords.unsqueeze(1).unsqueeze(3)
    out_dirs = dirs.view(1, 1, 1, K, 2)
    contacts = target - INFLATED_STONE_RADIUS_NORM * out_dirs

    source_to_target = source - target
    source_facing = (((contacts - target) * source_to_target).sum(dim=-1) > 0.0)

    source_exp = source.expand(-1, -1, N, K, -1)
    contacts_exp = contacts.expand(-1, N, -1, -1, -1)
    straight_clear = _segment_clearance_edges(node_coords, node_feats, node_mask, source_exp, contacts_exp)
    curl_clear = _curl_clearance_edges(node_coords, node_feats, node_mask, source_exp, contacts_exp)
    release_edge = release_sources.view(B, N, 1, 1)
    incoming_clear = torch.where(release_edge, curl_clear, straight_clear)

    src_idx = torch.arange(N, device=device).view(1, N, 1, 1)
    tgt_idx = torch.arange(N, device=device).view(1, 1, N, 1)
    edge_valid = (
        valid_source.view(B, N, 1, 1)
        & valid_target.view(B, 1, N, 1)
        & (src_idx != tgt_idx)
        & source_facing
    )
    reachable = edge_valid & (incoming_clear > 0.0)

    button = node_coords.new_tensor([BUTTON_X, BUTTON_Y]).view(1, 1, 1, 1, 2)
    to_button = button - target
    to_button = to_button / torch.norm(to_button, dim=-1, keepdim=True).clamp(min=1e-6)
    outward = target - button
    outward = outward / torch.norm(outward, dim=-1, keepdim=True).clamp(min=1e-6)
    score_align = torch.clamp((out_dirs * to_button).sum(dim=-1), min=0.0).expand(B, N, N, K)
    takeout_align = torch.clamp((out_dirs * outward).sum(dim=-1), min=0.0).expand(B, N, N, K)

    target_exp = target.expand(-1, N, -1, K, -1)
    out_dirs_exp = out_dirs.expand(B, N, N, -1, -1)
    score_goal = _goal_points(target_exp, out_dirs_exp, "score")
    opp_dist = _opponent_button_distances(node_coords, node_feats, node_mask).view(B, 1, N, 1).expand(-1, N, -1, K)
    takeout_goal = _goal_points(target_exp, out_dirs_exp, "takeout", opp_dist)
    score_open = _segment_clearance_edges(node_coords, node_feats, node_mask, target_exp, score_goal)
    takeout_open = _segment_clearance_edges(node_coords, node_feats, node_mask, target_exp, takeout_goal)

    inc_norm = _norm_positive_clearance(incoming_clear)
    score_open_norm = _norm_positive_clearance(score_open)
    takeout_open_norm = _norm_positive_clearance(takeout_open)

    neg = torch.full_like(incoming_clear, -1.0e9)
    score_useful = reachable & (score_open > 0.0) & (score_align > 0.0)
    takeout_useful = reachable & (takeout_open > 0.0) & (takeout_align > 0.0)
    score_logits = torch.where(score_useful, score_align + inc_norm + score_open_norm, neg)
    takeout_logits = torch.where(takeout_useful, takeout_align + inc_norm + takeout_open_norm, neg)
    score_weight = F.softmax(score_logits, dim=-1) * score_useful.to(dtype)
    takeout_weight = F.softmax(takeout_logits, dim=-1) * takeout_useful.to(dtype)
    score_weight = score_weight / score_weight.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    takeout_weight = takeout_weight / takeout_weight.sum(dim=-1, keepdim=True).clamp(min=1e-8)

    edge_ok = (
        valid_source.view(B, N, 1, 1)
        & valid_target.view(B, 1, N, 1)
        & (src_idx != tgt_idx)
    ).to(dtype)
    return {
        "edge_ok": edge_ok,
        "score_weight": score_weight,
        "takeout_weight": takeout_weight,
        "score_align": score_align,
        "takeout_align": takeout_align,
        "inc_norm": inc_norm,
        "score_open_norm": score_open_norm,
        "takeout_open_norm": takeout_open_norm,
        "score_useful": score_useful,
        "takeout_useful": takeout_useful,
    }


def _goal_contact_chain_1hop_edge_scalars(node_coords, node_feats, node_mask, c=None):
    d = _goal_contact_1hop_common(node_coords, node_feats, node_mask)
    edge_ok = d["edge_ok"]
    score_quality = (d["score_weight"] * d["score_align"]).sum(dim=-1, keepdim=True)
    takeout_quality = (d["takeout_weight"] * d["takeout_align"]).sum(dim=-1, keepdim=True)
    score_width = d["score_useful"].to(edge_ok.dtype).mean(dim=-1, keepdim=True)
    takeout_width = d["takeout_useful"].to(edge_ok.dtype).mean(dim=-1, keepdim=True)
    score_margin = (d["score_weight"] * d["score_open_norm"]).sum(dim=-1, keepdim=True)
    takeout_margin = (d["takeout_weight"] * d["takeout_open_norm"]).sum(dim=-1, keepdim=True)
    return _pad_edge_scalars(
        score_quality * edge_ok,
        takeout_quality * edge_ok,
        score_width * edge_ok,
        takeout_width * edge_ok,
        score_margin * edge_ok,
        takeout_margin * edge_ok,
    )


def _goal_contact_1hop_terms_edge_scalars(node_coords, node_feats, node_mask, c=None):
    d = _goal_contact_1hop_common(node_coords, node_feats, node_mask)
    edge_ok = d["edge_ok"]
    score_align = (d["score_weight"] * d["score_align"]).sum(dim=-1, keepdim=True)
    takeout_align = (d["takeout_weight"] * d["takeout_align"]).sum(dim=-1, keepdim=True)
    score_incoming_open = (d["score_weight"] * d["inc_norm"]).sum(dim=-1, keepdim=True)
    takeout_incoming_open = (d["takeout_weight"] * d["inc_norm"]).sum(dim=-1, keepdim=True)
    score_goal_open = (d["score_weight"] * d["score_open_norm"]).sum(dim=-1, keepdim=True)
    takeout_goal_open = (d["takeout_weight"] * d["takeout_open_norm"]).sum(dim=-1, keepdim=True)
    score_width = d["score_useful"].to(edge_ok.dtype).mean(dim=-1, keepdim=True)
    takeout_width = d["takeout_useful"].to(edge_ok.dtype).mean(dim=-1, keepdim=True)
    return _pad_edge_scalars(
        score_align * edge_ok,
        takeout_align * edge_ok,
        score_incoming_open * edge_ok,
        takeout_incoming_open * edge_ok,
        score_goal_open * edge_ok,
        takeout_goal_open * edge_ok,
        score_width * edge_ok,
        takeout_width * edge_ok,
    )


def _goal_contact_usefulness_2hop_edge_scalars(node_coords, node_feats, node_mask, c=None):
    """
    Four composite usefulness channels:
      score contact usefulness, takeout contact usefulness,
      two-hop score usefulness, two-hop takeout usefulness.
    """
    B, N, _ = node_coords.shape
    device = node_coords.device
    dtype = node_coords.dtype
    is_live = (node_feats[:, :, 3] > 0.5) & node_mask
    release_sources, _ = _source_landmark_masks(node_coords, node_feats, node_mask)
    valid_source = is_live | release_sources
    valid_target = is_live

    dirs = _contact_outgoing_directions(device, dtype)
    K = dirs.shape[0]
    source = node_coords.unsqueeze(2).unsqueeze(3)
    target = node_coords.unsqueeze(1).unsqueeze(3)
    out_dirs = dirs.view(1, 1, 1, K, 2)
    contacts = target - INFLATED_STONE_RADIUS_NORM * out_dirs

    source_to_target = source - target
    source_facing = (((contacts - target) * source_to_target).sum(dim=-1) > 0.0)

    source_exp = source.expand(-1, -1, N, K, -1)
    contacts_exp = contacts.expand(-1, N, -1, -1, -1)
    straight_clear = _segment_clearance_edges(node_coords, node_feats, node_mask, source_exp, contacts_exp)
    curl_clear = _curl_clearance_edges(node_coords, node_feats, node_mask, source_exp, contacts_exp)
    release_edge = release_sources.view(B, N, 1, 1)
    incoming_clear = torch.where(release_edge, curl_clear, straight_clear)
    incoming_open = _norm_positive_clearance(incoming_clear)

    src_idx = torch.arange(N, device=device).view(1, N, 1, 1)
    tgt_idx = torch.arange(N, device=device).view(1, 1, N, 1)
    edge_valid = (
        valid_source.view(B, N, 1, 1)
        & valid_target.view(B, 1, N, 1)
        & (src_idx != tgt_idx)
        & source_facing
    )
    reachable = edge_valid & (incoming_clear > 0.0)

    button = node_coords.new_tensor([BUTTON_X, BUTTON_Y]).view(1, 1, 1, 1, 2)
    target_exp = target.expand(-1, N, -1, K, -1)
    out_dirs_exp = out_dirs.expand(B, N, N, -1, -1)

    to_button = button - target
    to_button = to_button / torch.norm(to_button, dim=-1, keepdim=True).clamp(min=1e-6)
    outward = target - button
    outward = outward / torch.norm(outward, dim=-1, keepdim=True).clamp(min=1e-6)
    score_align = torch.clamp((out_dirs * to_button).sum(dim=-1), min=0.0).expand(B, N, N, K)
    takeout_align = torch.clamp((out_dirs * outward).sum(dim=-1), min=0.0).expand(B, N, N, K)

    score_goal = _goal_points(target_exp, out_dirs_exp, "score")
    opp_dist = _opponent_button_distances(node_coords, node_feats, node_mask).view(B, 1, N, 1).expand(-1, N, -1, K)
    takeout_goal = _goal_points(target_exp, out_dirs_exp, "takeout", opp_dist)
    score_open = _norm_positive_clearance(
        _segment_clearance_edges(node_coords, node_feats, node_mask, target_exp, score_goal)
    )
    takeout_open = _norm_positive_clearance(
        _segment_clearance_edges(node_coords, node_feats, node_mask, target_exp, takeout_goal)
    )

    score_use = score_align * incoming_open * score_open
    takeout_use = takeout_align * incoming_open * takeout_open
    score_valid = reachable & (score_align > 0.0) & (score_open > 0.0)
    takeout_valid = reachable & (takeout_align > 0.0) & (takeout_open > 0.0)
    score_contact_use = _normalized_logsumexp(score_use, score_valid, dim=-1)
    takeout_contact_use = _normalized_logsumexp(takeout_use, takeout_valid, dim=-1)

    def two_hop(goal, align_kind):
        hit, hit_idx, hit_point, l_coords, second_out, corridor_clear = _first_intersection_on_live_stone(
            node_coords, node_feats, node_mask, target_exp, goal
        )
        first_transfer_open = _norm_positive_clearance(corridor_clear)

        second_contact = l_coords - INFLATED_STONE_RADIUS_NORM * second_out
        second_contact_open = _norm_positive_clearance(
            _segment_clearance_edges(
                node_coords,
                node_feats,
                node_mask,
                target_exp,
                second_contact,
                exclude_src=True,
                exclude_tgt=True,
                exclude_extra_idx=hit_idx,
            )
        )

        if align_kind == "score":
            l_goal = _goal_points(l_coords, second_out, "score")
            l_to_button = button - l_coords
            l_to_button = l_to_button / torch.norm(l_to_button, dim=-1, keepdim=True).clamp(min=1e-6)
            second_align = torch.clamp((second_out * l_to_button).sum(dim=-1), min=0.0)
        else:
            opp_all = _opponent_button_distances(node_coords, node_feats, node_mask)
            l_opp = torch.gather(
                opp_all.view(B, 1, 1, 1, N).expand(-1, N, N, K, -1),
                4,
                hit_idx.unsqueeze(-1),
            ).squeeze(-1)
            l_goal = _goal_points(l_coords, second_out, "takeout", l_opp)
            l_outward = l_coords - button
            l_outward = l_outward / torch.norm(l_outward, dim=-1, keepdim=True).clamp(min=1e-6)
            second_align = torch.clamp((second_out * l_outward).sum(dim=-1), min=0.0)

        second_goal_open = _norm_positive_clearance(
            _segment_clearance_edges(
                node_coords,
                node_feats,
                node_mask,
                l_coords,
                l_goal,
                exclude_src=True,
                exclude_tgt=True,
                exclude_extra_idx=hit_idx,
            )
        )
        path_use = incoming_open * first_transfer_open * second_contact_open * second_align * second_goal_open
        valid = (
            reachable
            & hit
            & (corridor_clear > 0.0)
            & (second_contact_open > 0.0)
            & (second_align > 0.0)
            & (second_goal_open > 0.0)
        )
        return _normalized_logsumexp(path_use, valid, dim=-1)

    two_hop_score_use = two_hop(score_goal, "score")
    two_hop_takeout_use = two_hop(takeout_goal, "takeout")

    edge_ok = (
        valid_source.view(B, N, 1, 1)
        & valid_target.view(B, 1, N, 1)
        & (src_idx != tgt_idx)
    ).to(dtype)
    return _pad_edge_scalars(
        score_contact_use * edge_ok,
        takeout_contact_use * edge_ok,
        two_hop_score_use * edge_ok,
        two_hop_takeout_use * edge_ok,
    )


def _compute_source_button_region_spans(node_coords, node_feats, node_mask, shooter_team):
    """
    Visible angular span of the adaptive button-region circle from each source
    node. The target is the button-centered circle whose radius is the nearest
    opponent stone to the button center, capped by the house radius.
    """
    B, N, _ = node_coords.shape
    device = node_coords.device
    dtype = node_coords.dtype

    button_xy = node_coords.new_tensor([BUTTON_X, BUTTON_Y])
    is_live = (node_feats[:, :, 3] > 0.5) & node_mask
    button_idx, is_button_node = _button_node_indices(node_coords, node_feats, node_mask)
    button_region_radius = _compute_button_region_radii(node_coords, node_feats, node_mask, shooter_team)

    spans = torch.zeros(B, N, device=device, dtype=dtype)
    neg_inf_col = torch.full((B, 1), float("-inf"), device=device, dtype=dtype)
    pos_inf = torch.full((B, N), float("inf"), device=device, dtype=dtype)
    neg_inf = torch.full((B, N), float("-inf"), device=device, dtype=dtype)

    for source_idx in range(N):
        source_valid = node_mask[:, source_idx]
        if not bool(source_valid.any()):
            continue

        viewpoint = node_coords[:, source_idx:source_idx + 1, :]
        delta = node_coords - viewpoint  # (B, N, 2)
        dist = torch.norm(delta, dim=-1).clamp(min=STONE_RADIUS_NORM + 1e-6)
        angles = torch.atan2(delta[:, :, 0], delta[:, :, 1])
        half_angles = torch.asin((STONE_RADIUS_NORM / dist).clamp(max=1.0 - 1e-6))

        to_button = button_xy.view(1, 2) - node_coords[:, source_idx, :]
        button_dist = torch.norm(to_button, dim=-1).clamp(min=1e-6)
        button_angle = torch.atan2(to_button[:, 0], to_button[:, 1]).unsqueeze(1)
        target_half = torch.asin(
            (button_region_radius / button_dist).clamp(max=1.0 - 1e-6)
        ).unsqueeze(1)

        rel = _wrap_angle(angles - button_angle)
        low = -target_half.expand_as(rel)
        high = target_half.expand_as(rel)
        starts = torch.maximum(rel - half_angles, low)
        ends = torch.minimum(rel + half_angles, high)

        closer = dist < (button_dist.unsqueeze(1) - 1e-8)
        blockers = closer & is_live
        blockers[:, source_idx] = False
        valid = blockers & (ends > starts)

        masked_starts = torch.where(valid, starts, pos_inf)
        masked_ends = torch.where(valid, ends, neg_inf)
        order = torch.argsort(masked_starts, dim=1)
        sorted_starts = torch.gather(masked_starts, 1, order)
        sorted_ends = torch.gather(masked_ends, 1, order)

        prev_max_end = torch.cummax(sorted_ends, dim=1).values
        prev_max_end_exclusive = torch.cat([neg_inf_col, prev_max_end[:, :-1]], dim=1)
        newly_covered = (
            sorted_ends - torch.maximum(sorted_starts, prev_max_end_exclusive)
        ).clamp(min=0.0)
        covered = newly_covered.sum(dim=1)

        source_is_button = is_button_node[:, source_idx]
        target_span = (2.0 * target_half.squeeze(1) - covered).clamp(min=0.0)
        spans[:, source_idx] = torch.where(
            source_valid & ~source_is_button,
            target_span,
            torch.zeros_like(target_span),
        )

    return spans


def _compute_pairwise_thrower_masked_button_region_spans(node_coords, node_feats, node_mask, shooter_team):
    """
    Pairwise span variant requested by the user:
    - start from exact pairwise unoccluded spans,
    - fully occlude directions that point toward the thrower (-x direction),
    - for the button target node, replace the edge scalar with a source-specific
      visible span to an adaptive button-region circle.
    """
    B, N, _ = node_coords.shape
    spans = _compute_pairwise_unoccluded_angular_spans(node_coords, node_feats, node_mask).squeeze(-1)

    target_minus_source = node_coords.unsqueeze(1) - node_coords.unsqueeze(2)  # (B, source, target, 2)
    thrower_direction_mask = target_minus_source[..., 0] < 0.0
    spans = torch.where(thrower_direction_mask, torch.zeros_like(spans), spans)

    button_region_spans = _compute_source_button_region_spans(
        node_coords, node_feats, node_mask, shooter_team
    )  # (B, N)
    button_idx, has_button_mask = _button_node_indices(node_coords, node_feats, node_mask)

    for b in range(B):
        idx = int(button_idx[b].item())
        if idx >= 0:
            spans[b, :, idx] = button_region_spans[b]
            spans[b, thrower_direction_mask[b, :, idx], idx] = 0.0

    return spans.unsqueeze(-1)


def _compute_pairwise_thrower_masked_spans(node_coords, node_feats, node_mask):
    """Exact pairwise spans with only the -x thrower-direction mask applied."""
    spans = _compute_pairwise_unoccluded_angular_spans(node_coords, node_feats, node_mask).squeeze(-1)
    target_minus_source = node_coords.unsqueeze(1) - node_coords.unsqueeze(2)
    thrower_direction_mask = target_minus_source[..., 0] < 0.0
    spans = torch.where(thrower_direction_mask, torch.zeros_like(spans), spans)
    return spans.unsqueeze(-1)


def _compute_pairwise_button_region_spans(node_coords, node_feats, node_mask, shooter_team):
    """Exact pairwise spans plus adaptive button-region replacement, without thrower masking."""
    B, N, _ = node_coords.shape
    spans = _compute_pairwise_unoccluded_angular_spans(node_coords, node_feats, node_mask).squeeze(-1)
    button_region_spans = _compute_source_button_region_spans(
        node_coords, node_feats, node_mask, shooter_team
    )
    button_idx, _ = _button_node_indices(node_coords, node_feats, node_mask)
    for b in range(B):
        idx = int(button_idx[b].item())
        if idx >= 0:
            spans[b, :, idx] = button_region_spans[b]
    return spans.unsqueeze(-1)


def _clearance_for_candidate_angles(source_xy, target_xy, blocker_xy, blocker_mask, theta):
    """
    Evaluate straight-line bottleneck clearance for candidate shot angles.

    Args:
        source_xy: (B, 2)
        target_xy: (B, 2)
        blocker_xy: (B, N, 2)
        blocker_mask: (B, N) bool, True for live blocker stones
        theta: (B, K) absolute candidate angles in radians

    Returns:
        kappa: (B, K) bottleneck clearance along segment [source, first target
               hit]. Unreachable candidates receive a finite negative floor.
    """
    B, K = theta.shape
    device = source_xy.device
    dtype = source_xy.dtype

    u = torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1)  # (B, K, 2)

    target_vec = target_xy - source_xy  # (B, 2)
    target_vec_sq = (target_vec * target_vec).sum(dim=-1, keepdim=True)  # (B, 1)
    b_t = (u * target_vec.unsqueeze(1)).sum(dim=-1)  # (B, K)
    d_perp_sq_t = (target_vec_sq - b_t.square()).clamp(min=0.0)
    hit_margin = (INFLATED_STONE_RADIUS_NORM ** 2 - d_perp_sq_t).clamp(min=0.0)
    hits_target = (b_t > 0.0) & (d_perp_sq_t <= INFLATED_STONE_RADIUS_NORM ** 2 + 1e-8)
    lambda_t = torch.where(
        hits_target,
        b_t - torch.sqrt(hit_margin),
        torch.zeros_like(b_t),
    )  # (B, K)

    seg = lambda_t.unsqueeze(-1) * u  # (B, K, 2)
    seg_len_sq = seg.square().sum(dim=-1).clamp(min=1e-12)  # (B, K)

    blocker_rel = blocker_xy.unsqueeze(1) - source_xy.view(B, 1, 1, 2)  # (B, K, N, 2)
    blocker_rel_sq = blocker_rel.square().sum(dim=-1)  # (B, K, N)
    b_i = (blocker_rel * u.unsqueeze(2)).sum(dim=-1)  # (B, K, N)
    d_perp_sq_i = (blocker_rel_sq - b_i.square()).clamp(min=0.0)
    hit_blocker = blocker_mask.unsqueeze(1) & (b_i > 0.0) & (
        d_perp_sq_i <= INFLATED_STONE_RADIUS_NORM ** 2 + 1e-8
    )
    lambda_i = b_i - torch.sqrt(
        (INFLATED_STONE_RADIUS_NORM ** 2 - d_perp_sq_i).clamp(min=0.0)
    )
    blocker_hits_before_target = hit_blocker & (lambda_i <= lambda_t.unsqueeze(-1) + 1e-8)
    feasible = hits_target & ~blocker_hits_before_target.any(dim=-1)

    seg_expanded = seg.unsqueeze(2)  # (B, K, 1, 2)
    proj = (blocker_rel * seg_expanded).sum(dim=-1) / seg_len_sq.unsqueeze(-1)  # (B, K, N)
    proj = proj.clamp(0.0, 1.0)
    closest = source_xy.view(B, 1, 1, 2) + proj.unsqueeze(-1) * seg_expanded  # (B, K, N, 2)
    blocker_dist = torch.norm(blocker_xy.unsqueeze(1) - closest, dim=-1)  # (B, K, N)

    clearance = blocker_dist - INFLATED_STONE_RADIUS_NORM
    masked_clearance = torch.where(
        blocker_mask.unsqueeze(1),
        clearance,
        torch.full_like(clearance, CLEARANCE_CAP_NORM),
    )
    kappa = masked_clearance.min(dim=-1).values.clamp(max=CLEARANCE_CAP_NORM)  # (B, K)
    infeasible_floor = torch.full((B, K), -CLEARANCE_CAP_NORM, device=device, dtype=dtype)
    kappa = torch.where(hits_target, kappa, infeasible_floor)
    # If the exact first-hit ordering fails, keep the candidate finite but
    # non-positive so unreachable pairs do not inject infinities into training.
    kappa = torch.where(feasible, kappa, torch.minimum(kappa, -1e-4 * torch.ones_like(kappa)))
    return kappa


def _compute_pairwise_line_clearance(node_coords, node_feats, node_mask):
    """
    Approximate the straight-line max-clearance feature Phi_line(source, target),
    with exact first-hit feasibility at each candidate angle.

    For each ordered source/target pair we:
    1. form the target's admissible angular interval at the source using the
       inflated target disk,
    2. intersect blocker-forbidden angular intervals with that target interval,
    3. use interval boundaries and interval midpoints as candidate angles,
    4. evaluate bottleneck clearance along the segment ending at first target hit,
       then take the best candidate value.

    Returns:
        clearances: (B, N, N, 1)
    """
    B, N, _ = node_coords.shape
    device = node_coords.device
    dtype = node_coords.dtype

    is_live = (node_feats[:, :, 3] > 0.5) & node_mask
    clearances = torch.zeros(B, N, N, device=device, dtype=dtype)
    inf = torch.full((B, N), float("inf"), device=device, dtype=dtype)
    for source_idx in range(N):
        source_valid = node_mask[:, source_idx]  # (B,)
        if not bool(source_valid.any()):
            continue

        source_xy = node_coords[:, source_idx, :]  # (B, 2)
        source_to_nodes = node_coords - source_xy.unsqueeze(1)  # (B, N, 2)
        node_dist = torch.norm(source_to_nodes, dim=-1)  # (B, N)
        node_angle = torch.atan2(source_to_nodes[:, :, 1], source_to_nodes[:, :, 0])  # (B, N)

        for target_idx in range(N):
            target_live = is_live[:, target_idx] & source_valid
            if source_idx == target_idx or not bool(target_live.any()):
                continue

            target_dist = node_dist[:, target_idx]  # (B,)
            target_angle = node_angle[:, target_idx]  # (B,)
            well_defined = target_live & (target_dist > INFLATED_STONE_RADIUS_NORM + 1e-6)
            if not bool(well_defined.any()):
                continue

            target_delta = torch.asin(
                (INFLATED_STONE_RADIUS_NORM / target_dist.clamp(min=INFLATED_STONE_RADIUS_NORM + 1e-6))
                .clamp(max=1.0 - 1e-6)
            )  # (B,)
            low = -target_delta
            high = target_delta

            blocker_mask = is_live.clone()
            blocker_mask[:, source_idx] = False
            blocker_mask[:, target_idx] = False

            rel_angle = _wrap_angle(node_angle - target_angle.unsqueeze(1))  # (B, N)
            blocker_dist = node_dist
            blocker_beta = torch.asin(
                (INFLATED_STONE_RADIUS_NORM / blocker_dist.clamp(min=INFLATED_STONE_RADIUS_NORM + 1e-6))
                .clamp(max=1.0 - 1e-6)
            )
            interval_start = torch.maximum(rel_angle - blocker_beta, low.unsqueeze(1))
            interval_end = torch.minimum(rel_angle + blocker_beta, high.unsqueeze(1))
            interval_valid = blocker_mask & (interval_end > interval_start)

            boundaries = torch.cat([
                low.unsqueeze(1),
                high.unsqueeze(1),
                torch.where(interval_valid, interval_start, inf),
                torch.where(interval_valid, interval_end, inf),
            ], dim=1)  # (B, 2 + 2N)
            boundaries, _ = torch.sort(boundaries, dim=1)

            left = boundaries[:, :-1]
            right = boundaries[:, 1:]
            segment_valid = well_defined.unsqueeze(1) & torch.isfinite(left) & torch.isfinite(right) & ((right - left) > 1e-7)
            midpoint = 0.5 * (left + right)

            endpoint_eps = torch.minimum(
                torch.full_like(left, ANGLE_EPS),
                0.25 * (right - left).clamp(min=0.0),
            )
            edge_eps = torch.minimum(
                torch.full_like(target_delta.unsqueeze(1), ANGLE_EPS),
                0.5 * target_delta.unsqueeze(1),
            ).squeeze(1)

            candidate_rel = torch.cat([
                torch.zeros(B, 1, device=device, dtype=dtype),
                (low + edge_eps).unsqueeze(1),
                (high - edge_eps).unsqueeze(1),
                midpoint,
                left + endpoint_eps,
                right - endpoint_eps,
            ], dim=1)
            candidate_valid = torch.cat([
                well_defined.unsqueeze(1),
                well_defined.unsqueeze(1),
                well_defined.unsqueeze(1),
                segment_valid,
                segment_valid,
                segment_valid,
            ], dim=1)

            absolute_theta = target_angle.unsqueeze(1) + candidate_rel
            kappa = _clearance_for_candidate_angles(
                source_xy=source_xy,
                target_xy=node_coords[:, target_idx, :],
                blocker_xy=node_coords,
                blocker_mask=blocker_mask,
                theta=absolute_theta,
            )
            kappa = torch.where(candidate_valid, kappa, torch.full_like(kappa, -CLEARANCE_CAP_NORM))
            best_kappa = kappa.max(dim=1).values
            best_kappa = torch.where(
                torch.isfinite(best_kappa),
                best_kappa,
                torch.full_like(best_kappa, -CLEARANCE_CAP_NORM),
            )
            clearances[:, source_idx, target_idx] = torch.where(
                well_defined,
                best_kappa,
                torch.zeros_like(best_kappa),
            )

    return clearances.unsqueeze(-1)


def build_graph_batch(x, device):
    """
    Build batched graph from stone positions.

    Args:
        x: (B, 24) normalized stone positions

    Returns:
        node_feats: (B, max_nodes, NODE_FEAT_DIM) padded node features
        node_coords: (B, max_nodes, 2) padded node coordinates
        edge_feats: (B, max_nodes, max_nodes, edge_feat_dim) edge features
        node_mask: (B, max_nodes) bool mask for valid nodes
        n_nodes: (B,) number of real nodes per sample
    """
    B = x.size(0)
    stones = x.view(B, NUM_STONES, 2)

    # Detect live stones: not at (0,0) and not at (1,1)
    is_live = ((stones.sum(dim=-1) > 0.001) &
               (stones.max(dim=-1).values < 0.999))  # (B, 12) bool

    landmarks = _get_active_landmarks(device=device, dtype=stones.dtype)
    n_landmarks = landmarks.shape[0]

    # Max possible nodes = 12 stones + active landmark nodes
    max_nodes = NUM_STONES + n_landmarks

    button_xy = landmarks[0]  # (2,)

    # Pre-allocate
    node_feats = torch.zeros(B, max_nodes, NODE_FEAT_DIM, device=device)
    node_coords = torch.zeros(B, max_nodes, 2, device=device)
    node_mask = torch.zeros(B, max_nodes, dtype=torch.bool, device=device)
    n_nodes_list = []

    # Team indicator for stones: 0 for stones 0-5, 1 for stones 6-11
    team_ids = torch.zeros(NUM_STONES, device=device)
    team_ids[6:] = 1.0

    for b in range(B):
        live_mask = is_live[b]  # (12,)
        live_indices = live_mask.nonzero(as_tuple=True)[0]  # indices of live stones
        n_live = live_indices.shape[0]
        n_total = n_live + n_landmarks

        # Fill stone nodes
        for i, si in enumerate(live_indices):
            node_coords[b, i] = stones[b, si]
            node_feats[b, i, 0] = stones[b, si, 0]  # x
            node_feats[b, i, 1] = stones[b, si, 1]  # y
            node_feats[b, i, 2] = team_ids[si]        # team
            node_feats[b, i, 3] = 1.0                 # is_live
            node_feats[b, i, 4] = 0.0                 # is_landmark

        # Fill landmark nodes
        for j in range(n_landmarks):
            idx = n_live + j
            node_coords[b, idx] = landmarks[j]
            node_feats[b, idx, 0] = landmarks[j, 0]
            node_feats[b, idx, 1] = landmarks[j, 1]
            node_feats[b, idx, 2] = 0.0   # no team
            node_feats[b, idx, 3] = 0.0   # not live
            node_feats[b, idx, 4] = 1.0   # is landmark

        node_mask[b, :n_total] = True
        n_nodes_list.append(n_total)

    node_feature_mode = _resolve_node_feature_mode()
    if node_feature_mode == "button_visible_span":
        node_feats[:, :, 5:6] = _compute_unoccluded_angular_spans(node_coords, node_feats, node_mask)
    elif node_feature_mode == "release_reach_times_takeout":
        release_reach = _compute_release_reach_spans(node_coords, node_feats, node_mask)
        takeout = _compute_takeoutability_spans(node_coords, node_feats, node_mask)
        node_feats[:, :, 5:6] = release_reach * takeout
    elif node_feature_mode == "beats_nearest_opponent_to_button":
        node_feats[:, :, 5:6] = _compute_beats_nearest_opponent_to_button(node_coords, node_feats, node_mask)

    n_nodes = torch.tensor(n_nodes_list, device=device)
    return node_feats, node_coords, node_mask, n_nodes


def compute_edge_features(node_coords, node_feats, node_mask, c=None):
    """
    Compute edge features for fully connected graph.

    Edge features:
    - dx, dy: relative displacement (2)
    - distance (1)
    - same_team indicator (1)
    - edge scalar(s): up to five geometry channels, padded to width 5

    Returns: (B, max_nodes, max_nodes, 9)
    """
    B, N, _ = node_coords.shape
    device = node_coords.device

    # Relative displacement
    dx = node_coords.unsqueeze(2) - node_coords.unsqueeze(1)  # (B, N, N, 2)
    dist = torch.norm(dx, dim=-1, keepdim=True).clamp(min=1e-8)  # (B, N, N, 1)

    # Same team: both live stones and same team
    team = node_feats[:, :, 2]  # (B, N)
    is_live = node_feats[:, :, 3]  # (B, N)

    same_team = (team.unsqueeze(2) == team.unsqueeze(1)).float()  # (B, N, N)
    # Only meaningful when both are live stones
    both_live = (is_live.unsqueeze(2) * is_live.unsqueeze(1))  # (B, N, N)
    same_team = (same_team * both_live).unsqueeze(-1)  # (B, N, N, 1)

    shooter_team = _extract_shooter_team(c, node_coords)
    edge_scalar_mode = _resolve_edge_scalar_mode()
    if edge_scalar_mode == "thrower_masked_button_region_span":
        primary = _compute_pairwise_thrower_masked_button_region_spans(
            node_coords, node_feats, node_mask, shooter_team
        )
        edge_scalars = _pad_edge_scalars(primary)
    elif edge_scalar_mode == "thrower_masked_pairwise_span":
        primary = _compute_pairwise_thrower_masked_spans(node_coords, node_feats, node_mask)
        edge_scalars = _pad_edge_scalars(primary)
    elif edge_scalar_mode == "button_region_pairwise_span":
        primary = _compute_pairwise_button_region_spans(
            node_coords, node_feats, node_mask, shooter_team
        )
        edge_scalars = _pad_edge_scalars(primary)
    elif edge_scalar_mode == "button_visible_plus_thrower_masked_span":
        button_visible = _compute_unoccluded_angular_spans(node_coords, node_feats, node_mask)
        button_visible = button_visible.unsqueeze(1).expand(-1, N, -1, -1)
        reachability = _compute_pairwise_thrower_masked_button_region_spans(
            node_coords, node_feats, node_mask, shooter_team
        )
        edge_scalars = _pad_edge_scalars(button_visible, reachability)
    elif edge_scalar_mode == "button_visible_plus_release_reach_span":
        button_visible = _compute_unoccluded_angular_spans(node_coords, node_feats, node_mask)
        button_visible = button_visible.unsqueeze(1).expand(-1, N, -1, -1)
        release_reach = _compute_release_reach_spans(node_coords, node_feats, node_mask)
        release_reach = release_reach.unsqueeze(1).expand(-1, N, -1, -1)
        edge_scalars = _pad_edge_scalars(button_visible, release_reach)
    elif edge_scalar_mode == "button_visible_plus_release_reach_with_product":
        button_visible = _compute_unoccluded_angular_spans(node_coords, node_feats, node_mask)
        button_visible = button_visible.unsqueeze(1).expand(-1, N, -1, -1)
        release_reach = _compute_release_reach_spans(node_coords, node_feats, node_mask)
        release_reach = release_reach.unsqueeze(1).expand(-1, N, -1, -1)
        product = button_visible * release_reach
        edge_scalars = _pad_edge_scalars(button_visible, release_reach, product)
    elif edge_scalar_mode == "button_visible_plus_source_reach_takeout_edges_with_product":
        button_visible = _compute_unoccluded_angular_spans(node_coords, node_feats, node_mask)
        button_visible = button_visible.unsqueeze(1).expand(-1, N, -1, -1)
        pairwise = _compute_pairwise_unoccluded_angular_spans(node_coords, node_feats, node_mask)
        release_sources, takeout_sources = _source_landmark_masks(node_coords, node_feats, node_mask)
        release_edge_reach = pairwise * release_sources.view(B, N, 1, 1).float()
        takeout_edge_reach = pairwise * takeout_sources.view(B, N, 1, 1).float()
        edge_scalars = _pad_edge_scalars(
            button_visible,
            release_edge_reach,
            takeout_edge_reach,
            button_visible * release_edge_reach,
            button_visible * takeout_edge_reach,
        )
    elif edge_scalar_mode == "button_visible_plus_curl_arc_reach_with_outgoing":
        scorability = _compute_unoccluded_angular_spans(node_coords, node_feats, node_mask)
        scorability = scorability.unsqueeze(1).expand(-1, N, -1, -1)
        clearance, feasible, diversity, score_out, takeout_out = _compute_curl_arc_reachability_and_outgoing(
            node_coords, node_feats, node_mask, c=c
        )
        pairwise_reach = _compute_pairwise_unoccluded_angular_spans(node_coords, node_feats, node_mask)
        release_sources, _ = _source_landmark_masks(node_coords, node_feats, node_mask)
        release_edge = release_sources.view(B, N, 1, 1)
        straight_score, straight_takeout = _compute_pairwise_outgoing_compatibility(
            node_coords, node_feats, node_mask, c=c
        )
        curl_reach = feasible * diversity
        reach = torch.where(release_edge, curl_reach, pairwise_reach)
        score_out = torch.where(release_edge, score_out, straight_score)
        takeout_out = torch.where(release_edge, takeout_out, straight_takeout)
        edge_scalars = _pad_edge_scalars(
            scorability,
            reach,
            score_out,
            takeout_out,
            reach * torch.maximum(score_out, takeout_out),
        )
    elif edge_scalar_mode == "button_visible_plus_curl_arc_reach_clean":
        edge_scalars = _compute_clean_curl_reach_edge_scalars(
            node_coords, node_feats, node_mask, c=c, use_minkowski_straight=False
        )
    elif edge_scalar_mode == "button_visible_plus_curl_arc_minkowski_straight_reach_clean":
        edge_scalars = _compute_clean_curl_reach_edge_scalars(
            node_coords, node_feats, node_mask, c=c, use_minkowski_straight=True
        )
    elif edge_scalar_mode == "button_visible_plus_contact_arc_full":
        edge_scalars = _compute_contact_arc_edge_scalars(
            node_coords, node_feats, node_mask, c=c, stack="full"
        )
    elif edge_scalar_mode == "button_visible_plus_contact_arc_minimal":
        edge_scalars = _compute_contact_arc_edge_scalars(
            node_coords, node_feats, node_mask, c=c, stack="minimal"
        )
    elif edge_scalar_mode == "button_visible_plus_contact_arc_score_product":
        edge_scalars = _compute_contact_arc_edge_scalars(
            node_coords, node_feats, node_mask, c=c, stack="score_product"
        )
    elif edge_scalar_mode == "button_visible_plus_contact_arc_takeout_product":
        edge_scalars = _compute_contact_arc_edge_scalars(
            node_coords, node_feats, node_mask, c=c, stack="takeout_product"
        )
    elif edge_scalar_mode == "button_visible_plus_contact_arc_old_shape":
        edge_scalars = _compute_contact_arc_edge_scalars(
            node_coords, node_feats, node_mask, c=c, stack="old_shape"
        )
    elif edge_scalar_mode == "goal_contact_oldbest_plus_chain_2hop":
        edge_scalars = _goal_contact_oldbest_plus_chain_2hop_edge_scalars(
            node_coords, node_feats, node_mask, c=c
        )
    elif edge_scalar_mode == "goal_contact_chain_outside_2hop":
        edge_scalars = _goal_contact_chain_2hop_edge_scalars(
            node_coords, node_feats, node_mask, c=c
        )
    elif edge_scalar_mode == "goal_contact_chain_2hop":
        edge_scalars = _goal_contact_chain_2hop_edge_scalars(
            node_coords, node_feats, node_mask, c=c
        )
    elif edge_scalar_mode == "goal_contact_chain_1hop":
        edge_scalars = _goal_contact_chain_1hop_edge_scalars(
            node_coords, node_feats, node_mask, c=c
        )
    elif edge_scalar_mode == "goal_contact_chain_1hop_topk":
        edge_scalars = _goal_contact_chain_1hop_topk_edge_scalars(
            node_coords, node_feats, node_mask, c=c
        )
    elif edge_scalar_mode == "goal_contact_1hop_terms":
        edge_scalars = _goal_contact_1hop_terms_edge_scalars(
            node_coords, node_feats, node_mask, c=c
        )
    elif edge_scalar_mode == "goal_contact_usefulness_2hop":
        edge_scalars = _goal_contact_usefulness_2hop_edge_scalars(
            node_coords, node_feats, node_mask, c=c
        )
    elif edge_scalar_mode == "button_visible_plus_contact_arc_full":
        edge_scalars = _compute_contact_arc_edge_scalars(
            node_coords, node_feats, node_mask, c=c, stack="full"
        )
    elif edge_scalar_mode == "button_visible_plus_contact_arc_minimal":
        edge_scalars = _compute_contact_arc_edge_scalars(
            node_coords, node_feats, node_mask, c=c, stack="minimal"
        )
    elif edge_scalar_mode == "button_visible_plus_contact_arc_score_product":
        edge_scalars = _compute_contact_arc_edge_scalars(
            node_coords, node_feats, node_mask, c=c, stack="score_product"
        )
    elif edge_scalar_mode == "button_visible_plus_contact_arc_takeout_product":
        edge_scalars = _compute_contact_arc_edge_scalars(
            node_coords, node_feats, node_mask, c=c, stack="takeout_product"
        )
    elif edge_scalar_mode == "button_visible_plus_contact_arc_old_shape":
        edge_scalars = _compute_contact_arc_edge_scalars(
            node_coords, node_feats, node_mask, c=c, stack="old_shape"
        )
    elif edge_scalar_mode == "goal_contact_oldbest_plus_chain_2hop":
        edge_scalars = _goal_contact_oldbest_plus_chain_2hop_edge_scalars(
            node_coords, node_feats, node_mask, c=c
        )
    elif edge_scalar_mode == "goal_contact_chain_outside_2hop":
        edge_scalars = _goal_contact_chain_2hop_edge_scalars(
            node_coords, node_feats, node_mask, c=c
        )
    elif edge_scalar_mode == "goal_contact_chain_2hop":
        edge_scalars = _goal_contact_chain_2hop_edge_scalars(
            node_coords, node_feats, node_mask, c=c
        )
    elif edge_scalar_mode == "goal_contact_chain_1hop":
        edge_scalars = _goal_contact_chain_1hop_edge_scalars(
            node_coords, node_feats, node_mask, c=c
        )
    elif edge_scalar_mode == "goal_contact_chain_1hop_topk":
        edge_scalars = _goal_contact_chain_1hop_topk_edge_scalars(
            node_coords, node_feats, node_mask, c=c
        )
    elif edge_scalar_mode == "goal_contact_1hop_terms":
        edge_scalars = _goal_contact_1hop_terms_edge_scalars(
            node_coords, node_feats, node_mask, c=c
        )
    elif edge_scalar_mode == "goal_contact_usefulness_2hop":
        edge_scalars = _goal_contact_usefulness_2hop_edge_scalars(
            node_coords, node_feats, node_mask, c=c
        )
    elif edge_scalar_mode == "button_visible_plus_curl_arc_reach_clean":
        edge_scalars = _compute_clean_curl_reach_edge_scalars(
            node_coords, node_feats, node_mask, c=c, use_minkowski_straight=False
        )
    elif edge_scalar_mode == "button_visible_plus_curl_arc_minkowski_straight_reach_clean":
        edge_scalars = _compute_clean_curl_reach_edge_scalars(
            node_coords, node_feats, node_mask, c=c, use_minkowski_straight=True
        )
    elif edge_scalar_mode == "button_visible_plus_contact_arc_full":
        edge_scalars = _compute_contact_arc_edge_scalars(
            node_coords, node_feats, node_mask, c=c, stack="full"
        )
    elif edge_scalar_mode == "button_visible_plus_contact_arc_minimal":
        edge_scalars = _compute_contact_arc_edge_scalars(
            node_coords, node_feats, node_mask, c=c, stack="minimal"
        )
    elif edge_scalar_mode == "button_visible_plus_contact_arc_score_product":
        edge_scalars = _compute_contact_arc_edge_scalars(
            node_coords, node_feats, node_mask, c=c, stack="score_product"
        )
    elif edge_scalar_mode == "button_visible_plus_contact_arc_takeout_product":
        edge_scalars = _compute_contact_arc_edge_scalars(
            node_coords, node_feats, node_mask, c=c, stack="takeout_product"
        )
    elif edge_scalar_mode == "button_visible_plus_contact_arc_old_shape":
        edge_scalars = _compute_contact_arc_edge_scalars(
            node_coords, node_feats, node_mask, c=c, stack="old_shape"
        )
    elif edge_scalar_mode == "goal_contact_oldbest_plus_chain_2hop":
        edge_scalars = _goal_contact_oldbest_plus_chain_2hop_edge_scalars(
            node_coords, node_feats, node_mask, c=c
        )
    elif edge_scalar_mode == "goal_contact_chain_outside_2hop":
        edge_scalars = _goal_contact_chain_2hop_edge_scalars(
            node_coords, node_feats, node_mask, c=c
        )
    elif edge_scalar_mode == "goal_contact_chain_2hop":
        edge_scalars = _goal_contact_chain_2hop_edge_scalars(
            node_coords, node_feats, node_mask, c=c
        )
    elif edge_scalar_mode == "goal_contact_chain_1hop":
        edge_scalars = _goal_contact_chain_1hop_edge_scalars(
            node_coords, node_feats, node_mask, c=c
        )
    elif edge_scalar_mode == "goal_contact_chain_1hop_topk":
        edge_scalars = _goal_contact_chain_1hop_topk_edge_scalars(
            node_coords, node_feats, node_mask, c=c
        )
    elif edge_scalar_mode == "goal_contact_1hop_terms":
        edge_scalars = _goal_contact_1hop_terms_edge_scalars(
            node_coords, node_feats, node_mask, c=c
        )
    elif edge_scalar_mode == "goal_contact_usefulness_2hop":
        edge_scalars = _goal_contact_usefulness_2hop_edge_scalars(
            node_coords, node_feats, node_mask, c=c
        )
    elif edge_scalar_mode == "contact_geometry_release_products":
        edge_scalars = _contact_geometry_release_product_edge_scalars(
            node_coords, node_feats, node_mask
        )
    elif edge_scalar_mode == "contact_geometry_release_binary_reach_products":
        edge_scalars = _contact_geometry_release_product_edge_scalars(
            node_coords, node_feats, node_mask, binary_reach=True
        )
    elif edge_scalar_mode == "contact_geometry_release_products_plus_clearance":
        edge_scalars = _contact_geometry_release_product_edge_scalars(
            node_coords, node_feats, node_mask, include_clearance=True
        )
    elif edge_scalar_mode == "contact_geometry_release_binary_reach_products_plus_clearance":
        edge_scalars = _contact_geometry_release_product_edge_scalars(
            node_coords, node_feats, node_mask, include_clearance=True, binary_reach=True
        )
    elif edge_scalar_mode == "contact_geometry_release_products_plus_clearance_plus_onehop":
        edge_scalars = _contact_geometry_release_product_edge_scalars(
            node_coords, node_feats, node_mask, include_clearance=True, include_onehop=True
        )
    elif edge_scalar_mode == "contact_geometry_release_binary_reach_products_plus_onehop_allgoals":
        edge_scalars = _contact_geometry_release_product_edge_scalars(
            node_coords, node_feats, node_mask, include_clearance=False, include_onehop=True, binary_reach=True
        )
    elif edge_scalar_mode == "contact_geometry_release_binary_reach_products_plus_onehop_no_takeout_center":
        edge_scalars = _contact_geometry_release_product_edge_scalars(
            node_coords,
            node_feats,
            node_mask,
            include_clearance=False,
            include_onehop=True,
            include_kinds=("score", "semi"),
            binary_reach=True,
        )
    elif edge_scalar_mode == "contact_geometry_release_binary_reach_products_plus_onehop_no_center":
        edge_scalars = _contact_geometry_release_product_edge_scalars(
            node_coords,
            node_feats,
            node_mask,
            include_clearance=False,
            include_onehop=True,
            include_kinds=("score", "takeout", "semi"),
            binary_reach=True,
        )
    elif edge_scalar_mode == "contact_geometry_release_plus_stonepairs_full21":
        edge_scalars = _contact_geometry_release_plus_stonepairs_edge_scalars(
            node_coords, node_feats, node_mask, stone_stack="full21"
        )
    elif edge_scalar_mode == "contact_geometry_release_plus_stonepairs_products13":
        edge_scalars = _contact_geometry_release_plus_stonepairs_edge_scalars(
            node_coords, node_feats, node_mask, stone_stack="products13"
        )
    elif edge_scalar_mode == "contact_geometry_release_plus_stonepairs_basic9":
        edge_scalars = _contact_geometry_release_plus_stonepairs_edge_scalars(
            node_coords, node_feats, node_mask, stone_stack="basic9"
        )
    elif edge_scalar_mode == "contact_geometry_release_plus_stonepairs_rac5":
        edge_scalars = _contact_geometry_release_plus_stonepairs_edge_scalars(
            node_coords, node_feats, node_mask, stone_stack="rac5"
        )
    elif edge_scalar_mode == "contact_geometry_release_plus_stonepairs_score6":
        edge_scalars = _contact_geometry_release_plus_stonepairs_edge_scalars(
            node_coords, node_feats, node_mask, stone_stack="score6"
        )
    elif edge_scalar_mode == "contact_geometry_release_plus_stonepairs_scoresemi11":
        edge_scalars = _contact_geometry_release_plus_stonepairs_edge_scalars(
            node_coords, node_feats, node_mask, stone_stack="scoresemi11"
        )
    elif edge_scalar_mode == "contact_geometry_release_binary_plus_stonepairs_full21":
        edge_scalars = _contact_geometry_release_plus_stonepairs_edge_scalars(
            node_coords, node_feats, node_mask, stone_stack="full21", binary_reach=True
        )
    elif edge_scalar_mode == "contact_geometry_release_binary_plus_stonepairs_products13":
        edge_scalars = _contact_geometry_release_plus_stonepairs_edge_scalars(
            node_coords, node_feats, node_mask, stone_stack="products13", binary_reach=True
        )
    elif edge_scalar_mode == "contact_geometry_release_binary_plus_stonepairs_basic9":
        edge_scalars = _contact_geometry_release_plus_stonepairs_edge_scalars(
            node_coords, node_feats, node_mask, stone_stack="basic9", binary_reach=True
        )
    elif edge_scalar_mode == "contact_geometry_release_binary_plus_stonepairs_rac5":
        edge_scalars = _contact_geometry_release_plus_stonepairs_edge_scalars(
            node_coords, node_feats, node_mask, stone_stack="rac5", binary_reach=True
        )
    elif edge_scalar_mode == "contact_geometry_release_binary_plus_stonepairs_scoresemi11":
        edge_scalars = _contact_geometry_release_plus_stonepairs_edge_scalars(
            node_coords, node_feats, node_mask, stone_stack="scoresemi11", binary_reach=True
        )
    elif edge_scalar_mode == "contact_geometry_release_binary_plus_stonepairs_hop1_full21":
        edge_scalars = _contact_geometry_release_plus_stonepairs_edge_scalars(
            node_coords, node_feats, node_mask, stone_stack="full21", binary_reach=True, stone_hop1=True
        )
    elif edge_scalar_mode == "contact_geometry_release_binary_plus_stonepairs_hop1_products13":
        edge_scalars = _contact_geometry_release_plus_stonepairs_edge_scalars(
            node_coords, node_feats, node_mask, stone_stack="products13", binary_reach=True, stone_hop1=True
        )
    elif edge_scalar_mode == "contact_geometry_release_binary_plus_stonepairs_hop1_basic9":
        edge_scalars = _contact_geometry_release_plus_stonepairs_edge_scalars(
            node_coords, node_feats, node_mask, stone_stack="basic9", binary_reach=True, stone_hop1=True
        )
    elif edge_scalar_mode == "contact_geometry_release_binary_plus_stonepairs_hop1_rac5":
        edge_scalars = _contact_geometry_release_plus_stonepairs_edge_scalars(
            node_coords, node_feats, node_mask, stone_stack="rac5", binary_reach=True, stone_hop1=True
        )
    elif edge_scalar_mode == "contact_geometry_release_binary_concat_stonepairs_hop1_products13_allgoals21":
        edge_scalars = _contact_geometry_release_concat_stonepairs_edge_scalars(
            node_coords,
            node_feats,
            node_mask,
            release_include_clearance=True,
            release_include_alignment=True,
            release_include_reach_alignment=True,
            release_include_onehop=False,
            stone_stack="products13",
            binary_reach=True,
            stone_hop1=True,
        )
    elif edge_scalar_mode == "contact_geometry_release_binary_concat_stonepairs_hop1_products13_products9":
        edge_scalars = _contact_geometry_release_concat_stonepairs_edge_scalars(
            node_coords,
            node_feats,
            node_mask,
            release_include_clearance=False,
            release_include_alignment=False,
            release_include_reach_alignment=False,
            release_include_onehop=False,
            stone_stack="products13",
            binary_reach=True,
            stone_hop1=True,
        )
    elif edge_scalar_mode == "contact_geometry_release_binary_concat_stonepairs_hop1_products13_onehop_no_center14":
        edge_scalars = _contact_geometry_release_concat_stonepairs_edge_scalars(
            node_coords,
            node_feats,
            node_mask,
            release_include_clearance=False,
            release_include_alignment=False,
            release_include_reach_alignment=False,
            release_include_onehop=True,
            release_include_kinds=("score", "takeout", "semi"),
            stone_stack="products13",
            binary_reach=True,
            stone_hop1=True,
        )
    elif edge_scalar_mode == "contact_geometry_release_binary_concat_stonepairs_hop1_products13_onehop_no_takeout_center10":
        edge_scalars = _contact_geometry_release_concat_stonepairs_edge_scalars(
            node_coords,
            node_feats,
            node_mask,
            release_include_clearance=False,
            release_include_alignment=False,
            release_include_reach_alignment=False,
            release_include_onehop=True,
            release_include_kinds=("score", "semi"),
            stone_stack="products13",
            binary_reach=True,
            stone_hop1=True,
        )
    elif edge_scalar_mode == "oldbest_plus_stonepairs_full21":
        edge_scalars = _oldbest_plus_stonepairs_edge_scalars(
            node_coords, node_feats, node_mask, c=c, stone_stack="full21"
        )
    elif edge_scalar_mode == "oldbest_plus_stonepairs_products13":
        edge_scalars = _oldbest_plus_stonepairs_edge_scalars(
            node_coords, node_feats, node_mask, c=c, stone_stack="products13"
        )
    elif edge_scalar_mode == "oldbest_plus_stonepairs_basic9":
        edge_scalars = _oldbest_plus_stonepairs_edge_scalars(
            node_coords, node_feats, node_mask, c=c, stone_stack="basic9"
        )
    elif edge_scalar_mode == "oldbest_plus_stonepairs_rac5":
        edge_scalars = _oldbest_plus_stonepairs_edge_scalars(
            node_coords, node_feats, node_mask, c=c, stone_stack="rac5"
        )
    elif edge_scalar_mode == "oldbest_plus_stonepairs_score6":
        edge_scalars = _oldbest_plus_stonepairs_edge_scalars(
            node_coords, node_feats, node_mask, c=c, stone_stack="score6"
        )
    elif edge_scalar_mode == "oldbest_plus_stonepairs_scoresemi11":
        edge_scalars = _oldbest_plus_stonepairs_edge_scalars(
            node_coords, node_feats, node_mask, c=c, stone_stack="scoresemi11"
        )
    elif edge_scalar_mode == "contact_geometry_all_sources_plus_alignment_plus_clearance_allgoals":
        edge_scalars = _contact_geometry_release_product_edge_scalars(
            node_coords,
            node_feats,
            node_mask,
            include_clearance=True,
            include_alignment=True,
            include_reach_alignment=True,
            include_onehop=False,
            source_mode="all_sources",
        )
    elif edge_scalar_mode == "contact_geometry_release_binary_reach_products_plus_alignment_plus_clearance_allgoals":
        edge_scalars = _contact_geometry_release_product_edge_scalars(
            node_coords,
            node_feats,
            node_mask,
            include_clearance=True,
            include_alignment=True,
            include_reach_alignment=True,
            include_onehop=False,
            binary_reach=True,
        )
    elif edge_scalar_mode == "contact_geometry_release_products_plus_alignment_plus_clearance_allgoals":
        edge_scalars = _contact_geometry_release_product_edge_scalars(
            node_coords,
            node_feats,
            node_mask,
            include_clearance=True,
            include_alignment=True,
            include_reach_alignment=True,
            include_onehop=False,
        )
    elif edge_scalar_mode == "contact_geometry_release_products_plus_alignment_plus_onehop_allgoals":
        edge_scalars = _contact_geometry_release_product_edge_scalars(
            node_coords,
            node_feats,
            node_mask,
            include_alignment=True,
            include_clearance=False,
            include_onehop=True,
        )
    elif edge_scalar_mode == "contact_geometry_release_products_plus_onehop_allgoals":
        edge_scalars = _contact_geometry_release_product_edge_scalars(
            node_coords, node_feats, node_mask, include_clearance=False, include_onehop=True
        )
    elif edge_scalar_mode == "contact_geometry_release_products_plus_onehop_no_takeout_center":
        edge_scalars = _contact_geometry_release_product_edge_scalars(
            node_coords,
            node_feats,
            node_mask,
            include_clearance=False,
            include_onehop=True,
            include_kinds=("score", "semi"),
        )
    elif edge_scalar_mode == "contact_geometry_release_products_plus_onehop_no_center":
        edge_scalars = _contact_geometry_release_product_edge_scalars(
            node_coords,
            node_feats,
            node_mask,
            include_clearance=False,
            include_onehop=True,
            include_kinds=("score", "takeout", "semi"),
        )
    elif edge_scalar_mode == "oldbest_plus_contact_geometry_release_products_plus_clearance":
        edge_scalars = _oldbest_plus_contact_geometry_edge_scalars(
            node_coords, node_feats, node_mask, c=c, include_clearance=True, include_onehop=False
        )
    elif edge_scalar_mode == "oldbest_plus_contact_geometry_release_products_plus_clearance_plus_onehop":
        edge_scalars = _oldbest_plus_contact_geometry_edge_scalars(
            node_coords, node_feats, node_mask, c=c, include_clearance=True, include_onehop=True
        )
    elif edge_scalar_mode == "active_button_region_visible_plus_release_reach_with_product":
        scorability = _compute_active_button_region_scorability(
            node_coords, node_feats, node_mask, shooter_team
        )
        scorability = scorability.unsqueeze(1).expand(-1, N, -1, -1)
        release_reach = _compute_release_reach_spans(node_coords, node_feats, node_mask)
        release_reach = release_reach.unsqueeze(1).expand(-1, N, -1, -1)
        product = scorability * release_reach
        edge_scalars = _pad_edge_scalars(scorability, release_reach, product)
    elif edge_scalar_mode == "button_visible_release_reach_takeout_product":
        scorability = _compute_unoccluded_angular_spans(node_coords, node_feats, node_mask)
        scorability = scorability.unsqueeze(1).expand(-1, N, -1, -1)
        release_reach = _compute_release_reach_spans(node_coords, node_feats, node_mask)
        release_reach = release_reach.unsqueeze(1).expand(-1, N, -1, -1)
        takeout = _compute_takeoutability_spans(node_coords, node_feats, node_mask)
        takeout = takeout.unsqueeze(1).expand(-1, N, -1, -1)
        edge_scalars = _pad_edge_scalars(
            release_reach,
            scorability,
            takeout,
            release_reach * scorability,
            release_reach * takeout,
        )
    elif edge_scalar_mode == "button_visible_release_reach_takeout_only":
        scorability = _compute_unoccluded_angular_spans(node_coords, node_feats, node_mask)
        scorability = scorability.unsqueeze(1).expand(-1, N, -1, -1)
        release_reach = _compute_release_reach_spans(node_coords, node_feats, node_mask)
        release_reach = release_reach.unsqueeze(1).expand(-1, N, -1, -1)
        takeout = _compute_takeoutability_spans(node_coords, node_feats, node_mask)
        takeout = takeout.unsqueeze(1).expand(-1, N, -1, -1)
        edge_scalars = _pad_edge_scalars(
            release_reach,
            scorability,
            takeout,
            release_reach * scorability,
        )
    elif edge_scalar_mode == "button_visible_release_reach_takeout_products_only":
        scorability = _compute_unoccluded_angular_spans(node_coords, node_feats, node_mask)
        scorability = scorability.unsqueeze(1).expand(-1, N, -1, -1)
        release_reach = _compute_release_reach_spans(node_coords, node_feats, node_mask)
        release_reach = release_reach.unsqueeze(1).expand(-1, N, -1, -1)
        takeout = _compute_takeoutability_spans(node_coords, node_feats, node_mask)
        takeout = takeout.unsqueeze(1).expand(-1, N, -1, -1)
        edge_scalars = _pad_edge_scalars(
            release_reach,
            scorability,
            release_reach * scorability,
            release_reach * takeout,
        )
    elif edge_scalar_mode == "minkowski_scoring_triplet":
        scorability = _compute_minkowski_scorability_spans(
            node_coords, node_feats, node_mask, shooter_team
        )
        scorability = scorability.unsqueeze(1).expand(-1, N, -1, -1)
        reachability = _compute_minkowski_release_reach_spans(node_coords, node_feats, node_mask)
        reachability = reachability.unsqueeze(1).expand(-1, N, -1, -1)
        scoring_reachability = _compute_minkowski_scoring_reachability_spans(
            node_coords, node_feats, node_mask, shooter_team
        )
        scoring_reachability = scoring_reachability.unsqueeze(1).expand(-1, N, -1, -1)
        edge_scalars = _pad_edge_scalars(scorability, reachability, scoring_reachability)
    elif edge_scalar_mode == "minkowski_scoring_takeout_quintet":
        scorability = _compute_minkowski_scorability_spans(
            node_coords, node_feats, node_mask, shooter_team
        )
        scorability = scorability.unsqueeze(1).expand(-1, N, -1, -1)
        reachability = _compute_minkowski_release_reach_spans(node_coords, node_feats, node_mask)
        reachability = reachability.unsqueeze(1).expand(-1, N, -1, -1)
        scoring_reachability = _compute_minkowski_scoring_reachability_spans(
            node_coords, node_feats, node_mask, shooter_team
        )
        scoring_reachability = scoring_reachability.unsqueeze(1).expand(-1, N, -1, -1)
        takeoutability = _compute_minkowski_takeoutability_spans(node_coords, node_feats, node_mask)
        takeoutability = takeoutability.unsqueeze(1).expand(-1, N, -1, -1)
        takeout_reachability = _compute_minkowski_takeout_reachability_spans(
            node_coords, node_feats, node_mask
        )
        takeout_reachability = takeout_reachability.unsqueeze(1).expand(-1, N, -1, -1)
        edge_scalars = _pad_edge_scalars(
            scorability, reachability, scoring_reachability, takeoutability, takeout_reachability
        )
    elif edge_scalar_mode == "button_visible_times_release_reach":
        button_visible = _compute_unoccluded_angular_spans(node_coords, node_feats, node_mask)
        button_visible = button_visible.unsqueeze(1).expand(-1, N, -1, -1)
        release_reach = _compute_release_reach_spans(node_coords, node_feats, node_mask)
        release_reach = release_reach.unsqueeze(1).expand(-1, N, -1, -1)
        product = button_visible * release_reach
        edge_scalars = _pad_edge_scalars(product)
    elif edge_scalar_mode == "exact_line_clearance":
        primary = _compute_pairwise_line_clearance(node_coords, node_feats, node_mask)
        edge_scalars = _pad_edge_scalars(primary)
    elif edge_scalar_mode == "pairwise_unoccluded_span":
        primary = _compute_pairwise_unoccluded_angular_spans(node_coords, node_feats, node_mask)
        primary = primary.expand(-1, N, -1, -1)
        edge_scalars = _pad_edge_scalars(primary)
    elif edge_scalar_mode == "button_visible_span":
        primary = _compute_unoccluded_angular_spans(node_coords, node_feats, node_mask)
        primary = primary.unsqueeze(1).expand(-1, N, -1, -1)
        edge_scalars = _pad_edge_scalars(primary)
    else:
        raise AssertionError(f"Unhandled edge scalar mode: {edge_scalar_mode}")

    edge_feats = torch.cat([dx, dist, same_team, edge_scalars], dim=-1)  # (B, N, N, 9)

    # Zero out edges involving masked (padding) nodes
    mask_2d = (node_mask.unsqueeze(2) & node_mask.unsqueeze(1)).unsqueeze(-1)  # (B, N, N, 1)
    edge_feats = edge_feats * mask_2d.float()

    if edge_scalar_mode in {
        "button_visible_plus_curl_arc_reach_clean",
        "button_visible_plus_curl_arc_minkowski_straight_reach_clean",
    } or _is_contact_arc_edge_mode(edge_scalar_mode):
        is_landmark = (node_feats[:, :, 4] > 0.5) & node_mask
        landmark_pair = is_landmark.unsqueeze(2) & is_landmark.unsqueeze(1)
        edge_feats = edge_feats * (~landmark_pair).unsqueeze(-1).float()

    if _resolve_edge_prune_mode() == "stone_pair_zero_pairwise_span":
        pairwise = _compute_pairwise_unoccluded_angular_spans(node_coords, node_feats, node_mask).squeeze(-1)
        is_live = (node_feats[:, :, 3] > 0.5) & node_mask
        stone_pair = (is_live.unsqueeze(2) & is_live.unsqueeze(1))
        keep = (~stone_pair) | (pairwise > 1e-8)
        edge_feats = edge_feats * keep.unsqueeze(-1).float()

    return edge_feats


EDGE_FEAT_DIM = 4 + EDGE_SCALAR_DIM  # dx, dy, dist, same_team, edge scalar channels


# ─────────────────────────────────────────────────────────────────────
# EGNN: E(n)-Equivariant Graph Neural Network Message Passing Layer
# ─────────────────────────────────────────────────────────────────────

class EGNNLayer(nn.Module):
    """
    Single EGNN message passing layer.

    Message: m_ij = phi_e(h_i, h_j, ||x_i - x_j||^2, edge_feats_ij)
    Aggregation: m_i = sum_j m_ij
    Update: h_i' = h_i + phi_h(h_i, m_i)
    Coord update: x_i' = x_i + sum_j (x_i - x_j) * phi_x(m_ij)  [equivariant]
    """

    def __init__(self, hidden_dim, edge_feat_dim=EDGE_FEAT_DIM, dropout=0.1,
                 update_coords=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.update_coords = update_coords

        # Edge MLP: maps (h_i, h_j, dist^2, edge_feats) -> message
        edge_in = 2 * hidden_dim + 1 + edge_feat_dim
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_in, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        # Node update MLP
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Coordinate update (scalar weight per edge)
        if update_coords:
            self.coord_mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 1),
            )

        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, h, coords, edge_feats, node_mask):
        """
        Args:
            h: (B, N, hidden_dim) node features
            coords: (B, N, 2) node coordinates
            edge_feats: (B, N, N, edge_feat_dim)
            node_mask: (B, N) bool

        Returns:
            h_new: (B, N, hidden_dim)
            coords_new: (B, N, 2)
        """
        B, N, D = h.shape
        device = h.device

        # Pairwise distances squared
        diff = coords.unsqueeze(2) - coords.unsqueeze(1)  # (B, N, N, 2)
        dist_sq = (diff ** 2).sum(dim=-1, keepdim=True)  # (B, N, N, 1)

        # Build edge inputs: (h_i, h_j, dist_sq, edge_feats)
        hi = h.unsqueeze(2).expand(-1, -1, N, -1)  # (B, N, N, D)
        hj = h.unsqueeze(1).expand(-1, N, -1, -1)  # (B, N, N, D)
        edge_input = torch.cat([hi, hj, dist_sq, edge_feats], dim=-1)  # (B,N,N,2D+1+E)

        # Compute messages
        messages = self.edge_mlp(edge_input)  # (B, N, N, D)

        # Mask invalid edges
        mask_2d = (node_mask.unsqueeze(2) & node_mask.unsqueeze(1))  # (B, N, N)
        # Also mask self-loops
        eye = torch.eye(N, dtype=torch.bool, device=device).unsqueeze(0)
        mask_2d = mask_2d & ~eye
        messages = messages * mask_2d.unsqueeze(-1).float()

        # Aggregate messages (sum)
        agg = messages.sum(dim=2)  # (B, N, D)

        # Node update with residual
        h_new = h + self.node_mlp(torch.cat([h, agg], dim=-1))
        h_new = self.layer_norm(h_new)

        # Coordinate update (equivariant)
        coords_new = coords
        if self.update_coords:
            coord_weights = self.coord_mlp(messages)  # (B, N, N, 1)
            coord_weights = coord_weights * mask_2d.unsqueeze(-1).float()
            # Weighted sum of displacement vectors
            coord_shift = (diff * coord_weights).sum(dim=2)  # (B, N, 2)
            # Clamp to prevent explosions
            coord_shift = torch.clamp(coord_shift, -0.1, 0.1)
            coords_new = coords + coord_shift

        # Zero out padding nodes
        h_new = h_new * node_mask.unsqueeze(-1).float()
        coords_new = coords_new * node_mask.unsqueeze(-1).float()

        return h_new, coords_new


class ValueEGNN(nn.Module):
    """
    EGNN for curling value prediction.

    Builds a graph with live stones + landmark nodes.
    Passes messages through EGNN layers.
    Aggregates via masked mean pooling.
    Concatenates condition features and predicts value.
    """

    def __init__(self, input_dim=24, cond_dim=3, hidden_dim=128,
                 n_layers=3, n_heads=4, dropout=0.1, **kwargs):
        super().__init__()
        self.input_dim = input_dim
        self.cond_dim = cond_dim
        self.hidden_dim = hidden_dim

        # Node feature projection
        self.node_proj = nn.Linear(NODE_FEAT_DIM, hidden_dim)

        # EGNN layers
        self.layers = nn.ModuleList([
            EGNNLayer(hidden_dim, edge_feat_dim=EDGE_FEAT_DIM,
                      dropout=dropout, update_coords=(i < n_layers - 1))
            for i in range(n_layers)
        ])

        # Condition projection
        self.cond_proj = nn.Linear(cond_dim, hidden_dim)

        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x, c):
        B = x.size(0)
        device = x.device

        # Build graph
        node_feats, node_coords, node_mask, n_nodes = build_graph_batch(x, device)

        # Compute edge features
        edge_feats = compute_edge_features(node_coords, node_feats, node_mask, c=c)

        # Project node features
        h = self.node_proj(node_feats)  # (B, max_nodes, hidden)
        h = h * node_mask.unsqueeze(-1).float()

        coords = node_coords.clone()

        # Message passing
        for layer in self.layers:
            h, coords = layer(h, coords, edge_feats, node_mask)
            # Recompute edge features with updated coords
            edge_feats = compute_edge_features(coords, node_feats, node_mask, c=c)

        # Masked mean pooling
        mask_f = node_mask.unsqueeze(-1).float()  # (B, N, 1)
        h_pooled = (h * mask_f).sum(dim=1) / (mask_f.sum(dim=1).clamp(min=1.0))  # (B, D)

        # Condition
        c_proj = self.cond_proj(c)  # (B, D)

        # Predict
        combined = torch.cat([h_pooled, c_proj], dim=-1)  # (B, 2D)
        return self.value_head(combined)


# ─────────────────────────────────────────────────────────────────────
# Graph Transformer: Attention-based message passing
# ─────────────────────────────────────────────────────────────────────

class GraphTransformerLayer(nn.Module):
    """
    Multi-head attention with edge features.

    Attention: a_ij = softmax( (Q_i . K_j + bias(edge_ij)) / sqrt(d_k) )
    Output: o_i = sum_j a_ij * (V_j + edge_value(edge_ij))

    Edge features modulate both attention weights and values.
    """

    def __init__(self, hidden_dim, n_heads=4, edge_feat_dim=EDGE_FEAT_DIM,
                 dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.d_k = hidden_dim // n_heads
        assert hidden_dim % n_heads == 0

        self.W_q = nn.Linear(hidden_dim, hidden_dim)
        self.W_k = nn.Linear(hidden_dim, hidden_dim)
        self.W_v = nn.Linear(hidden_dim, hidden_dim)
        self.W_o = nn.Linear(hidden_dim, hidden_dim)

        # Edge feature projection for attention bias
        self.edge_attn = nn.Linear(edge_feat_dim, n_heads)
        # Edge feature projection for value modulation
        self.edge_value = nn.Linear(edge_feat_dim, hidden_dim)

        self.attn_dropout = nn.Dropout(dropout)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, h, edge_feats, node_mask):
        """
        Args:
            h: (B, N, D)
            edge_feats: (B, N, N, edge_feat_dim)
            node_mask: (B, N) bool

        Returns:
            h_new: (B, N, D)
        """
        B, N, D = h.shape
        H = self.n_heads
        dk = self.d_k

        # Multi-head attention
        Q = self.W_q(h).view(B, N, H, dk).transpose(1, 2)  # (B, H, N, dk)
        K = self.W_k(h).view(B, N, H, dk).transpose(1, 2)  # (B, H, N, dk)
        V = self.W_v(h).view(B, N, H, dk).transpose(1, 2)  # (B, H, N, dk)

        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(dk)  # (B, H, N, N)

        # Edge bias
        edge_bias = self.edge_attn(edge_feats)  # (B, N, N, H)
        edge_bias = edge_bias.permute(0, 3, 1, 2)  # (B, H, N, N)
        scores = scores + edge_bias

        # Mask invalid nodes (set attention to -inf for padding)
        mask_2d = node_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, N) - mask keys
        scores = scores.masked_fill(~mask_2d, float('-inf'))

        # Also mask out queries that are padding
        mask_q = node_mask.unsqueeze(1).unsqueeze(-1)  # (B, 1, N, 1)

        attn = F.softmax(scores, dim=-1)
        # Replace NaN from all-masked rows with 0
        attn = attn.masked_fill(torch.isnan(attn), 0.0)
        attn = self.attn_dropout(attn)

        # Value with edge modulation
        edge_v = self.edge_value(edge_feats)  # (B, N, N, D)
        edge_v = edge_v.view(B, N, N, H, dk).permute(0, 3, 1, 2, 4)  # (B, H, N, N, dk)

        # Attended values: standard attention + edge value contribution
        V_expanded = V.unsqueeze(3).expand(-1, -1, -1, N, -1)  # (B, H, N_q, N_k, dk)
        # Actually we want V[j] for each (i,j) pair
        V_for_edges = V.unsqueeze(2).expand(-1, -1, N, -1, -1)  # (B, H, N_i, N_j, dk)
        combined_v = V_for_edges + edge_v  # (B, H, N, N, dk)

        # Weighted sum
        attn_expanded = attn.unsqueeze(-1)  # (B, H, N, N, 1)
        out = (attn_expanded * combined_v).sum(dim=3)  # (B, H, N, dk)
        out = out.transpose(1, 2).contiguous().view(B, N, D)  # (B, N, D)

        # Residual + norm
        h = h + self.drop(self.W_o(out))
        h = self.norm1(h)

        # FFN with residual
        h = h + self.ffn(h)
        h = self.norm2(h)

        # Zero out padding
        h = h * node_mask.unsqueeze(-1).float()

        return h


class ValueGraphTransformer(nn.Module):
    """
    Graph Transformer for curling value prediction.

    Same graph construction as EGNN but uses multi-head attention
    with edge features instead of geometric message passing.
    """

    def __init__(self, input_dim=24, cond_dim=3, hidden_dim=128,
                 n_layers=3, n_heads=4, dropout=0.1, **kwargs):
        super().__init__()
        self.input_dim = input_dim
        self.cond_dim = cond_dim
        self.hidden_dim = hidden_dim

        # Node feature projection
        self.node_proj = nn.Linear(NODE_FEAT_DIM, hidden_dim)

        # Condition injection as a virtual global node
        self.cond_proj = nn.Linear(cond_dim, hidden_dim)
        self.global_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

        # Graph transformer layers
        self.layers = nn.ModuleList([
            GraphTransformerLayer(hidden_dim, n_heads=n_heads,
                                 edge_feat_dim=EDGE_FEAT_DIM, dropout=dropout)
            for _ in range(n_layers)
        ])

        # Value head: reads from global token
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x, c):
        B = x.size(0)
        device = x.device

        # Build graph
        node_feats, node_coords, node_mask, n_nodes = build_graph_batch(x, device)
        max_N = node_feats.shape[1]

        # Compute edge features
        edge_feats = compute_edge_features(node_coords, node_feats, node_mask, c=c)

        # Project node features
        h = self.node_proj(node_feats)  # (B, max_N, D)
        h = h * node_mask.unsqueeze(-1).float()

        # Add global (condition) token as extra node
        global_h = self.cond_proj(c).unsqueeze(1) + self.global_token.expand(B, -1, -1)  # (B, 1, D)
        h = torch.cat([global_h, h], dim=1)  # (B, max_N+1, D)

        # Expand edge features to include global node
        # Global node has zero edge features with all other nodes
        new_N = max_N + 1
        new_edge = torch.zeros(B, new_N, new_N, EDGE_FEAT_DIM, device=device)
        new_edge[:, 1:, 1:, :] = edge_feats  # existing edges

        # Expand node mask
        new_mask = torch.zeros(B, new_N, dtype=torch.bool, device=device)
        new_mask[:, 0] = True  # global token always valid
        new_mask[:, 1:] = node_mask

        # Message passing
        for layer in self.layers:
            h = layer(h, new_edge, new_mask)

        # Read from global token (index 0)
        global_out = h[:, 0, :]  # (B, D)

        return self.value_head(global_out)


# ─────────────────────────────────────────────────────────────────────
# Vectorized graph builder (much faster - no Python loops over batch)
# ─────────────────────────────────────────────────────────────────────

def build_graph_batch_fast(x, device):
    """
    Vectorized graph construction - no per-sample Python loops.

    All 12 stone slots + the active landmark set are always present as nodes.
    Dead/unthrown stones get is_live=0 and will be masked out in attention/messages.
    This allows fully batched operations.
    """
    B = x.size(0)
    stones = x.view(B, NUM_STONES, 2)

    landmarks = _get_active_landmarks(device=device, dtype=stones.dtype)
    n_landmarks = landmarks.shape[0]
    max_nodes = NUM_STONES + n_landmarks

    # Detect live stones
    is_live = ((stones.sum(dim=-1) > 0.001) &
               (stones.max(dim=-1).values < 0.999))  # (B, 12)

    # Build node coordinates: stones + active landmarks
    landmark_coords = landmarks.unsqueeze(0).expand(B, -1, -1)  # (B, n_landmarks, 2)
    node_coords = torch.cat([stones, landmark_coords], dim=1)  # (B, max_nodes, 2)

    # Build node features: (x, y, team, is_live, is_landmark)
    node_feats = torch.zeros(B, max_nodes, NODE_FEAT_DIM, device=device)

    # Stone features
    node_feats[:, :NUM_STONES, 0:2] = stones  # x, y
    team_ids = torch.zeros(NUM_STONES, device=device)
    team_ids[6:] = 1.0
    node_feats[:, :NUM_STONES, 2] = team_ids.unsqueeze(0).expand(B, -1)  # team
    node_feats[:, :NUM_STONES, 3] = is_live.float()  # is_live
    node_feats[:, :NUM_STONES, 4] = 0.0  # not landmark

    # Landmark features
    node_feats[:, NUM_STONES:, 0:2] = landmark_coords  # x, y
    node_feats[:, NUM_STONES:, 2] = 0.0  # no team
    node_feats[:, NUM_STONES:, 3] = 0.0  # not live stone
    node_feats[:, NUM_STONES:, 4] = 1.0  # is landmark

    # Node mask: live stones + all landmarks
    node_mask = torch.zeros(B, max_nodes, dtype=torch.bool, device=device)
    node_mask[:, :NUM_STONES] = is_live
    node_mask[:, NUM_STONES:NUM_STONES + n_landmarks] = True  # active landmarks always present

    node_feature_mode = _resolve_node_feature_mode()
    if node_feature_mode == "button_visible_span":
        node_feats[:, :, 5:6] = _compute_unoccluded_angular_spans(node_coords, node_feats, node_mask)
    elif node_feature_mode == "release_reach_times_takeout":
        release_reach = _compute_release_reach_spans(node_coords, node_feats, node_mask)
        takeout = _compute_takeoutability_spans(node_coords, node_feats, node_mask)
        node_feats[:, :, 5:6] = release_reach * takeout
    elif node_feature_mode == "beats_nearest_opponent_to_button":
        node_feats[:, :, 5:6] = _compute_beats_nearest_opponent_to_button(node_coords, node_feats, node_mask)

    n_nodes = node_mask.sum(dim=1)  # (B,)

    return node_feats, node_coords, node_mask, n_nodes


def compute_edge_features_fast(node_coords, node_feats, node_mask, c=None):
    """
    Vectorized edge feature computation.
    Same features as compute_edge_features, padded to a fixed 2-scalar edge channel.
    """
    B, N, _ = node_coords.shape
    device = node_coords.device

    # Relative displacement
    dx = node_coords.unsqueeze(2) - node_coords.unsqueeze(1)  # (B, N, N, 2)
    dist = torch.norm(dx, dim=-1, keepdim=True).clamp(min=1e-8)  # (B, N, N, 1)

    # Same team
    team = node_feats[:, :, 2]  # (B, N)
    is_live = node_feats[:, :, 3]  # (B, N)
    same_team = (team.unsqueeze(2) == team.unsqueeze(1)).float()
    both_live = is_live.unsqueeze(2) * is_live.unsqueeze(1)
    same_team = (same_team * both_live).unsqueeze(-1)  # (B, N, N, 1)

    shooter_team = _extract_shooter_team(c, node_coords)
    edge_scalar_mode = _resolve_edge_scalar_mode()
    if edge_scalar_mode == "thrower_masked_button_region_span":
        primary = _compute_pairwise_thrower_masked_button_region_spans(
            node_coords, node_feats, node_mask, shooter_team
        )
        edge_scalars = _pad_edge_scalars(primary)
    elif edge_scalar_mode == "thrower_masked_pairwise_span":
        primary = _compute_pairwise_thrower_masked_spans(node_coords, node_feats, node_mask)
        edge_scalars = _pad_edge_scalars(primary)
    elif edge_scalar_mode == "button_region_pairwise_span":
        primary = _compute_pairwise_button_region_spans(
            node_coords, node_feats, node_mask, shooter_team
        )
        edge_scalars = _pad_edge_scalars(primary)
    elif edge_scalar_mode == "button_visible_plus_thrower_masked_span":
        button_visible = _compute_unoccluded_angular_spans(node_coords, node_feats, node_mask)
        button_visible = button_visible.unsqueeze(1).expand(-1, N, -1, -1)
        reachability = _compute_pairwise_thrower_masked_button_region_spans(
            node_coords, node_feats, node_mask, shooter_team
        )
        edge_scalars = _pad_edge_scalars(button_visible, reachability)
    elif edge_scalar_mode == "button_visible_plus_release_reach_span":
        button_visible = _compute_unoccluded_angular_spans(node_coords, node_feats, node_mask)
        button_visible = button_visible.unsqueeze(1).expand(-1, N, -1, -1)
        release_reach = _compute_release_reach_spans(node_coords, node_feats, node_mask)
        release_reach = release_reach.unsqueeze(1).expand(-1, N, -1, -1)
        edge_scalars = _pad_edge_scalars(button_visible, release_reach)
    elif edge_scalar_mode == "button_visible_plus_release_reach_with_product":
        button_visible = _compute_unoccluded_angular_spans(node_coords, node_feats, node_mask)
        button_visible = button_visible.unsqueeze(1).expand(-1, N, -1, -1)
        release_reach = _compute_release_reach_spans(node_coords, node_feats, node_mask)
        release_reach = release_reach.unsqueeze(1).expand(-1, N, -1, -1)
        product = button_visible * release_reach
        edge_scalars = _pad_edge_scalars(button_visible, release_reach, product)
    elif edge_scalar_mode == "button_visible_plus_source_reach_takeout_edges_with_product":
        button_visible = _compute_unoccluded_angular_spans(node_coords, node_feats, node_mask)
        button_visible = button_visible.unsqueeze(1).expand(-1, N, -1, -1)
        pairwise = _compute_pairwise_unoccluded_angular_spans(node_coords, node_feats, node_mask)
        release_sources, takeout_sources = _source_landmark_masks(node_coords, node_feats, node_mask)
        release_edge_reach = pairwise * release_sources.view(B, N, 1, 1).float()
        takeout_edge_reach = pairwise * takeout_sources.view(B, N, 1, 1).float()
        edge_scalars = _pad_edge_scalars(
            button_visible,
            release_edge_reach,
            takeout_edge_reach,
            button_visible * release_edge_reach,
            button_visible * takeout_edge_reach,
        )
    elif edge_scalar_mode == "button_visible_plus_curl_arc_reach_with_outgoing":
        scorability = _compute_unoccluded_angular_spans(node_coords, node_feats, node_mask)
        scorability = scorability.unsqueeze(1).expand(-1, N, -1, -1)
        clearance, feasible, diversity, score_out, takeout_out = _compute_curl_arc_reachability_and_outgoing(
            node_coords, node_feats, node_mask, c=c
        )
        pairwise_reach = _compute_pairwise_unoccluded_angular_spans(node_coords, node_feats, node_mask)
        release_sources, _ = _source_landmark_masks(node_coords, node_feats, node_mask)
        release_edge = release_sources.view(B, N, 1, 1)
        straight_score, straight_takeout = _compute_pairwise_outgoing_compatibility(
            node_coords, node_feats, node_mask, c=c
        )
        curl_reach = feasible * diversity
        reach = torch.where(release_edge, curl_reach, pairwise_reach)
        score_out = torch.where(release_edge, score_out, straight_score)
        takeout_out = torch.where(release_edge, takeout_out, straight_takeout)
        edge_scalars = _pad_edge_scalars(
            scorability,
            reach,
            score_out,
            takeout_out,
            reach * torch.maximum(score_out, takeout_out),
        )
    elif edge_scalar_mode == "button_visible_plus_curl_arc_reach_clean":
        edge_scalars = _compute_clean_curl_reach_edge_scalars(
            node_coords, node_feats, node_mask, c=c, use_minkowski_straight=False
        )
    elif edge_scalar_mode == "button_visible_plus_curl_arc_minkowski_straight_reach_clean":
        edge_scalars = _compute_clean_curl_reach_edge_scalars(
            node_coords, node_feats, node_mask, c=c, use_minkowski_straight=True
        )
    elif edge_scalar_mode == "button_visible_plus_contact_arc_full":
        edge_scalars = _compute_contact_arc_edge_scalars(
            node_coords, node_feats, node_mask, c=c, stack="full"
        )
    elif edge_scalar_mode == "button_visible_plus_contact_arc_minimal":
        edge_scalars = _compute_contact_arc_edge_scalars(
            node_coords, node_feats, node_mask, c=c, stack="minimal"
        )
    elif edge_scalar_mode == "button_visible_plus_contact_arc_score_product":
        edge_scalars = _compute_contact_arc_edge_scalars(
            node_coords, node_feats, node_mask, c=c, stack="score_product"
        )
    elif edge_scalar_mode == "button_visible_plus_contact_arc_takeout_product":
        edge_scalars = _compute_contact_arc_edge_scalars(
            node_coords, node_feats, node_mask, c=c, stack="takeout_product"
        )
    elif edge_scalar_mode == "button_visible_plus_contact_arc_old_shape":
        edge_scalars = _compute_contact_arc_edge_scalars(
            node_coords, node_feats, node_mask, c=c, stack="old_shape"
        )
    elif edge_scalar_mode == "goal_contact_oldbest_plus_chain_2hop":
        edge_scalars = _goal_contact_oldbest_plus_chain_2hop_edge_scalars(
            node_coords, node_feats, node_mask, c=c
        )
    elif edge_scalar_mode == "goal_contact_chain_outside_2hop":
        edge_scalars = _goal_contact_chain_2hop_edge_scalars(
            node_coords, node_feats, node_mask, c=c
        )
    elif edge_scalar_mode == "goal_contact_chain_2hop":
        edge_scalars = _goal_contact_chain_2hop_edge_scalars(
            node_coords, node_feats, node_mask, c=c
        )
    elif edge_scalar_mode == "goal_contact_chain_1hop":
        edge_scalars = _goal_contact_chain_1hop_edge_scalars(
            node_coords, node_feats, node_mask, c=c
        )
    elif edge_scalar_mode == "goal_contact_chain_1hop_topk":
        edge_scalars = _goal_contact_chain_1hop_topk_edge_scalars(
            node_coords, node_feats, node_mask, c=c
        )
    elif edge_scalar_mode == "goal_contact_1hop_terms":
        edge_scalars = _goal_contact_1hop_terms_edge_scalars(
            node_coords, node_feats, node_mask, c=c
        )
    elif edge_scalar_mode == "goal_contact_usefulness_2hop":
        edge_scalars = _goal_contact_usefulness_2hop_edge_scalars(
            node_coords, node_feats, node_mask, c=c
        )
    elif edge_scalar_mode == "contact_geometry_release_products":
        edge_scalars = _contact_geometry_release_product_edge_scalars(
            node_coords, node_feats, node_mask
        )
    elif edge_scalar_mode == "contact_geometry_release_binary_reach_products":
        edge_scalars = _contact_geometry_release_product_edge_scalars(
            node_coords, node_feats, node_mask, binary_reach=True
        )
    elif edge_scalar_mode == "contact_geometry_release_products_plus_clearance":
        edge_scalars = _contact_geometry_release_product_edge_scalars(
            node_coords, node_feats, node_mask, include_clearance=True
        )
    elif edge_scalar_mode == "contact_geometry_release_binary_reach_products_plus_clearance":
        edge_scalars = _contact_geometry_release_product_edge_scalars(
            node_coords, node_feats, node_mask, include_clearance=True, binary_reach=True
        )
    elif edge_scalar_mode == "contact_geometry_release_products_plus_clearance_plus_onehop":
        edge_scalars = _contact_geometry_release_product_edge_scalars(
            node_coords, node_feats, node_mask, include_clearance=True, include_onehop=True
        )
    elif edge_scalar_mode == "contact_geometry_release_binary_reach_products_plus_onehop_allgoals":
        edge_scalars = _contact_geometry_release_product_edge_scalars(
            node_coords, node_feats, node_mask, include_clearance=False, include_onehop=True, binary_reach=True
        )
    elif edge_scalar_mode == "contact_geometry_release_binary_reach_products_plus_onehop_no_takeout_center":
        edge_scalars = _contact_geometry_release_product_edge_scalars(
            node_coords,
            node_feats,
            node_mask,
            include_clearance=False,
            include_onehop=True,
            include_kinds=("score", "semi"),
            binary_reach=True,
        )
    elif edge_scalar_mode == "contact_geometry_release_binary_reach_products_plus_onehop_no_center":
        edge_scalars = _contact_geometry_release_product_edge_scalars(
            node_coords,
            node_feats,
            node_mask,
            include_clearance=False,
            include_onehop=True,
            include_kinds=("score", "takeout", "semi"),
            binary_reach=True,
        )
    elif edge_scalar_mode == "contact_geometry_release_plus_stonepairs_full21":
        edge_scalars = _contact_geometry_release_plus_stonepairs_edge_scalars(
            node_coords, node_feats, node_mask, stone_stack="full21"
        )
    elif edge_scalar_mode == "contact_geometry_release_plus_stonepairs_products13":
        edge_scalars = _contact_geometry_release_plus_stonepairs_edge_scalars(
            node_coords, node_feats, node_mask, stone_stack="products13"
        )
    elif edge_scalar_mode == "contact_geometry_release_plus_stonepairs_basic9":
        edge_scalars = _contact_geometry_release_plus_stonepairs_edge_scalars(
            node_coords, node_feats, node_mask, stone_stack="basic9"
        )
    elif edge_scalar_mode == "contact_geometry_release_plus_stonepairs_rac5":
        edge_scalars = _contact_geometry_release_plus_stonepairs_edge_scalars(
            node_coords, node_feats, node_mask, stone_stack="rac5"
        )
    elif edge_scalar_mode == "contact_geometry_release_plus_stonepairs_score6":
        edge_scalars = _contact_geometry_release_plus_stonepairs_edge_scalars(
            node_coords, node_feats, node_mask, stone_stack="score6"
        )
    elif edge_scalar_mode == "contact_geometry_release_plus_stonepairs_scoresemi11":
        edge_scalars = _contact_geometry_release_plus_stonepairs_edge_scalars(
            node_coords, node_feats, node_mask, stone_stack="scoresemi11"
        )
    elif edge_scalar_mode == "contact_geometry_release_binary_plus_stonepairs_full21":
        edge_scalars = _contact_geometry_release_plus_stonepairs_edge_scalars(
            node_coords, node_feats, node_mask, stone_stack="full21", binary_reach=True
        )
    elif edge_scalar_mode == "contact_geometry_release_binary_plus_stonepairs_products13":
        edge_scalars = _contact_geometry_release_plus_stonepairs_edge_scalars(
            node_coords, node_feats, node_mask, stone_stack="products13", binary_reach=True
        )
    elif edge_scalar_mode == "contact_geometry_release_binary_plus_stonepairs_basic9":
        edge_scalars = _contact_geometry_release_plus_stonepairs_edge_scalars(
            node_coords, node_feats, node_mask, stone_stack="basic9", binary_reach=True
        )
    elif edge_scalar_mode == "contact_geometry_release_binary_plus_stonepairs_rac5":
        edge_scalars = _contact_geometry_release_plus_stonepairs_edge_scalars(
            node_coords, node_feats, node_mask, stone_stack="rac5", binary_reach=True
        )
    elif edge_scalar_mode == "contact_geometry_release_binary_plus_stonepairs_scoresemi11":
        edge_scalars = _contact_geometry_release_plus_stonepairs_edge_scalars(
            node_coords, node_feats, node_mask, stone_stack="scoresemi11", binary_reach=True
        )
    elif edge_scalar_mode == "contact_geometry_release_binary_plus_stonepairs_hop1_full21":
        edge_scalars = _contact_geometry_release_plus_stonepairs_edge_scalars(
            node_coords, node_feats, node_mask, stone_stack="full21", binary_reach=True, stone_hop1=True
        )
    elif edge_scalar_mode == "contact_geometry_release_binary_plus_stonepairs_hop1_products13":
        edge_scalars = _contact_geometry_release_plus_stonepairs_edge_scalars(
            node_coords, node_feats, node_mask, stone_stack="products13", binary_reach=True, stone_hop1=True
        )
    elif edge_scalar_mode == "contact_geometry_release_binary_plus_stonepairs_hop1_basic9":
        edge_scalars = _contact_geometry_release_plus_stonepairs_edge_scalars(
            node_coords, node_feats, node_mask, stone_stack="basic9", binary_reach=True, stone_hop1=True
        )
    elif edge_scalar_mode == "contact_geometry_release_binary_plus_stonepairs_hop1_rac5":
        edge_scalars = _contact_geometry_release_plus_stonepairs_edge_scalars(
            node_coords, node_feats, node_mask, stone_stack="rac5", binary_reach=True, stone_hop1=True
        )
    elif edge_scalar_mode == "contact_geometry_release_binary_concat_stonepairs_hop1_products13_allgoals21":
        edge_scalars = _contact_geometry_release_concat_stonepairs_edge_scalars(
            node_coords,
            node_feats,
            node_mask,
            release_include_clearance=True,
            release_include_alignment=True,
            release_include_reach_alignment=True,
            release_include_onehop=False,
            stone_stack="products13",
            binary_reach=True,
            stone_hop1=True,
        )
    elif edge_scalar_mode == "contact_geometry_release_binary_concat_stonepairs_hop1_products13_products9":
        edge_scalars = _contact_geometry_release_concat_stonepairs_edge_scalars(
            node_coords,
            node_feats,
            node_mask,
            release_include_clearance=False,
            release_include_alignment=False,
            release_include_reach_alignment=False,
            release_include_onehop=False,
            stone_stack="products13",
            binary_reach=True,
            stone_hop1=True,
        )
    elif edge_scalar_mode == "contact_geometry_release_binary_concat_stonepairs_hop1_products13_onehop_no_center14":
        edge_scalars = _contact_geometry_release_concat_stonepairs_edge_scalars(
            node_coords,
            node_feats,
            node_mask,
            release_include_clearance=False,
            release_include_alignment=False,
            release_include_reach_alignment=False,
            release_include_onehop=True,
            release_include_kinds=("score", "takeout", "semi"),
            stone_stack="products13",
            binary_reach=True,
            stone_hop1=True,
        )
    elif edge_scalar_mode == "contact_geometry_release_binary_concat_stonepairs_hop1_products13_onehop_no_takeout_center10":
        edge_scalars = _contact_geometry_release_concat_stonepairs_edge_scalars(
            node_coords,
            node_feats,
            node_mask,
            release_include_clearance=False,
            release_include_alignment=False,
            release_include_reach_alignment=False,
            release_include_onehop=True,
            release_include_kinds=("score", "semi"),
            stone_stack="products13",
            binary_reach=True,
            stone_hop1=True,
        )
    elif edge_scalar_mode == "oldbest_plus_stonepairs_full21":
        edge_scalars = _oldbest_plus_stonepairs_edge_scalars(
            node_coords, node_feats, node_mask, c=c, stone_stack="full21"
        )
    elif edge_scalar_mode == "oldbest_plus_stonepairs_products13":
        edge_scalars = _oldbest_plus_stonepairs_edge_scalars(
            node_coords, node_feats, node_mask, c=c, stone_stack="products13"
        )
    elif edge_scalar_mode == "oldbest_plus_stonepairs_basic9":
        edge_scalars = _oldbest_plus_stonepairs_edge_scalars(
            node_coords, node_feats, node_mask, c=c, stone_stack="basic9"
        )
    elif edge_scalar_mode == "oldbest_plus_stonepairs_rac5":
        edge_scalars = _oldbest_plus_stonepairs_edge_scalars(
            node_coords, node_feats, node_mask, c=c, stone_stack="rac5"
        )
    elif edge_scalar_mode == "oldbest_plus_stonepairs_score6":
        edge_scalars = _oldbest_plus_stonepairs_edge_scalars(
            node_coords, node_feats, node_mask, c=c, stone_stack="score6"
        )
    elif edge_scalar_mode == "oldbest_plus_stonepairs_scoresemi11":
        edge_scalars = _oldbest_plus_stonepairs_edge_scalars(
            node_coords, node_feats, node_mask, c=c, stone_stack="scoresemi11"
        )
    elif edge_scalar_mode == "contact_geometry_all_sources_plus_alignment_plus_clearance_allgoals":
        edge_scalars = _contact_geometry_release_product_edge_scalars(
            node_coords,
            node_feats,
            node_mask,
            include_clearance=True,
            include_alignment=True,
            include_reach_alignment=True,
            include_onehop=False,
            source_mode="all_sources",
        )
    elif edge_scalar_mode == "contact_geometry_release_binary_reach_products_plus_alignment_plus_clearance_allgoals":
        edge_scalars = _contact_geometry_release_product_edge_scalars(
            node_coords,
            node_feats,
            node_mask,
            include_clearance=True,
            include_alignment=True,
            include_reach_alignment=True,
            include_onehop=False,
            binary_reach=True,
        )
    elif edge_scalar_mode == "contact_geometry_release_products_plus_alignment_plus_clearance_allgoals":
        edge_scalars = _contact_geometry_release_product_edge_scalars(
            node_coords,
            node_feats,
            node_mask,
            include_clearance=True,
            include_alignment=True,
            include_reach_alignment=True,
            include_onehop=False,
        )
    elif edge_scalar_mode == "contact_geometry_release_products_plus_alignment_plus_onehop_allgoals":
        edge_scalars = _contact_geometry_release_product_edge_scalars(
            node_coords,
            node_feats,
            node_mask,
            include_alignment=True,
            include_clearance=False,
            include_onehop=True,
        )
    elif edge_scalar_mode == "contact_geometry_release_products_plus_onehop_allgoals":
        edge_scalars = _contact_geometry_release_product_edge_scalars(
            node_coords, node_feats, node_mask, include_clearance=False, include_onehop=True
        )
    elif edge_scalar_mode == "contact_geometry_release_products_plus_onehop_no_takeout_center":
        edge_scalars = _contact_geometry_release_product_edge_scalars(
            node_coords,
            node_feats,
            node_mask,
            include_clearance=False,
            include_onehop=True,
            include_kinds=("score", "semi"),
        )
    elif edge_scalar_mode == "contact_geometry_release_products_plus_onehop_no_center":
        edge_scalars = _contact_geometry_release_product_edge_scalars(
            node_coords,
            node_feats,
            node_mask,
            include_clearance=False,
            include_onehop=True,
            include_kinds=("score", "takeout", "semi"),
        )
    elif edge_scalar_mode == "oldbest_plus_contact_geometry_release_products_plus_clearance":
        edge_scalars = _oldbest_plus_contact_geometry_edge_scalars(
            node_coords, node_feats, node_mask, c=c, include_clearance=True, include_onehop=False
        )
    elif edge_scalar_mode == "oldbest_plus_contact_geometry_release_products_plus_clearance_plus_onehop":
        edge_scalars = _oldbest_plus_contact_geometry_edge_scalars(
            node_coords, node_feats, node_mask, c=c, include_clearance=True, include_onehop=True
        )
    elif edge_scalar_mode == "active_button_region_visible_plus_release_reach_with_product":
        scorability = _compute_active_button_region_scorability(
            node_coords, node_feats, node_mask, shooter_team
        )
        scorability = scorability.unsqueeze(1).expand(-1, N, -1, -1)
        release_reach = _compute_release_reach_spans(node_coords, node_feats, node_mask)
        release_reach = release_reach.unsqueeze(1).expand(-1, N, -1, -1)
        product = scorability * release_reach
        edge_scalars = _pad_edge_scalars(scorability, release_reach, product)
    elif edge_scalar_mode == "button_visible_release_reach_takeout_product":
        scorability = _compute_unoccluded_angular_spans(node_coords, node_feats, node_mask)
        scorability = scorability.unsqueeze(1).expand(-1, N, -1, -1)
        release_reach = _compute_release_reach_spans(node_coords, node_feats, node_mask)
        release_reach = release_reach.unsqueeze(1).expand(-1, N, -1, -1)
        takeout = _compute_takeoutability_spans(node_coords, node_feats, node_mask)
        takeout = takeout.unsqueeze(1).expand(-1, N, -1, -1)
        edge_scalars = _pad_edge_scalars(
            release_reach,
            scorability,
            takeout,
            release_reach * scorability,
            release_reach * takeout,
        )
    elif edge_scalar_mode == "button_visible_release_reach_takeout_only":
        scorability = _compute_unoccluded_angular_spans(node_coords, node_feats, node_mask)
        scorability = scorability.unsqueeze(1).expand(-1, N, -1, -1)
        release_reach = _compute_release_reach_spans(node_coords, node_feats, node_mask)
        release_reach = release_reach.unsqueeze(1).expand(-1, N, -1, -1)
        takeout = _compute_takeoutability_spans(node_coords, node_feats, node_mask)
        takeout = takeout.unsqueeze(1).expand(-1, N, -1, -1)
        edge_scalars = _pad_edge_scalars(
            release_reach,
            scorability,
            takeout,
            release_reach * scorability,
        )
    elif edge_scalar_mode == "button_visible_release_reach_takeout_products_only":
        scorability = _compute_unoccluded_angular_spans(node_coords, node_feats, node_mask)
        scorability = scorability.unsqueeze(1).expand(-1, N, -1, -1)
        release_reach = _compute_release_reach_spans(node_coords, node_feats, node_mask)
        release_reach = release_reach.unsqueeze(1).expand(-1, N, -1, -1)
        takeout = _compute_takeoutability_spans(node_coords, node_feats, node_mask)
        takeout = takeout.unsqueeze(1).expand(-1, N, -1, -1)
        edge_scalars = _pad_edge_scalars(
            release_reach,
            scorability,
            release_reach * scorability,
            release_reach * takeout,
        )
    elif edge_scalar_mode == "minkowski_scoring_triplet":
        scorability = _compute_minkowski_scorability_spans(
            node_coords, node_feats, node_mask, shooter_team
        )
        scorability = scorability.unsqueeze(1).expand(-1, N, -1, -1)
        reachability = _compute_minkowski_release_reach_spans(node_coords, node_feats, node_mask)
        reachability = reachability.unsqueeze(1).expand(-1, N, -1, -1)
        scoring_reachability = _compute_minkowski_scoring_reachability_spans(
            node_coords, node_feats, node_mask, shooter_team
        )
        scoring_reachability = scoring_reachability.unsqueeze(1).expand(-1, N, -1, -1)
        edge_scalars = _pad_edge_scalars(scorability, reachability, scoring_reachability)
    elif edge_scalar_mode == "minkowski_scoring_takeout_quintet":
        scorability = _compute_minkowski_scorability_spans(
            node_coords, node_feats, node_mask, shooter_team
        )
        scorability = scorability.unsqueeze(1).expand(-1, N, -1, -1)
        reachability = _compute_minkowski_release_reach_spans(node_coords, node_feats, node_mask)
        reachability = reachability.unsqueeze(1).expand(-1, N, -1, -1)
        scoring_reachability = _compute_minkowski_scoring_reachability_spans(
            node_coords, node_feats, node_mask, shooter_team
        )
        scoring_reachability = scoring_reachability.unsqueeze(1).expand(-1, N, -1, -1)
        takeoutability = _compute_minkowski_takeoutability_spans(node_coords, node_feats, node_mask)
        takeoutability = takeoutability.unsqueeze(1).expand(-1, N, -1, -1)
        takeout_reachability = _compute_minkowski_takeout_reachability_spans(
            node_coords, node_feats, node_mask
        )
        takeout_reachability = takeout_reachability.unsqueeze(1).expand(-1, N, -1, -1)
        edge_scalars = _pad_edge_scalars(
            scorability, reachability, scoring_reachability, takeoutability, takeout_reachability
        )
    elif edge_scalar_mode == "button_visible_times_release_reach":
        button_visible = _compute_unoccluded_angular_spans(node_coords, node_feats, node_mask)
        button_visible = button_visible.unsqueeze(1).expand(-1, N, -1, -1)
        release_reach = _compute_release_reach_spans(node_coords, node_feats, node_mask)
        release_reach = release_reach.unsqueeze(1).expand(-1, N, -1, -1)
        product = button_visible * release_reach
        edge_scalars = _pad_edge_scalars(product)
    elif edge_scalar_mode == "exact_line_clearance":
        primary = _compute_pairwise_line_clearance(node_coords, node_feats, node_mask)
        edge_scalars = _pad_edge_scalars(primary)
    elif edge_scalar_mode == "pairwise_unoccluded_span":
        primary = _compute_pairwise_unoccluded_angular_spans(node_coords, node_feats, node_mask)
        primary = primary.expand(-1, N, -1, -1)
        edge_scalars = _pad_edge_scalars(primary)
    elif edge_scalar_mode == "button_visible_span":
        primary = _compute_unoccluded_angular_spans(node_coords, node_feats, node_mask)
        primary = primary.unsqueeze(1).expand(-1, N, -1, -1)
        edge_scalars = _pad_edge_scalars(primary)
    else:
        raise AssertionError(f"Unhandled edge scalar mode: {edge_scalar_mode}")

    edge_feats = torch.cat([dx, dist, same_team, edge_scalars], dim=-1)  # (B, N, N, 9)

    # Mask
    mask_2d = (node_mask.unsqueeze(2) & node_mask.unsqueeze(1)).unsqueeze(-1)
    edge_feats = edge_feats * mask_2d.float()

    if edge_scalar_mode in {
        "button_visible_plus_curl_arc_reach_clean",
        "button_visible_plus_curl_arc_minkowski_straight_reach_clean",
    } or _is_contact_arc_edge_mode(edge_scalar_mode):
        is_landmark = (node_feats[:, :, 4] > 0.5) & node_mask
        landmark_pair = is_landmark.unsqueeze(2) & is_landmark.unsqueeze(1)
        edge_feats = edge_feats * (~landmark_pair).unsqueeze(-1).float()

    if _resolve_edge_prune_mode() == "stone_pair_zero_pairwise_span":
        pairwise = _compute_pairwise_unoccluded_angular_spans(node_coords, node_feats, node_mask).squeeze(-1)
        is_live = (node_feats[:, :, 3] > 0.5) & node_mask
        stone_pair = (is_live.unsqueeze(2) & is_live.unsqueeze(1))
        keep = (~stone_pair) | (pairwise > 1e-8)
        edge_feats = edge_feats * keep.unsqueeze(-1).float()

    return edge_feats


# ─────────────────────────────────────────────────────────────────────
# Fast versions of EGNN and Graph Transformer using vectorized builder
# ─────────────────────────────────────────────────────────────────────

class ValueEGNNFast(nn.Module):
    """Vectorized EGNN - same architecture but uses fast graph builder."""

    def __init__(self, input_dim=24, cond_dim=3, hidden_dim=128,
                 n_layers=3, n_heads=4, dropout=0.1, **kwargs):
        super().__init__()
        self.input_dim = input_dim
        self.cond_dim = cond_dim
        self.hidden_dim = hidden_dim

        self.node_proj = nn.Linear(NODE_FEAT_DIM, hidden_dim)

        self.layers = nn.ModuleList([
            EGNNLayer(hidden_dim, edge_feat_dim=EDGE_FEAT_DIM,
                      dropout=dropout, update_coords=(i < n_layers - 1))
            for i in range(n_layers)
        ])

        self.cond_proj = nn.Linear(cond_dim, hidden_dim)

        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x, c):
        B = x.size(0)
        device = x.device

        # Build graph (vectorized)
        node_feats, node_coords, node_mask, n_nodes = build_graph_batch_fast(x, device)

        # Compute edge features
        edge_feats = compute_edge_features_fast(node_coords, node_feats, node_mask, c=c)

        # Project node features
        h = self.node_proj(node_feats)
        h = h * node_mask.unsqueeze(-1).float()

        coords = node_coords.clone()

        # Message passing
        for layer in self.layers:
            h, coords = layer(h, coords, edge_feats, node_mask)
            # Recompute edge features with updated coords
            edge_feats = compute_edge_features_fast(coords, node_feats, node_mask, c=c)

        # Masked mean pooling
        mask_f = node_mask.unsqueeze(-1).float()
        h_pooled = (h * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)

        c_proj = self.cond_proj(c)
        combined = torch.cat([h_pooled, c_proj], dim=-1)
        return self.value_head(combined)


class ValueGraphTransformerFast(nn.Module):
    """Vectorized Graph Transformer - same architecture but uses fast graph builder."""

    def __init__(self, input_dim=24, cond_dim=3, hidden_dim=128,
                 n_layers=3, n_heads=4, dropout=0.1, **kwargs):
        super().__init__()
        self.input_dim = input_dim
        self.cond_dim = cond_dim
        self.hidden_dim = hidden_dim

        self.node_proj = nn.Linear(NODE_FEAT_DIM, hidden_dim)
        self.cond_proj = nn.Linear(cond_dim, hidden_dim)
        self.global_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

        self.layers = nn.ModuleList([
            GraphTransformerLayer(hidden_dim, n_heads=n_heads,
                                 edge_feat_dim=EDGE_FEAT_DIM, dropout=dropout)
            for _ in range(n_layers)
        ])

        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x, c):
        B = x.size(0)
        device = x.device

        # Build graph (vectorized)
        node_feats, node_coords, node_mask, n_nodes = build_graph_batch_fast(x, device)
        max_N = node_feats.shape[1]

        # Compute edge features
        edge_feats = compute_edge_features_fast(node_coords, node_feats, node_mask, c=c)

        # Project node features
        h = self.node_proj(node_feats)
        h = h * node_mask.unsqueeze(-1).float()

        # Add global (condition) token
        global_h = self.cond_proj(c).unsqueeze(1) + self.global_token.expand(B, -1, -1)
        h = torch.cat([global_h, h], dim=1)  # (B, max_N+1, D)

        # Expand edge features for global node
        new_N = max_N + 1
        new_edge = torch.zeros(B, new_N, new_N, EDGE_FEAT_DIM, device=device)
        new_edge[:, 1:, 1:, :] = edge_feats

        new_mask = torch.zeros(B, new_N, dtype=torch.bool, device=device)
        new_mask[:, 0] = True
        new_mask[:, 1:] = node_mask

        # Message passing
        for layer in self.layers:
            h = layer(h, new_edge, new_mask)

        # Read from global token
        global_out = h[:, 0, :]
        return self.value_head(global_out)


class ValueGraphTransformerGaussianFast(nn.Module):
    """Vectorized Graph Transformer with Gaussian mean/log-variance heads."""

    def __init__(self, input_dim=24, cond_dim=3, hidden_dim=128,
                 n_layers=3, n_heads=4, dropout=0.1, min_logvar=-6.0, max_logvar=3.5, **kwargs):
        super().__init__()
        self.input_dim = input_dim
        self.cond_dim = cond_dim
        self.hidden_dim = hidden_dim
        self.min_logvar = float(min_logvar)
        self.max_logvar = float(max_logvar)

        self.node_proj = nn.Linear(NODE_FEAT_DIM, hidden_dim)
        self.cond_proj = nn.Linear(cond_dim, hidden_dim)
        self.global_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

        self.layers = nn.ModuleList([
            GraphTransformerLayer(hidden_dim, n_heads=n_heads,
                                 edge_feat_dim=EDGE_FEAT_DIM, dropout=dropout)
            for _ in range(n_layers)
        ])

        self.mean_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.logvar_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x, c):
        B = x.size(0)
        device = x.device

        node_feats, node_coords, node_mask, n_nodes = build_graph_batch_fast(x, device)
        max_N = node_feats.shape[1]
        edge_feats = compute_edge_features_fast(node_coords, node_feats, node_mask, c=c)

        return self.forward_precomputed(node_feats, edge_feats, node_mask, c)

    def forward_precomputed(self, node_feats, edge_feats, node_mask, c):
        B = node_feats.size(0)
        device = node_feats.device
        max_N = node_feats.shape[1]

        h = self.node_proj(node_feats)
        h = h * node_mask.unsqueeze(-1).float()

        global_h = self.cond_proj(c).unsqueeze(1) + self.global_token.expand(B, -1, -1)
        h = torch.cat([global_h, h], dim=1)

        new_N = max_N + 1
        new_edge = torch.zeros(B, new_N, new_N, EDGE_FEAT_DIM, device=device)
        new_edge[:, 1:, 1:, :] = edge_feats

        new_mask = torch.zeros(B, new_N, dtype=torch.bool, device=device)
        new_mask[:, 0] = True
        new_mask[:, 1:] = node_mask

        for layer in self.layers:
            h = layer(h, new_edge, new_mask)

        global_out = h[:, 0, :]
        mean = self.mean_head(global_out)
        logvar = self.logvar_head(global_out).clamp(self.min_logvar, self.max_logvar)
        return mean, logvar


class ValueGraphTransformerGaussianPrecomputed(ValueGraphTransformerGaussianFast):
    """Gaussian Graph Transformer that consumes precomputed node/edge tensors."""

    def forward(self, node_feats, edge_feats, node_mask, c):
        return self.forward_precomputed(node_feats, edge_feats, node_mask, c)


# ─────────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────────

GNN_REGISTRY = {
    "egnn": ValueEGNNFast,
    "graph_transformer": ValueGraphTransformerFast,
    "graph_transformer_gaussian": ValueGraphTransformerGaussianFast,
    "graph_transformer_gaussian_precomputed": ValueGraphTransformerGaussianPrecomputed,
}
