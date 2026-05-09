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
EDGE_SCALAR_DIM = 5
TAKEOUT_OFFSET_NORM = 4.0 * STONE_RADIUS_NORM


def _resolve_edge_scalar_mode():
    mode = os.environ.get("GNN_EDGE_SCALAR_MODE", "thrower_masked_button_region_span").strip()
    allowed = {
        "thrower_masked_button_region_span",
        "button_visible_plus_thrower_masked_span",
        "button_visible_plus_release_reach_span",
        "button_visible_plus_release_reach_with_product",
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


def _resolve_node_feature_mode():
    mode = os.environ.get("GNN_NODE_FEATURE_MODE", "none").strip()
    allowed = {
        "none",
        "button_visible_span",
        "release_reach_times_takeout",
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
    }
    if mode not in allowed:
        raise ValueError(f"Unsupported GNN_RELEASE_NODE_MODE={mode!r}")
    return mode


def _get_active_landmarks(device=None, dtype=None):
    landmarks = LANDMARKS
    if _resolve_release_node_mode() == "single":
        landmarks = landmarks[[0, 2]]
    if device is not None or dtype is not None:
        landmarks = landmarks.to(
            device=device if device is not None else landmarks.device,
            dtype=dtype if dtype is not None else landmarks.dtype,
        )
    return landmarks


def _get_release_points(device=None, dtype=None):
    return _get_active_landmarks(device=device, dtype=dtype)[1:, :]


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

    if _resolve_edge_prune_mode() == "stone_pair_zero_pairwise_span":
        pairwise = _compute_pairwise_unoccluded_angular_spans(node_coords, node_feats, node_mask).squeeze(-1)
        is_live = (node_feats[:, :, 3] > 0.5) & node_mask
        stone_pair = (is_live.unsqueeze(2) & is_live.unsqueeze(1))
        keep = (~stone_pair) | (pairwise > 1e-8)
        edge_feats = edge_feats * keep.unsqueeze(-1).float()

    return edge_feats


EDGE_FEAT_DIM = 9  # dx, dy, dist, same_team, edge_scalar_1..edge_scalar_5


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


# ─────────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────────

GNN_REGISTRY = {
    "egnn": ValueEGNNFast,
    "graph_transformer": ValueGraphTransformerFast,
    "graph_transformer_gaussian": ValueGraphTransformerGaussianFast,
}
