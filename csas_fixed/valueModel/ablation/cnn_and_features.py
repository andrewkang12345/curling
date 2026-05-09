"""
CNN architectures and curling-specific feature engineering.

CNNs: Rasterize stone positions onto a 2D grid image, then use convolutions.
Features: Hand-crafted features based on curling rules and spatial relationships.
"""

import math
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

NUM_STONES = 12

# Curling sheet geometry (normalized coords, POS_MAX=4095)
BUTTON_X = 750.0 / 4095.0
BUTTON_Y = 800.0 / 4095.0
HOUSE_RADIUS_4FT = 200.0 / 4095.0   # 4-foot ring
HOUSE_RADIUS_8FT = 400.0 / 4095.0   # 8-foot ring
HOUSE_RADIUS_12FT = 600.0 / 4095.0  # 12-foot ring (house boundary)
BACKLINE_Y = 200.0 / 4095.0
HOGLINE_Y = 2900.0 / 4095.0
TEELINE_Y = BUTTON_Y


# ─────────────────────────────────────────────────────────────
# Curling Feature Engineering
# ─────────────────────────────────────────────────────────────

def compute_curling_features(x):
    """
    Compute hand-crafted curling features from stone positions.

    Input: x (B, 24) — normalized stone positions
    Output: (B, n_features) tensor of curling features

    Features computed:
    1. Per-team: n_inplay, n_inhouse, closest_dist_to_button, mean_dist,
       n_in_4ft, n_in_8ft, stone_spread (std of positions)
    2. Cross-team: n_counting_A, n_counting_B, shot_rock_team,
       closest_gap (difference in closest distances)
    3. Spatial: n_guards_A, n_guards_B (stones between tee and hog, outside house)
    """
    B = x.size(0)
    device = x.device
    stones = x.view(B, NUM_STONES, 2)

    button = torch.tensor([BUTTON_X, BUTTON_Y], device=device)

    # In-play mask: not (0,0) and not (1,1)
    is_inplay = ((stones.sum(dim=-1) > 0.001) &
                 (stones.max(dim=-1).values < 0.999))  # (B, 12)

    # Distance to button for each stone
    dist_to_button = torch.norm(stones - button, dim=-1)  # (B, 12)
    # Set out-of-play stones to large distance
    dist_to_button = torch.where(is_inplay, dist_to_button, torch.tensor(99.0, device=device))

    # Team masks
    team_a_mask = torch.zeros(NUM_STONES, dtype=torch.bool, device=device)
    team_a_mask[:6] = True
    team_b_mask = ~team_a_mask

    features = []

    for team_mask, team_name in [(team_a_mask, 'a'), (team_b_mask, 'b')]:
        tm = team_mask.unsqueeze(0).expand(B, -1)  # (B, 12)
        ip = is_inplay & tm  # (B, 12)

        # Number in play
        n_inplay = ip.float().sum(dim=-1, keepdim=True)  # (B, 1)

        # Number in house
        in_house = ip & (dist_to_button < HOUSE_RADIUS_12FT)
        n_inhouse = in_house.float().sum(dim=-1, keepdim=True)

        # Number in 4ft and 8ft rings
        in_4ft = ip & (dist_to_button < HOUSE_RADIUS_4FT)
        in_8ft = ip & (dist_to_button < HOUSE_RADIUS_8FT)
        n_in4ft = in_4ft.float().sum(dim=-1, keepdim=True)
        n_in8ft = in_8ft.float().sum(dim=-1, keepdim=True)

        # Closest distance to button (for this team)
        team_dists = torch.where(ip, dist_to_button, torch.tensor(99.0, device=device))
        closest_dist = team_dists.min(dim=-1, keepdim=True).values  # (B, 1)
        closest_dist = torch.clamp(closest_dist, max=1.0)

        # Mean distance of in-play stones
        mean_dist = (dist_to_button * ip.float()).sum(dim=-1, keepdim=True) / (n_inplay + 1e-8)
        mean_dist = torch.clamp(mean_dist, max=1.0)

        # Stone spread (std of positions for in-play stones)
        # Use distance from centroid
        ip_f = ip.float().unsqueeze(-1)  # (B, 12, 1)
        centroid = (stones * ip_f).sum(dim=1, keepdim=True) / (ip_f.sum(dim=1, keepdim=True) + 1e-8)
        diffs = stones - centroid  # (B, 12, 2)
        sq_dists = (diffs ** 2).sum(dim=-1)  # (B, 12)
        spread = ((sq_dists * ip.float()).sum(dim=-1, keepdim=True) / (n_inplay + 1e-8)).sqrt()

        # Guards: stones between tee line and hog line, outside the house
        y_coords = stones[:, :, 1]
        is_guard = ip & (y_coords > TEELINE_Y) & (y_coords < HOGLINE_Y) & (dist_to_button >= HOUSE_RADIUS_12FT)
        n_guards = is_guard.float().sum(dim=-1, keepdim=True)

        features.extend([n_inplay, n_inhouse, n_in4ft, n_in8ft,
                         closest_dist, mean_dist, spread, n_guards])

    # Cross-team features
    a_ip = is_inplay & team_a_mask.unsqueeze(0)
    b_ip = is_inplay & team_b_mask.unsqueeze(0)
    a_dists = torch.where(a_ip, dist_to_button, torch.tensor(99.0, device=device))
    b_dists = torch.where(b_ip, dist_to_button, torch.tensor(99.0, device=device))
    closest_a = a_dists.min(dim=-1).values  # (B,)
    closest_b = b_dists.min(dim=-1).values

    # Shot rock team: -1 if A closer, +1 if B closer, 0 if tied/none
    shot_rock = torch.sign(closest_a - closest_b).unsqueeze(-1)  # (B, 1)

    # Closest gap
    closest_gap = (closest_a - closest_b).unsqueeze(-1)  # (B, 1) positive = B advantage

    # Counting stones: how many of the winning team's stones are closer than the losing team's closest
    # For team A counting: A stones closer than B's closest
    n_counting_a = (a_ip & (dist_to_button < closest_b.unsqueeze(-1))).float().sum(dim=-1, keepdim=True)
    n_counting_b = (b_ip & (dist_to_button < closest_a.unsqueeze(-1))).float().sum(dim=-1, keepdim=True)

    features.extend([shot_rock, closest_gap, n_counting_a, n_counting_b])

    return torch.cat(features, dim=-1)  # (B, 20)


CURLING_FEAT_DIM = 20  # 8 per team × 2 + 4 cross-team


# ─────────────────────────────────────────────────────────────
# CNN: Grid rasterization
# ─────────────────────────────────────────────────────────────

def rasterize_stones(x, grid_size=32):
    """
    Rasterize stone positions onto a multi-channel 2D grid.

    Channels:
      0: Team A stones (Gaussian blobs)
      1: Team B stones (Gaussian blobs)
      2: All stones combined

    Input: x (B, 24) normalized positions
    Output: (B, 3, grid_size, grid_size)
    """
    B = x.size(0)
    device = x.device
    stones = x.view(B, NUM_STONES, 2)

    # Create coordinate grids
    grid_y = torch.linspace(0, 1, grid_size, device=device)
    grid_x = torch.linspace(0, 1, grid_size, device=device)
    gy, gx = torch.meshgrid(grid_y, grid_x, indexing='ij')  # (H, W)

    sigma = 1.5 / grid_size  # Gaussian spread relative to grid

    channels = torch.zeros(B, 3, grid_size, grid_size, device=device)

    for i in range(NUM_STONES):
        sx = stones[:, i, 0]  # (B,)
        sy = stones[:, i, 1]  # (B,)

        # In-play check
        is_ip = ((sx > 0.001) | (sy > 0.001)) & (sx < 0.999) & (sy < 0.999)  # (B,)

        # Gaussian blob: exp(-((gx-sx)^2 + (gy-sy)^2) / (2*sigma^2))
        dx = gx.unsqueeze(0) - sx.view(B, 1, 1)  # (B, H, W)
        dy = gy.unsqueeze(0) - sy.view(B, 1, 1)
        blob = torch.exp(-(dx**2 + dy**2) / (2 * sigma**2))  # (B, H, W)
        blob = blob * is_ip.view(B, 1, 1).float()

        team_ch = 0 if i < 6 else 1
        channels[:, team_ch] += blob
        channels[:, 2] += blob

    return channels


class ValueGridCNN(nn.Module):
    """
    Rasterize stones onto a 2D grid, apply CNN, combine with conditions.

    Supports variable grid_size and depth.
    """

    def __init__(self, input_dim=24, cond_dim=3, hidden_dim=128,
                 grid_size=32, n_conv_layers=4, dropout=0.1, **kwargs):
        super().__init__()
        self.input_dim = input_dim
        self.cond_dim = cond_dim
        self.grid_size = grid_size

        # CNN on rasterized grid
        in_ch = 3  # team_a, team_b, combined
        layers = []
        ch = in_ch
        for i in range(n_conv_layers):
            out_ch = hidden_dim // (2 ** max(0, n_conv_layers - 1 - i))
            out_ch = max(16, out_ch)
            layers.append(nn.Conv2d(ch, out_ch, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.GELU())
            if i < n_conv_layers - 1:
                layers.append(nn.MaxPool2d(2))
            ch = out_ch
        self.cnn = nn.Sequential(*layers)

        # Calculate flattened size
        with torch.no_grad():
            dummy = torch.zeros(1, 3, grid_size, grid_size)
            flat_size = self.cnn(dummy).view(1, -1).size(1)

        self.fc = nn.Sequential(
            nn.Linear(flat_size + cond_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x, c):
        grid = rasterize_stones(x, self.grid_size)  # (B, 3, H, W)
        features = self.cnn(grid).flatten(1)          # (B, flat)
        combined = torch.cat([features, c], dim=-1)
        return self.fc(combined)


class ValueFineCNN(nn.Module):
    """
    Higher-resolution CNN (64x64 grid) with deeper network.
    Focuses on the house area with a zoomed-in view.
    """

    def __init__(self, input_dim=24, cond_dim=3, hidden_dim=128,
                 grid_size=48, n_conv_layers=5, dropout=0.1, **kwargs):
        super().__init__()
        self.input_dim = input_dim
        self.cond_dim = cond_dim
        self.grid_size = grid_size

        in_ch = 3
        layers = []
        ch = in_ch
        for i in range(n_conv_layers):
            out_ch = min(hidden_dim, 32 * (2 ** i))
            layers.append(nn.Conv2d(ch, out_ch, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.GELU())
            if i < n_conv_layers - 1:
                layers.append(nn.MaxPool2d(2))
            ch = out_ch
        layers.append(nn.AdaptiveAvgPool2d(1))
        self.cnn = nn.Sequential(*layers)

        self.fc = nn.Sequential(
            nn.Linear(ch + cond_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x, c):
        grid = rasterize_stones(x, self.grid_size)
        features = self.cnn(grid).flatten(1)
        return self.fc(torch.cat([features, c], dim=-1))


class Value1DCNN(nn.Module):
    """
    1D CNN treating stones as a sequence of (x,y) pairs.
    Each stone is a "timestep" with 2 channels.
    """

    def __init__(self, input_dim=24, cond_dim=3, hidden_dim=128,
                 n_conv_layers=3, dropout=0.1, **kwargs):
        super().__init__()
        self.input_dim = input_dim
        self.cond_dim = cond_dim

        # Add team indicator as 3rd channel
        in_ch = 3  # x, y, team_indicator

        layers = []
        ch = in_ch
        for i in range(n_conv_layers):
            out_ch = hidden_dim // (2 ** max(0, n_conv_layers - 1 - i))
            out_ch = max(32, out_ch)
            layers.append(nn.Conv1d(ch, out_ch, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm1d(out_ch))
            layers.append(nn.GELU())
            ch = out_ch
        self.conv = nn.Sequential(*layers)

        self.fc = nn.Sequential(
            nn.Linear(ch + cond_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x, c):
        B = x.size(0)
        stones = x.view(B, NUM_STONES, 2)  # (B, 12, 2)

        # Add team channel
        team = torch.zeros(B, NUM_STONES, 1, device=x.device)
        team[:, 6:, :] = 1.0
        seq = torch.cat([stones, team], dim=-1)  # (B, 12, 3)
        seq = seq.transpose(1, 2)  # (B, 3, 12) for Conv1d

        features = self.conv(seq)  # (B, ch, 12)
        pooled = features.mean(dim=-1)  # (B, ch) global average pool

        return self.fc(torch.cat([pooled, c], dim=-1))


# ─────────────────────────────────────────────────────────────
# Feature-enhanced models
# ─────────────────────────────────────────────────────────────

class ValueFeatureMLP(nn.Module):
    """MLP with hand-crafted curling features concatenated to input."""

    def __init__(self, input_dim=24, cond_dim=3, hidden_dim=256,
                 n_layers=4, dropout=0.1, **kwargs):
        super().__init__()
        self.input_dim = input_dim
        self.cond_dim = cond_dim

        total_in = input_dim + cond_dim + CURLING_FEAT_DIM  # 24 + 4 + 20 = 48

        layers = []
        prev = total_in
        for _ in range(n_layers):
            layers += [nn.Linear(prev, hidden_dim), nn.GELU(), nn.Dropout(dropout)]
            prev = hidden_dim
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x, c):
        feats = compute_curling_features(x)
        return self.net(torch.cat([x, c, feats], dim=-1))


class ValueFeatureTransformer(nn.Module):
    """
    Transformer with curling features injected into the global token.
    Combines physics-informed per-stone features with global curling features.
    Uses team embeddings (no stone-index embeddings).
    """

    def __init__(self, input_dim=24, cond_dim=3, hidden_dim=256,
                 n_layers=4, n_heads=4, dropout=0.1, **kwargs):
        super().__init__()
        self.input_dim = input_dim
        self.cond_dim = cond_dim
        d = hidden_dim

        # Per-stone: (x, y, dist_button, angle, in_play, in_house) = 6
        self.stone_proj = nn.Linear(6, d)
        # Global: cond (4) + curling features (20) = 24
        self.global_proj = nn.Linear(cond_dim + CURLING_FEAT_DIM, d)

        self.team_embed = nn.Embedding(2, d)
        self.global_token = nn.Parameter(torch.randn(1, 1, d))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=n_heads, dim_feedforward=d * 4,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        self.value_head = nn.Sequential(
            nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1),
        )

    def _per_stone_features(self, x):
        B = x.size(0)
        stones = x.view(B, NUM_STONES, 2)
        button = torch.tensor([BUTTON_X, BUTTON_Y], device=x.device)
        delta = stones - button
        dist = torch.norm(delta, dim=-1, keepdim=True)
        angle = torch.atan2(delta[..., 0:1], delta[..., 1:2]) / math.pi
        is_ip = ((stones.sum(-1, keepdim=True) > 0.001) &
                 (stones.max(-1, keepdim=True).values < 0.999)).float()
        in_house = (dist < HOUSE_RADIUS_12FT).float()
        return torch.cat([stones, dist, angle, is_ip, in_house], dim=-1)

    def forward(self, x, c):
        B = x.size(0)
        device = x.device

        stone_feats = self._per_stone_features(x)
        tokens = self.stone_proj(stone_feats)

        team_ids = torch.zeros(NUM_STONES, dtype=torch.long, device=device)
        team_ids[6:] = 1
        tokens = tokens + self.team_embed(team_ids).unsqueeze(0)

        curling_feats = compute_curling_features(x)
        global_in = torch.cat([c, curling_feats], dim=-1)
        global_tok = self.global_proj(global_in).unsqueeze(1) + self.global_token.expand(B, -1, -1)

        all_tokens = torch.cat([global_tok, tokens], dim=1)
        out = self.encoder(all_tokens)
        return self.value_head(out[:, 0, :])


class ValueFeatureXGBInput(nn.Module):
    """
    Not a real model — just a feature extractor that outputs
    [positions, conditions, curling_features] for XGBoost.
    Used by the training loop to generate enhanced XGB inputs.
    """

    def __init__(self, **kwargs):
        super().__init__()

    def extract(self, x, c):
        feats = compute_curling_features(x)
        return torch.cat([x, c, feats], dim=-1)


# ─────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────

CNN_REGISTRY = {
    "grid_cnn": ValueGridCNN,
    "fine_cnn": ValueFineCNN,
    "cnn_1d": Value1DCNN,
    "feat_mlp": ValueFeatureMLP,
    "feat_transformer": ValueFeatureTransformer,
}
