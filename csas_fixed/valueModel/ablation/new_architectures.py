"""
New architecture variants for curling value prediction.

All models follow the same interface:
    forward(x, c) -> (B, 1)
where x is (B, 24) normalized stone positions and c is (B, 4) conditions.
"""

import math
import torch
from torch import nn
import torch.nn.functional as F

NUM_STONES = 12
BUTTON_X = 750.0 / 4095.0  # normalized button position
BUTTON_Y = 800.0 / 4095.0
HOUSE_RADIUS = 600.0 / 4095.0  # ~6 feet in normalized coords
POS_MAX = 4095.0
RELEASE_Y = 2900.0 / 4095.0
STONE_RADIUS_NORM = 0.012


def _wrap_angle(delta: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(delta), torch.cos(delta))


def _live_mask(stones: torch.Tensor) -> torch.Tensor:
    return (stones.sum(dim=-1) > 0.001) & (stones.max(dim=-1).values < 0.999)


def _visible_span_from_viewpoint(stones: torch.Tensor, viewpoint_xy: torch.Tensor) -> torch.Tensor:
    """
    Exact visible angular span of each live stone as seen from a single fixed
    viewpoint. Non-live stones receive 0.
    """
    B, N, _ = stones.shape
    device = stones.device
    dtype = stones.dtype
    is_live = _live_mask(stones)

    delta = stones - viewpoint_xy.view(1, 1, 2)
    dist = torch.norm(delta, dim=-1).clamp(min=STONE_RADIUS_NORM + 1e-6)
    angles = torch.atan2(delta[:, :, 0], delta[:, :, 1])
    half_angles = torch.asin((STONE_RADIUS_NORM / dist).clamp(max=1.0 - 1e-6))

    spans = torch.zeros(B, N, device=device, dtype=dtype)
    neg_inf_col = torch.full((B, 1), float("-inf"), device=device, dtype=dtype)
    pos_inf = torch.full_like(angles, float("inf"))
    neg_inf = torch.full_like(angles, float("-inf"))

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
    return spans.unsqueeze(-1)


def _release_center_reachability(stones: torch.Tensor) -> torch.Tensor:
    """
    Per-stone reachability from the release-center viewpoint: exact visible span
    from the release point with the backward (-x) half-plane masked out.
    """
    release_xy = stones.new_tensor([BUTTON_X, RELEASE_Y])
    spans = _visible_span_from_viewpoint(stones, release_xy)
    backward = (stones[..., 0] - release_xy[0]) < 0.0
    spans = torch.where(backward.unsqueeze(-1), torch.zeros_like(spans), spans)
    return spans


# ─────────────────────────────────────────────────────────────
# 1. MLP — strong tabular baseline
# ─────────────────────────────────────────────────────────────

class ValueMLP(nn.Module):
    """Simple MLP on concatenated [positions, conditions]."""

    def __init__(self, input_dim=24, cond_dim=3, hidden_dim=256, n_layers=4, dropout=0.1, **kwargs):
        super().__init__()
        self.input_dim = input_dim
        self.cond_dim = cond_dim
        total_in = input_dim + cond_dim

        layers = []
        prev = total_in
        for i in range(n_layers):
            layers.append(nn.Linear(prev, hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            prev = hidden_dim
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x, c):
        return self.net(torch.cat([x, c], dim=-1))


# ─────────────────────────────────────────────────────────────
# 2. DeepSets — permutation-invariant per-team pooling
# ─────────────────────────────────────────────────────────────

class ValueDeepSets(nn.Module):
    """
    DeepSets: phi(stone) per stone -> mean+max pool within each team -> rho(pooled, cond) -> value.
    Respects permutation invariance within team A (stones 0-5) and team B (stones 6-11).
    """

    def __init__(self, input_dim=24, cond_dim=3, hidden_dim=128, n_layers=3, dropout=0.1, **kwargs):
        super().__init__()
        self.input_dim = input_dim
        self.cond_dim = cond_dim

        # Per-stone encoder: (x, y) -> hidden
        phi_layers = [nn.Linear(2, hidden_dim), nn.GELU(), nn.Dropout(dropout)]
        for _ in range(n_layers - 1):
            phi_layers += [nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)]
        self.phi = nn.Sequential(*phi_layers)

        # Team indicator embedding (0 = team A, 1 = team B)
        self.team_embed = nn.Embedding(2, hidden_dim)

        # After pooling: 2 teams × (mean + max) × hidden_dim + cond_dim
        pool_dim = 2 * 2 * hidden_dim + cond_dim

        # Value head
        self.rho = nn.Sequential(
            nn.Linear(pool_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x, c):
        B = x.size(0)
        stones = x.view(B, NUM_STONES, 2)  # (B, 12, 2)

        # Per-stone features
        h = self.phi(stones)  # (B, 12, hidden)

        # Add team embeddings
        team_ids = torch.zeros(NUM_STONES, dtype=torch.long, device=x.device)
        team_ids[6:] = 1
        team_emb = self.team_embed(team_ids).unsqueeze(0).expand(B, -1, -1)  # (B, 12, hidden)
        h = h + team_emb

        # Pool within teams: mean + max
        h_a = h[:, :6, :]   # Team A
        h_b = h[:, 6:, :]   # Team B

        pool_a = torch.cat([h_a.mean(dim=1), h_a.max(dim=1).values], dim=-1)  # (B, 2*hidden)
        pool_b = torch.cat([h_b.mean(dim=1), h_b.max(dim=1).values], dim=-1)  # (B, 2*hidden)

        pooled = torch.cat([pool_a, pool_b, c], dim=-1)  # (B, 4*hidden + cond_dim)
        return self.rho(pooled)


# ─────────────────────────────────────────────────────────────
# 3. SetTransformer — attention without position embeddings
# ─────────────────────────────────────────────────────────────

class ValueSetTransformer(nn.Module):
    """
    Transformer with NO stone-index embeddings — only team embeddings.
    Truly permutation-equivariant within each team.
    """

    def __init__(self, input_dim=24, cond_dim=3, hidden_dim=256,
                 n_layers=4, n_heads=4, dropout=0.1, **kwargs):
        super().__init__()
        self.input_dim = input_dim
        self.cond_dim = cond_dim
        d = hidden_dim

        self.stone_proj = nn.Linear(2, d)
        self.cond_proj = nn.Linear(cond_dim, d)

        # Only team embedding, no per-stone-index embedding
        self.team_embed = nn.Embedding(2, d)

        # In-play indicator embedding
        self.inplay_embed = nn.Embedding(2, d)

        self.global_token = nn.Parameter(torch.randn(1, 1, d))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=n_heads, dim_feedforward=d * 4,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        self.value_head = nn.Sequential(
            nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1),
        )

    def forward(self, x, c):
        B = x.size(0)
        device = x.device

        stones = x.view(B, NUM_STONES, 2)  # (B, 12, 2)
        tokens = self.stone_proj(stones)     # (B, 12, d)

        # Team embedding (0 for stones 0-5, 1 for stones 6-11)
        team_ids = torch.zeros(NUM_STONES, dtype=torch.long, device=device)
        team_ids[6:] = 1
        tokens = tokens + self.team_embed(team_ids).unsqueeze(0)

        # In-play indicator: stone is in play if coords are not (0,0) and not (1,1)
        coords = stones  # already normalized to [0,1]
        is_inplay = ((coords.sum(dim=-1) > 0.001) & (coords.max(dim=-1).values < 0.999)).long()
        tokens = tokens + self.inplay_embed(is_inplay)

        # Global token from conditions
        global_tok = self.cond_proj(c).unsqueeze(1) + self.global_token.expand(B, -1, -1)

        all_tokens = torch.cat([global_tok, tokens], dim=1)  # (B, 13, d)
        out = self.encoder(all_tokens)
        return self.value_head(out[:, 0, :])


class ValueSetTransformerGaussian(nn.Module):
    """
    SetTransformer with a Gaussian value head.

    Returns (mean, log_variance), each shaped (B, 1). This keeps the same
    permutation-invariant tokenization as ValueSetTransformer but lets downstream
    search penalize or visualize uncertainty instead of treating every point
    estimate as equally certain.
    """

    def __init__(self, input_dim=24, cond_dim=3, hidden_dim=256,
                 n_layers=4, n_heads=4, dropout=0.1, min_logvar=-6.0, max_logvar=4.0, **kwargs):
        super().__init__()
        self.input_dim = input_dim
        self.cond_dim = cond_dim
        self.min_logvar = float(min_logvar)
        self.max_logvar = float(max_logvar)
        d = hidden_dim

        self.stone_proj = nn.Linear(2, d)
        self.cond_proj = nn.Linear(cond_dim, d)
        self.team_embed = nn.Embedding(2, d)
        self.inplay_embed = nn.Embedding(2, d)
        self.global_token = nn.Parameter(torch.randn(1, 1, d))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=n_heads, dim_feedforward=d * 4,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        self.mean_head = nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, 1),
        )
        self.logvar_head = nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, 1),
        )

    def forward(self, x, c):
        B = x.size(0)
        device = x.device

        stones = x.view(B, NUM_STONES, 2)
        tokens = self.stone_proj(stones)

        team_ids = torch.zeros(NUM_STONES, dtype=torch.long, device=device)
        team_ids[6:] = 1
        tokens = tokens + self.team_embed(team_ids).unsqueeze(0)

        is_inplay = ((stones.sum(dim=-1) > 0.001) & (stones.max(dim=-1).values < 0.999)).long()
        tokens = tokens + self.inplay_embed(is_inplay)

        global_tok = self.cond_proj(c).unsqueeze(1) + self.global_token.expand(B, -1, -1)
        out = self.encoder(torch.cat([global_tok, tokens], dim=1))[:, 0, :]
        mean = self.mean_head(out)
        logvar = self.logvar_head(out).clamp(self.min_logvar, self.max_logvar)
        return mean, logvar


class ValueSetTransformerMoE(nn.Module):
    """
    SetTransformer encoder with a gated mixture-of-experts value head on the
    global token.
    """

    def __init__(self, input_dim=24, cond_dim=3, hidden_dim=256,
                 n_layers=4, n_heads=4, dropout=0.1, n_experts=4, **kwargs):
        super().__init__()
        self.input_dim = input_dim
        self.cond_dim = cond_dim
        self.n_experts = n_experts
        d = hidden_dim

        self.stone_proj = nn.Linear(2, d)
        self.cond_proj = nn.Linear(cond_dim, d)
        self.team_embed = nn.Embedding(2, d)
        self.inplay_embed = nn.Embedding(2, d)
        self.global_token = nn.Parameter(torch.randn(1, 1, d))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=n_heads, dim_feedforward=d * 4,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        self.gate = nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, n_experts),
        )
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d, d),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d, 1),
            )
            for _ in range(n_experts)
        ])

    def forward(self, x, c):
        B = x.size(0)
        device = x.device

        stones = x.view(B, NUM_STONES, 2)
        tokens = self.stone_proj(stones)

        team_ids = torch.zeros(NUM_STONES, dtype=torch.long, device=device)
        team_ids[6:] = 1
        tokens = tokens + self.team_embed(team_ids).unsqueeze(0)

        is_inplay = ((stones.sum(dim=-1) > 0.001) & (stones.max(dim=-1).values < 0.999)).long()
        tokens = tokens + self.inplay_embed(is_inplay)

        global_tok = self.cond_proj(c).unsqueeze(1) + self.global_token.expand(B, -1, -1)
        all_tokens = torch.cat([global_tok, tokens], dim=1)
        out = self.encoder(all_tokens)
        global_out = out[:, 0, :]

        gate_logits = self.gate(global_out)
        gate_weights = F.softmax(gate_logits, dim=-1)
        expert_outputs = torch.cat([expert(global_out) for expert in self.experts], dim=-1)
        return (gate_weights * expert_outputs).sum(dim=-1, keepdim=True)


class ValueSetTransformerGeo(nn.Module):
    """
    SetTransformer with per-stone geometric augmentation:
    - scorability: visible angular span from the button
    - reachability: visible angular span from release-center throw location
    """

    def __init__(self, input_dim=24, cond_dim=3, hidden_dim=256,
                 n_layers=4, n_heads=4, dropout=0.1, **kwargs):
        super().__init__()
        self.input_dim = input_dim
        self.cond_dim = cond_dim
        d = hidden_dim

        self.stone_proj = nn.Linear(4, d)
        self.cond_proj = nn.Linear(cond_dim, d)
        self.team_embed = nn.Embedding(2, d)
        self.inplay_embed = nn.Embedding(2, d)
        self.global_token = nn.Parameter(torch.randn(1, 1, d))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=n_heads, dim_feedforward=d * 4,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        self.value_head = nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, 1),
        )

    def forward(self, x, c):
        B = x.size(0)
        device = x.device

        stones = x.view(B, NUM_STONES, 2)
        scorability = _visible_span_from_viewpoint(stones, stones.new_tensor([BUTTON_X, BUTTON_Y]))
        reachability = _release_center_reachability(stones)
        stone_feats = torch.cat([stones, scorability, reachability], dim=-1)
        tokens = self.stone_proj(stone_feats)

        team_ids = torch.zeros(NUM_STONES, dtype=torch.long, device=device)
        team_ids[6:] = 1
        tokens = tokens + self.team_embed(team_ids).unsqueeze(0)

        is_inplay = _live_mask(stones).long()
        tokens = tokens + self.inplay_embed(is_inplay)

        global_tok = self.cond_proj(c).unsqueeze(1) + self.global_token.expand(B, -1, -1)
        all_tokens = torch.cat([global_tok, tokens], dim=1)
        out = self.encoder(all_tokens)
        return self.value_head(out[:, 0, :])


# ─────────────────────────────────────────────────────────────
# 4. PairNet — explicit pairwise stone interactions
# ─────────────────────────────────────────────────────────────

class ValuePairNet(nn.Module):
    """
    Computes explicit pairwise features between all stone pairs,
    plus per-stone distance to button. Aggregates via mean pooling.
    """

    def __init__(self, input_dim=24, cond_dim=3, hidden_dim=128, n_layers=3, dropout=0.1, **kwargs):
        super().__init__()
        self.input_dim = input_dim
        self.cond_dim = cond_dim

        # Per-stone features: (x, y, dist_to_button, in_play, team)
        stone_feat_dim = 5

        # Per-stone MLP
        self.stone_mlp = nn.Sequential(
            nn.Linear(stone_feat_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
        )

        # Pairwise feature: (dist, same_team, both_inplay)
        pair_feat_dim = 3

        # Pairwise MLP
        self.pair_mlp = nn.Sequential(
            nn.Linear(pair_feat_dim + 2 * hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
        )

        # Value head: stone_pool + pair_pool + cond
        agg_dim = hidden_dim * 2 + cond_dim
        head_layers = []
        prev = agg_dim
        for _ in range(n_layers - 1):
            head_layers += [nn.Linear(prev, hidden_dim * 2), nn.GELU(), nn.Dropout(dropout)]
            prev = hidden_dim * 2
        head_layers.append(nn.Linear(prev, 1))
        self.value_head = nn.Sequential(*head_layers)

    def _compute_features(self, x):
        """Compute per-stone and pairwise features."""
        B = x.size(0)
        stones = x.view(B, NUM_STONES, 2)  # (B, 12, 2)

        # Distance to button
        button = torch.tensor([BUTTON_X, BUTTON_Y], device=x.device).view(1, 1, 2)
        dist_button = torch.norm(stones - button, dim=-1, keepdim=True)  # (B, 12, 1)

        # In-play indicator
        is_inplay = ((stones.sum(dim=-1, keepdim=True) > 0.001) &
                     (stones.max(dim=-1, keepdim=True).values < 0.999)).float()

        # Team indicator
        team = torch.zeros(B, NUM_STONES, 1, device=x.device)
        team[:, 6:, :] = 1.0

        # Per-stone features
        stone_feats = torch.cat([stones, dist_button, is_inplay, team], dim=-1)  # (B, 12, 5)

        return stones, stone_feats, is_inplay.squeeze(-1)

    def forward(self, x, c):
        B = x.size(0)
        stones, stone_feats, is_inplay = self._compute_features(x)

        # Per-stone embeddings
        h_stones = self.stone_mlp(stone_feats)  # (B, 12, hidden)

        # Pairwise features for all 66 pairs
        # Compute pairwise distances
        # stones: (B, 12, 2)
        diff = stones.unsqueeze(2) - stones.unsqueeze(1)  # (B, 12, 12, 2)
        pw_dist = torch.norm(diff, dim=-1, keepdim=True)  # (B, 12, 12, 1)

        # Same team indicator
        team_ids = torch.zeros(NUM_STONES, device=x.device)
        team_ids[6:] = 1.0
        same_team = (team_ids.unsqueeze(0) == team_ids.unsqueeze(1)).float()  # (12, 12)
        same_team = same_team.unsqueeze(0).unsqueeze(-1).expand(B, -1, -1, -1)  # (B, 12, 12, 1)

        # Both in play
        both_inplay = (is_inplay.unsqueeze(2) * is_inplay.unsqueeze(1)).unsqueeze(-1)  # (B, 12, 12, 1)

        # Concatenate: pair features + both stone embeddings
        hi = h_stones.unsqueeze(2).expand(-1, -1, NUM_STONES, -1)  # (B, 12, 12, hidden)
        hj = h_stones.unsqueeze(1).expand(-1, NUM_STONES, -1, -1)  # (B, 12, 12, hidden)
        pair_input = torch.cat([pw_dist, same_team, both_inplay, hi, hj], dim=-1)

        # Only use upper triangle (avoid duplicates)
        h_pairs = self.pair_mlp(pair_input)  # (B, 12, 12, hidden)

        # Pool: mean over all pairs and stones
        stone_pool = h_stones.mean(dim=1)  # (B, hidden)
        pair_pool = h_pairs.mean(dim=(1, 2))  # (B, hidden)

        agg = torch.cat([stone_pool, pair_pool, c], dim=-1)
        return self.value_head(agg)


# ─────────────────────────────────────────────────────────────
# 5. PhysicsTransformer — physics-informed features + transformer
# ─────────────────────────────────────────────────────────────

class ValuePhysicsTransformer(nn.Module):
    """
    Transformer that augments each stone token with physics-informed features:
    - distance to button
    - angle relative to centerline
    - whether stone is in play
    - whether stone is in the house

    Uses team embeddings instead of stone-index embeddings.
    """

    def __init__(self, input_dim=24, cond_dim=3, hidden_dim=256,
                 n_layers=4, n_heads=4, dropout=0.1, **kwargs):
        super().__init__()
        self.input_dim = input_dim
        self.cond_dim = cond_dim
        d = hidden_dim

        # Per-stone input: (x, y, dist_button, angle, in_play, in_house) = 6 features
        self.stone_proj = nn.Linear(6, d)
        self.cond_proj = nn.Linear(cond_dim, d)
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

    def _augment_stone_features(self, x):
        """Add physics features to each stone."""
        B = x.size(0)
        stones = x.view(B, NUM_STONES, 2)  # normalized coords

        button = torch.tensor([BUTTON_X, BUTTON_Y], device=x.device)

        # Distance to button
        delta = stones - button.unsqueeze(0).unsqueeze(0)
        dist = torch.norm(delta, dim=-1, keepdim=True)  # (B, 12, 1)

        # Angle relative to centerline (atan2)
        angle = torch.atan2(delta[..., 0:1], delta[..., 1:2])  # (B, 12, 1)
        angle = angle / math.pi  # normalize to [-1, 1]

        # In play
        is_inplay = ((stones.sum(dim=-1, keepdim=True) > 0.001) &
                     (stones.max(dim=-1, keepdim=True).values < 0.999)).float()

        # In house (within house radius of button)
        in_house = (dist < HOUSE_RADIUS).float()

        return torch.cat([stones, dist, angle, is_inplay, in_house], dim=-1)  # (B, 12, 6)

    def forward(self, x, c):
        B = x.size(0)
        device = x.device

        aug_stones = self._augment_stone_features(x)  # (B, 12, 6)
        tokens = self.stone_proj(aug_stones)           # (B, 12, d)

        # Team embedding
        team_ids = torch.zeros(NUM_STONES, dtype=torch.long, device=device)
        team_ids[6:] = 1
        tokens = tokens + self.team_embed(team_ids).unsqueeze(0)

        # Global token
        global_tok = self.cond_proj(c).unsqueeze(1) + self.global_token.expand(B, -1, -1)

        all_tokens = torch.cat([global_tok, tokens], dim=1)
        out = self.encoder(all_tokens)
        return self.value_head(out[:, 0, :])


# ─────────────────────────────────────────────────────────────
# 6. ResNet-style MLP with skip connections
# ─────────────────────────────────────────────────────────────

class ResBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.net(x)


class ValueResMLP(nn.Module):
    """Residual MLP with skip connections — designed for tabular data."""

    def __init__(self, input_dim=24, cond_dim=3, hidden_dim=256, n_layers=6, dropout=0.1, **kwargs):
        super().__init__()
        self.input_dim = input_dim
        self.cond_dim = cond_dim

        self.input_proj = nn.Linear(input_dim + cond_dim, hidden_dim)
        self.blocks = nn.Sequential(*[ResBlock(hidden_dim, dropout) for _ in range(n_layers)])
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x, c):
        h = self.input_proj(torch.cat([x, c], dim=-1))
        h = self.blocks(h)
        return self.head(h)


# ─────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────

ARCHITECTURE_REGISTRY = {
    "mlp": ValueMLP,
    "deepsets": ValueDeepSets,
    "set_transformer": ValueSetTransformer,
    "set_transformer_moe": ValueSetTransformerMoE,
    "set_transformer_geo": ValueSetTransformerGeo,
    "pairnet": ValuePairNet,
    "physics_transformer": ValuePhysicsTransformer,
    "resmlp": ValueResMLP,
}
