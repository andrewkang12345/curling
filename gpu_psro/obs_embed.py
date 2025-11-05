from __future__ import annotations
import torch

"""
Embeds env obs into a flat vector with ONLY:
 - Per-stone (x, y, owner_one_hot[team0, team1]) for ALIVE stones; dead stones -> zeros
 - Global flags: [hammer_flag, placement_code, turn_idx, powerplay_flag]

Shapes:
  pos   : [B, N, 2]
  alive : [B, N, 1]  (1.0 alive, 0.0 dead)
  team  : [B, N, 1]  (0.0 for team0, 1.0 for team1)
  aux   : [B, ?]     (we will recompute the subset we need from the raw obs)

Output dim:
  per-stone: 2 (pos) + 2 (owner_one_hot) = 4
  N stones (N=16) => 64
  + 4 flags => 68
"""

def embed_obs(obs: dict) -> torch.Tensor:
    pos   = obs["pos"]     # [B,N,2]
    alive = obs["alive"]   # [B,N,1]
    team  = obs["team"]    # [B,N,1], 0.0/1.0

    B, N, _ = pos.shape
    device = pos.device

    # Owner one-hot (team0, team1)
    t1 = team[..., 0]                                  # [B,N]
    owner_t1 = t1
    owner_t0 = 1.0 - t1
    owner = torch.stack([owner_t0, owner_t1], dim=-1)  # [B,N,2]

    # Mask out dead stones (zeros where dead)
    alive_mask = alive      # [B,N,1]
    pos_alive   = pos * alive_mask
    owner_alive = owner * alive_mask

    per_stone = torch.cat([pos_alive, owner_alive], dim=-1)  # [B,N,4]
    flat_stones = per_stone.reshape(B, -1)                    # [B, N*4]

    # Build the four requested flags
    # We re-use what's already in env.aux, but take exactly the fields you want:
    # aux from env has (in the sim) a superset; we reconstruct the subset explicitly to be safe.
    #   hammer_flag   : 1.0 if team0 has hammer else 0.0  (already encoded in env.aux as (hammer_team==0))
    #   placement_code: 0=A, 1=B, 2=PP_left, 3=PP_right
    #   turn_idx      : integer index of the number of throws so far in the end
    #   powerplay_flag: 1.0 if using PP_left/right, else 0.0
    aux_full = obs["aux"]    # [B, 7] in the sim; fields are [cur_team, t0_rocks, t1_rocks, hammer_flag, placement_code, turn_idx, powerplay_flag]
    hammer_flag    = aux_full[:, 3:4]
    placement_code = aux_full[:, 4:5]
    turn_idx       = aux_full[:, 5:6]
    powerplay_flag = aux_full[:, 6:7]
    flags = torch.cat([hammer_flag, placement_code, turn_idx, powerplay_flag], dim=-1)  # [B,4]

    return torch.cat([flat_stones, flags], dim=-1)  # [B, 68]
