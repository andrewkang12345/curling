# torch_curling_batched.py
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict

import torch
import torch.nn.functional as F

# -----------------------------------------------------------------------------
# Configs
# -----------------------------------------------------------------------------
@dataclass
class Sheet:
    length: float = 44.5
    width: float = 4.75
    tee_x: float = 0.0
    tee_y: float = 34.75
    hog_y: float = 21.95
    backline_y: float = 39.0
    # 4ft, 8ft, 12ft radii in meters
    house_radii: Tuple[float, float, float] = (0.61, 1.22, 1.83)

STONE_R = 0.145

@dataclass
class CurlingConfig:
    # integration & physics
    dt: float = 0.02
    mu_drag: float = 0.08
    curl_gain: float = 0.12
    v_stop: float = 0.03
    restit: float = 0.4
    restit_wall: float = 0.3
    # game rules
    stones_per_team: int = 5       # mixed doubles
    team0_hammer: bool = True
    use_preplaced: bool = True
    use_powerplay: bool = True
    default_placement: str = "A"   # "A","B","PP_left","PP_right", "EMPTY", "CURRICULUM_k"
    # execution noise
    noise_speed: float = 0.10
    noise_angle: float = 0.01
    noise_spin_flip_prob: float = 0.0
    # kernel limits
    max_sim_steps: int = 20000

    # ---- New: curriculum placement cache (fix #2) ----
    curriculum_cache_per_k: int = 128
    curriculum_cache_seed: int = 1234

# -----------------------------------------------------------------------------
# Torch Vector Utilities
# -----------------------------------------------------------------------------
def _norm2(xy: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(xy, dim=-1)

def _rot90(xy: torch.Tensor) -> torch.Tensor:
    return torch.stack([-xy[..., 1], xy[..., 0]], dim=-1)

def _clamp_(x: torch.Tensor, low: float, high: float):
    x.clamp_(min=low, max=high)
    return x

# -----------------------------------------------------------------------------
# Batched Curling Simulator (all on GPU)
# -----------------------------------------------------------------------------
class TorchBatchedCurlingSim:
    def __init__(self,
                 batch_size: int,
                 device: str = "cuda",
                 cfg: Optional[CurlingConfig] = None,
                 sheet: Optional[Sheet] = None):
        self.B = batch_size
        self.device = torch.device(device)
        self.cfg = cfg or CurlingConfig()
        self.sheet = sheet or Sheet()

        self.N = 16  # max stones capacity
        self._alloc_state()

        # per-end trackers
        self.turn_idx = torch.zeros(self.B, dtype=torch.int32, device=self.device)
        self.hammer_team = (torch.zeros(self.B, dtype=torch.int64, device=self.device)
                            if self.cfg.team0_hammer else torch.ones(self.B, dtype=torch.int64, device=self.device))
        self.cur_team = 1 - self.hammer_team
        self.rocks_left = torch.full((self.B, 2), self.cfg.stones_per_team, dtype=torch.int32, device=self.device)

        # placements stored as codes
        self.placement_code = torch.zeros(self.B, dtype=torch.int64, device=self.device)

        # count of stones already in play (created so far)
        self.count_stones = torch.zeros(self.B, dtype=torch.int64, device=self.device)

        # random generator
        self.rng = torch.Generator(device=self.device)
        self.rng.manual_seed(self.cfg.curriculum_cache_seed)

        # ---- FIX #2: precompute curriculum placement cache on GPU ----
        self._curr_cache = self._build_curriculum_cache(max_k=11,
                                                        per_k=self.cfg.curriculum_cache_per_k)

        # Pre-build a triangular mask for collisions (reused)
        self._tri_mask = torch.triu(torch.ones(self.N, self.N, device=self.device, dtype=torch.bool), diagonal=1)

    # ------------------ State init ------------------
    def _alloc_state(self):
        B, N = self.B, self.N
        self.pos = torch.zeros(B, N, 2, device=self.device, dtype=torch.float32)
        self.vel = torch.zeros(B, N, 2, device=self.device, dtype=torch.float32)
        self.alive = torch.zeros(B, N, 1, device=self.device, dtype=torch.float32)  # 1 if in play
        self.spin = torch.zeros(B, N, 1, device=self.device, dtype=torch.float32)
        self.team = torch.zeros(B, N, 1, device=self.device, dtype=torch.float32)

    # ------------------ Helpers ------------------
    def _placement_to_code(self, p: str) -> int:
        """
        Built-ins:
          A=0, B=1, PP_left=2, PP_right=3, EMPTY=4
        CURRICULUM_k: return 100 + k (k clamped to [1,11])
        """
        base = {"A": 0, "B": 1, "PP_left": 2, "PP_right": 3, "EMPTY": 4}
        if p in base:
            return base[p]
        if p.startswith("CURRICULUM_"):
            try:
                k = int(p.split("_", 1)[1])
                k = max(1, min(k, 11))
                return 100 + k
            except Exception:
                pass
        return 0  # default to "A"

    def _code_to_offset(self, code: torch.Tensor) -> torch.Tensor:
        x = torch.zeros_like(code, dtype=torch.float32, device=self.device)
        x = torch.where(code == 2, torch.full_like(x, -1.83), x)  # PP_left
        x = torch.where(code == 3, torch.full_like(x, +1.83), x)  # PP_right
        return x

    def set_execution_noise(self, sigma: float):
        self.cfg.noise_speed = max(0.0, float(sigma))
        self.cfg.noise_angle = max(0.0, float(sigma) * 0.2)

    # ------------------ FIX #2: Curriculum cache ------------------
    def _build_curriculum_cache(self, max_k: int, per_k: int) -> Dict[int, torch.Tensor]:
        """
        Build, once, a cache of non-overlapping in-house placements around the tee.
        Returns dict: k -> [per_k, k, 2] positions (float32, on device).
        This avoids per-batch Python loops at reset time.
        """
        cache: Dict[int, torch.Tensor] = {}
        center = torch.tensor([self.sheet.tee_x, self.sheet.tee_y],
                              device=self.device, dtype=torch.float32)

        min_sep = 2.0 * STONE_R + 1e-3
        r_max = float(self.sheet.house_radii[0])  # 4-ft ring

        for k in range(1, max_k + 1):
            # Over-generate candidates, then greedy pack per layout.
            # We generate [per_k, k*8] candidates and greedily keep the first k non-overlapping.
            cand_mult = 8
            C = k * cand_mult
            # Sample radii ~ sqrt(u)*r_max and angles ~ U(0,2pi)
            u = torch.rand(per_k, C, device=self.device, generator=self.rng)
            r = torch.sqrt(u) * max(r_max - STONE_R, 0.01)
            t = torch.rand(per_k, C, device=self.device, generator=self.rng) * (2.0 * math.pi)
            xs = center[0] + r * torch.cos(t)
            ys = center[1] + r * torch.sin(t)
            cand = torch.stack([xs, ys], dim=-1)  # [per_k, C, 2]

            # Greedy selection without Python loops across batch: iterate over picks dimension small k (≤11)
            selected = torch.empty(per_k, k, 2, device=self.device, dtype=torch.float32)
            # Start with a large "last chosen" tensor to compare against (we'll maintain a mask)
            # We do a small loop over j in [0..k-1], which is tiny and independent of B, so OK.
            # Track availability per candidate
            available = torch.ones(per_k, C, device=self.device, dtype=torch.bool)
            # Pre-place a "very far" point so first selection simply takes the first available
            for j in range(k):
                # pick the first available candidate for each row (vectorized argmax over ~C)
                # To encourage separation, pick the candidate with the largest min-distance to already selected
                if j == 0:
                    # first: just take index 0..(per_k-1) along C dimension uniformly
                    idx = torch.randint(low=0, high=C, size=(per_k,), device=self.device, generator=self.rng)
                else:
                    prev = selected[:, :j, :]  # [per_k, j, 2]
                    diff = cand.unsqueeze(2) - prev.unsqueeze(1)  # [per_k, C, j, 2]
                    d2 = (diff * diff).sum(-1)  # [per_k, C, j]
                    min_d = torch.where(available.unsqueeze(-1), d2, torch.full_like(d2, float("inf"))).amin(dim=-1)  # [per_k, C]
                    # forbid those closer than min_sep
                    ok = (min_d >= (min_sep * min_sep)) & available
                    # choose the farthest among ok; if none ok, just take first available
                    scores = torch.where(ok, min_d, torch.full_like(min_d, -1.0))
                    idx = scores.argmax(dim=1)  # [per_k]
                    # if all were -1 (no ok), argmax picks 0; ensure it's at least available
                    fallback = torch.arange(C, device=self.device).unsqueeze(0).expand(per_k, C)
                    first_avail = torch.where(available, fallback, torch.full_like(fallback, C)).amin(dim=1).clamp(max=C-1)
                    idx = torch.where((scores.max(dim=1).values > 0), idx, first_avail)

                pick = cand[torch.arange(per_k, device=self.device), idx, :]  # [per_k, 2]
                selected[:, j, :] = pick
                # mark any candidate too close to this pick as unavailable
                d = cand - pick.unsqueeze(1)  # [per_k, C, 2]
                mask_bad = (d * d).sum(-1) < (min_sep * min_sep)  # [per_k, C]
                available = available & (~mask_bad)
            cache[k] = selected  # [per_k, k, 2]

        return cache

    # ------------------ Reset ------------------
    @torch.no_grad()
    def reset(self, placements: Optional[List[str]] = None):
        self._alloc_state()
        self.turn_idx.zero_()
        self.hammer_team = (torch.zeros(self.B, dtype=torch.int64, device=self.device)
                            if self.cfg.team0_hammer else torch.ones(self.B, dtype=torch.int64, device=self.device))
        self.cur_team = 1 - self.hammer_team
        self.rocks_left[:] = self.cfg.stones_per_team
        self.count_stones.zero_()

        if placements is None:
            code = self._placement_to_code(self.cfg.default_placement)
            self.placement_code[:] = code
        else:
            assert len(placements) == self.B
            codes = torch.tensor([self._placement_to_code(p) for p in placements],
                                 device=self.device, dtype=torch.int64)
            self.placement_code = codes

        self._place_initial()
        return self._obs()

    @torch.no_grad()
    def _place_initial(self):
        """
        Places stones based on self.placement_code.
        - "EMPTY"
        - "CURRICULUM_k" (k in [1..11]) -> use cached random stones near the button (fix #2)
        - Original codes ("A","B","PP_left","PP_right") only if cfg.use_preplaced
        """
        B = self.B
        b_ar = torch.arange(B, device=self.device)

        # --- CURRICULUM LOGIC (FIX #2: pull from cache) ---
        is_curr = (self.placement_code >= 100)
        if is_curr.any():
            codes = self.placement_code[is_curr]
            ks = (codes - 100).clamp(min=1, max=11)
            # Choose a random cached layout index per example
            for unique_k in ks.unique().tolist():
                mask_k = is_curr.clone()
                mask_k[is_curr] = (ks == unique_k)
                idxs = torch.nonzero(mask_k, as_tuple=False).flatten()
                if idxs.numel() == 0:
                    continue
                layouts = self._curr_cache[unique_k]  # [L, k, 2]
                L = layouts.shape[0]
                sel = torch.randint(low=0, high=L, size=(idxs.numel(),), device=self.device, generator=self.rng)
                chosen = layouts[sel, :, :]  # [count, k, 2]
                # capacity guard
                k_eff = min(unique_k, int(self.N))
                # write positions/flags
                for j in range(k_eff):
                    idx = (self.count_stones[mask_k] + j).long()
                    self.pos[idxs, idx, 0] = chosen[:, j, 0]
                    self.pos[idxs, idx, 1] = chosen[:, j, 1]
                    self.vel[idxs, idx, :] = 0.0
                    self.alive[idxs, idx, 0] = 1.0
                    self.spin[idxs, idx, 0] = 0.0
                    # Non-hammer team owns cache stones
                    self.team[idxs, idx, 0] = (1.0 - self.hammer_team[mask_k].float())
                self.count_stones[mask_k] += k_eff

        # --- "EMPTY" (code 4): do nothing ---

        # --- ORIGINAL LOGIC (conditional) ---
        is_original_code = (self.placement_code <= 3)
        if not self.cfg.use_powerplay:
            self.placement_code = torch.where((self.placement_code == 2) | (self.placement_code == 3),
                                              torch.zeros_like(self.placement_code),
                                              self.placement_code)

        should_place_original = is_original_code & (self.cfg.use_preplaced)
        if should_place_original.any():
            b_idx_orig = b_ar[should_place_original].unsqueeze(-1)
            if b_idx_orig.shape[0] == 0:
                return

            hammer_team_orig = self.hammer_team[should_place_original]
            placement_code_orig = self.placement_code[should_place_original]

            guard_y_val = 0.5 * (self.sheet.hog_y + self.sheet.tee_y)
            inhouse_y_val = self.sheet.tee_y - 0.61 + STONE_R

            x_guard = torch.zeros(b_idx_orig.shape[0], device=self.device)
            y_guard = torch.full((b_idx_orig.shape[0],), guard_y_val, device=self.device)

            is_B = (placement_code_orig == 1)
            y_guard = torch.where(is_B, y_guard - 0.914, y_guard)

            is_PPL = (placement_code_orig == 2)
            is_PPR = (placement_code_orig == 3)
            pp_offset = self._code_to_offset(placement_code_orig)

            x_inhouse = torch.where(is_PPL | is_PPR, pp_offset, torch.zeros_like(pp_offset))
            y_inhouse_base = torch.full((b_idx_orig.shape[0],), inhouse_y_val, device=self.device)
            y_inhouse_pp = torch.full((b_idx_orig.shape[0],), self.sheet.tee_y, device=self.device)
            y_inhouse = torch.where(is_PPL | is_PPR, y_inhouse_pp, y_inhouse_base)

            x_guard = torch.where(is_PPL | is_PPR, pp_offset, x_guard)

            # two stones per selected batch
            idx0 = self.count_stones[should_place_original]
            idx1 = self.count_stones[should_place_original] + 1
            for k, (xs, ys, own_team, idx) in enumerate([
                (x_guard, y_guard, (1 - hammer_team_orig), idx0),
                (x_inhouse, y_inhouse, hammer_team_orig, idx1),
            ]):
                self.pos[b_idx_orig, idx.unsqueeze(-1), 0] = xs.unsqueeze(-1)
                self.pos[b_idx_orig, idx.unsqueeze(-1), 1] = ys.unsqueeze(-1)
                self.vel[b_idx_orig, idx.unsqueeze(-1), :] = 0.0
                self.alive[b_idx_orig, idx.unsqueeze(-1), 0] = 1.0
                self.spin[b_idx_orig, idx.unsqueeze(-1), 0] = 0.0
                self.team[b_idx_orig, idx.unsqueeze(-1), 0] = own_team.to(torch.float32).unsqueeze(-1)

            self.count_stones[should_place_original] += 2

    # ------------------ Throw + simulate ------------------
    @torch.no_grad()
    def throw(self, actions: torch.Tensor):
        """
        actions: [B, 3] -> (speed, angle, spin_code in [-1,1])
        Adds the next stone for each end (if rocks_left allows) and simulates until static.
        Returns obs, reward (0 mid-end), done flags.
        """
        B, N = self.B, self.N
        assert actions.shape == (B, 3)

        # Ensure capacity BEFORE allocating the slot
        if int(self.count_stones.max().item()) >= self.N:
            raise AssertionError("Stone capacity exceeded (pre-throw). Reduce preplaced stones or increase self.N.")

        # Apply execution noise
        speed = actions[:, 0] + torch.randn(B, device=self.device, generator=self.rng) * self.cfg.noise_speed
        angle = actions[:, 1] + torch.randn(B, device=self.device, generator=self.rng) * self.cfg.noise_angle
        speed = speed.clamp(min=0.0, max=3.2)
        angle = angle.clamp(min=-0.8, max=0.8)

        spin_raw = actions[:, 2]
        spin = torch.where(spin_raw < -0.33, torch.full_like(spin_raw, -1.0),
               torch.where(spin_raw > 0.33, torch.full_like(spin_raw, +1.0), torch.zeros_like(spin_raw)))

        if self.cfg.noise_spin_flip_prob > 0.0:
            flip = (torch.rand(B, device=self.device, generator=self.rng) < self.cfg.noise_spin_flip_prob).float()
            spin = spin * torch.where(flip > 0.5, torch.full_like(spin, -1.0), torch.ones_like(spin))

        # Compute throw velocities
        vx = speed * torch.sin(angle)
        vy = speed * torch.cos(angle)
        v = torch.stack([vx, vy], dim=-1)

        # Allocate next slot per batch
        idx = self.count_stones  # [B]
        assert int(idx.max()) < self.N, "Stone capacity exceeded"
        b_ar = torch.arange(B, device=self.device)
        self.pos[b_ar, idx, :] = 0.0
        self.vel[b_ar, idx, :] = v
        self.alive[b_ar, idx, 0] = 1.0
        self.spin[b_ar, idx, 0] = spin
        self.team[b_ar, idx, 0] = self.cur_team.to(torch.float32)

        self.count_stones = self.count_stones + 1

        # Simulate until static
        self._simulate_until_static()

        # advance turn
        self.turn_idx += 1
        r = self.rocks_left.clone()
        r[b_ar, self.cur_team.long()] -= 1
        self.rocks_left = r
        self.cur_team = 1 - self.cur_team

        done = (self.rocks_left.sum(dim=1) == 0)
        reward = torch.zeros(B, device=self.device, dtype=torch.float32)
        if done.any():
            term_mask = done
            rew = self._score()
            reward = torch.where(term_mask, rew, reward)

        return self._obs(), reward, done

    # ------------------ Physics core (vectorized) ------------------
    def _simulate_until_static(self):
        dt = self.cfg.dt
        for _ in range(self.cfg.max_sim_steps):
            alive_mask = (self.alive[..., 0] > 0.0)
            if not alive_mask.any():
                break

            v = self.vel
            spd = _norm2(v)
            moving = (spd > self.cfg.v_stop) & alive_mask
            if not moving.any():
                self.vel[spd < self.cfg.v_stop] = 0.0
                break

            v_hat = torch.where(spd.unsqueeze(-1) > 1e-8, v / (spd.unsqueeze(-1) + 1e-8), torch.zeros_like(v))
            a_fric = - self.cfg.mu_drag * spd.unsqueeze(-1) * v_hat

            v_hat90 = _rot90(v_hat)
            a_curl = self.cfg.curl_gain * self.spin * v_hat90

            a = a_fric + a_curl
            self.vel = self.vel + a * dt

            spd2 = _norm2(self.vel)
            self.vel[spd2 < self.cfg.v_stop] = 0.0

            self.pos = self.pos + self.vel * dt

            # ---- FIX #3: blocked collision resolution (no [B,N,N,2] giant tensor) ----
            self._resolve_collisions_blocked(block=4)

            self._resolve_walls_vectorized()

        spd = _norm2(self.vel)
        self.vel[spd < self.cfg.v_stop] = 0.0

    def _resolve_collisions_blocked(self, block: int = 4):
        """
        Resolve elastic collisions using small NxN tiles over stones (tiling over i,j),
        avoiding materializing [B,N,N,2]. Still fully batched over B.
        """
        B, N = self.B, self.N
        pos = self.pos
        vel = self.vel
        alive = (self.alive[..., 0] > 0.0)

        e = self.cfg.restit
        rad = (2 * STONE_R)

        # We will accumulate per-stone position/velocity corrections and then apply once.
        dpos_accum = torch.zeros_like(pos)
        dvel_accum = torch.zeros_like(vel)

        for i0 in range(0, N, block):
            i1 = min(N, i0 + block)
            # Self block (upper-tri within)
            pi = pos[:, i0:i1, :]                         # [B, bi, 2]
            vi = vel[:, i0:i1, :]                         # [B, bi, 2]
            ali = alive[:, i0:i1]                         # [B, bi]

            # intra-block pairs
            if i1 - i0 > 1:
                bi = i1 - i0
                tri = torch.triu(torch.ones(bi, bi, device=self.device, dtype=torch.bool), diagonal=1)  # [bi,bi]
                # broadcast within block: [B, bi, bi, 2]
                dij = pi.unsqueeze(2) - pi.unsqueeze(1)
                dist2 = (dij * dij).sum(-1)                                        # [B, bi, bi]
                collide = (dist2 <= (rad * rad)) & tri.unsqueeze(0) & ali.unsqueeze(2) & ali.unsqueeze(1)
                if collide.any():
                    dist = torch.sqrt(dist2 + 1e-12)
                    n = torch.where(dist.unsqueeze(-1) > 0, dij / dist.unsqueeze(-1), torch.zeros_like(dij))
                    overlap = (rad - dist + 1e-4).clamp(min=0.0)
                    shift = (overlap / 2.0).unsqueeze(-1) * n
                    shift = torch.where(collide.unsqueeze(-1), shift, torch.zeros_like(shift))

                    # position correction
                    dpos_accum[:, i0:i1, :] += (-shift.sum(dim=2) + shift.sum(dim=1))

                    # velocity correction (1D along normal)
                    vi_exp = vi.unsqueeze(2).expand(B, bi, bi, 2)
                    vj_exp = vi.unsqueeze(1).expand(B, bi, bi, 2)
                    vi_n = (vi_exp * n).sum(-1)
                    vj_n = (vj_exp * n).sum(-1)
                    vi_n_new = (vi_n * (1 - e) + vj_n * (1 + e)) / 2.0
                    vj_n_new = (vj_n * (1 - e) + vi_n * (1 + e)) / 2.0
                    dvi = (vi_n_new - vi_n).unsqueeze(-1) * n
                    dvj = (vj_n_new - vj_n).unsqueeze(-1) * n
                    dvi = torch.where(collide.unsqueeze(-1), dvi, torch.zeros_like(dvi))
                    dvj = torch.where(collide.unsqueeze(-1), dvj, torch.zeros_like(dvj))
                    dvel_accum[:, i0:i1, :] += (dvi.sum(dim=2) + dvj.sum(dim=1))

            # cross-block pairs (i block vs j block > i)
            for j0 in range(i1, N, block):
                j1 = min(N, j0 + block)
                pj = pos[:, j0:j1, :]                     # [B, bj, 2]
                vj = vel[:, j0:j1, :]                     # [B, bj, 2]
                alj = alive[:, j0:j1]                     # [B, bj]

                # pairwise diffs for the tile only
                dij = pj.unsqueeze(1) - pi.unsqueeze(2)   # [B, bi, bj, 2]
                dist2 = (dij * dij).sum(-1)               # [B, bi, bj]
                collide = (dist2 <= (rad * rad)) & ali.unsqueeze(2) & alj.unsqueeze(1)
                if not collide.any():
                    continue

                dist = torch.sqrt(dist2 + 1e-12)
                n = torch.where(dist.unsqueeze(-1) > 0, dij / dist.unsqueeze(-1), torch.zeros_like(dij))
                overlap = (rad - dist + 1e-4).clamp(min=0.0)
                shift = (overlap / 2.0).unsqueeze(-1) * n
                shift = torch.where(collide.unsqueeze(-1), shift, torch.zeros_like(shift))

                # positions: i gets -sum over j, j gets +sum over i
                dpos_accum[:, i0:i1, :] += -shift.sum(dim=2)
                dpos_accum[:, j0:j1, :] +=  shift.sum(dim=1)

                # velocities along normal
                vi_exp = vi.unsqueeze(2).expand(-1, -1, j1-j0, -1)  # [B, bi, bj, 2]
                vj_exp = vj.unsqueeze(1).expand(-1, i1-i0, -1, -1)  # [B, bi, bj, 2]
                vi_n = (vi_exp * n).sum(-1)
                vj_n = (vj_exp * n).sum(-1)
                vi_n_new = (vi_n * (1 - e) + vj_n * (1 + e)) / 2.0
                vj_n_new = (vj_n * (1 - e) + vi_n * (1 + e)) / 2.0
                dvi = (vi_n_new - vi_n).unsqueeze(-1) * n
                dvj = (vj_n_new - vj_n).unsqueeze(-1) * n
                dvi = torch.where(collide.unsqueeze(-1), dvi, torch.zeros_like(dvi))
                dvj = torch.where(collide.unsqueeze(-1), dvj, torch.zeros_like(dvj))
                dvel_accum[:, i0:i1, :] += dvi.sum(dim=2)
                dvel_accum[:, j0:j1, :] += dvj.sum(dim=1)

        # apply accumulators and zero out dead stones’ velocity
        self.pos = self.pos + dpos_accum
        self.vel = self.vel + dvel_accum
        self.vel[~alive] = 0.0

    def _resolve_walls_vectorized(self):
        B, N = self.B, self.N
        xlim = self.sheet.width / 2 - STONE_R
        ylim_low = -STONE_R
        ylim_high = self.sheet.length - STONE_R

        x = self.pos[..., 0]; y = self.pos[..., 1]
        vx = self.vel[..., 0]; vy = self.vel[..., 1]
        alive = self.alive[..., 0] > 0.0

        left = x < -xlim
        right = x > xlim
        top = y > ylim_high
        bottom = y < ylim_low

        vx = torch.where(left | right, -self.cfg.restit_wall * vx, vx)
        vy = torch.where(top | bottom, -self.cfg.restit_wall * vy, vy)

        x = torch.where(left, torch.full_like(x, -xlim), x)
        x = torch.where(right, torch.full_like(x, xlim), x)
        y = torch.where(bottom, torch.full_like(y, ylim_low), y)
        y = torch.where(top, torch.full_like(y, ylim_high), y)

        out = (y > (self.sheet.backline_y + 0.2))
        kill = out & alive
        self.alive[..., 0] = torch.where(kill, torch.zeros_like(self.alive[..., 0]), self.alive[..., 0])

        self.pos[..., 0] = x; self.pos[..., 1] = y
        self.vel[..., 0] = vx; self.vel[..., 1] = vy

    # ------------------ Scoring / Obs ------------------
    @torch.no_grad()
    def _score(self) -> torch.Tensor:
        B, N = self.B, self.N
        center = torch.tensor([self.sheet.tee_x, self.sheet.tee_y], device=self.device, dtype=torch.float32)
        diff = self.pos - center.view(1, 1, 2)
        d = torch.linalg.vector_norm(diff, dim=-1)
        in_house = d <= self.sheet.house_radii[2]
        alive = (self.alive[..., 0] > 0.0) & in_house

        d_masked = torch.where(alive, d, torch.full_like(d, 1e9))
        d_sorted, idx_sorted = torch.sort(d_masked, dim=1, stable=True)
        t = self.team[..., 0].long()
        t_sorted = torch.gather(t, dim=1, index=idx_sorted)

        first_is_valid = (d_sorted[:, 0] < 1e8)
        winner = torch.where(first_is_valid, t_sorted[:, 0], torch.zeros_like(t_sorted[:, 0]))
        eq = (t_sorted == winner.unsqueeze(-1)) & (d_sorted < 1e8)
        c = eq.to(torch.int32)
        run = torch.cumprod(torch.where(c > 0, torch.ones_like(c), torch.zeros_like(c)), dim=1)
        pts = run.sum(dim=1)

        score = torch.where(first_is_valid,
                            torch.where(winner == 0, pts.float(), -pts.float()),
                            torch.zeros_like(pts, dtype=torch.float32))
        return score

    @torch.no_grad()
    def _obs(self) -> Dict[str, torch.Tensor]:
        cur = self.cur_team.float().unsqueeze(-1)
        is_pp_code = (self.placement_code == 2) | (self.placement_code == 3)
        pp_flag = (self.cfg.use_powerplay & is_pp_code).float()

        aux = torch.stack([
            cur.squeeze(-1),
            self.rocks_left[:, 0].float(),
            self.rocks_left[:, 1].float(),
            (self.hammer_team == 0).float(),
            self.placement_code.float(),
            self.turn_idx.float(),
            pp_flag,
        ], dim=-1)
        return {
            "pos": self.pos,
            "vel": self.vel,
            "alive": self.alive,
            "spin": self.spin,
            "team": self.team,
            "aux": aux,
        }

# -----------------------------------------------------------------------------
# Minimal Gym-like wrapper (torch in/out)
# -----------------------------------------------------------------------------
class TorchBatchedCurlingEnv:
    def __init__(self, B: int, device: str = "cuda", cfg: Optional[CurlingConfig] = None, sheet: Optional[Sheet] = None):
        self.sim = TorchBatchedCurlingSim(B, device, cfg, sheet)

    def set_execution_noise(self, sigma: float):
        self.sim.set_execution_noise(sigma)

    @torch.no_grad()
    def reset(self, placements: Optional[List[str]] = None):
        return self.sim.reset(placements)

    @torch.no_grad()
    def step(self, actions: torch.Tensor):
        return self.sim.throw(actions)

    @property
    def device(self): return self.sim.device

    @property
    def B(self): return self.sim.B

# -----------------------------------------------------------------------------
# Quick smoke test
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    B = 64
    cfg = CurlingConfig(use_preplaced=True, max_sim_steps=200, v_stop=0.05, dt=0.03)
    env = TorchBatchedCurlingEnv(B, device=device, cfg=cfg)

    placements = ["EMPTY"] + [f"CURRICULUM_{k}" for k in range(1, 12)]
    placements = placements * (B // len(placements)) + placements[:(B % len(placements))]
    obs = env.reset(placements=placements)
    print("--- Testing Placements ---")
    print("Batch:", B)
    print("Stones Count (min,max):", int(env.sim.count_stones.min().item()), int(env.sim.count_stones.max().item()))

    actions = torch.rand(B, 3, device=device)
    obs, rew, done = env.step(actions)
    print("\n--- Testing Step ---")
    print("Device:", env.device)
    print("Done:", done.sum().item(), "| Reward mean:", float(rew.mean().item()))