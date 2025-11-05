from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
import os, json
import numpy as np
import torch
import torch.distributed as dist
from tqdm import tqdm, trange

from gpu_psro.obs_embed import embed_obs
from gpu_psro.policies import Policy, MixturePolicy, ActorCritic, ActorCriticCfg
from gpu_psro.ppo_br import PPOBestResponse, PPOCfg
from gpu_psro.meta import solve_zero_sum_minimax_mwu
from gpu_psro.io_utils import (
    save_policy_state_dict,
    save_json,
    save_numpy,
    ensure_dir,
    load_policy_state_dict,
)
from torch_curling_batched import TorchBatchedCurlingEnv, CurlingConfig

def _is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()
def _rank() -> int:
    return dist.get_rank() if _is_dist() else 0
def _world() -> int:
    return dist.get_world_size() if _is_dist() else 1
def _is_main() -> bool:
    return _rank() == 0
def _split(total: int, k: int, r: int) -> int:
    if k <= 1: return total
    base, rem = divmod(total, k)
    return base + (1 if r < rem else 0)

@dataclass
class PolicyRecord:
    policy: Policy
    label: str

@dataclass
class PSROCfg:
    device: str = "cuda"
    batch_eval: int = 8192
    episodes_eval_per_batch: int = 1
    exec_noise_sigma: float = 0.05
    max_iterations: int = 10
    out_dir: str = "checkpoints"

@dataclass
class PSRO:
    make_env_cfg: CurlingConfig
    placement: str = "A"
    psro: PSROCfg = field(default_factory=PSROCfg)
    # NEW OBS DIM: 16 stones * 4 features + 4 flags = 68
    ppo: PPOCfg = field(default_factory=lambda: PPOCfg(obs_dim=68, batch_size_envs=8192))

    def _local_cuda_str(self) -> str:
        if self.psro.device.startswith("cuda") and torch.cuda.is_available():
            return f"cuda:{torch.cuda.current_device()}"
        return "cpu"

    def _make_env(self, batch_size: int, device: str):
        env = TorchBatchedCurlingEnv(batch_size, device=device, cfg=self.make_env_cfg)
        env.set_execution_noise(self.psro.exec_noise_sigma)
        env.reset(placements=[self.placement] * batch_size)
        return env

    @torch.no_grad()
    def _eval_match_shard(self, pol0: Policy, pol1: Policy, B: int, device: str) -> Tuple[float, float]:
        total_throws = 2 * self.make_env_cfg.stones_per_team
        env = self._make_env(B, device)
        pbar = tqdm(total=total_throws, desc=f"Eval(rank={_rank()})", leave=False, unit="throws", disable=not _is_main())

        final_rew = torch.zeros(B, dtype=torch.float32, device=device)
        while True:
            obs = env.sim._obs()
            obs_vec = embed_obs(obs).to(device)
            cur_team = obs["aux"][:, 0].long()
            actions = torch.zeros(B, 3, device=device)
            mask0 = (cur_team == 0)
            mask1 = ~mask0
            if mask0.any():
                actions[mask0] = pol0.act(obs_vec[mask0], deterministic=False)
            if mask1.any():
                actions[mask1] = pol1.act(obs_vec[mask1], deterministic=False)
            _, rew, done = env.step(actions)
            pbar.update(1)
            if done.any():
                final_rew = torch.where(done, rew, final_rew)
                if done.all():
                    break

        pbar.close()
        mean_score = final_rew.mean().item()
        win_rate = ((final_rew > 0).float().mean() + 0.5 * (final_rew == 0).float().mean()).item()
        return float(mean_score), float(win_rate)

    @torch.no_grad()
    def _eval_match(self, pol0: Policy, pol1: Policy) -> Tuple[float, float]:
        ws = _world()
        rk = _rank()
        B_local = _split(self.psro.batch_eval, ws, rk)
        dev = self._local_cuda_str()

        s_sum = torch.tensor(0.0, device=dev)
        w_sum = torch.tensor(0.0, device=dev)
        cnt   = torch.tensor(0.0, device=dev)

        if B_local > 0:
            s_avg, w_avg = self._eval_match_shard(pol0, pol1, B_local, dev)
            s_sum += s_avg * float(B_local)
            w_sum += w_avg * float(B_local)
            cnt   += float(B_local)

        if _is_dist():
            dist.all_reduce(s_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(w_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(cnt,   op=dist.ReduceOp.SUM)

        denom = max(cnt.item(), 1.0)
        return float(s_sum.item() / denom), float(w_sum.item() / denom)

    # ---------- metrics helpers (unchanged) ----------
    @staticmethod
    def _compute_exploitability(payoff: np.ndarray, x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        v_meta = float(x @ payoff @ y)
        row_pay = payoff @ y
        col_pay = (x @ payoff)
        v_rowBR = float(row_pay.max())
        v_colBR = float(col_pay.min())
        return {"meta_value": v_meta, "row_best_response": v_rowBR, "col_best_response": v_colBR, "exploitability": v_rowBR - v_colBR}

    @staticmethod
    def _elo_from_winrates(W: np.ndarray, iters: int = 400, lr: float = 0.5) -> np.ndarray:
        N = W.shape[0]
        r = np.zeros(N, dtype=np.float64)
        for _ in range(iters):
            R = r[:, None] - r[None, :]
            P = 1.0 / (1.0 + np.exp(-R))
            grad = (W - P).sum(axis=1) - (W.T - (1 - P.T)).sum(axis=1)
            r += lr * grad
            r -= r.mean()
        return r

    @staticmethod
    def _relative_population_perf(W: np.ndarray) -> np.ndarray:
        N = W.shape[0]
        mask = ~np.eye(N, dtype=bool)
        sums = (W * mask).sum(axis=1)
        counts = mask.sum(axis=1)
        return sums / np.maximum(1, counts)

    # ---------- resume helpers (unchanged logic) ----------
    def _load_population_from_disk(self, device: str, obs_dim: int):
        out_dir = self.psro.out_dir
        npz_path = os.path.join(out_dir, "psro_state.npz")
        payoff_path = os.path.join(out_dir, "payoff.npy")
        win_path = os.path.join(out_dir, "winrate.npy")

        payoff = winmat = None
        start_iter = 0

        if os.path.exists(npz_path):
            try:
                bundle = np.load(npz_path, allow_pickle=False)
                payoff = bundle["payoff"]; winmat = bundle["winrate"]
                if "iter" in bundle.files: start_iter = int(bundle["iter"])
                if "placement" in bundle.files and str(bundle["placement"]) != str(self.placement) and _is_main():
                    print(f"[psro] Warning: placement in NPZ ({bundle['placement']}) != current ({self.placement})")
                if "exec_noise_sigma" in bundle.files and float(bundle["exec_noise_sigma"]) != float(self.psro.exec_noise_sigma) and _is_main():
                    print(f"[psro] Warning: noise sigma in NPZ ({float(bundle['exec_noise_sigma'])}) != current ({self.psro.exec_noise_sigma})")
            except Exception as e:
                if _is_main(): print(f"[psro] Failed to read psro_state.npz ({e}); falling back to .npy files")
                payoff = winmat = None

        if payoff is None or winmat is None:
            if os.path.exists(payoff_path) and os.path.exists(win_path):
                payoff = np.load(payoff_path); winmat = np.load(win_path)
                start_iter = max(0, min(payoff.shape[0], payoff.shape[1]) - 1)
            else:
                return [], [], None, None, 0

        m, n = payoff.shape
        p0, p1 = [], []

        path0 = os.path.join(out_dir, "p0_net0.pt")
        if not os.path.exists(path0): raise FileNotFoundError(f"Missing seed policy: {path0}")
        pol0 = ActorCritic(ActorCriticCfg(obs_dim=obs_dim, hidden=self.ppo.hidden), device=device)
        load_policy_state_dict(pol0, path0, map_location=device)
        p0.append(PolicyRecord(pol0, "p0_net0"))
        for i in range(1, m):
            path = os.path.join(out_dir, f"p0_br{i-1}.pt")
            if not os.path.exists(path): raise FileNotFoundError(f"Missing checkpoint for p0 index {i}: {path}")
            pol = ActorCritic(ActorCriticCfg(obs_dim=obs_dim, hidden=self.ppo.hidden), device=device)
            load_policy_state_dict(pol, path, map_location=device)
            p0.append(PolicyRecord(pol, f"p0_br{i-1}"))

        path1 = os.path.join(out_dir, "p1_net0.pt")
        if not os.path.exists(path1): raise FileNotFoundError(f"Missing seed policy: {path1}")
        pol1 = ActorCritic(ActorCriticCfg(obs_dim=obs_dim, hidden=self.ppo.hidden), device=device)
        load_policy_state_dict(pol1, path1, map_location=device)
        p1.append(PolicyRecord(pol1, "p1_net0"))
        for j in range(1, n):
            path = os.path.join(out_dir, f"p1_br{j-1}.pt")
            if not os.path.exists(path): raise FileNotFoundError(f"Missing checkpoint for p1 index {j}: {path}")
            pol = ActorCritic(ActorCriticCfg(obs_dim=obs_dim, hidden=self.ppo.hidden), device=device)
            load_policy_state_dict(pol, path, map_location=device)
            p1.append(PolicyRecord(pol, f"p1_br{j-1}"))

        start_iter = max(0, min(m, n) - 1)
        return p0, p1, payoff, winmat, start_iter

    def _save_psro_state(self, payoff: np.ndarray, winmat: np.ndarray, iter_idx: int, meta_value: float):
        if not _is_main(): return
        out_dir = self.psro.out_dir
        ensure_dir(out_dir)
        save_numpy(payoff, os.path.join(out_dir, "payoff.npy"))
        save_numpy(winmat, os.path.join(out_dir, "winrate.npy"))
        np.savez(os.path.join(out_dir, "psro_state.npz"),
                 payoff=payoff, winrate=winmat, iter=iter_idx, meta_value=meta_value,
                 placement=self.placement, exec_noise_sigma=self.psro.exec_noise_sigma,
                 stones_per_team=self.make_env_cfg.stones_per_team)

    # ---------- main loop ----------
    def run(self, br_steps: int = 1_000_000) -> Dict:
        device = self.psro.device
        obs_dim = self.ppo.obs_dim
        ensure_dir(self.psro.out_dir)

        p0, p1, payoff, winmat, start_iter = self._load_population_from_disk(device, obs_dim)

        if payoff is None:
            p0 = [PolicyRecord(ActorCritic(ActorCriticCfg(obs_dim=obs_dim, hidden=self.ppo.hidden), device=device), "p0_net0")]
            p1 = [PolicyRecord(ActorCritic(ActorCriticCfg(obs_dim=obs_dim, hidden=self.ppo.hidden), device=device), "p1_net0")]
            if _is_main():
                save_policy_state_dict(p0[0].policy, f"{self.psro.out_dir}/p0_net0.pt")
                save_policy_state_dict(p1[0].policy, f"{self.psro.out_dir}/p1_net0.pt")

            payoff = np.zeros((1, 1), dtype=np.float64)
            winmat = np.full((1, 1), 0.5, dtype=np.float64)
            s0, w0 = self._eval_match(p0[0].policy, p1[0].policy)
            payoff[0, 0] = s0
            winmat[0, 0] = w0
            self._save_psro_state(payoff, winmat, iter_idx=0, meta_value=float(s0))
            history = []
            start_iter = 0
        else:
            hist_path = os.path.join(self.psro.out_dir, "history.json")
            if os.path.exists(hist_path):
                try:
                    with open(hist_path, "r") as f:
                        hist_obj = json.load(f); history = hist_obj.get("history", [])
                except Exception:
                    history = []
            else:
                history = []

        it_bar = trange(start_iter, self.psro.max_iterations, desc="PSRO iters", unit="iter", disable=not _is_main())
        for it in it_bar:
            # Meta
            meta = solve_zero_sum_minimax_mwu(payoff)
            mix0 = MixturePolicy([rec.policy for rec in p0], meta.x.tolist(), device=device)
            mix1 = MixturePolicy([rec.policy for rec in p1], meta.y.tolist(), device=device)
            if _is_main():
                save_json({"iter": it, "meta_value": float(meta.value), "mix_p0": meta.x.tolist(), "mix_p1": meta.y.tolist()},
                          f"{self.psro.out_dir}/mixture_iter{it}.json")

            # BR for P0
            br0 = PPOBestResponse(
                env_ctor=self._make_env, p=0, opponent=mix1,
                cfg=PPOCfg(
                    obs_dim=obs_dim, device=device, total_env_steps=br_steps,
                    rollout_horizon=self.ppo.rollout_horizon, gamma=self.ppo.gamma,
                    gae_lambda=self.ppo.gae_lambda, clip_eps=self.ppo.clip_eps, lr=self.ppo.lr,
                    epochs=self.ppo.epochs, minibatch_size=self.ppo.minibatch_size,
                    entropy_coef=self.ppo.entropy_coef, value_coef=self.ppo.value_coef,
                    hidden=self.ppo.hidden, batch_size_envs=self.ppo.batch_size_envs,
                    ckpt_dir=self.psro.out_dir, ckpt_tag=f"p0_br{it}", ckpt_interval=250_000,
                    export_torchscript=False, ddp=(_world() > 1), world_size=_world(), rank=_rank(),
                ),
            ).train()
            p0.append(PolicyRecord(br0, f"p0_br{it}"))
            if _is_main(): save_policy_state_dict(p0[-1].policy, f"{self.psro.out_dir}/p0_br{it}.pt")

            # BR for P1
            br1 = PPOBestResponse(
                env_ctor=self._make_env, p=1, opponent=mix0,
                cfg=PPOCfg(
                    obs_dim=obs_dim, device=device, total_env_steps=br_steps,
                    rollout_horizon=self.ppo.rollout_horizon, gamma=self.ppo.gamma,
                    gae_lambda=self.ppo.gae_lambda, clip_eps=self.ppo.clip_eps, lr=self.ppo.lr,
                    epochs=self.ppo.epochs, minibatch_size=self.ppo.minibatch_size,
                    entropy_coef=self.ppo.entropy_coef, value_coef=self.ppo.value_coef,
                    hidden=self.ppo.hidden, batch_size_envs=self.ppo.batch_size_envs,
                    ckpt_dir=self.psro.out_dir, ckpt_tag=f"p1_br{it}", ckpt_interval=250_000,
                    export_torchscript=False, ddp=(_world() > 1), world_size=_world(), rank=_rank(),
                ),
            ).train()
            p1.append(PolicyRecord(br1, f"p1_br{it}"))
            if _is_main(): save_policy_state_dict(p1[-1].policy, f"{self.psro.out_dir}/p1_br{it}.pt")

            # Expand payoff/winrate and evaluate NEW row/col
            m_old, n_old = payoff.shape
            payoff_new = np.zeros((m_old + 1, n_old + 1), dtype=np.float64)
            winmat_new = np.full((m_old + 1, n_old + 1), 0.5, dtype=np.float64)
            payoff_new[:m_old, :n_old] = payoff
            winmat_new[:m_old, :n_old] = winmat

            # New row: p0_br[it] vs all p1
            for j in range(n_old + 1):
                s, w = self._eval_match(p0[m_old].policy, p1[j].policy)
                if _is_main():
                    payoff_new[m_old, j] = s
                    winmat_new[m_old, j] = w

            # New col: all p0 vs p1_br[it]
            for i in range(m_old + 1):
                s, w = self._eval_match(p0[i].policy, p1[n_old].policy)
                if _is_main():
                    payoff_new[i, n_old] = s
                    winmat_new[i, n_old] = w

            # Broadcast updated matrices to all ranks
            if _is_dist():
                shape_t = torch.tensor([payoff_new.shape[0], payoff_new.shape[1]], device=self._local_cuda_str(), dtype=torch.int64)
                dist.broadcast(shape_t, src=0)
                M, N = int(shape_t[0].item()), int(shape_t[1].item())
                buf_pay = torch.empty(M * N, device=self._local_cuda_str(), dtype=torch.float64)
                buf_win = torch.empty(M * N, device=self._local_cuda_str(), dtype=torch.float64)
                if _is_main():
                    buf_pay.copy_(torch.from_numpy(payoff_new.reshape(-1)).to(buf_pay.device))
                    buf_win.copy_(torch.from_numpy(winmat_new.reshape(-1)).to(buf_win.device))
                dist.broadcast(buf_pay, src=0)
                dist.broadcast(buf_win, src=0)
                payoff_new = buf_pay.cpu().numpy().reshape(M, N)
                winmat_new = buf_win.cpu().numpy().reshape(M, N)

            payoff = payoff_new; winmat = winmat_new

            # Metrics + save
            meta2 = solve_zero_sum_minimax_mwu(payoff)
            if _is_main() and it_bar is not None:
                it_bar.set_postfix(meta_value=f"{meta2.value:.4f}")

            ex = self._compute_exploitability(payoff, meta2.x, meta2.y)

            W0 = np.zeros((m_old + 1, m_old + 1))
            for i in range(m_old + 1):
                for k in range(m_old + 1):
                    lhs = float(meta2.y @ winmat[i, :]); rhs = float(meta2.y @ winmat[k, :])
                    W0[i, k] = float(lhs > rhs) + 0.5 * float(np.isclose(lhs, rhs))
            W0 = np.where(np.eye(W0.shape[0], dtype=bool), 0.5, W0)
            elo0 = self._elo_from_winrates(W0); rpp0 = self._relative_population_perf(W0)

            W1 = np.zeros((n_old + 1, n_old + 1))
            for j in range(n_old + 1):
                for l in range(n_old + 1):
                    lhs = float(winmat[:, j] @ meta2.x); rhs = float(winmat[:, l] @ meta2.x)
                    W1[j, l] = float(lhs < rhs) + 0.5 * float(np.isclose(lhs, rhs))
            W1 = np.where(np.eye(W1.shape[0], dtype=bool), 0.5, W1)
            elo1 = self._elo_from_winrates(W1); rpp1 = self._relative_population_perf(W1)

            self._save_psro_state(payoff, winmat, iter_idx=it, meta_value=float(meta2.value))
            if _is_main():
                save_json(
                    {"iter": it, "meta_value": float(meta2.value), "exploitability": ex,
                     "elo_p0": elo0.tolist(), "elo_p1": elo1.tolist(),
                     "rpp_p0": rpp0.tolist(), "rpp_p1": rpp1.tolist(),
                     "mix_p0": meta2.x.tolist(), "mix_p1": meta2.y.tolist(),
                     "m": int(payoff.shape[0]), "n": int(payoff.shape[1])},
                    f"{self.psro.out_dir}/metrics_iter{it}.json",
                )

            if _is_dist(): dist.barrier()

        final = solve_zero_sum_minimax_mwu(payoff)
        if _is_main():
            save_json(
                {"final_meta_value": float(final.value), "final_mix_p0": final.x.tolist(), "final_mix_p1": final.y.tolist()},
                f"{self.psro.out_dir}/final_mixture.json",
            )

        return {
            "meta_value": float(final.value),
            "mix_p0": [(rec.label, float(w)) for rec, w in zip(p0, final.x)],
            "mix_p1": [(rec.label, float(w)) for rec, w in zip(p1, final.y)],
            "payoff": payoff,
            "history": history if _is_main() else [],
        }
