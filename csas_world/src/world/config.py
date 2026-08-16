"""Typed configuration for csas_world.

Every component is gated by a flag so the model can be ablated:
  * policy/value only            -> heads.dynamics/consistency/decoder/outcome off
  * + latent consistency         -> heads.dynamics + loss.consistency on
  * + physical decoder           -> heads.decoder on
  * learned-model prefiltering   -> search.use_learned_model_prefilter on (later)

Configs are plain YAML; unknown keys raise so typos are caught early.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


# --------------------------------------------------------------------------- #
# Sub-configs
# --------------------------------------------------------------------------- #
@dataclass
class ModelCfg:
    # trunk (must match the canonical GraphTF prior to warm-start)
    hidden_dim: int = 256
    n_layers: int = 4
    n_heads: int = 4
    dropout: float = 0.1
    cond_dim: int = 3
    input_dim: int = 24
    action_dim: int = 4
    n_mixtures: int = 16
    # head toggles
    use_value: bool = True
    use_policy: bool = True
    use_dynamics: bool = True          # latent dynamics G(h,a)
    use_step_reward: bool = False      # EXP-009: scalar 2-step-return reward head (auxiliary)
    use_outcome: bool = False          # opt-in tactical end-outcome head (distributional margin)
    use_decoder: bool = False          # physical next-state decoder D(h)
    use_consistency: bool = True       # EMA-target latent consistency
    consistency_mode: str = "simsiam"  # {simsiam, mse, none}
    # outcome head: categorical over end-score margin bins
    outcome_bins: int = 17             # margin in [-8, 8]
    # EMA target encoder
    ema_decay: float = 0.99
    # warm-start
    warm_start_trunk: bool = True
    warm_start_policy_head: bool = True
    warm_start_value_head: bool = True


@dataclass
class LossCfg:
    policy_bc: float = 1.0          # human behaviour-cloning NLL
    policy_distill: float = 1.0     # MCTS weighted-action distillation NLL
    value: float = 1.0              # value regression (MSE + NLL)
    value_nll: float = 0.2          # weight of the gaussian-NLL term inside value
    # Train the value head only on realized ValueDiff (the value buffer), like the
    # dedicated baseline -- not on MCTS search-values (which it overfits). The
    # search still drives the policy via distillation.
    value_from_mcts: bool = False
    value_rank: float = 0.0        # EXP-064: margin rank loss on gated top-1/top-2 posts
    rank_margin: float = 0.25
    outcome: float = 0.0            # tactical end-outcome CE; default off (value head is sufficient)
    step_reward: float = 0.0        # EXP-009: weight of the 2-step-return reward-head Huber loss
    reward_action_conditioned: bool = False  # EXP-012: predict r(s,a) from the POST-action latent G(s,a)
                                             # (steps[k+1]) instead of the state latent steps[k]
    consistency: float = 2.0        # latent consistency (EfficientZero default ~2)
    decoder: float = 1.0            # physical next-state decoder
    # value target construction
    value_clip: float = 8.0


@dataclass
class ReplayCfg:
    unroll_steps: int = 5           # K for latent unrolling
    # mixture weights across sources (renormalised)
    mix_human: float = 0.30
    mix_value: float = 0.15
    mix_sim: float = 0.15
    mix_mcts: float = 0.40
    value_use_synthetic: bool = True
    capacity_per_source: int = 400_000
    shard_dir: str = "artifacts/replay"


@dataclass
class TrainCfg:
    epochs: int = 30
    batch_size: int = 1024          # per-rank
    lr: float = 3.0e-4              # LR for FRESH heads (dynamics/reward/decoder/consistency)
    lr_pretrained: float = 1.5e-5  # LR for warm-started trunk + policy + value heads
    weight_decay: float = 1.0e-4
    grad_clip: float = 1.0
    warmup_steps: int = 200
    num_workers: int = 4
    amp: bool = False              # off: csas _scale_tril (exp) is fp32-only under autocast
    seed: int = 0
    samples_per_epoch: int = 120_000
    augment: bool = True           # horizontal flip + team-slot swap (match baselines)
    checkpoint_metric: str = "val_policy_nll"  # "none" delegates selection to game evaluation
    early_stop_patience: int = 0   # >0: abort training when checkpoint_metric hasn't improved
                                   # for this many epochs (val-driven early stopping)
    select_value_guard: float = 0.0  # >0: an epoch is eligible for best.pt ONLY if the guard metric
                                     # stays <= this threshold (value-drift guard)
    select_value_guard_metric: str = "val_value_mse"  # az_v15+: use "val_value_mse_mcts" (held-out
                                     # realized-return MSE = MATCHUP-TRUE calibration) per the H2 fix;
                                     # the human-data default is kept for back-compat only
    train_value_head_only: bool = False  # az_v15-VH: freeze everything but the value head
                                     # (per-head continuation; see analysis doc + L4 certificate caveat)
    # distributed
    gpus: List[int] = field(default_factory=lambda: [0, 1, 2, 3])
    ddp_backend: str = "gloo"       # NCCL crashes in this image; gloo is safe
    log_every: int = 50
    ckpt_dir: str = "checkpoints"
    run_name: str = "csas_world_v1"


@dataclass
class SearchCfg:
    # candidate pool (the "96 legal + 96 diverse" set the user described)
    policy_candidates: int = 96
    structured_candidates: int = 24
    diverse_candidates: int = 96
    local_candidates: int = 48
    global_candidates: int = 16
    temperature: float = 1.35
    std_scale: float = 1.6
    kernel_bandwidth: float = 0.18
    uct_c: float = 0.02
    soft_topk: int = 24
    policy_temperature: float = 0.35
    noise_config: str = "configs/noise/v2_fullsheet.json"  # relative to csas_v3 (full sheet)
    noise_samples: int = 2
    rollout_greedy_steps: int = 5    # greedy rollout depth to harvest unroll targets
    use_learned_model_prefilter: bool = False  # hook for later ablation
    # value-model-free target generation: score each candidate by rolling the policy
    # to terminal and scoring the realized end with curling rules (Monte-Carlo). The
    # value model is then NOT used during collection (it becomes a pure trained head).
    terminal_rollout_scoring: bool = False
    rollout_temp: float = 0.6        # policy temperature for the in-rollout behavior policy
    # real multi-ply KR-UCT tree search for the policy target (value-model-free; leaves
    # evaluated by on-policy MC rollout to terminal + rule scoring). value targets are
    # realized terminal ValueDiff. Used for the iterative AlphaZero-style loop.
    use_mcts_tree: bool = False
    collect_step_reward: bool = False  # EXP-009: compute 2-step-return targets during collection
    value_leaf_bootstrap: bool = False # EXP-010: tree leaves use the value head (closed loop via --value-world)
    reward_leaf_select: bool = False   # EXP-013: 1-ply-robust candidate value = -r̂₂(post) (2-step reward head)
                                       # instead of -V(post); use with use_mcts_tree=false + noise_samples>0
    search_rollout_n: int = 1          # EXP-014: value-greedy "searched" rollout width (>1 = each ply picks the
                                       # best of N policy samples by value); use with terminal_rollout_scoring
    policy_target_kernel_visits: bool = False  # EXP-015/016: distillation target = kernel-effective visit count
                                       # W(a)=Σ_b K(a,b)n_b (KR-UCT), not value-softmax. Depth-1 value-leaf bandit.
    search_root_only: bool = True      # depth-1 KR-UCT bandit (no grandchildren); used with policy_target_kernel_visits
    value_target_kernel_root: bool = False  # EXP-016: write the kernel-regressed root value V̂_root as the MCTS-record
                                       # value target (consumed only when loss.value_from_mcts=true); else realized margin
    mcts_sims: int = 120             # simulations per root decision
    mcts_k_widen: float = 2.0
    mcts_alpha_widen: float = 0.5
    mcts_uct_c: float = 0.6
    mcts_max_depth: int = 0          # 0 = horizon-bound tree (default); 2 = the 2-ply training-time
                                     # operator (root expand → child eval via leaf fn, no grandchildren).
                                     # Use with use_mcts_tree=true + value_leaf_bootstrap=true.
    # EXP-069 "vectree" scorer (vectorised 4-ply tree targets, EXP-068-certified):
    vectree_budget: int = 16000     # simulator calls per decision
    vectree_depth: int = 4          # search plies
    vectree_out_cap: int = 8        # execution outcomes retained at interior chance nodes
    vectree_root_out_cap: int = 0   # 0 inherits out_cap; use >=64 for a stable root expectation
    vectree_inner_pool: int = 8     # learner proposals at interior decision nodes
    opponent_model_actions: int = 1 # deployed-opponent intentions integrated per explicit node
    opponent_model_deploy_depth: int = 1 # exact selector through this tree depth; raw policy deeper
    opponent_model_candidates: int = 16 # affordable approximation to deployed 48-candidate selector
    opponent_model_noise_samples: int = 2 # executions used to rank each modeled candidate
    # EXP-078 baseline-aware exact screen for opponent-model tree corrections.
    # The tree action is used only when paired exact continuations beat the
    # deployed incumbent fallback; otherwise the incumbent action is targeted.
    paired_gate_repeats: int = 8
    paired_gate_t: float = 0.5
    paired_gate_candidates: int = 48
    paired_gate_noise_samples: int = 8
    paired_gate_seed_salt: int = 1_077_000
    # EXP-063 "bigsel" scorer (big-budget self-distillation teacher, EXP-062 T2):
    bigsel_candidates: int = 192     # policy proposals (deployed-selection family)
    bigsel_k: int = 64               # noise realizations per candidate (CRN)
    bigsel_temp: float = 1.1         # deployed-selection proposal temperature
    bigsel_std: float = 1.2
    # EXP-058 "stt" scorer (searched + truncated rollout estimator, EXP-056 cell):
    stt_n_search: int = 6            # value-greedy candidates per in-rollout step
    stt_k_ego: int = 4               # noisy executions per root candidate (CRN)
    stt_trunc: int = 4               # rollout throws before the value-head leaf
    screen_topk: int = 8             # az_v12 screen_tree scorer: survivors kept from the noise-robust
                                     # flat screen for the stage-2 KR-UCT refinement tree.
    dist_sig_t: float = 0.0          # az_v13: >0 -> distill a ply ONLY if the screen's top-1 vs top-2
                                     # Q-gap is significant (t >= this); 0 disables (all plies distilled)
    dist_sig_lo: float = 0.8         # tie-break band: dist_sig_lo <= t < dist_sig_t triggers extra
                                     # realisations for the top-2 before the final significance decision
    dist_sig_extra_k: int = 16       # extra noisy realisations per top-2 candidate in the tie-break


@dataclass
class HorizonCfg:
    start_horizon: int = 1
    max_horizon: int = 10
    rounds_per_stage: int = 3            # collect->train rounds inside a stage
    roots_per_stage: int = 1500          # MCTS roots collected per round
    include_preplaced: bool = True
    # convergence: a round is "stronger" if winrate > 0.5+band OR mean score-margin >
    # converge_margin_band (the latter off when 0.0). Plateau for converge_patience rounds advances.
    converge_band: float = 0.04
    converge_margin_band: float = 0.0    # >0: a clear positive Δscore also counts as improvement
    converge_patience: int = 2
    h2h_games_per_order: int = 200       # per throwing order
    noisy_h2h: bool = True               # h2h uses robust selection (avg over noise_samples) + noisy
                                         # realized throws — selection/eval robust to execution noise


@dataclass
class PathsCfg:
    csas_v3_root: str = "/mnt/data/curling2/csas_v3"
    human_csv: str = "data/processed/inverse_realistic_fullsheet.csv"  # rel to csas_v3 (full sheet)
    preplaced_csv: str = "data/processed/preplaced/first_shots_inverse.csv"
    # raw 2026 data is the holdout-splittable real source (4 competitions, comp 0 = test)
    value_data_stones: str = "data/raw/2026/Stones.csv"
    value_data_ends: str = "data/raw/2026/Ends.csv"
    # synthetic terminal states are added as extra train-only rows (single comp)
    value_synth_stones: str = "data/processed/synthetic_terminal/Stones.csv"
    value_synth_ends: str = "data/processed/synthetic_terminal/Ends.csv"
    prior_policy_ckpt: str = "checkpoints/policy/human_prior_fullcov_fullsheet/best.pt"  # full sheet
    prior_value_ckpt: str = "checkpoints/value/holdout0/model.pt"  # sheet-agnostic (real-data value)
    # baselines to compare against in head-to-head
    baseline_policy_ckpts: List[str] = field(default_factory=lambda: [
        "checkpoints/policy/human_prior_fullcov/model.pt",
        "checkpoints/policy/mcts_horizon/h05/model.pt",
        "checkpoints/policy/mcts_horizon/h10/model.pt",
    ])
    baseline_value_ckpt: str = "checkpoints/value/holdout0/model.pt"


@dataclass
class Config:
    model: ModelCfg = field(default_factory=ModelCfg)
    loss: LossCfg = field(default_factory=LossCfg)
    replay: ReplayCfg = field(default_factory=ReplayCfg)
    train: TrainCfg = field(default_factory=TrainCfg)
    search: SearchCfg = field(default_factory=SearchCfg)
    horizon: HorizonCfg = field(default_factory=HorizonCfg)
    paths: PathsCfg = field(default_factory=PathsCfg)

    # ----------------------------------------------------------------- #
    def csas_path(self, rel_or_abs: str) -> Path:
        """Resolve a path that may be relative to the csas_v3 root."""
        p = Path(rel_or_abs)
        if p.is_absolute():
            return p
        return Path(self.paths.csas_v3_root) / p

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


# --------------------------------------------------------------------------- #
# (de)serialisation
# --------------------------------------------------------------------------- #
def _from_dict(cls, data: Dict[str, Any]):
    if not is_dataclass(cls):
        return data
    # `from __future__ import annotations` makes field.type a string, so resolve
    # the real annotations to detect nested dataclasses.
    import typing

    hints = typing.get_type_hints(cls)
    kwargs = {}
    known = {f.name for f in fields(cls)}
    for key, val in (data or {}).items():
        if key not in known:
            raise KeyError(f"Unknown config key '{key}' for {cls.__name__}")
        ftype = hints.get(key)
        if ftype is not None and is_dataclass(ftype) and isinstance(val, dict):
            kwargs[key] = _from_dict(ftype, val)
        else:
            kwargs[key] = val
    return cls(**kwargs)


def model_cfg_from_dict(d: Optional[Dict[str, Any]]) -> "ModelCfg":
    """Build a ModelCfg from a (possibly older) saved ``model_cfg`` dict.

    Renames deprecated keys and drops unknown ones so checkpoints written before a
    schema change still load (e.g. the per-step reward head was removed: its
    ``use_reward`` flag maps to the surviving ``use_outcome`` outcome head)."""
    d = dict(d or {})
    if "use_reward" in d:
        d.setdefault("use_outcome", d["use_reward"])
        d.pop("use_reward")
    known = {f.name for f in fields(ModelCfg)}
    return ModelCfg(**{k: v for k, v in d.items() if k in known})


def load_config(path: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> Config:
    data: Dict[str, Any] = {}
    if path is not None:
        with open(path, "r") as fh:
            data = yaml.safe_load(fh) or {}
    if overrides:
        data = _deep_merge(data, overrides)
    return _from_dict(Config, data)


def _deep_merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def save_config(cfg: Config, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        yaml.safe_dump(cfg.to_dict(), fh, sort_keys=False)


__all__ = [
    "Config", "ModelCfg", "LossCfg", "ReplayCfg", "TrainCfg",
    "SearchCfg", "HorizonCfg", "PathsCfg", "load_config", "save_config",
    "model_cfg_from_dict",
]
