# csas_world — Experiments Log

Chronological record of trained models, their setup, and measured performance.
Each experiment is a self-contained block following the schema in
[Template](#template-for-a-new-entry); the [Summary table](#summary-table) is the
one-line index. Append new entries at the bottom and add a row to the table — do
not rewrite old entries (correct them with a dated **Update:** note instead).

---

## Metric & protocol glossary

| Term | Meaning |
|---|---|
| **val/test value MSE** | Mean-squared error of the value head's mean vs. the realized signed end margin (`ValueDiff`), on the held-out competition split. Lower is better. The dedicated `csas_v3` Gaussian value model scores ≈ **2.22 / 2.09** (val/test) as a reference. |
| **val value NLL** | Gaussian negative log-likelihood of the value head (mean + log-variance) on the same split. |
| **val/test policy NLL** | Negative log-likelihood of the *human* action under the policy's full-covariance mixture. Lower = more human-like. This is **not** the training objective once `policy_bc=0`; it is reported only as a human-likeness diagnostic, so a search-distilled policy can be stronger while scoring a higher (worse) NLL. |
| **win rate / Δscore** | Game strength by **true alternating play to the terminal state under both throwing orders**. At each decision a player ranks `n_candidates` policy-sampled throws by *decision value* — for horizon > 1 this is its **own value head** applied to the simulated post-state, averaged over `sel_noise_samples` execution-noise realizations; at horizon 1 it is the exact rule score. The end is then rolled to terminal and scored by curling rules. **Win rate** is fraction of ends won; **Δscore** is the mean end-score differential (positive favours the evaluated model). |
| **Important** | Decision-value selection uses *each model's own value head*, so a model with a worse value head is handicapped at selection time even if its policy is unchanged — head-to-head numbers are only comparable when the value heads are of comparable quality. |

**Shared setup unless noted:** GraphTF trunk (hidden 256, 4 layers, 4 heads),
warm-started from `csas_v3` `human_prior_fullcov`; full-covariance MDN policy
(K=16) over `[speed, angle, spin, y0]`; Gaussian value head; authoritative JAX
simulator for all transitions; 4×A10G `gloo` DDP, batch 256, AMP off. Data:
inverse-recovered human actions (policy) + realized end outcomes from four mixed
-doubles competitions, held out by competition (value).

---

## Summary table

| ID | Date | Model / change | val policy NLL | val value MSE | Game strength | Verdict |
|---|---|---|---|---|---|---|
| EXP-001 | 2026-06 | `anchor_noisy` (execution-noise MCTS targets; value on clean buffer) | **4.19** | **2.18** | beats `csas_v3` `mcts_horizon/h10` ~59–64%; ≈0.51 vs noise-free anchor | **current best** |
| EXP-002 | 2026-06-10 | `az/iter1` — real KR-UCT tree + terminal-MC value (`value_from_mcts=true`), outcome head, fixed FGZ rule | 4.38 | 3.13 | 0.525 win / −0.308 Δscore vs `anchor_noisy` | did not beat anchor; **value head regressed** |
| EXP-003 | 2026-06-10 | `az/iter1_valclean` — option 2: `value_from_mcts=false` (value on clean buffer), reuses EXP-002 KR-UCT data | 4.51 | 2.30 | 0.525 win / −0.192 Δscore vs `anchor_noisy` | value head **fixed** (3.13→2.30); still parity |
| EXP-004 | 2026-06-10 | `az_v3/iter1` — option 1: KR-UCT `n_sims=120` (2.5× search) + `value_from_mcts=false` | 4.46 | 2.28 | 0.458 win / −0.167 Δscore vs `anchor_noisy` | stronger search did **not** help; ≈ parity-or-below |
| EXP-005 | 2026-06-12 | `anchor_noisy_simsiam_stopgrad_no_outcome` — direct online stop-gradient target; outcome head disabled | best **4.302**, final 4.501 | best **2.216**, final 2.319 | not run | no gain over EMA; continuation regressed |
| EXP-006 | 2026-06-12 | `anchor_noisy_ema_no_outcome` — original EMA target; outcome head disabled | best **4.302**, final 4.502 | best **2.216**, final 2.319 | not run | indistinguishable from EXP-005; continuation regressed |
| EXP-007 | 2026-06-12 | controlled iterative `outcome_on` vs `outcome_off`; fresh MCTS every 5 epochs, optimizer resumed, no BC/schedule | on **6.177**, off **5.371** | on **2.178**, off **2.162** | independent direct match: **0.500 win / -0.025 Δscore** for on vs off | outcome head provides **no measurable game-strength benefit** |
| EXP-008 | 2026-06-16 | **FULL SHEET** `exp_a_tuned` — tuned recipe (KR-UCT tree `n_sims=96`, value-model-free, `policy_bc=0`, `value=1.5`), 4-horizon A/B (h2-5) | ~14.5-15.3 (policy_bc=0; not a strength metric) | ~2.26 | h2-5 win ≈0.54/0.50/0.50/0.55 vs prior (≈0.52, parity) | **fixes regression → parity**; no decisive gain |
| EXP-009 | 2026-06-17 | **FULL SHEET** `exp_b_reward` — EXP-008 + auxiliary 2-step-return reward head | ~15.2-15.9 (policy_bc=0) | **2.200** (vs EXP-008 2.26 / baseline 2.22) | h2-5 win ≈0.58/0.50/0.50/0.49 vs prior (≈0.52, parity) | reward head **improves value head**; game strength parity |
| EXP-010 | 2026-06-17 | **FULL SHEET** `exp_c_valueloop` — EXP-009 + closed value loop (value-head leaf bootstrap + `--value-world`), 2 rounds/stage | ~15 (policy_bc=0) | **2.192** | deterministic 0.5335 vs prior, BUT **NOISY 0.5031 ± 0.018 → parity** | det. edge was an artifact; **parity under noise** |
| EXP-011 | 2026-06-17 | **FULL SHEET** `full_valueloop` — EXP-010 recipe across ALL horizons 1→10, to convergence (h2h winrate OR Δscore) | _pending_ | _pending_ | h01 ~parity (0.51/0.50) | **PAUSED at h02** (resume --start 2); deferred for EXP-012 |
| EXP-012 | 2026-06-17 | **FULL SHEET** `exp_d_reward_sa` — EXP-010 + **action-conditioned** reward `r(s,a)` (post-action latent), 4-horizon | ~15 (policy_bc=0) | **2.188** (best of all) | **NOISY: vs EXP-010 0.4825, vs prior 0.5044 — both parity** | action-cond = no benefit (use state-cond); parity under noise |
| EXP-013 | 2026-06-18 | **FULL SHEET** `exp_013_reward_robust` — **1-ply-ROBUST** `-r̂₂` selector (noise_samples=8 avg; restores robust selection lost since EXP-002), 4-horizon | ~15 (policy_bc=0) | n/a | **NOISY: vs prior 0.5407 ± 0.019 → BEATS; vs EXP-010 0.5446 → BEATS** (>2SE) | **FIRST to beat the prior under noise** — robust selection is the lever |

### ⚠ Noise-averaging (robust selection) status — which experiments averaged over execution noise

"Noise-averaging" = scoring each candidate (and the searched opponent reply) as the **mean decision value over K noisy executions** of the shot, so selection prefers shots that survive imperfect execution. It is active **only in the non-tree value path** (`search_state`: `use_mcts_tree=false` + `noise_samples>0`). The **KR-UCT tree commits one fixed noise draw per child and ignores `noise_samples`** (the knob is dead in tree configs). The in-loop h2h has additionally been **deterministic** throughout.

| Experiment | noise-averaged selection? | why |
|---|---|---|
| **EXP-001 `anchor_noisy`** | **YES (×8)** | non-tree value path, `noise_samples=8` |
| EXP-005 / EXP-006 / EXP-007 | YES (lineage) | trained on the noise-averaged `anchor_noisy` MCTS buffer / non-tree anchor_noisy-lineage |
| **EXP-002 / 003 / 004** | **NO** | KR-UCT tree (`use_mcts_tree=true`) → bypasses averaging |
| **EXP-008 / 009 / 010 / 011 / 012** (all full-sheet) | **NO** | all KR-UCT tree; `noise_samples:8` present but **dead** |
| **EXP-013 onward** | **YES (restored as default)** | noise-averaging in collection/search **and** noisy realized h2h eval — policy: *always* use the noisy average |

**Lost since:** EXP-002 (adoption of the KR-UCT tree). **Caveats:** (1) EXP-010's "beats prior 0.5335" was measured with non-robust selection + a zero-noise eval — **re-confirmed under noisy execution (2026-06-17): it collapses to 0.503 ± 0.018 = parity.** The deterministic edge was an artifact. (2) This plausibly explains EXP-004's "stronger search didn't beat anchor_noisy": the tree traded noise-robustness for depth.

---

## EXP-001 — `anchor_noisy` (baseline / current best)

- **Checkpoint:** `checkpoints/csas_world/anchor_noisy/model.pt` (+ `policy_csas.pt`)
- **Goal:** establish a strong unified policy/value model under realistic execution noise.
- **Setup**
  - Targets: MCTS-distilled policy with **execution-noise-averaged** candidate values (Bowling Student-t, `configs/noise/v1_bowling.json`), so the policy prefers robust, makeable shots.
  - **Value head trained on the realized-`ValueDiff` buffer** (real ends + synthetic terminal states), *not* on search values (`value_from_mcts=false`).
  - Heads active: policy, value, latent dynamics + EMA consistency, **per-step reward + outcome** (the per-step reward head was still present at this time), decoder.
  - Legality: the **pre-fix** free-guard-zone rule (over-restrictive — banned early *movement/contact*, not just removal; see EXP-002 for the correction).
- **Performance**
  - Value: val MSE **2.18**, test **2.08**, val value NLL ≈ 0.80 → on par / marginally better than the dedicated `csas_v3` value model (2.22 / 2.09).
  - Policy: val NLL **4.19**, test **4.18** (vs `human_prior_fullcov` 4.07 / 4.10 — slightly less human-like by design, search-distilled toward higher value).
  - Game strength: **beats** `csas_v3` `mcts_horizon/h10` at ~**59–64%** (h2/h6/h10) and edges the human prior; ≈0.51 vs the noise-free `anchor_v3` (execution-noise version is marginally stronger / more robust).
- **Verdict:** strongest model to date; serves as the warm-start and head-to-head reference for subsequent iterations.

---

## EXP-002 — `az/iter1` (KR-UCT tree + terminal-MC value)

- **Checkpoint:** `checkpoints/csas_world/az/iter1/model.pt`
- **Config:** `configs/anchor_mcts.yaml`; driver `scripts/az_converge.py` (collect → train → head-to-head vs previous → converge), warm-started from `anchor_noisy`.
- **Goal:** test whether a *real* multi-ply continuous-action search, with sound (terminal-grounded) value targets, improves on `anchor_noisy`.
- **What changed vs EXP-001**
  1. **Real multi-ply KR-UCT tree** (`src/world/search/kr_uct_tree.py`) against the authoritative simulator — progressive widening + kernel-regression UCT, leaves evaluated by on-policy MC rollout to terminal. Policy target = value-weighted soft-top-k over the searched root actions (`n_sims=48`, `roots=80`, horizons 1–10).
  2. **Value targets = realized terminal-MC `ValueDiff`** (rolled to the end under the policy), and **`value_from_mcts=true`** so the value head trains on those returns.
  3. **Per-step reward head removed**; tactical signal carried only by the distributional **outcome** head (curling reward is terminal-only ⇒ a per-step reward telescopes to the value difference, i.e. redundant with the value head).
  4. **Free-guard-zone rule corrected** (across `csas_v3` + `csas_fixed_moreMCTS`): illegal only if an *opponent* rock is *removed from play* before the 4th thrown stone (movement/contact and own-rock removal are legal); first 3 thrown stones protected (`horizon ≥ 8`). Contaminated MCTS data + the `sim` replay buffer were regenerated under the fixed rule.
- **Training (final):** train total 4.82 (policy_distill 4.30, value_mse 0.42, value_nll −0.33, outcome 0.61, consistency −0.42, decoder 0.11). **Val: policy NLL 4.38, value MSE 3.13, value NLL 1.58.**
- **Head-to-head vs `anchor_noisy`** (noisy decision-value selection, `n_candidates=48`, `sel_noise_samples=8`, full rollout to terminal):

  | Horizon | Win rate | Δscore |
  |---|---|---|
  | h04 | 0.567 | −0.050 |
  | h08 | 0.483 | −0.567 |
  | **avg** | **0.525** | **−0.308** |

  Within the ±0.04 convergence band ⇒ the loop declared convergence and **stopped at iter 1**.
- **Verdict:** **did not beat `anchor_noisy`.** Two issues:
  - **Value head regressed** (val MSE 2.18 → **3.13**) because `value_from_mcts=true` trained it on single-rollout terminal-MC returns (unbiased but very high variance) instead of the clean realized-`ValueDiff` buffer used in EXP-001.
  - Because head-to-head selection uses **each model's own value head**, the degraded value head **confounds** the result: the new model selected shots with a worse evaluator, so the −0.308 Δscore cannot be cleanly attributed to the policy.
- **Run cost:** collection ≈ 6.5 h (h07–h10 dominate at `n_sims=48` MC-to-terminal leaves) + train + h2h; ~7–8 h wall-clock for the single iteration.
- **Next:** two candidate follow-ups — (1) stronger search (`n_sims`↑, expensive) and (2) decouple the value head (`value_from_mcts=false`, train on the clean buffer; cheap, fixes the regression and the confound).

---

## EXP-003 — `az/iter1_valclean` (option 2: clean-buffer value head)

- **Checkpoint:** `checkpoints/csas_world/az/iter1_valclean/model.pt`
- **Config:** `configs/anchor_mcts_v2.yaml`; driver `scripts/run_value_clean.py` (train → export → head-to-head). **Reuses EXP-002's KR-UCT collection** (`artifacts/replay/mcts/az_iter1`) — no re-collection.
- **Goal:** test whether decoupling the value head (train on the realized-`ValueDiff` buffer instead of the noisy terminal-MC search returns) fixes the EXP-002 value regression and the head-to-head confound.
- **What changed vs EXP-002:** **only** `value_from_mcts` true → false. Everything else identical (same KR-UCT-distilled policy target, outcome head, fixed FGZ rule, warm-start from `anchor_noisy`). Confirmed the value buffer loads: training sources value=85,676 / sim=800 / mcts=2,960 records.
- **Training (final, epoch 24):** val policy NLL **4.51**, val value MSE **2.30** (epoch 0 was 2.16; drifts up slightly under joint training), val value NLL 0.85; train_total 4.13.
- **Head-to-head vs `anchor_noisy`** (same protocol as EXP-002):

  | Horizon | Win rate | Δscore |
  |---|---|---|
  | h04 | 0.533 | −0.167 |
  | h08 | 0.517 | −0.217 |
  | **avg** | **0.525** | **−0.192** |

- **Verdict:** soundness fix validated, but **still does not beat `anchor_noisy`**.
  - **Value head restored:** val MSE **3.13 → 2.30**, back near the anchor's 2.18 ⇒ the EXP-002 regression was indeed caused by training value on noisy terminal-MC returns.
  - **Confound confirmed & removed:** with *only* the value-target source changed, Δscore improved **+0.116** (−0.308 → −0.192), i.e. that much of EXP-002's deficit was the new model selecting shots through a degraded value head (decision-value selection uses each model's own value head).
  - **Still parity:** win rate 0.525, Δscore still marginally negative. With a sound value head, the `n_sims=48` KR-UCT search-distilled policy ≈ `anchor_noisy`; the search is not strong enough for its distillation target to beat the policy's own samples.
- **Run cost:** ~1 h (train + export + h2h; collection reused from EXP-002).
- **Next:** option 1 on this clean-value setup — raise `n_sims` (48 → 120+, optionally cap high-horizon roots to bound cost) so the search-distilled policy target is meaningfully stronger than the policy itself. Keep `value_from_mcts=false`.

---

## EXP-004 — `az_v3/iter1` (option 1: stronger search, clean-buffer value)

- **Checkpoint:** `checkpoints/csas_world/az_v3/iter1/model.pt`
- **Config:** `configs/anchor_mcts_v3.yaml`; driver `scripts/az_converge.py` via `scripts/_az_run_v3.sh` (`--work checkpoints/csas_world/az_v3`, fresh `az_v3_*` MCTS namespace). Warm-started from `anchor_noisy`.
- **Goal:** with the value head fixed (EXP-003), test whether a **stronger** KR-UCT search produces a distilled policy that beats `anchor_noisy`.
- **What changed vs EXP-003:** `mcts_sims` **48 → 120** (2.5× simulations per decision), full roots at every horizon 1–10 (no high-horizon cap); `value_from_mcts=false` kept. Fresh collection (not reused).
- **Training (final, epoch 24):** val policy NLL **4.46**, val value MSE **2.28** (value head stayed healthy — clean-buffer fix held), val value NLL 0.84; train_total 4.20.
- **Head-to-head vs `anchor_noisy`** (same protocol):

  | Horizon | Win rate | Δscore |
  |---|---|---|
  | h04 | 0.450 | +0.017 |
  | h08 | 0.467 | −0.350 |
  | **avg** | **0.458** | **−0.167** |

  Within the ±0.04 band ⇒ converged, **stopped at iter 1**.
- **Verdict:** **stronger search did not help — did not beat `anchor_noisy`.** Win rate 0.458 (anchor won ~54%), Δscore −0.167. With ~120 h2h games (std ≈ ±0.045), EXP-003 (0.525), EXP-004 (0.458) and 0.50 are mutually within noise ⇒ the honest reading is **both `n_sims` budgets sit at parity-or-below; 48→120 gave no detectable improvement.** Not a value confound (value MSE 2.28). The "search too weak" hypothesis is **not supported**.
- **Run cost:** collection ~13.3 h (03:30→16:51, `n_sims=120` MC-to-terminal leaves) + ~1 h train/h2h ≈ ~14.5 h.
- **Conclusion / next:** the iterative KR-UCT search-distillation from `anchor_noisy` has now been tested at `n_sims` ∈ {48, 120}, with the value head both broken and fixed, and **none beat the anchor**. The improvement operator itself (per-root, on-policy MC-to-terminal search distilled into the policy) appears to have plateaued at the anchor. **`anchor_noisy` remains the best model.** Beating it likely needs a different lever, not more search.
- **Update (candidate-diversity probe, `scripts/policy_diversity.py`):** the "distillation over-peaks the policy" hypothesis is **refuted**. Sampling the 48 deployment candidates (temp 1.1, std 1.2) at 30 boards, rel-spread (candidate std / dataset action std) = human 0.91, anchor 1.00, **az_v3 1.08**; mean pairwise z-dist = 2.48 / 2.70 / 2.93. The trained policies propose candidates *as diverse as or more diverse than* the human prior — diversity is **not** the bottleneck. Remaining suspects: value/selection discrimination among the (diverse) candidates, the anchor being near the pipeline ceiling, and the ~±0.045 h2h noise floor masking sub-0.05 gains.

---

## EXP-005 — `anchor_noisy_simsiam_stopgrad_no_outcome`

- **Checkpoint:** `checkpoints/csas_world/ablations/anchor_noisy_simsiam_stopgrad_no_outcome/{best,last,model}.pt`
- **Config:** `configs/ablations/anchor_noisy_simsiam_stopgrad_no_outcome.yaml`; driver `scripts/train_world.py`.
- **Goal:** replace the EMA target encoder with an EfficientZero/SimSiam-style direct online encoder under stop-gradient, while removing the end-outcome distribution head.
- **What changed vs EXP-001:**
  - Warm-started from the trained `anchor_noisy/model.pt` and continued for 25 epochs on the same value, simulator, and `anchor_noisy` MCTS replay.
  - `model.use_outcome=false` and `loss.outcome=0`.
  - `consistency_mode=simsiam`, with `ema_decay=1.0`: no target-trunk parameters are created; the online trunk encodes consistency targets inside `torch.no_grad()`.
  - All other active losses and data sources were retained. Training used GPUs 0 and 1 with `gloo` DDP, batch 256.
- **Command:** `PYTHONPATH=src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python scripts/train_world.py --config configs/ablations/anchor_noisy_simsiam_stopgrad_no_outcome.yaml --mcts-dir artifacts/replay/mcts/anchor_noisy --sim-dir artifacts/replay/sim --init checkpoints/csas_world/anchor_noisy/model.pt --out checkpoints/csas_world/ablations/anchor_noisy_simsiam_stopgrad_no_outcome --run-name anchor_noisy_simsiam_stopgrad_no_outcome`
- **Best checkpoint (epoch 1):** val policy NLL **4.301787**, value MSE **2.216401**, value NLL **0.816465**.
- **Final checkpoint (epoch 24):** val policy NLL **4.501451**, value MSE **2.318724**, value NLL **0.870550**; train consistency **-0.138934**.
- **Verdict:** did not improve the anchor. Even its best continuation checkpoint is worse than the original anchor's final policy NLL 4.192 and value MSE 2.180; continued training then overfits further. EXP-006 isolates whether this is caused by removing EMA.
- **Caveat:** this is a continuation from an already-converged anchor, not a retraining from the anchor's original initialization. A same-duration EMA+outcome continuation control would be needed to separate harm from outcome-head removal from generic continued-training drift.

---

## EXP-006 — `anchor_noisy_ema_no_outcome`

- **Checkpoint:** `checkpoints/csas_world/ablations/anchor_noisy_ema_no_outcome/{best,last,model}.pt`
- **Config:** `configs/ablations/anchor_noisy_ema_no_outcome.yaml`; driver `scripts/train_world.py`.
- **Goal:** isolate removal of the end-outcome distribution head while retaining the original EMA target encoder.
- **What changed vs EXP-005:** only restored the original EMA target trunk with `ema_decay=0.99`; the outcome head remained disabled. Data, initialization, losses, GPUs, batch size, and 25-epoch schedule were identical.
- **Command:** `PYTHONPATH=src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python scripts/train_world.py --config configs/ablations/anchor_noisy_ema_no_outcome.yaml --mcts-dir artifacts/replay/mcts/anchor_noisy --sim-dir artifacts/replay/sim --init checkpoints/csas_world/anchor_noisy/model.pt --out checkpoints/csas_world/ablations/anchor_noisy_ema_no_outcome --run-name anchor_noisy_ema_no_outcome`
- **Best checkpoint (epoch 1):** val policy NLL **4.301817**, value MSE **2.216399**, value NLL **0.816473**.
- **Final checkpoint (epoch 24):** val policy NLL **4.501528**, value MSE **2.318723**, value NLL **0.870484**; train consistency **-0.139348**.
- **Verdict:** numerically indistinguishable from EXP-005. At the final checkpoint, direct stop-gradient vs EMA differs by only **0.000077** policy NLL and **0.000001** value MSE. Under this warm-start and low trunk learning rate, EMA lag has no measurable effect. The shared regression is therefore not evidence against SimSiam-style targets; it is associated with the common no-outcome continuation setup.
- **Conclusion:** retain the original `anchor_noisy` checkpoint. These ablations provide no reason to replace EMA, and the outcome head remains a plausible useful auxiliary regularizer. To attribute causality cleanly, run either (a) EMA+outcome continuation for 25 epochs or (b) all variants from the same pre-anchor initialization.

---

## EXP-007 — controlled iterative outcome-head comparison

- **Checkpoints:**
  - Outcome on: `checkpoints/csas_world/ablations/outcome_controlled/outcome_on/best_game.pt` (round 1, epoch 34).
  - Outcome off: `checkpoints/csas_world/ablations/outcome_controlled/outcome_off/best_game.pt` (round 0, epoch 29).
- **Configs:** `configs/ablations/anchor_iterative_outcome_{on,off}.yaml`; driver `scripts/run_outcome_iterative_ablation.py`.
- **Goal:** isolate whether the terminal end-outcome classification auxiliary improves the already-trained `anchor_noisy` policy/value model.
- **Correctness fixes made before comparison:**
  - Root outcome is supervised from the root perspective.
  - Recurrent outcome is supervised only on real simulator transitions, not value-only/fake unroll steps.
  - The terminal margin sign is converted explicitly when the recurrent state's team perspective differs from the root.
  - Outcome loss uses a global valid-element denominator rather than averaging invalid recurrent steps.
  - Checkpoints now retain and restore AdamW state, global step, and any saved target-encoder state. The anchor's `simsiam` consistency mode is held fixed in both arms.
  - Legacy `reward_head.outcome_head.*` weights are mapped into the current standalone outcome head. Loaded parameters, including the trained outcome head, use the continuation learning rate; selected runs reported **zero fresh parameters**.
- **Controlled training protocol:**
  - Both arms warm-start from the identical trained `anchor_noisy/model.pt`; round 0 uses identical fresh MCTS replay.
  - Only the outcome head/loss differs. BC is disabled (`policy_bc=0`, no human-policy replay); BC remains initialization only.
  - No scheduler or warmup. Both pretrained and loaded-head learning rates are `2e-5`.
  - Train for exactly five epochs, evaluate game strength, and collect fresh noisy MCTS targets before another block if not converged. No 25-epoch reuse of stale replay.
  - AdamW state is preserved between blocks. Training/collection uses GPUs 0 and 1.
  - Checkpoint selection uses full terminal rollout win rate, with score differential as tie-breaker, against `anchor_noisy`. Both policies rank actions with the same frozen anchor value evaluator, removing value-head drift as a policy-strength confound.
- **Historical optimizer limitation:** the June 8 `anchor_noisy` checkpoint was written before optimizer serialization was implemented and contains no Adam moments. They cannot be reconstructed from model weights. Therefore both arms symmetrically start round 0 from the same trained weights with a new optimizer; from round 0 onward every checkpoint contains and restores the complete optimizer state and global step (137 parameter states without outcome, 141 with outcome).
- **Selected-checkpoint training diagnostics:**

  | Arm | Cumulative continuation epochs | val policy NLL | val value MSE | val value NLL |
  |---|---:|---:|---:|---:|
  | outcome on | 10 | 6.177 | 2.178 | 0.802 |
  | outcome off | 5 | 5.371 | 2.162 | 0.792 |

  Policy NLL was not used for selection because search distillation intentionally moves the policy away from human BC likelihood.
- **Model-selection rollout set** (192 ends per checkpoint; horizons 2/5/8/10):

  | Arm vs `anchor_noisy` | Win rate | Δscore |
  |---|---:|---:|
  | outcome on | 0.521 | +0.182 |
  | outcome off | 0.513 | +0.089 |

  Outcome on then failed to improve for two fresh-target blocks (0.505/+0.104 and 0.495/+0.109). Outcome off likewise failed twice (0.510/+0.063 and 0.503/+0.146), so both stopped by game-strength patience.
- **Independent confirmation** (new seed, 320 ends per comparison; horizons 2/5/8/10):

  | Matchup | Win rate for first model | Δscore |
  |---|---:|---:|
  | outcome on vs `anchor_noisy` | 0.497 | +0.019 |
  | outcome off vs `anchor_noisy` | 0.538 | +0.156 |
  | outcome on vs outcome off | **0.500** | **-0.025** |

  Full results: `checkpoints/csas_world/ablations/outcome_controlled/confirmation_seed361224/confirmation.json`.
- **Verdict:** the outcome head gives **no measurable game-strength improvement**. The direct comparison is exactly 50% and the small ordering against the anchor changes across seeds, so the selection-set difference was rollout noise. The no-outcome model works correctly with the existing Gaussian value head and is the simpler default. The independent outcome-off result above 50% is promising but not statistically decisive at 320 ends (win-rate standard error is about 2.8 percentage points).
- **Why longer continuation regressed:** fresh MCTS targets still define a noisy moving policy objective. More optimization improves fit to each search batch but does not monotonically improve terminal play; game-strength selection correctly rejected later checkpoints even while value MSE remained near 2.16-2.20.
- **Update 2026-06-16:** `Config()` plus the standard `base`, `anchor`, `anchor_noisy`, `anchor_az`, and `anchor_mcts*` configs now default to `use_outcome=false` and `loss.outcome=0.0`. Outcome-head training is opt-in through explicit ablation configs such as `anchor_iterative_outcome_on.yaml` or `full.yaml`.

---

# FULL-SHEET ERA (release 28.35 m, from 2026-06-16)

The sim/data/action-space migrated to the full sheet (see memory `curling-fullsheet-migration`).
New human prior `human_prior_fullcov_fullsheet` (physical NLL −2.40 ≈ truncated prior). Note:
full-sheet **policy NLL is ~6.6**, not comparable to the truncated ~4.2 (tighter action marginal,
different z-normalisation); use it only for relative comparisons within the full-sheet era. The
value model (`value/holdout0`) is sheet-agnostic (real-data ValueDiff), MSE baseline still ≈ **2.22**.

## Diagnosis — full-sheet curriculum did not improve past h01

First full-sheet curriculum (`curriculum_fullsheet`, 4-GPU) ran the **untuned `Config()` defaults**, not the
tuned `anchor_mcts*` recipe. h2h winrate vs the running champion: h01 0.53/0.57 (won), then h02 0.40/0.38,
h03 0.40, h04 0.33/0.40 — i.e. every stage after h01 *lost* to the early champion. Root causes (code-read):
- **Shallow search:** defaults have `use_mcts_tree=false` → a 1-ply value-greedy candidate pick, not the
  multi-ply KR-UCT tree. The distillation target is barely better than the policy itself.
- **Value model double-duty:** the 1-ply scoring AND the n-step bootstrap both use the *frozen baseline*
  value model; the trained world value head is never fed back into collection (loop not closed).
- **`policy_bc=1.0`** pins the policy to the human prior, so even a good target can't move it far.
- **Confound (per glossary):** h2h decision-value selection uses *each model's own value head*. The
  curriculum value MSE (~2.26) is slightly above baseline (2.22), so the trained model is *handicapped at
  selection* — some of the sub-0.5 winrate may be a value-head gap, not a worse policy.
- Augmentation was **already** applied to all replay sources (flip + team-swap, p=0.5) — not a culprit.
EXP-002/004 already showed the tuned KR-UCT recipe hit parity-or-below on the truncated sheet, so a plateau
is plausible even tuned; EXP-008/009 test it cleanly on the full sheet.

## EXP-008 — `exp_a_tuned` (full-sheet tuned recipe, 4-horizon A/B) — RUNNING

- **Checkpoint:** `checkpoints/csas_world/exp_a_tuned/h{02..05}/r0/model.pt`
- **Config:** `configs/exp_a_tuned_4h.yaml` ; driver `scripts/_run_curriculum_fullsheet.sh` (4 GPUs, GPU sim)
- **Goal:** does the tuned recipe break the full-sheet plateau at the exact horizons the default failed (h2-5)?
- **What changed vs the failed default run:** `use_mcts_tree=true` (multi-ply KR-UCT, value-model-FREE
  terminal-MC-rollout leaves, `mcts_sims=96`); `policy_bc=0` (pure search distillation, `mix_human=0`);
  `value=1.5, consistency=1.0, noise_samples=8`. Horizons 2-5, 1 round/stage, 64 roots, h2h 40/order.
- **Hypotheses tested:** #1 deep search, #3 drop policy_bc, value-model-free targets (the "double-duty" fix).
- **Result (2026-06-17):** **fixes the regression → parity with the prior.** Winrate vs champion: h02 0.537,
  h03 0.500, h04 0.500, h05 0.550 (overall ≈0.52), up from the default run's 0.40/0.40/0.33. So the tuned
  recipe stops the curriculum from *degrading* the model, but does not produce a decisive gain.
- **Caveats:** (a) order split is extreme and flips with horizon parity (e.g. h05 o0 0.88 / o1 0.23) — the
  h2h is **hammer-saturated** (near-equal models → last-stone team wins ~85%), so ±0.05 at 40 ends/order is
  noise; (b) `val_policy_nll` rose to ~14.5-15.3 (vs prior 6.6) because `policy_bc=0` lets the policy leave
  human-likeness — expected, not a strength signal. Echoes EXP-002/004's truncated parity-or-below.

## EXP-009 — `exp_b_reward` (EXP-008 + 2-step reward head, 4-horizon A/B) — QUEUED

- **Checkpoint:** `checkpoints/csas_world/exp_b_reward/h{02..05}/r0/model.pt`
- **Config:** `configs/exp_b_reward_4h.yaml` (= EXP-008 + `use_step_reward=true, step_reward=0.5, collect_step_reward=true`)
- **Goal:** does disentangling the near-term signal from the terminal value head help?
- **What changed vs EXP-008:** adds a scalar **2-step-return reward head** (`heads/reward_head.py`),
  trained (Huber) on the 2-step return = rule margin if the end ends within 2 plies, else the value model 2
  plies ahead (`collect._two_step_rewards`, gated by `collect_step_reward`). Auxiliary head; search + h2h
  unchanged vs EXP-008, so it isolates the effect of the extra grounded near-term signal on the trunk.
- **Result (2026-06-17):** **reward head improves the value head; game strength stays at parity.**
  Reward Huber loss trained 0.318 → 0.097 (head fits the 2-step return well). Winrate vs champion: h02
  0.575, h03 0.500, h04 0.500, h05 0.487 (≈0.52 — same parity as EXP-008). **Value MSE 2.200 vs EXP-008's
  ~2.26 and the 2.22 baseline** → the auxiliary 2-step signal moved the value head from *below* baseline to
  *above* it.
- **Verdict:** the 2-step reward head is a **real, measurable value-head gain** (validates the "value
  double-duty" hypothesis), and it de-confounds the h2h (its value head is no longer the weaker one), but it
  **does not break the game-strength parity** with the prior. Both EXP-008/009 reach the prior's strength,
  neither exceeds it — the bottleneck is no longer search depth or value double-duty.
- **Recommended next:** (a) **close the value loop** — feed this improved value head back into collection
  (`--value-world`; horizon_loop currently always uses the frozen baseline), so better value → better search
  targets → policy gains; (b) **hammer-controlled, higher-power eval** — the h2h is hammer-saturated (order
  splits like 0.90/0.07 that flip with horizon parity), so resolve sub-5% gains with many more ends and
  paired-by-hammer scoring before trusting any "improvement". Keep the reward head on (it's a free value gain).

## EXP-010 — `exp_c_valueloop` (close the value loop, 4-horizon) — RUNNING

- **Checkpoint:** `checkpoints/csas_world/exp_c_valueloop/h{02..05}/r{0,1}/model.pt`
- **Config:** `configs/exp_c_valueloop_4h.yaml` ; driver `scripts/_run_curriculum_fullsheet.sh` (4 GPUs)
- **Goal:** does closing the value loop break the EXP-008/009 parity? (recommended follow-up from EXP-009)
- **What changed vs EXP-009:** the KR-UCT tree leaves are now **bootstrapped by the model's own value
  head** (`search.value_leaf_bootstrap=true`, AlphaZero-style) instead of terminal MC rollouts; and
  `horizon_loop` feeds the **trained stage checkpoint's value head** into collection via `--value-world`
  (new `parallel_collect(value_world_ckpt=...)`), so the improving value guides the search. `rounds_per_stage=2`
  so round 0 (baseline value-leaf) → round 1 (trained value head) closes the loop. Reward head kept on.
- **Mechanics verified** (smoke): round 0 logs `tree leaves bootstrapped by the value head`; round 1 logs
  `value bootstrap = world value head` (the `--value-world` closure fires). Gated — EXP-008/009 unaffected.
- **Result (2026-06-17):** **best of the three; positive but sub-resolution.** Round 1 (closed loop) beat
  round 0 (baseline value) in 3/4 horizons: h02 .525→.537, h03 .537→**.588**, h04 .525→.487, h05 .500→.525.
  h03 r1 0.588 > 0.54 → became the champion (only round to beat the running best). **Value MSE ~2.192**
  (best across all exps: EXP-A 2.26, EXP-B 2.20, baseline 2.22). Reward Huber 0.35→0.075. Overall ≈0.53.
- **Verdict:** the closed loop is the strongest configuration — it produces the best value head and round-1 >
  round-0 (the loop self-improves within a stage) — but the overall winrate (~0.53) is still within noise of
  parity at 40 ends/order. Cumulative trend across EXP-008→010 is upward (value 2.26→2.19; win 0.52→0.53)
  yet sub-resolution. **The bottleneck is now the evaluation, not the training.** Next: high-N, both-orders
  (hammer-balanced) h2h of the EXP-010 champion (`h03/r1`) vs the prior to resolve whether the ~3% edge is real.
- **DECISIVE EVAL (2026-06-17, `scripts/_eval_highN.py` → `exp_c_valueloop/highN_eval.json`):** champion `h03/r1`
  vs the prior, 956 held-out games, both orders (hammer-balanced). **Overall winrate 0.5335 ± 0.0161 →
  BEATS the prior at >2 SE** (95% CI ≈ [0.501, 0.566]). Per-horizon: h02 0.511±.031 (dScore +0.12), h03
  **0.559±.032 (+0.22)**, h04 **0.555±.034 (+0.21)**, h05 0.513±.033 (−0.06). The gain is real but **modest
  and marginal** (CI lower bound barely above 0.5), concentrated at the **mid horizons** (h3/h4); the
  simplest (h2) and hardest in-regime (h5) are ~parity. **Conclusion:** the full recipe — deep KR-UCT search
  + 2-step reward head + closed value loop — is the first to *exceed* the human prior (not just match it); the
  earlier in-loop "parity" was the 40-ends/order h2h lacking the power to resolve a ~3% edge.

## EXP-011 — `full_valueloop` (full deployable model, all horizons, to convergence) — RUNNING

- **Checkpoint:** `checkpoints/csas_world/full_valueloop/` (champion = `curriculum_summary.json` best_ckpt)
- **Config:** `configs/full_valueloop.yaml` ; driver `scripts/_run_curriculum_fullsheet.sh` (4 GPUs); base=None
- **Goal:** train the EXP-010 winning recipe across **all horizons 1→10** to a deployable champion.
- **What changed vs EXP-010:** horizons 1-10 (was 2-5); `rounds_per_stage=3`; `roots_per_stage=80`;
  **convergence now also accepts a clear score-margin win** (`horizon.converge_margin_band=0.10` — new in
  config/horizon_loop: `stronger = winrate>0.5+band OR Δscore>margin_band`); `h2h_games_per_order=80`
  (40 was too noisy for reliable convergence). Everything else = EXP-010 (closed value loop + reward head).
- **Result:** **PAUSED 2026-06-17** after h01 (converged ~parity vs prior: r0 0.512, r1 0.500) — paused to
  run EXP-012 first. Resume: `bash scripts/_run_curriculum_fullsheet.sh configs/full_valueloop.yaml
  checkpoints/csas_world/full_valueloop artifacts/replay/sim_none --base
  /mnt/data/curling2/csas_world/checkpoints/csas_world/full_valueloop/h01/r1/model.pt --start 2`.

## EXP-012 — `exp_d_reward_sa` (action-conditioned r(s,a) reward head, 4-horizon) — DONE (noisy eval pending)

- **Checkpoint:** `checkpoints/csas_world/exp_d_reward_sa/h{02..05}/r{0,1}/model.pt`
- **Config:** `configs/exp_d_reward_sa_4h.yaml` ; driver `scripts/_run_curriculum_fullsheet.sh` (4 GPUs); base=None
- **Goal:** does an ACTION-conditioned reward head `r(s,a)` beat EXP-010's STATE-conditioned 2-step value head?
- **What changed vs EXP-010:** the reward head now regresses the 2-step return from the **post-action latent**
  `G(s_k,a_k) = steps[k+1]` instead of the state latent `steps[k]` (new gated `loss.reward_action_conditioned`;
  EXP-010/011 keep the state-conditioned default). Identical otherwise (closed value loop, h2-5, 2 rounds/stage).
- **Note:** reward head is still auxiliary (not yet wired into search/Q); this tests whether the corrected
  MuZero-style `r(s,a)` target is a better trunk/value signal than the state-conditioned version.
- **Result (2026-06-17):** **best value head yet, mixed in-loop winrate.** `val_value_mse=2.188` — best of
  all (EXP-008 2.26 → 009 2.20 → 010 2.192 → **012 2.188**), so action-conditioning further sharpens the value
  head. Reward Huber 0.36→0.072. In-loop winrate: h02 ~0.52, h03 ~0.52, **h04 r1 0.600** (champion), but
  **h05 regressed to ~0.41**. Champion `h04/r1`. Deterministic + non-noise-averaged in-loop numbers, so the
  decider is the **noisy high-N eval** (`scripts/_eval_noisy_compare.sh`): EXP-012 vs EXP-010 (reward-head
  choice) + both vs prior under noisy execution.
- **NOISY high-N verdict (2026-06-17, n=800/comparison, SE 0.018):** action- vs state-conditioned is a
  **wash** — EXP-012 vs EXP-010 = **0.4825 ± 0.018 (parity)**. And under noisy execution **neither beats the
  prior**: EXP-012 vs prior **0.5044**, EXP-010 vs prior **0.5031** (both parity). → **Decision: use the
  STATE-conditioned reward head** (action-conditioning gave nothing; matches the "default B" rule).
- **MAJOR caveat — supersedes EXP-010's deterministic claim:** EXP-010's "0.5335 beats prior at >2 SE" was
  measured **deterministically with non-robust selection**; under realistic noisy execution it **collapses to
  parity (0.503)**. So the entire tuned/closed-loop/reward-head program reaches **parity with the human prior
  under noise — it does not beat it.** This is exactly why EXP-013/014 (noise-robust selection, restored since
  it was lost at EXP-002) are the real test.

## EXP-013 — `exp_013_reward_robust` (1-ply-robust r̂₂ selector) — RUN DONE, noisy eval running

- **Config:** `configs/exp_013_reward_robust.yaml`; champion `exp_013_reward_robust/h05/r1/model.pt`.
- **Goal:** the actual test of **noise-robust selection** (lost since EXP-002) — does selecting shots that
  survive execution noise beat the prior under noise, where the non-robust loop only reached parity?
- **What changed vs EXP-010/012:** **no deep tree.** Collection distillation target = soft-top-k of candidates
  scored by the **mean over K=8 noisy executions of −r̂₂(post)** (state-cond 2-step reward head; round 0 falls
  back to −V, round 1+ uses the trained head via `--value-world`). This is the 1-ply-robust ego selection
  (`search.reward_leaf_select`, `use_mcts_tree=false`, `noise_samples=8`). In-loop h2h NOISY (`noisy_h2h=true`).
  New gated code: `reward_leaf_select` + `_WorldReward` adapter + `reward_model` threaded through `_raw_q`.
- **Scope (v1):** robust selection applied at **collection** (shapes the distilled policy); the h2h player
  still re-ranks by V; reward target uses the existing (policy-sampled-opponent) 2-step return. The
  searched-opponent target and deployment-side r̂₂ re-rank are v2 refinements if v1 shows promise.
- **In-loop (NOISY, 30/order, vs running champion):** h02 .50/.47, h03 .47/**.60**, h04 .47/.42, h05 .52/**.55**
  — produced champions at h03 r1 and h05 r1. Low-N + vs-champion, so decisive read = the high-N noisy eval.
- **Result (2026-06-18) — FIRST genuine win under noise.** High-N NOISY eval (n=700, SE 0.019): EXP-013 vs
  **prior = 0.5407 ± 0.019 → BEATS (>2SE)** (h02 .564, h03 .554, h04 .507, h05 .557, **h08 .521** [FGZ]; all
  Δscore positive, +0.13..+0.33 mid-game). EXP-013 vs **EXP-010** (non-robust closed loop) = **0.5446 ± 0.021
  → BEATS (>2SE)**. So **noise-robust selection (lost since EXP-002) is the lever that breaks the plateau**:
  where EXP-010/012 were dead parity under noise (0.503/0.504), EXP-013 beats both the prior and EXP-010.
- **Verdict:** modest but real (winrate lower-CI ~0.503; Δscore the stronger signal). **First config to beat the
  prior under realistic execution noise.** Validates: *always select robustly (average over execution noise).*
- **Next (think-again options):** (a) EXP-014 (robust terminal-rollout selector) to compare leaf evaluators;
  (b) v2 refinements — searched-opponent reward target + deployment-side r̂₂ re-rank; (c) scale the EXP-013
  recipe to the **full 1→10 deployable model** (it now beats the prior). Cheap collection (~20s/round) makes (c) attractive.

## EXP-014 — `exp_014_terminal_robust` (1-ply-robust value-greedy TERMINAL-rollout selector, no reward head) — OOM'd, SKIPPED per user

- **Config:** `configs/exp_014_terminal_robust.yaml` ; 4-horizon (2→5), `roots_per_stage=48`, `noisy_h2h`.
- **Goal:** contrast EXP-013's `r̂₂` leaf — select via a value-model-free **terminal rollout** (each candidate noise-averaged over K=4; rollout itself value-greedy: each ply picks best of `search_rollout_n=12` policy samples by `argmin V_next`). Only the terminal rule score, no reward head, no bootstrap.
- **Outcome:** h02 weak (winrate_vs_prev_best 0.567→0.500), then **CUDA OOM at h03 collection**: the value-greedy rollout built a ~9k-state GNN value-eval batch (192 cands × k_ego 4 × n_search 12) → 4.94 GiB curl-arc edge features. **Fixed** by chunking the eval to ≤2040 states in `_mc_rollout_terminal_batch` (collect.py) — fix stays in for any future terminal-rollout use. **Skipped the ~3h restart per user** (EXP-013 already beats the prior; the W-target line, EXP-015/016, took priority).

## EXP-015 — `exp_015_kernel_visits` (KR-UCT kernel-effective visit count W(a) as the POLICY target) — DONE: PARITY vs prior (NEGATIVE result), champion `h03/r1/model.pt`

- **Config:** `configs/exp_015_kernel_visits.yaml` ; 4-horizon (2→5), `roots_per_stage=48`, `noisy_h2h`.
- **Goal:** replace the value-softmax distillation target with **W(a) = Σ_b K(a,b)·n_b** — the continuous/stochastic analog of an AlphaZero visit target (Yee/Lisý/Bowling 2016, KR-UCT; curling is their domain). The kernel-mass selection is itself a noise-robustness mechanism: forgiving regions (good neighbours) accrue mass; knife-edges get suppressed.
- **Mechanism (new code):** depth-1 root KR-UCT bandit (`search_root_only`, `kr_uct_tree.mcts_search(root_only=True)`) with a **value-head leaf** + **fresh execution noise per visit** (revisits average execution uncertainty; the kernel pools across neighbours). `collect_root_record` exposes `W = kernel_effective_counts(...)` (the `eff_n` `kr_smooth_scores` previously discarded), then `π ∝ W^{1/τ}` via `soft_topk(log W, τ)` with **τ=1 → π ∝ W** (AlphaZero training convention); deployment-equivalent selection = `argmax W`. `kernel_bandwidth=0.30` for robust pooling.
- **Value head:** UNCHANGED — `value_from_mcts=false` trains it only on the realized-ValueDiff buffer (W is used for **policy only**). This is the EXP-016 contrast.
- **Validation:** kernel helpers unit-tested (closest-cluster action gets highest W; isolated suppressed); depth-1 bandit runs clean (40 sims → all visits accrued); real collection path writes healthy W-distill targets (`dist_weights` spread `[0.47,0.16,0.09]`, not collapsed) — refinements (τ=1, bw=0.30) avoid the over-sharpening the random-leaf toy showed.
- **RESULT (2026-06-18) — NOISY high-N (N=700, n≈196–266/horizon), via `scripts/_eval_parallel.py`:** W vs **prior** = **0.4939 ± 0.015 → parity, trending below**; per-horizon h02 0.515 / h03 0.504 / h04 0.518 / h05 0.475 / h08 0.449 (dScore +0.13 → −0.23). **Degrades at long horizons.** Clearly weaker than EXP-013 (0.541). **Conclusion: the kernel-effective-visit-count target as a POLICY target is a NEGATIVE result** — it does not beat the prior and underperforms robust value/reward *selection* (EXP-013). Interpretation: at depth-1, `W(a)` rewards where visits piled and the kernel over-smooths the policy toward safe-but-bland shots, discarding the fine value ranking that `argmax`-value selection keeps; hence the long-horizon decay. **Robust value/reward selection (EXP-013), not the visit-count target, is the lever.**

## EXP-016 — `exp_016_kernel_visits_value` (EXP-015 + kernel-regressed root value as the VALUE target) — BUILT + VALIDATED, SKIPPED per user

- **Config:** `configs/exp_016_kernel_visits_value.yaml` ; same as EXP-015 except the value target.
- **What changed vs EXP-015:** `value_target_kernel_root=true` writes the **kernel-regressed root value** `V̂_root = Σ_a W(a)V̂(a)/Σ_a W(a)` into the MCTS records, and `value_from_mcts=true` makes the value head train on it (vs EXP-015's realized buffer). Tests whether the smoother, noise-robust *kernel-regressed* search value avoids the search-value overfitting that made plain `value_from_mcts` lose historically.
- **Observable in h2h:** `WorldPlayer` scores candidates with its **own** trained value head, so the EXP-015↔016 value-target difference shows up at selection time — the ablation is meaningful.
- **Validation:** real collection path on CPU writes `value_target[0] = ±2.284` (non-integer kernel-regressed `V̂_root`, vs EXP-015's integer ±3 margin) — the value-target switch fires correctly.
- **Noise-averaging:** EXP-015/016 are noise-robust by construction — the depth-1 bandit re-executes each action under fresh execution noise on every visit (per-action averaging) **and** the kernel pools robust evidence across neighbouring actions; noisy h2h on (`noisy_h2h=true`).
- **Status:** built + validated (collection writes the non-integer kernel-regressed `V̂_root` value target correctly), but **SKIPPED per user** after EXP-015's W policy target came in at parity: the W-policy ceiling is parity, so EXP-016 (same policy + better value head) was judged unlikely to beat EXP-013. Config kept (`exp_016_kernel_visits_value.yaml`) if the value-target question is revisited.

## EXP-017 — `exp_017_deploy_robust` (scale the EXP-013 robust-r̂₂ recipe to the FULL 1→10 deployable model) — DONE: BEATS prior at full scale, champion `h07/r0/model.pt`

- **Config:** `configs/exp_017_deploy_robust.yaml`. Recipe IDENTICAL to EXP-013 (state-cond 2-step reward head, candidate value = mean over K=8 noisy executions of −r̂₂(post), EZ 1-ply path + noise averaging = the 1-ply-robust scorer, closed r̂₂ loop via `--value-world` round 1+, `use_mcts_tree=false`, `value_from_mcts=false`).
- **What changed vs EXP-013:** scaled to deployable — `start_horizon=1`, `max_horizon=10`, `roots_per_stage=80`, `h2h_games_per_order=80`, `rounds_per_stage=2`.
- **REPORTING NOTE (corrected per user):** the right metrics are **per-horizon WITH HAMMER** (model as the to-move team — hammer is fixed by horizon parity: model HAS it at ODD h, not at EVEN h) plus **odd+even PAIR averages** (h01+h02, … — equal weight, breaks the hammer = pure skill). Do NOT pool across horizons (end-weighting let the big-pool horizons dominate) and do NOT quote a single throw-order-averaged number. (The earlier "OVERALL 0.5224 end-weighted" is superseded.)
- **RESULT — NOISY high-N vs PRIOR (N=700, `_eval_parallel.py`, 4-GPU within-horizon sharded, per-order margins), champion `h07/r0/model.pt`:**
  - **Per-horizon (winrate w/ hammer | w/o hammer ; dScore w/ | w/o):** h01 .712|.344 (+0.99|−0.78), h02 .752|.346, h03 .725|.343, h04 .734|.385, h05 .797|.314 (+1.35|−0.91), h06 .792|.248, h07 .808|.192, h08 .867|.204 (+1.77|−1.45), h09 .821|.284, **h10 (PRE-PLACED) .823|.258 (+1.88|−1.25)**. The with/without-hammer split is large (hammer decisive), as expected.
  - **Hammer-neutral skill (equal-weight pairs):** h01+h02 **0.529** (+0.12), h03+h04 **0.555** (+0.28), h05+h06 **0.522** (+0.09), h07+h08 **0.506** (+0.03), h09+h10 **0.540** (+0.11) → **mean 0.530**, every pair ≥ 0.50, all dScore positive. **Beats the prior on pure skill across the whole end, including the deepest real horizon (h09+h10, with h10 = the true pre-placed start-of-end).**
- **CAVEAT — late-stage GPU-0 OOM corrupted h08+ training (since FIXED in trainer.evaluate):** the original run's h08 val-eval OOM'd (GPU 0 = rank-0 train + collection + h2h; the eval loaded the whole val split onto GPU 0). Fix applied: `trainer.evaluate` now `empty_cache()` + per-batch device moves + eval bs 512→256. Champion `h07/r0` predates the corruption (clean, value MSE 2.23) and generalizes; the h8/h9 resume stages trained cleanly but didn't beat it.
- **The "h10 collection bug" was DATA COVERAGE, now FIXED:** `throws_remaining==10` = the first thrown stone, which begins from PRE-PLACED stone states (standard/pp_left/pp_right) the annotators skipped. Ported pre-placed generation into `src/world/preplaced.py` (canonical boards via csas full-sheet `compact_m_to_raw`, dead=POS_MAX) and wired `build_roots(include_preplaced=)` + `h2h_eval` to serve them at h10. Verified: h10 collection now writes full records (was 5/shard); h10 eval here used the 387 real pre-placed val states. Config: `max_horizon=10`, `include_preplaced=true`.
- **Eval tooling:** `_eval_parallel.py` now shards **within** each horizon across GPUs (h10 went from ~5 h single-GPU to part of a 64-min full run) and reports the per-horizon hammer split + pairs; `head_to_head` returns per-order margins; `h2h_eval`/`_eval_highN` take `--root-shard`.
- **h10-EXTENSION RESUME (2026-06-20) — h10 trains clean, but causes late-horizon FORGETTING.** Resumed from `h09/r1`, `--start 10`, pre-placed h10 (`include_preplaced=true`, `max_horizon=10`). h10 trained **cleanly this time** (no divergence/OOM/nan; full 20/shard both rounds — the data-coverage + `trainer.evaluate` OOM fixes held). h10 r0/r1 winrate vs prev_best 0.487/0.500 (didn't beat the carried champion). **NOISY 1→10 eval of the h10-trained `h10/r1` vs prior (pre-placed h10):** hammer-neutral pair skill — h01+h02 0.544, h03+h04 0.546, h05+h06 0.537, **h07+h08 0.461↓, h09+h10 0.487↓**, MEAN **0.515** (vs h07/r0's 0.530). So extending to h10 nudged the early game up but **regressed the late game** (h07+h08 0.506→0.461 on comparable human roots, ~1.5 SE) — classic curriculum forgetting. **`h07/r0` (0.530) remains the stronger all-rounder; `h10/r1` only adds true h10 (opening) coverage at the cost of h7–10.** Pre-placed h10 eval itself: model w/ hammer 0.786, no-hammer 0.238 (hammer decisive at the opening).
- **VIZ:** `scripts/viz_game_match.py` (generalizes `viz_game.py` to any player pair + winrate-faithful `--noisy-select`/`--realize-noise`). Rendered full-end (pre-placement + 10 throws) games for `h10/r1`: `artifacts/figures/game_ours_vs_ours` (self-play, clean) + `game_ours_vs_prior` (winrate setup). Reference style = `game_anchor_vs_v3`.
- **Next:** to get a strong full-end model WITHOUT forgetting: (a) fresh EXP-017 1→10 with **horizon mixing** (sample roots from all horizons each round, or replay earlier-horizon records during h8–10) instead of pure sequential stages; (b) or lower-LR / fewer-epoch late stages; (c) v2 refinements (searched-opponent r̂₂ target, deployment-side r̂₂ re-rank). For deployment now: `h07/r0` (best all-round) or `h10/r1` (adds opening, weaker h7–10).

## EXP-018 — `exp_018_consolidate` (CONSOLIDATION joint-train on the union of all per-horizon buffers — anti-forgetting full-end model) — DONE: BEST full-end model, champion `last.pt` (0.557)

- **RESULT — NOISY 1→10 vs PRIOR (per-horizon hammer split + equal-weight pairs, pre-placed h10), `last.pt`:** pairs h01+h02 0.533, h03+h04 0.561, h05+h06 0.548, **h07+h08 0.577, h09+h10 0.565** → **MEAN 0.557** (all dScore +0.12…+0.30). **Beats both baselines: h07/r0 0.530, h10/r1 0.515.** The late game h10/r1 wrecked (h07+h08 0.461) is now the STRONGEST (0.577); h09+h10 (real pre-placed opening) 0.565; **every pair ≥ 0.53 — uniformly strong across the whole end.** `best.pt` (val-nll-selected, more human-like) = 0.544 < `last.pt` 0.557 → the most-trained checkpoint is the deploy model (val_policy_nll is the wrong selector for in-game strength). Training stable (no nan, 20 epochs).
- **Conclusion:** forgetting was a *policy* problem; one joint pass over the union of the 20 saved buffers (no re-collection, ~75 min train + fresh optimizer from h07/r0) removes it AND improves on the per-stage champion. **New deploy champion = `checkpoints/csas_world/exp_018_consolidate/last.pt` (0.557).**

- **Idea:** the forgetting is a *policy* problem (the value head already trains on the horizon-agnostic real value buffer; only the per-horizon policy distillation gets overwritten). We have **all 20 saved MCTS buffers** (h01–h10 × r0/r1 = 1600 records, incl pre-placed h10) at `hXX/rY/mcts/`. So train ONE policy on the **union** (sample uniformly across all horizons every batch) → no horizon is overwritten. Efficient: **reuses the already-collected search targets, no re-collection** — one DDP training run.
- **Setup:** `scripts/_run_consolidate.sh` → `run_consolidate.py` → `trainer.launch(mcts_shard_dir=<union>, init_ckpt=h07/r0/model.pt)`. Union dir = flat symlinks to all 20 `mcts` shards (`load_shards` rglobs `*.npz`). Warm-start from **`h07/r0`** (best all-round, 0.530) + **fresh optimizer** (the ckpt has no optimizer state → fresh Adam; epoch loop runs fresh `range(epochs)`). Recipe = EXP-017 (`policy_distill` on union mcts, value head on real buffer via `value_from_mcts=false`, reward head trained too). **Augment ON + VERIFIED correct**: `augment_batch` flips state + `a_raw` + `dist_actions_raw` (negates angle/spin/y0) + cond consistently, leaves value invariant, excludes dead (POS_MAX) stones, and handles pre-placed (flipped pp_left = pp_right). 4 GPUs (DDP), 20 epochs.
- **Goal:** a single model that matches the per-horizon bests across ALL h1–10 (no forgetting) → beat both h07/r0 (0.530) and h10/r1 (0.515) on the hammer-neutral pair metric, including a real h10 opening.
- **Verify:** NOISY 1→10 eval (`_eval_parallel.py`, per-horizon hammer-split + pairs, pre-placed h10) vs prior; compare MEAN-of-pairs to 0.530 / 0.515.

## EXP-019 — `exp_019_consolidate` (EXP-018 joint consolidation + MODE-BALANCED h10 buffer) — DONE: no measurable gain; EXP-018 stays champion

- **RESULT:** overall 1→10 NOISY MEAN-of-pairs **0.542** (pairs h01+h02 0.559, h03+h04 0.559, h05+h06 0.523, h07+h08 0.536, h09+h10 0.532) vs EXP-018's **0.557** — **within ~1 SE (indistinguishable)**. Balancing the h10 buffer did not move the overall, as expected (pp is a small slice: ~⅓ of h10, 1/10 of horizons). **EXP-018 `last.pt` (0.557) remains the deploy champion.**
- **pp-specific test INCONCLUSIVE (methodological error):** ran the mode-split DETERMINISTICALLY for speed, but at h10 deterministic play is **saturated** (hammer wins ~100%, no-hammer ~0% → pooled ~0.5 for every mode), so it cannot reveal a pp skill difference — exp018/exp019 came out byte-identical at pp_left (0.489)/pp_right (0.500) purely from that saturation, NOT "no change." A NOISY mode-split (slow on CPU; needs proper GPU sharding) is required to actually measure pp.
- **Honest takeaway:** the pp UNDER-COLLECTION was real and is fixed in code (`balance=True`, kept for future runs), but its *effect* on play is unconfirmed — the original "evidence" (2 blowout viz games) was too weak to establish a deficit, and neither the deterministic mode-split nor the (pp-light) overall eval can confirm one. Since pp is rare (Power Play = once/game) and the overall is unchanged, deployment impact is negligible either way. Pursue a noisy mode-split only if power-play robustness specifically matters.

- **Why:** the h10 pre-placed states are CANONICAL (`board_norm` → only **6 distinct states** = 3 modes × 2 guard-slots), so the data's 77/11/12 mode skew is NOT a data limit. EXP-018's h10 collection sampled the data rows *proportionally* → the 6 states got `[70,63,9,8,7,3]` (2 standard ~133, the 4 pp ~27 combined). That under-collection of pp_left/right was a decision (reused the data-driven loader), not a constraint — and the 2 viz blowouts hinted at weak pp play.
- **Fix (free):** `_preplaced_rows(balance=True)` (now default in `build_preplaced_roots`) samples EQUALLY across the 6 canonical (mode, guard-slot) groups — verified `[29,27,27,26,26,25]` (balanced) vs `[70,63,9,8,7,3]`. Same 80-root budget.
- **EXP-019 = EXP-018's anti-forgetting JOINT consolidation, NOT the sequential curriculum:** (1) re-collect ONLY the h10 buffer balanced (CPU, h09 policy + h08/r1 value-world, reward-leaf — same target quality, balanced modes); (2) union = exp_017 h1–9 buffers UNCHANGED + balanced h10; (3) joint-consolidate (warm-start h07/r0, fresh optimizer, 20 epochs, augment on). So both fixes stack: joint train (no forgetting) + balanced h10 (no pp under-collection). `scripts/_run_exp019.sh`.
- **Verify:** (a) **per-mode h10 winrate before(exp018)/after(exp019)** — deterministic, parallelized one (model,mode) job/GPU (`_eval_h10_by_mode --only-mode`) — did pp_left/right close the gap to standard? (the standalone mode eval was too slow noisy on CPU, hence deterministic + sharded). (b) full 1→10 NOISY MEAN-of-pairs vs prior, compare to EXP-018's 0.557 (no regression). Watcher `_exp019_watch_eval.sh`.

## EVAL-PROPER-3WAY (2026-06-24) — fixed-protocol head-to-head vs prior for the paper

- **Setup:** NOISY alternating play, both throwing orders, free-guard-zone rule. Each decision uses each player's own 1-ply robust selector ($M{=}48$ candidates, mean decision value over $K{=}8$ noisy executions); the realized throw is a noisy sample of the intended action. Per horizon we use the full val pool (no random subsampling: $N{=}400$ cap, but max pool $=387$ at h$=$10, so the cap never binds). Per-horizon $n_{\text{per\_order}}$: $h{=}1$–$9$ = 95–133 distinct recorded human boards, $h{=}10$ = 387 frequency-weighted draws from canonical pre-placed openings (effectively 9 distinct $(\text{mode},\,\text{guard\_slot},\,\text{thrower\_block})$ canonical states). Total games per horizon = $2n$. 4-GPU within-horizon sharding via `_eval_parallel.py`; results aggregated by `scripts/_aggregate_proper_eval.py` and stored at `eval_out/proper/summary_3way.json`.
- **Models compared (all vs human prior):**
  - **EXP-019 consolidated** (`exp_019_consolidate/last.pt`) — joint train on union of per-horizon buffers + mode-balanced canonical h10 (the deploy candidate).
  - **EXP-017 per-stage champion** (`exp_017_deploy_robust/h07/r0/model.pt`) — best single-stage iterate, has not seen any canonical h10 in training.
  - **EXP-017 sequential h10/r1** (`exp_017_deploy_robust/h10/r1/model.pt`) — naive 1→10 sequential curriculum (the forgetting baseline).
- **Per-horizon with-hammer winrate / score differential (EXP-019):**
  | h | n | wr w/ h | dScore w/ h | wr w/o h | dScore w/o h |
  |---|---|---|---|---|---|
  | 1 | 125 | 0.704 ± 0.041 | +0.87 | 0.328 ± 0.042 | −0.76 |
  | 2 | 133 | 0.722 ± 0.039 | +1.00 | 0.331 ± 0.041 | −0.74 |
  | 3 | 118 | 0.814 ± 0.036 | +1.26 | 0.297 ± 0.042 | −0.93 |
  | 4 | 109 | 0.706 ± 0.044 | +1.05 | 0.339 ± 0.045 | −0.69 |
  | 5 | 118 | 0.839 ± 0.034 | +1.55 | 0.288 ± 0.042 | −0.87 |
  | 6 | 113 | 0.832 ± 0.035 | +1.62 | 0.265 ± 0.042 | −1.15 |
  | 7 |  99 | 0.828 ± 0.038 | +1.38 | 0.222 ± 0.042 | −1.37 |
  | 8 |  98 | 0.878 ± 0.033 | +1.85 | 0.245 ± 0.043 | −1.40 |
  | 9 |  95 | 0.884 ± 0.033 | +1.91 | 0.263 ± 0.045 | −0.95 |
  | 10 | 387 | 0.848 ± 0.018 | +1.89 | 0.269 ± 0.023 | −1.32 |
- **Hammer-neutral pair averages (A-as-to-move) — overall mean of pairs across all 3 models:**
  | pair | EXP-019 | per-stage h07/r0 | sequential h10/r1 |
  |---|---|---|---|
  | h1+h2 | 0.517 ± 0.029 (+0.06) | 0.525 ± 0.029 (+0.11) | 0.513 ± 0.029 (+0.07) |
  | h3+h4 | **0.577 ± 0.029 (+0.29)** | 0.577 ± 0.030 (+0.36) | 0.561 ± 0.029 (+0.10) |
  | h5+h6 | **0.552 ± 0.027 (+0.20)** | 0.540 ± 0.029 (+0.13) | 0.515 ± 0.029 (−0.08) |
  | h7+h8 | **0.537 ± 0.029 (−0.01)** | 0.521 ± 0.027 (+0.05) | 0.527 ± 0.030 (−0.12) |
  | h9+h10 | **0.576 ± 0.020 (+0.29)** | 0.536 ± 0.024 (+0.01) | 0.491 ± 0.025 (−0.19) |
  | **MEAN-of-pairs** | **0.552** | 0.540 | 0.521 |
  | **MEAN dScore** | **+0.167** | +0.133 | **−0.042** |
- **Headline read:**
  - **EXP-019 wins on overall skill (0.552 > 0.540 > 0.521)** and on score differential (+0.17 > +0.13 > −0.04 points/end). It's positive at all five pair averages (≥0.52).
  - **Per-stage champion is a strong baseline** at mid-end horizons (h3+h4 0.577 tied), but does NOT generalize to opening: h9+h10 = 0.536 (vs EXP-019's 0.576).
  - **Sequential curriculum has clearly forgotten** the late game: pair h9+h10 = **0.491 below 0.50** (i.e.\ *worse* than the prior); MEAN dScore is *negative* (−0.04 points/end). This is the forgetting signature consolidation was designed to fix.
  - **Per-end score margin is large** (consolidated: +0.17 pts/end hammer-neutral); over 8 ends in a real game this compounds to >+1 expected total — a decisively winning team, even though the per-end winrate looks "only" 0.55.

## EXP-020 — `exp_020_consolidate_valuemcts` (EXP-019 + value head ALSO supervised by MCTS records) — RUNNING

- **One-line diff vs EXP-019** (`configs/exp_020_consolidate_valuemcts.yaml`): `loss.value_from_mcts: false → true`. Every other knob (union buffer `exp019_union_mcts`, warm-start `exp_017_deploy_robust/h07/r0/model.pt`, 20 epochs, augment, all loss weights including value=1.5 and value_nll=0.2, all replay mix, all search/horizon settings) is identical.
- **What the flag changes:** in `losses.py:81-86`, when `value_from_mcts=true` the `vmask *= is_value_src` mask is removed, so the value-head MSE+NLL loss also fires on the MCTS-source records' `value_target` (the realized terminal margin from the policy rollout, sign-flipped per perspective, broadcast across the K+1 unroll steps). In EXP-019 those targets were stored but masked out; here they become live supervision. The MCTS records contribute K+1 = 6 value-target rows each.
- **Why this is interesting:** in EXP-019 the value head sees ONLY recorded human ends (SOURCE_VALUE buffer; horizon-agnostic real-game ValueDiff). It does not see any "value under our policy" supervision. EXP-020 adds the policy's on-policy MC returns (terminal margins of consolidated-policy rollouts from real / pre-placed root states). These are unbiased MC returns for the policy we deploy, so they should align the value head with the policy's actual play distribution. Risk: distribution shift away from real human ends (could hurt the auxiliary aux-MSE test metric); benefit: tighter coupling between value estimate and the policy that's actually played.
- **Expected effect on game strength:** ambiguous a priori. If the value head was previously a slight bottleneck (mis-calibrated for self-play), adding policy-aligned supervision should help selection at decision time. If it was already well-calibrated by the real-data buffer, this could regress slightly via distribution drift. The eval will tell.
- **Eval protocol:** identical to EXP-019 (`_eval_parallel.py`, NOISY alternating play, both throwing orders, full val pool per horizon, $N{=}400$ cap that never binds, $h{=}1..10$, 4-GPU within-horizon shards, results at `eval_out/proper/exp020_valuemcts_vs_prior/`). Will compare MEAN-of-pairs + per-horizon win/dScore vs EXP-019 0.552 / +0.17, EXP-017 per-stage 0.540 / +0.13, sequential 0.521 / -0.04.
- **Status:** training launched (~75 min); watcher `_exp020_watch_eval.sh` will auto-run the eval on completion.
- **RESULT (2026-06-25) — NEGATIVE: value_from_mcts=true HURTS.** Training tail: `val_policy_nll = 19.62`, `val_value_mse = 3.45`, `val_value_nll = ~0.40` (vs EXP-019's 19.97 / 2.23 / 0.80) — the MCTS-source value targets dragged the value head ~50% farther from the real ValueDiff distribution; `value_mse` on the train side stayed high (~1.06) throughout, never converging to the EXP-019 level (~0.12). Game strength regressed across the board:
  | model (vs prior) | h1+h2 | h3+h4 | h5+h6 | h7+h8 | h9+h10 | MEAN | MEAN dScore |
  |---|---|---|---|---|---|---|---|
  | EXP-019 (false) | 0.517 | 0.577 | 0.552 | 0.537 | 0.576 | **0.552** | **+0.17** |
  | EXP-020 (true) | 0.486 | 0.572 | 0.497 | 0.496 | 0.521 | 0.515 | -0.02 |
  Every pair regressed; h01+h02 dropped below parity (0.486). **Interpretation:** the MCTS-record value targets are realized terminal margins from the search-distilled policy's rollouts, which are unbiased MC returns for that policy's play distribution but distributionally distinct from the real-data value buffer (recorded human ends). Mixing them at the existing 60/30 mcts/real-value ratio overwhelms the value head with policy-self-play distribution and corrupts its calibration on real states. **Conclusion: `value_from_mcts=false` (EXP-019) remains the right choice;** the value head should be trained only on the real value buffer. EXP-020 ckpt kept for the record but not deployed.

---

## EXP-021 — `exp_021_valuemcts_earlystop` (EXP-020 + held-out MCTS val + early-stop ckpt) — DONE, best.pt is competitive with EXP-019

- **Setup:** same recipe as EXP-020 (`value_from_mcts=true`, all else equal to EXP-019), plus a held-out MCTS val partition. Per stage we hold out shard `_s3of4` (~25%) of the union buffer; train uses shards 0/1/2 = 60 shards = 1200 records, val = 20 shards = 400 records. `trainer.evaluate()` extended (new function `evaluate_mcts_losses()`) to iterate the val partition each epoch and call `compute_losses` per batch, reporting per-loss val metrics `val_{policy_distill,value_mse,value_nll,consistency,step_reward,total}_mcts` alongside the external `val_policy_nll`/`val_value_mse` benchmarks. `cfg.train.checkpoint_metric=val_value_mse_mcts` so `best.pt` is the lowest-val-MSE-on-MCTS-distribution epoch (effective early stop via ckpt selection).
- **What the new val signal showed:** clear overfitting on the model's own training distribution, not just distribution drift away from real ends. `val_value_mse_mcts` trajectory across 20 epochs:
  `2.284 (e104, warm-start) → 2.071 → 1.750 → 1.547 (e107, min) → 1.571 → 1.584 → 1.633 → ... → 1.673 (e123)`.
  Train value MSE meanwhile went down monotonically 1.66→1.06. So the value head IS learning the MCTS distribution for the first ~3 epochs then starts to overfit it (val rises). The external `val_value_mse` (on real data) rose monotonically the whole time (2.96→3.52), which is the *additional* distribution-drift effect we saw in EXP-020.
- **Game strength** (NOISY 1→10 vs prior, $N{=}400$ per horizon, full pool, 4-GPU sharded):
  | model | h1+h2 | h3+h4 | h5+h6 | h7+h8 | h9+h10 | MEAN | MEAN $\Delta$score |
  |---|---|---|---|---|---|---|---|
  | EXP-019 (`value_from_mcts=false`)         | 0.517 | 0.577 | 0.552 | 0.537 | 0.576 | **0.552** | +0.17 |
  | EXP-020 (`value_from_mcts=true`, last.pt) | 0.486 | 0.572 | 0.497 | 0.496 | 0.521 | 0.515 | -0.02 |
  | EXP-021 (`value_from_mcts=true`, last.pt) | 0.526 | 0.555 | 0.540 | 0.512 | 0.529 | 0.532 | +0.07 |
  | **EXP-021 best.pt (epoch 107, early-stop)** | 0.501 | 0.561 | **0.605** | 0.547 | **0.597** | **0.562** | **+0.21** |
- **Two findings:**
  1. **Held-out val mechanism works as designed:** without it (deploying `last.pt` at epoch 123) EXP-021 lands at 0.532, still below EXP-019. With it, we recover a checkpoint at epoch 107 reaching 0.562. The val-MSE-on-MCTS signal correctly identified the overfitting onset.
  2. **best.pt slightly beats EXP-019 on the MEAN-of-pairs** (0.562 vs 0.552, +0.010 — within ~1 SE on the aggregate). Per-pair the gains concentrate at mid/late game (h5+h6: +0.053 ≈ 1.9 SE, h9+h10: +0.021), with a small regression at h1+h2 (-0.016). MEAN dScore is the clearer signal: **+0.21 pts/end** vs EXP-019's +0.17.
- **Deploy decision:** `EXP-021 best.pt` is a marginally-better candidate than EXP-019 last.pt. The mean improvement is within noise, but the late-game pair gains look real, and MEAN dScore is meaningfully higher (+0.21 vs +0.17). Both are reasonable deploy choices; EXP-019 is the more conservative pick (simpler recipe, no early-stop dependency), EXP-021 best.pt is the marginally-stronger pick if the late-game gains are real.
- **Methodology takeaway for the paper:** when training with mixed-distribution value supervision, hold out a chunk of the actual training distribution as val and select on the per-loss val signal there. The standard "val on the external human-data benchmark" misses the overfitting that matters for game strength.

---

## EXP-022 — `exp_022_exp019_earlystop` (EXP-019 recipe + held-out MCTS val + early stop) — DONE, NO overfit + major eval-variance finding

- **Setup:** identical to EXP-019 (`value_from_mcts=false`, all else equal) plus the held-out MCTS val partition and `checkpoint_metric=val_total_mcts`. Tests whether `policy_distill + consistency + step_reward` were overfitting in EXP-019, where we had no per-loss MCTS val signal.
- **Per-loss val curves (val_total_mcts trajectory):** monotonically decreasing through all 20 epochs, no overfitting onset. Components: `val_policy_distill_mcts` 6.179 → 5.885 (monotonic ↓), `val_consistency_mcts` -0.32 → -0.39 (monotonic ↓ = better), `val_step_reward_mcts ≈ 0` throughout. **So EXP-019 was NOT overfitting on any of its MCTS-trained heads.** `best.pt` ended up identical to `last.pt` (both at epoch 123, 0 total absolute weight diff across all 241 layers; md5 only differs in save-time metadata).
- **MAJOR methodological finding — eval variance:** since `best.pt = last.pt` (byte-equivalent weights), the eval-output gap between them is pure variance from re-running the same model. Two evals of the same model:
  | pair | run1 (as last.pt) | run2 (as best.pt) | gap |
  |---|---|---|---|
  | h1+h2 | 0.498 | 0.556 | +0.058 |
  | h3+h4 | 0.540 | 0.607 | +0.067 |
  | h5+h6 | 0.507 | 0.535 | +0.028 |
  | h7+h8 | 0.526 | 0.547 | +0.021 |
  | h9+h10 | 0.588 | 0.575 | −0.013 |
  | MEAN | **0.532** | **0.564** | **+0.032** |
  **Single-eval noise on MEAN-of-pairs is ±0.03.** Source: per-shard `sample_actions_z` (mixture sampling for the 1-ply robust selector's candidate pool) uses unseeded torch RNG, so the candidate sets — and thus the noisy decisions — differ across runs. Execution noise (`env_nz`) IS seeded by horizon, but candidate sampling is not.
- **Implications for previous claims:**
  - EXP-021 best.pt 0.562 vs EXP-019 0.552 (+0.010 gap) — **sits inside ±0.03 noise**, not robustly an improvement.
  - EXP-020 last 0.515 vs EXP-019 0.552 (−0.037 gap) — just outside one-σ but not robustly so.
  - The pair-level differences we've been treating as "≈2 SE signal" should be interpreted with the larger eval-variance envelope in mind.
- **Fix for confident future claims:** add `torch.manual_seed(args.seed + shard_id)` at the top of `_eval_highN.py` (and the launcher pass a seed). Then run **3–5 seeds per model** and report mean ± SD across seeds. Cheap (each eval is ~1 h on 4 GPUs; 5 seeds × 1 model is 5 GPU-hours; we'd typically do this only for the headline deploy candidates). For everything else, treat single-run mean-of-pairs as ±0.03.
- **Bottom line for deploy:** EXP-019 was already at a good stopping point (no overfit on its trained heads). The held-out val infrastructure is still valuable for any future recipe that mixes training distributions (it correctly caught EXP-021's overfit). For the paper's deploy champion, EXP-019 is the safer call; EXP-021 best.pt is statistically indistinguishable from it given current eval variance.

---

## EXP-023 — `exp_023_exp019_longtrain` (EXP-022 recipe at 40 epochs) — DONE, no measurable game-strength gain past 20 epochs

- **Setup:** same as EXP-022 (EXP-019 recipe + held-out MCTS val + early-stop by val_total_mcts) but trained for 40 epochs from h07/r0 instead of 20. Tests whether the val-loss headroom at epoch 123 (still decreasing logarithmically) translates to deployable gains.
- **Val curve over 40 epochs:** logarithmic slowdown. First 20 epochs: 5.860 → 5.491 (−0.369, −0.019/epoch). Next 20 epochs: 5.491 → 5.446 (−0.045, −0.002/epoch). Minimum at epoch 138 (5.436); small wiggles thereafter. `policy_distill_mcts` 6.179 → 5.862 (still decreasing), `consistency_mcts` -0.320 → -0.416 (still becoming more negative).
- **Game strength:** EXP-023 last.pt = MEAN-of-pairs **0.540** (MEAN dScore +0.126). EXP-023 best.pt (epoch 138) = MEAN-of-pairs **0.541** (MEAN dScore +0.064). Both **within ±0.03 eval-variance** of EXP-019 (0.552) and EXP-022 (0.532-0.564, same model evaluated twice).
- **Conclusion:** the additional val-loss improvement past 20 epochs **does not translate to deployable game strength**. The model is sharpening its fit to a *fixed* search-target dataset (1200 records, ~700 repetitions per record over 40 epochs); after epoch 123 we're just compressing to a frozen ceiling. The data scale is the bottleneck, not training duration. To improve beyond 0.55-ish mean-of-pairs we'd need either (a) more search budget at collection time (better targets), or (b) more roots (broader coverage), not more epochs on the existing union. **EXP-019's 20-epoch budget was already at the practical ceiling.**
- **Methodology note:** could have resumed from `exp_022_exp019_earlystop/last.pt` + 20 more epochs instead of restarting from h07/r0 + 40 epochs (the trainer's optimizer-state-save/restore makes resume identical to continuous training, modulo dataloader seeding). ~75 min of redundant compute that we'd save next time.

---

## EXP-024 — `exp_024_finetune_valuemcts` (fine-tune EXP-022 best with value_from_mcts=true) — DONE, lands within eval-noise band of EXP-019

- **Setup:** warm-start from `exp_022_exp019_earlystop/best.pt` (= EXP-019-equivalent, fully consolidated with value head fit only to real data). Fine-tune 20 epochs with `value_from_mcts=true`, held-out MCTS val + early-stop by `val_value_mse_mcts`. Tests whether *adding* MCTS-value supervision to an already-converged real-data value head is better than EXP-021's "redo consolidation from h07/r0 with the new loss".
- **Val signal worked cleanly:** `val_value_mse_mcts` started high (2.27 — value head had never seen MCTS-value targets), dropped to **min 1.48 at epoch 130** (6 epochs in), then rose to 1.71 by epoch 143. Classic U-shape; best.pt selected at epoch 130. Meanwhile, `val_value_mse` on real data rose monotonically 2.22 → 3.60 (real-data calibration gets dragged ~60% by the new MCTS supervision).
- **Game strength:** EXP-024 last.pt MEAN-of-pairs = **0.547** (+0.08 dScore), EXP-024 best.pt MEAN-of-pairs = **0.534** (+0.08 dScore). Both within ±0.03 eval-variance of EXP-019's 0.552. (Curiously best.pt evaluated *lower* than last.pt here; that's within eval noise but it shows the val-MSE-MCTS signal is not perfectly aligned with game strength.)
- **Consolidation-runs summary (all warm-started from h07/r0 unless noted):**
  | run | value_from_mcts | epochs | early-stop | MEAN | dScore |
  |---|---|---|---|---|---|
  | EXP-019 | false | 20 | n/a | 0.552 | +0.17 |
  | EXP-020 last | true | 20 | no | 0.515 | -0.02 |
  | EXP-021 best | true | 20 | yes (e107) | 0.562 | +0.21 |
  | EXP-021 last | true | 20 | no | 0.532 | +0.07 |
  | EXP-022 | false | 20 (= EXP-019) | yes (no overfit) | 0.532-0.564 (single model, 2 evals) | ... |
  | EXP-023 last | false | 40 | no | 0.540 | +0.13 |
  | EXP-023 best | false | 40 | yes (e138) | 0.541 | +0.06 |
  | EXP-024 last (warm exp_022) | true | 20 fine-tune | no | 0.547 | +0.08 |
  | EXP-024 best (warm exp_022) | true | 20 fine-tune | yes (e130) | 0.534 | +0.08 |
- **Honest conclusion:** every consolidation variant lands in **0.515-0.564** = roughly ±0.025 around 0.54, matching the established eval-variance. The `value_from_mcts` toggle, early-stop ckpt selection, warm-start choice (h07/r0 vs exp_022 best), and longer training (20 → 40 epochs) are **all within eval noise** of each other. The held-out MCTS val mechanism is *methodologically validated* (correctly diagnoses overfit onset in value_from_mcts=true runs, correctly reports no overfit for value_from_mcts=false) but does NOT yield reliable game-strength improvements within current eval variance.
- **The real bottleneck is the dataset, not training-time hyperparameters.** ~1200 fixed MCTS records, ~700 repetitions each in 20 epochs — we're saturating what can be extracted from this collection. Next-best paths: (a) larger joint collection, (b) better search at collection time (multi-ply KR-UCT instead of 1-ply robust); (c) deterministic eval seeding + multi-seed reporting to shrink the eval-noise floor below the current ±0.03 so smaller training-recipe gains become resolvable.

---

## AZ-v4 / EXP-025 — `az_v4_iter1` (one-step AZ policy improvement: warm-start EXP-021 best, collect fresh data with EXP-021 policy, re-train) — DONE, converged at iter 1 with NO improvement

- **Setup (the apples-to-apples AZ policy-improvement step):**
  - **Collector:** `exp_021_valuemcts_earlystop/best.pt` (current best).
  - **Search recipe:** `configs/exp_017_deploy_robust.yaml` — `use_mcts_tree: false`, EZ 1-ply + `noise_samples=8` robust scorer. **Exactly the recipe used to collect the data that trained EXP-021's baseline** (so iter-1 vs iter-0 isolates the POLICY change only, no search-depth confound). Earlier attempts used `anchor_mcts_v3.yaml` (KR-UCT n_sims=120 + multi-ply tree) which has 10× the per-record compute and a different recipe from baseline — abandoned for clean comparison.
  - **Data per iter:** 1,600 records (= EXP-021's training set size, 75/25 train/val split: 1,200 train + 400 val). 4 shards × 40 records × 10 horizons. Collected at ~17 min/horizon, ~2.4h total. Train and val partitions are an in-iter shard split (shards 0-2 train, shard 3 val).
  - **Train:** `configs/exp_021_valuemcts_earlystop.yaml` (20 epochs, value_from_mcts=true, held-out val + early-stop by `val_value_mse_mcts`). Warm-start from `exp_021/best.pt`.
  - **Eval:** standard 10-horizon × N=400 × both throw orders × 4-shard GPU split via `_eval_parallel.py`. Same methodology used throughout the EXP-019..024 chain.
  - **Convergence rule:** declare converged if Δ(mean_wr) ≤ 0.03 AND Δ(mean_dScore) ≤ 0.10 vs the previous iter, i.e. the new iter fails to beat the previous by more than the established eval-noise floor (winrate ±0.03 from EXP-022's repeated-eval discovery, dScore band loosened to ±0.10 based on observed per-eval dScore spread).
  - **Driver:** `scripts/az_converge_v2.py` + `scripts/_az_v4_launch.sh` (single-instance PID lock added after the racing incident, see "Methodology lesson" below).

- **Clean results (after the racing incident was identified and fixed):**

  | | mean_wr | mean_dScore | per-pair wr |
  |---|---|---|---|
  | iter-0 (EXP-021 best.pt re-eval) | **0.547** | **+0.194** | [0.502, 0.576, 0.549, 0.552, 0.554] |
  | iter-1 (this experiment) | **0.515** | **+0.012** | [0.482, 0.565, 0.510, 0.466, 0.550] |
  | Δ vs iter-0 | **−0.032** | **−0.182** | down on 4 of 5 pairs |

  **Convergence rule triggered immediately at iter 1**: Δwr=−0.032 ≤ 0.03 ✓ AND ΔdS=−0.182 ≤ 0.10 ✓ — iter 1 failed to clear noise on winrate (and clearly regressed on dScore). The loop stopped after one iter.

- **Cost:** ~2.4h collect + ~30m train + ~50m eval ≈ **~4h on 4 GPUs** for one iter. (Compare: a heavier `anchor_mcts_v3.yaml`-style collection would have been ~80h per iter — abandoned).

- **Interpretation — AZ-style policy improvement does NOT detectably improve on EXP-021 at this data scale.** The bet was: same-recipe collection but with an improved policy will yield better policy-distillation targets and train a better model. The bet did not pay off. Three plausible reasons (not mutually exclusive):
  1. **Data scale is too small.** 1,600 records, repeated ~700× over 20 epochs, is saturating what can be extracted at this size; the marginal gain from a slightly better collector falls below the ±0.03 noise floor.
  2. **The "policy-improvement operator" isn't strong enough with 1-ply EZ.** A real multi-ply MCTS-tree would provide larger search-improved targets per record (closer to the policy improvement theorem's premise), but we already showed that going to `mcts_tree=true` blows compute up 10× — would need ~80h per iter just to test this.
  3. **The collector's exact behavior on val-pool roots may not match its h2h-vs-prior behavior**: EXP-021 was already strong, so the marginal policy-distillation gain on roots where EXP-021 already plays near-optimally is tiny.

- **Three independent single-eval realizations of EXP-021 best.pt** (the noise-floor data point this run produced as a side benefit):
  - **0.562** (original cached eval, used for the EXP-021/022/024 entries above)
  - **0.531** (post-contamination fresh eval — see "Methodology lesson")
  - **0.547** (this experiment's iter-0 clean re-eval)
  - **Mean ≈ 0.547, range 0.031, SD ≈ 0.013.** This is direct evidence that single-eval numbers in this regime have ~±0.015–0.030 noise. Numbers like "EXP-021 best 0.562" or "EXP-019 0.552" should be read as ~0.55 ± 0.03, not as point estimates that can be compared to each other below ~0.03.

- **Methodology lesson: a multi-orchestrator race caused a chain of bad numbers before the clean re-run.**
  - Earlier `pkill` against `_eval_parallel.py` killed only the parent process, not its child `_eval_highN.py` workers, which kept writing to their `--out` paths. After we'd `rm -rf`'d the iter-0 dir and replaced it with a symlink to the cached `eval_out/proper/exp021_best_vs_prior/` shards, the stray writes resolved through the symlink and **overwrote our cached EXP-021 best.pt shards** (the original 0.562 → 0.531 partial overwrite was the first symptom).
  - A subsequent kill+restart cycle additionally produced **three concurrent orchestrators** (each `az_converge_v2.py` parent assumed it was alone), all training and evaluating to the same `iter1/` and `iter1_vs_prior/` paths. The current best.pt got overwritten mid-training; the eval shards interleaved from two parallel `_eval_parallel.py` runs on different best.pt files; an `iter-1: 0.5317 / +0.130` entry got written to history.json that we cannot trust.
  - **Fix:** added a PID-lock to `scripts/_az_v4_launch.sh` (refuses to start if another launcher is alive, releases on EXIT). And changed `az_converge_v2.py` iter-0 logic to always do a fresh eval into its own dir (never a symlink to a shared external dir).
  - **Forensic artifacts preserved** at: `checkpoints/csas_world/az_v4/iter1_corrupted_race_*` (mixed two-training output), `eval_out/az_v4/iter0_vs_prior_contaminated_symlink` (polluted cached shards), `eval_out/az_v4/iter1_vs_prior_corrupted_race_*` (mixed two-eval output), `checkpoints/csas_world/az_v4/history.json.contaminated` (the wrong −0.031 wr entry).

- **Bottom line for deploy:** EXP-021 best.pt (or EXP-019, indistinguishable from it within noise) remains the deploy champion. **No same-data-scale AZ iter we've tested has cleared the ±0.03 noise floor**: az_iter1 (0.525 vs parent, 2026-06-09), az_v3_iter1 (0.458 vs parent, 2026-06-10), and now az_v4_iter1 (−0.032 vs baseline). The bottleneck is data scale, not training-time hyperparameters. To move past this, the leverage move is either (a) significantly more data per iter (≥5× current), (b) a stronger policy-improvement operator (multi-ply MCTS), or (c) multi-seed eval to shrink ±0.03 → ±0.01 so smaller gains become resolvable.

---

## AZ-v5 / EXP-026 — `az_v5_novaluemcts` (one-step AZ policy improvement WITHOUT value_from_mcts) — DONE, iter-1 IMPROVED on baseline (within noise band, still declared converged)

- **Hypothesis under test:** az_v4_iter1 regressed (Δwr=−0.032, ΔdS=−0.182 vs baseline). Two candidate causes — value-head drift from `value_from_mcts=true` (EXP-024 showed this flag drags `val_value_mse` on real data 2.22→3.60), or the degenerate self-distillation problem with the 1-ply EZ improvement operator. This experiment isolates the FIRST by swapping ONLY the train config; everything else identical to az_v4_iter1.
- **What changed vs az_v4 (this is the only delta):**
  - **Train config:** `configs/exp_022_exp019_earlystop.yaml` (value_from_mcts=false, early-stop by `val_total_mcts`) — i.e. the EXP-019/022 recipe with no MCTS-target value supervision.
- **What stays the same vs az_v4 (clean control):**
  - **Warm-start ckpt:** `exp_021_valuemcts_earlystop/best.pt` (same)
  - **Collector policy:** `exp_021_valuemcts_earlystop/best.pt` (same)
  - **Search recipe:** `configs/exp_017_deploy_robust.yaml` — `use_mcts_tree: false`, EZ 1-ply (same)
  - **Collected data:** **literally the same 1,600 records** — `artifacts/replay/mcts/az_v5_novaluemcts_iter1` is a symlink to `az_v4_iter1`. No re-collection (saves 2.4h + eliminates collect-time noise as a confound).
  - **Eval methodology:** `_eval_parallel.py` at N=400 × 10 horizons × both throw orders × 4-GPU sharding (same)
  - **Convergence rule:** Δwr ≤ 0.03 AND ΔdS ≤ 0.10 (same)
- **Driver:** `scripts/_az_v5_novaluemcts_launch.sh` → `scripts/az_converge_v2.py --train-config configs/exp_022_exp019_earlystop.yaml --work checkpoints/csas_world/az_v5_novaluemcts ...` (the loop driver now takes `--train-config` so multiple recipes can share infrastructure; paths derive from `Path(args.work).name` so az_v4 and az_v5 don't collide).
- **Possible outcomes and what each tells us:**
  1. **iter-1 lands close to baseline (e.g. wr in [0.52, 0.57], dS not collapsed)** → value-head drift was the issue. The improvement operator works; we just had a recipe-time bug.
  2. **iter-1 regresses about as much as az_v4 (Δwr ≈ −0.03, ΔdS ≈ −0.18)** → the operator IS the problem (degenerate self-distillation from 1-ply EZ on a single fixed policy collecting all horizons). Need a real multi-ply MCTS to make progress. Closes that question.
  3. **iter-1 modestly improves on baseline (Δwr > 0)** → value-head drift was MASKING a real (small) gain from the policy-improvement step. Big news; suggests iterating could compound.
- **Cost projection:** ~50min iter-0 fresh eval + ~30m iter-1 train + ~50m iter-1 eval ≈ **~2h 10m** to one decisive data point (no re-collect; data shared).

- **RESULTS (this experiment landed cleanly at 09:31 UTC, 2026-06-30):**

  | | mean wr | mean dScore | per-pair wr |
  |---|---|---|---|
  | iter-0 (EXP-021 best.pt, fresh re-eval into az_v5 dir) | 0.5330 | +0.148 | [0.485, 0.591, 0.544, 0.496, 0.549] |
  | iter-1 (this experiment) | **0.5506** | **+0.190** | [0.506, 0.570, 0.561, 0.526, 0.589] |
  | Δ vs iter-0 | **+0.018** | **+0.042** | up on 4 of 5 pairs (pair-1 basically flat) |

  Convergence rule (Δwr ≤ 0.03 AND ΔdS ≤ 0.10) fires, so the loop declared "converged at iter 1" — but this time from ABOVE (positive deltas) rather than below (az_v4's negative deltas).

- **Interpretation (which is the primary finding of the az_v4 → az_v5 comparison):**

  This is the definitive causal test we set up. Only ONE variable changed between az_v4 and az_v5:
  - az_v4 (value_from_mcts=**true**): Δwr=−0.032, ΔdS=−0.182
  - az_v5 (value_from_mcts=**false**): Δwr=+0.018, ΔdS=+0.042

  **The −0.182 dScore regression in az_v4 was caused primarily by value_from_mcts=true**, not by the 1-ply improvement operator being too weak. Flipping the flag alone recovers the whole ~+0.22 dScore gap. This matches the EXP-024 finding (fine-tuning with value_from_mcts=true drove val_value_mse on real data from 2.22 → 3.60) — value-head calibration on real data is fragile, and forcing the value head to also fit MCTS-record targets from a single collector pulls it off-calibration in a way that costs game strength.

- **Deploy status:** az_v5 iter-1 (`checkpoints/csas_world/az_v5_novaluemcts/iter1/best.pt`) is a marginal but real improvement over EXP-021 best.pt on both winrate (0.551 vs ~0.547 mean over 4 noise draws) and dScore (+0.190 vs ~+0.19 across draws). Well within noise, but it's the first same-scale AZ iter that hasn't ended below baseline on either metric.

---

## AZ-v6 / EXP-027 — `az_v6_2ply_unfrozen` (TWO-PLY collect + value_from_mcts=true train) — DONE, best dScore of any AZ iter so far (2-ply operator makes value_from_mcts=true safe)

- **Hypothesis under test:** az_v4 iter-1 regressed substantially on dScore (Δ=−0.182) with value_from_mcts=true. Two candidate causes: (1) value-head drift from `value_from_mcts=true`, or (2) the 1-ply EZ improvement operator is too weak (degenerate self-distillation from a fixed policy collecting all horizons). az_v5 (running) tests (1) by flipping to `value_from_mcts=false` at fixed 1-ply collect; az_v6 (this experiment) tests (2) by replacing the 1-ply EZ collect with a **two-ply KR-UCT tree** while keeping `value_from_mcts=true`.
- **Why 2-ply specifically:** the AlphaZero policy-improvement theorem requires the search operator to produce targets π' > π. With 1-ply EZ + a single fixed collector, the operator collapses to "π picks π's own best action" — degenerate self-distillation. A 2-ply tree ranks root candidates by *what happens after the opponent's (or one's own) response*, which is strictly more information than 1-ply value at the post-state. Costs ~2× 1-ply, far cheaper than full multi-ply (~10×). Important framing: **2-ply is a NEW training-time-only operator** — neither the deployed inference (which is 1-ply with value-head bootstrap per `_decision_values`) nor any prior canonical EXP recipe uses it. We are not "matching inference"; we are using a stronger lookahead at training time than at inference time, which is the standard AZ/MuZero asymmetry.
- **Implementation:**
  - **New knob `mcts_max_depth` in `Config.search`** (default 0 = horizon-bound, the existing behavior). When > 0, caps the tree recursion at that depth. `src/world/search/kr_uct_tree.py`: `_simulate` now takes `max_depth, depth` kwargs; a fresh child at the cap is evaluated via `rollout_value_fn` (no further descent into already-expanded grandchildren). `mcts_search` threads the new arg through.
  - **`src/world/search/collect.py`** — the existing `use_mcts_tree=true` branch already supported `value_leaf_bootstrap` (which configures `rollout_value_fn` to be a value-head call rather than an on-policy MC rollout). Now it also passes `cfg.mcts_max_depth` to `mcts_search`.
  - **`configs/exp_026_2ply_valuemcts.yaml`** (new) — collect-time config. Diff vs the baseline `configs/exp_017_deploy_robust.yaml`: `use_mcts_tree: true`, `mcts_sims: 60` (shallow tree fills fast), `mcts_max_depth: 2`, `value_leaf_bootstrap: true`, `search_root_only: false` (we want grandchildren now), `reward_leaf_select: false` (tree-q ranks candidates, not r̂₂).
  - **`scripts/az_converge_v2.py`** — added `--collect-config` CLI flag (default = baseline 1-ply); `collect_iter` now takes the collect-config path; the main loop's collect-phase log line now prints the cfg basename.
  - **`scripts/_az_v6_2ply_unfrozen_launch.sh`** (new) — clone of v5 launcher with `--collect-config exp_026_2ply_valuemcts.yaml --train-config exp_021_valuemcts_earlystop.yaml --work .../az_v6_2ply_unfrozen`. PID lock as in v4/v5.
  - **`scripts/_az_v6_chain_after_v5.sh`** (new) — sequencer: polls the az_v5 PID lock until it releases (process gone), waits 15s for GPU memory release, then launches the v6 launcher. Fire-and-forget.
- **What's kept the same vs az_v4 (so iter-1 deltas isolate the collect operator):**
  - Warm-start: `exp_021_valuemcts_earlystop/best.pt`
  - Train config: `configs/exp_021_valuemcts_earlystop.yaml` (value_from_mcts=true)
  - Eval methodology: `_eval_parallel.py`, N=400, 10 horizons, both orders, `--noisy`
  - Convergence rule: Δwr ≤ 0.03 AND ΔdS ≤ 0.10
  - Data scale: 1,600 records per iter (1,200 train + 400 val)
- **Possible outcomes:**
  1. **iter-1 lands close to or above baseline** (Δwr ≥ -0.03 AND ΔdS not collapsed): the 2-ply operator did meaningful work; the degenerate-self-distillation hypothesis was the dominant cause of az_v4's regression. Strong evidence to iterate further at this recipe.
  2. **iter-1 regresses similarly to az_v4** (both Δwr ≈ -0.03 AND ΔdS ≈ -0.18): neither the operator depth nor the train config moves the needle alone. Combined with az_v5's result, the bottleneck is most likely data scale, not operator/training-config.
  3. **iter-1 dScore recovers (Δds ≥ 0) but wr still flat/below**: the operator gave better value targets (less calibration drift) without unlocking a real policy gain. Suggests the value head was the more brittle piece, but policy improvement at this data scale is hard.
- **Cost projection:** ~50min iter-0 fresh eval + **~2-3h iter-1 collect** (2-ply with value-head leaves; comparable to 1-ply EZ since the 5-ply greedy rollout is gone — the tree itself only descends 2 plies) + ~30m iter-1 train + ~50m iter-1 eval ≈ **~4-5h** to one data point. Same order as az_v5.
- **Sequencing:** `nohup bash scripts/_az_v6_chain_after_v5.sh` is what fires the whole thing — it sleeps until az_v5's lock releases, then launches v6 cleanly.

- **RESULTS (this experiment landed cleanly at 18:22 UTC, 2026-06-30, ~9 hours after az_v5 finished — chain worked):**

  | | mean wr | mean dScore |
  |---|---|---|
  | iter-0 (EXP-021 best.pt, fresh re-eval into az_v6 dir) | 0.5413 | +0.148 |
  | iter-1 (this experiment) | **0.5430** | **+0.225** |
  | Δ vs iter-0 | +0.002 (within noise) | **+0.076** (real signal) |

  Convergence declared at iter 1 (both deltas within band). Per-iter collect cost was ~2.3-2.6h (h04-h10 each ~2000-2400s — the 2-ply tree + value-head leaves added only ~2× the 1-ply EZ cost, not 10× the multi-ply cost, as projected).

- **Interpretation — the 2-ply operator changed the value_from_mcts=true story:**

  Same warm-start (exp_021/best.pt), same train config (value_from_mcts=true), same warm-start policy for collection (exp_021/best.pt). The only change from az_v4 was the collect operator (1-ply EZ → 2-ply tree). Result: **the value-head drift disappeared** — instead of −0.182 dScore we see +0.076.

  This is meaningful causally: it means the 2-ply operator produces MCTS-value targets **clean enough that the value head can absorb them without losing real-data calibration**. Under 1-ply EZ, MCTS-value targets are degenerate (π self-distilling) and the value head learning them at training time drifts. Under 2-ply, the MCTS-value targets encode a real +1-ply lookahead beyond what π sees, so the value head learning them stays useful.

  Comparison across the three isolate-a-variable experiments:

  | experiment | collect | train VFM | Δwr | ΔdS | note |
  |---|---|---|---|---|---|
  | az_v4 | 1-ply EZ | true  | −0.032 | −0.182 | value drift dominates |
  | az_v5 | 1-ply EZ | false | +0.018 | +0.042 | VFM off, minor gain from just-data-change |
  | az_v6 | 2-ply    | true  | +0.002 | **+0.225** | 2-ply lets VFM=true actually help |

  **az_v6 has the best dScore of any AZ iter we've run** (+0.225 for the trained model, +0.076 over its own iter-0 baseline). Winrate delta is within noise but positive, and the model plays with a meaningfully bigger margin of victory when it does win. That last bit is what dScore captures.

- **Deploy candidates now (all within ~±0.03 wr noise; picking on dScore):**
  - **az_v6 iter-1** — wr 0.543, dScore +0.225 (best dScore)
  - **az_v5 iter-1** — wr 0.551, dScore +0.190 (slightly better wr, simpler recipe)
  - EXP-021 best.pt / EXP-019 — the historical baselines
  - Difference between top two is within noise. az_v6 has the strongest signal but is a more complex recipe.

- **Next-step candidates (in order of interest):**
  1. **Continue iterating az_v6** past iter-1. Convergence declared but with positive delta — one more iter would confirm whether the improvement compounds or plateaus. Only ~4-5h more.
  2. **Multi-seed eval** of az_v5 iter-1 and az_v6 iter-1 (say 3 seeds each) — shrinks noise ±0.03 → ±0.017, might make the winrate deltas resolvable and confirm which is truly stronger.
  3. **2-ply with `value_from_mcts=false`** (az_v7) — would tell us whether 2-ply's benefit is *only* through value-head training or also through better policy-distill targets. Cheap to run.

---

## AZ-v6b / EXP-028 — `az_v6b_iter2` (CONTINUATION of az_v6 for one more iter) — DONE, PLATEAU (no compounding)

- **Question this settles:** does the +0.076 dScore improvement az_v6 got over its own iter-0 baseline **compound** on iter-2, or is it a one-shot single-iter effect that plateaus?
- **Setup — everything identical to az_v6, only the warm-start changes:**
  - **Warm-start ckpt:** `checkpoints/csas_world/az_v6_2ply_unfrozen/iter1/best.pt` (az_v6's winning ckpt: wr 0.543, dScore +0.225 vs human prior; val_value_mse_mcts = 2.54 at ckpt-time).
  - Collect config: `configs/exp_026_2ply_valuemcts.yaml` (2-ply, `mcts_max_depth=2`, `value_leaf_bootstrap=true`, `mcts_sims=60`). Same as az_v6.
  - Train config: `configs/exp_021_valuemcts_earlystop.yaml` (value_from_mcts=true, early-stop by `val_value_mse_mcts`). Same as az_v6.
  - Eval: `_eval_parallel.py --N 400`, 10 horizons, both orders, `--noisy`. Same as everything else.
  - **`--max-iters 1`** — one iter, no loop; the compound-or-plateau question wants a single clean data point.
- **In this experiment's naming, "iter-0" is a fresh re-eval of az_v6's iter-1 model, and "iter-1" is what az_v6 would have called "iter-2".** The Δ from this experiment's iter-0 to its iter-1 IS the compound signal.
- **Possible outcomes:**
  1. **Δwr > 0 AND ΔdS > 0 (both positive again, roughly matching az_v6's own +0.002 wr / +0.076 dS):** AZ improvement is COMPOUNDING at 2-ply. First real evidence of that in this project. Strong argument to keep iterating (v6c, v6d).
  2. **Δwr ≈ 0 AND ΔdS ≈ 0 (both flat):** the improvement was a one-shot effect from the initial 2-ply + value-head-relearning combo; the recipe plateaus at this data scale. Focus shifts to more data / better operator.
  3. **Any regression (Δwr < 0 OR ΔdS < 0 by more than noise):** the value head or the policy drifted from a stronger starting point (az_v6/iter1); either the second iter's collected data has different statistics than the first, or the training pulled the model off a good spot. Interesting failure mode either way.
- **Driver:** `scripts/_az_v6b_iter2_launch.sh` (single-instance PID lock). Fired at 02:25 UTC 2026-07-01.
- **Cost projection:** ~8h wall-clock (~50min iter-0 eval + ~5.9h 2-ply collect + ~30min train + ~50min iter-1 eval), matching az_v6's timing.

- **RESULTS (landed at 11:14 UTC, 2026-07-01, ~8h 49m after launch):**

  | | mean_wr | mean_dScore |
  |---|---|---|
  | iter-0 (fresh re-eval of `az_v6_2ply_unfrozen/iter1/best.pt`) | 0.5412 | +0.180 |
  | iter-1 (this experiment — az_v6's would-be iter-2) | **0.5300** | **+0.101** |
  | Δ vs iter-0 | **−0.011** (within ±0.03 wr noise) | **−0.079** (within ±0.10 dS noise, but negative) |

  Convergence declared at iter-1. Δ values within the noise band, so formally "no improvement above noise", but both are slightly negative — so the AZ improvement did NOT compound; if anything it slightly retraced.

- **Interpretation — matches Outcome 2 (plateau) with a mild lean toward Outcome 3 (dScore retracts):**

  The +0.076 dScore boost that az_v6 got on its first iter was a **one-shot effect**, not the beginning of a compounding trajectory. The mechanism is now clear:
  - The 2-ply operator produces MCTS-value targets that are strictly more informative than 1-ply (real +1-ply lookahead).
  - The value head, initially calibrated only on the realized-value buffer (EXP-021's real-data), had "room to absorb" the 2-ply MCTS-value signal → az_v6's +0.076 dScore gain.
  - **Once absorbed, there's no new information for a second iter to add.** The value head has already integrated whatever the 2-ply signal offers at this data scale; re-collecting from the newer, slightly-better policy doesn't provide fresh calibration information; retraining on similar-distribution targets doesn't produce a further step.
  - The mild negative slide (Δwr = −0.011, ΔdS = −0.079) is consistent with normal iter-2 noise around a plateau — the process randomly walks slightly below the peak.

  Reading across the full AZ chain:

  | run | warm-start | collect | train VFM | Δwr | ΔdS | note |
  |---|---|---|---|---|---|---|
  | az_v4 | exp_021/best | 1-ply | true | −0.032 | −0.182 | value drift dominates |
  | az_v5 | exp_021/best | 1-ply | false | +0.018 | +0.042 | VFM off, minor gain |
  | az_v6 | exp_021/best | 2-ply | true | +0.002 | **+0.225 model / +0.076 Δ** | 2-ply makes VFM=true safe, one-shot boost |
  | **az_v6b** | **az_v6/iter1/best** | 2-ply | true | **−0.011** | **−0.079** | **NO compounding — plateau confirmed** |

- **Bottom-line finding: at this data scale (~1,600 records/iter), the AZ policy-improvement loop yields a one-shot benefit from moving to a stronger operator (2-ply) and stops there.** Iterating further at the same recipe + same data scale does not compound the gain. To break out, the leverage moves are:
  1. **More data per iter** (5-10× current). The gating factor is that the value head can absorb the current MCTS-value signal in one pass; more data would spread the fitting over a wider state distribution and might reveal further gains.
  2. **A stronger operator** (multi-ply full-tree with `mcts_max_depth: 4+`). More lookahead depth = strictly more information; also strictly more compute (~10× per record).
  3. **Different eval methodology (multi-seed)** to shrink the noise floor to ±0.01. Would let us resolve gains that currently disappear in ±0.03 noise, potentially picking up compounding signal az_v6b is missing.

- **Deploy conclusion:** `az_v6_2ply_unfrozen/iter1/best.pt` (wr 0.543, dScore +0.225) remains the strongest ckpt across all AZ experiments. Neither iterating further (az_v6b) nor using the simpler VFM=false recipe (az_v5, wr 0.551, dScore +0.190) beats it decisively. If the paper deploys one number, it's a wash between az_v6 (best dScore) and az_v5 (best wr, simpler recipe) — both are within noise of the EXP-021 baseline.

---

## AZ-v7 / EXP-029 — `az_v7_3ply_from_v6` (3-ply collect from az_v6/iter1, VFM=true train) — DONE, REGRESSED (operator depth is NOT the lever; plateau confirmed as data-scale-bound)

- **Question this settles:** az_v6b showed 2-ply plateaued when iterated from az_v6/iter1 (Δwr=−0.011, ΔdS=−0.079). Does a DEEPER operator (3-ply, one more ply of real lookahead) unlock the compounding that 2-ply couldn't?
- **Rationale:** the plateau in az_v6b suggests the value head absorbed the 2-ply MCTS-value signal in one pass, then had nothing new to learn on re-iter. If that's the case, giving it a *different, deeper* signal (3-ply lookahead) should provide fresh information the value head hasn't seen yet. That's the AZ progression premise applied to search depth rather than to iteration count.
- **Setup — only the collect depth changes vs az_v6b:**
  - **Warm-start ckpt:** `checkpoints/csas_world/az_v6_2ply_unfrozen/iter1/best.pt` (same as az_v6b, so the "does it compound" question is against the same baseline).
  - **Collect config:** `configs/exp_029_3ply_valuemcts.yaml` — clone of `exp_026` with **`mcts_max_depth: 3`** (was 2). Everything else identical: `use_mcts_tree=true`, `value_leaf_bootstrap=true`, `mcts_sims=60`, `search_root_only=false`, same noise/candidate/kernel settings.
  - **Train config:** `configs/exp_021_valuemcts_earlystop.yaml` (VFM=true, same as az_v6/az_v6b — so the value-head-drift is not confounded).
  - **`--max-iters 1`** — one clean data point.
  - Eval methodology and convergence rule identical to prior runs.
- **What "iter-0" and "iter-1" mean here:** iter-0 = fresh re-eval of `az_v6/iter1/best.pt` (same ckpt az_v6b evaluated as its baseline at wr 0.5412 / dScore +0.180). iter-1 = the new 3-ply-collected + VFM-trained ckpt. The Δ tests **operator depth as a lever**, given the compounding-by-iteration lever plateaued.
- **Possible outcomes:**
  1. **Δwr and ΔdS both positive (matching or exceeding az_v6's +0.076 dS first-iter gain):** deeper operator IS a real lever; further depth-scaling might keep the improvement going. Best outcome — argues for az_v8 = 4-ply.
  2. **Δwr flat, ΔdS positive:** value head gets more calibration signal from 3-ply than from 2-ply, but the policy has hit a ceiling at this data scale. Still useful — means deploying the deeper-operator model has a real margin-of-victory advantage.
  3. **Δwr and ΔdS both flat or negative (matches az_v6b):** operator depth doesn't help either. Data scale is the real bottleneck — more data per iter is the only remaining lever.
- **Cost projection:** each of the 60 sims does 3 simulator steps instead of 2, so per-record cost ≈ 1.5× 2-ply. Per horizon (h04-h10) probably ~40-55 min (2-ply was ~35-42 min). Total iter ~9-11h. Launched at $(will fill in from run.log when it fires).
- **Driver:** `scripts/_az_v7_3ply_from_v6_launch.sh` (single-instance PID lock).

- **RESULTS (landed 01:14 UTC, 2026-07-02, ~8h 52m after launch):**

  | | mean_wr | mean_dScore |
  |---|---|---|
  | iter-0 (fresh re-eval of `az_v6_2ply_unfrozen/iter1/best.pt`) | 0.5410 | +0.168 |
  | iter-1 (3-ply collect + VFM=true train) | **0.5068** | **+0.074** |
  | Δ vs iter-0 | **−0.034** | **−0.095** |

  Per-pair wr dropped on 4 of 5 pairs (worst: h05+h06 0.562→0.477). Train val curve: `val_value_mse_mcts` best 2.07 (early-stop caught it), but real-data `val_value_mse` ended at 3.13 — the familiar VFM=true calibration pull.

- **Direct comparison — the depth question, same warm-start (`az_v6/iter1/best.pt`), same train config:**

  | second-iter recipe | Δwr | ΔdS |
  |---|---|---|
  | az_v6b: 2-ply again | −0.011 | −0.079 |
  | az_v7: 3-ply | −0.034 | −0.095 |

  3-ply did **not** unlock compounding — it landed slightly below the 2-ply repeat (both within noise of each other, both clearly not-positive). Outcome 3 of the pre-registered outcomes: operator depth is not the lever.

- **Cost note:** 3-ply collect was ~same wall-clock as 2-ply (~38-43 min/horizon; total chain 8.9h vs az_v6b's 8.8h). The extra ply is cheap because leaves are value-head calls — but it bought nothing.

- **Interpretation — the az_v4→v7 arc is now complete and internally consistent:**
  1. az_v6's +0.076 dScore first-iter gain was a **one-shot absorption effect**: a value head calibrated only on realized-value data gets one useful dose of search-derived value signal, then saturates.
  2. Neither re-dosing (az_v6b, 2-ply again) nor a stronger dose (az_v7, 3-ply) produces a second step. If anything, repeated VFM=true training from an already-absorbed warm-start slowly degrades real-data value calibration (val_value_mse 3.13 here) — each re-iter is downside risk, not upside.
  3. **At ~1,600 records/iter, the AZ loop is data-scale-bound.** Operator depth, iteration count, and the VFM toggle are all exhausted as levers at this scale.

- **Deploy conclusion (unchanged): `az_v6_2ply_unfrozen/iter1/best.pt`** (wr 0.543, dScore +0.225) remains the strongest ckpt; `az_v5_novaluemcts/iter1/best.pt` (wr 0.551, dScore +0.190) the simpler-recipe runner-up. az_v7's iter-1 ckpt is NOT a deploy candidate.

- **Next-step decision (post az_v7):** the two remaining levers, ranked:
  1. **Multi-seed eval of the deploy shortlist** (az_v6/iter1, az_v5/iter1, EXP-021 best, EXP-019) — 3 seeds × ~1h each ≈ 3-5h total. Shrinks the ±0.03 noise floor to ~±0.017 and firmly ranks the shortlist for the paper's deploy claim. Cheap, directly useful, no new training risk.
  2. **5-10× data per iter** (~8k-16k records, ~1-2.5 days of 2-ply collect) — the only remaining hypothesis for making the AZ loop compound. Expensive, and the one-shot-absorption mechanism above predicts the benefit saturates in the value head rather than compounding; run only if the paper needs a "we scaled it" data point.

---

## MULTISEED-EVAL / EXP-030 — multi-draw eval of the deploy shortlist — DONE, FOUR-WAY PARITY (no AZ variant beats the consolidation baselines)

- **Goal:** every AZ-arc conclusion so far was hostage to the ±0.03 single-eval noise floor. Get ≥3 independent N=400 draws per shortlist candidate, shrink the SE to ~±0.006-0.023, and rank the shortlist for real.
- **Method:** each draw = one full `_eval_parallel.py` run (N=400, 10 horizons, both orders, NOISY). Draws are independent because candidate sampling (`sample_actions_z`) is unseeded. Reused existing on-disk draws (az_v6/iter1 already had 3: its own iter-1 eval + az_v6b's and az_v7's iter-0 re-evals; exp_021 had 3 on-disk + the recorded 0.5624 original whose shards were lost to the racing incident); ran 4 new draws (2× az_v5/iter1, 2× exp_019 last). Driver `scripts/_multiseed_eval.sh`, aggregator `scripts/_aggregate_multiseed.py`, results at `eval_out/multiseed/aggregate.json`.

- **RESULTS (2026-07-02):**

  | candidate | n draws | wr ± SE | dScore ± SE |
  |---|---|---|---|
  | exp_021 best (baseline champion) | 4 | **0.5458 ± 0.0062** | +0.176 ± 0.017 |
  | exp_019 last (conservative baseline) | 3 | 0.5456 ± 0.0038 | +0.164 ± 0.015 |
  | az_v6_iter1 (2-ply + VFM=true) | 3 | 0.5417 ± 0.0006 | **+0.191 ± 0.017** |
  | az_v5_iter1 (1-ply, VFM=false) | 3 | 0.5411 ± 0.0066 | +0.160 ± 0.023 |

  All four are within ~0.005 winrate of each other — smaller than every pairwise combined SE. No candidate separates from any other on either metric.

- **Key deflations:**
  1. **az_v6's celebrated +0.225 dScore was its luckiest draw.** Its 3-draw mean is +0.191 ± 0.017 — statistically indistinguishable from exp_021's +0.176 ± 0.017. The "2-ply one-shot boost" survives as at most a ~+0.015 dScore lean, well under 1 combined SE.
  2. **az_v6_iter1's winrate is remarkably STABLE across draws** (0.5430/0.5412/0.5410, SE 0.0006) — but stable at a mean slightly *below* the baselines, not above.
  3. **az_v5's first draw (0.5506) was also its luckiest**; its mean is 0.5411 ± 0.0066.
  4. The apparent Δ improvements in az_v5/az_v6 iter-1-vs-iter-0 comparisons were, in hindsight, largely their iter-0 baselines drawing LOW (0.533, 0.541) while their iter-1s drew HIGH — regression to the mean across the pair of draws.

- **Bottom line for the paper and for deploy:** the **entire AZ arc (az_v4 → az_v7) produced checkpoints at statistical parity with the consolidation baselines** (EXP-019/EXP-021). The honest claims are: (a) value_from_mcts=true without a strong enough operator actively HURTS (az_v4, confirmed −0.18 dScore, far outside noise); (b) the 2-ply operator neutralizes that harm (az_v6); (c) nothing in the arc produced a real game-strength gain over consolidation at this data scale. **Deploy champion: keep EXP-021 best.pt (or EXP-019 last.pt — identical within noise; EXP-019 is the simpler recipe).** az_v6's dScore lean (+0.015) is not a reason to switch.
- **Methodological deliverable:** multi-draw eval (3-4 draws) shrinks the effective noise floor from ±0.03 to ~±0.006-0.013 SE at ~50min per draw on 4 GPUs. Any future claimed improvement should clear ~2× combined SE (~±0.02) on the multi-draw mean before being believed. This protocol (and the racing-incident lesson about unseeded eval draws) belongs in the paper's evaluation section.

---

## AZ-v8 / EXP-031 — `az_v8_ratchet` (structurally-correct AZ loop: accumulating buffer + promotion gate + concentrated 2-ply) — DONE, NO PROMOTION IN 3 ITERS (ceiling confirmed; loop-structure hypothesis rejected)

- **Goal:** find a loop where more iterations make the policy stronger. The az_v4..v7 arc never tested this properly because the loop had three structural defects, each now fixed:
  1. **Data discard → accumulating replay buffer.** az_converge_v2 trained each iter on ONLY that iter's 1,600 records, discarding all previous buffers — a fresh small-data fit every time. This alone explains one-shot absorption + plateau: iteration never increased total training signal. az_v8 trains iter N on the union of iters 1..N (train and held-out-val partitions both accumulate): 1,600 → 3,200 → 4,800 records.
  2. **Backward drift → promotion gate (ratchet).** az_v6b/az_v7 slid backward because each iter unconditionally became the next collector/warm-start. az_v8 evaluates each iter with a 3-draw eval (SE ~0.006-0.013) and promotes to incumbent ONLY if mean wr beats the incumbent's by > 1× combined SE. Otherwise the incumbent stays collector + warm-start (the new data still accumulates). Monotone by construction.
  3. **Tree starvation → concentrated sims.** At az_v6's mcts_sims=60 / k_widen=2.0, the 2-ply tree gave ~15 root children × ~4 visits, ~1 visit per opponent-response sample — thin targets (kernel regression pools across neighboring actions, but per-action counts were low). az_v8 uses `configs/exp_031_2ply_sims120.yaml`: **mcts_sims=120, mcts_k_widen=1.5** → ~16 root children × ~7-8 visits, ~2 visits per grandchild. ~2× per-action evidence; collect ~2× wall-clock (~12h/iter).
- **Why not 5-ply:** no dose-response from depth in az_v6/v7 (2-ply ≈ 3-ply ≈ nothing); with 60-120 sims, depth ≥3 levels get ~1 visit each (pure variance — depth-d needs sims ~ branching^d, i.e. thousands for 5-ply = 3-12 days/iter); and with value-head leaves, deep search can't out-know its V — the compounding channel is V improving between iterations (via VFM=true on search-grounded returns), not depth within an iteration.
- **Recipe:** collect 2-ply/sims120 with incumbent policy → accumulate → train exp_021 recipe (VFM=true, early-stop on accumulated held-out val) warm-started FROM INCUMBENT → 3-draw eval → gate. 3 iterations.
- **Driver:** `scripts/az_ratchet.py` (new orchestrator; reuses collect/train/eval helpers from az_converge_v2) + `scripts/_az_v8_ratchet_launch.sh` (PID lock). Launched 10:02 UTC 2026-07-02.
- **Iter-0 baseline seeded** from the 3 existing on-disk draws of exp_021/best.pt (copied, not symlinked — the racing-incident lesson): **wr 0.5403 ± 0.0040, dScore +0.164 ± 0.015**. Gate for iter-1 promotion: 3-draw mean wr > ~0.548.
- **Cost:** ~15h/iter (12h collect + 35m train + 2.6h 3-draw eval); 3 iters ≈ 2 days (baseline was free via seeding).

- **RESULTS (landed 05:59 UTC, 2026-07-04, ~44h total):**

  | iter | accum records | 3-draw wr ± SE | 3-draw dScore ± SE | Δwr vs incumbent | gate (1×comb SE) | decision |
  |---|---|---|---|---|---|---|
  | 0 (exp_021 best, seeded) | — | 0.5403 ± 0.0040 | +0.164 ± 0.015 | — | — | incumbent |
  | 1 | 1,600 | 0.5411 ± 0.0039 | +0.159 ± 0.021 | **+0.0008** | 0.0056 | kept incumbent |
  | 2 | 3,200 | 0.5359 ± 0.0039 | +0.124 ± 0.033 | −0.0044 | 0.0056 | kept incumbent |
  | 3 | 4,800 | 0.5396 ± 0.0047 | +0.150 ± 0.032 | −0.0007 | 0.0062 | kept incumbent |

  Final incumbent after 3 iterations: **still `exp_021_valuemcts_earlystop/best.pt`** — the loop never found a checkpoint worth promoting. Collect cost matched projection (~11.6h/iter at sims=120).

- **Interpretation — this is the clean negative result the az_v4..v7 arc couldn't provide:**
  1. **All three structural defects were fixed** (data accumulation 1,600→4,800; ratchet gate preventing backward drift; 2× per-action search evidence) **and the policy still did not improve.** Every iteration landed within ±0.005 of the incumbent's 0.5403 — far inside the gate. The earlier plateau was NOT an artifact of loop structure.
  2. **The ratchet mechanism itself worked exactly as designed**: no backward step was ever taken (contrast az_v6b/az_v7, which drifted −0.01..−0.03 when promotion was unconditional). The gate never fired because there was genuinely nothing to promote.
  3. **The tripled buffer produced no trend**: iter-3 (4,800 records) ≈ iter-1 (1,600 records) ≈ baseline. If data scale at THIS collection recipe were the binding constraint, iter-3 should have ticked up; it didn't. More of the same data is not the answer either.
  4. **Remaining hypotheses for the ~0.54 ceiling** (in decreasing plausibility): (a) **model capacity** — 7.3M params may be saturated; (b) **root-pool diversity** — every iter collects from the same human-data root pool, so accumulating iterations adds noise-resamples of the same states, not new states (a true self-play loop would generate novel positions by playing full games from the opening); (c) **intrinsic headroom** — under Student-t execution noise, the achievable edge of a near-optimal policy over the strong human prior may simply be ~0.54 at these horizons.
  5. **The full AZ investigation (az_v4 → az_v8) is now a complete, internally-consistent story for the paper**: one-shot value-signal absorption (v6), no compounding from iteration (v6b), no dose-response from depth (v7), and no rescue from correct loop structure (v8). Combined with the multi-seed protocol (EXP-030), the negative result is solidly established, not noise-hostage.

- **Recommended next steps:** (a) fold the arc into the paper (negative-result + eval-methodology contribution; Tables should use multi-draw means ± SE); (b) if pursuing strength further, the untested lever with the strongest rationale is **full-game self-play collection** (novel states beyond the human root pool) and/or **larger model capacity** — both are new projects, not iterations of this loop.

---

## AZ-v9 / EXP-032 — `az_v9_selfplay` (FULL-GAME SELF-PLAY ratchet: search-at-every-ply, incumbent value at leaves, dScore-primary gate) — IN FLIGHT

- **Two corrections discovered while building this (errata for earlier entries):**
  1. **Record volumes in az_v4..v8 entries are understated 4×.** `--max-roots 160` applies PER SHARD (the root pool is sharded first, then truncated), so each AZ iter collected **6,400 records (4,800 train + 1,600 val)**, not 1,600 — and az_v8 accumulated to **14,400 train records** (12× the baseline's 1,200) by iter-3. Verified by counting the npz contents. This *strengthens* az_v8's conclusion: 12× more same-recipe data produced zero trend.
  2. **The search operator never actually improved across az_v6..v8 iterations.** The collect calls passed only `--value` (the frozen csas_v3 value model) and never `--value-world`, so tree leaves were always evaluated by the same frozen external value model. VFM=true training improved the model's value head each iter, but **search never saw it** — the improvement operator was literally constant. Another independent reason compounding was impossible.

- **What az_v9 changes (the first structurally-COMPLETE AZ loop in this project):**
  1. **Full-game self-play collection** (`src/world/search/selfplay.py`, new): play entire ends from the pre-placed openings with the incumbent policy; fresh 2-ply KR-UCT search at EVERY ply; executed action = one execution-noise realisation of the searched intent. One game = 10 records (one per horizon, auto-balanced), each anchored at a **policy-induced state**. The training distribution now evolves with the policy each iteration — the compounding channel the frozen human root pool could never provide. Targets per record: dist = soft-top-k of the searched root actions; value = the game's **realized final margin** (grounded MC return, per-state perspective — the AlphaZero value target); unroll = the real game continuation; reward = 2-step returns along the real trajectory. Smoke-verified: horizons 10→1 per game, perspective-alternating margins, correct unroll truncation (5,5,5,5,5,5,4,3,2,1), normalized dist weights.
  2. **Incumbent value head at tree leaves** (`--value-world <incumbent>`): value-head improvements now feed back into the search operator each iteration (fixes errata #2).
  3. **dScore-primary promotion gate** (per project metric convention): promote iff Δds > 1× combined SE AND Δwr > −1× combined SE (no-clear-winrate-loss guard). Winrate is supplementary.
- **Unchanged:** ratchet harness (accumulating buffer, incumbent-anchored warm-start/collect, 3-draw eval), collect search recipe (`exp_031`: 2-ply, sims=120, k_widen=1.5), train recipe (`exp_021`, VFM=true — now consuming grounded MC value targets), volume (160 games/shard × 4 × 10 rec/game = 6,400/iter), seeded iter-0 baseline (wr 0.5403 ± 0.0040, ds +0.164 ± 0.015).
- **Driver:** `scripts/_az_v9_selfplay_launch.sh` → `scripts/az_ratchet.py --selfplay-games 160 --gate-metric ds` (new flags). Launched 13:01 UTC 2026-07-04.
- **What would count as success:** any iter promoted on the dScore gate (needs Δds ≳ +0.03-0.04 on the 3-draw mean). Distinct failure modes are informative too: if self-play + closed value loop still plateaus, the ceiling is capacity/headroom, not data distribution.
- **Cost:** ~15h/iter (collect ~11-12h + train ~40m + 3-draw eval ~2.6h); 3 iters ≈ 2 days.

- **RESULTS — iters 1-3 (landed 07:54 UTC 2026-07-06); iters 4-6 running via `--resume` continuation:**

  | iter | accum train records | 3-draw wr ± SE | 3-draw dScore ± SE | gate (dScore-primary) | decision |
  |---|---|---|---|---|---|
  | 0 (exp_021 best, seeded) | — | 0.5403 ± 0.0040 | +0.164 ± 0.015 | — | incumbent |
  | 1 | 4,800 | 0.5484 ± 0.0008 | +0.159 ± 0.002 | Δds −0.005 < +0.015 | kept incumbent (Δwr +0.008 ≈ 2× SE, noted) |
  | **2** | **9,600** | **0.5742 ± 0.0054** | **+0.263 ± 0.044** | **Δds +0.099 > +0.046** | **PROMOTED** |
  | 3 | 14,400 | 0.5583 ± 0.0039 | +0.217 ± 0.020 | Δds −0.046 vs iter-2 | kept incumbent |

- **Interim interpretation:**
  1. **The self-play loop produced the largest, clearest improvement of the entire project at iter-2**: wr +0.034 (~6× combined SE) and dScore +0.099 (2.1× the gate) over the exp_021 baseline. Every one of iter-2's individual draws (wr 0.570/0.585/0.567) exceeds the best single draw ever recorded for any earlier model (0.562). This is far outside any noise story.
  2. **iter-1 → iter-2 is the first compounding step observed in the project** (one data dose: small wr nudge; two doses + twice-refreshed value head: decisive jump). The mechanistic fixes (policy-induced states, incumbent value head at tree leaves, accumulation) are validated as the missing ingredients — az_v4..v8's null results were caused by their absence, not by a capacity ceiling at ~0.54.
  3. **iter-3 dipped from iter-2's mark** (0.558/+0.217 — still far above the original baseline). Consistent with either (a) iter-2's numbers being a high draw-set around a true ~0.56-0.57 plateau, or (b) training-on-accumulation noise. The ratchet did its job: iter-2 remains incumbent and collector.
  4. **Deploy champion (as of iter-3): `az_v9_selfplay/iter2/best.pt`** — wr 0.574 ± 0.005, dScore +0.263 ± 0.044 vs the human prior; clearly ahead of the entire multiseed shortlist (all ~0.54/+0.16-0.19).
- **Continuation (iters 4-6) landed 03:53 UTC 2026-07-08:**

  | iter | accum train records | 3-draw wr ± SE | 3-draw dScore ± SE | decision |
  |---|---|---|---|---|
  | 4 | 19,200 | 0.5529 ± 0.0062 | +0.177 ± 0.015 | kept incumbent |
  | 5 | 24,000 | 0.5445 ± 0.0001 | +0.157 ± 0.011 | kept incumbent |
  | 6 | 28,800 | 0.5532 ± 0.0035 | +0.174 ± 0.024 | kept incumbent |

  Final incumbent after 6 iterations: **`az_v9_selfplay/iter2/best.pt`** — confirmed at **wr 0.5657 ± 0.0048, dScore +0.234 ± 0.023 over 7 independent draws** (the original 3-draw 0.5742/+0.263 was mildly winner's-curse-inflated). This is the project's deploy champion.

- **Full-trajectory interpretation (pending one confirmation, below):**
  1. The loop produced **one decisive promotion (iter-2)** and then plateaued: iters 3-6 cluster at wr ~0.545-0.558 / dScore +0.16-0.22 — consistently at-or-above the original baseline (0.5403/+0.164) but 3-5 SE below iter-2's mark, despite being warm-started FROM iter-2 and trained on supersets of its data.
  2. **CONFIRMED (7-draw eval, 2026-07-08): iter-2's strength is real.** All 7 independent draws: wr 0.5705/0.5849/0.5673/0.5656/0.5432/0.5703/0.5580 → **wr = 0.5657 ± 0.0048, dScore = +0.234 ± 0.023**. vs baseline: **Δwr = +0.025 (~4.1× combined SE), ΔdS = +0.070 (~2.5× combined SE)** — decisive on both metrics. The original 3-draw estimate (0.574/+0.263) was mildly optimistic (standard winner's-curse from gate selection), but the honest 7-draw number still clearly exceeds both the baseline AND the iters 3-6 descendant cluster (~0.545-0.558). So reading (b) holds with a correction: iter-2 is genuinely ~0.566, and each post-peak re-train on the growing buffer loses ~0.01-0.02 to churn — which the ratchet gate correctly refused four times.
  3. Either way, the self-play loop's headline stands: it produced the project's strongest checkpoint and its first statistically-decisive improvement (iter-2's promotion cleared the dScore gate at 2.1× and winrate at ~6× combined SE), validating the mechanistic fixes (policy-induced states, incumbent value at leaves, accumulation) over the az_v4..v8 null results.
  4. The ratchet design proved its worth twice: it refused four post-peak iterations that would each have degraded the deployed model by ~0.02 under an unconditional-promotion loop.

- **POST-HOC DIAGNOSIS (2026-07-08) — why nothing improves past iter-2. Three probes, one conclusion:**
  1. **Target-sharpness probe:** distillation targets collected by exp_021 have entropy 1.54 nats (top-action weight 0.52); targets collected by iter-2 have entropy 2.24 nats (top-weight 0.28). When the improved policy+value runs the same 2-ply/120-sim search, the search can no longer confidently rank the policy's own proposals — the operator's Q-surface flattened over the policy's candidates. This is what a distillation loop converging to its operator's fixed point looks like in the data.
  2. **Distill-loss cross-table:** iter-2 fits the current operator's targets far better than exp_021 does (6.37 vs 7.28 NLL); the residual gap vs its old training set (5.49) is ≈ fully explained by the entropy rise — i.e., near self-confirmation on a disagreement basis.
  3. **Fresh-window retrain (the decisive test):** warm-start from `iter2/best.pt`, train ONLY on it5+it6 (9,600 fresh iter-2-generation records, ZERO stale exp_021-era targets, volume-matched to iter-2's own training set), then 3-draw eval → **wr 0.5496 ± 0.0077, dScore +0.171 ± 0.017** — squarely in the iters-3-6 cluster, nowhere near the champion. **Stale-generation drag is ruled out as the cause of the post-peak decline.** The flat new-generation targets themselves cannot sustain training at the champion's level: weak distillation signal + continued VFM churn nets ~−0.015 wr per retrain from the peak. (Driver: `scripts/_az_v9_freshwin_launch.sh`; ckpt `checkpoints/csas_world/az_v9_freshwin/`.)

  **Closure: the 2-ply/120-sim operator's signal is genuinely extracted at iter-2.** The loop converged to (near) its operator's fixed point in one promotion; every further retrain — on stale mixture or pure fresh data — is net-negative churn that the ratchet gate correctly refused. Further improvement requires a sharper operator (e.g. mcts_sims 300+, which reduces the search's MC error and re-sharpens Q-discrimination among near-equal candidates, and/or true 3-ply with adequate sims), not more iterations of this one. **Champion stands: `az_v9_selfplay/iter2/best.pt` (wr 0.5657 ± 0.0048, dScore +0.234 ± 0.023 over 7 draws).**

---

## AZ-v10 / EXP-033 — `az_v10_terminal` (VALUE-FREE dense-candidate terminal-MC ratchet from the champion) — IN FLIGHT

- **Motivation (from the az_v9 closure + the certification discussion):** az_v9 converged to the fixed point of its 2-ply/value-leaf operator — a certificate conditioned on the learned V, which admits (π,V) co-adaptation fixed points. The tractable path to a *real* optimality certificate in a finite-horizon alternating zero-sum game is a **value-free operator**: dense candidate proposals scored by MC rollouts to TERMINAL with rule scoring — no learned model anywhere in the improvement operator. With self-play collection at every ply, later plies' improved policy feeds earlier plies' rollout scores — backward-induction-flavored without exponential trees.
- **Operator (per self-play ply; `search_state` with `terminal_rollout_scoring=true`, config `exp_033_terminal_dense.yaml`):**
  - **Dense proposal** (~119 after dedup): 48 policy samples + 16 structured + 32 diverse-grid + 24 local perturbations + 8 global uniform — *eventually-dense*, so the operator can find shots the policy assigns ~zero mass (the fundamental gap of policy-only proposals at any depth).
  - **Scoring:** each candidate executed under k_ego=4 noise realisations; each realisation rolled to terminal by the current policy; Q = mean realized rule-scored margin. Value-model-FREE (`search_rollout_n=1`).
  - **Cost:** smoke-timed h10 ≈ 35s, h6 ≈ 13s, h2 ≈ 2s → ~2.3 min/game; 128 games/shard ≈ 5-7h/iter collect.
- **Certification diagnostics** (new, written per shard as `.diag.json`): per horizon, the margin `Qbest_all − Qbest_policyOnly` — how much the dense proposal still beats the policy's own best sample under the value-free operator. Convergence of these margins toward ≤ MC-resolution IS the optimality certificate. (Smoke observation: at the h10 opening the dense proposal beat the policy's best sample by +0.75 — the champion may still be improvable early in the end.)
- **Loop:** ratchet harness, warm-start + collector = **`az_v9_selfplay/iter2/best.pt` (the champion)**, train cfg exp_021 (VFM=true; value targets remain grounded realized-game margins), dScore-primary gate at 1× combined SE, 3-draw evals, accumulating buffer, 2 iterations. Iter-0 baseline seeded from the champion's unbiased confirmation draws 4-6 (time-ordered, no selection): **wr 0.5597 ± 0.0084, dScore +0.222 ± 0.031**.
- **Outcomes:** promotion ⇒ the value-free operator expresses improvement the 2-ply/V operator couldn't (co-adaptation fixed point broken) — new champion + evidence the certificate chase has room. No promotion + shrinking margins ⇒ the champion gains the strongest available un-improvability certificate (dense proposal, MC resolution, on-policy states). Either is a headline result.
- **Driver:** `scripts/_az_v10_terminal_launch.sh` → `az_ratchet.py --selfplay-scorer terminal` (new flag; `selfplay.py --scorer terminal` branch). Launched 17:01 UTC 2026-07-08; ETA ~1 day for 2 iters.
- **RESULTS (landed 04:30 UTC 2026-07-09) — NO PROMOTION; un-improvability at this operator's resolution:**

  | iter | 3-draw wr ± SE | 3-draw dScore ± SE | gate | decision |
  |---|---|---|---|---|
  | 0 (champion, seeded draws 4-6) | 0.5597 ± 0.0084 | +0.222 ± 0.031 | — | incumbent |
  | 1 | 0.5456 ± 0.0107 | +0.149 ± 0.049 | Δds −0.073 | kept incumbent |
  | 2 | 0.5458 ± 0.0051 | +0.169 ± 0.010 | Δds −0.053 | kept incumbent |

  Same retrain-churn signature as az_v9 iters 3-6 and the fresh-window probe: every retrain from the peak lands ~0.55/+0.15-0.17.

- **Margin diagnostics — the certificate, with the necessary winner's-curse correction:**
  - Raw per-horizon margins (Qbest_all − Qbest_policyOnly): ~+0.09 at h1-h6 rising to ~+0.22 at h10; overall +0.129 (iter-1) and +0.130 (iter-2).
  - **These are consistent with PURE selection noise, not real improvement.** With per-candidate MC error σ ≈ 0.5-0.75 pts (k_ego=4 rollouts, margins are integer-ish), the expected max-of-noise gap between 119 and 48 candidates is ≈ σ(√(2ln119)−√(2ln48)) ≈ +0.15 — the measured magnitude. Two corroborations: (a) margins grow with horizon exactly as rollout-length-driven MC variance does; (b) margins did NOT shrink after training on the targets (+0.129 → +0.130) — real improvements would have been distilled away, noise artifacts persist.
  - **Certificate statement: the champion's policy proposals are un-improvable by the dense proposal at this operator's MC resolution (ε ≈ 0.5-0.75 pts per candidate, k_ego=4), under raw-policy continuations, on on-policy states.** The resolution is loose; tightening it needs adaptive allocation (more rollouts on the near-ties) — which is az_v11.
- **Comparison with previous best:** champion unchanged — `az_v9_selfplay/iter2/best.pt` (wr 0.5657 ± 0.0048, dScore +0.234 ± 0.023 over 7 draws). az_v10's checkpoints are not deploy candidates; its contribution is the (loose-resolution) value-free un-improvability certificate + the clean demonstration that flat uniform-budget operators can't resolve below their winner's-curse floor.
- **az_v11 (`az_v11_tree2term`) auto-launched by the chain** (az_v10 had no promotion): dense-root KR-UCT depth-2 + terminal-rollout leaves — visit-count adaptive allocation concentrates MC precision on the near-ties, directly attacking the resolution limit that capped az_v10's certificate.

---

## AZ-v11 / EXP-034 — `az_v11_tree2term` (dense-root KR-UCT depth-2 + terminal-rollout leaves) — DONE, NO PROMOTION; weakest retrains of the series — noise-starved targets

- **Operator (config `exp_034_tree2_terminal.yaml`, scorer `tree_terminal`):** the classic MCTS shape, value-free — root pre-seeded with the ~48-candidate dense proposal (new `mcts_search(root_candidates=...)`), KR-UCT visit-count adaptive allocation + exploration bonus to depth 2, leaves evaluated by on-policy MC rollout to terminal + rule scoring. sims=96, 64 games/shard (2,560 rec/iter). Collect ~15.7h/iter (sequential tree rollouts — no batching).
- **RESULTS (landed 19:48 UTC 2026-07-10):**

  | iter | 3-draw wr ± SE | 3-draw dScore ± SE | decision |
  |---|---|---|---|
  | 0 (champion, seeded draws 5-7) | 0.5571 ± 0.0078 | +0.196 ± 0.024 | incumbent |
  | 1 | 0.5372 ± 0.0085 | **+0.060 ± 0.030** | kept incumbent |
  | 2 | 0.5368 ± 0.0069 | **+0.043 ± 0.012** | kept incumbent |

  The **weakest retrained checkpoints of the entire series** (previous retrains: ~0.546-0.558 / +0.12-0.22). Champion unchanged.
- **Post-mortem — the operator was noise-starved (operator-quality lesson #2):** with 96 sims over 48 dense root candidates, the unvisited-first rule leaves each candidate ~2 fresh-noise evaluations, versus 4 in az_v10's flat operator and 8 in the deployed selector. In this execution-noise-dominated domain that violates the project's oldest hard-won lesson (EXP-013: noise-robust candidate evaluation is what first beat the prior). The distillation targets were soft-topk over ~2-sample Q estimates — sharp-looking winner's-curse artifacts — and training on them degraded the policy more than any previous target set. **Noise-robustness of the evaluation matters more than adaptive allocation or lookahead depth; an operator that trades it away produces actively harmful targets.**
- **Margin diagnostics:** raw +0.063 (iter-1) / +0.071 (iter-2), stable across training (→ artifact, not signal, same reasoning as EXP-033), and consistent with the 2-visit noise floor after kernel-pooling shrinkage. No evidence of real improvement over the champion's proposals.
- **az_v12 (`az_v12_screentree`) auto-launched:** the corrected two-stage operator — batched noise-robust screen (k_ego=8 per dense candidate) → KR-UCT depth-2 terminal-leaf tree over the top-8 survivors (~6 fresh-noise adaptive visits each). Restores EXP-013 robustness while keeping density, adaptivity, and 2-ply search-based continuations. Smoke diag: stage-1 margins ≈ 0 and the stage-2 winner originated from a POLICY candidate at all 10 plies of the smoke game.

---

## AZ-v12 / EXP-035 — `az_v12_screentree` (noise-robust screen → depth-2 tree over survivors) — DONE, NO PROMOTION; the operator ladder closes with the champion standing

- **Operator (config `exp_035_screen_tree.yaml`, scorer `screen_tree`):** the properly-constructed value-free operator for this noise-dominated domain — stage 1 screens the ~48-candidate dense proposal with the batched flat terminal-MC scorer at **k_ego=8** noisy executions per candidate (deployed-selector robustness, uniform precision); stage 2 refines the top-8 survivors with a KR-UCT depth-2 terminal-leaf tree (48 sims → ~6 adaptive fresh-noise visits per finalist). Sequential-halving-with-a-tree, value-free end to end. Collect ~8.3h/iter (64 games/shard, 2,560 rec/iter).
- **RESULTS (landed 20:09 UTC 2026-07-11):**

  | iter | 3-draw wr ± SE | 3-draw dScore ± SE | decision |
  |---|---|---|---|
  | 0 (champion, seeded draws 4/5/7) | 0.5556 ± 0.0066 | +0.202 ± 0.031 | incumbent |
  | 1 | 0.5453 ± 0.0045 | +0.157 ± 0.019 | kept incumbent |
  | 2 | 0.5467 ± 0.0077 | +0.134 ± 0.024 | kept incumbent |

  Retrains back in the normal churn band (~0.545/+0.13-0.16) — the noise-robustness fix repaired az_v11's target-quality collapse (+0.04-0.06), confirming that diagnosis — but still no improvement over the champion.
- **Diagnostics:** stage-1 margins +0.070/+0.071 (stable across iters → artifact, not signal; at k_ego=8 the expected pure winner's-curse gap between 48 and 24 candidates is ≈ +0.06 — the measured magnitude). Winner-origin: **~76% of executed moves came from the policy's own samples** after robust screening + tree refinement (~0.80 at h1-h6, dropping to ~0.60 at h9-h10). The drop at early plies is consistent with near-tie randomness over policy-heavy survivor pools rather than clear non-policy superiority; no per-horizon margin exceeds its noise floor.

- **CERTIFICATE SYNTHESIS (az_v10 + v11 + v12 — the final word on the optimality question):**
  The champion (`az_v9_selfplay/iter2/best.pt`) has now survived, without a single promotion, three families of value-free improvement operators on its own self-play states:
  1. flat dense terminal-MC, uniform k_ego=4 (az_v10);
  2. dense-root KR-UCT depth-2 with terminal leaves (az_v11 — noise-starved, its failure itself informative);
  3. robust-screen → adaptive depth-2 tree (az_v12 — density + robustness + adaptivity + 2-ply search continuations).
  In every case the improvement margins sat at each operator's winner's-curse floor and did not distill away, and every retrain landed below the peak (the universal ~0.01-0.02 retrain-churn). **Certificate: the champion's policy is un-improvable, at ε ≈ 0.2-0.3 pts MC resolution (k_ego=8) against an eventually-dense proposal under raw-policy continuations, on its on-policy state distribution — and no operator in the ladder could construct training targets that move it.** Residual caveats, quantified where possible: candidate density δ (~48-119 proposals/ply), MC resolution ε, rollout-continuation gap (raw samples vs deployed robust selection), and state coverage (on-policy from the 3 canonical openings).
  **Operator-quality lessons for the paper:** (1) noise-robust candidate evaluation dominates depth and adaptivity (az_v11 vs az_v12); (2) improvement margins from max-over-candidates comparisons MUST be winner's-curse-corrected or they fabricate signal (az_v10); (3) retraining past an operator's fixed point is universally net-negative churn — a promotion-gated ratchet is what converts a noisy loop into a monotone one.
- **FINAL STANDINGS (whole project):** champion `az_v9_selfplay/iter2/best.pt` — **wr 0.5657 ± 0.0048, dScore +0.234 ± 0.023** (7 draws) vs the human prior; +0.025 wr / +0.070 dScore over the pre-self-play consolidation ceiling (exp_021/exp_019 at ~0.546/+0.17). Produced by: self-play collection from pre-placed openings, 2-ply KR-UCT with the incumbent's value head at leaves, grounded MC value targets, accumulating buffer, dScore-gated ratchet — one promotion to convergence, then certified un-improvable by the az_v10-v12 ladder at their resolutions.

---

## AZ-v13 / EXP-036 — `az_v13_step0` (degradation-channel-fixed fine-tune of the champion) — IN FLIGHT

- **Goal:** the user wants to extend the improvement loop before concluding capacity exhaustion. Every post-champion retrain (9 of them, 4 data recipes) landed ~0.01-0.02 wr below the champion — not "no signal" but *negative* signal, via three identified channels. Step 0 fixes all three and retrains on EXISTING champion-generation data (zero recollection) to test whether the channels were the whole story.
- **Key enabling discovery:** the champion-generation targets are NOT uniformly flat — flatness was operator-dependent. Top-weight ≥0.5 fractions: az_v9-tree targets 14%, az_v10 flat-MC 23%, **az_v12 screen-tree 46% (mean top-w 0.511 ≈ the 0.521 of the data that trained the champion)**. There are ~6.5k confident distill plies sitting in existing buffers.
- **The three fixes:**
  1. **Flat-target erosion → significance-gated distillation** (data-side, no trainer change — `dist_mask` already gates the loss per record, `losses.py:72`): `_build_confident_buffers.py` writes filtered *copies* of az_v9 it3-6 + az_v10 + az_v12 (az_v11 excluded, noise-starved), zeroing dist_mask below per-source top-weight thresholds (0.55; 0.60 for az_v10's noisier k_ego=4 targets). Result: 30,720 train records (4,876 active-distill, 15.9%) + 10,240 val (1,587). Value/consistency/reward targets stay active on ALL records.
  2. **VFM drift → val-driven early stopping + drift guard on selection** (per user direction, replacing the crude epoch cap): new `train.early_stop_patience=4` (abort when the selection metric stops improving; epochs=20 is only a ceiling) and `train.select_value_guard=2.70` (an epoch is best.pt-eligible only if real-data val_value_mse ≤ champion's 2.3987 + 0.3). VFM=true kept — self-play value targets are grounded MC returns, the loop's compounding channel.
  3. **Misaligned checkpoint selection → `checkpoint_metric: val_policy_distill_mcts` on the confidence-filtered val partition** = "matches the search where the search is significant", replacing val_value_mse_mcts (twice measured strength-misaligned: EXP-024, az_v9b).
- **Success criteria:** retrain lands AT the champion (3-draw mean within ~1 SE) → channels fixed, churn eliminated; ABOVE the gate → residual signal existed, loop extends (→ Step 1: fresh screen_tree collection with collect-time significance masking + top-2 tie-break rollouts). At-or-below despite the fixes → genuine exhaustion evidence at this capacity.
- **Driver:** `scripts/_az_v13_step0_launch.sh` (train from champion + 3-draw eval). Launched 06:09 UTC 2026-07-13. Champion 7-draw reference: wr 0.5657 ± 0.0048, dScore +0.234 ± 0.023.

- **STEP-0 RESULTS (landed 09:58 UTC 2026-07-13):** wr **0.5514 ± 0.0051** (Δ −0.0143, −2.0× comb SE), dScore **+0.2373 ± 0.0213** (Δ +0.0034, **parity** — first retrain in project history to hold the champion's dScore; all 9 previous retrains lost 0.05-0.19). Mechanics worked as designed: early stop at epoch 136+4; drift guard never triggered (real-data val_value_mse actually improved, 2.29 vs champion's 2.40); selection metric declined smoothly to its optimum. **Verdict: degradation channels fixed on the primary metric; no promotion — the ~4.9k confident plies from existing buffers carried no new signal** (consistent with much post-hoc "confidence" being operator noise-sharpness rather than true Q-gaps — the post-hoc top-weight proxy can't distinguish them; the collect-time t-test in Step 1 can). Residual wr −0.014 at 2σ: either remaining trunk-drift churn or 3-draw luck (draw 1 was 0.5413; draws 2-3 0.5582/0.5546).
- **→ Step 1 (launched next): fresh screen_tree collection with collect-time SIGNIFICANCE masking** — per-candidate Q means AND standard errors from the k_ego=8 screen; dist_mask=1 only when the top-1 vs top-2 gap passes t ≥ 2, with a batched tie-break round (extra realisations for the top-2) when 0.8 < t < 2. This distills only statistically-real preferences — the cleanest possible test of whether ANY extractable signal remains at this capacity.
- **STEP-1 SMOKE FINDING (the most clarifying number of the operator investigation): only ~10% of plies (2/20 in the smoke) pass t ≥ 2** — vs az_v12's 63% apparent "confidence" by top-weight. ~85% of the sharp-looking search preferences were MC noise dressed up by soft-topk (temperature 0.35 turns ±0.25 noise into confident-looking weights). The operator genuinely discriminates on ~1-2 plies per game. Consequence for the launch: the Step-0 buffers are pre-seeded as **dist-neutral anchors** (az_v13_anchor_*: dist_mask zeroed everywhere; value/consistency targets only) so that distillation in Step 1 trains EXCLUSIVELY on statistically-real preferences (~500 significant plies/iter expected). Driver `scripts/_az_v13_step1_launch.sh`: exp_037 collect (sig-gated screen_tree) + exp_036 train (aligned selection, early stop, drift guard), 2 ratchet iterations from the champion, dScore gate. Launched 10:29 UTC 2026-07-13.

- **STEP-1 RESULTS (landed 10:05 UTC 2026-07-14):**

  | iter | significant plies (t≥2) | 3-draw wr ± SE | 3-draw dScore ± SE | decision |
  |---|---|---|---|---|
  | 0 (champion, seeded 4-6) | — | 0.5597 ± 0.0084 | +0.222 ± 0.031 | incumbent |
  | 1 | 209/2,560 (8.2%) | **0.5571 ± 0.0052** | **+0.225 ± 0.030** | kept (**FULL PARITY — first ever, both metrics**) |
  | 2 | 226/2,560 (8.8%) | 0.5514 ± 0.0008 | +0.177 ± 0.015 | kept (small dip, near-band) |

- **FINAL VERDICT of the az_v13 program — the loop is now provably clean and provably empty at this data rate:**
  1. **The retrain-degradation problem is solved.** Iter-1 held the champion on BOTH metrics (Δwr −0.003, Δds +0.003) — no prior retrain (10 attempts) ever did. Significance-gated distillation + dist-neutral anchoring + aligned selection + val-driven early stop + drift guard together eliminate the universal ~0.01-0.02 churn.
  2. **The true improvement-signal rate is ~8.5% of plies (~1.7 plies/game), stable across iterations** — and distilling exactly those plies (≈200-435 examples/iter) produces no measurable game-strength gain.
  3. **Honest interpretation — two readings, not one:** (a) capacity exhaustion at 7.3M params; or (b) **signal sparsity**: the significant plies are real but ~400 examples over a continuous 24-dim state space is too thin to learn a *generalizable* correction — the policy would need ~10× more significant-ply examples (~40-90h collection per iteration, partially cost-reducible by dropping stage-2 for non-significant plies) before "unlearnable" is distinguishable from "at capacity". The az_v13 verdict licenses capacity scaling per the user's criterion, but does NOT rule out the sparsity reading.
  4. **Champion unchanged through 14 post-champion retrains and 5 operator families: `az_v9_selfplay/iter2/best.pt`** (wr 0.5657 ± 0.0048, dScore +0.234 ± 0.023 over 7 draws).

---

## ANALYSIS (2026-07-15) — why post-champion iterations regressed instead of tying

See **`analysis_why_iterations_regressed.md`** (repo root). TL;DR: the pre-az_v13 systematic
regression was H3-class (misaligned checkpoint selection) + flat/noisy-target erosion — proven
by the az_v13 fixes reaching full parity; the post-fix residual is peak-selection statistics
(the champion is the promoted upper tail of a stochastic process; unbiased retrains scatter
at-or-below it when no signal remains); the human-derived value anchor (mix_value slice +
drift guard treating human-play calibration as ground truth) is a real uncontrolled tension —
az_v13 retrains IMPROVED human-value fit (2.30-2.32 vs champion's 2.40) without gaining
strength; and game-theoretic self-play drift is second-order (no monotone decay vs the prior)
but enters through the value function and through the t≥2 gate defining "improvement" under
self-continuations rather than vs the eval opponent. Cost-ordered follow-ups listed in the doc
(mix_value ablation; self-play value buffer; mixed-opponent collection vs the prior).

---

## AZ-v14 / EXP-038 — `az_v14_big` (capacity scaling: 23.2M params from scratch) — IN FLIGHT

- **Goal:** test reading (a) of the az_v13 verdict — is the ~0.566 ceiling a capacity limit of the 7.3M model? Per the plan, this is the quicker experiment (~1 day); the 10× significant-ply collection (reading (b), signal sparsity) runs later on a cheaper CPU instance since collection is JAX-CPU-dominated.
- **Model:** `hidden_dim=384, n_layers=6, n_heads=6` → **23.2M params (3.2×)**. All dims config-driven; verified instantiation + forward. The canonical 256-dim prior can't be absorbed, so `build_model` now gracefully skips the prior warm-start on dim mismatch (true from-scratch; the human-BC replay slice bootstraps the policy: `policy_bc=0.3, mix_human=0.15`). action_mean/std buffers come from canonical defaults (warm-start-independent).
- **Corpus** (assembled by `_az_v14_launch.sh`, hygiene per COLLECTIONS.md): az_v9 it1-2 raw (9,600 — the champion's diet, sharp exp021-gen targets) + az_v13_conf_train (30,720 champion-gen, 0.55-filtered dist) + az_v13_ratchet it1-2 (3,840, t≥2-gated) = **44,160 train records**; 14,720 val (incl. sig-masked partitions for the aligned selection metric); plus the usual human/value/sim replay slices. VFM=true (grounded MC returns).
- **Training:** az_v13-fixed recipe — `checkpoint_metric=val_policy_distill_mcts`, `early_stop_patience=6`, `select_value_guard=3.0` (looser: from-scratch calibration), epochs=60 ceiling, samples_per_epoch=120k. ~8-10 min/epoch at 23.2M.
- **Then:** 3-draw eval vs prior; compare to the champion (wr 0.5657 ± 0.0048, dScore +0.234 ± 0.023). Outcomes: big model ≥ champion + gate → capacity was binding, new lineage begins (self-play ratchet with the big model next); big model ≈ champion → capacity NOT binding at 3.2× → the sparsity reading gains weight and the 10× collection becomes the decisive experiment; big model < champion → from-scratch-on-corpus underperforms the curriculum lineage (informative about training path dependence, would retry with distill-from-champion init before concluding).
- **First launch crashed** on the prior warm-start dim mismatch (fixed as above); relaunched 07:24 UTC 2026-07-15.

- **RESULTS (landed 14:52 UTC 2026-07-15): outcome (3), extreme form — the from-scratch 23.2M model is far BELOW even the human prior.** 3 draws: wr 0.4920/0.4771/0.4524 → **0.4738 ± 0.0116, dScore −0.152 ± 0.049** (champion: 0.5657/+0.234). Yet its training metrics looked healthy — val conf-distill 5.24, human-value fit 2.36, all 60 epochs used with the selection metric still improving. **Reading: this run measured PATH DEPENDENCE, not capacity.** The champion's strength rests on the canonical human-prior policy (a dedicated full-covariance MDN trained extensively in csas_v3) as the foundation of its proposal distribution; a 15% human-BC replay slice cannot replicate that foundation, and the deployed 48-sample selector turns a diffuse proposal distribution into losing play regardless of how well the corpus targets are fit. The capacity question remains OPEN.
- **az_v14b RESULTS (landed 21:18 UTC 2026-07-15): distill-then-finetune ALSO broken — wr 0.4689 ± 0.0094, dScore −0.211 ± 0.022** (same territory as from-scratch), despite phase-1 matching the champion's VALUE outputs nearly perfectly (val_value_mse_mcts 0.021) and suspiciously identical distill losses (~5.23-5.24) across all big-model runs on different val sets.
- **ROOT CAUSE FOUND (diagnostics 2026-07-15): the action-normalization buffers.** Both big models skipped the prior warm-start (dim mismatch) and fell back to `_DEFAULT_ACTION_MEAN/STD` in `model.py` — which are the STALE PRE-FULLSHEET stats (speed mean 1.26 vs the true fullsheet 2.45; std 0.39 vs 0.159; spin std 0.84 vs 2.66). The champion inherits correct stats from the prior's warm-start. Measured consequence: the student's sampled proposals are ~2× too wide on the speed dimension *despite a better-than-teacher NLL on the teacher's samples* (mode-covering MDN pathology: sharp density at the sample points, spurious spread under sampling), which wrecks the deployed 48-sample selector. **Both az_v14 runs are therefore INVALID as capacity evidence.** Latent bug note: the fullsheet migration (2026-06-16) never updated the world-model defaults; the bug only bites models built without warm-start — az_v14 was the first ever.
- **→ az_v14c (launched 21:32 UTC): identical distill-then-finetune, but phase 1 initializes from a seed ckpt (`az_v14c_seed.pt`) = fresh 23.2M model with the CHAMPION's action buffers copied in.** Same distill set (reused), same exp_039/exp_040 recipe, 3-draw eval. This is the corrected capacity test.
- **az_v14c RESULTS (landed 03:51 UTC 2026-07-16): buffer fix recovered ~+0.025 wr but the gap remains — wr 0.4954 ± 0.0078, dScore −0.099 ± 0.037.** Post-fix diagnostics: value transfer near-perfect on corpus states (MSE 0.02), but the student's per-state proposal cloud is still systematically wider than the champion's (speed-spread ratio median 1.34, p90 2.25) — uniform-24-sample distillation transfers the coarse distribution, not per-state sharpness (MDN mode-covering). Untested additional gap: student V matched on corpus x0 only; the deployed selector queries V on arbitrary candidate post-states. NLL comparisons across differently-normalized models are Jacobian-shifted and meaningless (explains the earlier "suspicious 5.24 constant"). **Conclusion: the 384-wide route is a foundation-transfer engineering problem, not a capacity readout. Parked.**
- **→ az_v14d (launched 03:57 UTC 2026-07-16): DEPTH-EXTENDED CHAMPION — the clean capacity design.** Keep d=256 (champion width) and grow 4→8 layers: ALL champion weights (trunk 0-3, every head, action buffers) load natively; new layers 4-7 initialized near-identity (zeroed W_o/ffn-out, identity norm affines — post-LN means not bit-exact: encode Δ ≈ 10% of feature scale, policy-mu Δ ≈ 0.055 z-units, re-adapted by fine-tuning). 13.65M params (1.9×). Fine-tune on the az_v14 corpus with the az_v13 recipe (`exp_041_depth_extend.yaml`, guard 2.80), then 3-draw eval. Any gain over the champion = capacity effect with zero transfer risk; parity = capacity not binding at 1.9× depth.
- **az_v14d RESULTS (landed 07:51 UTC 2026-07-16) — STATISTICAL PARITY; the capacity question is answered:**

  | | wr ± SE | dScore ± SE |
  |---|---|---|
  | champion (7.3M) | 0.5657 ± 0.0048 | +0.2339 ± 0.0232 |
  | az_v14d (13.65M depth-extended champion) | 0.5580 ± 0.0029 | **+0.2465 ± 0.0067** |
  | Δ | −0.0077 (−1.4× SE) | +0.0126 (+0.5× SE) |

  Early stop at epoch 8+4; drift guard clean. az_v14d's dScore is nominally the highest 3-draw mean recorded (with unusually tight draws), but under the dScore-primary gate this is parity, not promotion. Deploy stays with the 7.3M champion (half the parameters, same strength); az_v14d/best.pt is a parity-equivalent alternative.

- **CAPACITY-ARC CLOSURE (az_v14 a/b/c/d):**
  1. **Capacity is NOT the binding constraint at ~2×**: giving the champion 4 extra near-identity layers and fine-tuning on the full 44k-record corpus with the fixed recipe reproduces its strength exactly — no capacity dividend from the same data.
  2. The 3.2×-wide route (v14a/b/c) never produced a capacity readout — it produced a catalog of **foundation-transfer failure modes**, each diagnosed: stale pre-fullsheet `_DEFAULT_ACTION_MEAN/STD` in `model.py` (latent since the 2026-06-16 fullsheet migration; only bites warm-start-free construction); uniform-sample MDN distillation transferring coarse distribution but not per-state sharpness (mode-covering; sampled proposals 1.34-2.25× too wide); value-head generalization to off-corpus post-states unverified; cross-normalization NLL comparisons Jacobian-invalid.
  3. **Combined with az_v13's verdict, the project's ceiling story is now fully determined**: the loop is clean (full-parity retrains), the extractable signal rate is ~8.5% of plies, distilling exactly those plies at ~400 examples/iter moves nothing, and 1.9× capacity doesn't change any of it. **The front-running explanation for the remaining ceiling is SIGNAL SPARSITY — and the decisive next experiment is the ~10× significant-ply collection, planned for the cheaper CPU instance** (collection is JAX-CPU-dominated; see analysis_why_iterations_regressed.md §7 for the also-pending H2/H1 ablations, which are config-cheap and can run before or alongside).
- Original az_v14b design notes follow (mechanics unchanged in v14c):
  **DISTILL-then-FINETUNE.** Phase 0: champion-distillation set — 24 policy samples (temp 1.0, uniform weights) + champion V targets on all 44k corpus states (`_build_champion_distill_set.py`; 26 train + 2 val shards). Phase 1 (exp_039): distill champion → 23.2M (30-epoch ceiling, patience 5, selection = matches-the-champion distill metric). Phase 2 (exp_040): az_v13-recipe fine-tune on the real corpus (--init phase-1 best, guard 2.9). Then 3-draw eval. If the distilled+finetuned big model ≥ champion + gate → capacity was binding after all; ≈ champion → capacity not binding at 3.2×, sparsity reading strengthens; still ≪ champion → distillation notes to be examined before further conclusions (e.g., mixture-collapse under uniform-weight sampling).

---

## METAGAME / EXP-042 — meta-game payoff matrix (PSRO step 0; the missing BR proof) — IN FLIGHT

- **Question (raised 2026-07-16, see analysis_why_iterations_regressed.md §6b):** does the training
  paradigm provably find a best response to its TRAINING opponent, even where that doesn't transfer
  to the human prior? **Evidence hole confirmed: every eval in az_v9→v14 was vs the prior; the modern
  loop never measured new-vs-incumbent head-to-head.** The t≥2 convergence diagnosis is an approximate
  self-play-equilibrium claim (un-improvable vs self at ε), not a BR claim; and the operator is a
  policy-iteration-style BR *approximator* (one-ply deviations, learner continuations, self-play V),
  not an exact oracle.
- **If BR-ability is confirmed, two consequences follow** (the user's argument, endorsed): (i) the
  human-value anchor (H2) becomes unjustifiable — a validated BR-improver should price positions by
  realized margins from the actual matchup, not human-play values; (ii) **PSRO/double-oracle** becomes
  the principled escalation with our loop as the BR oracle — and it fixes H1 exactly (the prior sits
  IN the population; BRs are trained vs the Nash mixture; certificate = "BR gains ≤ ε vs mixture").
- **Infrastructure fit:** ~80% exists (population of provenance-tracked ckpts; multi-draw noisy h2h =
  matrix evaluator; ratchet gate = admission rule; sig-gated collection = oracle targets). The one
  build item: **opponent-aware collection** (parity-switched policies in the self-play game loop and
  MC rollouts) — which simultaneously implements the H2 fix (matchup-true value targets) and the H1
  fix (significance gate computed under the actual opponent).
- **Run:** `scripts/_metagame_matrix.sh` → `eval_out/metagame/` — noisy N=400 × 10-horizon h2h for the
  6 new pairs among {exp_021 best, champion (az_v9/iter2), az_v13-it1, az_v14d}; vs-prior column reused
  from existing multi-draw evals. Launched ~08:5x UTC 2026-07-16, ~5h.
- **Pre-registered decision tree:** (a) BR confirmed + non-transitivity → full PSRO (LP meta-solver +
  opponent-aware oracle); (b) BR confirmed + transitive → minimal H1+H2 fix only ("az_v15: ratchet vs
  prior-inclusive mixture"); (c) BR not confirmed → reinterpret the az_v9 promotion (likely value-head
  driven), revise the analysis doc, PSRO premature.
- **RESULTS (landed 14:0x UTC 2026-07-16) — the matrix OVERTURNS the plateau narrative:**

  dScore payoff matrix (row's margin/end vs column; h2h entries single-draw, ±0.03-0.05; vs-prior column = multi-draw means):

  | | prior | exp021 | champ | v13it1 | v14d |
  |---|---|---|---|---|---|
  | prior | — | −0.176 | −0.234 | −0.225 | −0.246 |
  | exp021 | +0.176 | — | −0.013 | −0.111 | −0.161 |
  | champ | +0.234 | +0.013 | — | −0.036 | −0.106 |
  | v13it1 | +0.225 | +0.111 | +0.036 | — | −0.082 |
  | **v14d** | **+0.246** | **+0.161** | **+0.106** | **+0.082** | — |

  1. **The meta-game is PERFECTLY TRANSITIVE** — all 10 pairwise signs align into one order, no cycles: **v14d > v13it1 > champ > exp021 > prior**. Restricted-game Nash is PURE: az_v14d, weight 1.0.
  2. **BR-ability: CONFIRMED (for the fixed-recipe retrains).** v13it1 and v14d — both trained on champion-generation data — beat the champion head-to-head (+0.036, +0.106 dScore). The paradigm does improve against its training opponent. (champ > exp021 is only +0.013, within single-draw noise; the ordering is supported by the global sign-consistency.)
  3. **The H1 transfer gap, quantified:** v14d beats the champion by +0.106/end head-to-head while their vs-prior dScores are statistically tied (+0.2465 vs +0.2339). Improvements against the self-play lineage transfer only ~10-15% to the prior matchup. **The vs-prior gate has been REJECTING genuinely stronger models** — az_v13-it1 and az_v14d were both "kept incumbent" while actually dominating the champion.
  4. **The "plateau" was a measurement artifact of a saturating fixed-opponent metric.** In head-to-head terms the model sequence exp021 → champ → v13it1 → v14d is a strictly improving chain — self-play iterations were compounding all along, invisible at ~+0.24 dScore vs the prior where that matchup appears to saturate. **This also revises the az_v14d capacity verdict: the depth-extended model IS the strongest policy produced by the project (dominance/Nash sense); capacity DID pay — just not in the vs-prior column.**
  5. **Decision-tree branch: (b) — transitive** → PSRO's population machinery reduces to a ratchet; the correct minimal fix is an **incumbent-relative gate** (promotion = beat the incumbent head-to-head, dScore-primary — the classic AlphaZero gate, which our matrix retroactively validates) plus the H2 fix (drop the human-value anchor; value targets from the actual training matchup).
- **CONFIRMATION DRAWS (landed 2026-07-16): az_v14d's dominance over the champion is CONFIRMED — h2h dScore −0.1018 ± 0.0052 over 3 draws (−0.106/−0.108/−0.091, ~20× SE).** Notable: the h2h winrate is dead-even (0.5003) while dScore is decisive — v14d wins by larger margins; a winrate-primary gate would have missed this promotion entirely (vindicates the dScore-primary convention). v13it1-vs-v14d remains unresolved within noise (−0.032 ± 0.046).
- **★ FORMAL PROMOTION (2026-07-16): `checkpoints/csas_world/az_v14d/best.pt` (13.65M, depth-extended lineage) is the GLOBAL CHAMPION** — beats the previous champion h2h by +0.102/end (3-draw), best vs-prior dScore ever (+0.2465 ± 0.0067), pure Nash of the restricted meta-game, dominates every model in the population. The 7.3M az_v9/iter2 remains the strongest small model and the reference for the L4 scaling leg.
- **az_v15 L4 leg auto-launched on draw completion** (corrected loop: incumbent-relative dScore gate, matchup-true value guard, mix_value 0.10; from v13it1; stop after 2 consecutive non-promotions).

---

## AZ-v15 / EXP-043-045 — L4 leg of the scaling study (corrected loop, train-to-capacity certificate) — LEG DONE, certificate QUALIFIED pending the mix_value A/B

- **Setup:** the corrected loop's first run — incumbent-relative dScore gate (h2h vs incumbent, promote iff ds > 1×SE with wr guard; `--gate-opponent incumbent --stop-after-nonpromotions 2`), matchup-true value guard (`select_value_guard_metric: val_value_mse_mcts`, 3.60), human anchor demoted (exp_043: mix_value 0.10, mix_mcts 0.80), sig-gated screen_tree collection, from **v13it1** (strongest L4 model).
- **RESULTS (gate-converged 17:51 UTC 2026-07-17):**

  | iter | h2h vs incumbent (ds ± SE) | wr | sig rate | decision |
  |---|---|---|---|---|
  | 1 | −0.0676 ± 0.0260 | 0.4998 | 9.3% | kept |
  | 2 | −0.0320 ± 0.0317 | 0.5054 | 8.5% | kept → stop rule fired |

  **L4 certificate (provisional): v13it1 is the 7.3M ceiling under this loop.** Convergence signals: gate ✓ (2 non-promotions), operator ✓ (sig rate flat at ~8.5-9% floor), value ✗ (matchup-true val MSE still improving 3.21→2.79 across iters — signal 3 NOT plateaued). Per-head trajectories: consistency/dynamics flat (−0.118→−0.116; first direct evidence on the dynamics-undertraining question — it is not visibly improving, soft due to partition drift); human-value fit degraded vs v13it1's 2.32 (2.75/2.47) as expected under the reduced anchor.
- **Two qualifications, hence the A/B (EXP-044, launched immediately):** (a) the h2h trend was improving toward parity (−0.068 → −0.032) when the pre-registered stop rule fired; (b) the recipe changed simultaneously with the gate — under the old mix_value=0.30 recipe the first az_v13 retrain reached +0.036 h2h over its incumbent, vs −0.068 here under 0.10 — suggestive that the human-value slice was load-bearing REGULARIZATION for deployed play (H2's sign possibly wrong for deployment even if right for calibration). **EXP-044 = single-factor A/B**: identical accumulated data, identical incumbent + warm-start + guard, mix_value 0.30/mix_mcts 0.60, 3-draw h2h gate vs v13it1. Compare directly against iter-2's exp_043 result (−0.032 ± 0.032). If the 0.30 variant promotes → recipe corrected, leg resumes; if it matches iter-2 → mix_value exonerated, certificate stands as-is.

- **A/B VERDICT (landed 21:14 UTC 2026-07-17): the human-value anchor is EXONERATED as a factor — and in fact hurts.** exp_044 (mix_value 0.30): h2h vs v13it1 = **ds −0.0745 ± 0.0031** (draws −0.081/−0.073/−0.070) vs exp_043's −0.0320 ± 0.0317. Restoring the anchor made the retrain clearly worse head-to-head. The earlier suggestive comparison (az_v13-it1's +0.036 under 0.30) was confounded by its weaker incumbent, as suspected. **mix_value 0.10 stays; the H2 fix is confirmed in the deployed-play sense too.**
- **★ L4 CERTIFICATE ACCEPTED: `az_v13_ratchet/iter1/best.pt` is the trained-to-capacity 7.3M model.** Both recipe variants fail to beat it on identical data; gate + operator signals fired; the sole unfired signal (matchup-true value-val still improving across iters) is the certificate's residual caveat.
- **→ L8 leg launched (2026-07-17 ~21:3x UTC):** corrected loop from the global champion az_v14d (13.65M), config exp_045 (= exp_043 at n_layers 8), incumbent-relative gate, stop after 2 non-promotions, up to 8 iters. Pre-launch fix: `export_policy` now embeds the MODEL's arch in the exported csas policy (it hardcoded default L4 — would have silently broken L8 collection); the L8 export→load→sample round-trip was smoke-verified before launch. (L8 iter-1: h2h −0.0145 ± 0.0260 → kept incumbent.)

- **→ EXP-046 / az_v15-VH — VALUE-HEAD-ONLY continuation (chained after the L8 leg; addresses the L4 certificate's unfired value signal).** Rationale: joint "train until all val losses bottom" is unsafe (val/strength misalignment measured thrice: EXP-024, az_v14, the L4 leg itself) because heads bottom at different times — but the *per-head surgical form* is sound: new `train.train_value_head_only` freezes everything except the value head (trunk included → deployed proposals bit-identical; any h2h change causally attributable to V; with one trainable head, `val_value_mse_mcts` becomes an ALIGNED selection/stopping criterion by construction). Recipe exp_046: from v13it1 on the L4-leg accum data, value losses only, early-stop patience 5, then 3-draw h2h gate vs v13it1. **Promotion ⇒ the L4 certificate was premature and a "value-refinement phase" joins the train-to-capacity protocol (would apply to L8's certificate too); parity ⇒ certificate stands with the caveat resolved.** This is the first live use of the progressive-freezing instrument from the per-head training-schedule discussion.

---

## AZ-v15 L8 leg / EXP-045 — DONE, gate-converged: az_v14d is the trained-to-capacity 13.65M model

- **RESULTS (gate-converged 20:47 UTC 2026-07-18):** iter-1 h2h vs incumbent −0.0145 ± 0.0260 (kept), iter-2 −0.0414 ± 0.0341 (kept) → stop rule fired after 2 non-promotions. **L8 certificate: `az_v14d/best.pt` is the ceiling at 13.65M under this loop** (same residual caveats as L4; the VH continuation applies to this certificate too and is queued).
- **Scaling-relevant observations:** (a) the significance rate is LOWER at L8 (6.9-7.2%) than at L4 (8.5-9.3%) — the operator finds less teachable improvement against the stronger model, consistent with the dominance ordering; (b) unlike L4, matchup-true value-val did not improve across iterations (2.71 → 2.91, partition-drift-confounded), so the value-refinement question is weaker here.
- **Scaling-study status:** two certified points — L4 ceiling = v13it1, L8 ceiling = v14d. **The headline edge (converged-L8 vs converged-L4 = v14d vs v13it1) is still statistically unresolved (−0.032 ± 0.046 over 3 widely-dispersed draws)** — 4 additional draws chained after the VH run (`_scaling_edge_draws.sh`) to bring the SE to ~±0.025. Queue: L8-leg done → VH continuation (value-refinement test on the L4 certificate) → scaling-edge draws.

---

## EXP-046 VH + SCALING EDGE — both resolved (2026-07-19); L12 creation (az_v16 / EXP-047) launched

- **Scaling edge RESOLVED: converged-L8 beats converged-L4 by +0.052 ± 0.019/end** (v13it1 vs v14d, 7 draws; the +0.060 draw remains the lone outlier; h2h wr dead-even 0.5004 — margin-of-victory dominance again). The scaling study's first two certified points: **L4 ceiling (v13it1) < L8 ceiling (v14d) by ~0.05/end.** Capacity produces real converged-strength gains.
- **EXP-046 VH verdict: value-head-only refinement does NOT extend the L4 ceiling.** The freeze worked (12 trainable / 216 frozen tensors after the DDP-ordering fix — freeze must precede the DDP wrap, reducer hooks bind at construction); the value head reached a better val optimum (2.62 vs the leg's 2.79) with deployed proposals bit-identical — and still landed slightly BELOW the incumbent in play (h2h −0.027 ± 0.014). **Better global value calibration ≠ better candidate ranking at decision-relevant states. The L4 certificate now stands in full** (gate + operator + value signals all resolved); the same reasoning retroactively firms the L8 certificate (where value-val wasn't even improving).
- **→ az_v16 / EXP-047 (launched 05:07 UTC 2026-07-19): L12 creation** — v14d + 4 near-identity layers (surgery essentially EXACT this time: encode Δ 0.0006, vs 0.046 for v14d's own creation — v14d's fine-tuned features are nearly LN-invariant), 19.99M params, fine-tuned on the SAME az_v14 corpus as v14d's creation (clean +capacity analogy), corrected recipe (exp_047 = exp_043 at L12), then 3-draw h2h gate vs v14d. If it beats v14d like v14d beat the champion (+0.102), depth scaling is compounding; the L12 loop leg would follow.

- **EXP-047 VERDICT (7 draws, 2026-07-19): PARITY — the depth ladder FLATTENS at L12.** L12-create vs v14d: wr 0.5043 ± 0.0070, dScore **−0.018 ± 0.024** (early positive draws regressed to the mean; the familiar first-draw winner's curse). Same corpus, same recipe, +46% params → nothing absorbed (early stop at epoch 6, like v14d's — but this time with no strength gain to show). **No promotion; az_v14d remains global champion.**

- **★ SCALING STUDY — COMPLETE (first pass). The table:**

  | size | model | converged/created strength (h2h chain) | loop certificate |
  |---|---|---|---|
  | 7.3M (L4) | v13it1 | baseline of chain | certified (gate+operator+VH-resolved) |
  | 13.65M (L8) | **v14d (champion)** | **+0.052 ± 0.019/end over L4 ceiling** | certified (gate+operator) |
  | 19.99M (L12) | az_v16_create | +0.000 (parity with v14d: −0.018 ± 0.024) | creation flat → leg not warranted |

  **Unified conclusion of the entire az_v9→v16 program: DATA is now the binding constraint.** Every size trained on the same ~44k-record corpus; capacity paid once (L4→L8) and then flattened (L8→L12); loop extractability is certified exhausted at two sizes (~7-9% significance rates, immediate gate convergence); per-head value refinement resolved negative; the vs-prior matchup saturates. All training-side axes on this box are spent. **The unlocking move is COLLECTION SCALING — the user's planned experiment on a cheaper CPU instance** (collection is JAX-CPU-dominated): a 5-10× corpus would give every rung of the ladder (and the flattened L12 rung in particular) something new to absorb, after which the depth ladder and the corrected loop are both ready to re-run with known costs and certified instruments.

---

## AZ-v17 / EXP-048 — BIG-CORPUS COLLECTION (the data-wall experiment) — IN FLIGHT on g5.4xlarge

- **Motivation:** the completed scaling study's unified conclusion — every training-side axis exhausted; DATA is the binding constraint (capacity paid L4→L8 then flattened at L12 on the same 44k corpus; loop significance floors at ~7-9%; ~400 significant plies/generation too sparse to learn from). This collection is the unlocking experiment.
- **Setup (instance switched to g5.4xlarge: 16 vCPU, 1× A10G):** collector = global champion az_v14d (policy export `az_v15_L8/incumbent0_policy_csas.pt`); operator = the certified sig-gated screen_tree (exp_037: robust k_ego=8 screen → depth-2 terminal-leaf tree over top-8, collect-time t≥2 significance masking with tie-break rollouts). Driver `scripts/_collect_bigcorpus.sh`: rounds of 6 workers × 16 games sharing the single GPU, fully resumable, per-shard manifests + certification diags, graceful stop via `artifacts/replay/mcts/az_v17_bigcorpus/STOP`.
- **Single-GPU port required two fixes:** (a) **new `POLICY_BATCH_CAP` env chunking** in `_mc_rollout_terminal_batch` — the curl-arc edge features spike to ~7-10 GB per worker at rollout batch ≈380 (fine with per-worker GPUs, OOM-fatal with 8 workers on one A10G); cap=96 → ~2-3 GB/worker; (b) `PYTORCH_CUDA_ALLOC_CONF=expandable_segments`. Benchmarks: 8 workers uncapped = 7/8 OOM; 6 workers capped = clean, **26.5 games/hour (~635/day)**.
- **Target: 5,000 games = 50,000 records ≈ ~4,000 significant distill plies (~10× any previous generation's pool), ETA ~8 days.** Corpus is checkpoint-useful at any point (round-based shards); progress in `collect.log` (cumulative games + per-round sig rates). Launched 05:47 UTC 2026-07-20.
- **On completion:** train the next generation on the big corpus (likely at L8 and L12 — does the flattened rung wake up with 10× data?), corrected-loop legs with the certified instruments. Training/eval will need the GPU box again (DDP configs assume 4 GPUs).

- **INCIDENT LOG (the collection took 3 attempts):** (1) Jul 20-23: 41 rounds crashed silently at ~90 min/worker — root cause `vm.max_map_count` = 65,530 default; the JAX CPU JIT allocates one mmap per compiled kernel section and the screen_tree operator compiles thousands of shape-specialized sim kernels per long-lived worker (caught red-handed: a dying worker at 65,531 maps). Fix: sysctl → 1,048,576 (persisted in /etc/sysctl.d/99-jax-maps.conf) + 90s staggered starts + 6h/worker timeout. The 2-game benchmark had passed because it never accumulated enough kernels — sizing tests must match production shape counts. The driver also logged rounds as "done" without verifying shards on disk — never trust a driver's own summary. (2) A second schedule loss (~10h idle): the test-chain's drain-wait `pgrep -f world.search.selfplay` matched stale harness shells containing that literal text — the second self-matching-pgrep incident.
- **PAUSED at 1,904 games (user decision: early peek)** = 19,040 records, ~1,390 significant plies (~3× the largest pool ever trained on; the full 5k target would be ~9×).

## AZ-v18 / EXP-050 — PAUSED-CORPUS TRAINING TESTS (L8 data-wall + L12 wake-up) — DONE, DOUBLE NULL at 3×

- **Setup (single-GPU box):** corpus split 95 train / 24 val shards; single-GPU configs (exp_048 = exp_045@gpus[0], exp_049 = exp_047@gpus[0]); eval with 4 shard-workers sharing GPU 0 (protocol otherwise identical — numbers comparable). Run 1: L8 fine-tune from champion az_v14d. Run 2: L12 fine-tune from az_v16_create. Both: corrected recipe, 3-draw h2h gate vs az_v14d.
- **RESULTS (landed 14:10 UTC 2026-07-27):**

  | test | h2h vs champion (3 draws) | verdict |
  |---|---|---|
  | L8 + 19k fresh records | wr 0.5016 ± 0.0045, ds **−0.019 ± 0.036** | parity — no promotion |
  | L12 + same corpus | wr 0.5037 ± 0.0081, ds **−0.009 ± 0.029** | parity — no promotion |

  Both trainings healthy (best epochs 10/9, guards clean, early stops normal).
- **Interpretation (pre-registered asymmetry applies):** a **double null at ~3× significant plies** — tripling the fresh champion-generation data moves neither the champion nor the flattened L12 rung. This does NOT refute the sparsity hypothesis (which predicted ~10×) but shifts weight toward the harder readings: the remaining headroom at this operator resolution may be genuinely small (the champion may sit near the noise-dominated practical ceiling), or improvements need qualitatively different data (e.g., off-self-play states), not just more of the same. Decision: **resume collection toward the full 5,000 games** — the 9× test is the one the hypothesis actually specified, and it remains the cheapest decisive experiment (~5 more days).
- **Champion unchanged: `az_v14d/best.pt`.**

---
---

## INFRA-051 — Curling Arena (standard head-to-head interface) — 2026-07-27

Built `arena/`: FastAPI app + web UI + agent JSON/text protocol for playing full
mixed-doubles matches against the champion on the authoritative stack (env_bridge
default-params physics, rules scoring, early-takeout legality, v2_fullsheet noise;
champion = WorldPlayer az_v14d, 48 candidates × 8 noise realizations — stronger
noise-averaging than the ×2 eval default, logged per match). Survey of prior
webapps: took mixed_doubles_game's interaction shape (canvas + live guide +
trajectory playback), threw away its stale stack (contact_mild physics,
SetTransformer value), rebuilt clean in csas_world. Input modalities: raw params,
draw-to-rest, contact-point+weight, move-hit-stone-to / takeout (path-bank NN init
+ real-board CEM; achieved error always reported). Draw solves ≈1–2 cm noiseless;
contact/tap honest at 0.2–0.5 m when guards physically constrain the shot.
**Semantics discovery (documented in AGENTS.md):** in the training convention,
stones driven through the back are NOT removed — `in_play` masks stones only off
the raw grid — so takeouts park victims behind the house ("spent" zone) and the
early-takeout forfeit fires only on literal off-grid removals. The arena keeps
these semantics exactly (match outcomes are byte-identical to the eval harness
transition). **[Superseded 2026-07-27: the user flipped the stack default to real
takeout rules — see the EXP-052 addendum. The historical convention is now opt-in via
WORLD_BOUNDARY_REMOVAL=0.]** Matches persist as JSON logs (intended vs realized actions, per-throw
champion evals). Power-play window: champion waits at an end's first throw while a
human/agent hammer side may still call its power play. `bash arena/run.sh 8020`.
---

## EXP-052 / az_v19 — NEW-RULES RETRAIN (boundary removal) — IN FLIGHT 2026-07-27

**Question.** The training convention never removes stones driven behind the house (the
raw-grid `in_play` mask kills only off-grid stones), so takeout victims park "spent" behind
the house and the early-takeout forfeit almost never fires (INFRA-051 discovery). What
happens if we flip to REAL-curling removal — stones die past the back line (center >
1.974 m) or on the side boards (|lateral| > 2.23 m) — making the no-takeout rule bind as
intended, and retrain the champion in that world?

**Mechanism.** `WORLD_BOUNDARY_REMOVAL=1` in `env_bridge` (post-processing on every
simulator transition; all csas_world sim consumers route through env_bridge — verified).
Unit-verified: through-the-back stones die; opponent removal at h10 → illegal + board
restored; same removal at h7 → legal; own-stone removal legal. Hogged/short stones
unchanged (out of scope).

**Design** (`scripts/_exp052_newrules_loop.sh`, config `exp_052_L8_newrules` = exp_048
corrected-loop recipe):
1. Collect ~600 games champion (az_v14d exported policy) self-play under NEW rules,
   sig-gated screen_tree (exp_037 operator), 6 workers × 16 games/round →
   `artifacts/replay/mcts/az_v19_newrules/`.
2. Fine-tune az_v14d (L8) on the corpus (shard 5 = val).
3. Eval h2h vs unadapted az_v14d: 3 draws UNDER NEW RULES (primary; dScore-primary,
   N=400×10 horizons×2 orders per draw) + 1 draw under OLD rules (regression check).

**Pre-registered outcomes.**
- Adapted > champion under new rules (ds > 2·SE): rules change is strategically material
  and the loop exploits it (expected mainly via legal-takeout discipline at h≥8 and
  changed hit-and-roll value: shooters that roll through the back now die).
- Null: az_v14d's play already transfers (plausible: removal mostly deletes stones that
  were strategically irrelevant when parked; the binding legality rule is rare in
  practice IF the champion rarely peels early).
- Adapted < champion under old rules (secondary): adaptation cost, quantifies rules gap.

**Known caveats (accepted, documented):** mix_sim 0.10 consistency buffers are old-rules
transitions (representation-level only); human value anchor (0.10) is real-curling data —
consistent with the NEW rules, arguably a better anchor here than in every prior run.
Corpus scale ~6k records is az_v9-iteration scale, below the EXP-050 3× corpus — a null at
this scale is weak evidence (same asymmetry as EXP-050); a positive is a positive.

**Addendum (2026-07-27, later).** User directive: the NEW rules are now the STANDARD.
- `WORLD_BOUNDARY_REMOVAL` default flipped to ON in env_bridge (opt-out `=0`). Every future
  collection/training/eval runs real takeout rules unless explicitly opted out. All certified
  numbers up to az_v14d / EXP-050 are OLD-rules numbers — never compare across the flip.
- The arena switched to the new rules immediately (az_v14d still the resident champion until
  EXP-052 resolves; both sides play the same environment, so matches stay fair).
- The in-flight chain script could not be edited (bash reads scripts lazily); its "old-rules"
  eval draw merely unset the env var, which now means NEW rules — so a placeholder
  `eval_out/az_v19_newrules/oldrules_run1/summary.json` makes the chain SKIP that draw, and the
  true old-rules regression draw must be run manually with `WORLD_BOUNDARY_REMOVAL=0` after
  training completes. Collection itself is unaffected (workers carry the flag explicitly).

**Status.** Collection launched 2026-07-27. Raw eval aggregates are auto-appended by the
chain script on completion; verdict + draw-level ±SE finalised by hand below.
---

## EXP-053 / depth certification — IS SEARCH DEPTH A LEVER? (Phase 0, no training) — IN FLIGHT 2026-07-27

**Motivation.** az_v7/EXP-029's "operator depth is not the lever" conclusion is unsound by
today's standards: 60 sims for a 3-ply tree (~1 visit at depth 3), value-head leaves (search
can't out-know its own V), no per-node noise robustness, no significance gating, no
operator-as-player certification, unconditional promotion. Depth was never retested with the
sound machinery (az_v12 screening, EXP-037 gating, incumbent gate). Post-EXP-050 (same-operator
data at 3x = null), a deeper operator is the leading "qualitatively different data" candidate.
EXP-052 collection PAUSED mid-round-1 for this (user directive; nothing lost, resumable).

**New machinery.**
- CRN (common random numbers) in `LocalNoise.sample_batch(crn=True)`: same underlying
  standard-t draws across candidates (per-candidate scale mapping preserved) -> paired
  candidate comparisons; verified spin/y0/speed offsets identical, angle correlated >0.999.
- `world/search/beam.py`: `screen_beam_choose` — depth-3 recursive screen-beam, minimax
  backup on the deterministic spine, value-free terminal-MC leaves, CRN screens
  (root 48-cand robust screen -> top 6; opponent 16x8 -> 3 most dangerous; our 12x8 reply;
  opponent re-picks under the deeper estimate). Illegal interior throws need no masking —
  forfeit semantics make them self-penalising. Plus `screen_tree_choose` = the exact d2
  collection operator as a callable.

**Design** (`scripts/exp053_depth_cert.py`, 4 shards, resumable JSONL): 40 val roots per
horizon h in {4,6,8,10} (preplaced at h10). Per ply, three choosers on identical states:
d2 (exp_037 knobs), d2p (COMPUTE-MATCHED control: stage-1 k_ego 48 + 128 tree sims ~= d3's
~2.9k rollout chains), d3 (beam). Disagreements (noise-normalised action distance dn > 2.5)
adjudicated by (a) paired terminal-MC k=64 CRN [secondary; biased toward behavior-policy
continuations] and (b) paired GUIDED PLAYOUTS T=8 [primary]: end played out from each action
with the deployed champion (az_v14d WorldPlayer, robust selection) moving BOTH sides, common
noise streams across branches. Runs under the NEW rules (stack default).

**Pre-registered.** PRIMARY: among playout-RESOLVED (|Δ|>2SE) d3-vs-d2p disagreements, d3's
win fraction; depth is certified a lever iff d3 > d2p at binomial p<0.05 AND the d3-vs-d2
comparison agrees in direction. If d3 only beats d2 but NOT compute-matched d2p, the gain is
budget, not depth (spend it on wider screens instead). Null -> depth dead at this
branching/noise level; the amortised latent-tree route becomes the only depth path.
Secondary: disagreement rates and per-horizon strata (expect depth to matter mid-end, h 4-8);
mean adjudicated Δ/end as the effect-size estimate for a Phase-1 retrain decision.

**Caveats.** Guided playouts use the OLD-rules-trained champion as the continuation policy
under NEW rules — symmetric across branches, so pairwise adjudication stays fair; absolute
Δ magnitudes carry that caveat. Root proposals come from the same exported champion policy
for all three operators (cancels).

**Standing directive (user, 2026-07-27):** if the pre-registered primary comes back
promising, proceed DIRECTLY to Phase 1 — d3-target collection (sig-gated, CRN) +
corrected-loop retrain + incumbent-relative gate — without further approval. All
experiments from now on run the REAL dead-stone logic (boundary_removal ON, the stack
default); selfplay and _eval_highN now print a RULES banner so every run log records
which convention was active.

**RESULTS (first look, 160/160 plies, 2026-07-28).** Wall-clock/ply: d2 61s, d2p 194s,
d3 121s — the "compute-matched" control actually got ~1.6x d3's budget (tree overhead is
serial), making the comparison conservative AGAINST d3. Operators disagree on ~99% of plies
at dn>2.5 (single-argmax choice under near-ties is noise-dominated — consistent with
EXP-037's ~8.5% true-significance rate). PRIMARY (playout-resolved d3 vs d2p): **16/25 =
64%, binom p=0.115** — direction positive but NOT at the p<0.05 bar; d3-vs-d2 agrees
(10/17, mean Δ +0.085/end). Adjudicator split matches the pre-registered bias prediction:
behavior-policy MC scores d3 at 43% while realistic strong-play playouts score it 64% —
d3 appears to buy adversarial robustness that behavior-rollout estimands can't see.
Resolution rate was only 25/158 at T=8 → the study is underpowered, not negative.

**EXTENSION (EXP-053b, pre-registered second look, launched 2026-07-28).** Searches are
stored; re-adjudicate every stored d3-vs-d2p disagreement with 8 additional paired guided
playouts (pooled T=16, exact two-group pooling, fresh disjoint seeds). Sequential-look
correction: certify depth iff pooled playout-resolved d3 wins at **binom p < 0.03** (and
d3-vs-d2 direction unchanged). If certified -> Phase 1 fires per the standing directive;
if not -> depth-via-sim is declared not-a-lever at this branching/noise level, the
latent-tree route inherits the depth question, and EXP-052 collection resumes.

**FINAL (EXP-053b pooled, T=16, 158/158 pairs, 2026-07-28): NOT certified.** Resolved
pairs 23 (down from 25 — some T=8 "resolved" pairs were regression-to-the-mean, exactly
what the extension guards against); d3 wins **16/23 = 69.6%, binom p = 0.0466** vs the
pre-registered corrected bar p<0.03. Mean adjudicated Δ **+0.056/end ± 0.056**. Per-horizon
resolved wins: h04 4/6, h06 1/3, h08 8/11, h10 3/3. Direction agreed everywhere with the
first look and with d3-vs-d2.

**Interpretation.** The pre-registered rule says depth-via-sim is NOT certified as a lever
at this branching/noise level. Honest caveats pointing the other way: p=0.0466 would pass a
naive one-look 0.05 bar; the point estimate is a ~70% decision-win rate; the control had
1.6x d3's wall-clock; and the effect concentrates late-end (h08+h10: 11/14). Verdict stands
(no third look — that would be p-hacking), but a Phase-1 depth retrain remains a live,
user-callable option AFTER the rules-change retrain resolves, ideally with budget-parity d3
and CRN-tightened screens. The latent-tree (dynamics-head) route formally inherits the
depth question. **Decision executed: EXP-052 collection resumed.** CRN itself is a keeper
regardless — it is free variance reduction for every future screen/gate.
---

## EXP-054 / depth x compute dose-response — QUEUED behind EXP-052 (pre-registered 2026-07-28)

**Question (user).** Does the depth-2 vs depth-3 comparison change at LARGER budgets for
both? EXP-053's single-point comparison had a 1.6x wall-clock mismatch (against d3) and a
99% noise-driven disagreement rate; its pooled verdict was a near-miss (16/23=69.6%,
p=0.0466 vs the corrected 0.03 bar).

**Design** (`scripts/exp054_depth_dose.py`; fresh val roots seed 54, h in {6,8,10}, 32/h =
96 plies; new rules; same exported-champion proposals everywhere). Four operators per ply:
d2_lo (exp_037: ke 8 / 48 sims), d3_lo (EXP-053 beam), d2_hi (ke 56 / 384 sims — CALIBRATED
up from ke 32/192 after the first 3 plies measured d2_hi at 191s vs d3_hi 363s, before any
adjudication outcomes were inspected; those 3 plies were discarded and the study restarted),
d3_hi (ke 12, beams 8/4, opp 20, mine 16; ~360s/ply measured at h6). Measured seconds
recorded per op for honest accounting. Noise-averaging paradigm everywhere
(k_ego noisy executions per intended throw at every screen, CRN-paired in d3; fresh-noise
visits in the d2 tree; T=16 paired noisy playouts in adjudication). Interior beam nodes
condition on the deterministic parent post (documented approximation).

**Adjudication.** Paired guided playouts T=16 (deployed champion both sides) on the two
DEPTH contrasts: d3_hi-vs-d2_hi and d3_lo-vs-d2_lo; paired MC k=64 on those plus the
within-depth dose contrasts (d2_hi-vs-d2_lo, d3_hi-vs-d3_lo).

**Pre-registered readouts (single look).**
1. PRIMARY: playout-resolved d3_hi vs d2_hi wins, binom p < 0.05 -> depth certified at high
   budget; Phase-1 depth retrain fires per the standing directive (post-EXP-052 champion).
2. Dose-response: hi beats lo within each depth (MC) — sanity that budget buys quality.
3. Interaction: depth gap (d3-d2) larger at hi than lo -> depth is the scaling lever;
   gap shrinking at hi -> budget-not-depth (spend compute on wider robust screens).
4. d3_lo-vs-d2_lo = EXP-053 replication on fresh roots (consistency check).

**Status.** RUNNING since 2026-07-28 ~10:00 UTC (user directive: EXP-052 paused again to
prioritize the depth comparison; its collection had banked 4 complete rounds = 384/600
games before the pause, resumable).

**RESULTS (96/96 plies, 2026-07-29): DEPTH IS NOT A LEVER — decisive null + failed replication.**
Wall-clock/ply: d2_lo 69s, d3_lo 140s, d2_hi 483s, d3_hi 378s (the hi control ended up 1.28x
d3_hi at h8/h10 — biased toward the control, i.e. conservative FOR the null).
- PRIMARY (playout-resolved d3_hi vs d2_hi, T=16): **3/6, p=0.66** — nothing. Resolution
  collapsed at high budget (6/95): when the well-fed operators disagree, the alternatives are
  near-equivalued.
- **EXP-053 REPLICATION FAILED**: d3_lo vs d2_lo on fresh roots = 5/12 resolved wins, mean
  Δ = −0.139/end (was 16/23 = 69.6%, +0.056 on the original roots). The EXP-053 near-signal
  (p=0.0466) was noise; the pre-registered corrected bar (p<0.03) correctly refused it.
- Dose-response sanity: hi > lo within each depth per MC (11/17 and 13/21, both ns, mean Δ
  +0.104 / +0.026) — budget buys a little decision quality; DEPTH does not (no interaction).

**VERDICT & decision executed.** Simulator-tree depth-3 is declared not-a-lever at this
branching/noise level, now on a proper dose-response with honest budget accounting. Phase-1
depth retrain does NOT fire. The latent-tree (dynamics-head) route formally inherits the
depth question. CRN remains adopted for all future screens/gates. **EXP-052 resumed**
(collection was at 384/600 games).
---

## EXP-055 / depth-null diagnosis — WHY is depth flat? (pre-registered 2026-07-29)

**Context.** EXP-054 closed depth as not-a-lever, but two mechanistic explanations remain
untested, and the pooled CI (|Δ| ≲ 0.1/end) still contains effect sizes that would matter.
User directive: run both diagnostics.

**055a — amortization hypothesis (--mode weak).** The EXP-053 primary contrast (d3 beam vs
compute-matched d2, ke48/128 sims) rerun with the HUMAN PRIOR as proposal/rollout policy
inside both operators (adjudication unchanged: champion playouts T=20). Pre-registered: if
the depth effect (mean adjudicated playout Δ) is clearly positive here while flat with the
champion policy, the null is explained — champion-distilled rollouts already amortize the
extra ply, and depth will never pay inside a loop that keeps strengthening the policy.

**055b — spine-bias hypothesis (--mode sb).** Stochastic-branching d3: ply 2 branches on 3
realized-post medoids (greedy farthest-point over k_ego noisy executions, weighted by
assignment counts) so the opponent replies to boards that actually happen; ply 3 stays on
the per-realization spine. ~2.7x d3_lo cost (~375s/ply, vs stored d2_hi's 483s — control
stays over-budgeted, conservative). Evaluated on EXP-054's EXACT seed-54 roots and
adjudicated against the STORED d2_hi actions (T=20 playouts, primary) and stored d3_hi
actions (paired MC, secondary — did the fix change/improve the decisions?). Pre-registered:
a clear positive mean Δ vs d2_hi (t>2), where deterministic-spine d3_hi was flat (+0.032 ±
0.036), certifies the spine as the artifact and the sb-operator as the Phase-1 candidate.

**Both flat ->** the depth question is closed for simulator trees: noise + strong amortized
policy genuinely shorten the useful lookahead horizon (backgammon-like), and only the
latent-space route remains. Primary readout for both modes: mean adjudicated Δ with
draw-level t-test (not just resolved-wins binomial — the fat-tailed resolution floor was
EXP-053/054's power problem).

**RESULTS (2026-07-30): BOTH HYPOTHESES REJECTED — depth question CLOSED with mechanism.**
- 055a weak-policy (128 plies, prior rollouts): playout meanΔ **−0.049 ± 0.050** (t=−0.98),
  MC −0.009 ± 0.040. No depth edge appears even with a weak policy — amortization by the
  champion policy is NOT what hides depth's value; there is no hidden value to amortize.
- 055b stochastic branching (96 plies, EXP-054's exact roots vs stored actions): d3_sb vs
  d2_hi playout meanΔ **−0.105 ± 0.062** (t=−1.68, leaning NEGATIVE), MC +0.002; d3_sb vs
  deterministic-spine d3_hi MC +0.033 ± 0.047 — fixing the spine does not unlock depth.
- Reading: both arms leaning ≤0 is consistent with minimax over noisy continuous-action Q
  estimates AMPLIFYING estimator noise (hard min/max over sampled replies) versus the
  soft, noise-integrating terminal rollouts of the 1-ply screen. In this execution-noise
  regime the game is backgammon-like: the reply distribution is what matters, and rollouts
  already integrate it; explicit adversarial sharpening adds bias, not information.

**CLOSURE.** Simulator-tree depth (2->3 ply, any budget tested, deterministic or stochastic
spine, strong or weak policy) does not improve decisions over the certified noise-robust
2-ply screen_tree operator. Five studies agree (053, 054, 055a, 055b + the EXP-053
replication failure). Keep: CRN screens; the operator of record stays exp_037 screen_tree.
The only remaining depth avenue is the latent-space (dynamics-head) tree, which changes the
cost model rather than the statistics, and should be motivated by dynamics-head fidelity
first. dScore-primary reporting throughout; champion unchanged (az_v14d).
---

## EXP-056 / rollout-estimator factorial (the Sage-3T question) — IN FLIGHT 2026-07-30

**Motivation (user).** Frontier backgammon engines (Open Sage 3T / XG Roller++, 7/2026
benchmark) don't win by deep root trees — 4-ply fixed search (0.41 PR) LOSES to sampled
truncated rollouts whose every internal decision is 3-ply, evaluated by the value net at a
~7-turn horizon (0.21 PR). Our EXP-053/054/055 arc tested (and rejected) exactly the losing
axis — root-tree depth. The winning axis — IN-ROLLOUT policy strength + truncation — was
never tested. Our operator of record = rollouts with a RAW 0-ply behavior policy to
terminal: their known-weak configuration of the winning family.

**Design** (`scripts/exp056_rollout_estimator.py`; h {6,8,10} x 32 val roots seed 56; all
factorial arms score the SAME dense candidate pool per ply — proposal variance excluded):
2x2 on the rollout estimator + reference:
  RT  raw->terminal, ke 8 (current stage-1)      | RtT raw->trunc@4 + champion-V leaf, ke 8
  ST  searched (EXP-014 value-greedy, n=6)->terminal, ke 4 | StT searched + trunc@4 + V, ke 4
  record = full exp_037 screen_tree (deployment reference).
Truncated leaf = new `max_steps`/`leaf_value_model` in `_mc_rollout_terminal_batch`
(V from root perspective at the frontier; az_v10-12's value-leaf objection is weaker now:
the champion V is matchup-calibrated and sits 4+ plies from the root). Searched arms halve
k_ego (variance-reduced rollouts; budget). Smoke wall-clocks/ply at h6: RT 22s, RtT 15s,
ST 46s, StT 33s, record 39s — primary contrast StT(33s) vs record(39s) is budget-fair.

**Adjudication & pre-registered readouts.** Paired guided playouts T=20 on PRIMARY
StT-vs-record (mean adjudicated Δ, t-test); paired raw-terminal MC k=64 CRN on the
factorial contrasts (each cell vs RT) + record-vs-RT (the tree stage's marginal value,
never separately measured). Predictions: (1) StT > record, gains concentrated h>=8 where
terminal-rollout variance is worst; (2) factorial shows which component pays (truncation vs
searched steps); (3) if StT wins -> it becomes the collection operator and its truncated
value estimates double as TD-flavored value targets -> Phase-1 retrain.

**RESULTS (96/96 plies, 2026-07-30) — FIRST POSITIVE OPERATOR SIGNAL, pending replication.**
Wall-clock/ply: RT 20s, RtT 12s, ST 63s, StT 32s, record 54s. PRIMARY (StT vs record,
playouts T=20): **meanΔ = +0.108 ± 0.063/end (t=1.72), resolved 7/9 (p=0.090), positive at
ALL horizons** (h6 +0.114 / h8 +0.075 / h10 +0.136) — and StT costs **60% of record**. MC
secondary −0.036 (the MC adjudicator is itself a raw-terminal estimator — biased toward the
record's estimand; playouts pre-registered primary for exactly this). Factorial (MC): ST
alone −0.089 (t=−1.80; searched-to-terminal at ke 4 is worse — truncation is what makes
searched steps affordable/effective), RtT alone +0.033 ns, record-vs-RT (tree stage's
marginal value, first measurement) +0.064 ns.

**EXP-056b (pre-registered fresh-roots replication, queued behind EXP-057).** t=1.72 is
EXP-053 déjà vu; discipline requires replication before promotion. Power analysis: between-
pair variance dominates (per-pair T=20 noise is a minority), so the extension adds PAIRS,
not playouts: +96 fresh val roots (h {6,8,10} x 32, seed 57), arms StT + record only,
T=20 playout adjudication. Verdict rule (two-look pooled, n≈192): promote StT to collection
operator iff pooled playout meanΔ t ≥ 2.1 AND the replication's standalone meanΔ > 0.
If promoted -> Phase-1: StT-collected corpus (its truncated leaf values double as
TD-flavored value targets), corrected-loop retrain, incumbent gate with k=8 confirmation
draw (per EXP-057 policy).

**EXP-056b RESULT (96/96 fresh plies, 2026-07-31): REPLICATION FAILED — StT NOT promoted.**
Replication alone: playout meanΔ **−0.034 ± 0.064/end** (t=−0.52; wave-1 was +0.108 ± 0.063).
Pooled two-wave (n=191 pairs): **+0.038 ± 0.045/end, t=0.83** — both pre-registered
promotion conditions fail. Operator of record remains exp_037 screen_tree.

Honest residuals (hypothesis-generating only, both post-hoc after a failed primary):
- StT runs at ~60% of record's cost with pooled strength a statistical tie — but the pooled
  lower 95% CI (−0.05/end) is exactly at the non-inferiority boundary, so a collection-
  operator swap for throughput would need its own certification (plus estimand-shift risk).
- h-strata consistent across BOTH waves: h10 positive twice (+0.136, +0.086; pooled +0.111
  ± 0.077), h6 the drag — truncation+V may pay only when terminal is far (the pre-registered
  prediction), suggesting a horizon-gated hybrid as a future candidate.

**Meta-lesson (now 3-for-3: EXP-053, EXP-056, and the EXP-054-lo sign flip):** in this
domain, operator signals at t≈1.7 on 96-192 adjudicated pairs DO NOT survive fresh-roots
replication. True per-decision operator differences, if any, are ≲0.05/end — at the floor of
affordable adjudication and small against training-side levers (az_v14d's training-side win
was +0.19/end at deployment settings per EXP-057). **Scoping correction (user, 2026-07-31):** this closes the DROP-IN inference-operator
question only. EXP-056 ranked a candidate pool distilled from screen_tree's preferences,
with a value head never calibrated on StT frontier states, adjudicated under incumbent-
champion continuations — the self-consistent question (train StT, infer StT) vs (train
screen, infer screen) is untested, and the ecosystem mismatch could have hurt OR helped.
The search-operator program is closed for zero-training swaps; the self-consistent StT
system is tested by EXP-058 below.
---

## EXP-057 / eval-protocol validation — is k=2 robust selection ranking-stable at k=8? (pre-registered 2026-07-30)

**Question (user).** Training targets average 8 noise realizations; the canonical eval
protocol selects with 2 (both players, symmetric — sound as a comparison); the arena deploys
8. Two caveats need closing: (1) k=2 selection noise dilutes skill gaps (promotions
conservative, nulls weaker); (2) rankings are certified at k=2 and could in principle
reorder at k=8 (the arena's deployed configuration was never itself certified).

**Design** (`scripts/_exp057_k8_stability.sh`, queued behind EXP-056's last shard). One k=8
draw per decided matchup, each at its OWN certified rules, changing ONLY k (new
`--sel-noise` override threaded through `_eval_parallel`/`_eval_highN`, printed as a
PROTOCOL OVERRIDE banner):
- M1 az_v14d vs az_v9 champion, OLD rules — certified k=2: ds +0.102 ± 0.005/end.
- M2 az_v19_newrules vs az_v14d, NEW rules — certified k=2: ds +0.001 ± 0.012/end (parity).

**Pre-registered.** Validation passes iff M1's k=8 dScore stays positive (>2·SE) — ranking
stable — and M2 stays within noise of parity or the SAME sign region (a small positive at
k=8 would sharpen the EXP-052 null's interpretation, not overturn a promotion). Expected per
dilution: |M1 k=8| >= |M1 k=2|. Standard going forward regardless of outcome: k=2 remains
the canonical gate protocol (cost + 50-experiment comparability); any future PROMOTION adds
one k=8 confirmation draw; the arena's k=8 deployment is recorded per match.
---

## EXP-058 / az_v20 — SELF-CONSISTENT StT ECOSYSTEM TEST — IN FLIGHT 2026-07-31

**Question (user).** After collecting StT targets and retraining self-consistently, is an
StT-trained system stronger than the screen_tree-trained control — i.e., (train StT, infer
deployed) vs (train screen, infer deployed)? EXP-056b only refuted StT as a zero-training
drop-in inside the incumbent ecosystem.

**Design.** The control arm ALREADY EXISTS: az_v19_newrules = az_v14d fine-tuned on 672
new-rules screen_tree-target games (exp_052 recipe). New arm az_v20_stt: identical in every
respect except the per-ply distillation-target operator = the StT estimator (searched
value-greedy steps n=6, k_ego 4 CRN, truncated@4 + incumbent-V leaf; new selfplay scorer
"stt", --value-world az_v14d). Same generator policy, same rules, same sig-gating (t>=2,
tie-break pooling), same target size (~600 games), same trainer recipe/init. Value targets
stay realized returns (one change at a time; the truncated-estimate-as-value-target variant
is a follow-up arm if this one moves).

**Gate (pre-registered; SHORTENED per user for an early conclusion).** Primary: az_v20 vs
az_v19 (the matched train-side contrast), ONE draw at k=4 selection, N=250 x 10 horizons x 2
orders = 5,000 ends (~2h; SE(ds) ≈ 0.020, and k=4 halves the k=2 selection dilution per
EXP-057, so per-end discrimination is sharper). The full canonical battery (3 draws k=2 +
k=8 confirmation + vs-incumbent) runs only if the early draw is promising (|ds| > 2·SE). Outcomes: az_v20 > az_v19
=> the StT estimand teaches something screen_tree doesn't (ecosystem effect real; iterate);
parity => one-generation StT self-consistency adds nothing beyond the drop-in null (bounds
ONE generation only — the policy only partially reshapes toward StT support in one cycle,
pre-registered caveat); az_v20 < az_v19 => StT targets are WORSE to learn from at this
noise level. Cost: ~16-20h collection (StT is ~60% of screen_tree's per-ply cost), ~3h
train, ~10h evals.
---

## EXP-059 / az_v21 — 2x DATA SCALE FOR THE StT TEACHER — IN FLIGHT 2026-08-01

**Question (user).** Is StT-as-teacher truly useless, or was the 1x parity signal
starvation? Doubling the StT corpus does double duty: (a) measures the StT family's
data-scale response, and (b) at ~1,250 games the StT corpus's significant-ply count
(~650 at its 5.2% rate) finally MATCHES the az_v19 control's (~570 at 8.5%), removing the
fewer-teaching-signals confound flagged in EXP-058.

**Design** (`scripts/_exp059_stt2x_loop.sh`). Continue the az_v20_stt corpus to ~1,250
games (3 workers — the certified GPU ceiling for value-model-resident StT workers, with
shards-on-disk verification per round); retrain az_v21_stt2x from az_v14d with the identical
recipe; shortened k=4 N=250 gates (per the EXP-058 amendment): PRIMARY vs az_v19 control,
SECONDARY vs az_v20_stt (within-family scale response).

**Pre-registered outcomes.** (1) az_v21 ≈ az_v19 AND ≈ az_v20: StT flat in data — the
teacher is genuinely inert at this resolution; closure stands reinforced. (2) az_v21 >
az_v19 by >2·SE: StT responds to scale — conditional follow-up REQUIRED before any
promotion claim: a 2x screen-target control (~600 more screen games) to separate
estimand-vs-data effects. (3) az_v21 > az_v20 but ≈ az_v19: pure data-scale effect, teacher
identity irrelevant — redirects effort to the generic 9x data test. Cost: ~20h collection +
~3h train + ~4h gates.
---

## EXP-060 — FIXED-POINT DIAGNOSTIC (is there anything left to distill?) — IN FLIGHT 2026-08-03

Per stored az_v19 collection record (x0, c0, teacher's top-weighted dist action), compute the
champion's own deployed selection (WorldPlayer k=8) twice with independent seeds. Report
teacher-vs-student material-disagreement rate (dn>2.5) MINUS the student-vs-student baseline
(stochastic selection among near-ties), stratified sig-gated vs non-sig and by horizon.
Pre-registered reading: excess disagreement on sig plies <~15-20% => policy is at its
teacher's fixed point; the 9x same-operator fine-tune is predictably parity. >~40% => real
distillation headroom; 9x justified. All ~570 sig plies + 400 sampled non-sig; ~1h, 4 shards.

**Design note for the diverse-openings program (user, 2026-08-03):** NO jittered
pre-placements. Mechanism = HIGH-TEMPERATURE PREFIXES (2-4 burn-in throws from canonical
roots by a hot policy / human prior / arena-match prefixes, under execution noise — states
sound by construction, then normal collection at reduced horizon) + COVERAGE CONTROL
(stratify by state descriptor: stones in play, house occupancy, guard count, canonicalized
symmetry; sample prefixes to fill under-covered buckets).

**VERDICT (906 plies, 2026-08-03): AT THE FIXED POINT — excess disagreement ≈ 0.**
On sig-gated plies: teacher-vs-student material disagreement 98.6%, student-vs-student
baseline 99.2% -> **EXCESS −0.6%** (non-sig: +0.5%; every horizon within ±3%). The
teacher's choice is statistically indistinguishable from another draw of the student
itself. Pre-registered reading: the policy IS its teacher's fixed point -> **the 9x
same-operator fine-tune is predicted parity with high confidence; do not spend the week.**

**Bonus discovery (explains EXP-053-059 coherently):** deployed selection SELF-disagrees
at the action level ~99% of the time (dn>2.5 across independent draws) — the decision
landscape at collection states is a PLATEAU of near-equivalued shots, and every chooser
(teacher, student, StT, d3...) is a lottery over the same plateau. That is why all operator
pairs "disagreed" on ~99% of plies while all adjudicated value gaps were <=0.05/end: the
action-level comparisons were lotteries; plateau HEIGHT is the only real quantity, and
everything sits on the same plateau on-distribution.

**Implication + next diagnostic.** Distillation gradients vanish on-distribution at any
data scale (targets ~ the policy's own draw distribution). The live question for the
diverse-openings lever: does teachable headroom exist OFF-distribution? Follow-up
(headroom map): generate hot-prefix states (the (ii) mechanism), and measure the VALUE GAP
E[Q(teacher choice) − Q(student choice)] by paired MC per state — the saturated
disagreement metric is uninformative there; the value gap is the distillable signal. If
the gap is ~0 there too, the loop is closed at this noise level and (iv) full-training /
value-channel (iii) are the only remaining levers.
---

## EXP-061 — HEADROOM MAP RESULT: THE TEACHER IS BEHIND THE STUDENT — 2026-08-03

Δ = Q(teacher screen_tree choice) − Q(student deployed-selection choice), paired terminal-MC
k=64 CRN, 480 states:
- CONTROL (on-distribution sig plies): **Δ = −0.266 ± 0.056/end (t=−4.7)**
- HOT-PREFIX (off-distribution):       **Δ = −0.126 ± 0.027/end (t=−4.7)** (negative at every
  prefix length; most negative in guard-heavy strata: guard=2/3 buckets −0.17..−0.31)

**Reading.** The deployed selection (policy proposals + value-head ranking + k=8 noise
robustness) now picks BETTER shots than the collection operator that trains it. Mechanism:
the teacher argmaxes ~200 candidates on 8-sample terminal-MC estimates (SE ~±0.25) — a
WINNER'S-CURSE selection whose chosen action regresses under the 64-sample re-measure —
while the student ranks with a low-variance value head and carries no such curse. On the
decision plateau (EXP-060), the low-variance ranker wins. Caveat: the adjudicator shares
the teacher's estimand (raw-policy terminal MC), which if anything should FAVOR the
teacher — making the negative sign conservative. Part of the magnitude is measurement
mechanics (curse regression), but the curse applies to the actual TARGETS too: collection
distills noise-selected actions.

**Consequences.**
1. The distillation loop as constituted is not just saturated (EXP-060) — its improvement
   operator is INVERTED. Sig-gating + replay mixes + guards explain why recent fine-tunes
   were parity rather than negative: they limited the damage.
2. The diverse-openings bet dies FOR THIS TEACHER: off-distribution Δ is also negative.
   It may return with a stronger teacher.
3. Teacher quality is worst exactly in guard-heavy/congested states — compounding the
   defensive-teaching starvation (EXP-060 horizon table).
4. NEW LEVER exposed — **big-budget self-distillation**: the strongest affordable operator
   is now the student's OWN decision rule at a larger budget (more candidates x more noise
   samples, value-head ranked — EXP-057 measured k=8 >> k=2 at play). A teacher =
   deployed-selection@(96-192 cands x k=32-64) has no MC winner's curse (shared value-head
   bias cancels teacher/student) and distills the measured k-scaling gain into 48x8 play.
   This is the AZ loop with the search replaced by the certified robust-selection operator.

**Falsifiable next step (proposed):** verify a big-budget teacher actually beats the
student per-decision (same paired-MC + playout harness, teacher = k=64/192-cand selection);
if yes, one collection+retrain cycle with those targets.
---

## EXP-062 — BIG-BUDGET SELF-DISTILLATION TEACHER VERIFICATION — IN FLIGHT 2026-08-04

**Question.** EXP-061 inverted the loop: the MC-argmax teacher is BEHIND the student
(winner's curse). Candidate replacement teacher = the student's OWN deployed decision rule
at a larger budget (value-head ranked, no MC curse; shared value bias cancels in the
contrast). EXP-057 already showed budget scaling of this rule buys real play strength
(k=2 -> k=8). Verify per-decision superiority BEFORE any collection cycle.

**Design** (`scripts/exp062_bigbudget_teacher.py`; EXP-061's state distribution: 80
on-distribution sig plies + 160 hot-prefix states). Per state: S = 48x8 deployed selection;
S' = independent second draw (null anchor, Δ ≈ 0 expected); T1 = 96x32 (~8x); T2 = 192x64
(~32x). Δ(X−S) by paired terminal-MC k=64 CRN (fair here: neither side optimizes the MC
estimand); guided playouts T=16 on every 4th state for T2−S confirmation.

**Pre-registered.** Teacher certified iff Δ(T1−S) > 0 and Δ(T2−S) >= Δ(T1−S) (dose-response)
with the playout confirm agreeing in sign, and Δ(S'−S) ≈ 0 (anchor sanity). If certified ->
one collection+retrain cycle with big-budget-selection targets (sig-gating via paired value
margins), the first loop restart supported by every diagnostic.

**VERDICT (240 states, 2026-08-04): CERTIFIED.** Monotone dose-response with a clean null
anchor: Δ(S'−S) = +0.019 ± 0.025 (anchor ≈ 0 ✓); Δ(T1−S) = +0.035 ± 0.026 (> 0 ✓);
**Δ(T2−S) = +0.062 ± 0.027 (t=+2.28)** (>= T1 ✓); playout confirm Δ(T2−S) = **+0.110 ±
0.068** (sign agrees ✓). Unlike the EXP-053/056 near-signals, this one has structure
(3-level monotone ladder + null anchor), an independent game-level confirmation (EXP-057:
budget scaling of this exact rule, >6σ at n=2,790 ends), and a understood mechanism
(variance reduction of the argmax over the plateau; no winner's curse — value bias shared).
Anchor-corrected per-decision edge ≈ +0.04-0.09/end. Proceeding to the pre-registered
cycle: EXP-063 below.
---

## EXP-063 / az_v22 — BIG-BUDGET SELF-DISTILLATION CYCLE — IN FLIGHT 2026-08-04

**The pre-registered consequence of EXP-062's certification.** Teacher = the certified T2
operator (192 policy proposals x k=64 CRN noisy executions, value-head ranked — the
deployed selection at ~32x budget; per-decision edge +0.06/end MC, +0.11 playout-confirmed).
New selfplay scorer `bigsel` replicates the deployed `_decision_values` semantics exactly
(proposal temp 1.1/std 1.2, mean -V(post) over noise realizations, illegal-if-any masking),
distill targets = soft-topk over the robust values, sig-gate = top-1-vs-top-2 t>=2 on the
value-sample SEs (CRN-conservative). Smoke: 1 game -> 10 records ok.

**Plan** (`scripts/_exp063_bigsel_loop.sh`): ~400 games (3 workers — value-eval-heavy),
target dir az_v22_bigsel; fine-tune az_v14d with the exp_052 recipe; SHORTENED GATE (per
the standing amendment): one k=4 N=250 draw **vs the incumbent az_v14d itself** — the real
question is finally "does the loop beat the champion again". Full 3-draw battery + k=8
confirmation only if |ds| > 2·SE. Expected timeline: ~1.5-2 days collection + 3h train +
2h gate.

**Why this can work where everything since az_v14d failed:** the teacher is (a) certified
STRONGER than the student per-decision (unlike screen_tree, which EXP-061 showed is
weaker), (b) curse-free (value-ranked, not MC-argmax), (c) cheap enough per-ply to gate
generously, and (d) its gains concentrate exactly where EXP-057 showed budget matters.
Risk (pre-registered): self-distillation of a value-ranked teacher can only teach the
policy to propose what the value head already prefers — if the value head is the binding
constraint, the cycle returns parity and the (iii) value-rank lever becomes the priority.
---

## EXP-064 / az_v23 — DECISION-RELEVANT VALUE TRAINING (rank loss) — IN FLIGHT 2026-08-05

**The (iii) lever, now the priority** (three independent results point at the value head as
the binding constraint: EXP-062, the az_v15-VH test, EXP-063's negative). Deployed
selection uses V only to RANK candidate posts; we now train that directly.

**Machinery.** New rank fields in the record schema (zero-filled on legacy shards —
backward compatible); `scripts/backfill_rank_posts.py` simulates RANK_R=4 noisy executions
of the stored top-1/top-2 target actions per sig-gated ply (h>=2) and stores the
post-states + next_cond (az_v19 corpus: 294 train / 72 val pairs); margin rank loss in
losses.py: relu(margin − [Q(top1)−Q(top2)]) with Q(a) = −V(post,next_cond) mean over the R
posts — the exact deployed sign convention. loss.value_rank=0.3, margin 0.25.

**Smoke finding (baseline):** az_v14d's value head orders only **70.4%** of the gated
top-1/top-2 pairs correctly — a third of the teacher's confident comparisons are misranked
by the head that deployed selection relies on. That is the measured headroom.

**A/B (perfectly matched).** az_v23_rank = az_v14d + exp_052 recipe + rank loss on the
rank-backfilled az_v19 corpus vs the EXISTING az_v19_newrules control (same corpus, same
recipe, no rank loss). Shortened gate: one k=4 N=250 draw vs the control; a second vs the
incumbent az_v14d if the first is promising. Readouts: gate dScore (primary), val_rank_acc
trajectory (does the loss actually fix the orderings), val_value_mse_mcts guard (does rank
training damage calibration).
---

## EXP-065 / az_v25 — EXPLOITABILITY PROBE (asymmetric best response to az_v14d) — IN FLIGHT 2026-08-05

**Question (user).** Was SELF-play itself the limitation? EXP-042's meta-game matrix was
perfectly transitive (Nash = pure az_v14d), which under transitivity makes PSRO collapse to
our incumbent ratchet — but that matrix covered ~5 one-lineage siblings, and we have NEVER
trained an asymmetric best response TO the champion (every corpus ever was symmetric
self-play). This is the last cheap experiment that could overturn the ceiling conclusion.

**Design.** New selfplay BR mode (--opponent-world): the fixed champion plays one block per
game with its deployed 48x8 selection; the LEARNER's plies get bigsel targets at
EXPLORATORY proposal temperature (1.35/1.6 vs deployed 1.1/1.2 — the single-cycle
asymmetry: exploring responses deployed play would not try); value targets = realized
MATCHUP returns (value-against-az_v14d specifically, the BR value function). Opponent plies
carry conf=0 (no distillation) but contribute matchup value/unroll signal. ~400 games,
3 workers; fine-tune az_v14d (exp_052 recipe); shortened gate: one k=4 N=250 draw vs
az_v14d.

**Pre-registered outcomes.** (1) BR ≈ parity: az_v14d is near-unexploitable within our
model class — the transitivity verdict holds at the level that matters, PSRO is
conclusively unnecessary, and the noise-ceiling reading gets its strongest evidence.
(2) BR wins > 2·SE: self-play left exploitability on the table; the matrix was
population-limited; a population/BR loop becomes the live path (a working BR operator IS
the "teacher that stays ahead"). Either outcome is decisive. Caveat: one BR generation
from a champion warm start bounds one iteration of exploitation, not the BR-loop limit.
---

## EXP-066 — SEARCH-VALIDATION BENCHMARK (simple-regret scaling curve) — QUEUED 2026-08-05

**Question (user's verdict, accepted).** EXP-053-055/061 refuted OUR budget-constrained,
statically-branched, chance-node-free trees — not correctly-implemented stochastic search.
Code audit CONFIRMS the structural defect: kr_uct_tree commits ONE fresh noise realization
per visit (no explicit chance/afterstate nodes; min/max mixes with execution randomness;
~2-6 effective samples per candidate at historical budgets). Before any search conclusions
stand, an instrumented benchmark must show whether a correctly-structured tree's SIMPLE
REGRET decreases monotonically with budget.

**Design** (`scripts/exp066_search_validation.py` + `src/world/search/hybrid_tree.py`).
60 h=2 states (40 tactical: >=4 live or >=2 in-house, hot-prefix generated; 20 control),
one SHARED 128-candidate pool per state (96 policy + 32 structured; all arms choose from
the pool). REFERENCE: two-stage chance-correct expectimax (A: 128 cands x 32 root-CRN x
[48 opp x 8]; B: top-16 at 128 x [64 x 16]) — GPU-JAX (the new 29k throws/s unlock makes
this ~1-2 min/state). ARMS at budgets {1k, 4k, 16k, 64k} simulator calls:
flat_width (bigsel family), screen_tree (budget-scaled operator of record), and
hybrid_tree — NEW module implementing the full spec: explicit decision/chance nodes
(min/max only at decisions), double progressive widening (actions AND outcomes),
min-evidence gate (>=8 effective samples on serious candidates before opening more),
policy-prior PUCT bonus over kernel-regressed stats (bandwidth in execution-noise units),
mixed V+periodic-rollout leaves, anytime budget checkpoints from one 64k run.

**Pre-registered readouts.** (1) Monotone regret decrease per arm = implementation sound;
non-monotone = fix search before concluding anything about the game. (2) If all arms'
64k regret ~ reference noise floor (esp. on TACTICAL strata) => the plateau/ceiling
conclusion stands at the level of the game, closing the search question permanently.
(3) hybrid_tree regret << flat_width at equal budget on tactical states => a correctly
funded stochastic tree DOES add value; its targets become candidates for training
(gated: only after monotonicity holds). Chained behind EXP-065 (`_exp066_chain.sh`).


## Template for a new entry

```markdown
## EXP-00N — <short name>

- **Checkpoint:** <path>
- **Config:** <yaml> ; driver <script>
- **Goal:** <one line>
- **What changed vs <baseline EXP-id>:** <bulleted deltas only>
- **Training (final):** <key train + val metrics>
- **Head-to-head vs <reference>:** <table: horizon | win rate | Δscore ; + avg>
- **Verdict:** <beat / parity / regressed + why>
- **Run cost:** <wall-clock, bottleneck>
- **Next:** <follow-ups>
```

Also add a one-line row to the [Summary table](#summary-table).

**EXP-052 raw eval aggregates (auto-appended by _exp052_newrules_loop.sh):**

- newrules_run1: winrate 0.5048 ± 0.0095, dScore +0.0136 ± 0.0329 /end (n=2790 ends, 40 shard-cells)
- newrules_run2: winrate 0.5039 ± 0.0095, dScore +0.0125 ± 0.0350 /end (n=2790 ends, 40 shard-cells)
- newrules_run3: winrate 0.4887 ± 0.0095, dScore -0.0219 ± 0.0336 /end (n=2790 ends, 40 shard-cells)
- oldrules_run1: (no results)

**FINAL VERDICT (2026-07-29): PARITY — az_v14d's play TRANSFERS to the real takeout rules.**
Pooled over the 3 pre-registered draws (draw-level): dScore **+0.001 ± 0.012/end**, winrate
0.499 ± 0.005 (n=8,370 ends). The corrected-loop fine-tune on 672 new-rules self-play games
(6,720 records; train early-stopped at epoch 13, best epoch 9) does NOT improve on the
unadapted champion under the new rules. Interpretation (pre-registered outcome 2): boundary
removal mostly deletes stones that were strategically irrelevant when parked, and the
champion's style rarely relied on early peels that now forfeit — the rules change is real
but az_v14d already plays it near-optimally relative to what this corpus can teach.
**Champion: az_v14d/best.pt UNCHANGED, now certified at parity under the new-rules standard.**
az_v19_newrules/best.pt retained as the new-rules-native alternative (not promoted).
The manual OLD-rules regression draw (WORLD_BOUNDARY_REMOVAL=0, adaptation-cost footnote)
is running; its aggregate will be appended below.

**OLD-rules regression draw (manual, WORLD_BOUNDARY_REMOVAL=0):** winrate 0.4935, dScore +0.0072 ± 0.0284/end (n=2790) — no adaptation cost detectable; the new-rules fine-tune did not damage old-rules play.

**EXP-057 k=8 aggregates (auto-appended):**

- m1_v14d_v9_old: k=8 winrate 0.5256, dScore +0.1910 ± 0.0293/end (n=2790)  [certified k=2: +0.102 ± 0.005 (k=2, 3 draws)]
- m2_v19_v14d_new: k=8 winrate 0.4989, dScore -0.0197 ± 0.0272/end (n=2790)  [certified k=2: +0.001 ± 0.012 (k=2, 3 draws)]

**VERDICT (2026-07-31): PROTOCOL VALIDATED.** M1's ranking holds at k=8 with the edge
nearly DOUBLED (+0.102 -> +0.191/end, >6σ positive) — the dilution prediction confirmed:
k=2 gate numbers are conservative understatements of deployed strength. M2 stays at parity
(−0.020 ± 0.027) — the EXP-052 null is robust across protocols, not a dilution artifact.
Consequences: (1) both pre-registered pass conditions met; the canonical k=2 gate protocol
and every historical ranking stand; (2) the arena's k=8 deployment of az_v14d is now itself
certified (its true edge over the previous champion at deployment settings is ~+0.19/end);
(3) promotions henceforth include one k=8 confirmation draw (policy adopted above).

**EXP-058 raw eval aggregates (auto-appended):**

- vsctrl_k4: winrate 0.4968, dScore -0.0258 ± 0.0383/end (n=2516)

**INCIDENT + rerun (2026-08-01).** The first train/gate above is INVALID by our own
standards: collection workers 3-4 OOM'd every round (the 3-per-GPU StT-worker ceiling —
each carries the value-world model — plus the resident arena), so the val split (keyed to
shard 4) was EMPTY, training ran without validation ("best at epoch none": no early-stop
selection, no value-drift guard — the az_v13 misaligned-selection failure mode), and the
gate evaluated the raw last epoch. The 624-game corpus itself is intact (39 shards, 5.2%
mean sig rate — note: LOWER than screen_tree's ~8.5%, the StT estimator's CRN-marginal SEs
gate more conservatively). Fixed: proper split rebuilt (34 train / 5 val shards), retrain
with the full exp_052 selection machinery, fresh k=4 gate draw (vsctrl_k4_v2). Corrected
result to be appended below.

**CORRECTED RESULT + VERDICT (2026-08-01): PARITY — one-generation self-consistent StT
adds nothing detectable.** With valid selection (best @ epoch 10, guard active):
az_v20_stt vs az_v19 control at k=4: winrate 0.5050, dScore **+0.018 ± 0.034/end**
(n=2,516). |ds| << the shortened gate's 2·SE bar -> the full battery does not fire.
Combined with EXP-056b: the StT estimator is now null BOTH as a drop-in chooser AND after
one generation of self-consistent retraining on its own targets.

Standing caveats (pre-registered): one-generation bound only (proposal support reshapes
gradually); the StT corpus carried ~5.2% sig plies vs the control's ~8.5% (CRN-marginal
gating is more conservative — fewer teaching signals at equal games); the shortened gate
resolves only |ds| >~ 0.07. A multi-generation StT loop or a sig-rate-equalised corpus
could still differ, but nothing in five experiments (053-058) suggests the effect size
would justify the cost.

**PROGRAM CLOSURE (search/target operators, EXP-053 -> EXP-058).** Neither deeper trees,
nor budget, nor the Sage-3T rollout estimator — as chooser or as teacher — moves this
system at its current data scale. The certified stack stands: exp_037 screen_tree targets,
1-ply-robust deployed selection (k=2 gate / k=8 deploy), az_v14d champion. Remaining
strength levers, in order of evidence: (1) DATA SCALE (the interrupted 9x significant-ply
test — az_v17 old-rules 1,904 games banked; new-rules az_v19 672 + az_v20 624 corpora
reusable for mixed-target experiments); (2) architecture at larger data; (3) latent-space
search, gated on dynamics-head fidelity.

**EXP-059 raw eval aggregates (auto-appended):**

- vs19ctrl_k4: winrate 0.5004, dScore +0.0322 ± 0.0364/end (n=2516)
- vs20stt1x_k4: winrate 0.5127, dScore +0.0060 ± 0.0412/end (n=2516)

**VERDICT (2026-08-02): OUTCOME 1 — StT-as-teacher is flat in data scale; inert confirmed.**
The 2x corpus (1,296 games / 12,960 records / 697 sig plies at 5.4% — now EXCEEDING the
control's ~570 sig plies, confound removed) trained cleanly (best @ epoch 12, guard active),
and az_v21 sits at parity with BOTH the az_v19 screen control (+0.032 ± 0.036/end) and the
1x StT arm (+0.006 ± 0.041/end). No scale response within the StT family, no catch-up
against the screen control once teaching signals are matched. Per the pre-registered map:
the teacher identity is irrelevant at this resolution — the EXP-053→058 closure stands
REINFORCED, and the data-scale question decouples from the operator question entirely.
The generic data lever (the 9x significant-ply test) is now the sole ranked-first open
direction; the combined new-rules corpora (az_v19 672 + az_v20/21 1,296 games ≈ 2,000
games, ~1,270 sig plies across both estimands) are banked toward it.

**EXP-063 raw eval aggregates (auto-appended):**

- vsinc_k4: winrate 0.4942, dScore -0.0684 ± 0.0354/end (n=2516)

**VERDICT (2026-08-05): NOT PROMISING — leaning NEGATIVE (−0.068 ± 0.035, t≈−1.9, k=4
single draw).** No full battery, no promotion; az_v14d unchanged. Chain valid (432 games,
15.3% sig rate = ~2x historical as predicted; best @ epoch 16 with guards; one accounting
note: the 1/3 val split left ~440 TRAIN sig plies — comparable to, not double, the
controls').

**Reading.** The pre-registered risk materialized, plus a sharper mechanism the negative
lean suggests: distilling soft-topk of VALUE-RANKED candidates narrows the policy toward
the value head's preferred modes — but deployed strength = proposal DIVERSITY x value
discrimination. Sharpening proposals toward V's argmax makes the 48 play-time candidates
redundant, shrinking the effective search width that robust selection feeds on. Self-
distillation eats the exploration that made selection strong. (Testable: compare candidate
diversity / effective sample size of az_v22 vs az_v14d proposals.) Caveat per our own
meta-lesson: a single ~2σ read shouldn't be over-interpreted — but direction alone kills
promotion.

**Where this leaves the program.** The POLICY-distillation channel is now closed from every
direction: weak teacher (screen_tree — actually behind the student, EXP-061), alternative
estimand (StT, 056/058/059), off-distribution (061), and now the certified-stronger
self-teacher (one generation). Three independent results point at the VALUE head as the
binding constraint: EXP-062 (all decision-quality gains live in value ranking at big k),
the az_v15-VH test (val-MSE improvement ≠ play), and this cycle. Priority lever is now
**(iii) decision-relevant value training** (rank loss on gated top-1/top-2 pairs — cheap:
backfill posts on existing corpora incl. this one, add margin loss, A/B with the shortened
gate), with (iv) full capacity x data retraining as the big-swing alternative.

**EXP-064 raw (auto):** az_v23_rank vs az_v19 control k=4: winrate 0.4984, dScore -0.0191 ± 0.0371/end (n=2516)

**EXP-064 POST-MORTEM + 064b (2026-08-05).** The A/B tested nothing: val rank_acc fell
monotonically 69.6% -> 56.5% over epochs 10-13 while val_value_mse stayed flat — **294
training pairs overfit within one epoch**; the (correct) early-stop selected epoch 9 ≈ the
unchanged baseline, so the parity gate compared baseline-to-baseline. Concept untested;
pair count was the failure.

**EXP-064b (relaunched, matched at 2.2x pairs + pair augmentation):** backfilled the
az_v20/21 StT corpus (+362/46 pairs; merged 656 train / 118 val), and extended the
flip/team-swap batch augmentation to the rank fields (states flip, rank_cond block swaps —
consistent transforms, ~4x effective pair diversity). Because the union corpus changes ALL
losses' data, the control is retrained too: az_v24b_ctrl = same union + recipe, NO rank
loss. Gate: one k=4 N=250 draw, rank-vs-ctrl. Watch val_rank_acc trajectory for the same
overfit signature; if it recurs at 656 pairs, the honest conclusion is that rank
generalization needs pair counts only a dedicated big collection can provide.

**EXP-064b raw (auto):** rank vs ctrl k=4: winrate 0.5095, dScore +0.0342 ± 0.0284/end (n=2516)

**VERDICT (2026-08-05): overfit FIXED, generalization NOT achieved, gate not promising.**
At 656 pairs + rank-aware augmentation the val rank_acc no longer collapses (held ~65-72%
across epochs vs 064's monotone fall to 56%) — but it also never IMPROVED past the ~70%
baseline, and val rank loss stayed flat (~0.20). The gate leaned positive (+0.034 ± 0.028,
t=+1.2, winrate 0.510) but is below the 2·SE bar and squarely in known noise territory.
Per pre-registration: no escalation.

**Where (iii) stands.** The 30% misranking headroom measurement is solid and unchanged; the
margin-rank objective at O(10^2-10^3) pairs stabilizes but does not teach a better ordering
function. Closing the headroom needs O(10^4+) certified pairs, and MC-grounded pair
generation costs ~1-2 min/pair — a dedicated multi-day collection. That is the price tag
for the value channel; park it as costed-but-unfunded. The remaining untested lever is
(iv) full capacity x data retraining. Champion: az_v14d, unchanged since 2026-07-12.

**EXP-065 raw eval aggregates (auto-appended; header said EXP-063 — driver-clone artifact):**

- vsinc_k4 (v1): winrate 0.4988, dScore -0.0207 ± 0.0288/end (n=2516)

**v1 INVALID by the EXP-058 standard (2026-08-06).** "best at epoch none": the value guard
(3.60, calibrated for self-play corpora) is unreachable on a BR corpus — exploratory-temp
learner play inflates outcome variance, so val_value_mse_mcts starts ~4.9 at epoch 9 and
never passes; the gate evaluated unselected epoch-13 weights. Corpus itself is fine
(432 games, 430 sig plies at 10.0%). Retraining with guard 5.5 (fits the corpus's
variance floor), re-gating; corrected verdict below.

**v2 gate (valid ckpt, best @ epoch 11, guard 5.5): winrate 0.5022, dScore +0.0604 ±
0.0368/end (t≈1.64, n=2,516).** The largest positive lean any challenger has shown vs
az_v14d — but below the pre-registered 2·SE bar and in the t≈1.6-1.7 replication-death zone
(EXP-053/056/EXP-054-lo). PRE-REGISTERED BEFORE DRAW 2 (2026-08-06): run one more identical
k=4 draw (vsinc_k4_v2c); certify EXPLOITABILITY FOUND iff pooled two-draw dScore t >= 2.1;
else the verdict is 'no certified exploitability' with the lean recorded. Note v1
(unselected ckpt) read −0.021 — checkpoint selection moved the readout by +0.08, itself
evidence the guard fix mattered.

**VERDICT (2026-08-06): EXPLOITABILITY CERTIFIED.** Draw 2: +0.1157 ± 0.0302. POOLED
two-draw: **dScore +0.0788 ± 0.0238/end, t=+3.31** (n=5,032 ends) — far above the
pre-registered 2.1 bar. The first certified positive challenger since az_v14d's creation,
after ~12 failed symmetric attempts — and it took only 432 asymmetric BR games
(exploratory bigsel targets + MATCHUP value returns vs the fixed champion).

**What this overturns and what it doesn't.** (1) az_v14d IS exploitable within our own
model class — symmetric self-play left real learnable surface on the table; the user's
PSRO instinct is vindicated, and EXP-042's transitivity conclusion was population-limited
exactly as suspected. (2) The noise-ceiling reading WEAKENS: there is learnable structure;
the symmetric loop just couldn't see it (its opponent moved with the learner; the BR's
fixed opponent + matchup-grounded values could). (3) NOT yet established: whether az_v25_br
is generally stronger or an anti-v14d specialist (classic exploiter trade-off), and whether
BR iteration compounds (league ratchet). Follow-up gates launched: az_v25_br vs
az_v19_newrules k=4 (generality) + vs az_v14d k=8 (EXP-057 confirmation-draw policy).
Promotion decision deferred to the meta-game view (champion should approximate the
population Nash, not the last exploiter).

**FOLLOW-UP GATES (2026-08-06) — BOTH CERTIFY; PROMOTION EXECUTED.**
- GENERALITY: az_v25_br vs az_v19_newrules (never seen in training) k=4: **+0.1002 ±
  0.0312/end** (wr 0.508) — beats a foreign strong model as hard as its training target;
  NOT an anti-v14d specialist.
- CONFIRMATION at deployment settings (EXP-057 policy): vs az_v14d k=8: **+0.1065 ±
  0.0228/end (t≈4.7)** — the edge GROWS at k=8, exactly the dilution pattern of a real gap.
- Meta-game check: v25 > v14d, v25 > v19, v14d ≈ v19 — transitive with v25 on top; Nash =
  pure az_v25_br. **az_v25_br/best.pt IS THE NEW GLOBAL CHAMPION** — first promotion since
  az_v14d (2026-07-16), achieved by ONE asymmetric best-response generation (432 games).
  Arena deployment updated. az_v14d retained as the reference incumbent.

**EXP-067 (launched): BR ITERATION 2 — does the league ratchet compound?** Same machinery,
one generation up: opponent = az_v25_br (fixed), learner init = az_v25_br, exploratory
bigsel targets + matchup returns, ~400 games, guard 5.5, k=4 gate vs az_v25_br + k=8
confirmation if certified. If iteration 2 also certifies, the BR/league loop is the
self-improvement engine this project spent a month proving self-play could not be; if it
returns parity, one BR step was a one-time harvest of the symmetric loop's blind spot —
either answer sets the roadmap.
