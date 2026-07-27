# Why post-champion iterations regressed instead of tying

*Analysis, 2026-07-15. Companion to `experiments_log.md` (az_v9 .. az_v13). Question posed:
after the champion (`az_v9_selfplay/iter2/best.pt`, wr 0.5657 ± 0.0048 / dScore +0.234 ± 0.023
over 7 draws), why did every further iteration land BELOW it — not even at a tie? Three
candidate explanations were proposed: (H1) a game-theoretic weakness of self-play MCTS —
training vs a stronger self drifts strength away from the fixed weaker eval opponent;
(H2) mis-anchoring on "real value" derived from human play; (H3) a bad validation /
early-stopping mechanism. Verdict up front: **H3 plus flat-target erosion explains most of
the pre-az_v13 regression (proven by fixing it); a fourth, unnamed mechanism — peak-selection
statistics — explains most of the post-fix residual; H2 is a real, live tension we have
never controlled for and is cheaply testable; H1 is real but second-order at our current
distance from the human lineage, and matters mainly through the value function and the
definition of "improvement" used by the significance gate.***

---

## 1. The phenomenon, precisely

Fourteen post-champion retrains across five operator families, all warm-started from (or
gated against) the champion:

| run | recipe | wr | dScore | Δwr vs champ |
|---|---|---|---|---|
| az_v9 it3-6 | 2-ply V-leaf, accumulate, VFM | 0.545-0.558 | +0.16-0.22 | −0.008..−0.021 |
| fresh-window | champion-gen data only | 0.5496 | +0.171 | −0.016 |
| az_v10 it1-2 | flat terminal-MC | 0.5456/0.5458 | +0.149/+0.169 | −0.020 |
| az_v11 it1-2 | noise-starved tree | 0.537 | +0.04-0.06 | −0.029 (worst) |
| az_v12 it1-2 | robust screen→tree | 0.5453/0.5467 | +0.157/+0.134 | −0.019 |
| az_v13 step0 | + post-hoc conf filter, fixed selection/stopping | 0.5514 | **+0.237 (ds parity)** | −0.014 |
| **az_v13 it1** | **+ collect-time t≥2 gating, anchors** | **0.5571** | **+0.225** | **−0.003 (FULL parity)** |
| az_v13 it2 | same | 0.5514 | +0.177 | −0.008 |

Two distinct regimes: **(i) a systematic bias-down of ~0.01-0.03 before the az_v13 fixes**,
and **(ii) a residual "never above, occasionally slightly below" scatter after them**.
These have different explanations and should not be conflated.

---

## 2. H3 — the validation / early-stopping mechanism (LARGELY CONFIRMED, now fixed)

What was wrong, in order of measured harm:

1. **Checkpoint selection by `val_value_mse_mcts` was strength-misaligned.** Twice measured
   directly: EXP-024 (best.pt *worse* than last.pt in games) and az_v9b. The selected epoch
   was the one whose value head best fit search-value targets — not the strongest player.
2. **Distillation targets were mostly noise, and training on them is actively harmful, not
   neutral.** Soft-topk at temperature 0.35 converts ±0.25 pts of MC error into
   confident-looking weights; the collect-time t-test later showed **only ~8.5% of plies
   carry a statistically real preference** (az_v12's apparent "63% confident" collapsed to
   ~10% under t ≥ 2). Distilling near-uniform or noise-ranked targets erodes/randomizes the
   policy's sharp proposal distribution — the thing the deployed 48-sample selector depends
   on. This is the *flat-target erosion* channel; az_v11 (2 noise realisations per
   candidate) is its extreme demonstration (−0.029, dScore collapsed to +0.04).
3. **Stopping was governed by epoch counts or monotone losses**, exposing every fine-tune to
   post-optimum drift epochs.

**The proof is the fix**: significance-gated distillation + aligned selection metric
(`val_policy_distill_mcts` on a sig-masked held-out partition) + val-driven patience
stopping + drift guard produced, in az_v13 it1, the **first retrain ever to hold the
champion on both metrics** (Δwr −0.003, Δds +0.003). H3-class defects were therefore the
dominant cause of regime (i). What H3 does *not* explain: why nothing lands *above* — see §5.

---

## 3. H2 — anchoring on human-derived "real value" (REAL, UNCONTROLLED, TESTABLE)

The mechanism, stated sharply: **two pieces of our pipeline treat human-play value as ground
truth, and both push the value head toward V^{human} on every retrain**:

- the **`mix_value = 0.30` replay slice** — realized-ValueDiff targets from *human games*
  (states valued under human continuations);
- the **drift guard** (az_v13) — an epoch is best.pt-eligible only if `val_value_mse`
  *on human-derived data* stays low. What we called "VFM drift" and guarded against may
  partly be **correct adaptation** of V from V^{human} toward V^{π} that we then penalized.

For a policy that now differs from human play, V^{π} ≠ V^{human} precisely in the states
where the policy's improvements live. The deployed selector ranks candidates with this V
every ply, so mis-anchoring costs strength multiplicatively.

Evidence (new, pulled for this analysis): the az_v13 retrains all **improved human-value
fit beyond the champion's while losing or merely tying strength** — champion 2.399;
step0 2.302, it1 2.316, it2 2.316 (val_value_mse, human data). The champion sits at a
*less* human-calibrated point than its descendants. Consistent with (not proof of) the
story that each retrain "re-humanizes" V a little and pays a small strength tax, and that
our guard institutionalizes this. Counterpoint keeping this honest: it1 (full parity) and
it2 (dip) have near-identical human-fit, so human-fit is not a sufficient statistic —
scatter dominates at this margin, and the champion itself was trained WITH the same 0.30
anchor.

There is also a subtle interaction with H1 pulling the *opposite* way: our VFM value
targets are realized margins **under self-play**, while evaluation plays **against the
prior** — so pure V^{self} is also not the right value for the eval matchup. The champion
plausibly sits at an accidental sweet spot on the human↔self-play value axis; any retrain
re-mixes the two anchors and jiggles away from it. A "re-equilibration" story of this kind
explains a *symmetric, direction-free* small loss — matching regime (ii) better than any
directional mechanism.

**Cheap decisive tests** (config-level, ~4h each with the fixed az_v13 recipe):
1. mix_value 0.30 → 0.10, guard off → if strength ≥ parity or improves, the human anchor is taxing us.
2. Replace the human value slice with a **self-play value buffer** built from existing
   collections (realized margins are already in every record; zero recollection).
3. The sharpest version: value targets from games **vs the prior** (see H1 remedy below) —
   the value that actually prices the eval matchup.

---

## 4. H1 — game-theoretic self-play drift (REAL BUT SECOND-ORDER HERE; enters through V and through the gate's definition of "improvement")

The concern is legitimate in general: self-play optimizes toward (locally) best response to
*self*; the eval is vs a fixed, different, weaker opponent; in non-transitive regions those
objectives diverge. But three observations bound its size in our setting:

- **Curling vs a fixed noise model is two-player zero-sum with full observability**; the
  minimax policy is robust against *any* opponent. Self-play drift hurts when training
  chases exploitable idiosyncrasies of self; it cannot push a policy *below* its minimax
  value vs a weaker opponent unless the function approximator trades off states the prior
  visits for states only self-play visits. That trade is possible (finite capacity!) but
  bounded.
- **The signature is absent**: drift predicts monotone decay vs the prior with continued
  self-play iterations. az_v9 it3→it6 shows no trend (0.558, 0.553, 0.545, 0.553), and all
  iterations stayed *above* the pre-self-play baseline vs the prior. az_v13 it1's full
  parity with purely self-play-sourced updates is hard to square with strong drift.
- **Where H1 genuinely bites is inside the machinery, twice**: (a) the value head learns
  V under self-play continuations but prices candidates vs the prior at eval (states where
  the prior blunders are undervalued as opportunities); (b) the **t ≥ 2 "significant"
  plies are improvements under the assumption that BOTH continuations are played by our own
  policy** — their value vs the prior is attenuated or even irrelevant. This is a clean
  partial explanation for az_v13's "signal exists (8.5% of plies) but distilling it moves
  nothing vs the prior."

**Remedy worth running** (moderate cost, uses existing infra): **mixed-opponent collection** —
some fraction (e.g., 50%) of collection games played against the frozen prior (both throw
orders), with value targets = realized margins *from those games* and the significance gate
computed under the actual eval matchup. Since our declared metric IS strength vs the prior,
optimizing partially against it is legitimate, not cheating — though it specializes the
policy toward this opponent, which should be stated in the paper if used.

---

## 5. H4 — the unnamed fourth mechanism: peak-selection statistics (EXPLAINS THE POST-FIX RESIDUAL)

The champion is not a typical draw of the training process — it is the **promoted upper
tail** of a stochastic pipeline (SGD run × data draw × gate). The 7-draw confirmation pins
its true strength (0.5657), but the *process* that produced it has a mean lower than its
peak: the post-fix retrain distribution is centered around ~0.55-0.557 with run-to-run
scatter of roughly ±0.005-0.01 (visible directly: az_v13 it1 = 0.5571, it2 = 0.5514 —
same recipe, same-sized data, different draws).

Under this reading, once az_v13 removed the bias (regime i), the observation "retrains tie
at best and sometimes dip slightly, never exceed" is **exactly the expected signature of an
unbiased process with zero remaining improvement signal, gated by a ratchet**: you only
beat an upper-tail incumbent by drawing the upper tail again *plus* real signal; with no
signal, ~half of clean retrains land slightly below and none clear the gate. No additional
degradation mechanism is required to explain regime (ii) — though H2's re-equilibration
plausibly contributes to the scatter.

Corollary worth internalizing: **"no retrain ever tied the champion" was never the right
summary** — it conflated the biased era with the unbiased one. After the fixes, one of two
retrains achieved statistical parity immediately.

---

## 6. Synthesis — attribution

| observation | dominant explanation |
|---|---|
| universal ~0.01-0.03 loss, pre-az_v13 | H3 (misaligned selection) + flat/noisy-target erosion; worst case az_v11 = noise-starved targets |
| dScore collapse in az_v11 (+0.04) | pure target-noise erosion (2 realisations/candidate) |
| post-fix residual (at-parity to −0.008) | H4 peak-selection scatter; possibly + H2 re-equilibration tax |
| "significant plies teach nothing vs prior" | H1(b): significance defined under self-continuations; + sparsity (~400 examples) |
| retrains improving human-value fit while not gaining strength | H2: human anchor is not the strength-relevant calibration |
| no monotone decay over self-play iterations | bounds H1(a) to second order |

## 6b. Addendum (2026-07-16): the missing proof, and the PSRO argument

A sharper question was raised against §3-§4: **do we have proof that the training paradigm at
least finds a best response (BR) to what it plays against during training, even if that BR is
not optimal against the human prior?** If yes, two consequences follow with force:
(i) the human-value anchor (H2) loses its last justification — a validated BR-improver should
price positions under the *actual matchup being optimized* (realized margins from
learner-vs-opponent games), not under human play; and (ii) **PSRO** (Policy-Space Response
Oracles / double oracle) becomes the principled escalation, using the current MCTS-style loop
as the BR oracle.

**Answer: we do NOT have that proof — the evidence hole is real.** Every gate and multi-draw
eval in az_v9→v14 compared models *vs the human prior*; the modern loop never once played
new-vs-incumbent head-to-head. What exists is only suggestive and equilibrium-flavored, not
BR-flavored: the t≥2 convergence diagnosis (§ az_v13) shows the champion is un-improvable
*against itself* by one-shot ply deviations at ε resolution — an approximate self-play
equilibrium claim. Note also the oracle-quality caveat: our operator improves via single-ply
deviations with learner continuations and a self-play value function — a policy-iteration-style
BR *approximator*, not an exact BR oracle. (PSRO tolerates approximate oracles — convergence is
then to approximate equilibria — but the BR ability should be established empirically first.)

**The missing proof is cheap and is literally PSRO step 0: the meta-game payoff matrix.**
Launched 2026-07-16 (`scripts/_metagame_matrix.sh`, `eval_out/metagame/`): noisy N=400 h2h for
the 6 new world-vs-world pairs among {exp_021, champion, az_v13-it1, az_v14d} (the vs-prior
column already exists from past multi-draw evals). It answers three things at once:
1. Did promoted models beat their data-generating opponents? (paradigm validation)
2. Do the parity-vs-prior retrains (az_v13-it1, az_v14d) beat the champion head-to-head?
   (direct evidence for H1's transfer gap if yes)
3. Is the meta-game transitive? (decides whether PSRO's population machinery buys anything)

**Why PSRO fits this codebase unusually well** (~80% of the stack exists): the population is
5-8 provenance-tracked checkpoints; the multi-draw noisy h2h machinery IS the matrix evaluator;
the ratchet gate becomes the population-admission rule; significance-gated collection becomes
the BR oracle's target generator. The one real build item is **opponent-aware collection**
(self-play games + MC rollouts currently step both sides with one policy; the BR oracle needs
learner-by-search / opponent-by-fixed-policy with the opponent drawn from the meta-mixture per
game — a contained change to `selfplay.py` and `_mc_rollout_terminal_batch`, parity-switched
policy per ply). That single change simultaneously delivers the **H2 fix** (value targets =
realized margins from the true matchup) and the **H1 fix** (the t≥2 gate computed under the
actual opponent — "significant" finally means "significant against who we're measured on").

**Decision tree on matrix results:**
- (a) BR-ability confirmed + non-transitivity present → build full PSRO (opponent-aware oracle
  + LP meta-solver over the population; double-oracle convergence certificate).
- (b) BR-ability confirmed + matrix transitive → Nash degenerates to the top policy; build only
  the opponent-aware oracle and run "az_v15: ratchet vs prior-inclusive mixture" — the minimal
  H1+H2 fix (population machinery buys nothing).
- (c) BR-ability NOT confirmed → this document's §3-§4 need revision: the loop never was a BR
  finder; the az_v9 promotion likely reflects value-head improvement rather than policy
  improvement; PSRO is premature and the oracle itself is the binding constraint.

**MATRIX RESULTS (2026-07-16): branch (b), and the plateau narrative is overturned.**
Full payoff table in experiments_log.md (EXP-042). Headlines:
1. Perfectly transitive: **v14d > v13it1 > champ > exp021 > prior** (all 10 pairwise dScore
   signs consistent; no cycles; restricted Nash = pure az_v14d).
2. **BR-ability confirmed for the fixed-recipe retrains**: models trained on champion-generation
   data beat the champion head-to-head (+0.036, +0.106 dScore/end).
3. **H1 quantified**: v14d's +0.106/end edge over the champion transfers to only ~+0.013 vs the
   prior (~10-15% transfer). The vs-prior ratchet gate was therefore rejecting genuinely
   stronger models (az_v13-it1, az_v14d) — **this document's §5 "peak-selection scatter around a
   signal-less process" reading is now REVISED: the post-fix retrains were not scattering around
   the champion; they were IMPROVING in head-to-head strength, invisible to a saturating
   fixed-opponent metric (top-3 cluster at +0.23-0.25 vs prior).** The regression phenomenon this
   document set out to explain was one part training-side defects (§2, fixed) and one part
   metric artifact (§4/H1 — larger than initially assessed).
4. Consequences adopted: **incumbent-relative gating** (promotion = beat the incumbent h2h,
   dScore-primary — the AlphaZero gate, retroactively validated), **drop the human-value anchor**
   (H2; value from actual training matchups), and no PSRO meta-solver needed (transitive).
   Confirmation draws for the re-crowning edges running before az_v14d is formally promoted.

## 7. Recommended experiments (cost-ordered)

1. **H2 ablation A** (~4h, config-only): az_v13 fine-tune with `mix_value 0.30→0.10`,
   guard off. Read: does parity hold or improve without the human anchor?
2. **H2 ablation B** (~5h): self-play value buffer (build from existing records) replacing
   the human value slice; same recipe.
3. **H1/H2 combined** (~1-1.5 days): mixed-opponent collection (50% games vs the frozen
   prior), vs-prior value targets, significance gate under the eval matchup, one ratchet
   iteration. The single most on-target experiment for our declared metric.
4. **az_v14 capacity scaling** (running): orthogonal to all of the above; tests whether any
   of these ceilings is capacity-set.

*Files: champion `checkpoints/csas_world/az_v9_selfplay/iter2/best.pt`; fixed recipe
`configs/exp_036_confident_finetune.yaml` + `configs/exp_037_sig_screen_tree.yaml`;
value-calibration numbers pulled from each run's `results.json` (2026-07-15).*
