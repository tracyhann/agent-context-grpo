# CCPO hypotheses — brainstorm and tracker

Running list of what we believe, what we have tested, and what is worth testing
next. Add a hypothesis here **before** running the arm that tests it, so the
prediction is on record and cannot be rewritten after the fact.

**Legend** — `[ ]` open · `[~]` in flight · `[x]` settled · `[!]` refuted

Each settled entry carries: the arm that tested it, the numbers, and what follows.

---

## Bottom line as of 2026-09-06

**CCPO's central mechanism is refuted on ALFWorld.** Six φ variants — the policy's
hidden state, explicit `{t, n_unique, revisit, progress}`, both concatenated,
bag-of-words, and whole-episode memory, under two targets — all give `λ = 0.000`
and `phi_rel_corr ≈ 0`. φ-similarity does not predict return-similarity inside a
bucket, measured both through the baseline's mean shift (λ) and directly through
within-bucket ordering (`phi_rel_corr = −0.0096`, positive in 42% of buckets
against a 50% chance rate).

**What survives** is not the thesis but two components corrected along the way:

* the **leave-one-out exclusion** (`i_u ≠ i_v`) — GiGPO and G²PO both include the
  scored trajectory in its own baseline; this does not;
* the **successor-value target** (`ACG_CCPO_TARGET=nextnode`) — G²PO's component 1.

Together they move the credit a long way from GiGPO (`r_vs_gigpo` +0.977 → **+0.319**)
without any context conditioning at all. Whether that helps the *policy* is
untested — it needs a matched baseline arm, which the budget has not allowed.

**What this does not say.** Nothing here rules out context conditioning on a
benchmark where situation similarity does predict outcome similarity. On ALFWorld,
from a given observation, what happens next is dominated by which action is chosen
now rather than by how the agent arrived — and φ is computed from the prompt,
before the action exists.

---

## Open — ranked by expected value

### [ ] H-G. The relevance signal is heavy-tailed across buckets, not uniformly absent

**Observation, seven steps of `ccpo-mem-20260906`.** The mean `phi_rel_corr` is
**+0.024**, yet the fraction of buckets with positive correlation averages **42%**
— consistently *below* the 50% chance rate, at every step but one.

Those two facts are only consistent if the distribution is **heavy-tailed**: a
minority of buckets where φ predicts target-distance strongly, pulling the mean up,
against a majority that are flat or slightly negative. The signal is not uniformly
absent — it is concentrated.

| step | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|
| `phi_rel_corr` | −0.010 | +0.002 | +0.060 | +0.038 | +0.009 | −0.006 | +0.074 |
| buckets > 0 | 42% | 41% | 51% | 46% | 40% | 36% | 41% |

**Why this matters for H-A.** Every shrinkage rule tried so far asks a
*batch-level* question — `eb_pooled` and `eb_hier` estimate `tau^2` over the whole
batch, and `eb` asks a per-occurrence question but through the underpowered
mean-shift test. None asks "is φ relevant *in this bucket*". If relevance is
concentrated, a batch-level estimate averages it away, and a per-occurrence
mean-shift test cannot detect it in the few trajectories a bucket holds.

**Proposed:** `λ_bucket` driven by that bucket's own `phi_rel_corr` — a direct
estimate of grouping relevance rather than an indirect one via the baseline's mean.
This is the sharpest form of the original H-A idea.

**Honest caveat before anyone builds it.** A bucket of ~9 members gives ~36 pairs,
so the per-bucket correlation is itself noisy, and selecting the buckets that look
relevant *by the same statistic that then sets their weight* is a garden-of-forking-
paths risk — the selection would capitalise on noise. It needs a held-out split
within the bucket, or a permutation test per bucket, before it can be trusted.
**Do not ship it on the strength of this observation alone.**

Also blocked by H-F: measured on a policy stuck near the floor.

### [ ] H-F. The harness plateaus far below published, and we do not know why

**The elephant.** Every arm run here sits at 5-10% training success and 0.06-0.21
held-out, while published GiGPO reports **90.16** on this exact model and protocol.

`ccpo-mem-20260906` solved, per step of 128: 4, 10, 9, 5, 9, 5, 3 — flat-to-noisy,
not climbing. `ccpo-ctx` was the same shape (7, 9, 15, 9, 7). Held-out 0.0625 at
step 5 here; 0.070 there; the earlier hardened-prompt GiGPO arm managed 0.195 at
step 5 and 0.211 at step 10, then flattened.

**I have been assuming this is a budget artifact** — published runs train 100-150
steps and ours reach 5-20. That may be right; ALFWorld curves can be slow to lift.
But it is an assumption, not a measurement, and it undercuts everything else:
a harness stuck near the floor may simply not exercise the credit-assignment
differences these arms exist to test. An estimator cannot show its worth on a
policy that solves nothing.

**Sharpened at step 12 — the policy has learned exactly one task type.**
Per-type success on the training draw:

| step | pick_and_place | pick_clean | pick_cool | pick_heat | pick_two_obj |
|---|---|---|---|---|---|
| 10 | **0.438** | 0.062 | 0.000 | 0.000 | 0.000 |
| 11 | **0.250** | 0.062 | 0.000 | 0.000 | 0.000 |
| 12 | **0.062** | 0.000 | 0.000 | 0.000 | 0.000 |

`pick_and_place` — the simplest type, one object one destination — carries
essentially every success. The four compositional types are at ~0 after 12 steps,
and `pick_two_obj_and_place` has **never succeeded once**. Published GiGPO's 90.16
requires solving all five.

So the plateau is not uniform underperformance. It is a policy that has learned the
easy type and nothing else. That is consistent with the benign reading — the
compositional types plausibly need far more than 12 steps — but it also says the
overall success rate is the wrong thing to watch: **per-type success is the
progress signal**, and the first compositional type to lift is the event that
would settle H-F.

**Train success is a poor progress signal at this batch size.** Each step draws 16
fresh tasks, so the number of `pick_and_place` tasks in the draw dominates. Observed
sd across 12 steps is 0.024 against a binomial 0.019 at the same mean — the excess
is task composition. Only the held-out set (fixed 128 episodes) is comparable
across steps, and step-to-step swings in train success (0.086 -> 0.008) carry
almost no information. Several earlier readings in this file leaned on that signal
and should not have.

**Health is fine** — no collapse to blame: KL 0.019 -> 0.022 bounded, entropy
0.79-0.86 stable, grad norm 1.5-3.2, clip fraction 0.005-0.011,
`valid_action_ratio` 0.996-0.999.

**Candidate causes, none tested:**
* genuinely needs 100-150 steps (the benign reading)
* something in the config still differs from the reference in a way that matters —
  the deltas we know about are hardware-forced (sdpa vs flash-attn is now closed,
  4 GPUs vs 8, tp=1 vs 2), but "known deltas are benign" has not been verified
* the reference prompt underperforms on this model — the hardened prompt reached
  0.195 at step 5 against 0.062-0.070 here, though that comparison is confounded
  with the estimator
* evaluation protocol mismatch we have not spotted

**Cheapest discriminating test:** one arm to step 30-40 and look at the shape. If
it lifts, budget. If it stays flat, the harness. Nothing else here is worth
trusting until this is settled — including the refutation, which was measured on
a policy that never got off the floor.

### [ ] H-C. The learned successor-feature φ — now the only surviving route, and a long shot

**Reprioritised down, not up.** Five φ variants — the policy's hidden state,
explicit `{t, n_unique, revisit, progress}`, both concatenated, bag-of-words, and
whole-episode memory — all give `phi_rel_corr ≈ 0`. A learned encoder would have to
find within-bucket ordering structure that none of them sees, and Phase 1 already
found no advantage over a null control. Cost is high (plumbing plus reward-free TD
training inside the rollout loop).

*(original entry below)*

`ccpo/learned.py` — `LearnedPhi` + action-conditioned successor features +
learned affinity, TD-trained reward-free. This is §02's actual method and **has
never been wired in**. Cost: plumbing plus TD training inside the rollout loop.
Prior evidence is weak (Phase 1 found no advantage over a null control), so this
ranks below H-A and H-B despite being the headline method.

### [ ] H-D. Similarity gate raises bucket occupancy

`ACG_CCPO_SIM=0.95` — GiGPO ships this (`are_similar`, SequenceMatcher). Would
raise `n_eff` and cut the 33% singleton rate. **Deprioritised**: support is not the
binding constraint (see H-2), and GiGPO has the same gate natively, so a
CCPO-with-gate vs GiGPO-without comparison would confound the gate with the
estimator.

### [ ] H-E. G²PO's edge term

`ACG_CCPO_EDGE_W>0` adds `V(next) − V(current)`, standardised per task. Untested.
Note `V(g)` carries trajectory-length information through `γ^(T−t)`, so watch
`acc_len_corr` for a length bias.

---

## Settled — refuted

### [~] H-A. Uncertainty should measure *grouping relevance*, not penalise within-bucket variance

> **SETTLED 2026-09-06 by `ccpo-mem-20260906` — the second branch.**
> φ-distance does **not** predict target-distance inside a bucket — measured
> directly, independent of whether φ shifts the baseline's mean. Two consecutive
> non-degenerate batches:
>
> | step | `phi_rel_corr` | buckets with corr > 0 | `tau2` | λ |
> |---|---|---|---|---|
> | 1 | −0.0096 | 42% | 0.00000 | 0.000 |
> | 2 | **+0.0016** | 41% | 0.00000 | 0.000 |
> | 3 | **+0.0600** | **51%** | 0.00000 | 0.000 |
> | 4 | **+0.0377** | 46% | 0.00000 | 0.000 |
> | 5 | **+0.0090** | 40% | 0.00000 | 0.000 |
>
> **RESOLVED AT STEP 10 (the pre-registered re-read), n=10 — "real but too weak
> to exploit".**
>
> | statistic | slope/step | r | p | first half -> second half |
> |---|---|---|---|---|
> | `phi_rel_corr` | +0.0057 | +0.512 | 0.092 | +0.020 -> +0.043 |
> | frac buckets > 0 | +0.0151 | +0.576 | **0.046** | 0.440 -> 0.490 |
>
> **Caveat added at step 11 — that significance was fragile.** One more point moved
> the bucket-fraction trend from p = 0.046 to **p = 0.112**, and the correlation
> trend from 0.092 to 0.085. With n ~ 10 a single observation flips the verdict, so
> "significant at the 0.05 level" was over-read. The *direction* is stable across
> every window (first half below chance, second half at or above it); the p-value is
> not. Read the direction, not the threshold.
>
> **The upward drift is probably real.** The bucket-fraction trend reached the
> 0.05 level at step 10: the *proportion* of buckets in which φ-distance predicts
> target-distance genuinely rises as the policy differentiates. That is the
> mechanism predicted when this was reopened, and it means the flat categorical
> claim "φ carries no signal" was **wrong**.
>
> **But the magnitude stays too small to matter.** Mean `phi_rel_corr` +0.032,
> second half +0.043, and the fit projects **+0.114 at step 20** — below the 0.15
> threshold registered in advance. At that magnitude an R²-driven λ would be
> ~0.013. So λ = 0 remains the correct verdict at every magnitude reachable here,
> and the substance of the refutation stands: **the conditioning cannot be made to
> do useful work on ALFWorld at this scale**, not because φ is uninformative but
> because it is informative far too weakly.
>
> **Correction to the earlier framing.** The refutation should not be stated as
> "φ-similarity does not predict return-similarity". It should be: *φ-similarity
> predicts return-similarity weakly and increasingly, at roughly r = 0.03-0.09 over
> the first ten steps, which is one to two orders of magnitude below what the
> estimator would need.* The first version is false; the second is what was measured.
>
> **Still live for a longer horizon.** The trend does not plateau within ten steps.
> Whether it saturates near 0.1 or keeps climbing over 100-150 steps — the published
> training length — is unmeasured, and is the single most interesting open question
> about this method. It is also gated on H-F.
>
> ---
>
> *(superseded) STATUS AT STEP 9: the trend test says drifting upward, p ~ 0.07.* Regressing
> `phi_rel_corr` on step over nine points: slope **+0.0073/step**, r = **+0.560**,
> p ~ 0.073 (normal approximation on n=9, indicative only). Positive-bucket
> fraction: slope +0.0124/step, r = +0.448. First three steps average **+0.017**,
> last three **+0.062**; step 9 is +0.086 with **60%** of buckets positive, the
> first clearly above-chance reading.
>
> Extrapolated, that reaches ~0.15 by step 20 — the pre-registered "keeps climbing"
> threshold, which calls for rerunning the variant table at matched steps.
>
> **Not calling it yet.** n=9, p above 0.05, and I have already whipsawed twice on
> this series by reacting to pairs of points. The pre-registered re-read is at step
> 10 and the arm runs to 20; the trend test is the instrument, not the latest value.
>
> ---
>
> *(superseded) RE-CLOSED at step 5 on the pre-registered criterion "falls back toward 0".*
> The series is non-monotonic — mean **+0.020**, and the positive-bucket fraction
> averages **44%**, below the 50% chance rate at every step except one. Steps 3-4
> were an excursion, not a trend.
>
> Both of my readings were over-reactions to two points. The closure at steps 1-2
> was premature; the reopening at steps 3-4 was equally so. **The right reading of
> five points is: small, noisy, centred near zero, with no trend** — which is what
> the pre-registered third criterion says, and why writing the criteria down before
> the data arrived was worth doing.
>
> Residual caveat, honestly stated: mean +0.020 is *slightly* positive rather than
> exactly zero, and five steps of an early-training policy is not many. If a longer
> arm ever runs, this is worth re-reading at step 20+. It does not change the
> verdict at any magnitude reachable here — an R²-driven λ would be ~0.0004.
>
> Original framing, retained: this is H-A's **second** branch, not the first:
>
> * `phi_rel_corr > 0` while `λ = 0` ⇒ φ carries signal the shrinkage rule is
>   discarding, and the fix is in the rule. **Act on it.**
>
> A plausible mechanism for a rising trend: at step 1 the policy is near-uniform, so
> sibling trajectories inside a bucket barely differ and there is no context
> structure to find. As training differentiates them, context starts to matter.
> If so, the early nulls measured a property of the *untrained policy*, not of the
> benchmark — and every arm here has been read at steps 1-3.
>
> **Do not re-close until step 10.** If the trend holds, the whole refutation needs
> revisiting, including the six-variant table below, all of which was read at
> equally early steps.
>
> **But do not overclaim either.** +0.05 is a weak correlation. An R²-driven λ — the
> replacement proposed below — would give λ ≈ corr² ≈ 0.0025, still effectively
> zero. So at *this* magnitude λ = 0 remains the right verdict and the rule is not
> discarding anything usable; what has changed is that the quantity is no longer
> flat at zero, and it is rising. The decision point is whether it keeps rising:
>
> | `phi_rel_corr` by step ~10 | reading |
> |---|---|
> | plateaus near 0.05 | the signal is real but too weak to exploit; refutation stands in substance |
> | keeps climbing (≳0.15) | the early nulls were an artefact of the untrained policy; **rerun the variant table at matched steps** |
> | falls back toward 0 | steps 3-4 were noise; original closure stands |
>
> This was the escape hatch for the four earlier nulls: λ only detects a *mean*
> shift, so a φ that ordered neighbours correctly without moving that mean would
> have read as zero signal. It doesn't order them either. **The premise does not
> hold on ALFWorld, and no shrinkage rule repairs that** — the R²-driven
> replacement proposed below would be estimating an R² of zero.
>
> The inversion identified below (high `s²` suppressing conditioning, when high
> `s²` is the evidence of conflation) is still a genuine design flaw and would
> matter on a benchmark where φ carried signal. It is simply not what is blocking
> CCPO here. **Do not spend compute on it.**

*(original entry retained below)*

**The inversion.** `λ_i = τ² / (τ² + ρ²·s²·var_gain)`. Within-bucket variance `s²`
sits in the **denominator**, so **high within-bucket variance → lower λ → less
conditioning**. But the method's premise is that high within-bucket return variance
*is the evidence of conflation* — distinct states sharing one observation — which
is exactly when conditioning should matter most. The rule suppresses conditioning
precisely where the thesis says it is needed.

It is defensible as pure bias-variance reasoning (a noisy neighbourhood gives a
noisy LOO estimate), but it cannot distinguish:

* **`s²` is noise** — nothing to condition on, back off. Correct behaviour.
* **`s²` is structure φ could explain** — conflation, condition harder. Currently
  penalised identically.

**Also: λ tests the wrong thing for judging φ.** `d = b_obs − b_LOO` only detects
whether φ shifts the *mean* of the baseline. A φ that correctly orders neighbours
by similarity, but whose reweighting happens not to move that mean, reads as
exactly zero signal. So "λ = 0" is weaker evidence about φ than it looks.

**Proposed replacement.** Decompose `s²` into the part φ explains and the part it
does not, and let *that* drive trust:

```
R²_bucket = 1 − Var(target | phi-weighted) / Var(target)
lambda    driven by R², not by s² alone
```

**First step, already shipped:** `ccpo/phi_rel_corr` — within-bucket correlation
between φ-distance and |target difference|, over the pairs that actually feed the
estimator. Diagnostic only; never fed back into the weights, so the reward-free
leave-one-out argument is untouched.

**Predictions.**
* `phi_rel_corr > 0` while `λ = 0` ⇒ φ carries signal the rule is discarding, and
  the fix is in the rule. **Act on it.**
* `phi_rel_corr ≈ 0` ⇒ the premise itself does not hold on ALFWorld; no rule
  change repairs that. **Stop pursuing conditioning.**

Status: measured for the first time by `ccpo-mem-20260906`.

### [!] H-B. Compaction makes φ informative — and has never actually been evaluated

> **SETTLED 2026-09-06 by `ccpo-mem-20260906`.** The digest reaches the prompt
> (`prompt_length/mean` 549 → **775**, ~226 tokens of digest) and λ is **still
> 0.000**, `phi_rel_corr` −0.0096, on a non-degenerate batch. Whole-episode history
> in the prompt does not make φ carry within-bucket signal.
>
> Training is healthy — 4 → 10 of 128 solved over the first two steps, prompt
> 775 → 802 tokens, `episode/length/mean` 49.6 → 48.0.
>
> Two sub-results worth keeping:
> * **The digest-order bug was the real damage.** 0/128 episodes solved with the
>   reversed digest, **4/128** with it chronological, against 7/128 memory-off. 4 vs
>   7 is ~1.2 binomial sd — compaction is roughly **neutral**, not harmful. 0 vs 7
>   was ~2.7 sd.
> * **The mechanism claim does not hold either.** `episode/length/mean` = 49.6 of a
>   50-turn cap: remembering what it already tried did not shorten episodes, which
>   was the stated justification (58.7% of turns revisit an already-seen observation).
>
> So "frozen φ + memory beats GRPO" (0.625 vs 0.604) remains unexplained. It was
> obtained with the history inverted, and with the digest fixed the component is
> neutral on this model at step 1.

*(original entry retained below)*

Inside a bucket the observation is **constant by construction**, and without a
digest the prompt carries only `step_count` plus the most recent `history_length`
(2) turns. So φ can separate "step 5 from step 15" but not "has this agent already
searched here twice". `n_unique`, `revisit` and `progress` summarise the whole
episode and appear nowhere in the prompt unless compaction is on.

Prior work found **frozen φ + memory was the only arm beating plain GRPO**
(0.625 vs 0.604) — but see B-3 below: that result was obtained with the digest
rendered backwards, so compaction has never been evaluated as designed.

Independent support from outside: HGPO's `K=2 → K=4` (more prompt history) bought
**more than their entire estimator contribution** (+6.77 vs +5.40 out-of-distribution).

Arm: `ccpo-mem-20260906`. Watch `phi_rel_corr`, `λ`, then `episode/length/mean`
(the mechanism claim is that remembering where it searched shortens episodes:
58.7% of turns revisit an already-seen observation, failures 2.68× as often).


### [!] H-1. Context conditioning improves credit assignment on ALFWorld

The central hypothesis. **λ = 0.000 in every variant tried**, ~1,500 credited
samples per step:

| arm | φ | target | λ | effect_rel | r_vs_gigpo |
|---|---|---|---|---|---|
| `ccpo-base` | hidden | return | 0.000 | 0.0000 | +0.977 |
| `ccpo-nextnode` | hidden | nextnode | 0.000 | 0.0000 | +0.530 |
| (mis-gated, ran bow) | bow + explicit ctx | nextnode | 0.000 | 0.0000 | +0.523 |
| `ccpo-ctx` | hidden+ctx | nextnode | 0.000 | 0.0000 | +0.505 |

`Var(d) = s²(1/n_eff − 1/J)` is **exactly the permutation null** — verified
numerically at 1.0005 ± 0.0155 over 400 configurations × 4000 shuffles. So λ asks
"does the actual φ-weighting shift the baseline more than a *random* reweighting of
the same neighbours?" and answers no, every time.

This is the production-scale test of the write-up's own **E5**, previously a null
on 15–22 pairs and explicitly not a refutation. It is now one — **for the mean
shift**. See H-A: the ordering question is still open.

### [!] H-2. The inertness is thin bucket support

Refuted. `n_eff` 4.86–4.93 of a maximum 7, mean bucket size 8.4, 90% of samples
credited, 758 buckets per step. Every condition the method needs is met.

**Correction:** the `n_eff ≈ 2.1` figure quoted earlier came from a smoke run with
`group_size=4`, where J ≤ 3 by construction — 2.1 was 70% of the ceiling, not thin
support. Never quote a support number from a reduced-group run.

### [!] H-3. The inertness is the shrinkage rule

Refuted. `eb`, `eb_pooled` and `eb_hier` independently agree. Two of the three were
built specifically to rule this out:

* `eb_pooled` — James-Stein, one λ per batch. Fixed the ~2-degrees-of-freedom
  problem in `eb`; still 0.
* `eb_hier` — `τ² = max(0, E[d²] − E[Var(d)])` from the batch, then
  `λ_i = τ²/(τ² + ρ²Var_i)` per occurrence. Keeps ensemble power *and* per-occurrence
  adaptivity (λ sd 0.196 over 0.34–1.00 on a synthetic fixture, against `eb_pooled`'s
  0.000); still 0 on real data.

### [!] H-4. The inertness is noise in the target

Refuted, and this one was mine. Reasoning was: return-to-go on ALFWorld is
near-binary (0 or 10), so `Var(d)` swamps any reweighting — conditioning the
*baseline* cannot remove noise living in the *target*. So switch to `V(next node)`,
pooled over every sibling visit and far lower variance.

The target change **worked as designed** — `r_vs_gigpo` +0.977 → +0.530, `r_vs_g2po`
+0.393 → +0.719 — and **λ was still 0**. The target was not the obstacle.

---

## Settled — confirmed (corrections, not hypotheses)

### [x] C-1. The reference estimator was mislabelled

`G[i] − mean(G[bucket])` is **GiGPO's** step advantage in `mean_norm` mode, not
G²PO's. So the repo's headline "corr(A_CC, A_G2PO) = 0.956" says CCPO tracks
**GiGPO**; against G²PO's actual estimator it had never been compared. Now
measured: **+0.39**. The two references are genuinely different quantities.
G²PO port verified against the published implementation to ≤2.1e-07
(`tests/test_g2po_port.py`).

### [x] C-2. The two advantage terms were on mismatched scales

Episode term mean-centred in reward units (|A| ≈ 2.5–7.5 on ALFWorld), step term
always standardised (|A| ≈ 1). So `step_advantage_w = 1` was in effect a ~5×
**down-weight** of the entire context contribution. Nothing chose this; it was two
defaults meeting. Both now follow one `mode`, defaulting to `mean_std_norm` — what
the G²PO reference sets, and where verl's GRPO arm already is. Measured
`|A_EP|/|A_CC|` = 1.06–1.17 after the fix.

**Candidate explanation for "the context term is inert" in §05c that does not
require the estimator to be wrong.**

### [x] C-3. The compaction digest was rendered backwards

The digest was shown in its **eviction** order (informative before no-effect,
recent before old), so the agent read its own history most-recent-first —
inverting the causality of a sequential plan, which is what the digest exists to
convey. Caught when a memory-on arm solved **0 of 128** episodes at step 1 where
memory-off solved 7–16, with actions still 124/128 valid and 120 admissible.

Eviction priority and display order are different concerns. Fixed.

**This ordering has been in the codebase since compaction was written**, so the
"frozen φ + memory beats GRPO" result was obtained with the history inverted.
Compaction has never been evaluated as designed.

### [x] C-4. φ never saw the trajectory context

E4 says φ must be a function of both observation and context. The `bow` path
thermometer-codes `{t, n_unique, revisit, progress}`; the `hidden` path computes
`derive_context()` and **never consults it**. Added `ACG_CCPO_PHI=hidden+ctx`,
with `tests/test_phi_context.py` measuring `effect_rel` 0.0007 → 0.0169 → 0.1058
(hidden → hidden+ctx → bow) on a fixture where returns depend on context.

### [x] C-5. The ALFWorld prompt had drifted from the reference

Ours told the model to reason "concisely" and gave a one-shot example with a
~25-token think block — added for Qwen3, which could not reliably emit action tags.
Reverted to reference wording. Response length 54 → 96 tokens.

**But the argument for reverting was wrong**: I expected longer reasoning to help;
train success went *down*. And the 0.843 valid-action ratio I worried about
self-corrected to 0.996 within three steps — RL fixed the format unaided.

---

## Measurement notes — things that will mislead you

* **`E_w` is not evidence about φ.** `τ_b` is the bucket's *median* distance, so
  `E_w ≈ exp(−1) ≈ 0.37` for essentially any distance distribution. It is
  scale-invariant by construction and absorbs the collapse it was meant to detect.
* **`perf/max_memory_reserved_gb` is an aggregate across pools.** It read 103.8 GB
  against a 102.6 GB card. Sample `nvidia-smi` during the update instead.
* **Never quote `n_eff` from a reduced-`group_size` run** — it is capped by `J`.
* **`episode/success_rate` on the training draw is mostly task-composition noise.**
  16 fresh tasks per step, and only `pick_and_place` is ever solved, so the metric
  tracks how many easy tasks were drawn. Use the held-out set, or per-type rates.
* **A p-value at n~10 is not a finding.** The bucket-fraction trend read p=0.046 at
  step 10, 0.112 at step 11, 0.312 at step 12. Read the direction across windows,
  never the threshold crossing.
* **A degenerate batch fakes agreement.** With every episode reward 0, all node
  values are 0 and `r_vs_gigpo`/`r_vs_g2po` both read ~0.97. Check `reward_sum`
  before reading any correlation.
* **`phi_is_hidden` had an exact-match bug** (`== "hidden"` vs `hidden+ctx`), which
  silently ran `bow` and made the guard metric report a φ failure that was really a
  gating failure. Same bug in the capture gate in `fsdp_workers.py`.
