# CCPO hypotheses — brainstorm and tracker

Running list of what we believe, what we have tested, and what is worth testing
next. Add a hypothesis here **before** running the arm that tests it, so the
prediction is on record and cannot be rewritten after the fact.

**Legend** — `[ ]` open · `[~]` in flight · `[x]` settled · `[!]` refuted

Each settled entry carries: the arm that tested it, the numbers, and what follows.

---

## Bottom line as of 2026-09-06

**CCPO's central mechanism is real but far too small to matter on ALFWorld.**
Six φ variants — the policy's hidden state, explicit `{t, n_unique, revisit,
progress}`, both concatenated, bag-of-words, and whole-episode memory, under two
targets — all give `λ = 0.000`. Over 14 steps `phi_rel_corr` averages **+0.0306**,
95% CI [+0.0140, +0.0473], which excludes zero: φ-similarity *does* predict
return-similarity inside a bucket. It is just ~3 orders of magnitude below what the
estimator needs (implied R² = 0.0009).

H-H then bounded what any better φ could buy: **6.1% of target variance**, from a
400-permutation ICC test. That 6.1% is CCPO's entire theoretical margin over GiGPO,
because GiGPO already uses the uniform bucket mean and context conditioning only
competes for the within-bucket, between-trajectory slice. So λ = 0.000 was never a
bug — the shrinkage correctly reported that there is almost nothing to shrink
toward.

(An earlier version of this paragraph quoted `phi_rel_corr = −0.0096` as the
headline. That was one step's value, and the multi-step interval above contradicts
it. The premise is true and negligible, which is a different claim from false.)

**What survives** is not the thesis but two components corrected along the way:

* the **leave-one-out exclusion** (`i_u ≠ i_v`) — GiGPO and G²PO both include the
  scored trajectory in its own baseline; this does not;
* the **successor-value target** (`ACG_CCPO_TARGET=nextnode`) — G²PO's component 1.

Together they move the credit a long way from GiGPO (`r_vs_gigpo` +0.977 → **+0.319**)
without any context conditioning at all. Whether that helps the *policy* is now
being tested by `ccpo-long-20260906`, which runs to step 100 — the reference
protocol's own length — so its held-out number is directly comparable to the
published GiGPO/G²PO figures without spending a GPU-month on baseline arms.

**The largest known gap to the baselines is training length, not method.** Every
arm so far ran 20 steps against a published 100, and held-out was still rising
monotonically at the cutoff (0.0625 → 0.0781 → 0.1250 → 0.1719, partial credit
0.226 → 0.803). Nothing about φ can be judged against published numbers until an
arm has run the published length.

**What this does not say.** Nothing here rules out context conditioning on a
benchmark where situation similarity does predict outcome similarity. On ALFWorld,
from a given observation, what happens next is dominated by which action is chosen
now rather than by how the agent arrived — and φ is computed from the prompt,
before the action exists.

---

## Open — ranked by expected value

### [~] H-I. Will 100 steps actually close the gap? — prediction on record, 2026-09-06

Stated **before** `ccpo-long-20260906` reports, so it cannot be rewritten after.

The gap is large. Published GiGPO (K=2) is **90.16** in-distribution; we are at
**17.2** at step 20. Calling that "just under-trained" is a real claim and it
deserves a falsifiable prediction rather than a shrug.

**Why the optimistic reading is defensible.** The held-out series is
0.0625 → 0.0781 → 0.1250 → 0.1719, accelerating rather than flattening, and
partial credit is at **0.803** — the agent already completes most sub-goals and is
failing on the last conversion to a finished task. On ALFWorld that final step
tends to convert in a burst, so a sigmoid rather than a line is the right prior.

**Prediction.** At step 100, held-out success lands in **0.45–0.70**. Concretely:

* **> 0.45** ⇒ H-F's "under-trained" reading is confirmed and length was indeed the
  binding constraint; the remaining distance to 90 is then a method/scale question
  worth spending on.
* **0.25–0.45** ⇒ length was *part* of it but something else is also wrong. Look
  first at `history_length=2` and the prompt, not at φ.
* **< 0.25** ⇒ "under-trained" is **refuted** and the harness has a real deficit
  that 5x the compute does not fix. That would make every φ result in this file
  provisional, since they were all measured on a crippled harness.

The third branch is the one to take seriously. If it happens, the honest move is to
stop method work entirely and diff our rollout loop against `baselines/verl-agent`
turn by turn.

**Checkpoints to watch:** step 40 should clear ~0.25 and step 60 ~0.35 if the
0.45–0.70 landing is live. If step 60 is still under 0.25, kill the run rather than
spend the remaining 4 h — early stopping is armed at step 60 with patience 8, but
it triggers on *plateau*, not on being behind schedule, so this is a manual call.

---

**Tracking — `ccpo-long-20260906`, held-out (128 fixed episodes):**

| step | 5 | 10 | 15 | 20 | 25 | 30 | 35 | 40 |
|---|---|---|---|---|---|---|---|---|
| success | .0625 | .0781 | .1250 | .1719 | .2422 | .2500 | .3047 | **.3438** |
| partial | .226 | .386 | .454 | .803 | 1.043 | 1.278 | 1.475 | **1.821** |

(Steps 5–20 are `ccpo-mem`; 25+ are the warm-started continuation. Same config, same
constant LR, so the series is one curve.)

**Step-40 checkpoint: comfortably ahead of schedule.** The marker was ~0.25 by step
40; actual is **0.3438**, and the step-60 marker of ~0.35 is nearly met twenty steps
early. Eight consecutive evaluations, every one higher than the last.

The `< 0.25` branch — the one that would have refuted "under-trained" and made every
φ result in this file provisional — is now **dead**. That was the branch worth
worrying about, and it did not happen.

**My 0.45–0.70 band now looks conservative,** and I want that on record before the
answer arrives rather than after. Naive linear continuation of the last four
evaluations (~+0.045 per 5 steps) reaches ~0.85 by step 100, which I do not believe —
success curves saturate, and the remaining tasks are the compositional ones that
convert last. But the honest statement is that the band's *upper* half is the live
region, not its middle. I am not revising the prediction; a band you move once it
starts resolving is not a prediction.

**One caution against reading the increments too finely:** at p≈0.3 with n=128 the
binomial SE is 0.040, so any single step-to-step move under ~0.08 is inside noise.
The step-30 flat point (+0.008) was noise and resolved as such at step 35. What
carries weight here is the monotone run of eight, not any one increment.


> **Scheduling decision, 2026-09-06, following H-H below.** The remaining budget
> goes to *length*, not to another φ variant.
>
> H-H bounds every φ hypothesis at 6.1% of target variance, and that 6.1% *is*
> CCPO's entire theoretical margin over GiGPO — GiGPO already uses the uniform
> bucket mean, so context conditioning only ever competes for the within-bucket,
> between-trajectory slice. No encoder turns that into a SOTA-sized lever.
>
> H-F meanwhile resolved the other way: held-out success is climbing monotonically
> and the run is simply under-trained at 20 steps. The reference protocol
> (`baselines/G2PO/examples/g2po_trainer/run_alfworld.sh`) is
> `train_data_size=16, group_size=8, total_epochs=100` — **100 steps**, five times
> what any arm here has had.
>
> So: `ccpo-long-20260906` warm-starts from `ccpo-mem-20260906/global_step_20` and
> runs to an absolute step 100, matching the reference length exactly. ~10 h on
> 4 GPUs. Warm-starting is legitimate here — same config, same constant LR, so it
> is the same run continued, not a new arm.
>
> This is the honest ordering: **the largest remaining gap to the baselines is
> training length, and it is not a method question.** Nothing about φ can be
> evaluated against published numbers until an arm has run the published length.


### [x] H-H. **The variance budget — how much is on the table for *any* φ**

Before spending another GPU arm on a better encoder, measure the ceiling. All of
H-C/H-D/H-E/H-G are attempts to make φ predict target-similarity better; none of
them can beat the amount of target variance that is *conditionable in principle*.

**Experiment** `analysis/variance-budget` (offline, CPU, on the
`ccpo-mem-20260906` dump: 90,728 rows, steps 1–16, 4,855 buckets with J≥3).

Decompose the target's variance against the bucket gate:

| component | var | share of total |
|---|---|---|
| between-bucket (bucket means) | 0.2856 | 0.557 |
| within-bucket | 0.2274 | **0.443** |

The uniform bucket baseline `b_obs` already removes the between-bucket 55.7%.
The 44.3% within-bucket is everything CCPO's context conditioning is competing for.

Now split the within-bucket part by trajectory. ICC(trajectory), against a null
that permutes trajectory labels *within* each bucket at matched group sizes:

```
buckets with >=2 distinct trajectories : 1204
ICC real  mean +0.0221
ICC null  mean -0.1163   null 95% range [-0.1556, -0.0785]   (400 permutations)
bias-corrected ICC = real - null = +0.1384
permutation p = 0.0000   (0 of 400 permutations reached the real mean)
```

The null is negative because the one-way ICC estimator is biased at small k; the
permutation preserves each bucket's group sizes, so that bias is matched and the
difference is the unbiased estimate. (A first pass quoted +0.1226 with a CI taken
from the real ICC's spread alone, which ignored the null's own variability — the
permutation test above supersedes it and is the stronger result.)

**Confound checked, and it does not explain the result.** Two occurrences in the
same trajectory share their future, so targets would cluster by trajectory
mechanically. But same-trajectory *revisits* of the same bucket still carry
var 0.1579, against between-trajectory var 0.2759 (one occurrence sampled per
trajectory, so no two points share a trajectory). The ICC is neither degenerate
nor purely shared-future — 56% of occurrences per (bucket, trajectory) are
repeats, mean 4.17.

**Conclusion — the ceiling, and it is low.**

```
conditionable share of total target variance  ~=  0.1384 x 0.443  =  0.061
what the current phi actually captures        ~=  phi_rel_corr^2 =  0.0009
```

So there is real structure — the signal is **~65x larger than what φ currently
extracts**, which says the estimator is *encoder-limited, not signal-limited*, and
that is the strongest argument H-C has ever had.

But the oracle is **6.1% of target variance**. Even a perfect trajectory-level φ
buys a ~5% variance reduction on the step term, which is itself one of two
advantage components. That is not a SOTA-sized lever, and it retroactively
explains λ = 0.000 across every variant tried: **the shrinkage is correct.** There
is almost nothing to shrink toward, and the estimator has been reporting that
faithfully for 15 steps while I looked for a bug in it.

**This bounds every remaining φ hypothesis.** H-C, H-D and H-G are all competing
for the same 6.1%. They should be ranked below anything that changes the *other*
advantage term or the harness, and H-C is worth at most one arm — as a
measurement of how much of the 6.1% a learned encoder recovers, not as a
SOTA attempt.


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

### [x] H-F. The harness plateaus far below published — RESOLVED: under-trained, not broken

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

> **RETRACTED at step 15, and H-F resolves toward the benign reading.** The
> held-out set reports per-type success, and it says the opposite of the training
> draw:
>
> | task type | s5 | s10 | s15 | s20 |
> |---|---|---|---|---|
> | look_at_obj_in_light | 0.000 | 0.167 | 0.226 | 0.125 |
> | pick_and_place | 0.119 | 0.142 | 0.210 | **0.255** |
> | pick_clean_then_place | 0.105 | 0.075 | 0.150 | **0.192** |
> | pick_cool_then_place | 0.000 | 0.045 | 0.077 | 0.000 |
> | pick_heat_then_place | 0.000 | 0.071 | 0.000 | 0.000 |
> | pick_two_obj_and_place | 0.071 | 0.000 | 0.050 | **0.214** |
> | **OVERALL** | 0.062 | 0.078 | 0.125 | **0.172** |
> | partial-credit score | 0.226 | 0.386 | 0.453 | **0.803** |
>
> **Final, run complete (clean exit at step 20, zero error lines.)** Overall
> held-out rose at every single evaluation — 0.0625 → 0.0781 → 0.1250 → 0.1719 —
> and the partial-credit score nearly doubled again at s20 to 0.803. `pick_two_obj`
> went 0.050 → 0.214, the first *compositional* type to lift, which is the event
> this entry named in advance as the one that would settle H-F.
>
> Per-type remains noisy at 128 episodes (the two `_then_place` types sit at 0.000
> at s20 after being non-zero at s15); the OVERALL and partial-credit series are
> the trustworthy ones, and both are monotone across four evaluations.
>
> **Four of six types are rising**, held-out success has doubled, and the
> partial-credit score has doubled — the agent is making substantial progress on
> tasks it does not finish. This is a healthy learning trajectory at step 15 of a
> process the published work runs for 100-150 steps. **The plateau was not real;
> it was the training-draw metric.**
>
> The claim below — that the policy had learned exactly one task type — came from
> per-type rates on the 16-task training draw, where each type gets a handful of
> samples per step. It was wrong, and it is a second instance of the same mistake
> the gotchas section already warns about: **never read progress off the training
> draw.**
>
> Bearing on H-A: relevance did *not* grow over 14 steps even though the policy was
> demonstrably improving over the same window. So "the early nulls measured an
> undifferentiated policy" is refuted too — the policy differentiated, and
> `phi_rel_corr` stayed at ~0.03.
>
> ---
>
> *(retracted) Sharpened at step 12 — the policy has learned exactly one task type.*
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
raise `n_eff` and cut the singleton rate, measured at **0.364** at step 15.

**Re-prioritised, and the earlier reasoning here was wrong on two counts.**

1. *It is not confounded with the baseline.* GiGPO's own default is
   `enable_similarity: False` (`baselines/verl-agent/verl/trainer/config/ppo_trainer.yaml:253`),
   and no ALFWorld run script overrides it. So the published GiGPO numbers were
   produced with **exact** anchor matching, exactly what our `ccpo_sim=0.0` does.
   There is no fairness gap to close, and enabling the gate is a genuine
   improvement *over* the baseline's grouping rather than a correction to it.

2. *It is not bounded by H-H.* The 6.1% ceiling was measured on buckets with J≥3 —
   it caps how much better a φ can predict *within buckets that already exist*.
   The gate changes something different: whether an occurrence lands in a usable
   bucket **at all**. 36.4% of occurrences currently get no step credit from any
   estimator, and that mass is outside H-H's denominator entirely.

That makes H-D the **highest-value open φ-adjacent hypothesis** — it is the only
one whose upside is not capped at 6.1%. It is still deprioritised behind the
reference-length run, because a gate change is untestable against published
numbers until an arm has run the published length, but it is now the first thing
to try after `ccpo-long`.

### [ ] H-E. G²PO's edge term

`ACG_CCPO_EDGE_W>0` adds `V(next) − V(current)`, standardised per task. Untested.
Note `V(g)` carries trajectory-length information through `γ^(T−t)`, so watch
`acc_len_corr` for a length bias.

---

## Settled — refuted

### [!] H-A. Uncertainty should measure *grouping relevance*, not penalise within-bucket variance

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
> **FINAL, n=14 — the premise is true and negligible.** Enough points to stop
> re-litigating:
>
> ```
> phi_rel_corr    mean +0.0306  sd 0.0318  95% CI [+0.0140, +0.0473]   excludes zero
> frac buckets>0  mean 0.464                                           below 0.50 chance
> trend p-value   s10 0.046 -> s11 0.112 -> s12 0.312 -> s13 0.095 -> s14 0.324
> implied R^2     0.00094   ->  an R^2-driven lambda of ~0.0009
> ```
>
> **Three conclusions, all needed together:**
>
> 1. **φ-similarity does predict return-similarity.** The mean is +0.031 with a 95%
>    CI that excludes zero. The categorical claim "φ carries no signal" — which this
>    file asserted twice — is **false**.
> 2. **There is no trend.** The apparent climb through step 10 was a small-sample
>    artefact; the p-value wandered 0.046 -> 0.32 -> 0.095 -> 0.32 as points
>    arrived. The relevance is roughly constant, not growing with training.
> 3. **It is negligible.** r = 0.031 means R^2 = 0.0009. Even the sharpest possible
>    relevance-driven shrinkage rule — the one proposed in this entry — would set
>    λ ~ 0.001. λ = 0 is not a failure of the rule; it is the right answer, arrived
>    at from a rule that happens to be testing something else.
>
> The mean being positive while only 46% of buckets exceed zero is the heavy-tail
> signature from H-G: a minority of buckets carry the signal. That remains the one
> structurally interesting thing here, and it is still gated on H-F.
>
> ---
>
> *(superseded) Caveat added at step 11 — that significance was fragile.* One more point moved
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
