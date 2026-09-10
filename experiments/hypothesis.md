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

### [!!! REFUTED — five ways] H-AK. Inferred state equivalence carries no signal on ALFWorld

The CCPO thesis is that state equivalence should be **inferred** rather than asserted by
exact match. Every form of that claim has now been measured and every one fails.

| form of the claim | measurement |
|---|---|
| phi as pairwise similarity across a task | r = **+0.011** with abs(target difference) |
| affordance-Jaccard as pairwise similarity | r = **-0.013** (and the wrong sign) |
| phi-weighting inside a bucket vs uniform (H-Q) | **2-10 R^2 points WORSE** than uniform |
| ON/OFF ablation, end to end | **-0.01 points** over 9 paired evaluations |
| soft rescue of orphaned occurrences (below) | closer **20.9%** of the time (50% = chance) |

**The orphan test was the last surviving form**, and the most favourable one. Refining
the key to `(task, obs, aff)` buys 29.1% of within-node variance but leaves 4.9% of
occurrences in singleton nodes with no leave-one-out baseline -- a gap G2PO simply does
not fill. Soft affordance-weighted rescue over the parent node against G2PO's only
option, the parent mean:

```
orphaned occurrences tested            11,051
(A) parent-node mean  [G2PO's option]  MAE 0.3952
(B) affordance-soft neighbourhood      MAE 0.3949    +0.1%
soft closer on                         2,312/11,051  = 20.9%
```

20.9% is *below* chance: the weighting is worse than uniform for most orphans.

**The structural lesson, and it is the real result.** The signal lives in **partition
refinement**, not in **metric learning**:

* `aff` as a partition refinement inside an observation node -> **29.1% of variance**
* `aff` as a similarity metric across a task -> **~0**

Same feature, same data. Refining a hard partition works; softening it into a metric
does not. **No better phi rescues this** -- the architecture is what fails, and two
independent representations (hidden-state phi, affordance sets) reach the same place.

**What this means for the project.** CCPO as designed will not reach SOTA on ALFWorld.
The honest contribution is the negative result: a five-way refutation of a plausible
idea, with a quantified account of why the exact method wins. The useful refinement
(affordances) is free from the environment API and G2PO already takes it.

**The one constructive thread left** is not CCPO's thesis: G2PO's invisible-state repair
uses hand-written ALFWorld vocabulary (`"You heat"`, `"You cool"`), while ours derives
failures generically from `obs_{t+1} == obs_t`. If the generic version matches the
hand-coded one, that is a modest but real contribution. `g2po-aff` measures it.


### [!!! MEASURED] H-AJ. Admissible actions explain 29.1% of within-node value variance — 5x context

Offline on `ccpo-hardedge`'s 264,058 dumped occurrences. That arm ran the HARD gate, so
its `bucket` is `(task, observation)` -- G2PO's key minus the admissible-action list.
Within each node, how much of the variance in `target` (= V(next)) does the admissible
set explain?

```
nodes with >=4 occurrences                         12,632
nodes where `aff` splits them (>=2 distinct sets)   3,924   (31.1%)
within-node variance in V(next) explained by aff    29.1%
```

**Compare H-H's ICC of 6.1% for context conditioning.** The admissible-action set carries
roughly **five times** the signal of the thing this project was built on.

**G2PO already captures it.** Their anchor is `observation + admissible actions`
(`env_manager.py`), so those splits happen inside their grouping for free. Ours is
observation-only, so we pool states whose available actions differ -- destroying 29.1% of
the within-node variance in exactly the quantity the step term predicts. **This is a
quantified mechanism for part of the 12-20 point gap, not a hypothesis.**

It also explains why `ccpo-hardedge` failed despite using G2PO's key: the key was right,
but the anchor feeding it was too coarse.

**PRE-REGISTERED prediction for `g2po-harness-20260910`** (running now: their estimator,
our un-refined anchor, `anchor_aff=0`):

* it should land **between** our CCPO arms (79.7 endpoint) and the reference (91.4) --
  it gains their estimator but not their state resolution
* if it matches the reference, the anchor does not matter and H-AJ is a red herring
* if it matches our arms, the estimator does not matter and the anchor carries everything

**Next arm after it: same config with `anchor_aff=1`.** That is the direct test of whether
our generic `aff` field recovers what their hand-built anchor gets. Unlike their
`PATTERNS = ["You heat", "You cool", ...]`, ours needs no domain vocabulary -- which is
the one place a contribution over G2PO still plausibly exists.


### [SETTLED] H-AH-FINAL. G2PO reproduces on our stack at 91.4/95.3; CCPO loses by 12-20 points

`g2po-ref-20260909` ran their tree to step 100 on our hardware, no deviation from their
published configuration.

```
trajectory: 12 12 29 36 47 55 65 67 72 80 82 85 87 86 82 78 89 88 95 91

                      G2PO    ours    diff
endpoint (step 100)   91.4    79.7   +11.7
window (60-100)       86.8    66.8   +20.0
window sd              5.0    13.7
peak                  95.3    79.7      -     <- DO NOT QUOTE (max of 20 draws, H-Z)

published G2PO 95.0; it touched 95.3 at step 95.
```

**The reproduction question is closed.** Their number is real on this hardware, this
environment build, this evaluation draw. Nothing about our stack caps performance.

**CCPO loses decisively.** +11.7 endpoint, +20.0 on the converged window -- far outside
the run-to-run variance measured from the accidental replicate (mean 6.5, max 14.1).
Our arm is also far less stable: window sd 13.7 against their 5.0.

**Everything else is eliminated.** Config identical (H-AB); harness diff reduced to the
two anchor mechanisms (H-AD/H-AE); KL clamp inert (Qwen3-era); our port of their
estimator verified faithful (H-AF); eval split and protocol identical.

**The design error, stated plainly.** `ACG_CCPO_GATE=global` uses context conditioning to
REPLACE state grouping -- one bucket per task, with phi expected to recover structure.
phi measures inert four ways, so in practice each action is compared against a task-wide
average. G2PO compares an action against other actions *from the same state*, which is
the mechanism the whole GiGPO/G2PO family rests on. **We swapped a working mechanism for
one carrying no signal.**

**But the grouping key alone does not explain it.** `ccpo-hardedge` uses the identical
`(task, anchor)` key and still reaches 19.5/41.4 against G2PO's 46.9/79.7. Something
beyond the key -- baseline form (leave-one-out + lambda shrinkage vs self-inclusive
within-node standardisation), or the anchor conflation itself -- carries the rest.

**Next: `--arm g2po`** (their estimator, vendored verbatim, in OUR harness) separates
estimator from harness. Launched.


### [!!! CONFIRMED at step 50] H-AH-final. G2PO reaches our 100-step result in 50 steps

```
step     5   10   15   20   25   30   35   40   45   50
G2PO    12   12   29   36   47   55   65   67   72   80
ours    10   11   17   21   21   31   36   23   41   50
gap     +2   +1  +12  +15  +26  +24  +29  +45  +31  +30
```

**G2PO's step-50 held-out is 79.7% -- exactly our best arm's FINAL value at step 100.**
It reaches our best result in half the budget, with 50 steps still to run and 15 points
to the published 95.0.

**This clears the noise objection that H-AI raised.** The measured within-method spread
(attempt 1 vs attempt 2, same config and seed) is mean 6.5 / max 14.1 points. A 30-point
separation is 2-4x that, and unlike the early steps it is sustained across six
consecutive evaluations. The conservative test -- the WORSE G2PO run against ours -- was
positive at every step where both existed.

**Caveat that remains.** From step 40 on there is only one G2PO trajectory (attempt 1
died at 35), so the within-method control no longer runs alongside. The variance estimate
comes from steps 5-35. At a 30-point gap that does not threaten the conclusion, but a
precise figure still wants a CCPO replicate.

**Standing conclusion: CCPO is substantially worse than G2PO on identical hardware, data,
protocol and evaluation draw.** Not a harness artifact -- config verified identical
(H-AB), harness diff reduced to the two anchor mechanisms (H-AD/H-AE), the KL clamp shown
inert, and our port of their estimator verified faithful (H-AF).

### [!!!] H-AI. The accidental replicate: G2PO's own run-to-run spread is ~10 points

The OOM restart produced something we had never had -- **a same-config, same-seed
replicate of the same method.** Attempt 2 differs from attempt 1 only in
`gpu_memory_utilization` (0.6 -> 0.3) and `save_freq` (-1 -> 20). Neither touches the
objective; the first changes vLLM's KV-cache size, hence batching, hence sampling order.

```
step   G2PO a1   G2PO a2   ours     a1-a2   a2-ours
   5      10.2      12.5   10.2      -2.3      +2.3
  10      21.1      11.7   10.9      +9.4      +0.8
```

**At step 10 the two G2PO runs differ by 9.4 points, and attempt 2 is level with our
arm.**

**This undercuts H-AH.** The "+10 to +15 point G2PO lead" measured over seven
evaluations is the same magnitude as G2PO's own nondeterministic spread. With one run per
method I cannot separate "G2PO is better" from "attempt 1 drew well".

**What survives.** G2PO does *run* on our stack and reaches ~50 by step 35 in at least
one run -- the reproduction question is answered. What is NOT established is the size, or
even the existence, of a method gap.

**What this vindicates.** The caveat attached to every arm this session -- *seed variance
has never been measured* -- was the right one, and it has now bitten the headline result
rather than a side finding. The estimated between-arm paired sd of ~5.3 was, if anything,
optimistic.

**What it changes about what to run next.** `--arm g2po` was queued to locate the gap
between estimator and harness. **That is premature: there may be no gap to locate.** The
first requirement is now replicates -- at minimum a second CCPO run at identical config,
so both methods have a spread rather than a point. Without that, no comparison on this
task means anything at ~10 points.

### [!!! RUNNING — decisive] H-AH. G2PO reproduces on our stack and is BEATING us outright

**Update at step 25: the gap is widening, not constant.**

```
step   G2PO   ours   diff   (ours reaches G2PO's value at step)
   5   10.2   10.2   +0.0      10
  10   21.1   10.9  +10.2      30
  15   27.3   17.2  +10.1      30
  20   28.9   21.1   +7.8      30
  25   42.2   21.1  +21.1      50
```

**G2PO at step 25 has reached what our best arm reached at step 50 -- half our budget.**
Windowed over steps 1-25: train +3.6, held-out +9.8. On this trajectory it passes our
FINAL 79.7 somewhere around step 45-55.

**CORRECTION at step 30: the gap is ~10 points and STABLE, not widening.**

```
diffs:  +0.0  +10.2  +10.1  +7.8  +21.1  +10.1
excluding step 5: mean +11.9, sd 5.3, MEDIAN +10.1
step 25's +21.1 is 1.8 sd above the mean -- an excursion
```

I called step 25 "widening" and extrapolated that G2PO would pass our final 79.7 by step
45-55. **Withdrawn.** The curves show +21.1 was OUR stall, not their spike: ours sat flat
at 21.1 across steps 20 and 25 before jumping to 30.5, while theirs actually dipped
42.2 -> 40.6. One point does not make a trend, and I built an extrapolation on it.

**The train gap is genuinely growing, though**: +0.3, +2.0, +3.6, +5.0 over successive
windows, against a held-out gap of +9.9. So the earlier reading -- that this was purely
an evaluation-temperature sharpening effect, since train was level -- is **half wrong**.
G2PO is learning faster *and* its advantage roughly doubles at T=0.4. Both are real.

**This is no longer "the published number transfers". It is "CCPO is materially worse
than G2PO on identical hardware, data, protocol and evaluation draw."** The comparison
is paired (same seed, same 128 games) and every configuration key was verified identical
beforehand.

**The harness is now nearly exonerated.** After removing the KL clamp (Qwen3-era, inert
here) and the length penalty (never enabled), and after showing the `global_seqlen`/
memory/timing differences are purely 2-vs-4 GPUs (total tokens ratio 1.004), the only
code difference left on the ALFWorld path is the two anchor mechanisms -- and those are
small next to a 21-point gap.

**So the remaining explanation is the estimator itself**, i.e. CCPO. The experiment that
confirms it is already built and one command away: `--arm g2po` runs their estimator,
vendored verbatim, inside OUR harness.

* tracks the reference -> the fault is CCPO's advantage
* tracks our arms -> something in our harness outside the diff

**The dissociation persists** (held-out gap 2.7x the train gap), which points at policy
sharpness: their advantage produces a policy that gains far more from the T=1.0 -> T=0.4
evaluation than ours does.

### [SETTLED — negative, and it strengthens H-AD/H-AE] H-AF. Our G2PO port is faithful

Before attributing the gap to the harness, the alternative had to be excluded: that we
mis-ported G2PO's algorithm. `core_ccpo.py` vendors two pieces of their code, and a
silent error in either would explain a great deal. Checked line by line against
`baselines/G2PO/g2po/core_g2po.py`.

**`g2po_node_values` vs `compute_group_aggregation_values`:**

| detail | G2PO | ours | |
|---|---|---|---|
| node key | `obs2idx` reset per task -> (task, anchor) | `(task, str(anchor_obs[i]))` | match |
| discount | `0.95 ** (len(traj) - step_count)` | `gamma ** (T - t)`, t 0-indexed | match |
| terminal exponent | last step gets **gamma^1**, not gamma^0 | same | match |
| aggregation | sum / len(members) | `acc[key] / cnt[key]` | match |
| terminals | -1 -> SUCCESS_REWARD, -2 -> 0 | `_TERM_OK`/`_TERM_BAD` | match |

The gamma^1-at-terminal detail is the off-by-one that would have been easiest to get
wrong and hardest to notice. It is right.

**`g2po_step_advantage` vs `compute_step_level_advantage`:**

| detail | G2PO | ours | |
|---|---|---|---|
| comp 1 std | `torch.std` = **unbiased, n-1** | `v.std(ddof=1)` | match |
| singleton nodes | skipped, left at 0 | `if len(ids) < 2: continue` | match |
| invalid penalty | comp 1 only, on successor value | `v_pen` in comp 1 only | match |
| comp 2 gain | **unpenalised** `next - current` | `v_raw - V(NODE)` | match |
| comp 2 scope | standardised across the task | same | match |

`torch.std` defaulting to n-1 while `np.std` defaults to n is the other trap here, and
the port uses `ddof=1`. The only divergence is a `len < 2` guard our comp 2 adds and
theirs lacks -- immaterial, since a task carries 8 rollouts x many steps.

**Why a negative result matters here.** H-O concluded that our advantage correlates
~0.90 with G2PO's and therefore the estimator is not the gap. **That conclusion rests
entirely on `g2po_step_advantage` being a faithful reference** -- a broken reference
would have made the correlation meaningless in either direction. It is faithful, so H-O
stands, and with the estimator, config (H-AB), eval protocol, training length (H-Z),
context conditioning and memory all excluded, the two anchor differences (H-AD, H-AE)
are what remains.

### [!!! RUNNING — `ccpo-anchor-20260909`, both flags] H-AG. Step-1 grouping confirms anchor_aff dominates

Relaunched after the container rebuild with **both** `obs_repair=1` and `anchor_aff=1`
(the earlier attempt had only the first, and never reached step 1). Config diff against
`ccpo-global-20260907`: those two flags are the only differences.

```
                base    +repair    +repair+aff
n_buckets        758       797         1172
singleton_frac  0.335     0.227        0.291
step-1 success  0.0547    0.0547      0.0547   <- identical: rollouts unchanged
```

**`anchor_aff` is the dominant mechanism, by an order of magnitude.** It adds **+375
nodes** over repair-only, against the repair's own +39 -- a 55% rise in node count over
base.

**That figure cross-checks against an independent measurement.** The `aff` field was
added on 2026-09-03 because "42.8% of observations map to >1 admissible set". If each
ambiguous observation splits in two, 758 x 0.428 ~ 324 extra nodes; we observe +375.
The grouping change matches the ambiguity rate measured a week earlier by different
means, which is good evidence the flag does what it claims rather than fragmenting
arbitrarily.

**The predicted tension resolved favourably.** Finer nodes push singletons UP
(0.227 -> 0.291) while the failure-anchor repair pushes them DOWN; the net 0.291 is
still **below base's 0.335**. So the pair delivers 55% more state resolution *and* fewer
occurrences lacking a leave-one-out baseline than the 79.7% arm -- a combination neither
flag reaches alone. (Repair-only had fewer singletons but almost no added resolution;
aff-only would have had resolution at a singleton cost.)

**Clean ablation confirmed:** step-1 `success_rate` is identical to base at 0.0547, so
the rollouts are the same and only credit assignment differs.

### [STOPPED at step 32] H-AG RESULT — the anchor fix did not convert

Stopped by user direction to free the cards for the G2PO reference run.

| step | anchor | base | diff |
|---|---|---|---|
| 5 | 9.4 | 10.2 | -0.8 |
| 10 | 8.6 | 10.9 | -2.3 |
| 15 | 14.1 | 17.2 | -3.1 |
| 20 | 15.6 | 21.1 | -5.5 |
| 25 | 11.7 | 21.1 | -9.4 |
| 30 | 20.3 | 30.5 | -10.2 |

mean paired difference **-5.21**, **6/6 negative**,
monotone. Train over steps 1-32: anchor 15.1 vs base 17.4
(-2.3).

**The step-1 diagnostics were unambiguously good and predicted nothing.** Node count rose
758 -> 1172 (+55%), the singleton fraction fell 0.335 -> 0.291, and the +375 nodes from
`anchor_aff` cross-validated against an independently measured 42.8% ambiguity rate.
Rollouts were identical at step 1, so the ablation was clean. None of it reached
held-out success.

**Lag, not obviously a ceiling.** Matching each anchor value to the base step that first
reached it gives lags of 0, 5, 0, 5, 10, 10 -- the arm ran ~10 steps behind and the lag
grew slowly. Whether it would have stabilised is **unresolved**: the run was stopped at
32, before the step-50 test that would have separated "slower" from "capped".

**This is the third structural improvement that did not convert**, after `effect_rel`
moving a hundredfold for -0.01 points and the digest's cost fix returning to baseline.
**Grouping-quality metrics have no demonstrated predictive value for held-out success on
this task.** That is now a strong enough prior to require an accuracy result before
believing any further grouping change.

**What it does NOT establish.** Seed variance has still never been measured. A -4 mean
difference sits inside the ~5.3 between-arm paired sd estimated earlier, so a single
seed cannot separate "harmful" from "unlucky". The honest verdict is *unconverted*, not
*refuted*.

### STEP-30 — 6/6 negative, but the right question is LAG vs CEILING

```
paired diff:  5:-0.8  10:-2.3  15:-3.1  20:-5.5  25:-9.4  30:-10.2
sign test 6/6 negative -> p = 0.031
```

The direction is now established. But the raw gap conflates two very different failures,
and the fix is to ask **which base step reached each anchor value**:

| anchor step | value | base step at that level | lag |
|---|---|---|---|
| 5 | 9.4 | 5 | 0 |
| 10 | 8.6 | 5 | 5 |
| 15 | 14.1 | 15 | 0 |
| 20 | 15.6 | 15 | 5 |
| 25 | 11.7 | 15 | 10 |
| 30 | 20.3 | 20 | 10 |

**The arm is ~10 steps behind, and the lag is growing slowly.** That is a materially
different diagnosis from "capped lower":

* **constant lag ~10** -> the arm is merely slower and would land near base's step-90
  value (~75) at step 100. Bad but not a refutation of the mechanism.
* **growing lag** -> a genuine ceiling. Extrapolating the current rate (10 steps of lag
  per 30) puts step 100 at base's step-67 level, near 50.

**Step-50 test, fixed now.** Base at 50 is 50.0; its step-40 value is 22.7.

* anchor@50 **>= ~40** -> lag shrinking or small; the mechanism is not harmful
* anchor@50 **~23** -> constant ~10-step lag; slower, not capped
* anchor@50 **< ~20** -> lag growing; **refuted**, and the conclusion is that
  credit-assignment structure is not where the remaining points live on this task

**Caveat that keeps this honest:** base's own curve has a step-to-step sd of 7.9 and
contains drops of -14.8 and -13.3 that it recovered from, and base's step-40 value (22.7)
is itself one of those dips. Read step 50 against the local trend, not a single base
point.

### STEP-20 RESULT — passes the kill rule, but 4/4 negative

```
paired diff (anchor - base), same eval draw at each step
  step  5   -0.8
  step 10   -2.3
  step 15   -3.1
  step 20   -5.5      mean -2.93, monotone
  train window 11-20  -3.7
```

**Kill rule PASSES** (15.6% against a ~11% floor), so the arm runs to 50.

**But the per-evaluation "within noise" labels understate this.** Those bands treat each
evaluation as two independent 128-episode samples; the arms are PAIRED -- same seed, same
game sequence at each step -- so the differences are the right unit. Four of four are
negative, the magnitude grows monotonically, and the train window agrees at -3.7.

**Why this is still not a verdict.** The four differences are not independent (they track
the same two evolving policies), a sign test gives p=0.125, and **seed variance has never
been measured** -- the between-arm paired sd was estimated at ~5.3, so -2.9 sits well
inside what a single seed could produce. Base was at 21.1% here and reached 79.7%, and
early-trajectory calls have been wrong four times in this project.

**What it costs to keep going: nothing.** GPUs 4-5 carry another tenant and host RAM will
not fit a second arm, so these cards have no alternative use.

**The pattern worth naming.** This is the THIRD clean structural improvement that has not
converted: `effect_rel` a hundredfold in the hard-gate comparison (-0.01 points), the
digest's cost fix (back to baseline, no gain), and now a 55% resolution increase with
fewer singletons, cross-validated against an independent 42.8% ambiguity measurement,
producing nothing positive over 20 steps. **Grouping-quality metrics have zero
demonstrated predictive value for held-out success on this task.** Any future arm
justified by them should carry that prior.

**Still to prove.** Grouping quality has failed to translate into accuracy once already:
`ccpo-hardedge` vs `ccpo-global` moved `effect_rel` a hundredfold for **-0.01 points**
of held-out success. Judge this arm at steps 20 / 50 / 100 against base's
21.1 / 50.0 / 79.7.

### [!!] H-AE. `aff` was built to fix anchor ambiguity and never wired into the grouping

Continuing the harness diff past the three files of H-AD, across the whole
`agent_system` tree:

```
environments/env_manager.py      the anchor repair (H-AD) + the item below
memory/memory.py                 fetch_sep -- DEAD CODE, zero call sites in their tree
environments/prompts/appworld.py irrelevant (different environment)
multi_turn_rollout/rollout_loop.py  enable_thinking only -- verified no-op on Qwen2.5
```

`memory.py` and `rollout_loop.py` are eliminated. What remains is a **second** anchor
difference, and it may matter more than the first:

```python
# G2PO, env_manager.py, before the repair runs:
text_obs[i] += f"\nAdmissible actions:\n {reformatted_admissible_actions}\n"
```

**G2PO's node identity is observation text PLUS the admissible-action list. Ours is the
observation text alone** -- `key = (task, str(anchor_obs[i]))`, core_ccpo.py:409.

**We already knew this was a problem and already built the fix.** The `aff` field was
added to this tree on 2026-09-03 with the comment "42.8% of observations map to >1
admissible set". It is threaded from `env_manager` through `anchor_aff` in the batch to
`aff_labels` in `ccpo_step_advantage` -- and there it is used at exactly one place,
core_ccpo.py:795, **to write a column into the diagnostic CSV.** It never enters a node
key. We measured the ambiguity, built the disambiguator, and then grouped without it.

On our own measurement that leaves **42.8% of observations** conflating states that
G2PO separates -- a larger population than the anchor repair touches.

**Implemented as `ACG_ANCHOR_AFF`** (default 0), ordered to match G2PO exactly: the
append happens BEFORE the history append and before the repair tests, both of which are
`startswith()` and so survive a trailing append. G2PO's reset seeds `history_obs` from
`full_text_obs` while its step stores obs+admissible; that inconsistency is reproduced
deliberately (it only ever reaches `history[-2]` on the first step).

**`tests/test_obs_repair.py` now drives the REAL `AlfWorldEnvironmentManager.reset/step`
against a stub env.** That was added because the first version of this change had
`self.history_obs = [[o] for o in full_text_obs]` placed BEFORE `full_text_obs` was
assigned -- it parsed cleanly and would have raised `NameError` eight minutes into a GPU
run. The live-path test catches that class of bug in five seconds.

**Revised next arm: BOTH flags.** `ccpo-anchor` ran with `obs_repair=1` only. To match
G2PO's harness the arm needs `obs_repair=1 anchor_aff=1`. Ablate the halves later, only
if the pair moves the number.

### [!!! RUNNING — `ccpo-anchor-20260909`] H-AD. G2PO repairs the anchor; we never did

H-AB closed off config, eval protocol, split, training length, context conditioning,
memory and the estimator, leaving "a harness difference outside those keys" as one of
two survivors. **That did not need a training run to check -- both trees are on disk.**

Diffed our harness against the G2PO checkout:

```
agent_system/environments/prompts/alfworld.py        0 differing lines  (identical)
agent_system/environments/env_manager.py           138 differing lines
agent_system/multi_turn_rollout/rollout_loop.py     86 differing lines
```

The prompt template is byte-identical. Inside `env_manager.py` sits a mechanism G2PO
has and **we do not, and never did** (`git log -S` finds nothing; `baselines/verl-agent`,
i.e. GiGPO's original, does not have it either -- so this is a G2PO harness
CONTRIBUTION, not something we deleted):

```python
PATTERNS = ["You heat", "You cool", "You clean", "You turn on"]
if not text_obs[i].startswith("Nothing happens"):
    self.history_obs[i].append(text_obs[i])
is_invisible_obs = len(self.history_obs[i]) >= 2 and any(
    self.history_obs[i][-2].startswith(p) for p in PATTERNS)
text_obs[i] = " ".join(self.history_obs[i][-2:]) if is_invisible_obs else self.history_obs[i][-1]
```

**Critically, this is not a prompt change.** `full_text_obs` -- the prompt -- is built
before this block and is untouched. What it rewrites is **`anchor`**, which in our tree
feeds `anchor_obs` -> `ccpo_step_advantage` and `derive_context`: it is the STATE
IDENTITY used for node grouping in the step term. A wrong anchor means occurrences that
are in different states share a baseline.

**Two independent repairs, and they are not equally supported.**

* **(B) a failed action does not advance the anchor.** Without it our anchor becomes the
  literal `"Nothing happens."` on every failure, so unrelated failed states anywhere in
  the batch collapse to ONE anchor. Applies to every task type.
* **(A) invisible-state concatenation.** ALFWorld does not restate the result of
  heat/cool/clean/turn-on, so "egg heated" and "egg not heated" carry identical anchor
  text. Task-type specific.

**Honest check that partly undercuts (A).** The obvious test is whether our per-type
deficits fall on the four pattern-affected types. **They do not:**

```
pattern types    (heat/cool/clean/turn-on):  87.1, 78.2, 80.4, 65.6  -> mean 77.8
non-pattern      (pick_and_place, two_obj):  87.6, 69.6              -> mean 78.6
```

Essentially equal. I first read the correspondence as striking -- worst type
`look_at_obj_in_light` is the "You turn on" case -- but it does not survive averaging.
**(B) is untested by this comparison** (it affects all types alike) and is the more
plausible of the two.

**Why this is still the best lead.** It is the ONLY substantive harness difference
between our tree and the reference implementation that produced 95.0, it sits directly
on the grouping our step term depends on, and our own code already flags anchor
conflation as a known problem -- the `aff` field was added because "42.8% of
observations map to >1 admissible set". `aff` cannot fix either repair: admissible
actions are the same before and after heating, and the same at every failure.

**Implemented behind `ACG_OBS_REPAIR` (default 0, so every prior run stays
reproducible).** `tests/test_obs_repair.py` transcribes G2PO's loop literally and checks
ours against it: **matches on 9/9 steps.** On the fixture, 2 of 9 raw anchors are the
bare failure string (collapsing batch-wide to one) and the post-heat anchor is
byte-identical to an earlier pre-heat one; after repair, neither holds.

**Next arm: `ccpo-global` + `obs_repair=1`, 100 steps, everything else identical to the
79.7% configuration.** This is a single-flag ablation against the best result, and
unlike the memory arms it is not a new idea -- it is adopting a mechanism from the
implementation we are trying to match.

**LAUNCHED as `ccpo-anchor-20260909`** on GPUs 0-3. Config diffed against the 79.7% arm:
`obs_repair` is the ONLY real difference (`compact_stall=0` is cosmetic -- the digest is
off entirely at `compact_budget=0`).

**Step-1 falsification, before any success number.** The repair rewrites anchors, and
anchors determine node grouping, so the grouping structure MUST change. Base at step 1:
`n_buckets` 758, `bucket_size_mean` 8.44, `bucket_singleton_frac` 0.335. **If
`n_buckets` comes back at 758 the flag is not taking effect** and the arm is
meaningless -- stop it rather than run it for its success rate.

Expected direction if the repair works as reasoned: **more** buckets and a **higher**
singleton fraction, because states that previously collapsed onto the shared
`"Nothing happens."` anchor now carry the last real observation instead, and post-heat
states separate from pre-heat ones.

**STEP-1 RESULT: the repair is active, and my predicted DIRECTION was half wrong.**

| metric | base | anchor | delta |
|---|---|---|---|
| `n_buckets` | 758 | 797 | **+39** |
| `bucket_size_mean` | 8.44 | 8.03 | -0.41 |
| `bucket_size_p90` | 16 | 20 | +4 |
| `bucket_singleton_frac` | 0.335 | **0.227** | **-0.108** |
| `effect_rel` | 0.0945 | 0.1159 | +0.021 |
| `episode/success_rate` | 0.0547 | 0.0547 | **0.000** |
| `episode/valid_action_ratio` | 0.8431 | 0.8422 | -0.001 |

I predicted more buckets **and a higher singleton fraction**. Buckets rose, but
singletons **fell by 10.8 points** -- and the reason identifies which repair is doing
the work:

* **(B) drives the singleton drop.** A failed action at state S used to be anchored to
  the shared string `"Nothing happens."`; now it carries S's last real observation and
  joins S's node. States that occurred once therefore also collect their failure turns
  and stop being singletons. `bucket_size_p90` rising 16->20 is the same effect.
* **(A) drives the +39 nodes**, separating post-heat/cool/clean/turn-on states from the
  pre-action states they were byte-identical to.

**Why the singleton drop is the substantive part.** A singleton node has NO leave-one-out
baseline -- with one occurrence there is nothing to leave out. 0.335 -> 0.227 moves
about **11% of all occurrences from "no usable baseline" to "has neighbours."** That is
a direct increase in the estimator's coverage, and it is the mechanism by which this
could actually help.

**Clean ablation confirmed.** `success_rate` is identical at step 1 (0.0547 both) and
`valid_action_ratio` matches to 0.001, so the rollouts are the same and ONLY credit
assignment differs.

**The caution that must stay attached.** Structural improvement is necessary, not
sufficient, and this exact inference has failed before: `ccpo-hardedge` vs `ccpo-global`
moved `effect_rel` from 0.001 to 0.119 -- a hundredfold -- and changed held-out success
by **-0.01 points**. A better-looking grouping has already once bought nothing. Judge
this arm on held-out success at steps 20/50/100, not on its grouping metrics.

**Kill rule, fixed in advance.** This arm is an ablation against the best result rather
than a speculative lever, so the bar is different from the memory arms: run to 50 unless
held-out at step 20 is more than 2 SE (about 10 points) BELOW base's 21.1%, i.e. below
~11%. Judge it at step 50 against base's 50.0%, and at 100 against 79.7%.


**Status board.** Entries below carry results as often as they carry questions; this
says which is which, so the section can be read without opening every one.

| | what it is | state |
|---|---|---|
| **H-AC** | `ccpo-cheapmem`, digest at 192 tok / replace mode | **running** |
| **H-S / H-X** | epistemic weighting `w_u = J_u/(J_u+c)` — the ONE untested uncertainty variant | **open** |
| **H-AB** | G2PO on our harness — the only remaining way to attribute 79.7% | **open, declined twice** |
| H-Z, H-AA, H-V, H-W, H-Y, H-Y' | | resolved, kept here for the reasoning |

**Two cheap unrun checks**, neither needing a training run:
* eval-only pass on `ccpo-global` step-100 vs step-125/135 checkpoints **on the same
  draw** — settles the ceiling free of the resume-draw confound
* a second seed of `ccpo-global` — every number in this file, the headline 79.7%
  included, rests on a single seed

**Why localstd's negative result does not close the uncertainty question.** `localstd`
divided A_CC by sigma, damping HIGH-VARIANCE neighbourhoods. H-S weights by reliability,
damping POORLY-SAMPLED ones. The two disagree precisely where variance and support are
both high, so -2.08 on the first says nothing about the second.


### [RESULT — passes its kill rule, but the mechanism is null] H-AC. `ccpo-cheapmem` — the digest at a twentieth of the cost

H-AA established that the digest's cost is real and its stated mechanism is not. H-Y's
per-type split, pre-registered, still put both memory-demanding types at ranks 1 and 2
of 6 (p = 1/15 = 0.067). So the lever is kept and the cost is cut, rather than the same
arm being re-run.

Two changes, both measured offline first (`tests/test_compact_mode.py`):

* **`ACG_COMPACT_MODE=replace`.** The digest substitutes the recent window instead of
  stacking on it. `build_digest` already covers those turns, so `prepend` shipped them
  twice. **This alone is NOT the fix** -- an earlier version of this entry claimed it
  was. The window is ~88 tokens against a 512-budget digest's ~500, so replace reclaims
  only about a quarter of the overhead.
* **Budget 512 -> 192.** This is the real lever, and the sweep is why:

| budget | digest tok | informative lines | avg overhead | % of base prompt |
|---|---|---|---|---|
| 128 | 123 | 3/15 | +13 | 2.7% |
| **192** | **184** | **5/15** | **+35** | **7.4%** |
| 256 | 252 | 7/15 | +59 | 12.7% |
| 512 | 494 | 14/15 | +146 | 31.4% |

512 more than doubles the prompt whenever it fires, which is what `ccpo-memory` and
`ccpo-gatedmem` both paid. 192 keeps a third of the informative lines at a fifth of the
cost.

**Isolation.** Diffed against `ccpo-global-20260907`: the only real differences are
`compact_budget` 0->192, `compact_mode` ->replace, `compact_stall` ->2. (`ccpo_std`,
`ccpo_std_floor`, `ccpo_step_norm` read as differences only because they are now written
explicitly at what were already the code defaults -- `task`, 0.25, `mode`.)

**Falsifiable at step 1, before any success number.** The offline sweep predicts
`prompt_length/mean` ~500 against base 466 and gatedmem's 656. If step 1 lands much
above ~555, the sweep's model of the cost is wrong and the arm should be stopped
immediately rather than run for its success rate.

**RESULT: 544 at step 1 — passes the threshold, but the point prediction was 2.2x off,
and the reason matters.** Predicted +35 overhead, measured +78. The cost identity is
`overhead = P(fire) x (digest - window) = P(fire) x (184 - 88) = P(fire) x 96`, so +78
implies the gate firing on **~81%** of turns, not the 36% the sweep assumed. That 36%
was measured on a TRAINED policy; at step 1 the policy is untrained, stalls constantly,
and trips the >=2-turn gate far more often. **The cost model is sound; it was fed a
firing rate from the wrong regime.**

Follow-on prediction, falsifiable across the run: as training reduces stalling,
`prompt_length/mean` should DECLINE toward +35. `ccpo-gatedmem` is the control -- its
prompt length stayed flat at 648-695 across all 20 steps, because at a 512 budget the
digest dominates however often it fires.

Cost reduction actually achieved so far: **+78 against gatedmem's +190, i.e. 2.4x, not
the 5.5x claimed above.**

**Kill rule, fixed in advance, same form as H-Y's:** stop at step 20 if train mean over
steps 10-20 is below 0.130 (base 0.160), or held-out at step 20 is below 15%.
Continue only on the pre-registered per-type concentration, never on overall success
alone.

### RESULT at step 20 — the cost fix worked, the memory hypothesis did not

```
1. train mean 10-20   cheapmem 0.148   base 0.160   gatedmem 0.107   PASS (>0.130)
2. held-out @20       cheapmem 16.4%   base 21.1%   gatedmem 15.6%   PASS (>15%)
3. per-type pooled    memory +1.5      others -0.1                   "satisfied"
```

**What genuinely improved: the cost.** Train success over steps 10-20 is 0.148 against
base's 0.160, where BOTH earlier digest arms sat at ~0.107. Cutting the budget 512->192
and replacing the window instead of prepending removed essentially all of the digest's
drag. That part of H-AA's diagnosis was correct and the fix worked.

**What did not improve: anything the memory story predicts.** Pooled over four
evaluations:

| type | s5 | s10 | s15 | s20 | mean |
|---|---|---|---|---|---|
| *look_at_obj_in_light | +7 | +10 | -7 | +6 | **+3.9** |
| *pick_two_obj_and_place | +0 | +0 | +0 | -4 | **-0.9** |
| four non-memory types | | | | | **-0.1** |

The continue condition is technically met (+1.5 vs -0.1) but **the margin is 1.6 points
against a pooled per-type SE of ~5**, and it rests entirely on `look_at_obj_in_light`,
the type with the LARGEST SE (6.8). `pick_two_obj_and_place` -- the tightest measured
deficit and the clearest memory demand -- is **-0.9 pooled and exactly +0 at three of
four evaluations.** The step-20 "satisfied" reading is produced by the non-memory types
drawing badly (cool and heat both 0% against base's 16-17%), not by memory types gaining.

**The honest summary: cheapmem's achievement is being *not worse* than base.** Held-out
at step 20 is 16.4% against base's 21.1% -- still behind. Fixing the cost of an
intervention that does not help returns you to baseline; it does not move you toward 95.

**Three arms have now been spent on the digest** (`ccpo-memory`, `ccpo-gatedmem`,
`ccpo-cheapmem`). The lever is characterised: its cost is real and now removable, its
benefit is not measurable at this scale. Recommend NOT resuming to 100 unless the
alternative uses are worse -- the kill rule permits continuation, it does not mandate it.

**Prompt-overhead prediction: REFUTED.** I predicted the +78 step-1 overhead came from
the untrained policy stalling more often than the sweep's 36%, and would therefore
decline toward +35. Over 20 steps it never moved: +79 +79 +82 +85 +90 +78 +84 +81 +80
+85 ... +81. Either stalling does not fall over this range, or the firing rate was never
the explanation.


### [OPEN — blocked on a decision] H-AB. Our config matches G2PO's published script exactly — the gap is not setup

Diffed `experiments/ccpo-global-20260907/outputs/resolved_config.json` against
`baselines/G2PO/examples/g2po_trainer/run_alfworld.sh`, the script that produced the
95.0 number. **Every substantive hyperparameter is identical:**

```
lr 1e-6              kl_loss_coef 0.01        kl_loss_type low_var_kl
gamma 0.95           use_kl_in_reward False   use_invalid_action_penalty True (coef 0.1)
max_steps 50         group_size 8             train_batch_size 16
max_prompt_length 2048   max_response_length 512
val temperature 0.4  val do_sample True       test_freq 5
ppo_mini_batch_size 256  ppo_micro_batch_size_per_gpu 32
eval split: eval_in_distribution (both)
```

The 16 "differing" keys are paths, run names, logger, `n_gpus_per_node` (8 vs 4),
`tensor_model_parallel_size`, `gpu_memory_utilization`, and the estimator itself.

**`data.val_batch_size` 128 vs our 64 is not a real difference.** `test.parquet` holds
128 rows in both; ours is chunked into 2 batches and both are evaluated
(`val/success_rate` = 51/64 = 102/128 exactly). Same 128 tasks, same split, same
sampling temperature. **Our held-out numbers are directly comparable to the published
ones, and n really is 128.**

**What this closes.** The 15-point gap to G2PO is now excluded from: configuration,
evaluation protocol, evaluation split, training length (H-Z), context conditioning
(H-H/H-Q/ablation), memory (H-Y/H-AA), and the advantage estimator itself in the sense
H-O measured (r ~ 0.90 with G2PO's). Those were the candidates.

**What remains, and it is now the whole question.** Either (i) something in our harness
outside these config keys differs from the G2PO checkout -- prompt template, thinking
format, action parsing, reward shaping -- or (ii) the published 95.0 does not reproduce
on this hardware/stack. Both are answered by exactly one experiment: **run
`algorithm.adv_estimator=g2po` on OUR harness at these settings.**

That experiment has been declined twice, on the reasoning that baseline numbers are
already known. That reasoning was sound when method tuning still had untried levers.
It no longer holds: every lever has been tried and the config is now proven identical,
so a G2PO arm is no longer a baseline-reproduction exercise -- it is the only remaining
way to attribute our 79.7. If G2PO scores ~80 here, 79.7 is competitive and the gap is
environmental. If it scores ~95, the gap is in code we can diff line by line, since both
trees are on disk.

**Note on the per-type numbers.** Per-type rates are aggregated as a MEAN OF THE TWO
BATCH-LEVEL RATES, not as a pooled proportion (e.g. pick_clean 0.759524 = mean of 13/21
and 9/10). When the two batches hold different numbers of a type, that unweighted mean
is a slightly biased estimate of the pooled rate. It does not affect the overall number
and is too small to move H-Y's conclusions, but per-type values should not be treated as
exact proportions over 128.


### [RESOLVED] H-AA. The digest's stated mechanism is refuted, paired, at step 15

`ccpo-gatedmem` and `ccpo-global` share `env.seed=0`, `train_batch_size=16` and
`env.rollout.n=8`, so they see the **same training-task sequence** and can be compared
step-by-step as paired samples. Over the 15 matched steps so far (exact sign test):

| metric | mean diff | gatedmem higher | p |
|---|---|---|---|
| `response_length/mean` | **+6.32 tok** | **15/15** | 6.1e-5 |
| `prompt_length/mean` | +194.3 tok | 15/15 | 6.1e-5 |
| `actor/kl_loss` | **-0.009** | **0/15** | 6.1e-5 |
| `episode/success_rate` | -0.024 | 4/15 | 0.12 |

**The response-length result refutes the digest's justification.** `compact.py` motivates
the digest as going "in the prompt, so the agent stops re-deriving state inside
`<think>`" -- which predicts SHORTER responses. Responses are longer, at every single
step, by ~8%. The digest does not replace the agent's own re-derivation; it is read
*in addition to* it. So it costs ~194 prompt tokens and ~6 response tokens, and buys no
reduction anywhere. **The premise in the module docstring is wrong as written and should
be corrected rather than restated.**

**Lower KL at 15 of 15 steps is the learning-rate signature.** The digest arm stays
closer to the reference policy at every step. Less divergence from reference is less
policy movement per step, which is exactly the slower-learning pattern the ungated arm
showed (train 0.108 vs 0.160 over steps 10-22). It suggests the digest suppresses the
update rather than corrupting it -- consistent with the ungated arm eventually reaching
0.9997 valid-action ratio while still trailing on success.

**What is NOT yet established.** Success rate is -0.024 at 4/15, p=0.12 -- the direction
matches the ungated arm but it does not clear significance, and held-out is within noise
at every evaluation so far (step 5 -0.8+/-7.4, step 10 +0.0+/-7.8, step 15 -3.9+/-9.0).
No instability: grad norm 0.890 vs base 1.700, clipfrac 0.003, and
`prompt_length/clip_ratio` is 0.000, so nothing is being truncated.

**This does not by itself trigger the kill rule**, which keys on the steps 10-20 train
mean and the step-20 held-out per-type split. But it removes the mechanism that
motivated the arm: whatever the digest does, it is not saving the agent from
re-deriving state.


### [RESOLVED] H-Y. The digest never had a format problem

I proposed that the deciding diagnostic for the digest arm was whether
`valid_action_ratio` recovers to >=0.999, reasoning that a recovery with flat success
would isolate the digest's *content* as unhelpful rather than its *cost*. **That test
was already passed by the arm it was meant to explain, and I had not checked.**

| step | base (79.7% arm) | ccpo-memory (failed) |
|---|---|---|
| 1 | 0.8431 | 0.7864 |
| 5 | 0.9986 | 0.9977 |
| 10 | 0.9992 | 0.9995 |
| 22 | 0.9991 | **0.9997** |

Both arms start near 0.81 (untrained policy) and both are at 0.999 by step 5. The
failed arm ends *above* base. There was never a format-compliance failure to recover
from, so the diagnostic could not discriminate anything.

**Where the harm actually is.** On TRAINING episodes, not just held-out draws:

```
train success, steps 10-22:   base 0.160   memory 0.108   (-0.053)
memory behind at 18 of 22 individual steps
```

It learns slower, full stop. The one clear difference is prompt budget: 745-806 tokens
against base's ~470.

**Bearing on `ccpo-gatedmem`.** The stall gate cuts step-1 prompt length to 656 (base
466, ungated 774) -- roughly a third of the overhead removed, so about two thirds of the
suspected cost remains. Against that, the gate is not merely a smaller digest but a
*conditional* one: it fires only after >=2 turns with no new observation, i.e. only when
the agent is looping, whereas the failed arm showed a digest even when the current
observation sufficed. That is a mechanistic difference and is the only reason to expect
a different outcome. Prior stays at 30%.

**Kill rule, fixed now so it cannot be fitted to the result.** Stop the arm at step 20 if
either holds:

1. train success over steps 10-20 is below base's 0.160 by more than 0.03 (i.e. it
   reproduces the failed arm's deficit rather than closing it), **or**
2. held-out at step 20 is below 15% (base: 21.1%).

Continue to 50 only if the step-20 per-type split shows the pre-registered concentration
on `pick_two_obj_and_place` / `look_at_obj_in_light` (H-Y'). Overall success alone is not
sufficient grounds to continue -- at n=128 its SE is ~3.6 points, which cannot separate
the hypotheses.

Measured cost: 460 s/step, so step 20 is ~2.6 h and step 50 ~6.4 h.


### [RESOLVED — prediction held, p=0.067] H-Y'. Where the 15 points actually are

Pooling the nine converged evaluations of `ccpo-global-ext` (steps 105–145) gives
per-type numbers with 3x less noise than any single draw:

| task type | mean | sd | SE | shortfall vs 95 |
|---|---|---|---|---|
| look_at_obj_in_light | 65.6 | 20.4 | 6.8 | −29.4 |
| **pick_two_obj_and_place** | **69.6** | 6.4 | **2.1** | −25.4 |
| pick_cool_then_place_in_recep | 78.2 | 7.7 | 2.6 | −16.8 |
| pick_clean_then_place_in_recep | 80.4 | 7.7 | 2.6 | −14.6 |
| pick_heat_then_place_in_recep | 87.1 | 8.5 | 2.8 | −7.9 |
| pick_and_place | 87.6 | 7.9 | 2.6 | −7.4 |
| OVERALL | 79.7 | 4.7 | 1.6 | −15.3 |

**The gap is not spread evenly.** Two types are already near 87 and account for almost
none of it. `pick_two_obj_and_place` at 69.6 with SE 2.1 is the tightest and most
trustworthy deficit; `look_at_obj_in_light` is lower still but its SE of 6.8 makes it
suggestive rather than established.

**Why this is a prediction and not a post-hoc read.** Those two types are precisely the
ones whose failure mode is *forgetting what you already did*: two-object placement needs
to know which instance has already been moved, and lamp-search needs to know which
receptacles have already been opened. Everything else is a single fetch-and-transform
where the current observation suffices — which is also why context conditioning measured
null on the pooled set (H-H, H-Q, the ON/OFF ablation).

**Registered before `ccpo-gatedmem` reports:**

* If the digest works through the mechanism claimed, the gain must be
  **concentrated in `pick_two_obj_and_place` and `look_at_obj_in_light`**, and roughly
  absent on `pick_and_place`/`pick_heat`, which have little room and no memory demand.
* A *uniform* lift across all six types would falsify the mechanism even if the overall
  number improves — that pattern means the digest is acting as generic prompt
  scaffolding, not as memory.
* A drop on the two already-strong types with a rise on the two weak ones is the
  **cost** signature: the digest is buying memory by spending prompt budget.
* Overall success is the *weakest* of these readouts. Judge the arm on the per-type
  pattern, and only on types where SE is small enough to carry a claim.

**This also reframes the target.** Reaching 95 overall does not require +15 everywhere;
it requires closing ~25 points on two types while holding the other four. That is a
narrower and more tractable problem than "the method underperforms," and it is the first
result that points at a specific capability rather than at the estimator.


### [RESOLVED] H-Z. **150 steps buys nothing** — the 100-step ceiling, and a warning about best-checkpoint selection

`ccpo-global-ext-20260908` warm-started the 79.7% arm at step 100 and ran to 145,
where early stopping fired. Nine evaluations:

```
85.9  79.7  77.3  75.8  78.1  78.9  84.4  71.1  85.9

mean 79.69   sd 4.97   range 71.1 – 85.9
step-100 baseline        79.69
difference               −0.00
trend across 50 steps    −0.14 pts/5 steps
```

**The mean over 50 additional steps equals the step-100 value to two decimals.** The
curve had already converged; the apparent headroom at step 100 (a fitted tail slope of
+2.19 pts/5 steps) was noise in the last few evaluations, not a trend.

**Consequence for the GiGPO comparison.** GiGPO reports 86.7 at *its* 150-iteration
protocol. We now have a 150-step number and it is ~80. **The 7-point gap is not a
training-length gap** — it is method or harness, and H-O already located it outside
the estimator.

### The best-checkpoint trap, stated plainly

`best.json` records **85.9% at step 105**, and it would be tempting to quote that.
It should not be quoted. With sd ≈ 4.97 over 9 draws, the expected maximum sits about
1.5 sd above the mean — **79.69 + 7.5 ≈ 87**. We observed 85.9.

**The "best checkpoint" is the luckiest draw, not a better policy.** Selecting the max
of a noisy series is selecting noise, and it is systematically biased upward by roughly
1.5 sd whenever the evaluation is this variable.

This matters beyond this run: **every `stepN-best` checkpoint in this project is
selected the same way.** Report the mean over evaluations, or re-evaluate the chosen
checkpoint on a fresh draw — never the selecting maximum.

**Caveat that does not change the conclusion.** This run predates the
validation-draw alignment fix, so its evaluations replay the *early* draw sequence.
That biases the comparison against the step-100 number in an unknown direction — but
the extension's *internal* trend (−0.14 pts/5 steps over nine of its own evaluations,
all on the same replayed sequence) is unaffected by the offset, and it is flat.


### [OPEN — untested] H-X. **"Uncertainty" was never one dial** — aleatoric vs epistemic, and only one is untested

Every uncertainty result in this file has conflated two different quantities:

| | asks | where it enters | measured |
|---|---|---|---|
| **aleatoric** — σ | how much do outcomes vary *here*? | `ccpo-localstd` divides A_CC by σ | **−2.08 paired, trending harmful** |
| **epistemic** — n_eff, J | how well do I *know* the mean? | H-S weights A_CC by reliability | **untested** |

`ccpo-localstd` divides by the neighbourhood's spread, so it **damps high-variance
neighbourhoods** — plausibly the states that most discriminate good actions from bad.
That is a coherent mechanism for a mildly harmful result rather than a neutral one.

H-S weights by *reliability* (`w_u = J_u/(J_u+c)`), damping **poorly-sampled**
neighbourhoods instead — a different target, measured at 1.36× reliability difference
between J≤3 and J≥6. **The two rules disagree exactly where variance is high AND
support is high**: localstd shrinks those, H-S keeps them.

**So `localstd` trending negative does not refute H-S.** What it refutes is the
framing carried through most of this session — that uncertainty is a single dial that
either helps or does not.

**What to abandon:** uncertainty about *whether to trust φ's neighbourhood*. Settled
four ways — τ² ≈ 4e-6, λ = 0.011 whenever free, H-Q showing φ-weighting inside a
bucket is actively worse than uniform, and the on/off ablation at −0.01. There is
nothing to be uncertain about because there is no signal.

**What may still be worth one arm:** epistemic weighting (H-S). Predicted effect
remains small; it is variance reduction on the ~13% of occurrences with J≤3.

---

### [RESOLVED — ran as ccpo-gatedmem, killed at step 20] H-Y. **Uncertainty-gated memory** — spend the digest only where the agent is lost

The memory arm (H-W) failed with a *diagnosis*, not a verdict: `valid_action_ratio`
0.9869 against ≥0.999 elsewhere, and it started degraded **at step 1** — before
training could adapt — so the ~300-token digest was costing output format from the
first rollout. **The digest's value was never disproven; its cost was identified.**

That cost is paid every turn while the benefit only exists when the agent is
repeating itself. Measured on `gate-probe-20260907`:

| gate | fires on | mean V(next) there |
|---|---|---|
| always (what H-W ran) | 100.0% | 2.896 |
| `revisit_count > 0` | 54.3% | 2.306 |
| **`stall ≥ 2`** | **36.0%** | **1.964** |
| `stall ≥ 5` | 14.4% | 1.330 |

Gating on `stall ≥ 2` cuts exposure to ~a third, and the turns it selects have **mean
V(next) 1.96 against 2.90 overall** — it fires precisely on the turns going badly,
which is the population the digest exists to rescue. The untouched 64% keep the
reference protocol exactly.

`stall` is already computed in `derive_context()` (steps since the last new
observation) and needs no plumbing. Implementation is a condition on the
`ACG_COMPACT_BUDGET` block in `env_manager.py`.

**Why this ranks above H-S:** it fixes a defect we measured rather than adding a
mechanism, and it has a measured targeting signal. H-S has a real quantity but a
predicted effect too small for one arm to resolve against a paired sd of 5.2.

**Both held** until `ccpo-localstd` finishes — it is at step 45 with the comparator's
decisive stretch (65–100) still ahead, and stopping on t = −1.15 would repeat the
step-30 misread.


### [RESOLVED — superseded by H-AA/H-AC] H-W. The memory digest is **hurting** at 20 steps, with an identified cost mechanism

`ccpo-memory-20260907` — the digest arm, a single config change (`compact_budget`
0 → 512) from the 79.7% run. Interim at step 20 of 100:

| step | memory | no-memory | delta |
|---|---|---|---|
| 5 | 8.6 | 10.2 | −1.6 |
| 10 | 8.6 | 10.9 | −2.3 |
| 15 | 12.5 | 17.2 | −4.7 |
| 20 | **9.4** | **21.1** | **−11.7** |

**Four for four, widening monotonically, mean −5.08.** More telling than the gap is
the *internal* trend: memory **+0.62 pts/5 steps** against the comparator's **+3.90**.
The memory arm has gone 8.6 → 9.4 across 20 steps while the comparator doubled. It is
not merely behind — it is barely learning.

This is qualitatively different from the first three readings, which I correctly
declined to read as signal. Four consecutive negatives with a monotone widening *and*
a flat internal trend is a pattern; three noisy points were not.

**Identified cost mechanism:** `valid_action_ratio` is **0.9869**, against ≥0.999 in
every other arm — roughly 1.3% of actions failing to parse. Consistent with the
~300-token digest (prompt 796 vs ~496) crowding the prompt and degrading output
format. **The digest is costing action validity, not just tokens.** That is a
concrete, fixable defect rather than a verdict on the idea: a shorter budget, or
placing the digest after the observation rather than before it, would test whether
the format cost is separable from the memory content.

**Not refuted.** 20 steps is short, the comparator's own curve was flat until step 15,
and the digest could still pay off over a longer horizon. But it is the
weakest-performing arm at this stage, it has a diagnosable defect, and it is occupying
GPUs that H-V — the best-supported open idea — is waiting on.

**H-R note.** `phi_rel_corr` is +0.0067 here against +0.0113 in the no-memory arm.
Consistent with the digest changing φ's neighbourhoods as H-R predicted it might,
though n is still too small to call. If the arm is paused this stays unresolved.

**Recommendation: pause at the step-20 checkpoint and pivot to H-V.** `step20-best`
preserves the question for a resume.


### [RESOLVED] H-V. **Coherent standardisation** — RESULT: neutral-to-mildly-harmful, 16 paired evaluations

**The question that produced it** (user, 2026-09-08): *can we standardise using our
context-conditioned grouping rather than G²PO's?* I had been framing the choice as
"our grouping vs theirs" for *membership*, and treating the scale as a separate
decision between per-task and per-node. The real option is to use the kernel for
**both** jobs.

We already compute a φ-weighted *mean* over the soft neighbourhood. The φ-weighted
*variance* over the same neighbourhood was simply never computed:

```
b_u = Σ w_uv · target_v / Σ w_uv                          (have)
σ_u = sqrt( Σ w_uv (target_v − b_u)² / Σ w_uv )           (missing)
A_CC = (target_u − b_u) / σ_u
```

**Measured** (`gate-probe-20260907`, n=6,912, soft global neighbourhoods, τ=0.15),
split-half reliability of the step advantage:

| scale | reliability |
|---|---|
| unstandardised | 0.8299 |
| task-level sd (**what we ship now**) | 0.8446 |
| per-node sd (**G²PO's choice**) | 0.8968 |
| **CONTROL: σ shuffled within task** | 0.8860 |
| **φ-weighted local σ** | **0.9711** |

**The shuffle control was run first this time**, after H-U's reliability claim turned
out to be an artifact. Give every occurrence someone else's σ — same distribution of
scales, no information about *which* occurrence it belongs to — and reliability is
0.886. The real σ reaches 0.971. So **≈0.085 of the gain is σ being the right scale
for that occurrence**, not merely dividing by a varying number. That is exactly the
test H-U failed.

It also beats G²PO's per-node scale by 0.074: **the kernel is a better scale estimator
than the observation node.**

**Why this is coherent rather than opportunistic.** G²PO standardises within a hard
node because a hard node is all they have. We *deleted* the node (H-M) and then fell
back to per-task scale — which the table shows is the worst of the three options. The
kernel that decides membership should also decide scale; anything else is a mismatch
between the two halves of the estimator.

**Caveats, recorded before implementation:**

* σ has **p10 = 0.000** — degenerate neighbourhoods (all-identical targets) would
  divide by ~0. A floor is required, and its value is a real hyperparameter.
* Measured with **bag-of-words φ** on one checkpoint, not the hidden-state φ we run.
* Split-half reliability is a proxy for gradient quality, not a success rate. It is
  a better proxy than anything else available offline, but H-J and H-U are both
  reminders that a proxy can move without the policy following.
* The trainer's existing per-task standardisation must be **disabled** when this is
  on, or the advantage is standardised twice.

---

## RESULT — `ccpo-localstd-20260908`, stopped at step 80

| step | localstd | task-std | delta |
|---|---|---|---|
| 5 | 18.0 | 10.2 | +7.8 |
| 30 | 19.5 | 30.5 | −10.9 |
| 50 | 51.6 | 50.0 | +1.6 |
| 65 | 47.7 | 47.7 | +0.0 |
| 70 | 57.0 | 63.3 | −6.2 |
| 75 | 59.4 | 68.0 | −8.6 |
| **80** | **64.1** | **71.9** | **−7.8** |

```
PAIRED  mean −2.34   sd 5.29   SE 1.32   t = −1.77   n = 16
        95% CI [−4.94, +0.25]
```

**Not formally significant, but the CI now excludes anything better than +0.25**, and
the last three evaluations are all substantially negative in exactly the stretch where
the comparator accelerated (65 → 100 took it from 47.7 to 79.7).

**Plausible mechanism.** Dividing by the neighbourhood's spread damps precisely the
**high-disagreement** states — the ones where sibling trajectories most disagree about
what happens next, which is where the discriminating signal lives. That cost grows as
the policy improves enough for those states to be informative, which matches the
deltas being near zero early and consistently negative after step 65.

**This is the fourth proxy to improve offline without the policy following** — after
ICC (H-J), per-node reliability (H-U, itself an artifact), and grouping relevance
(H-1). Split-half reliability rewards *consistency*; policy learning apparently does
not reward the same thing. **Reliability is not a safe proxy for gradient quality on
this benchmark**, and that is now established four times over.

**What survives:** the aleatoric/epistemic distinction (H-X). This arm tested
dividing by σ. H-S — weighting by sample-size reliability — remains untested and is a
different quantity.

**Rank: superseded by the result above. Previously: above H-U and above the remaining φ work.** Unlike every φ hypothesis it is
not bounded by H-H's ceiling — it changes the *scale* of the credit, not the quality
of the grouping — and unlike H-U it has an intrinsic criterion that survives its own
control.


### [ ] H-U. **Standardisation level — per task (ours) vs per node (G²PO).** The largest measured deviation

H-O cleared the LOO exclusion (0.009), φ (0.011) and the gate (0.012) and left ~0.09
of the divergence from G²PO unexplained. **This is it.**

G²PO calls `step_norm_reward(..., step_group_uids, ...)` — standardising `A_NC`
**within each node**. We standardise **per task** (`ray_trainer.py:414`, looping over
`uid`). Measured on the hard-gate dump (264,058 samples, bucket = observation node):

| standardisation of our step advantage | corr with the G²PO reference |
|---|---|
| raw | 0.8372 |
| **per TASK (ours)** | 0.8364 |
| **per NODE (G²PO)** | **0.9272** |
| | **+0.0908** |

That single change accounts for essentially the whole residual gap.

**Is per-node better? On reliability, NO — the first measurement was an artifact.**

I first reported split-half reliability of 0.9770 per node against 0.8749 per task,
and flagged as an unresolved caveat that both halves were standardised by their *own*
statistics, which can induce agreement mechanically. **It did.** Controlling for it —
scaling both halves by an independent quantity, the node's target spread:

| | reliability |
|---|---|
| unstandardised | 0.8128 |
| per-node, each half scaled by itself | 0.9770 ← **artifact** |
| **per-node, artifact-free** | **0.8544** |
| **per-task, artifact-free** | **0.8502** |

**The real gap is +0.004, not +0.10.** Per-node standardisation is *not*
demonstrably more reliable. (The small-node worry is separately minor: nodes with ≤2
occurrences are 15.7% of nodes but 2.1% of occurrences.)

**What survives is only the correlation result** — per-node makes our advantage
0.927-agreeing with G²PO instead of 0.836. That is real, but it is an argument of the
form *"be more like the method that scores 95.0"*, not evidence that per-node is
intrinsically better. Circumstantial, and much weaker than this entry first claimed.

**The structural point, and it corrects a claim I made in conversation.** I called
this "a one-line change". It is not, for the shipped configuration: **under the global
gate there is no node.** `bucket = (task_uid,)` — deleting the node and replacing
membership with `exp(−d/τ)` is the entire point of that gate. So per-task
standardisation was never a *choice*; it is a **consequence** of having no nodes left
to standardise within.

Adopting G²PO's level therefore means reintroducing observation nodes as a **second,
parallel grouping used only for scale**, alongside the kernel that decides the
baseline:

* **nodes → the scale** of the credit signal (every state contributes zero-mean,
  unit-variance, so a wide-spread node cannot swamp a narrow one);
* **kernel → the membership** of the baseline.

That separation is coherent and arguably the right design, but it is a real change.

**Why it may matter beyond correlation.** Per-node standardisation is precisely the
variance-reduction property G²PO's own ablation credits to their group-aggregation
component. Per-task lets high-variance nodes dominate the gradient.

**Rank: DOWNGRADED after the artifact control.** It remains the single largest
measured deviation from G²PO and is not bounded by H-H's ceiling — but with the
reliability evidence withdrawn, the case is purely "match the winner". That no longer
clearly outranks finishing the memory arm, which tests the one component of the
original design never cleanly evaluated.

**Do not pause a running arm for this.** I recommended exactly that on the strength of
the 0.977 figure, before running the control that refuted it. Before this earns GPU
time it needs an *intrinsic* criterion — some test that per-node standardisation is
better on its own terms rather than by resemblance to G²PO. If no such test exists
offline, it is a bet, and should be run only after memory resolves and labelled as one.


### [!] H-T. Memory **summaries** in φ — refuted offline; memory **content** still open

Asked how to get memory into φ so it conditions the soft weighting. Two routes.

**Route A — summaries derived from the observation sequence alone (no plumbing).**
`derive_context()` already walks each trajectory in order, so it can compute
`null_frac` (share of past actions that left the observation unchanged), `stall`
(steps since anything new) and `revisit_count` for free.

Well-distributed — 59% of occurrences have non-zero `null_frac`, `stall` p90 = 6 —
but they predict nothing new:

| target | existing ctx R² | + memory feats | Δ |
|---|---|---|---|
| return-to-go | 0.2144 | 0.2158 | **+0.0013** |
| nextnode | 0.1681 | 0.1715 | **+0.0034** |

**Cause: collinearity with what ctx already has.**

```
corr(revisit_count, progress) = −0.615
corr(stall,         progress) = −0.572
corr(null_frac,     progress) = −0.467
```

`progress = n_unique/(t+1)` is *already* a stall detector. **Refuted** — do not add
observation-derived memory summaries to φ; they restate `progress`.

**Route B — the digest's content. Untested, and the only channel carrying anything
new.** What the digest has that the observation sequence does not is the **actions**:
which things were tried and which failed. `"open cabinet 3 → no effect (×3)"` is not
recoverable from observations.

Implementation sketch: carry the digest as a `non_tensor_batch` field the way
`anchor_obs` travels; hash it into a bag plus two scalars (distinct failed actions,
distinct successful ones via `state_signature`); concatenate under a new
`ACG_CCPO_PHI=hidden+ctx+mem` with its own weight so it stays a single-flag ablation.

This would also **fix the H-R confound**: memory currently reaches φ only
incidentally, because the digest sits in the prompt the hidden state is computed
over. Explicit featurisation makes it a deliberate, ablatable channel.

**Test before building** — the deciding measurement is whether *action* history
predicts the target beyond ctx, i.e. the same regression above with digest features.
Needs the digest added to `ACG_CCPO_GDUMP` (two lines) and a 15-minute 2-step probe.
**Prior: also small**, given Route A returned +0.003 and φ's measured relevance is
R² 0.00013 — but it is the one remaining channel the current φ cannot see, so 15
minutes to measure beats 11 hours to guess.


### [OPEN — untested] H-S. **Uncertainty as a weight on A_CC, not a choice between baselines** — the reframing that survives

Every uncertainty result in this file is about **whether to trust φ's neighbourhood
over the uniform one**. That question is settled and the answer is "don't": τ² ≈ 4e-6,
λ = 0.011 whenever it is free to choose, and H-Q showed φ-weighting inside a bucket
is actively worse than uniform. There is nothing to be uncertain *about*, so the dial
has nothing to arbitrate.

**A different uncertainty is large, measurable, and untouched:** how precisely
*this occurrence's* baseline is estimated at all.

**Measurement** (`gate-probe-20260907`, 6,912 occurrences). Split each occurrence's
reference trajectories in half, compute A_CC from each half, correlate:

| J (distinct ref. trajectories) | n | corr(A₁,A₂) | Spearman-Brown reliability |
|---|---|---|---|
| 2 | 336 | 0.522 | 0.686 |
| 3 | 441 | 0.681 | 0.810 |
| 4 | 421 | 0.762 | 0.865 |
| 5 | 573 | 0.771 | 0.871 |
| 6 | 913 | 0.810 | 0.895 |
| 7 | 3266 | 0.839 | 0.912 |

**Monotone in support, 1.36× between J≤3 and J≥6, and 13.1% of occurrences sit in
the unreliable region.** This has nothing to do with φ — it is arithmetic about
sample size, which is why it is not blocked by any of the nulls above.

**The proposal.** Uncertainty modulates the step term's *weight*, not the choice
between two baselines:

```
current:   A = A_EP + w · A_CC        w = 1 for every occurrence
proposed:  A = A_EP + w_u · A_CC      w_u = J_u / (J_u + c)   ~ reliability
```

This is the classical correction for a noisy regressor: an unreliable estimate should
contribute in proportion to its reliability or it injects variance into the gradient
without carrying signal. Same bias-variance logic the method was built on, aimed at a
quantity that actually varies.

**Why it should survive where λ did not.** λ required φ to be informative. This
requires only that a baseline from 2 trajectories is noisier than one from 7, which
is arithmetic and is measured at 1.36×.

**Predicted upside, stated before testing: small.** The affected 13% would be
down-weighted ~25%, so this is variance reduction on a minority of occurrences, not
new signal. **A single training arm almost certainly cannot resolve it** against
±13-point evaluation noise — so this should be settled *offline*, not with GPU time.

**Offline test** (no training run): on the existing dump, compare uniform vs
reliability weighting on (a) the variance of A_CC, and (b) its correlation with the
target. The proposal wins if variance falls while correlation holds. If correlation
falls proportionally, the weighting is removing signal along with noise and should be
dropped.

**Relation to H-N.** H-N proposed shrinking toward the *hard-gate* baseline instead of
the uniform task mean — still a choice-between-baselines framing. H-S supersedes it as
the more promising use of uncertainty, because it does not depend on either baseline
being better than the other.


### [~] H-R. The memory arm moves **two** channels, not one — caveat recorded before its result

`ccpo-memory-20260907` is a single *config* change from the 79.7% arm
(`compact_budget` 0 → 512). It is **not** a single *causal* change.

The digest is prepended to `memory_contexts`, so it enters the **prompt**
(`env_manager.py:203`). φ's observation half is the reference policy's hidden state
at position `seqlen − response_length − 1` — the **last prompt token**
(`dp_actor.py:262`). So the digest is inside the sequence the hidden state is
computed over, and turning memory on changes:

1. **what the policy conditions on** — the intended test; and
2. **what φ sees**, hence which occurrences are neighbours — unintended.

The *context scalars* are unaffected: `derive_context()` builds
`{t, n_unique, revisit, progress}` from the observation sequence alone and never
reads the digest. Only the hidden half is coupled.

**Consequence.** If this arm beats 79.7%, the gain is not attributable to the policy
channel without a third arm — digest in the prompt, φ computed from a digest-free
prompt — which needs a second forward pass and does not currently exist.

**Prediction on record (from prior nulls, not from measurement):** the φ channel
contributes ~nothing, because `phi_rel_corr` is 0.011 with implied R² 0.00013 and
τ² ≈ 4e-6 across 100 steps. Any effect should be the policy channel. If the arm wins
*and* `phi_rel_corr` is unchanged, that reading is supported; if `phi_rel_corr` moves
materially, it is not, and the third arm becomes necessary.

This should have been flagged at launch rather than after the fact.


### [!] H-Q. Hard gate **with** context conditioning (λ=1) — the missing cell, REFUTED offline

The two arms confound gate with λ: `ccpo-global` is global gate + λ=1 (conditioning
ON), `ccpo-hardedge` is hard gate + λ=0.011 (conditioning OFF). **Never run: hard
gate with λ forced to 1** — φ-weighting *inside* GiGPO's own bucket. That is
arguably the truest test of the original thesis, since it keeps the baselines'
grouping and changes only how members are weighted.

**Measured offline** on `gate-probe-20260907`, LOO residual R² within exact-obs
buckets:

| | return-to-go | nextnode |
|---|---|---|
| uniform LOO (λ=0, what `ccpo-hardedge` runs) | **0.4579** | **0.7434** |
| φ-weighted, τ=0.10 | 0.3589 (−0.099) | 0.6736 (−0.070) |
| φ-weighted, τ=0.15 | 0.3955 (−0.063) | 0.6976 (−0.046) |
| φ-weighted, τ=0.25 | 0.4381 (−0.020) | 0.7228 (−0.021) |
| φ-weighted, τ=0.50 | 0.4653 (+0.007) | 0.7394 (−0.004) |
| φ-weighted, τ=1.00 | 0.4673 (+0.009) | 0.7434 (−0.000) |

**φ-weighting inside a bucket is worse than uniform at every useful τ**, and the
curve is monotone toward the uniform baseline — it reaches parity only where τ is
large enough that `exp(−d/τ) ≈ 1` for everyone, i.e. **the best weighting is no
weighting**. Under `nextnode` it never exceeds uniform at all.

**This vindicates the shrinkage.** λ = 0.011 is not the estimator missing a signal:
conditioning inside an exact-observation bucket *actively destroys* information, and
empirical Bayes correctly refuses to pay for it. Every earlier entry in this file
that treated λ→0 as a symptom to be fixed (H-2, H-3, and the `eb_pooled`/`eb_hier`
work) was chasing a component that was behaving correctly.

**It also isolates what the global gate actually does.** There, φ is not re-weighting
a set that was already well chosen — it is *choosing* the set. **Membership and
weighting are different jobs, and φ is only competent at the first.** That is the
sharpest statement of what survives from the context-conditioning idea.

**Decision: do not run this arm.** Offline says it would land 2–10 R² points below
the arm already paused. The probe cost ~1 minute of CPU against ~11 GPU-hours.


### [~] H-P. **The `ccpo-hardedge` ablation is not gate-vs-gate — it is CCPO ON vs OFF**

I launched `ccpo-hardedge-20260907` calling it "hard gate vs global gate, edge term
held fixed". That label is wrong, and the truth makes it a *better* experiment.

Under the hard gate the empirical-Bayes shrinkage collapses, as it has in every arm
of this project:

| | λ | `effect_rel` | λ>0.5 | `r_vs_g2po` |
|---|---|---|---|---|
| hard gate | **0.0106** | **0.0014** | 1.0% | 0.9090 |
| global gate | 1.0000 | 0.1185 | 100% | 0.8909 |

λ = 0.011 puts **98.9% of the weight on `b_obs`**, the plain uniform bucket mean.
φ, the affinity weights and the whole context-conditioning apparatus contribute
~1%, and the credit departs from the uniform baseline by **0.14%**. That arm is
therefore *not* CCPO: it is **GiGPO's grouping + G²PO's node-value target + G²PO's
edge term, with context conditioning switched off**, which is why it sits closer to
G²PO (0.909) than the global arm does.

**So the comparison actually running is:**

> context conditioning fully ON (λ=1, `effect_rel` 0.119)
> vs fully OFF (λ=0.01, `effect_rel` 0.001),
> with target, edge term, prompt, seed and every other knob identical.

That is the **cleanest test of CCPO's central thesis this project has run** — every
earlier attempt was confounded, and the two 20-step arms never separated the
estimator from the harness.

**Result at step 40 of 100:** mean delta **+0.87 pts**, 95% CI containing zero over
eight evaluations, sign flipping four times.

**If this holds to step 100** the conclusion is stated plainly: *CCPO's context
conditioning is inert on ALFWorld even when the machinery is made to work.* H-M
fixed the inertness as a mechanism (`effect_rel` 0.000 → 0.119) and bought nothing
in policy performance. The project's real results would then be:

1. the **negative result** — context-conditioned grouping does not help on this
   benchmark, established across φ variants (H-1), targets, shrinkage rules,
   a measured 6.1% ceiling (H-H), and now a clean ON/OFF ablation;
2. the **79.7%** number, which came from restoring G²PO's edge term, not from CCPO.

Hold the conclusion until step 100 — the global arm's gains appeared late in its
curve (63.3 → 79.7 between steps 70 and 100), so the informative region is ahead.


### [!] H-O. **The estimator is NOT why we underperform** — three suspects tested, all cleared

Prompted by "why are we still underperforming": 79.7 (ours) vs 95.0 (G²PO) at an
identical protocol. Decomposed the divergence offline on the 485,888-sample dump
from `ccpo-global-20260907`, plus the two arms' own diagnostics.

**Suspect 1 — the leave-one-out exclusion.** Ours excludes the scored trajectory;
G²PO includes it. I had treated this as a correctness fix and never tested it.

**Suspect 2 — the φ weighting.**

**Suspect 3 — the gate** (global task-bucket vs G²PO's per-observation node).

| variant | corr with G²PO | sign disagree |
|---|---|---|
| ours as shipped (φ-weighted LOO) | 0.7360 | 17.3% |
| drop φ, uniform LOO | 0.7250 | 18.4% |
| self-inclusive mean (G²PO-style) | **0.7449** | 17.1% |

| arm | `r_vs_g2po` | `r_vs_gigpo` |
|---|---|---|
| global gate (task = bucket) | 0.8909 | 0.4089 |
| hard gate (obs = node) | **0.9028** | 0.4517 |

**All three cleared.** LOO costs 0.009. φ *adds* 0.011. The gate moves agreement by
0.012. Our step term is ~**0.90 correlated with G²PO's**, and every knob we own
moves that by about one point.

**Conclusion: a 15-point performance gap does not follow from a 0.10 correlation
gap in the advantage.** The estimator is close to G²PO's; the estimator is
therefore very unlikely to be the cause. This *retracts* the ranking given earlier
in this session, which put "reduce divergence from G²PO" and "switch the target"
at the top — both aim at a component that measures fine.

**What remains, and it is now the only cheap-to-eliminate unknown:** the harness.
ALFWorld build, env seeding, vLLM version, tokenisation, reward wiring, or seed
variance (ours is 1 seed against their 3-seed mean, and G²PO reports ±0.8).

**The decisive experiment is a G²PO arm on our harness**, and after this analysis
it is no longer a matter of methodological preference — it is the only remaining
way to locate the gap. Two outcomes, both informative:

* **~95** → our harness is fine and our method is genuinely 15 points worse
  *despite* a 0.90-correlated advantage. That is a surprising result in its own
  right and would point at the composition (`A_EP + w·A_CC`) or the
  standardisation level rather than the step term.
* **~80** → our harness caps every method, and CCPO is at parity with a SOTA
  method. Every number in this file would then need restating as relative rather
  than absolute.

Also worth noting: `r_vs_gigpo` is 0.41-0.45, so we are far from GiGPO and close to
G²PO. Whatever we are, we are a G²PO variant.


### [ ] H-N. **Shrink toward the hard gate, not the uniform mean** — restores uncertainty as a live component

**The problem this fixes.** H-M's global gate runs with **λ = 1.0, hard-set**. Every
uncertainty quantity is still computed and logged, and none of them touches the
advantage:

```
lam_u_mean      1.0000   forced, not estimated
tau2            0.0000   signal variance: still "no signal"
lam_eb_obs      0.0173   what the EB rule WOULD give if consulted
rho             0.5900   metric confidence -- computed, unused
```

So `ccpo-global` tests context-conditioned **grouping** and does **not** test
uncertainty at all. If that arm wins, it wins on grouping, and the write-up must
say so. Uncertainty is one of the two ideas the method is built on; right now it is
decorative again, in a new way.

**Why λ=1 was necessary and not merely lazy.** Under a global gate `b_obs`
degenerates to the uniform task mean, measured at R² **0.4248** against the hard
gate's **0.4579** — *worse*. Shrinking toward it moves the estimator toward the
worse baseline, and since the EB rule still reports λ≈0.017 that is exactly where
it would land. λ=1 was the only way to use the estimate that measured better
(φ-weighted `b_loo`, 0.4841).

**The fix.** Make the shrinkage target the **exact-observation bucket mean** rather
than the uniform task mean. Then

```
A_CC  =  target − [ λ · b_loo(global φ-weighted)  +  (1−λ) · b_hard(exact-obs bucket) ]
```

and λ trades two *defensible* estimators instead of one good and one bad:

* **λ → 1**: the global soft neighbourhood (R² 0.4841)
* **λ → 0**: GiGPO's exact-observation gate (R² 0.4579) — a **floor**, not a cliff

This makes the hard gate the worst case rather than the uniform mean, so the
estimator cannot do worse than the published baselines' grouping however badly φ
behaves. It also restores the original bias-variance story: shrink toward the
conservative local estimate exactly where the global neighbourhood is thin or φ is
untrustworthy.

**Implementation** (~20 lines in `ccpo_step_advantage`): compute the exact-obs
bucket mean alongside the global one, carry it as `b_hard` in `_rec`, and use it in
place of `b_obs` in the shrinkage when `_GATE == "global"`. Occurrences whose
exact-obs bucket has <2 distinct trajectories — the ~35% the hard gate cannot serve —
fall back to λ=1, which is correct: there is no local estimate to shrink toward.

**Predicted outcome, on record before running.** λ will still be small, because
`lam_eb_obs` says so and nothing about the target changes. So this arm should land
*close to* `ccpo-global` rather than above it, and its value is (a) removing the
downside risk of a bad φ and (b) making uncertainty a live, measurable component
again. **If λ stays under 0.05 the honest conclusion is that CCPO's uncertainty
half does not earn its place on ALFWorld**, and the method should be presented as
context-conditioned grouping alone.

Run after `ccpo-global-20260907` reports.


### [x] H-M. **Global context-conditioned grouping** — RESULT: 79.7%, best of the project — the first CCPO-shaped idea that measures positive

**The idea.** Drop the hard `(task_uid, observation)` gate. Make the whole task one
bucket and let the φ kernel `exp(−d/τ)` decide the neighbourhood softly. The hard
gate becomes the **τ → 0 limit** of the soft one rather than a separate mechanism —
a genuine unification of CCPO's context conditioning with G²PO's global view.

**Why it is not bounded by H-H.** That 6.1% ceiling was measured *inside* exact-
observation buckets, so it bounds the baseline **within** a gate. This changes what
the gate is. Same reason H-D and H-L escaped the cap — but unlike those two, this
one measures positive.

**Offline evidence** — `gate-probe-20260907`, 6,912 occurrences, LOO residual R²,
φ = bag-of-words(obs) ⊕ thermometer context:

| baseline | return-to-go | nextnode |
|---|---|---|
| exact-obs gate (GiGPO / G²PO) | 0.4579 | 0.7434 |
| uniform global (LOO) | 0.4248 | 0.7512 |
| **global φ-weighted** | **0.4841** (τ=0.15) | **0.7604** (τ=0.25) |
| | **+0.026** | **+0.017** |

The τ curve is an **inverted U** under both targets — 0.30 at τ=0.02, peak at
0.15–0.25, decaying toward the uniform mean beyond. The optimum is interior, which
is the whole argument: neither hard local matching nor global averaging, but a soft
global neighbourhood.

**Why this works where progress banding (H-J) failed.** Banding made buckets
sharper *and smaller*, and the sample loss ate the gain. A soft kernel makes them
sharper *without* shrinking support: low weights replace exclusion instead of
discarding occurrences. `live_frac` goes 0.88–0.93 → **1.000**; nothing falls dead.

**Implementation trap, caught before launching.** Under a global gate `b_obs`
degenerates to the **uniform task mean**, which measured 0.4248 — *worse* than the
hard gate. The quantity that measured better is the φ-weighted `b_loo`, which
carries weight λ. Since λ has been **exactly 0.000 in every run of this project**,
a naive global gate would have shrunk onto the worse baseline and lost points while
looking like a faithful implementation. So the global gate takes **λ = 1**: the
soft kernel is the gate, and there is nothing left to shrink toward.

**First evidence the estimator is no longer inert.** `effect_rel` — how far the
credit departs from the uniform baseline — has been *exactly* 0.0000 at every step
of every arm in this project. Under the global gate it is **0.3629**.

**Experiment** `ccpo-global-20260907`: `gate=global, tau=0.15, edge_w=1.0,
target=nextnode, phi=hidden+ctx`, 100 iterations (G²PO's own length).

**Confound, stated up front — and it is worse than two things.** Comparing
`ccpo-global` against the hard-gate curve (`ccpo-mem` steps 1-20 + `ccpo-long`
steps 21-100) differs in **four** ways, not one:

| | hard arm | ccpo-global |
|---|---|---|
| gate | hard | **global** |
| `edge_w` | 0.0 | **1.0** |
| effective λ | 0.000 | **1.0** |
| `compact_budget` | **512** (steps 1-20), 0 after | 0 throughout |

The memory column is not merely a confound, it is *inconsistent within the
comparator*: steps 5/10/15/20 compare against a memory-ON hard arm, step 25+
against a memory-OFF one, because that is where the `ccpo-long` warm start took
over. **The baseline changed configuration partway through the series.** Any
matched-step delta read across that boundary — including the +0.039 lead at steps
5-20 and the −0.031 reversal at step 25 — is contaminated by it.

`ccpo-global` remains valid **on its own terms**: a clean 100-step run from base at
a fixed config, comparable to G²PO's published 95.0 without needing any internal
comparator. Use that comparison, not the matched-step one.

A genuine gate-vs-gate ablation needs a hard-gate arm at `edge_w=1.0`,
`compact_budget=0` throughout. Until that exists, **no causal claim about the gate
is supported.**

The original note, still true: this arm also changes `edge_w` 0.0 → 1.0. The latter is not a variable under test — it is
restoring G²PO's edge-centric advantage, which we had switched off while trying to
beat G²PO, and which their ablation credits for their margin over GiGPO. It is a
fix, not a treatment. But if this arm wins, the split between the two is unknown
without an ablation, and that ablation should be run before any claim is made.

---

## RESULT — `ccpo-global-20260907`, clean exit at step 100, 0 errors

**79.69% held-out**, the run's best and still climbing at the cutoff
(tail slope +2.19 pts/5 steps over the last six evaluations).

```
5    10   15   20   25   30   35   40   45   50
10.2 10.9 17.2 21.1 21.1 30.5 35.9 22.7 41.4 50.0
55   60   65   70   75   80   85   90   95   100
35.2 41.4 47.7 63.3 68.0 71.9 75.8 75.0 78.1 79.7
```

| method (same protocol, 100 iters) | ALFWorld All |
|---|---|
| G²PO | 95.0 ± 0.8 |
| GiGPO | 86.7 ± 1.7 |
| **CCPO global gate (ours, 1 seed)** | **79.7** |
| GRPO | 72.8 ± 3.6 |
| CCPO hard gate (previous arm) | 70.3 |
| RLOO | 69.7 ± 2.5 |
| PPO | 54.4 ± 3.1 |

**+9.4 points over the previous arm.** Passes PPO, RLOO and GRPO; short of GiGPO
and G²PO. Per-type at step 100 is uniform — every category between 73 and 92,
with no dead category, unlike earlier arms:

```
pick_and_place 91.7   look_at_obj 80.0   pick_heat 78.6
pick_clean 76.0       pick_two_obj 75.0  pick_cool 73.2
```

**The mechanism ran, and that is the headline for the method rather than the
number.** `effect_rel` averaged **0.119** against *exactly 0.0000* in every prior
arm; `live_frac` 1.000 (nothing dead, against ~0.65 usable under the hard gate);
`edge_cov` 1.000. CCPO's estimator was doing something other than reproducing the
uniform bucket baseline for the first time.

**What this does NOT establish.** The four-way confound stands: gate, `edge_w`,
λ and memory all differ from the previous arm. **The +9.4 cannot be attributed to
the global gate.** `edge_w=1.0` restored G²PO's edge-centric advantage, which their
own ablation credits for their margin over GiGPO, and that alone could account for
most or all of the gain. A hard-gate arm at `edge_w=1.0` is required before any
gate claim is made, and it is now the highest-value next experiment.

Also: one seed against the baselines' three, and `phi_rel_corr` stayed at 0.011 —
φ still barely predicts return-similarity, so whatever the global gate bought, it
was not by making φ informative.

**Target to beat** (G²PO Table 1, Qwen2.5-1.5B, ALFWorld, *identical* protocol,
100 iterations, 3 seeds): G²PO **95.0 ± 0.8**, GiGPO 86.7 ± 1.7, GRPO 72.8 ± 3.6,
RLOO 69.7 ± 2.5. Ours to date: **70.3** (1 seed) — below GRPO.


### [!] H-J. Progress banding the gate — PROPOSED AND REFUTED THE SAME NIGHT

**Experiment** `gate-probe-20260907` (2 steps warm-started from `ccpo-long`
step 100, no val, no checkpoints; 6,912 occurrences with full observation text
dumped via the new `ACG_CCPO_GDUMP`). Analysis: `scripts/analyse_gate.py`.

**The claim I made:** bucketing on `(task, obs, progress_band)` instead of
`(task, obs)` fixes the baselines' time-blindness, and since it changes bucket
*membership* rather than the baseline within a bucket, it escapes H-H's 6.1% cap.

**First measurement (ICC) appeared to confirm it, and was the wrong metric.**
Bias-corrected ICC rose 0.023 → 0.103, which I read as "4.5x more exploitable
signal". ICC measures the *share* of within-bucket variance that is
between-trajectory. It says nothing about whether the estimator can exploit that
share, because it ignores the noise in estimating the baseline from a smaller
bucket.

**Correct measurement — variance the LOO baseline actually explains:**

| gate | dead | R² (return-to-go) | R² (nextnode) |
|---|---|---|---|
| `(task, obs)` — GiGPO / G²PO | 0.073 | **0.4673** | 0.7458 |
| `+ progress band` | 0.141 | 0.4398 (−0.028) | 0.7125 (−0.033) |
| visited-set signature | 0.947 | 0.5263 | 0.7658 |
| task only, no obs gate | 0.000 | 0.4248 | 0.7512 |

Banding **loses under both targets.** Bucket occupancy falls 5.11 → 3.66
trajectories and dead weight nearly doubles; the noisier baseline more than eats
the ICC gain. **Refuted.**

**Lesson: ICC is not a sufficient criterion for a gate change.** It is a
signal-share statistic; the decision needs a signal-to-noise statistic. Use the
LOO residual R² from `analyse_gate.py`, which prices both the sharper bucket and
the smaller sample. The same error killed the visited-set proposal an hour
earlier from the opposite direction (great ICC, 95% dead).

---

### [ ] H-K. **The nextnode target hollows out the anchor gate** — the real finding

Same dump. Read the two R² columns above *against each other*:

* under **return-to-go** (GiGPO's target), the anchor gate beats no-gate by
  **+0.043** — GiGPO's grouping is doing real work;
* under **nextnode** (G²PO's successor value, which *we* run), the anchor gate is
  worth **−0.005** — ignoring the observation entirely does slightly *better*.

The successor value already encodes where the trajectory ended up, so gating on
where it started is close to redundant. **We adopted these two components
separately and never tested them together, and the combination appears to cancel.**

This is a live candidate for part of the 70.3 vs 90.16 gap: our configuration may
have quietly neutralised the one component that gives GiGPO its edge over GRPO.

**Cheap test, no new training:** `analyse_gate.py` already scores it offline. The
decisive on-policy test is one arm with `ACG_CCPO_TARGET=return` and everything
else fixed. But it should be ranked behind the GiGPO baseline, which is now more
informative than it looked: this analysis produced a concrete mechanism by which
our own config could cost points against the published number.

---

### [ ] H-L. Relaxed visited-set gate

The visited-set signature has the best R² under both targets (+0.059 under
return-to-go) but 94.7% dead, so nearly all of that comes from the task-mean
fallback rather than from the gate. What it actually shows is that **the 5.3% it
serves, it serves well** — precise state identity works, it just has no support in
a 128-trajectory batch.

Worth testing: coarsen it until support appears — `state_signature()`'s
order-invariant successful-action set, or a backoff that falls to `(task, obs)`
when a visited-set bucket has fewer than 2 trajectories. That keeps the precision
where it is affordable and the baselines' gate everywhere else.


### [x] H-I. Will 100 steps actually close the gap? — RESOLVED: yes, mostly — prediction on record, 2026-09-06

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

| step | 5 | 10 | 15 | 20 | 25 | 30 | 35 | 40 | 45 | 50 |
|---|---|---|---|---|---|---|---|---|---|---|
| success | .0625 | .0781 | .1250 | .1719 | .2422 | .2500 | .3047 | .3438 | .2891 | .4219 |
| partial | .226 | .386 | .454 | .803 | 1.043 | 1.278 | 1.475 | 1.821 | 1.391 | 2.371 |

| step | 55 | 60 | 65 | 70 | 75 | 80 | 85 | 90 | 95 | **100** |
|---|---|---|---|---|---|---|---|---|---|---|
| success | .4141 | .4141 | .5703 | .6406 | .5938 | .6172 | .6406 | .6172 | .6719 | **.7031** |
| partial | 2.152 | 2.038 | 3.178 | 3.477 | 2.900 | 3.321 | 3.312 | 3.048 | 3.076 | **3.752** |

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

**Step-60 checkpoint: the pre-registered marker is cleared, but a plateau has
appeared.** The marker was ~0.35 by step 60; actual is **0.4141**. By the criterion
written down in advance, the run continues.

Against that: three consecutive evaluations at 0.4219 / 0.4141 / 0.4141 — 15 steps
with no improvement — and partial credit drifting *down* across the same window
(2.371 → 2.152 → 2.038). Partial credit is the less noisy indicator, so a decline
there is worth more than the flatness in success.

**Decision: continue to 100.** Reasoning, recorded now so it can be judged later:

* The marker was set in advance precisely so a plateau would not be re-litigated
  mid-run. 0.414 clears 0.35 comfortably.
* This series has flattened twice before and resumed both times — step 30 was flat
  (+.008) then jumped to .3047; step 45 *fell* to .2891 then jumped to .4219. A
  third flat window after two false alarms is weak evidence.
* Early stopping is armed and handles the true-plateau case without my judgement:
  patience 8 with two stale evaluations banked means it fires at **step 90**, so
  the exposure is 30 steps (~3.5 h), not 40.
* `step50-best` is preserved regardless, so a plateau costs compute, not the result.

**What would change the decision:** step 65 or 70 coming in *below* 0.39 — that
would be a decline rather than a plateau, and partial credit falling for a fourth
consecutive evaluation would corroborate it. In that case kill immediately rather
than waiting for step 90; early stopping fires on plateau, and a decline is worse
than a plateau.

> **Resolved at step 65: 0.5703.** The kill criterion was 0.39; actual came in
> +0.156 above the plateau, ~4 SE. Continuing was correct.
>
> **The lesson worth keeping is about this curve's shape, not about this call.**
> It has now flattened or fallen and resumed *three* times — step 30 (+.008), step
> 45 (−.055), steps 55–60 (flat for 15 steps). Every one looked like a plateau in
> the moment and none was. On a 128-episode held-out with SE ≈ 0.04, a 10–15 step
> flat window is simply not resolvable from noise, so **plateau calls on this setup
> need ~20+ steps of evidence, not 15.** Early stopping's patience of 8 evaluations
> (40 steps) is correctly sized; my instinct to look at 15 was not.
>
> Also worth recording: partial credit declined across the plateau (2.371 → 2.152 →
> 2.038) and I weighted that as corroborating evidence because it is the less noisy
> indicator. It then jumped to 3.178. Being less noisy than a noisy thing does not
> make it a reliable leading indicator over three points.

---

## RESOLVED at step 100: **0.7031** held-out. Run complete, clean exit, 0 errors.

**The prediction was 0.45–0.70. The answer was 0.7031** — at the top edge, a hair
outside. The band was slightly conservative, as flagged at step 40, and it was not
revised mid-run.

| | held-out success |
|---|---|
| CCPO @ step 20 (previous arm) | 17.2 |
| **CCPO @ step 100 (this run)** | **70.3** |
| GRPO (published) | 72.8 |
| GiGPO (G²PO Table 1, same protocol) | 86.7 |
| HGPO K=2 (published) | 92.77 |

Per-type at step 100 — every type well above zero, including all four
compositional ones that sat at 0.000 through step 20:

```
pick_heat_then_place_in_recep   0.792     pick_clean_then_place_in_recep  0.731
pick_and_place                  0.789     pick_two_obj_and_place          0.600
pick_cool_then_place_in_recep   0.756     look_at_obj_in_light            0.292
```

**H-F is settled conclusively: the harness was never broken, only under-trained.**
5x the training turned 17.2 into 70.3.

**The curve had not saturated at the cutoff.** Linear fit over the last six
evaluations: **+0.0196 per 5 steps, r = +0.909**. Extending is worth real points.

### What this cost me in credibility, recorded so it is not repeated

I called a plateau or leaned toward one **four times** on this curve — steps 30,
45, 55–60, and 85–90 — and was wrong every time. The step-90 call was the worst:
I stated "the curve has converged around 0.62" and *reversed a compute
recommendation* on it. The next three evaluations were .6172 → .6719 → .7031, the
three highest of the run.

The mistake was not impatience, and "wait 20 steps instead of 15" was the wrong
fix — it was a tighter version of the same error. The real problem: **at n=128 the
binomial SE is ~0.042, and the tail's true slope is ~0.02 per 5 steps.** The signal
is half the noise. Six consecutive evaluations cannot distinguish a flat curve from
this one, so *no* reading of this series could have supported a saturation call.
The instrument could not answer the question I kept asking it.

**Rule for future arms:** never call saturation from the 128-episode series alone.
Either widen the evaluation (n=512 → SE 0.021) or judge from the training-draw
trend, which was climbing steadily (0.63 → 0.85) throughout the window I called
converged and was the correct signal all along.

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

### Why our anchor is observation-only: we treated the hard key as the baseline to BEAT

Not an oversight. Traced 2026-09-10:

* `aff` was added **2026-09-03** with the comment *"so the credit assigner can label
  states the observation text conflates (42.8% of observations map to >1 admissible
  set)"*. **The conflation was measured and known.**
* But it was threaded through as `aff_labels`, and reaches exactly one use:
  `core_ccpo.py:795`, writing a column into the diagnostic CSV. **It never enters a node
  key.**

**The word "label" is the tell.** The intent was to MEASURE the conflation, to show phi
was resolving it -- not to FIX it in the grouping. In CCPO's design phi was supposed to do
that job: a soft kernel over a task-wide bucket would find the right neighbours, so
enriching the hard anchor looked redundant. **The anchor was the fallback we expected to
outperform.**

That inverted the dependency. G2PO puts affordances INTO the state identity, where they
partition. We kept them BESIDE it, where they only annotate.

```
aff as a partition refinement  ->  29.1% of within-node value variance
aff as a similarity metric     ->  ~0   (r = -0.013)
phi as a similarity metric     ->  ~0   (r = +0.011)
```

**We had the right feature from the start and put it where it does nothing**, because the
architecture assumed the learned component would make the hard key irrelevant. That
assumption is exactly what H-AK refutes five ways.

**Transferable lesson:** when a method proposes to replace an existing mechanism, the
existing mechanism is the thing to build on until the replacement is *demonstrated*
superior -- not the thing to leave un-improved so the replacement looks better. Every
enrichment we withheld from the anchor was an enrichment withheld from our own step term.


### CORRECTION: our anchor repair is only HALF domain-agnostic

I claimed repeatedly that G2PO's invisible-state repair uses hand-written ALFWorld
vocabulary "while ours derives failures generically". **Half true, and the borrowed half
is copied verbatim:**

```
ours:   _INVISIBLE_PATTERNS = ("You heat", "You cool", "You clean", "You turn on")
theirs: PATTERNS           = ["You heat", "You cool", "You clean", "You turn on"]
```

| mechanism | ours | generic? |
|---|---|---|
| failed action does not advance the anchor | `obs_{t+1} == obs_t` | **YES** |
| invisible-state carry-forward | copied `PATTERNS` | **NO — theirs, verbatim** |
| admissible actions in the key (`aff`) | env API | **YES** |

**Consequence for `g2po-aff`.** As configured (`obs_repair=1 anchor_aff=1`) it CANNOT
support the claim "a domain-agnostic disambiguator matches a hand-coded one", because one
of its three components IS the hand-coded one. If it reaches ~91.4 we will not know
whether the generic parts or the borrowed vocabulary did the work.

**Ablation required: `anchor_aff=1, obs_repair=0`.** That is fully generic -- admissible
actions from the environment API, no vocabulary. Queued as `g2po-affonly`.

* `affonly` ~ `aff` -> the generic component does the work; **the contribution is real**
* `affonly` << `aff` -> the ALFWorld vocabulary carries it; **no generic contribution**

**A genuinely generic replacement for PATTERNS may exist and is worth testing later:**
heat/cool/clean/turn-on all CHANGE the admissible-action set, so an `aff` transition may
detect exactly the state changes `PATTERNS` was written to catch. That would make the
whole repair vocabulary-free. Not testable on the current dumps -- they carry no
within-trajectory turn index -- so it needs an instrumented run.


### CORRECTION: the global gate does NOT average over the whole task

I said repeatedly today that `ACG_CCPO_GATE=global` "compares each action against a
task-wide average" and that phi being inert makes the kernel "effectively uniform".
**Both are wrong, and the metric was logged all along.**

```
ccpo-global, mean over 100 steps:
  bucket_size_mean   7.85     <- occurrences in the bucket
  n_eff_mean         5.84     <- EFFECTIVE neighbours after exp(-d/tau) weighting
  phi_rel_corr       0.011    <- phi-distance vs |value difference|
  r_vs_g2po          0.891
  effect_rel         0.119
```

`n_eff` 5.84 against a bucket of 7.85 means **the kernel concentrates**. It selects a
small effective neighbourhood; it does not flatten to the task mean.

**The real failure is sharper than the one I described.** The kernel picks ~6 of ~8
neighbours *confidently*, and `phi_rel_corr` = 0.011 says that choice is uncorrelated
with the quantity being baselined. **It is not vague, it is confidently wrong.**

**What CCPO actually is, stated correctly.** Not a different idea from G2PO -- the SAME
structure with a learned similarity substituted for an exact one:

| | G2PO | CCPO global |
|---|---|---|
| partition | hard, by `obs + admissible` | soft kernel over one bucket/task |
| similarity | string equality on a rich anchor | `exp(-d/tau)` on phi |
| baseline | self-inclusive node mean | phi-weighted leave-one-out |
| shrinkage | none | lambda, forced to 1 under global |

As tau -> 0, `exp(-d/tau)` becomes an indicator on d = 0, i.e. **G2PO's hard grouping is
the tau -> 0 limit of ours**. `r_vs_g2po` = 0.891 confirms the advantages are largely the
same; the CC term moves 12% of |A| and that 12% measured at -0.01 points.

**So the method's premise reduces to one claim: that a learned similarity beats an exact
one.** On ALFWorld it does not, and H-AJ says why -- their "exact" similarity is on
`obs + admissible-action set`, whose admissible component alone explains 29.1% of
within-node value variance.


### The KL pre-clamp is a Qwen3 fix, inert on Qwen2.5 — and the obvious check is circular

`core_algos.py` clamps `ref_logprob - logprob` to [-20, 20] before `exp()` in the k3 /
`low_var_kl` estimator. Its comment claimed it was load-bearing, citing NaN grad norms
from step 13 and 132 skipped updates. **Those observations are from the Qwen3-1.7B
generation** (the clamp arrived in the initial CCPO commit, 2026-09-04, alongside the
Blackwell sdpa/flash-attn-stub work).

**The tempting check is circular.** Every Qwen2.5 arm shows zero NaN/inf grad norms
(max 3.45 over 100 steps) and zero skipped updates -- but the clamp *prevents* exactly
that symptom, so its absence says nothing about whether the clamp is firing.

**The non-circular evidence is the reference run.** `g2po-ref-20260909` trains the same
model on the same data with the same `kl_loss_type=low_var_kl`, running their tree
**without this clamp**, and has passed 20 steps with no NaN. If Qwen2.5 produced
|kl| ~ 90 tokens in this setup, their run would have died the way ours did on Qwen3.
Corroborated by `actor/kl_loss` ~ 0.026: under k3, kld ~ kl^2/2, so RMS |kl| ~ 0.23.

**Conclusion: not a difference from G2PO on this backbone.** The code is kept (free, and
guards a return to Qwen3) with the comment corrected. Removed from the list of candidate
explanations for the gap.

**General lesson:** when a guard suppresses the symptom it was added for, you cannot use
the symptom's absence to judge the guard. Find a system running without the guard.


### Train success is NOT below held-out — and a single train step will fool you

Asked whether we hold train success rates, I first reported `ccpo-global-ext` at
**95.3% train against 79.7% held-out** and called it a 15-point generalization gap, with
the flourish that our policy reaches on training what G2PO reports on held-out.
**That was wrong.** 95.3 is the value at the LAST step, and single-step train values
swing enormously: 128 episodes drawn over only 16 tasks, so one favourable task draw
moves the number 20 points.

Averaging train over the 5 steps around each evaluation:

| arm | step | train | held-out | gap |
|---|---|---|---|---|
| ccpo-global | 100 | 77.7 | 79.7 | **-2.0** |
| ccpo-global-ext | 145 | 83.4 | 85.9 | **-2.5** |
| ccpo-long | 100 | 67.7 | 70.3 | -2.7 |
| ccpo-localstd | 80 | 64.7 | 64.1 | +0.6 |

**Held-out sits slightly ABOVE train, consistently.** That is what the sampling
temperatures predict: held-out runs at T=0.4 (greedy) against training's T=1.0
(exploratory). There is no overfitting to correct.

**Consequences.**

1. **The gap to G2PO is a LEARNING gap, not a transfer gap.** We do not reach their
   level on either split. Any hypothesis framed as "we overfit the training games" is
   dead on arrival.
2. **`ccpo-anchor` should be judged on held-out level alone** -- there is no train/test
   gap for it to close.
3. **Never quote a single train step.** Use a window. This is the same error class as
   the `stepN-best` trap: reading a maximum, or an endpoint, off a noisy series.

Composition does not explain anything either: `valid_seen` is if anything *easier* than
`train` -- 13.1% `pick_two_obj_and_place` (the hardest type) against training's 16.7%,
and more `pick_and_place_simple`.


### `eval_in_distribution` is much closer to training than its name suggests

Checked 2026-09-09 after the question "is G2PO's 95% train/test leakage?".

**Not leakage.** `train` and `valid_seen` are different directories, and:

```
shared trial ids for an overlapping config      0
identical initial_state.pddl hashes             0   (400 train sampled vs all 251 valid_seen)
```

**But the split is close by construction.** 240 of `valid_seen`'s 242 top-level
directory names also appear in `train`. Those names encode
`task-object-receptacle-scene`, so an eval game is a **different trial of the same task
configuration, in the same room, with the same objects** as a training game. Example:
`look_at_obj_in_light-AlarmClock-None-DeskLamp-323` has two trials under `train` and a
third, different one under `valid_seen`.

That is ALFWorld's own design -- `valid_unseen` holds 85 novel scenes against
`valid_seen`'s 242 overlapping ones -- and `eval_in_distribution` maps to `valid_seen`
(`config_tw.yaml`: `eval_id_data_path: $ALFWORLD_DATA/json_2.1.1/valid_seen`).

**Also worth knowing: the parquets are placeholders.** `examples/data_preprocess/prepare.py`
carries the upstream note "We do NOT use the data in 'hiyouga/geometry3k', instead we
only use it to indicate the modality and the data size." The rows set batch sizes; every
actual game comes from the environment. So train/test parquet overlap is not a
meaningful question, and regenerating the parquets only redraws sizes -- which is
nevertheless why the G2PO reference arm skips that step, to avoid disturbing anything.

**What follows:**

1. **This does NOT explain the 15-point gap.** We use the identical split -- it is the
   code default both trees inherit, verified in H-AB. Both 95.0 and 79.7 are valid_seen
   numbers.
2. **It DOES inflate the absolute level.** 95% on seen scenes is not general ALFWorld
   competence, and neither would ours be. Any writeup should say *which* split.
3. **Every published baseline shares it**, so the ranking is fair even though the level
   is optimistic. Do not "correct" for it in a comparison; do disclose it.


### The binding constraint is the pid budget, not the GPU count

The container now shows **6 GPUs**, two of them idle. **That does not mean two arms can
run.** Measured 2026-09-08:

```
cgroup pids.max                       8192
one 4-GPU training arm alone         ~7700   (94%)
+ a second Ray cluster                8133   (99%)  -> second run dies
```

The second run died **silently immediately after "Started a local Ray instance"** --
no traceback, no error, one line then nothing. That is the signature to recognise: a
silent death right after Ray init is pid exhaustion, not a config or CUDA fault.

**Check `cat /sys/fs/cgroup/pids.current` BEFORE launching anything alongside a running
arm.** With ~400 pids of headroom, nothing else fits: not a second training arm, not an
eval-only pass, not a 2-GPU probe. The env workers dominate the count (128 envs), so
the footprint barely depends on how many GPUs a run is given.

Consequence for planning: **more GPUs did not buy concurrency.** Extra GPUs can only
make a single arm faster (or allow a wider batch), and multiple seeds must be run
SEQUENTIALLY. Any plan that assumes "8 GPUs means two 4-GPU arms" is wrong on this box
unless the per-run pid footprint is cut first.

### Killing by pattern matches your own shell

`pgrep -f evalck` / `pkill -f <pattern>` match the inline bash command that CONTAINS the
pattern -- i.e. the shell running the kill. This has now caused two incidents. Kill by
PID read from `train.pid`, and check `$$` before killing anything a pattern returned.


### A RESUMED run restarts its evaluation draw — resumed series are OFFSET (2026-09-08)

The checkpoint contains only `actor/` and `data.pt`. **No environment state.** And
`make_envs()` runs at `main_ppo.py:71`, *before* `_load_checkpoint()` at
`ray_trainer.py:1140`. So a resumed run rebuilds its validation environments with
fresh per-seed iterators and **replays the game sequence from the beginning.**

Detrended residual correlation (quadratic trend removed, so this tests the shared
*draw* rather than the shared learning curve):

| pair | n | residual corr | perm p |
|---|---|---|---|
| `global` vs `hardedge` (both fresh) | 10 | +0.667 | 0.083 |
| `global` vs `localstd` (both fresh) | 16 | +0.669 | **0.008** |
| `hardedge` vs `localstd` (both fresh) | 10 | +0.773 | **0.006** |
| `global` vs `long` (**long RESUMED**) | 16 | +0.235 | 0.371 |

**Fresh runs share the draw. Resumed runs do not.**

**Affected runs:** `ccpo-long-20260906` (resumed at step 20 — its whole series from
step 25 is offset) and `ccpo-global-ext-20260908` (resumed at 100 — its step-105
evaluation uses the games the original saw at **step 5**).

**Consequence for the extension's "saturation" reading:** comparing
`ccpo-global-ext` numbers (early draws) against `ccpo-global`'s step-100 number (late
draw) is **not the same measurement**. The flat readings may still be real, but that
comparison is confounded by whatever difficulty difference exists between draws.
**Do not quote a ceiling from it** until the step-100 and step-125 checkpoints are
evaluated on the *same* draw — a short eval-only job.

### And a correction to how the pairing claim was first justified

The original note claimed shared draws on the strength of **raw** correlations
(+0.946 etc.). That statistic was confounded: both arms climb from ~10% to ~80%, and
that alone produces high correlation regardless of the draw. The **detrended residual**
above is the correct test. It happens to agree for fresh runs — but the first
justification was wrong, and the resumed case is exactly where raw correlation misleads
(+0.929 raw, +0.235 detrended).

### Other notes


### Arms SHARE their evaluation draw — matched-step comparisons are PAIRED (2026-09-08)

Validation environments are built with `seed = env.seed + 1000`, identically in every
run (`env_manager.py:665`). Each worker shuffles the game list with its own seed and
iterates in lockstep, so **at step *k* every arm evaluates the same 128 games.**

Measured correlation between arms' held-out series at matched steps:

| pair | evaluations | corr |
|---|---|---|
| `global` vs `hardedge` | 10 | **+0.946** |
| `localstd` vs `global` | 8 | **+0.810** |
| `global` vs `memory` | 4 | +0.482 |

Both `localstd` and `global` dropping from 35.9 to ~22 at step 40 is not coincidence —
it is the same task draw being hard for both.

**This corrects a mistake I made throughout the session.** I repeatedly quoted a
**±13-point** band and used it to dismiss between-arm differences. That figure is
correct for *one arm's* evaluation-to-evaluation movement (the task set genuinely
changes between steps, per the note below). It is **wrong for comparing two arms at
the same step**, because the shared draw cancels.

```
localstd vs global, 8 PAIRED evaluations
  mean delta  −1.27
  sd of delta   5.21      <- not 13
  SE            1.84
  95% CI    [−4.88, +2.34]
```

The paired CI is roughly **three times tighter** than the unpaired band implies.

**Consequences for earlier readings:**

* The `localstd` step-30 delta of −11.0 was a genuinely large deviation (≈2 sd), not
  routine noise. Its reversion at step 35 is what made it uninformative, not its size.
* The CCPO on/off ablation's mean of **−0.01 over nine paired evaluations** is a
  *tighter* null than presented — the CI there is roughly ±2.8, not ±13.
* Several "both readings are inside the noise band" dismissals were too generous.

**Rule:** use the **paired delta sd** for between-arm comparisons at matched steps,
and the ±13 single-evaluation band only for judging one arm's own trajectory. Compute
the delta sd from the arms being compared rather than assuming either figure.

### Other notes


### Ad-hoc offline φ used Python's salted `hash()` — findings reproduce, examples did not

Several offline analyses (H-Q, H-T, H-V, the gate probe) rebuilt a bag-of-words φ with
a local helper using Python's built-in `hash()`. **That is salted per process**, so the
hashed projection differed between runs. The production code does not have this bug —
`FrozenPhi._bag` uses `_seed_of` (sha1).

**Impact on the recorded numbers: none that matters.** H-V re-run with the production
hash under `PYTHONHASHSEED=0`:

| | deterministic | salted |
|---|---|---|
| unstandardised | 0.8300 | 0.8299 |
| task-level sd | 0.8446 | 0.8446 |
| per-node sd (G²PO) | 0.8979 | 0.8968 |
| CONTROL σ shuffled | 0.8814 | 0.8860 |
| **φ-weighted local σ** | **0.9712** | 0.9711 |

Every figure within 0.005. A 1024-bin hashed bag is a random projection either way and
the salt only permutes which words collide, so aggregate statistics over thousands of
samples are stable.

**What it did change: which individual pairs surface.** Two runs of the same
"show me representative neighbours" query returned different pairs, because the
ranking by φ-distance is salt-dependent even when its distribution is not.

**Rule:** any offline analysis that inspects *specific* occurrences — worked examples,
qualitative pairs, anything quoted individually — must use `_seed_of`, not `hash()`,
or it cannot be reproduced. Aggregate statistics tolerate the salt; examples do not.

### Other notes


### The three baseline papers use THREE DIFFERENT training budgets — corrected 2026-09-07

| paper | ALFWorld budget | headline (Qwen2.5-1.5B) | source |
|---|---|---|---|
| **G²PO** | **100 iterations** | G²PO 95.0 | "each for 100 iterations" |
| **GiGPO** | **150 iterations** | GiGPO 86.7 | "each for 150 iterations" |
| **HGPO** | **160 iterations** | HGPO 92.77, GiGPO 90.16 | "160 training iterations" |

**Consequence: our 100-iteration runs are a matched comparison to G²PO ONLY.**
Every table in this file that placed our number beside GiGPO's 86.7 or HGPO's
90.16/92.77 was comparing across a 50-60 iteration deficit. Those rows are *not*
matched-budget and must be labelled as such.

Note also that **G²PO compared their own 100-iteration result against GiGPO's
150-iteration numbers** (their Table 1 baseline rows are byte-identical to GiGPO's
Table 1, standard deviations included) and still reported a win. Their 95.0 is
therefore a genuinely hard, conservatively-established bar.

### Retraction: GiGPO's ALFWorld hyperparameters are identical to ours

Earlier today, asked why we underperform, I claimed GiGPO's paper used
`train_data_size=256`, `group_size=5` and validation temperature 0.0, and offered
that as a candidate source of the gap. **That was a misreading** — those are the
*Search-Augmented QA* hyperparameters. GiGPO's ALFWorld section specifies:

> group size 8, 16×8 = 128 environments, rollout temperature 1.0, validation
> temperature 0.4, mini-batch 256, KL coefficient 0.01, γ 0.95, prompt 2048,
> response 512, 50 environment steps, lr 1e-6, reward 10 / −0.1 invalid

Every one of those matches our configuration. There is no hyperparameter deviation
from GiGPO, and that line of suspicion was mine, not the data's.

### Neither G²PO nor GiGPO labels the evaluation split

Zero occurrences of "unseen", "in-distribution", "out-of-distribution",
`valid_seen` or `valid_unseen` in either paper. The split is settled only by code:
`eval_dataset` defaults to `eval_in_distribution` in both repos and neither ALFWorld
script overrides it. The In-Success/Out-Success split comes from **HGPO**, a
separate re-implementation — which is why HGPO reports GiGPO at 90.16 where GiGPO's
own paper reports 86.7.

### Other notes


### The evaluation set is NOT fixed — corrected 2026-09-07

Every table and comparison earlier in this file describes validation as "128 fixed
held-out episodes". **That is wrong.** ALFWorld's TextWorld env
(`textworld/gym/envs/textworld_batch.py`) does:

```
rng = np.random.RandomState(seed)          # shuffle the game order
gamefiles = [next(self._gamefiles_iterator) for _ in range(self.batch_size)]
```

Each worker shuffles the eval pool with its seed and **iterates**. `reset()`
advances to the next game, so **every validation evaluates a different draw of 128
episodes.** There is no held-out set held constant across steps.

**Consequences, and they are large:**

* The ±13-point noise floor measured on `ccpo-global` is not merely temperature-0.4
  sampling — it is a fresh task draw each time. Step-to-step deltas below ~13
  points carry no information, and I misread several as signal during that run.
* Any matched-step comparison between two arms compares *different task sets*.
  The `ccpo-global` vs `ccpo-hardedge` deltas are noisier than they look.
* Only endpoint numbers, or means over many evaluations, are worth weighing.

**This is NOT a deviation from the baselines** — it is the reference harness's own
behaviour, so G²PO/GiGPO/HGPO numbers are produced the same way. It does explain
why they report a mean over **3 seeds** and we report 1 draw.

### `val_batch_size` 64 vs G²PO's 128 — a real but narrow deviation

`_val_envs` is built once with `env_num = val_batch_size` and each worker is seeded
`seed + i`, so:

* **G²PO:** 128 workers × 1 game per validation = 128 games from 128 shuffles.
* **Ours:** 64 workers × 2 games per validation = 128 games from 64 shuffles.

Same pool, same count, different draw structure. Set to 64 because of the cgroup
pid ceiling (`pids.max=8192`; we run at ~7,750 threads during training), so raising
it needs testing rather than a config flip. Everything else in the protocol matches:
split `eval_in_distribution`, 128 episodes, val temperature 0.4, `do_sample=True`,
`test_freq=5`, 50 env steps, reward 10 / −0.1 invalid.

### Other notes


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
