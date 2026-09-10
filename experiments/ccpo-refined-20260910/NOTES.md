# ccpo-refined-20260910 — the design-error fix, with context forced ON

## Why this arm exists

The shipped CCPO (`ccpo-global`, 79.7) used context conditioning to **replace** state
grouping: `ACG_CCPO_GATE=global` puts every occurrence of a task in one bucket and relies
on phi to recover structure. phi is inert (`phi_rel_corr` 0.011), so the kernel
concentrates confidently on arbitrary neighbours. G2PO (91.4 on this box) instead
partitions by `(task, observation + admissible actions)`.

This arm makes context **refine** a good partition instead of replacing it:

| key | 79.7 arm | this arm | why |
|---|---|---|---|
| `ccpo_gate` | global | **hard** | partition by `(task, anchor)`, the one that works |
| `anchor_aff` | — | **1** | admissible set in the anchor: 29.1% of within-node V(next) variance (H-AJ) |
| `obs_repair` | — | **1** | failures stop collapsing onto one shared anchor |
| `ccpo_shrink` | eb | **one** | **forces lambda = 1** (see below) |

These are the ONLY four differences from `ccpo-global-20260907` (verified by config diff).

## Why `ccpo_shrink=one` is required, not cosmetic

Under the hard gate the empirical-Bayes lambda collapses: `tau^2` measures **0.00e+00**,
so lambda -> 0.011 and **98.9% of the weight goes to the plain observation-only
baseline**. With `shrink=eb` this arm would be `g2po-affonly` with a vestigial 1% context
term -- not a context-conditioned method at all. `one` overrides the EB verdict so the
question can be asked. **Not a defensible production setting**: it forces a term the data
calls worthless. It is a probe.

## Validity check at step 1

`ccpo/lam_u_mean` **must read 1.0**, and `effect_rel` should sit well above
`ccpo-hardedge`'s 0.0014. If lambda is not 1, the flag did not take effect and the arm is
invalid. Monitored automatically.

## Prior, recorded before any result

**Expect harm.** H-Q measured phi-weighting inside a bucket at 2-10 R^2 points WORSE than
uniform, and the refined partition has smaller buckets (45,978 vs 17,607), giving phi less
room. Five independent measurements (H-AK) put inferred state equivalence at no signal.

**What would change that:** beating `g2po-affonly` (same partition, no phi) -- the only
comparison that isolates a positive contribution from the soft weighting. A win over the
79.7 arm alone would NOT be enough, since the partition change by itself should account
for most of any gain.

## Step-1 validity check: PASSED (2026-09-10)

```
lam_u_mean   1.0000   (ccpo-hardedge 0.0110)   shrink=one took effect
effect_rel   0.1637   (ccpo-hardedge 0.0014)   context term moves ~16% of |A|
n_buckets    1172     singleton 0.293          matches ccpo-anchor's refined partition
tau2         1.10e-03                          see below -- interpretation pending
```

**This arm genuinely tests context conditioning on a refined partition.** It is not
`g2po-affonly` in disguise: the context term carries ~117x the weight it did under the
obs-only hard gate.

**tau2 is nonzero here, and the like-for-like comparison says it is not step-1 noise.**

| arm | gate | anchor | tau2@1 | tau2 > 0 on |
|---|---|---|---|---|
| **ccpo-refined** | hard | obs + aff + repair | **1.10e-03** | 1/1 steps so far |
| ccpo-hardedge | hard | obs only | 0 | **0 / 50** |
| ccpo-long | hard | obs only | 0 | **0 / 80** |
| ccpo-global | global | obs only | 0 | 1% |
| ccpo-anchor | global | obs + aff + repair | 0 | 19% |

Every other arm read exactly 0 at step 1. **Compare within a gate**, since tau2 is estimated
ACROSS buckets and the gate changes what a bucket is (a node under `hard`, a whole task
under `global`). Within the hard gate: obs-only anchor -> positive on none of 130 steps;
refined anchor -> positive on step 1. The global-gate arms move the same way (1% -> 19%
when affordances are added).

**Only one step. Tracked automatically; do not quote until it has run 20+.**

See H-AL in `experiments/hypothesis.md` for what it would imply.

## Kill rule and H-AL decision — fixed at step 10, BEFORE step 20 exists

Held-out so far: step 5 7.8 (base 10.2), step 10 7.0 (base 10.9) — both within noise.
tau2 positive on 5/10 steps.

**At step 20 (base there = 21.1):**

* **STOP** if held-out < 11% — more than 2 SE below base, i.e. clearly failing. This is the
  same threshold used for `ccpo-anchor`, which shares this partition. The queue behind
  this arm waits on it, so running a failing arm to 100 costs ~9 h of GPUs 0-3.
* Otherwise **run to 50** and judge against base's 50.0. The question this arm answers
  (phi-weighting vs uniform inside the same node) is only settled against `g2po-aff`,
  which is queued behind it — so a middling result here is not a reason to continue
  past 50 on its own.

**H-AL at step 20:** tau2 > 0 on **>= 11/20** steps -> launch `ccpo-refined-eb`
(`shrink=eb` on this partition, the defensible data-driven version). Otherwise H-AL is dead
and H-AK stands as written. Currently 5/10.

These decisions are independent: the arm can fail its kill rule and still pass H-AL (the
partition carries structure even if forced lambda=1 misuses it), which would be the
strongest possible argument for the `eb` arm.

## H-AL resolved at step 15: the eb follow-up is cancelled

The tau2 proxy (9/15 positive) will likely pass at step 20, but the direct measure of what
`shrink=eb` would do -- `lam_eb_obs`, logged throughout -- averages **0.0398** (per step
0.015-0.096, no trend). EB would put ~4% of the advantage on context. A
`ccpo-refined-eb` arm would therefore be `g2po-affonly` with a 4% context term, and is
**not being launched** unless `lam_eb_obs` exceeds ~0.2 by step 20. Full reasoning, and
the admission that the pre-registered proxy was the wrong quantity, in H-AL.

This does not affect this arm's own kill rule at step 20.

## Step 20: kill rule PASSED; H-AL proxy passed but the eb arm stays cancelled

| step | ccpo-refined | base | diff |
|---|---|---|---|
| 5 | 7.8 | 10.2 | -2.3 |
| 10 | 7.0 | 10.9 | -3.9 |
| 15 | 14.1 | 17.2 | -3.1 |
| 20 | 14.8 | 21.1 | -6.2 |

**Kill rule (fixed at step 10): stop below 11%. 14.8 -> PASS, runs to 50.** All four
evaluations trail base, each within noise -- the same shape as `ccpo-anchor` (same refined
partition, 15.6 at step 20), which went on to finish unconverted. Next judgement: step 50
against base's 50.0, and ultimately against `g2po-aff`.

**H-AL:** tau2 positive on 13/20 (65%), so the pre-registered proxy passes. The direct
measure does not move: `lam_eb_obs` mean 0.0413 (0.0424 over steps 1-10, 0.0402 over
11-20), max 0.0956, 0.037-0.060 across steps 16-20. Below the ~0.2 exception by a factor
of four. **`ccpo-refined-eb` stays cancelled.**

## VERDICT at step 50: FAIL. Forcing context onto G2PO's partition hurts.

| step | 5 | 10 | 15 | 20 | 25 | 30 | 35 | 40 | 45 | 50 |
|---|---|---|---|---|---|---|---|---|---|---|
| ccpo-refined | 7.8 | 7.0 | 14.1 | 14.8 | 22.7 | 25.0 | 28.9 | 20.3 | 25.8 | **36.7** |
| base | 10.2 | 10.9 | 17.2 | 21.1 | 21.1 | 30.5 | 35.9 | 22.7 | 41.4 | **50.0** |
| diff | -2.3 | -3.9 | -3.1 | -6.2 | +1.6 | -5.5 | -7.0 | -2.3 | -15.6 | **-13.3** |

* Pooled over all 10 evaluations: **-5.8**, with 9 of 10 negative. Over the window of
  steps 30-50: **-8.7**. Steps 45 and 50 individually exceed 2SE (±11.6 and ±12.3).
* The G2PO reference was at 79.7 by step 50, so the gap to SOTA is far larger than the gap to base.
* By task type, the losses concentrate in clean, cool and heat (-18 to -24 at step 50), plus
  pick_two (-18). Only look_at_obj is ahead (+7).
* Validity held throughout: lam = 1.000 on every step, and context carries 17.6% of |A|.
  So this is a genuine test of the context term, and it did not help.
* tau2 was positive on 34/50 steps (68%), against 0/130 under the observation-only
  partition. Once the partition is right, context has real structure to explain. But
  empirical Bayes would weight it at ~4% (`lam_eb_obs`), and forcing it to 100% costs
  success rate. **The context structure exists, but it is too weak to act on at full
  weight.**

**Stopped at step 50** (total_epochs was 100, and early stopping would not fire because
step 50 was the best yet). This was the pre-registered judgement point, and the user
queued HGPO to follow it. The trainer and Ray tree were SIGKILLed after
`latest_checkpointed_iteration.txt` read 50. Checkpoints kept: `step50-best` and
`step50-last`, the same step. To resume: `resume_from=.../global_step_50` under a NEW
name, because `launch()` rm -rf's its target directory.
