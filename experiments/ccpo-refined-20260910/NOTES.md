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
