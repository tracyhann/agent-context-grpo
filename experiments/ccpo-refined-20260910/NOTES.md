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
