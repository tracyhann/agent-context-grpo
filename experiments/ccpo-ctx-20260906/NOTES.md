# ccpo-ctx-20260906

**What this arm is, in practice.** Launched as `phi=hidden+ctx`,
`target=nextnode`, `shrink=eb_hier`. Measured at step 1: **lambda = 0.0000**, so
phi is inert and the estimator reduces to

    A_CC(u) = V(next(u)) - mean over OTHER trajectories of V(next(v))

i.e. **G2PO's component 1 with leave-one-out instead of self-inclusive**. No
configuration reachable here changes what it computes, so it runs on rather than
being relaunched. `r_vs_gigpo = +0.505`, `r_vs_g2po = +0.727` — genuinely
different credit from both references.

**The conditioning result this arm completes.** Four phi/target combinations, all
null, each ~1500 live samples per step:

| phi | target | lambda | effect_rel | r_vs_gigpo |
|---|---|---|---|---|
| `hidden` | return | 0.000 | 0.0000 | +0.977 |
| `hidden` | nextnode | 0.000 | 0.0000 | +0.530 |
| `bow` (explicit context) | nextnode | 0.000 | 0.0000 | +0.523 |
| `hidden+ctx` | nextnode | 0.000 | 0.0000 | +0.505 |

`Var(d) = s^2 (1/n_eff - 1/J)` is the variance of `b_obs - b_LOO` under the null
that the weights are uncorrelated with the returns. `tau^2 = max(0, E[d^2] -
E[Var(d)]) = 0` says the realised disagreement never exceeds that null. **phi
similarity does not predict return similarity inside a bucket** — not from the
policy's hidden state, not from explicit {t, n_unique, revisit, progress}, not
from both.

Support is not the explanation: `n_eff` 4.86-4.93 of a maximum 7, mean bucket size
8.4, 90% of samples credited, 758 buckets. Nor is the shrinkage rule: `eb`,
`eb_pooled` and `eb_hier` independently agree.

This is the production-scale test of the write-up's own E5 ("the central
hypothesis has no evidence yet"), previously a null on 15-22 pairs.

**Watch.** Held-out success against a GiGPO arm on the same reference prompt.
What is being measured here is the target change and the leave-one-out exclusion,
not context conditioning.

**Result.** _pending_

**Reading.** _pending_
