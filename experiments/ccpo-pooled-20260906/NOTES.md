# ccpo-pooled-20260906

**Question.** Does CCPO's context-conditioned step credit beat the GiGPO step
credit on an identical harness?

**Arm.** `--arm ccpo` against `gigpo-repro-20260906`. Everything else is the shared
default; `diff <(jq -S .config .../config.json) ...` should show only `arm` and the
CCPO knobs.

**Watch, in order.**
1. `plots/ccpo.png` → "lambda* by rule". If `ccpo/lam_eb_obs` sits at 0 the
   per-occurrence rule is inert and CCPO *is* the uniform baseline: the arm cannot
   differ from GiGPO no matter how long it trains, and `ccpo/r_vs_gigpo` will read
   1.0 to confirm it. `ccpo/lam_pooled_obs` on the same panel says what the pooled
   rule would have used.
2. `ccpo/effect_rel` — the estimator's departure from the uniform baseline
   relative to |A|. Near zero means the same thing by a different route.
3. `ccpo/bucket_singleton_frac` and `ccpo/live_frac` — how much of the batch the
   step term can act on at all.
4. Only then the success rate.

**Result.** _pending_

**Reading.** _pending_
