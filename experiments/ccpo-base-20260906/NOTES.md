# ccpo-base-20260906

**Question.** Does CCPO's context-conditioned step credit beat the GiGPO step credit on an identical harness?

**Cell of the 2x2.** estimator = **CCPO**, memory = **off**. See
[`../PLAN.md`](../PLAN.md). Differs from `gigpo-repro-20260906` in exactly
the estimator (`arm`, and `ccpo_shrink` which has no meaning outside CCPO) — verify with:

```bash
diff <(jq -S .config ../gigpo-repro-20260906/config.json) <(jq -S .config config.json)
```

**Budget.** 20 steps, four held-out evaluations. Held-out evaluation carries
~±0.048 on 128 episodes, so only differences beyond ~0.1 are readable at this
budget; anything inside that band should be reported as not separable, not ranked.

**Watch.** `plots/ccpo.png` first, before any success curve. `ccpo/lam_eb_obs` vs `ccpo/lam_pooled_obs` — the shipped `eb` rule measured λ=0.000 on a real batch, which would make this arm numerically identical to the baseline; `eb_pooled` is used for exactly that reason. `ccpo/effect_rel` ≈ 0 or `ccpo/r_vs_gigpo` ≈ 1.0 means inert. Also `ccpo/n_eff_mean` and `ccpo/bucket_singleton_frac`, which are the first valid production measurements of bucket support this project will have.

**Result.** _pending_

**Reading.** _pending_
