# ccpo-mem-20260906

**Question.** Do the estimator and compaction compose, or does each capture the same gain?

**Cell of the 2x2.** estimator = **CCPO**, memory = **on**. See
[`../PLAN.md`](../PLAN.md). Differs from `gigpo-repro-20260906` in exactly
the estimator and `compact_budget` together — verify with:

```bash
diff <(jq -S .config ../gigpo-repro-20260906/config.json) <(jq -S .config config.json)
```

**Budget.** 20 steps, four held-out evaluations. Held-out evaluation carries
~±0.048 on 128 episodes, so only differences beyond ~0.1 are readable at this
budget; anything inside that band should be reported as not separable, not ranked.

**Watch.** Whether it exceeds both single-factor arms. The prior work's only arm beating plain GRPO was frozen-phi + memory (0.625 vs 0.604) but never separated the two; this cell plus the other three does. Compaction also feeds phi — inside a bucket the observation is constant by construction, so the digest is the only block that can tell siblings apart.

**Result.** _pending_

**Reading.** _pending_
