# gigpo-mem-20260906

**Question.** Does compaction help on its own, without any estimator change?

**Cell of the 2x2.** estimator = **GiGPO**, memory = **on**. See
[`../PLAN.md`](../PLAN.md). Differs from `gigpo-repro-20260906` in exactly
`compact_budget` — verify with:

```bash
diff <(jq -S .config ../gigpo-repro-20260906/config.json) <(jq -S .config config.json)
```

**Budget.** 20 steps, four held-out evaluations. Held-out evaluation carries
~±0.048 on 128 episodes, so only differences beyond ~0.1 are readable at this
budget; anything inside that band should be reported as not separable, not ranked.

**Watch.** Held-out success against `gigpo-repro`. Also `prompt_length/mean` (the digest costs prompt tokens) and `episode/length/mean` — the mechanism claim is that remembering where it already searched shortens episodes. 58.7% of turns revisit an already-seen observation, and failures revisit 2.68x as often as successes.

**Result.** _pending_

**Reading.** _pending_
