# ccpo-mem-20260906

**Question.** Does compaction make phi carry signal? This is the last untried
lever on the one mechanism that has never fired.

**Why this is the remaining candidate.** Across four phi/target variants lambda is
exactly 0, and the two obvious causes are ruled out:

* **Not the shrinkage rule.** `Var(d) = s^2 (1/n_eff - 1/J)` is exactly the
  permutation null -- verified numerically at 1.0005 +/- 0.0155 over 400
  configurations x 4000 permutations. So lambda asks the right question ("does the
  actual phi weighting shift the baseline more than a *random* reweighting of the
  same neighbours?") and answers no.
* **Not the kernel.** From 17,352 dumped samples, `n_eff/J` = 0.877 and only 35.6%
  of occurrences have near-uniform weights. The weights vary; they vary in a
  direction uncorrelated with the returns. Sharpening the kernel would enlarge
  `|d|` without making it informative, and the permutation null would keep
  correctly reporting zero.

So the fix has to be a phi that carries signal. Compaction is the mechanism:
inside a bucket the observation is constant by construction, and without a digest
the prompt carries only `step_count` plus the most recent 2 turns, so phi cannot
see `n_unique`, `revisit` or `progress`. The digest puts the whole episode in the
prompt and hence in the hidden state -- siblings then differ by *what they have
already tried*. It is also the only configuration prior work found beating plain
GRPO (0.625 vs 0.604), never separated from the estimator until now.

**Arm.** `compact_budget=512`; differs from `ccpo-ctx-20260906` in that alone.

**Watch.** `ccpo/lam_u_mean` at step 1 settles it in ~7 minutes. Then
`prompt_length/mean` (the digest costs prompt tokens) and `episode/length/mean`
(the mechanism claim is that remembering where it searched shortens episodes:
58.7% of turns revisit an already-seen observation, and failures revisit 2.68x as
often as successes).

**Caveat.** Compaction changes the prompt, so it moves the policy as well as phi.
That confound is irrelevant to the lambda diagnostic but would matter for any
success-rate claim, which needs a memory-on baseline arm too.

**Result.** _pending_

**Reading.** _pending_
