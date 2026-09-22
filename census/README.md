# Census evaluation — play every held-out task exactly once

Ported from the 1.5B line, where it replaced multi-seed sampled evaluation entirely.

## The problem it fixes

Neither environment hands out its held-out tasks exactly once.

* **ALFWorld** gives each worker its own shuffled copy of the 140-game pool and
  `reset()` takes the next one. Games collide across workers.
* **WebShop** samples without replacement *within* a reset, then redraws
  independently on the next one, so multi-reset passes repeat goals.

An N-episode pass over a pool of M therefore lands on about `M·(1−(1−1/M)^N)`
distinct tasks and repeats the rest — roughly 84 of ALFWorld's 140, roughly 63%
of WebShop's 500. Repeated tasks are counted twice, so the reported figure is a
re-weighted average over a sample rather than the split average, and **the
weights change with the seed**.

It also means the *number of validation envs* silently changes which games are
played. 128 envs × 1 reset and 64 envs × 2 resets are different samples, not two
chunks of one — halving `VAL_BATCH` is not a batching-only change.

## What it gives you

With coverage pinned, every task is played once and **sampling variance is
exactly zero**. Two consequences:

1. There is no draw left for `env.seed` to move, so it is fixed and the
   **decoding seed** becomes the only source of run-to-run difference. Vary
   `+actor_rollout_ref.rollout.seed` to put an error bar on the census.
2. Any sampled evaluation can be reproduced **offline** from the per-episode
   dump. Five seeds of GPU time become one pass plus a free re-draw — and you get
   the whole sampling distribution rather than five points from it.

## Use

```bash
census/apply_census_patch.py /workspace/baselines/verl-agent    # once per tree
census/apply_census_patch.py /workspace/baselines/G2PO

scripts/census.sh hgpo 7b alfworld <ckpt> 0,1,2,3 101           # decode seed 101

census/census_draw.py <run>/outputs/episodes.jsonl --seeds 997 101 3173 869 2917
census/census_draw.py <run>/outputs/episodes.jsonl --draws 20000
census/census_draw.py <run>/outputs/episodes.jsonl --enumerate 20000 --top 3
```

`--draws N` reports the sampling distribution of a 128-task evaluation: mean, sd,
the analytic sd, min/max and p5/p95. The analytic sd is
`sqrt(S²/n · (1 − n/N))` with `S²` the finite-population variance; the
`(1 − n/N)` correction is not optional — on WebShop it is 0.744 and dropping it
overstates the spread by about 16%, and on ALFWorld it is 0.086, which is why a
128-of-140 draw is already almost as precise as the full pass.

## Two traps

**ALFWorld task ids are not unique by task directory.** Three tasks in the split
have two trials each, so 140 games carry only 137 distinct task-dir names. The
patch uses `<task dir>/<trial dir>`. A dump written before that fix is still
usable — `census_draw.py` disambiguates repeats by order of appearance, which is
exact because census mode pins worker *i* to game *i* from the same deterministic
`game_files` list, so the k-th row with a given label is the same game in every
arm's dump.

**Per-task-type keys are not per-episode.** ALFWorld's `<type>_success_rate`
lists hold one entry per episode *of that type*, so they are shorter than the
batch and indexing them by batch position pairs a row with a different episode's
result. The dump writes only series whose length equals the batch; the type is
recoverable from the task id.

## Interpreting the numbers

On ALFWorld the finite-population correction makes a 128-of-140 draw nearly as
tight as the census (sd ≈ 0.9–1.2 points on the 1.5B arms), so the multi-seed
spread that motivated five seeds is mostly **repeat-sampling noise this patch
removes**, not irreducible evaluation noise. On WebShop a 128-of-500 draw still
carries sd ≈ 3.3–3.7, so there the census buys a great deal more.

A caution that survives all of this: the census fixes the *task* draw, not the
policy. One census is one decoding realisation; quote several decode seeds as
mean ± std before treating a few-point gap between methods as real.
