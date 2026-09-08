# Handoff — 2026-09-08

State at container relaunch. Everything below is committed; nothing lives only in a
running process.

## Where the numbers stand

| arm | steps | held-out | note |
|---|---|---|---|
| **`ccpo-global-20260907`** | 100 | **79.7%** | **best result. the comparator for everything.** |
| `ccpo-long-20260906` | 100 | 70.3% | hard gate, no edge term |
| `ccpo-localstd-20260908` | 80 (stopped) | ~59% @75 | coherent standardisation, H-V |
| `ccpo-global-ext-20260908` | 145 | **79.7% mean** | 150-step probe: +0.00 over step 100 |
| `ccpo-hardedge-20260907` | 50 (paused) | 41.4% | CCPO ON/OFF ablation |
| `ccpo-memory-20260907` | 22 (stopped) | 9.4% @20 | digest, H-W |

Published at the same protocol (Qwen2.5-1.5B, ALFWorld, 100 iters, 3 seeds):
**G²PO 95.0**, GiGPO 86.7 (at *150* iters), GRPO 72.8, RLOO 69.7, PPO 54.4.

## The two results that matter

**1. CCPO's context conditioning is inert.** `ccpo-hardedge` vs `ccpo-global` is
context conditioning fully OFF (λ=0.011, effect_rel 0.001) vs fully ON (λ=1,
effect_rel 0.119), everything else identical. Mean difference over 9 paired
evaluations: **−0.01 points**. Corroborated four independent ways — `phi_rel_corr`
0.011 (R² 0.00013), H-H's ICC ceiling of 6.1%, H-Q showing φ-weighting inside a
bucket is *worse* than uniform, and this ablation.

**2. The 79.7% came from G²PO's edge term, not from CCPO.** `edge_w` had defaulted to
0.0 while we were trying to beat G²PO. Turning it on coincided with 70.3 → 79.7.

## Do not quote a `stepN-best` number

Best-checkpoint selection takes the maximum of a noisy series and is biased upward by
~1.5 sd. `ccpo-global-ext` shows it cleanly: mean 79.69, sd 4.97, and `best.json` reads
85.9 — almost exactly the expected maximum of 9 draws. Report the mean over
evaluations, or re-evaluate on a fresh draw.

## Two measurement corrections that change how to read everything

* **Arms share their evaluation draw.** Validation workers are seeded identically in
  every run and iterate in lockstep, so at step *k* every arm sees the same 128 games
  (corr +0.946 between arms). Matched-step comparisons are therefore **paired**, with
  a delta sd of ~5.3 — *not* the ±13 single-evaluation band. Use the paired sd for
  between-arm claims, ±13 only for one arm's own trajectory.
* **A RESUMED run restarts its evaluation draw.** Checkpoints hold no env state and
  `make_envs()` runs before `_load_checkpoint()`, so a resumed run replays the game
  sequence from the start. Detrended residual correlation: fresh-vs-fresh +0.67 to
  +0.77 (p<0.01), fresh-vs-resumed +0.235 (p=0.37). **`ccpo-long` and
  `ccpo-global-ext` are offset from the fresh runs**, so their numbers are not
  directly comparable at matched steps. Settle any such comparison by re-evaluating
  both checkpoints on the same draw.
* **The evaluation set is not fixed.** TextWorld shuffles per worker seed and
  iterates, so every validation draws different games. There is no held-out set held
  constant across steps.

## Open, ranked

1. **H-Y — uncertainty-gated memory.** The digest arm failed with a *diagnosis*:
   `valid_action_ratio` 0.987 from step 1, before training could adapt, so it cost
   output format on every turn. Gate it on `stall >= 2` (already computed in
   `derive_context`): fires on 36% of turns, and those turns have mean V(next) 1.96
   against 2.90 overall. One condition on the `ACG_COMPACT_BUDGET` block in
   `env_manager.py`.
2. **H-S — epistemic weighting.** `w_u = J_u/(J_u+c)` on the step term. Real quantity
   (reliability 0.686 at J=2 vs 0.912 at J=7) but a small predicted effect; a single
   arm probably cannot resolve it against a paired sd of 5.3. Needs multiple seeds or
   an offline criterion first.
3. **G²PO baseline on our harness.** Never run. It is the only way to separate "our
   method is worse" from "our harness caps everyone" — H-O showed our advantage is
   ~0.90 correlated with G²PO's, so a 15-point gap does not follow from the estimator.

## Do not re-run without changing something

* Progress banding the gate (H-J) — loses under both targets.
* Observation-derived memory features in φ (H-T route A) — restate `progress`,
  +0.003 R².
* Hard gate with λ=1 (H-Q) — φ-weighting inside a bucket is 2–10 R² points worse
  than uniform.
* Per-node standardisation on reliability grounds (H-U) — that evidence was an
  artifact; the real gap is +0.004.

## Resuming

Every arm is resumable: `scripts/exp_run.py --name X --arm ccpo --set
resume_from=experiments/<arm>/outputs/checkpoints/global_step_N --set
total_epochs=<absolute target>`. `total_epochs` is absolute, not an increment. Safe
here because the LR is constant with zero warmup.

More GPUs: `train_batch_size` must divide the GPU count. 16 works with 1/2/4/8, not
6. With 8 GPUs, `--set gpus=0,1,2,3,4,5,6,7`.

Full reasoning for every entry above is in `experiments/hypothesis.md`.
