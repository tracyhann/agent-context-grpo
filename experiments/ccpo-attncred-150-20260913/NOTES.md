# ccpo-attncred-150-20260913

**Question.** Does the phi-attention readout (H-AL), a null over 100 steps, separate from
the hard-gate control when the budget runs to 150 -- the horizon GiGPO publishes at?

**Arm.** `ccpo-attncred-20260912` re-run at `total_epochs=150`, plus `pin_steps=100` so the
step-100 checkpoint survives the rolling window. Every other key is byte-identical to the
100-step arm: `ccpo_lam_fix=1.0`, `ccpo_prior_kappa=2.0`, `ccpo_backoff_task=1`,
`ccpo_jweight_c=0.0`, `ccpo_phi=hidden+ctx`, `ccpo_gate=hard`, `ccpo_target=return`,
`ccpo_rho=0.0`, `keep_ckpts=1`, FLASH_ATTN, seed 0. GPUs are assigned at launch.

This is a FRESH run, not a continuation: the 100-step arm's `global_step_100` was pruned
by the rolling window (only `best.json` and the iteration marker remain), so there is
nothing to resume from. Steps 1-100 should reproduce the 0912 run up to sampling noise,
which is itself a free replicate -- the yardstick from `fbjw` is 15 points on a single
step, 1.79 on a window mean.

**Watch.** `plots/progress.png`, held-out `val/success_rate` every 5 steps.
- Steps 1-100: does it track the 0912 curve (window 70-100 was 75.56)? A large divergence
  means seed-level noise dominates anything this ladder has measured.
- Steps 100-150: the question. `ret-hard` ended at 84.4 @100 and was never run past it,
  so there is NO paired control beyond step 100 -- a 150-step number here is comparable
  to GiGPO's published 86.7 @150 and to nothing else in this project.
- `ccpo/effect_rel` should sit at 0.24-0.35 as before; if it collapses to 0.000 the
  readout is not switched on and the run is void.
- Early stop is armed (`patience=8`, `min_steps=40`): 40 steps without a new best ends it
  before 150. That is the intended behaviour, not a failure.

**Result.** _(fill in with the step each number came from; quote the best checkpoint and
the mean of the last few evaluations, never a single point)_

**Reading.** _(including the negative case: if 150 steps does not separate it from 84.4,
the horizon was not the constraint either, and H-AL stays closed.)_
