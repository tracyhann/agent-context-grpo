# g2po-harness-resume-20260910

**Question.** One sentence: what does this run decide?

**Arm.** How it differs from the comparison arm — name the exact config keys.

**Watch.** Which curves in `plots/` answer the question, and what value would count
as a positive result. Fill this in *before* the run finishes.

**Result.** Numbers with the step they came from. Held-out evaluation here is
stochastic by design (T=0.4, 128 episodes, ~±0.048), so quote the best-checkpoint
score and the mean of the last few evaluations, never a single point.

**Reading.** What it means, including for the negative case.

## Launch (2026-09-11 05:35, from the queue after HGPO)

Resumed correctly from `g2po-harness-20260910/.../global_step_5`: actor weights loaded on
all 4 ranks, and `[acg] resumed at step 5; advancing the validation draw by 2 resets so it
matches a fresh run`, so the held-out games are identical to a fresh run's.

**Caveat: no dataloader state.** The step-5 checkpoint was taken mid-pause and has no
`data.pt` (`No dataloader state found ... will start from scratch`). So the TRAINING
task order restarts at the beginning of the (seeded) dataset: steps 6-10 see the same task
batches as steps 1-5, and the whole order is shifted by 5 relative to a clean run. Weights,
optimizer state and evaluation are unaffected. It is a small departure (5 repeated
batches out of 100), and it should be named whenever this arm is compared step-for-step.

GPUs at launch: ours on 0-3 (~25 GB each at startup); the other tenant is on 4 (17 GB)
and 5 (28 GB).

## STOPPED at step 6 (2026-09-11 ~05:50), user decision

The user asked for this arm not to run and for the GiGPO reference to take GPUs 0-3
instead. The queue watcher was stopped first, so it could not hand the GPUs to `g2po-aff`,
and then the trainer and Ray tree were SIGKILLed. No checkpoint beyond the source's
`global_step_5` exists, so nothing was lost. The harness-vs-estimator question this arm
was meant to answer stays open. `g2po-aff` / `g2po-affonly` (still queued, behind GiGPO)
answer the more useful version of it.
