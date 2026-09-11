# g2po-aff-20260910

**Question.** One sentence: what does this run decide?

**Arm.** How it differs from the comparison arm — name the exact config keys.

**Watch.** Which curves in `plots/` answer the question, and what value would count
as a positive result. Fill this in *before* the run finishes.

**Result.** Numbers with the step they came from. Held-out evaluation here is
stochastic by design (T=0.4, 128 episodes, ~±0.048), so quote the best-checkpoint
score and the mean of the last few evaluations, never a single point.

**Reading.** What it means, including for the negative case.

## Launch (2026-09-11 20:22, from the queue after GiGPO)

pid 305250 on GPUs 0-3. (The `-20260910` suffix is the queue's fixed date, which keeps its
pidfile paths deterministic; the actual launch was on 09-11.) Verified in the live
process environment and command line:

* `adv_estimator=g2po`: G2PO's estimator, vendored verbatim as `g2po/core_g2po.py`.
* **`ACG_OBS_REPAIR=1`, `ACG_ANCHOR_AFF=1`: our state-grouping fixes are ON.** This is the
  single-variable contrast against the G2PO reference (91.4 @100) and `g2po-harness`.
* `VLLM_ATTENTION_BACKEND=TRITON_ATTN`, 100 epochs, early stop from step 40 (patience 8).
* The `ACG_CCPO_*` variables exp_run sets are **inert here**: `ray_trainer.py` reads them
  only inside the CCPO branch (lines 348-457) and in `core_ccpo.py`. The G2PO branch (l.458)
  calls `core_g2po` alone.

**What it answers:** whether our anchor refinement (failure repair + admissible-action
key), on top of G2PO's own estimator, beats G2PO. Judge it on windowed held-out against
the G2PO reference at matched steps, never on stepN-best.

## Step 1 healthy (20:29); G2PO diagnostics added for the NEXT arm only

Step 1: train success 5.5%, valid-action 0.84, KL 0.016. About 7.8 GB per GPU and 7.6k pids.

**This arm logs no method-specific G2PO terms**: its 74 metrics are verl's standard set.
After it launched, a LOGGING-ONLY block was added to the G2PO branch of `ray_trainer.py`
(mirrored in `patches/`). It writes `g2po/node_size_mean`, `node_size_p90`, `nodes_per_task`,
`singleton_sample_frac` (comp1 is zero for these), `terminal_success_frac`,
`terminal_failure_frac`, `invalid_action_frac` and `adv_absmean`. It reads only; nothing
feeds back into the advantage. It was checked against hand-computed values on a mock batch.
This process imported `ray_trainer` at startup, so the change does not affect it;
**`g2po-affonly` will be the first arm with these metrics.**
