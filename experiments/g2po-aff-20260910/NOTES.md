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
