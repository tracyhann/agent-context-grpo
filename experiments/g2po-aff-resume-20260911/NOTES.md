# g2po-aff-resume-20260911

**Question.** One sentence: what does this run decide?

**Arm.** How it differs from the comparison arm — name the exact config keys.

**Watch.** Which curves in `plots/` answer the question, and what value would count
as a positive result. Fill this in *before* the run finishes.

**Result.** Numbers with the step they came from. Held-out evaluation here is
stochastic by design (T=0.4, 128 episodes, ~±0.048), so quote the best-checkpoint
score and the mean of the last few evaluations, never a single point.

**Reading.** What it means, including for the negative case.
# g2po-aff-resume-20260911 — FAILED AT STARTUP (kept as the record)

Relaunch of `g2po-aff-20260910` (killed externally at step 16) from its `global_step_15`.
It died in `init_workers` -> vLLM `sleep(level=1)`:

    AssertionError: Memory usage increased after sleeping.

**Cause: another tenant took GPUs 0-3.** After killing every process of ours (verified:
0 compute apps under our uid), GPUs 0-3 still showed **36-38 GB used each at 72-90%
utilization**, and 4-5 were also busy (67 / 49 GB). From inside the container nvidia-smi
lists only our own processes, so that load is another container's. vLLM measures free
memory, then sleeps and asserts that memory did not grow; a neighbour allocating in that
window trips it. Nothing about the resume itself is wrong.

**This probably also explains the 22:27 kill of `g2po-aff`** (no traceback, no OOM, a bash
watcher dying with it) - consistent with the GPUs being taken over - but that remains
unproven, so it stays recorded as unexplained.

**Decision: wait, do not share.** The standing instruction is not to interfere with other
users. Our arm peaks at ~45-60 GB per GPU and only ~60 GB is free per card, so sharing
risks OOMing their job and ours. The queue now waits for GPUs 0-3 to be genuinely free
(>=80 GB each, up to 24 h) and then relaunches the resume as `g2po-aff-r2-20260911`.

## Correction (22:36): the neighbour's load FLUCTUATES

At 22:35:28 all four of GPUs 0-3 reported **>=80 GB free**; 35 s later they were back at
36-38 GB used with 42-65% utilization, with **zero compute processes of ours**. So the
tenant is real (the load is not our dying run), but it cycles between phases the way our
own arms do between rollout and training.

**Consequence:** a single-sample resource check can be fooled by a trough, which is very
likely how the resume came to launch at 22:31 and then hit vLLM's
"Memory usage increased after sleeping" -- it started in a gap and the neighbour
re-allocated during rollout init. `wait_ram` in `scripts/queue_anchor_aff.sh` now requires
**5 consecutive free samples 60 s apart** before launching.
