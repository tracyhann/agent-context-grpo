# g2po-harness-20260910

**Question.** One sentence: what does this run decide?

**Arm.** How it differs from the comparison arm — name the exact config keys.

**Watch.** Which curves in `plots/` answer the question, and what value would count
as a positive result. Fill this in *before* the run finishes.

**Result.** Numbers with the step they came from. Held-out evaluation here is
stochastic by design (T=0.4, 128 episodes, ~±0.048), so quote the best-checkpoint
score and the mean of the last few evaluations, never a single point.

**Reading.** What it means, including for the negative case.

## PAUSED at step 4/5 (2026-09-10) — how to resume, and why the checkpoint is odd

Paused to free GPUs 0-3 for `ccpo-refined`. `checkpoints/global_step_5` is **partially
written but resumable**:

| part | state | matters for resume? |
|---|---|---|
| FSDP shards (model / optim / extra, all 4 ranks) | **complete** — 21 files, each within 1% of a known-good checkpoint | **yes — this is what loads** |
| `actor/huggingface/` export (12 files) | missing | no — inference only |
| `data.pt` (dataloader position) | missing | no — verl warns and restarts the dataloader; rows are placeholders and the env draw is seeded |
| `latest_checkpointed_iteration.txt` | missing | **only for `resume_mode=auto`** |

**Resume with an EXPLICIT path, never auto** (auto looks for the missing marker):

    python3 scripts/exp_run.py --name g2po-harness --arm g2po \
      --set resume_from=/workspace/experiments/g2po-harness-20260910/outputs/checkpoints/global_step_5 \
      --set gpus=0,1,2,3 --set total_epochs=100 --set compact_budget=0 \
      --set obs_repair=0 --set anchor_aff=0

Last *logged* step is 4. Step 5's metrics and evaluation were never written, so the
resumed run's first evaluation will be step 10.

**Why it is partial — a bug in the pause chain.** `pause_and_launch_refined.sh` waited
for the checkpoint DIRECTORY to exist, which happens when the save *starts*, then sent
SIGTERM. The trainer ignored SIGTERM (second recorded occurrence) and was SIGKILLed during
the HF export. Correct gate: wait for `latest_checkpointed_iteration.txt` to name the
step, which verl writes only after the save completes.
