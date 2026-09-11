# gigpo-ref-20260911 — GiGPO reference reproduction, 4 GPUs

**Published: 86.7 in-distribution @ 150 iterations**, Qwen2.5-1.5B (GiGPO paper; the same
value G2PO's and HGPO's tables report). `baselines/verl-agent` *is* the GiGPO repo, so this
runs its own `verl.trainer.main_ppo` with `adv_estimator=gigpo`, unmodified, via `run.sh`.

## Why now

On 2026-09-11 the user stopped `g2po-harness-resume` (at step 6) and asked for the GiGPO
baseline on the freed GPUs 0-3. With G2PO (91.4 @100) and HGPO (79.7 @100, 93.0 @160)
already reproduced, this completes the reproduced baseline set on our stack.

## Deviations from their script (none touch the objective; full list in `run.sh`)

TRITON_ATTN, no fa_stub, HOME, gpu_mem 0.3, TP 1 on 4 GPUs, console logger (metrics
mirrored from the Ray worker log), prepare skipped (same `test.parquet`), and **checkpoints
added**: theirs saves none (`save_freq=-1`). Here `save_freq=25` + `max_actor_ckpt_to_keep=2`,
with step 100 preserved as `step100-budget` by `scripts/snapshot_ckpt.sh`.

`history_length` = 2 comes from the config default (their script does not set it), the same
K as HGPO and our arms. `val_before_train=True` is kept, so there is a step-0 evaluation.

## Comparability

Same 128-game `test.parquet`, T=0.4, test_freq 5. Report **step 100 (budget-matched to our
arms and G2PO)** and **step 150 (their protocol)**, labelled, with windowed means.

## Launch (2026-09-11 05:49)

GPUs 0-3 free (4/4) after `g2po-harness-resume` was stopped; GPU 5 carries the other tenant.
Hydra composed the full config first (`--cfg job`): gigpo, step_advantage_w 1.0,
mean_std_norm, history_length 2, 150 epochs, save_freq 25, max_actor_ckpt_to_keep 2,
val_before_train true.

**Pid-file slip, fixed within a minute.** Launched from the Claude shell (which has job
control), `setsid ... &` forks, so `$!` (238167) was a wrapper that exited at once. The
queue watcher read that as "GiGPO finished" and could have launched `g2po-aff` onto GPUs
0-3 during GiGPO's low-memory startup. It was killed before it did. `train.pid` was
rewritten to the real `run.sh` pid (**238169**, trainer 238180), and the metrics mirror
and queue were restarted on it with `setsid -f`. Nothing else was affected.
