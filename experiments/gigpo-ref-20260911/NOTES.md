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

## Progress through step 50 (2026-09-11)

| held-out % | 0 | 5 | 10 | 15 | 20 | 25 | 30 | 35 | 40 | 45 | 50 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gigpo-ref | 10.2 | 6.2 | 14.8 | 14.1 | 14.8 | 18.8 | 25.8 | 23.4 | 28.1 | 37.5 | 35.9 |
| G2PO ref | - | 12.5 | 11.7 | 28.9 | 35.9 | 46.9 | 54.7 | 64.8 | 67.2 | 71.9 | 79.7 |
| HGPO | - | 11.7 | 10.9 | 17.2 | 14.1 | 23.4 | 22.7 | 37.5 | 30.5 | 34.4 | 34.4 |

Window 35-50: **GiGPO 31.2**, HGPO 34.2, G2PO 70.9. GiGPO tracks HGPO closely and trails G2PO
by ~40 points at this stage; HGPO's big climb came at steps 50-70. Pace ~454 s/step.

## BUDGET-MATCHED RESULT at step 100 (2026-09-11 16:31)

| held-out % | 85 | 90 | 95 | **100** | window 85-100 |
|---|---|---|---|---|---|
| **gigpo-ref** | 69.5 | 68.8 | 82.0 | **76.6** | **74.2** |
| HGPO (reproduced) | 66.4 | 76.6 | 81.2 | 79.7 | 76.0 |
| ccpo-global (ours, best) | 75.8 | 75.0 | 78.1 | 79.7 | 77.2 |
| G2PO (reproduced) | 89.1 | 87.5 | 95.3 | 91.4 | 90.8 |

**At 100 iterations, GiGPO, HGPO and our ccpo-global form one cluster (74-77 windowed,
within noise of each other), and G2PO sits ~14 points above all three.** GiGPO plateaued
around 50-55 over steps 55-75, then climbed again from step 80.

Weights: `checkpoints/step100-budget` (snapshot 16:31:06, 19 GB counting links, 2 links
per shard, so it survives the trainer's rotation at step 150). Pace over the first 100
steps is ~385 s/step, so the run should end around 22:00, when the queue starts `g2po-aff`.

## FINAL (2026-09-11 20:18): GiGPO reproduces its published number exactly

| held-out % | 105 | 110 | 115 | 120 | 125 | 130 | 135 | 140 | 145 | **150** |
|---|---|---|---|---|---|---|---|---|---|---|
| gigpo-ref | 76.6 | 68.8 | 81.2 | 79.7 | 81.2 | 88.3 | 82.8 | 83.6 | 88.3 | **86.7** |

* **Endpoint 86.7 vs published 86.7. Reproduced.** Window 135-150: **85.4**.
* **Budget-matched at step 100: 76.6** (window 85-100: 74.2), in the HGPO/ccpo-global
  cluster, ~14-17 points below G2PO.
* Finished normally (150/150, "Final validation metrics", step-150 checkpoint saved).
  Wall-clock 14.3 h.

Checkpoints: `step100-budget` (budget-matched), plus `step150-best` = `step150-last` ->
`global_step_150` (best over the retained 125/150: 81.2 vs 86.7; `best.json` records it).
`global_step_125` (19 GB) is neither best nor last. As with HGPO's `global_step_140`,
deleting it waits for the user's OK. `global_step_25`-`_100` hold only `data.pt`.
