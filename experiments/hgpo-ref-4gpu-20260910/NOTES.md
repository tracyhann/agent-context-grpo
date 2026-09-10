# hgpo-ref-4gpu-20260910 — HGPO reference reproduction, 4 GPUs, this container

**Published: 92.77 in-distribution @ 160 iterations**, Qwen2.5-1.5B, K=2. HGPO ships in
`baselines/verl-agent` as `recipe/hgpo`; this runs it unmodified via `run.sh`.

## Why a new id, not `hgpo-ref-20260910`

That directory is the configuration record for the run handed to **another container**
(2 GPUs, 4-5). On 2026-09-10 the user asked for HGPO to be chained here as well, on 4
GPUs after `ccpo-refined`. A separate id keeps the two runs from sharing one output
path; the objective-side config is identical (diff the two `run.sh`).

## Deviations from `hgpo-ref-20260910` (none touch the objective)

* GPUs `4,5` -> **`0,1,2,3`**: same GPU count as our arms and the G2PO reference. TP=1
  gives 4 data-parallel ranks; train_batch 16 and val_batch 128 divide 4.
* `save_freq` 40 -> **20** with **`max_actor_ckpt_to_keep=2`**: the trainer rotates, so at
  most two ~25 GB checkpoints ever exist (140 + 160 at the end). Their trainer has no
  best-checkpoint logic; after the run the retained ones are labelled `stepX-best` /
  `stepX-last` (best = over retained checkpoints, stated as such).
* `RAY_TMPDIR=/tmp/ray_hgpo4`.

All deviations from HGPO's own script (TRITON_ATTN, no fa_stub, HOME, gpu_mem 0.3, TP=1,
console logger, prepare skipped, explicit checkpoint dir) are listed in `run.sh`.

## Comparability

* Same `test.parquet` (128 games, T=0.4, test_freq 5) as every arm and the G2PO reference.
* **Budget:** headline at 160 iterations; our arms and G2PO run 100. Report **both** its
  step-100 held-out (budget-matched) and its 160 endpoint, labelled.
* **Step-100 results and weights are kept (user requirement, 2026-09-10).** Results: step
  100 is a test_freq=5 evaluation point, so its held-out numbers land in `metrics.jsonl`.
  Weights: rotation (`max_actor_ckpt_to_keep=2`) would delete `global_step_100` at step
  140, so `scripts/snapshot_ckpt.sh` (started 2026-09-10, log `outputs/snapshot.log`)
  waits for the step-100 save to complete and hard-links it to
  **`checkpoints/step100-budget`**. Final checkpoint set: `step100-budget`, plus
  `stepX-best` / `stepX-last` from the retained 140/160.
* Metrics: verl console output lands in the Ray worker log; `scripts/ref_metrics.sh`
  mirrors it to `outputs/metrics.jsonl` every 5 min and re-renders the plots.

## Launch

Chained by `scripts/queue_anchor_aff.sh`: starts after `ccpo-refined` exits and GPUs
0-3 / RAM / disk are free; the G2PO-harness arms follow it. Expected wall-clock ~15-17 h
(G2PO's 100 steps took ~9.3 h on 4 GPUs; HGPO has 1.6x the steps and a 4096-token prompt cap).

## Launch (2026-09-10 13:26) and resource footprint

Launched by the queue after ccpo-refined was stopped at step 50 (resource check: host RAM
265G, GPUs 0-3 free, disk 592G). No vLLM memory warnings in the log. At 13:30, GPU memory
was 3.6-27 GB per card (of 96 GB).

**Pid footprint: ~13.8k of the container's 20k pids.max**, against ~7.7k for one of our
arms. It is nearly all threads: 256 `AlfworldWorker` actors (128 train + 128 val, both
built at startup by `make_envs`, so evaluation adds no new environment workers) plus ~225
`ray::IDLE` workers, each carrying dozens of threads. **No second arm can run
alongside it in this container.** A monitor alerts at 16k, 18k and 19.5k.

## Pace (measured at step 4, 14:00)

**442 s/step**, so 160 steps take ~19.7 h and the run should end around **09:00 on
2026-09-11**. Step 100 (the budget-matched checkpoint) lands around **01:30 on
2026-09-11**. That is slower than the ~15-17 h estimated from G2PO (~335 s/step): HGPO
uses a 4096-token prompt cap against G2PO's 2048. The first three metric rows parsed
cleanly (73 fields per step: KL loss, pg loss, rewards, advantages, valid-action ratio).
