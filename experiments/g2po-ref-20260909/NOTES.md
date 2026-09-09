# g2po-ref-20260909 — G²PO reference reproduction

**Runs `baselines/G2PO` itself**, not our port. The question is whether G²PO reaches its
published **95.0** on this hardware and stack.

| outcome | reading |
|---|---|
| ~95 | the 15-point gap is **ours**. Both trees run here, so it is diffable. |
| ~80 | our **79.7 is competitive**; the published number does not transfer to this stack. |

Either way this is the number that tells us what 79.7 means. It has been the top open
item since H-AB established that our config is identical to theirs.

## Why run their tree rather than our vendored estimator

Both exist now. They answer different questions:

* **this arm** (their tree) — reproduction. Isolates *everything* of theirs from
  *everything* of ours.
* `--arm g2po` in `scripts/exp_run.py` (our tree, `g2po/core_g2po.py` vendored verbatim)
  — isolates the **estimator** from the **harness**. Worth running only if this arm
  scores ~95, because then the follow-up 2×2 locates the gap precisely.

## Deviations from their `run_alfworld.sh`

All forced by this box; each is listed in `run.sh` with its reason. The ones that could
matter, and why they do not:

1. `VLLM_ATTENTION_BACKEND` XFORMERS → **TRITON_ATTN**. XFORMERS has no sm_120 kernel.
   Our arms use Triton too, so this is not a handicap applied only to them.
2. **Their `data_preprocess.prepare` step is skipped deliberately.** It regenerates the
   parquets, which could draw a different validation set and destroy comparability with
   every number we hold. We point at the existing 128-row `test.parquet` — the size
   their own script requests.
3. 2 GPUs, TP=1 (theirs: 8 GPUs, TP=2); `gpu_memory_utilization` 0.6 → 0.45 because
   another tenant shares these cards. Both change gradient accumulation and KV-cache
   size, not the objective.
4. Console logger only (no wandb). `scripts/parse_g2po_log.py` converts the console log
   into our `metrics.jsonl` format — validated against a known log, where it recovers
   0.797 and the per-type numbers exactly.

**Unchanged**: lr, kl_loss_coef/type, γ, invalid-action penalty, max_steps, group size,
train_batch_size, prompt/response lengths, val temperature and sampling, test_freq,
mini/micro batch sizes, `save_freq=-1`, total_epochs.

`save_freq=-1` is *their* default: this arm writes **no checkpoints**, which also keeps
50 GB off a disk at 97%.

## Comparability

Same 128 `eval_in_distribution` tasks, same T=0.4 sampling, same `test_freq=5`, same
backbone. Directly comparable to `ccpo-global-20260907` (79.7) and to the published 95.0.


## Operational notes discovered at launch (2026-09-09)

Three things bit on the way up; all are in `run.sh` now.

1. **`docker/fa_stub` must NOT be on `PYTHONPATH`.** The real flash-attn (2.8.3.post1)
   is installed and its `unpad_input` is pure PyTorch, so their
   `use_remove_padding=True` works on sm_120. Putting the stub first shadows the real
   package and every call raises. Our own arms never load the stub. **This means NO
   deviation from their config was needed** -- `use_remove_padding` stays `True`.
2. **`HOME` must be a real, writable home** (`/home/claude`). `env -i` clears it, and
   `HOME=/root` makes every Ray worker log `bash: /root/.bashrc: Permission denied`.
   Cosmetic, but the noise looks like a failure.
3. **Their metrics do not reach `train.log`.** verl's console backend `print()`s inside
   the TaskRunner ray actor; Ray forwards only the tqdm bar (stderr) to the driver. The
   metrics land in `/tmp/ray_g2po/ray/session_latest/logs/worker-*.out`.
   `scripts/g2po_metrics.sh` mirrors that file into `outputs/worker_metrics.log` (it is
   volatile) and parses it into `metrics.jsonl`.

**Do not judge liveness from GPU memory.** Startup loads 494 games and the model before
allocating; a healthy run shows ~4 MiB for several minutes. Watch the
`Training Progress` bar in `train.log` instead.

**Measured cost: ~652 s/step on 2 GPUs -> ~17.4 h for 100 steps.**


## Attempt 1 died at step 37: CUDA OOM, and the warning was in the data from step 1

`perf/max_memory_allocated_gb` read **98.8 GB mean / 99.4 peak on a ~95.6 GB card** for
the entire run, against 39.5 for our 4-GPU arms. vLLM sleeps and wakes its KV pool every
step; at 99% of the card each wake is a coin flip, and one failed inside
`cumem_allocator.wake_up`.

**Cause:** `gpu_memory_utilization=0.6` was copied from their script *because* it was
theirs -- but theirs runs 8 GPUs at TP=2, where a rank carries a quarter of what it does
here on 2 GPUs. Fidelity to a value is not fidelity to a configuration.

I had already seen the 97.8-vs-39.2 figure in a ranked metric comparison and dismissed
it as "expected from GPU count". It was the failure, visible six hours early.

**Two fixes, both deviations from their script, both documented in `run.sh`:**

* `gpu_memory_utilization` 0.6 -> **0.3**. Sizes the KV cache, not the result.
* `save_freq` -1 -> **20**. Theirs writes no checkpoints, so the OOM destroyed 7 hours
  with nothing to resume from. ~50 GB against 283 GB free.

**Attempt 1's data is preserved** as `metrics_attempt1_oom.jsonl` / `train_attempt1_oom.log`
-- 7 evaluations, steps 5-35, showing a stable ~10-15 point lead over our arm. That
result stands on its own; the restart is to see whether G2PO reaches its published 95.0
by step 100.
