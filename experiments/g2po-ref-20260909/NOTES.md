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
