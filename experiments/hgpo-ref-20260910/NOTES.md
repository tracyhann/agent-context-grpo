# hgpo-ref-20260910 — HGPO reference reproduction

**Published: 92.77 in-distribution @ 160 iterations**, Qwen2.5-1.5B, K=2 — the strongest
published number on this benchmark, and the one "SOTA" means at our `history_length=2`.

**No clone was needed.** HGPO ships *inside* `baselines/verl-agent` (the GiGPO repo) as
`recipe/hgpo`, with its own ray trainer, env manager, config and an official ALFWorld
script at exactly our scale: `run_qwen2.5_1.5b_alfworld_train.sh`.

## Where HGPO's config differs from G²PO's — worth knowing before comparing

| | HGPO | G²PO | ours |
|---|---|---|---|
| `total_epochs` | **160** | 100 | 100 |
| `max_prompt_length` | **4096** | 2048 | 2048 |
| `truncation` | **left** | error | error |
| `ppo_micro_batch_size_per_gpu` | 16 | 32 | 32 |
| `save_freq` | 40 | −1 | 5 |

Their headline is at **160 iterations**, not 100, so it is *not* budget-matched to G²PO's
95.0 or to our 79.7. Any table must say so.

## Deviations from their script (all forced by this box, all in `run.sh`)

1. `VLLM_ATTENTION_BACKEND` XFORMERS → **TRITON_ATTN** (no sm_120 xformers kernel)
2. **`docker/fa_stub` off `PYTHONPATH`** — real flash-attn 2.8.3 is installed and its
   `unpad_input` is pure PyTorch, so `use_remove_padding=True` works on Blackwell
3. `HOME=/home/claude` — `env -i` clears it and `/root` is unwritable
4. `gpu_memory_utilization` 0.6 → **0.3** — at 0.6 the G²PO reference allocated ~99 GB on
   a 96 GB card and died in vLLM's `cumem` wake_up at step 37
5. `tensor_model_parallel_size` 2 → **1** — 2 GPUs at TP=1 gives 2 data-parallel ranks;
   16 divides 2
6. console logger only; metrics parsed from the **ray worker log**, not the driver
7. **`data_preprocess.prepare` SKIPPED** — it regenerates the parquets and could draw a
   different validation set, destroying comparability
8. `default_local_dir` set **explicitly** — their default is relative and would write
   inside the baselines checkout

**Unchanged:** everything touching the objective — `weight_type=length`,
`length_weight_alpha=1.0`, `base_group=False`, `mode=mean_std_norm`, lr, KL, γ,
invalid-action penalty, batch/group sizes, history_length 2, val temperature, test_freq.

## Status

**Queued, not launched.** Host RAM was 55 GB against the ~90–105 GiB a second arm needs.
Runs when the queue ahead of it drains.

## Status update 2026-09-10: executed in ANOTHER container

Removed from this container's queue at the user's direction. This directory remains as
the configuration record (`run.sh`, `config.json`, the deviation list above); the run
itself is happening elsewhere, so `outputs/` here will not be populated by this
container. Compare against it only once its metrics are brought back with the same
evaluation protocol (128-game `eval_in_distribution`, T=0.4) -- and remember its
headline is at 160 iterations, not 100.

## 2026-09-10: a 4-GPU copy is chained in THIS container too

At the user's request HGPO was also queued here, after `ccpo-refined`, on GPUs 0-3:
see `experiments/hgpo-ref-4gpu-20260910`. Same objective config; separate id and output path.
