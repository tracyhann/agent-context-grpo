# Experiments

One directory per run, named `<arm>-<what-it-tests>-<YYYYMMDD>`. Every directory is
self-contained: the configuration that produced the numbers sits beside them, so a
result can always be traced back to the exact command that made it.

```
<exp-id>/
  config.json     FULL resolved configuration -- hydra overrides, ACG_* env vars,
                  git commit + dirty flag, torch/vllm/transformers versions, and
                  which parts of the G2PO reference protocol are matched vs not
  run.sh          the exact command, regenerated from config.json
  NOTES.md        what this arm tests, what to look at, what happened
  outputs/
    train.log             stdout/stderr
    metrics.jsonl         one JSON object per training step -- every metric
    resolved_config.json  verl's own view of the config
    ccpo_samples.csv      per-sample estimator diagnostics (CCPO arms)
    ccpo_samples.csv.len.csv   (response length, A_CC) pairs
    checkpoints/
      stepN-best/   hardlinked copy of the best held-out checkpoint
      stepN-last/   symlink to the most recent
      best.json     which step is best, and its score
  plots/
    progress.png    success rate, reward, KL, entropy, length, grad norm, clip
    ccpo.png        lambda, n_eff, bucket occupancy, effect size, r_vs_*, term scales
```

Checkpoints are deliberately **not** kept for every step: `trainer.save_freq` writes a
rolling `global_step_N`, of which only the two newest survive, and the run keeps exactly
one `stepN-best` (hardlinked, so it costs no extra disk) and one `stepN-last`.

## Running

```bash
scripts/exp_run.py --name <name> --arm <ccpo|grpo|gigpo> [--set key=value ...]
scripts/exp_run.py --name <name> --arm ccpo --dry-run       # write config only
scripts/plot_metrics.py --compare experiments/a experiments/b -o experiments/compare.png
```

`--set` accepts any key in `DEFAULTS` (`scripts/exp_run.py`). Arms must differ ONLY in
the keys named on the command line, so comparing two runs is a diff of two `config.json`
files:

```bash
diff <(jq -S .config experiments/A/config.json) <(jq -S .config experiments/B/config.json)
```

## Comparability with the published baselines

Defaults mirror `baselines/G2PO/examples/g2po_trainer/run_alfworld.sh`, which is where
the published G2PO ALFWorld numbers come from, and which GiGPO/HGPO also follow. The
matched settings and the remaining deltas are recorded in every `config.json` under
`reference_protocol`. The deltas are hardware-forced, not choices:

| | reference | here | why |
|---|---|---|---|
| attention | flash-attn | sdpa + `use_remove_padding=False` | flash-attn has no sm_120 build |
| rollout attention | XFORMERS | `TRITON_ATTN` | Blackwell; Triton JIT-compiles per arch |
| GPUs | 8, `tp=2` | 6, `tp=1` | what this box has |

Published reference points on ALFWorld / Qwen2.5-1.5B-Instruct (HGPO Table 1,
in-distribution / out-of-distribution success):

| method | In-Success | Out-Success |
|---|---|---|
| GRPO | 72.8 | 70.1 |
| GiGPO (K=2) | 90.16 | 84.76 |
| HGPO (K=2) | 92.77 | 90.16 |
| GiGPO (K=4) | 93.29 | 91.53 |
| HGPO (K=4) | 94.85 | 92.12 |

`K` is `env.history_length`: it sets both how many past turns the prompt carries and,
for HGPO, the depth of its group hierarchy. Our default is 2, so the K=2 rows are the
comparable ones.

## Picking up a run in flight

Runs are detached (`start_new_session`), so they outlive the shell that started
them. To see where one is:

```bash
scripts/exp_status.py                                     # every run, one line
tail -f experiments/<exp>/outputs/train.log | grep '\[turn\]'   # rollout cadence
nvidia-smi --query-compute-apps=pid --format=csv,noheader  # which phase
```

The process name tells you the phase: `generate_sequences` (rollout),
`compute_log_prob` / `ref_compute_ref_log_prob` (the two forward passes),
`update_actor` (PPO). A silent log with the GPUs at 100% is normal during the
forwards and the update; the `[turn]` lines cover the rollout.

To stop one cleanly: `kill $(cat experiments/<exp>/outputs/train.pid)`, then
`.venv/bin/ray stop --force`, then check `nvidia-smi --query-compute-apps` for a
lingering context and kill that pid too — ray occasionally leaves one holding
tens of GB, which will OOM the next launch.

## Environment constraints on this box

Two things bind, and both are recorded here because they cost hours to find:

**The pid ceiling.** The cgroup allows 8192 pids while `nproc` reports 256, and
Ray sizes its gRPC/asio pools from `nproc`. verl-agent also gives **every ALFWorld
environment its own Ray actor**, so a reference batch is 128 train + 128 validation
actors. Untuned that is ~115 threads per actor, ~29k threads, and every worker
aborts with `thread: Resource temporarily unavailable`. Two fixes, both in
`scripts/exp_run.py`:

- `RAY_num_server_call_thread`, `RAY_num_grpc_internal_threads` and
  `RAY_object_manager_rpc_threads_num` set to 1 take an actor from ~115 to ~24
  threads (measured);
- `val_batch_size=64` keeps the full 128-episode evaluation but halves the
  persistent validation pool. verl iterates the whole val dataloader, so the
  protocol is unchanged — only the actor count is.

At 128 + 128 actors the run reached 8165/8192 threads and died. At 128 + 64 it fits.
**Consequence: arms cannot run in parallel on this box** — two arms would need
~384 actors. Runs are sequential, each on all available GPUs.

**A cold start looks exactly like a hang.** The first `generate()` call compiles the
model and captures CUDA graphs for every vLLM engine. With four engines that pins
the GPUs at 100% for several minutes and writes nothing to the log — and it cost one
run, killed on the assumption that generation was pathologically slow. It is not:
benchmarked standalone, `TRITON_ATTN` does **6127 tok/s** on this model (32
sequences, 512 max tokens, 2.4 s). `scripts/exp_run.py` now points
`VLLM_CACHE_ROOT` and `TORCHINDUCTOR_CACHE_DIR` at the data volume, where a warm
engine starts in ~16 s, and the rollout logs one line per turn so a stall is
distinguishable from progress. Before concluding a run is stuck, check
`nvidia-smi --query-compute-apps` — the process name says which phase it is in
(`generate_sequences`, `compute_log_prob`, `ref_compute_ref_log_prob`,
`update_actor`).

**Micro-batch sizes cannot follow the reference directly.** The reference sets
`ppo_micro_batch_size_per_gpu=32` for both the update and the log-prob passes, but
it also runs `use_remove_padding=True`, so its micro-batches are packed. Without
flash-attn we pad every sequence to `max_prompt_length + max_response_length` =
2560 tokens, and the same 32 OOMs in the **backward** pass of `update_actor`
("Tried to allocate 9.27 GiB") while being perfectly fine for the forward-only
log-prob passes. So the two are separate knobs here: `ppo_micro_batch_size_per_gpu`
16 for the update, `log_prob_micro_batch_size_per_gpu` 32 for the forwards.
`gpu_memory_utilization` is also cut to 0.35 — vLLM reserves that fraction of the
card for the whole run and only needs KV cache for ~32 short generations; the rest
is worth more to the backward pass.

**Batch/GPU divisibility.** verl asserts `train_batch_size * rollout.n % n_gpus == 0`.
The reference `train_batch_size=16` with `group_size=8` gives 128, so 4 GPUs
divides cleanly and 6 does not. Arms run on 4 GPUs and **two GPUs sit idle by
choice**: the alternative is `train_batch_size=18` (144 episodes/step), which would
buy ~1.5x throughput at the cost of no longer matching the batch the published
numbers were produced with. For `gigpo-repro`, matching that batch *is* the
experiment. The idle GPUs cannot be used for a second arm either — the pid ceiling
above, not GPU count, is what caps parallelism.

## Index

Regenerate the live view with `scripts/exp_status.py --md`.

| experiment | arm | purpose | status |
|---|---|---|---|
| `gigpo-repro-20260906` | gigpo | reproduce published GiGPO (90.16 / 84.76) to validate the harness | running |

Analysis helpers:

```bash
scripts/exp_status.py                       # one line per run + the published targets
scripts/analyse_dump.py experiments/<exp>   # per-sample estimator diagnostics
scripts/plot_metrics.py --compare a b -o experiments/compare.png
```
