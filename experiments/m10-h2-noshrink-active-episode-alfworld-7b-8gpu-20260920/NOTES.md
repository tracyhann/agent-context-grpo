# M10 H2 no-shrink + active episode: 7B / eight GPUs

**Prepared only; not launched or queued.** Fresh `Qwen/Qwen2.5-7B-Instruct`, seed 0,
150 training steps, eight GPUs (`0,1,2,3,4,5,6,7`). GPU IDs are configuration, not a reservation.
Registry key: `future-progress-h2-no-credit-shrinkage-active-episode`. Canonical ID: `ccpo-attncred-abl-fph2-noshrink-ep-alfworld-7b`.

## Method and comparison

History length 2, future horizon 2, context statistics enabled, soft exponential
similarity (`ccpo_tau=0.15`, whitening removes 3 directions). Whole query trajectories
are excluded from the contextual readout. All usable history/current/future
baselines use their context estimate at full strength (`ccpo_lk_fix=1`,
`ccpo_lam_fix=1`); kappa 2 is retained but cannot shrink the baseline. Existing
fallback, unsupported-row handling, terminal 10/0 potentials and detached features
are unchanged. Original edge weight 0; history/future weights 1/1.

Let `H = Y - C[Y]`, `V_s = C_s[Z]`, and
`F = z_task(V_min(t+2,T) - V_t)`. Current/future readouts use their respective
observation groups. Y includes the existing invalid-action penalty; Z is the
unpenalized discounted potential target. The masked actor advantage is
`A = mask * (E + N_CC(H + F))`. N_CC is task mean/std normalization on ALFWorld and
identity on WebShop. Future F is task-standardized on both benchmarks.
Episode coefficient is **1**. E is the existing trainer episode channel, added after contextual normalization without normalizing the final sum.
E uses the existing task-level turn-row mean/std on ALFWorld and mean subtraction
on WebShop, including the -0.1 invalid-action penalty. No new episode normalization
or reward definition is introduced. KL remains a separate loss.

Full math: [main-method definition](../MAIN_METHOD.md) and
[active-episode implementation](../m10-h2-noshrink-active-episode-alfworld-1.5b-2gpu-20260920/NOTES.md).

Paired 7B arm: [notes](../m10-h2-noshrink-alfworld-7b-8gpu-20260920/NOTES.md). Only `ccpo_ep_w` differs between the
pair's learning settings. [config-diff-from-pair.json](config-diff-from-pair.json)
records that change and the separate experiment ID.
The matched 1.5B control is `experiments/m10-h2-noshrink-active-episode-alfworld-1.5b-2gpu-20260920`. Scale changes are only model,
resolved model path, GPU allocation, rollout TP1→2, and experiment identity;
see [config-diff-from-1.5b.json](config-diff-from-1.5b.json).

## Protocol and resource preparation

16 tasks × 8 rollouts = 128 episodes per training step; eight GPUs do **not**
change the number of rollouts per task. Rollout TP 2 gives four replicas; actor/ref
FSDP use eight ranks. Gradient checkpointing and dynamic token batching stay on.
Existing per-GPU token budgets remain update 12288 / log-prob 24576 and require an
8-GPU memory smoke test before claiming runtime readiness. There is no GPU test
or memory-fit claim from this preparation.

Training and validation turn ceiling: **50**. This is the standard
ALFWorld 50-turn protocol;
WebShop retains its 1K catalog and original scorer. Prompt/response limits,
optimizer, minibatches, validation sampling and all other benchmark settings
match the recorded 1.5B control. Learning rate 1e-6, discount 0.95, separate KL 0.01;
validation/checkpoint every 5 steps, early stopping off; best, step 100 and last
checkpoints retained. History/current/future and applied episode diagnostics
continue through the existing estimator and plot code.

## Reproduce / launch later

The preparation command is recorded in [prepare-command.json](prepare-command.json).
To prepare all four arms in a new dated directory, run:

```bash
python scripts/prepare_main_7b.py --date YYYYMMDD --gpus 0,1,2,3,4,5,6,7
```

The generator refuses existing experiment directories, runs the launcher only
with `--dry-run`, and preserves the full paired 1.5B protocol. The public registry
also accepts `--backbone 7b`; use the prepared full config when reproducing the
local benchmark-specific settings.

When these eight GPUs are assigned and launch is requested:

```bash
bash experiments/m10-h2-noshrink-active-episode-alfworld-7b-8gpu-20260920/run.sh
```

The full command
and exported method flags are in [config.json](config.json). The model field is
7B and the resolved checkpoint cannot point at the old 1.5B weights. If 7B weights
are not cached, `model_path` is the public Hugging Face model ID; a future launch
needs access to those weights. No weights are downloaded by this preparation.
Environment/setup details and source hashes are retained. [VALIDATION.json](VALIDATION.json)
records CPU checks separately from the pending GPU smoke test.
