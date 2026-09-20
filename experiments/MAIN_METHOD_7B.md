# Main method: 7B / eight-GPU preparation

**Prepared only, 2026-09-20. No training or queue entries created.** All four
experiments use **Qwen2.5-7B-Instruct, 150 steps, seed 0, eight GPUs per run**.
ALFWorld is M10; its corresponding WebShop arm remains named M11.

| Benchmark | Variant | Episode weight | Prepared experiment |
|---|---|---:|---|
| ALFWorld | H2 no shrinkage, no episode (main method) | 0 | [M10 7B](m10-h2-noshrink-alfworld-7b-8gpu-20260920/NOTES.md) |
| ALFWorld | H2 no shrinkage, active episode | 1 | [M10 7B + episode](m10-h2-noshrink-active-episode-alfworld-7b-8gpu-20260920/NOTES.md) |
| WebShop | H2 no shrinkage, no episode (main method) | 0 | [M11 7B](m11-h2-noshrink-webshop-7b-8gpu-20260920/NOTES.md) |
| WebShop | H2 no shrinkage, active episode | 1 | [M11 7B + episode](m11-h2-noshrink-active-episode-webshop-7b-8gpu-20260920/NOTES.md) |

## Preserved method

[The designated main method](MAIN_METHOD.md) remains history 2 / future 2,
full-strength usable contextual baselines (`ccpo_lk_fix=1`, `ccpo_lam_fix=1`),
frozen hidden state plus context statistics, soft exponential similarity,
original edge weight 0, history/future weights 1/1, and binary-return targets.
Kappa 2 is recorded but does not shrink a usable baseline. Existing task fallback,
whole-trajectory exclusion, endpoint grouping and terminal potentials remain.

Writing H for historical contextual advantage and F for the task-standardized
H2 potential difference, the two actor advantages are:

\[
A^{\mathrm{main}}=M\,N_{CC}(H+F),\qquad
A^{+EP}=M\,[E+N_{CC}(H+F)].
\]

N_CC is task standardization on ALFWorld and identity on WebShop. E uses the
existing trainer episode channel and is added **after** contextual normalization;
there is no final normalization of the sum. With episode weight 0, the raw episode
channel remains logged but has zero actor contribution. Episode rewards still
supply return/potential labels in both variants. Gradient-isolation and active
fusion checks exercise the actual estimator/trainer helpers.

Within each benchmark's 7B pair, `ccpo_ep_w` is the only learning-setting change.
Relative to each matching prepared 1.5B arm, only model, resolved checkpoint,
GPU allocation, rollout tensor parallelism and experiment identity change.
Each folder records both exact config diffs.

## Eight-GPU runtime

- Configured devices: `0,1,2,3,4,5,6,7`; these IDs are not reservations.
- Rollout tensor parallelism **2**, giving four rollout replicas; actor/reference
  FSDP world size **8**. This restores the existing eight-GPU reference layout.
- Training batch stays **16 tasks × 8 rollouts = 128 episodes**. More GPUs do not
  change group size, method coefficients, optimizer minibatches or reward targets.
- Existing dynamic per-GPU token budgets remain **12,288 update / 24,576 log-prob**;
  activation checkpointing and FlashAttention stay enabled. Memory fit is not
  established by CPU preparation and needs an eight-GPU smoke run at launch time.
- ALFWorld keeps its **50-turn** train/eval horizon; WebShop keeps **15 turns**, the
  1K catalog and original scorer. This is separate from the 30-turn ablations.
- Learning rate 1e-6, gamma 0.95, separate KL coefficient 0.01; eval/save every
  five steps, no early stopping, best + step100 + final checkpoint retention.

The 7B weights are not cached on this host at preparation time. The full resolved
config uses `Qwen/Qwen2.5-7B-Instruct` as the model path; it never retains a 1.5B
snapshot. A later launch must have access to the 7B weights. No model download,
GPU allocation or runtime-readiness claim is part of this preparation.

## Reproduction and launch

[`scripts/prepare_main_7b.py`](../scripts/prepare_main_7b.py) generates all four
experiments from their recorded 1.5B controls and the shared method registry.
It calls the actual launcher with `--dry-run`, validates rendered Hydra flags,
checks pairwise differences, writes shell scripts and refuses existing folders.

```bash
# Use a new YYYYMMDD to prepare another copy; this date already exists.
python scripts/prepare_main_7b.py --date YYYYMMDD --gpus 0,1,2,3,4,5,6,7
```

[`ablations/run.py`](../ablations/run.py) now accepts `--backbone 7b`; default
1.5B IDs and the existing override API are preserved. To inspect the 7B registry:

```bash
python ablations/run.py --list --backbone 7b
```

Each experiment contains `config.json`, `run.sh`, `PREPARED.json`, `NOTES.md`,
config diffs, preparation command, source hashes and `VALIDATION.json`. When a
run is scheduled, execute that folder's `run.sh` to reproduce the full prepared
local protocol. All four require eight GPUs each; the scripts do not schedule
or preempt existing jobs.

CPU verification covers the backbone/CLI selection, the 7B model path, eight-GPU
and TP2 flags, exact method pairing, environment exports, no-overwrite behavior,
existing active-episode gradients and zero-weight isolation. GPU execution
remains untested.
