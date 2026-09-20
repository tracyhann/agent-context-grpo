# H2 component ablations: history-only and future-only

Prepared 2026-09-20, **not launched or queued**. Four Qwen2.5-1.5B-Instruct
experiments, 150 steps, seed 0, two GPUs each. No credit shrinkage. ALFWorld
has episode weight 0; WebShop retains active episode weight 1, as requested.

| Benchmark | Variant | History weight | Future weight | Episode weight | Prepared notes |
|---|---|---:|---:|---:|---|
| ALFWorld M10 | history-only | 1 | 0 | 0 | [NOTES](m10-h2-noshrink-history-only-alfworld-1.5b-2gpu-20260920/NOTES.md) |
| ALFWorld M10 | future-only | 0 | 1 | 0 | [NOTES](m10-h2-noshrink-future-only-alfworld-1.5b-2gpu-20260920/NOTES.md) |
| WebShop M11 | history-only + episode | 1 | 0 | 1 | [NOTES](m11-h2-noshrink-history-only-active-episode-webshop-1.5b-2gpu-20260920/NOTES.md) |
| WebShop M11 | future-only + episode | 0 | 1 | 1 | [NOTES](m11-h2-noshrink-future-only-active-episode-webshop-1.5b-2gpu-20260920/NOTES.md) |

## Equations and interpretation

C_t[q] denotes the existing kernel-weighted contextual readout of q from peers
outside the query's trajectory. Peers come from the same task/observation group;
existing task backoff is retained. No-shrink means lambda_k=1 on usable readouts;
no supported peer anywhere still produces no usable contextual credit.
The kernel-versus-uniform mixture also stays at lambda_u=1.

\[
H_{i,t}=Y_{i,t}-C_{i,t}[Y],\qquad
Z_{i,t}=\gamma^{T_i-t}R_i,\qquad V_{i,t}=C_{i,t}[Z],
\]
\[
F_{i,t}=z_{\mathrm{task}}\!\left(V_{i,\min(t+2,T_i)}-V_{i,t}\right),
\qquad A_{i,t,\ell}=M_{i,t,\ell}\left[w_E E_{i,t}
+N_{\mathrm{CC}}(w_H H+w_F F)_{i,t}\right].
\]

Y is the existing penalized return target; Z is the unpenalized discounted
potential target, gamma=0.95. Current and future potentials use their own
observation groups. The terminal potential is 10 for success and 0 otherwise.
F uses per-task mean/sample-standard-deviation normalization with epsilon 1e-6.
N_CC applies the existing per-task mean/sample-std normalization on ALFWorld
and is identity on WebShop. E is the existing episode channel: per-task turn-row
standardization on ALFWorld, mean subtraction on WebShop. E is added **after**
N_CC; the final sum is not normalized again. KL remains a separate loss.

| Arm | Masked policy advantage |
|---|---|
| ALFWorld history-only | M N_CC(H) |
| ALFWorld future-only | M N_CC(F) |
| WebShop history-only | M (E + H) |
| WebShop future-only | M (E + F) |

N_CC's supported-row population is exactly
`(w_H > 0 AND history_live) OR (w_F > 0 AND future_eligible)`.
A disabled component cannot change the active component's normalization through
its support mask. Unsupported-row and singleton behavior follow the existing
trainer. Advantage tensors are detached before the PPO loss.

**Future-only means no historical-residual advantage H.** History length 2,
frozen hidden states and accumulated context-statistics vectors still construct
the potentials. Removing those inputs would be a different representation ablation.
**History-only means no future-progress advantage F.** The shared H2 estimator
still computes raw future diagnostics at weight 0; they cannot affect the update.
Episode weight 0 disables only the separate E channel, not the episode-return
labels required by H and F.

## Matched protocol

ALFWorld controls against [M10 H2 no-shrink EP0](m10-h2-no-credit-shrinkage-alfworld-1.5b-2gpu-20260919/NOTES.md).
WebShop controls against [M11 H2 no-shrink EP1](m11-h2-noshrink-active-episode-webshop-1.5b-2gpu-20260920/NOTES.md).
Each new config differs from its matched control only in the disabled component's
coefficient and experiment identity. Legacy configs' implicit history coefficient
1 is materialized explicitly without rewriting their records.

Retained: history length 2, future horizon 2, context statistics weight 1, soft
exponential similarity with tau 0.15 and whitening 3, eight rollouts per task,
16 tasks per training batch, original edge weight 0, 150 steps, seed 0, fresh
1.5B weights, standard 50-turn ALFWorld / 15-turn WebShop protocol, evaluation
and checkpoints every 5 steps. WebShop uses the existing 1K catalog and scorer.
Configured GPU pairs are ALFWorld 0,1 and WebShop 2,3; they are not reservations.
Runs sharing a pair need separate scheduling.

## Implementation and logging

- [Estimator](../ccpo/future_progress.py): `history_weight` defaults to 1;
  `progress_weight` retains default 1. Weighted components and active support
  drive the actual step credit.
- [Launcher](../scripts/exp_run.py): `ccpo_progress_history_weight` maps to
  `ACG_CCPO_PROGRESS_HISTORY_WEIGHT`; `ccpo_progress_weight` maps to
  `ACG_CCPO_PROGRESS_WEIGHT`. Negative/nonfinite weights are rejected.
- [Trainer overlay](../patches/verl-agent/verl/trainer/ppo/ray_trainer.py): forwards
  both weights; the live trainer has the identical change.
- [Registry](../ablations/ablations.py): four benchmark-specific component arms.
- [Preparation helper](../scripts/prepare_component_ablations.py): full-control
  protocol cloning, single-coefficient diff checks, no overwrite and dry-run only.

Snapshots retain raw `history_adv`, `progress_normalized`, potential/support/
shrinkage estimates, plus `weighted_history`, `weighted_progress`,
`history_applied`, `future_applied`, `episode_applied`, and `actor_applied`.
Disabled weighted/applied arrays are exactly zero. The runtime actor-sum identity
check remains enabled. The existing plots consume actual history/future/episode
weights, so a zero coefficient is shown as zero while raw diagnostic terms remain
available. `progress_live_frac` reports the enabled components' support union.

## Reproduction and validation

```bash
# Prepare all four arms with a new date; this does not launch training.
python scripts/prepare_component_ablations.py --date YYYYMMDD

# CPU tests, including actual trainer function and PPO loss/gradient isolation.
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=tests:. \
  .venv/bin/python -m unittest test_future_progress_components -v
```

Each folder records config, environment, Hydra command, run.sh, control diff,
PREPARED.json, preparation command, source hashes and NOTES.md. Validation results
are recorded in VALIDATION.json; this is code/config preparation, not measured
benchmark performance or a GPU smoke test.

Validation completed: **53 CPU unit tests passed**, including seven component
checks, real trainer/PPO gradient isolation across all four variants, existing
H1/H2, active/disabled episode channels and hidden-only compatibility. Registry
and documentation guards, shell syntax, Python compilation and trainer-overlay
parity passed. The broad `ccpo/test_arms.py` guard separately fails existing
FlashAttention absence/stub probes against the installed package, and its global
absolute-path scanner (including generated local launchers). Details and logs
are retained in each folder's VALIDATION.json; those broad checks are not claimed
as passing. No GPU training or benchmark evaluation was started.
