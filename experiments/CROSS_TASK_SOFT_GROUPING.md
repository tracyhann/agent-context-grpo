# M10/M11 H2 no shrinkage, EP0: cross-task contextual soft grouping

This variant removes task and observation matching from the selected
[main method](MAIN_METHOD.md). Both 1.5B runs retain episode weight zero.
Registry: `future-progress-h2-no-credit-shrinkage-cross-task`.
Variant: `CCPO-ATTNCRED-FUTURE-PROGRESS-TWO-STEP-NOSHRINK-CROSS-TASK`.

| Benchmark | Canonical ID | Prepared experiment |
|---|---|---|
| ALFWorld | `ccpo-attncred-abl-fph2-noshrink-cross-task-alfworld-1.5b` | [M10 notes](m10-h2-noshrink-cross-task-alfworld-1.5b-2gpu-20260928/NOTES.md) |
| WebShop | `ccpo-attncred-abl-fph2-noshrink-cross-task-ws-1.5b` | [M11 notes](m11-h2-noshrink-cross-task-webshop-1.5b-2gpu-20260928/NOTES.md) |

Both configurations change only `ccpo_gate=hard` to `ccpo_gate=cross_task`
plus experiment identity relative to their no-shrink EP0 control. Legacy
implicit LOO and history coefficient defaults are recorded explicitly.

## Peer pool and weights

For canonical rollout batch B and query occurrence i:

```text
P_i = {j in B: trajectory(j) != trajectory(i)}
w_ij = exp(-||phi_i - phi_j|| / tau)
C_i[q] = sum_{j in P_i} w_ij q_j / sum_{j in P_i} w_ij
```

No condition on task UID or exact observation appears in P_i. Cross-task means
all tasks in one benchmark's rollout batch; it does not mix ALFWorld/WebShop,
previous iterations, or an external replay buffer. The encoder and prompts
still contain task meaning; only the hard matching requirement is removed.
The existing processed frozen hidden+context features, PCA-3, context weight 1,
Euclidean distance and exponential kernel remain. Tau is 0.15 times the median
distance over all distinct row pairs in the pool, with the existing degenerate
median fallback of 1. The bandwidth pool includes same-trajectory pairs;
readout weights exclude the entire query trajectory. Weights use a common
exponential shift for stability, which cancels after normalization.

There is no top-k approximation, task-prior mixing, or same-task fallback.
The inherited kappa=2 and backoff flag are inert for this gate. Real repeated
visits retain occurrence weight; trainer padding copies are deduplicated before
all statistics. J counts other trajectories; effective support is
`1 / sum_j (normalized weight mass of trajectory j)^2`. No other trajectory
means no supported historical or future credit.

The old `ccpo_gate=global` is unchanged: it removes observation matching only
and still creates one pool per task.

## Credit and protocol

```text
H_i = Y_i - C_i[Y]
Z_i = gamma^(T_i-t_i) R_i; V_i = C_i[Z]
F_i = z_task(V_min(t_i+2,T_i) - V_i)
ALFWorld: A = mask * N_task(H + F)
WebShop:  A = mask * (H + F)
```

Y retains the existing invalid-action penalty; Z does not. Nonterminal future
endpoints use their own context and the batch-wide pool. Terminal potential
remains success 10 / failure 0. Gamma=0.95. H/F/EP weights are 1/1/0 and the
original edge weight is zero. Lambda_u=lambda_k=1. Task IDs remain available
for task-wise normalization and diagnostics, not neighbor eligibility.

Both use Qwen2.5-1.5B-Instruct, seed 0, 150 steps, 16 tasks x 8 rollouts,
history length 2, future horizon 2, and two configured GPUs. All environment
and optimizer settings follow the control: ALFWorld 50 turns, WebShop 15
turns with the 1K catalogue. These are prepared configurations, not results.

## Implementation and verification

- [Shared readout](../ccpo/cross_task.py), integrated into the existing history
  and potential estimator in [core_ccpo.py](../ccpo/core_ccpo.py).
- [Tests](../tests/test_cross_task_ablations.py): independent numerical kernel,
  cross-task reward influence, task/observation label independence with frozen
  features, whole-trajectory LOO, padding, missing peers, stable narrow kernels,
  configuration propagation and real trainer PPO gradients in both modes.
- Exact distances use about 313 MiB for the maximum 6,400-row ALFWorld batch;
  median selection temporarily adds about 156 MiB. The implementation avoids
  per-pair Python records and the much larger `(rows, rows, features)` temporary.

Metrics identify `ccpo/progress_cross_task_grouping=1`. Snapshots include
`{history,current,future}_cross_task_mass`, `_peer_tasks`, `_kernel_tau`, `_J`,
`_n_eff`, `_peer_rows`, and `_same_traj_mass`. Current exact/backoff fractions
are zero in this mode. Level 0 denotes the primary batch pool; it does not
mean exact observation matching. Terminal future rows have no neighborhood.

Prepare new dated configurations (refuses existing directories):

```bash
.venv/bin/python scripts/prepare_cross_task_ablations.py --date YYYYMMDD
```

Each experiment includes its configuration diff, source hashes, preparation
command and validation record. Preparation does not start or queue training.

## Validation on 2026-09-28

The focused CPU suite passed 67 tests, including the new cross-task tests and
existing future-progress/uniform-peer regression tests. The separate WebShop
environment passed all 9 cross-task tests. The ablation registry guard passed.

A synthetic full-size readout (6,400 rows, 1,573 features, 16 tasks, 128
trajectories) took 4.11 seconds with one BLAS thread; process peak RSS was
594.8 MiB. All 6,400 readouts were finite and each excluded its own 50-turn
trajectory. This measures one CPU readout, not encoding, whitening, both
channels together, or end-to-end training.

The full arm guard still reports two pre-existing undocumented uniform-peer
variant names and 338 absolute-path findings in host-local/generated scripts
(including the two new ignored launch scripts generated by the standard
launcher). The two missing names were confirmed against HEAD. The new variant
is documented; configuration, trainer wiring and dependency checks pass.
See each experiment's VALIDATION.json for the exact scope and test logs.
