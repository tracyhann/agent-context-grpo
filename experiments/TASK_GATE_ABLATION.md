# M10 EP0 / M11 EP1 H2 no shrinkage: task-gate ablation

**Prepared tooling only. No run has been launched.**

This ablation removes exact-observation matching from the selected
[main method](MAIN_METHOD.md) and keeps task identity. It answers the reviewer's
question directly: how much of the method depends on observations recurring across
rollouts of a task?

It is not the [cross-task arm](CROSS_TASK_SOFT_GROUPING.md). That arm drops the
task condition and the observation condition together, so its result cannot be
attributed to either one.

| Benchmark | Registry / variant | Canonical ID |
|---|---|---|
| ALFWorld EP0 | `future-progress-h2-no-credit-shrinkage-task-gate` / `CCPO-ATTNCRED-FUTURE-PROGRESS-TWO-STEP-NOSHRINK-TASK-GATE` | `ccpo-attncred-abl-fph2-noshrink-task-gate-alfworld-1.5b` |
| WebShop EP1 | `future-progress-h2-no-credit-shrinkage-active-episode-task-gate` / `CCPO-ATTNCRED-FUTURE-PROGRESS-TWO-STEP-NOSHRINK-ACTIVE-EPISODE-TASK-GATE` | `ccpo-attncred-abl-fph2-noshrink-ep-task-gate-ws-1.5b` |

Both change only `ccpo_gate=hard` to `ccpo_gate=global`, plus experiment identity,
relative to their no-shrink controls: ALFWorld
`m10-h2-no-credit-shrinkage-alfworld-1.5b-2gpu-20260919` (EP0) and WebShop
`m11-h2-noshrink-active-episode-webshop-1.5b-2gpu-20260920` (EP1).

## Peer pool and weights

```text
P_i = {j: task(j) = task(i), trajectory(j) != trajectory(i)}
w_ij = exp(-||phi_i - phi_j|| / tau)
C_i[q] = sum_{j in P_i} w_ij q_j / sum_{j in P_i} w_ij
```

No condition on the observation appears in P_i. Features, the exponential kernel,
whole-trajectory exclusion, targets and normalization are the control's. Tau is
0.15 times the median distance over all row pairs of the task.

This is the readout the control already applies to a turn that has no
exact-observation peer, here applied to every turn. The equivalence is exact: on
a batch whose observations never repeat across rollouts, the hard gate and the
task gate return the same credit to the last bit, and the task gate returns that
credit whether or not observations repeat.

There is one level, so the exact and backoff fractions both read zero, and the
task prior does not apply. A task with a single rollout has no supported credit.

`ccpo_gate=global` already existed for arms without future progress; the only
estimator change is that the future-progress path now accepts it. Gates other than
`hard`, `global` and `cross_task` are rejected at launch for future-progress arms.

## Credit and protocol

```text
H_i = Y_i - C_i[Y]
Z_i = gamma^(T_i-t_i) R_i; V_i = C_i[Z]
F_i = z_task(V_min(t_i+2,T_i) - V_i)
ALFWorld: A = mask * N_task(H + F)
WebShop:  A = mask * (E + H + F)
```

Qwen2.5-1.5B-Instruct, seed 0, 150 steps, 16 tasks x 8 rollouts, history length 2,
future horizon 2, two configured GPUs, with every environment and optimizer
setting taken from the control.

## Reading the result

Compare success with the control, GRPO and GiGPO at the same step.

- At or near the control: the method does not need observations to recur.
- Between GRPO and the control: contextual weighting carries part of the gain by
  itself and observation matching the rest.
- At or below GRPO: the method needs recurring observations, and the task-level
  fallback is not a substitute for them.

`ccpo/progress_task_grouping=1` identifies the variant in the metrics.

## Prepare

On the training host, from the repository root:

```bash
bash scripts/sync_patches.sh
```

```bash
.venv/bin/python tests/test_rebuttal_task_gate_ccpo.py
```

```bash
.venv/bin/python scripts/prepare_rebuttal_task_gate_ccpo.py
```

Preparation compares every key with the control, including its absolute paths, so
it succeeds only in the checkout location the control was recorded in. It writes
`experiments/m10-h2-noshrink-task-gate-ep0-alfworld-1.5b-2gpu-<date>/` and
`experiments/m11-h2-noshrink-task-gate-ep1-webshop-1.5b-2gpu-<date>/` and starts
nothing.

## Verification

[Tests](../tests/test_rebuttal_task_gate_ccpo.py), CPU only: the registry and
launcher deltas; exact equality with the hard gate's fallback; an independent
kernel computation; invariance to observation labels; task locality, in contrast
to cross-task; whole-trajectory exclusion; single-rollout tasks; the prepare
script; and the real trainer's advantage assembly on both benchmarks. With the
pinned verl-agent overlay synced in a scratch copy all of these pass, and every
other test file has the same outcome as before the change. No GPU run or training
result is claimed.
