# ALFWORLD H2 no shrinkage, EP0: cross-task soft grouping

**Prepared only; not launched or queued.** Qwen2.5-1.5B-Instruct, 150 training
steps, seed 0, two GPUs (`0,1` configured, not reserved).
Registry key `future-progress-h2-no-credit-shrinkage-cross-task`; canonical ID `ccpo-attncred-abl-fph2-noshrink-cross-task-alfworld-1.5b`.

## Change from control

Change only `ccpo_gate: hard -> cross_task` from `experiments/m10-h2-no-credit-shrinkage-alfworld-1.5b-2gpu-20260919`.
[Full config diff](config-diff-from-control.json) also records experiment identity;
legacy implicit history weight and whole-trajectory LOO are made explicit.
Both benchmarks retain EP=0, lambda_u=lambda_k=1, history=2, future horizon=2,
binary-return targets, and the original soft kernel with tau scale 0.15.

For query i, P_i contains every occurrence in this canonical rollout batch
except occurrences belonging to i's own trajectory. Neither task UID nor
observation equality restricts P_i. ALFWorld and WebShop have separate runs and
separate batches; cross-task does not mean pooling the two environments.
Padding copies are deduplicated first; real repeated visits retain their mass.

```text
phi = existing batch-whitened frozen hidden state + trajectory-context features
w_ij = exp(-||phi_i - phi_j|| / tau), j in P_i
C_i[q] = sum(w_ij * q_j) / sum(w_ij)
tau = 0.15 * median distance over all distinct row pairs in this batch
H_i = Y_i - C_i[Y]
Z_i = gamma^(T_i-t_i) * R_i; V_i = C_i[Z]
F_i = z_task(V_min(t_i+2,T_i) - V_i)
```

The median includes same-trajectory pairs, following the existing bandwidth
rule; the readout always excludes them. A degenerate median uses 1 before the
0.15 multiplier. The normalized kernel uses a numerically stable common shift.
No hard task/observation gates, top-k cutoff, task prior, or task fallback apply.
`ccpo_prior_kappa` and `ccpo_backoff_task` remain recorded control values but are
inactive under `cross_task`. With no other trajectory, contextual credit is
unsupported and zero. Lambda_u=lambda_k=1 wherever supported.

History Y retains the existing invalid-action penalty; potential labels Z do
not. Terminal potentials remain success 10 / failure 0, gamma=0.95. The future
endpoint uses its own context to weight the same batch-wide candidate pool.
Task IDs still define task-wise advantage normalization, not peer eligibility.

**A = mask * N_task(H + F)**. H/F/EP weights are **1 / 1 / 0** on BOTH benchmarks, original
edge weight zero. ALFWorld retains its combined per-task mean/sample-std
normalization. WebShop retains mean_norm, without final contextual normalization.
The actor/reference prompts retain history length 2 and the task goal.

## Protocol and diagnostics

16 tasks x 8 rollouts, 150 steps, 50-turn cap, save/evaluate every
5 steps. Prompt/response budgets, optimizer, reward and environment settings
match the control: ALFWorld 2048/512; WebShop 4096/512 with its 1K catalog.
The exact distance matrix uses about 313 MiB at the 6,400-row ALFWorld ceiling;
median selection temporarily adds about 156 MiB. No additional model forward.

`ccpo/progress_cross_task_grouping=1` identifies the variant. History/current/
future snapshots include cross-task normalized weight mass, peer task count,
peer rows, effective trajectory support and kernel bandwidth. Same-trajectory
mass is zero; cross-task mass can be zero for a one-task batch. Terminal future
rows have no neighborhood statistics. Existing H/F/EP actor-sum logging remains.

```bash
python scripts/prepare_cross_task_ablations.py --date YYYYMMDD
# Launch from the project root when ready:
bash experiments/m10-h2-noshrink-cross-task-alfworld-1.5b-2gpu-20260928/run.sh
```

[Shared definition](../CROSS_TASK_SOFT_GROUPING.md).
[config.json](config.json), [PREPARED.json](PREPARED.json),
[prepare-command.json](prepare-command.json), [source hashes](prepared-source-sha256.json)
and [VALIDATION.json](VALIDATION.json) record preparation and CPU validation.
No training result or GPU smoke test is claimed.
