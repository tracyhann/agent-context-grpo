# M10 H2 no shrinkage, no episode: no LOO

**Prepared only; not launched or queued.** Qwen2.5-1.5B-Instruct, 150 training
steps, seed 0, 16 tasks x 8 rollouts, two GPUs (`0,1` configured,
not reserved). Turn ceiling 50; evaluation/checkpoints every 5 steps.

Registry key `future-progress-h2-no-credit-shrinkage-no-loo`; canonical ID `ccpo-attncred-abl-fph2-noshrink-noloo-alfworld-1.5b`.
Control: `experiments/m10-h2-no-credit-shrinkage-alfworld-1.5b-2gpu-20260919`. [Full config diff](config-diff-from-control.json) changes
only experiment identity and `ccpo_loo: 1 -> 0`; the previously implicit history
weight 1 and LOO default 1 are resolved explicitly for comparison.

## Exact ablation

**Full self-inclusion:** matching same-trajectory turns AND the query occurrence
itself are allowed in the exact (task, observation) peer group. This applies to
the history baseline, current potential and future endpoint potential. Different
observations do not become exact peers; DP padding copies are still deduplicated.
Current and future potentials retain their respective observation groups.

All usable readouts use full kernel strength: lambda_u=lambda_k=1. History/future
weights are 1/1; episode/original-edge weights are 0/0. History length 2 and future
horizon 2, hidden+context-statistics features, soft kernel tau scale 0.15, and
three-direction whitening are retained. Episode rewards still label the returns;
the separate episode advantage has zero applied PPO weight on BOTH benchmarks.

For q in {Y,Z}, let C_t^all[q] be the context-weighted mean over all exact
matches, including self. Then H_t=Y_t-C_t^all[Y], V_t=C_t^all[Z],
F_t=z_task(V_min(t+2,T)-V_t), and A=mask*N_CC(H+F).
Y is the historical penalized return; Z=gamma^(T-t)*R_episode is the unpenalized
potential label, gamma=0.95. N_CC is task mean/std normalization on ALFWorld and
identity on WebShop. Terminal V is success 10 / failure 0.

A single matching trajectory is sufficient, including a singleton query. A
singleton has H=0 and V=its own Z; F may remain nonzero from discounting or a
change at the endpoint. J counts distinct represented trajectories, now including
the query trajectory, while occurrence weights retain the existing aggregation.
The estimator stays detached, but allowing its own outcome into the baseline
removes the old trajectory-exclusion protection; this is a statistical ablation.

Equations, exact support semantics and diagnostics: [NO_LOO_ABLATIONS.md](../NO_LOO_ABLATIONS.md).
Self-occurrence and own-trajectory kernel mass are logged for history/current/
future, alongside peer row counts, J, n_eff, actual shrinkage coefficients,
component advantages and actor identity checks. Terminal future peer metrics are
undefined and excluded from means. Future-progress plots include self-mass panels.

## Reproduction

Prepare both benchmark arms for a fresh date:

```bash
python scripts/prepare_no_loo_ablations.py --date YYYYMMDD
```

For a later authorized launch from the project root:

```bash
bash experiments/m10-h2-noshrink-noloo-alfworld-1.5b-2gpu-20260921/run.sh
```

[config.json](config.json), [run.sh](run.sh), [prepare-command.json](prepare-command.json),
and [prepared-source-sha256.json](prepared-source-sha256.json) capture the resolved
launch recipe and sources. [VALIDATION.json](VALIDATION.json) records CPU checks.
WebShop retains the control's 15-turn, 1K-catalog benchmark protocol and scorer.
No training result or GPU validation is claimed by preparation.
