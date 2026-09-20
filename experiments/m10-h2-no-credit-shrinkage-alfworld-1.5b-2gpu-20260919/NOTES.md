# M10 H2 without credit shrinkage

**Main-method designation — 2026-09-20:** This H2 / no-credit-shrinkage /
episode-weight-zero arm is now the selected main method for this benchmark.
See [the shared definition and ablation comparisons](../MAIN_METHOD.md).
This documentation update does not change the recorded training configuration.

**Prepared only. Not launched or queued.** ALFWorld, fresh Qwen2.5-1.5B-Instruct,
seed 0, 150 steps, two GPUs (0,1), future horizon 2, context-statistics vector enabled.
GPU IDs are the copied protocol setting, not a new reservation or launch.

Control: [recorded M10 H2](../m10-h2-ccpo-attncred-ctxadv-future-progress-alfworld-1.5b-2gpu-20260918/NOTES.md).
The only method delta is `ccpo_lk_fix: empty → 1.0`. All other config values are identical; the
experiment ID and output/cache locations are distinct.

## Baseline and advantage

For target q ∈ {Y, Z}, C_s[q] is the context-weighted baseline excluding the entire
query trajectory, U_s[q] is the task prior, and J_s counts distinct peer trajectories.
On supported exact observation groups, `B_s[q] = λ_s C_s[q] + (1−λ_s) U_s[q]`.
This arm uses **`λ_s = 1; B_s[q] = C_s[q]`**. A usable context baseline is used at full strength, including J=1. The recorded κ=2 cannot affect the pinned weight; the task prior is still available for diagnostics.

The setting applies to the historical return baseline and both current/future
potential readouts. Each endpoint uses its own group. No exact peer retains the
existing context-weighted task-bucket fallback at λ=1; no cross-trajectory peer
anywhere remains unsupported. There is no new confidence threshold. Terminal
potentials use 10/0 and have no estimated shrinkage coefficient.

`H_t = Y_t − B_t[Y]`; `Z_s = γ^(T−s) R_episode`; nonterminal `V_s = B_s[Z]`;
`F_t = z_task(V_min(t+2,T) − V_t)`; `A_t = mean_std_norm_task(H_t + F_t)`.
History/future weights stay 1/1, episode/original-edge weights stay 0/0. The separate
kernel-vs-uniform weight is already fixed at 1 in the control.

## Implementation and validation

[Experiment plan and math](../experiments.md), [ablation registry](../../ablations/ablations.py),
[resolved config](config.json), [exact config delta](config-diff-from-m10-h2.json),
[preparation status](PREPARED.json), [source hashes](prepared-source-sha256.json),
[CPU regression](../../tests/test_future_progress_ablation.py), [validation results](VALIDATION.json).

The existing `ACG_CCPO_PRIOR_KAPPA` / `ACG_CCPO_LK_FIX` code paths implement the
ablation; the estimator is shared with M10 H2. Logs already expose history/current/
future kernel baselines, task priors, J, λ, and fallback levels. Check the realised
λ values, rather than the unrelated diagnostic empirical-Bayes λ curve.

The prepared source includes [verified reference capture](../future-progress-verified-capture-20260919/NOTES.md).
For a controlled comparison, use the same code revision for the κ=2 control and
both ablations. Old GPU runs can differ in reference batching and floating-point
execution. CPU validation does not replace a GPU smoke test on the eventual host.

The local [run.sh](run.sh) uses the recorded training virtual environment and exact
protocol, and passes `bash -n`. It has not been executed. It embeds local paths and
is gitignored under the repository's normal convention; regenerate commands from
config/registry on another host. No controller or chain entry was created.

Validation completed: **25 CPU tests passed** (5 new H2 ablation checks plus 20 existing future-progress checks). The ablation registry guard and the full arm/document guard on a clean source export passed. The scan of this working container includes ignored compiled caches and generated launchers with local absolute paths; that environmental failure is recorded in VALIDATION.json.
