# M10 H2: no credit shrinkage + no context-stat vector

**Prepared only; not launched or queued.** ALFWorld, fresh
Qwen2.5-1.5B-Instruct, **150 steps**, seed 0, two GPUs, 8 rollouts per task.
GPU IDs `0,1` are copied protocol placeholders, not a reservation.

Registry key: `future-progress-h2-no-credit-shrinkage-no-context-vector`.
Variant: `CCPO-ATTNCRED-FUTURE-PROGRESS-TWO-STEP-NOSHRINK-NOCTX`.
Canonical ID: `ccpo-attncred-abl-fph2-noshrink-noctx-alfworld-1.5b`.

This combines the two existing H2 interventions. Relative to the
[M10 H2 control](../m10-h2-ccpo-attncred-ctxadv-future-progress-alfworld-1.5b-2gpu-20260918/NOTES.md),
there are exactly two method changes:

```
ccpo_ctx_w:  1.0 → 0.0
ccpo_lk_fix: ""  → 1.0
```

## Paired controls

| Arm | Context-stat weight | Usable exact-group λ_k |
|---|---:|---:|
| M10 H2 control | 1 | J/(J+2) |
| M10 H2 NOCTX | 0 | J/(J+2) |
| M10 H2 NOSHRINK | 1 | 1 |
| **This joint arm** | **0** | **1** |

All four have future horizon 2. Against
[H2 NOCTX](../m10-h2-noctx-alfworld-1.5b-2gpu-20260919/NOTES.md), this isolates
shrinkage when using hidden-only similarity. Against
[H2 NOSHRINK](../m10-h2-no-credit-shrinkage-alfworld-1.5b-2gpu-20260919/NOTES.md),
it isolates the context-statistics block under full-strength readout. Comparison
with the H2 parent changes both components. Use the same source revision for
these controls; the existing H1 M10-NOCTX run has a different future horizon.

## Representation and advantage

`ccpo_phi=hidden+ctx` remains the compatible interface flag, while
`ccpo_ctx_w=0` skips the context-statistics block entirely. The actual similarity
representation is **1536-dimensional processed frozen hidden features** on 1.5B:
batch centering, removal of the top three principal components, then L2
normalization. The equivalent `ccpo_phi=hidden` spelling is also supported by the
current runtime, but is not needed as a third config change.

The actor/reference prompt still contains its normal two-turn observation-action
history. This removes the additional accumulated statistics vector, not the
historical text. Context-statistic perturbations cannot affect the estimator.

For each target q in {Y, Z}, C_s[q;φ] is the kernel-weighted baseline excluding
the entire query trajectory:

```
φ_s = normalize(remove_top_3_PC(h_s − mean(h)))
w_sj = exp(−||φ_s − φ_j||₂ / τ_group)
B_s[q] = C_s[q;φ]                (λ_k = 1 on usable readouts)

H_t = Y_t − B_t[Y]
Z_s = γ^(T−s) R_episode,         V_s = B_s[Z]
F_t = z_task(V_min(t+2,T) − V_t)
A_t = mean_std_norm_task(H_t + F_t)
```

The hidden-only/full-strength readout is shared by the history baseline and
current/future potential baselines. Each endpoint uses its own observation group.
The context kernel remains enabled; no-context-vector does not mean uniform
weighting. `ccpo_prior_kappa=2` remains recorded but is inert under the explicit
`ccpo_lk_fix=1` override. Kernel/uniform λ_u remains pinned to 1.

With no exact-group peers, use the existing task-bucket fallback with hidden-only
similarity; with no cross-trajectory peers anywhere, the estimate remains
unsupported. Terminal endpoints retain fixed 10/0 success/failure potentials and
no estimated λ. No new minimum-support threshold is added.

History/future fusion weights remain 1/1. Episode/original-edge weights remain
0/0. No shrinkage removes baseline mixing with the task prior; future task
standardization and ALFWorld's final combined standardization remain enabled.

## Logging and preparation

Existing snapshots/metrics retain history/current/future kernel estimates,
task priors, support J, effective support, λ_k, fallback levels, hidden/processed
features and applied H/F components. On usable readouts, λ_k must be 1 and the
baseline must equal its kernel estimate. Nonterminal future φ must be the
hidden-only vector at the t+2 endpoint (clipped at terminal). Terminal λ is masked.

Files: [config](config.json), [exact H2 delta](config-diff-from-m10-h2.json),
[status](PREPARED.json), [validation](VALIDATION.json),
[source hashes](prepared-source-sha256.json). The local gitignored `run.sh` is
prepared and unexecuted. No controller or queue entry was created.

The [H2 CPU regression](../../tests/test_future_progress_ablation.py) validates
the joint combination: 1536-dimensional features, context-statistics invariance,
full-strength readouts including J=1, inert κ, retained endpoints, fallback and
terminal behavior, unsupported rows, and detached finite advantages. Existing
future-progress tests cover trainer normalization, gradient isolation and strict
feature checks. Source includes verified reference-feature capture. GPU execution
is untested; copied runtime settings and package versions need checking at launch.

Validation: **35 future-progress CPU tests passed** in the ALFWorld environment, including the joint-ablation regression. Registry/document guards, generated shell syntax and diff formatting checks passed. No GPU was used.
