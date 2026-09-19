# M11 WebShop H2 without credit shrinkage

**Prepared only; not launched or queued.** Fresh Qwen2.5-1.5B-Instruct, two GPUs,
150 steps, seed 0, 8 rollouts per task. Recorded GPU IDs `2,3` are inherited
configuration placeholders, not a reservation. Prompt history remains two turns,
future horizon is two, and the hidden + accumulated context-stat vector is retained.

Registry key: `future-progress-h2-no-credit-shrinkage`, now enabled for both
ALFWorld and WebShop. Variant: `CCPO-ATTNCRED-FUTURE-PROGRESS-TWO-STEP-NOSHRINK`.
Canonical WebShop ID: `ccpo-attncred-abl-fph2-noshrink-ws-1.5b`.

## Control and interpretation

The sole method delta from the [prepared M11 H2 control](../m11-h2-ccpo-attncred-ctxadv-future-progress-webshop-1.5b-2gpu-20260918/NOTES.md)
is **`ccpo_lk_fix: "" → 1.0`**. `ccpo_prior_kappa=2` remains recorded but is inert
under this explicit full-strength override. Kernel-versus-uniform λ_u is already
1 in the parent and stays 1. Fusion weights do not change.

| Setting | M11 H2 control | This arm |
|---|---:|---:|
| Prompt history / future horizon | 2 / 2 | 2 / 2 |
| λ_k on usable exact groups | J / (J+2) | **1** |
| Context-stat weight | 1 | 1 |
| History / future weights | 1 / 1 | 1 / 1 |
| Episode / original-edge weights | 0 / 0 | 0 / 0 |

Relative to the completed original M11 H1, both horizon and shrinkage change.
Use the H2 κ=2 control to isolate shrinkage, on the same source revision and
verified reference-feature capture implementation.

The local experiment audit before preparing this arm found no M11 WebShop
no-shrinkage training record: completed M11 and M11-NOCTX use default κ=2
shrinkage. The earlier [step-150 analysis](../step150-posthoc-analysis-20260919/NOTES.md)
was an offline replay of the original **H1** snapshot: no shrinkage lowered
potential MSE by about 10% and increased conditional progress SD by about 45.1%.
Those are not results from a trained no-shrinkage policy and do not predict the
H2 arm's success rate. The previously prepared no-shrinkage arm was ALFWorld.

## Method

For target q in {Y, Z}, C_s[q] is the context-weighted baseline excluding the
entire query trajectory; U_s[q] is the correspondingly excluded task prior.

```
control:   B_s[q] = λ_s C_s[q] + (1 − λ_s) U_s[q],  λ_s = J_s/(J_s+2)
no-shrink: B_s[q] = C_s[q],                        λ_s = 1

H_t = Y_t − B_t[Y]
Z_s = γ^(T−s) R_episode,         V_s = B_s[Z]
F_t = z_task(V_min(t+2,T) − V_t)
A_t = H_t + F_t                (WebShop mean_norm)
```

The full-strength baseline applies to **history, current potential and future
potential**, including exact groups with only one peer trajectory. Each endpoint
uses its own observation group; history and potential still use their separate
return targets. Task standardization of F remains enabled; WebShop's mean_norm
path does not add a combined task-standard-deviation rescaling.

If no exact observation peer is available, retain the existing contextual task
fallback. If there is no cross-trajectory peer anywhere, the row remains
unsupported; no estimate is invented. Terminal endpoints retain fixed 10/0
success/failure potentials and no estimated shrinkage coefficient. The context
representation, grouping, invalid-action penalty, fallback support rules,
1,000-product catalogue and original item-option scorer are unchanged.

## Logging and preparation

Existing diagnostics retain history/current/future kernel baselines, task priors,
support J, effective support, λ_k and fallback levels, plus raw/normalized future
and applied H/F components. On usable readouts, λ_k must be 1 and baseline must
equal its kernel estimate. Task priors remain available for diagnostics even
though they have zero blend weight. Terminal future λ entries remain masked.

Files: [config](config.json), [delta from H2](config-diff-from-control.json),
[delta from original M11](config-diff-from-m11.json), [status](PREPARED.json),
[validation](VALIDATION.json), [source hashes](prepared-source-sha256.json).
The local gitignored `run.sh` contains the exact prepared command and environment;
it has not been executed. No controller/queue entry was created.

The [WebShop tests](../../tests/test_future_progress_webshop_ablation.py) exercise
all three full-strength readouts, J=1 support, unchanged component scaling,
inert κ under λ_k=1, detached finite advantages and exact config/runtime deltas.
The shared [H2 tests](../../tests/test_future_progress_ablation.py) also cover
fallback, terminal behavior and unsupported rows. GPU execution is untested.

Validation: **34 future-progress CPU tests passed** in the WebShop environment. The ablation registry guard, arm/document registry guard, shell syntax and diff formatting checks passed. No GPU was used.
