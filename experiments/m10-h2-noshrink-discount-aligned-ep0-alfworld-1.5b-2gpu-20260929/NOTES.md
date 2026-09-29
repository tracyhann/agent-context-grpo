# Historical + discount-aligned future residual: alfworld, EP0

**Prepared only; not launched or queued.** Qwen2.5-1.5B-Instruct, 150 steps,
seed 0, two GPUs (`0,1` configured, not reserved).
Registry: `future-progress-h2-no-credit-shrinkage-discount-aligned`. Matched control: `experiments/m10-h2-no-credit-shrinkage-alfworld-1.5b-2gpu-20260919`.

Only `ccpo_progress_mode: difference -> discount_aligned` changes the method;
[the full diff](config-diff-from-control.json) also records experiment identity
and previously implicit defaults. This arm retains task-observation matching.

```text
Z_s = gamma^(T-s) * R
V_s = C_s[Z]                    # same contextual estimator and peer weights
H_t = Y_t - C_t[Y]              # unchanged historical residual
u = min(t+2, T); h = u-t        # actual elapsed turns, not always the configured 2
F_TD_t = z_task(gamma^h * V_u - V_t)
A_actor = mask * N_task(H + F_TD)
```

Gamma is 0.95. Near the terminal, h can be 1; terminal potentials remain
success 10 / other endings 0. There is no new immediate-reward term. Historical
invalid-action penalties stay in Y, outside the unpenalized value labels Z.
History/future/episode weights are 1/1/0; lambda_u=lambda_k=1.
The frozen reference, hidden+context features, whole-trajectory LOO, soft kernel,
tau, horizon, target, training protocol and task-wise future standardization
remain matched. ALFWorld normalizes H+F together; WebShop retains mean_norm and
adds its episode channel after the step channel, exactly as the EP1 control.

## Reviewer question

The original raw difference decomposes as
`V_u - V_t = (gamma^h*V_u - V_t) + (1-gamma^h)*V_u`.
This experiment removes the second term BEFORE task-wise z-scoring. Subtracting
separately standardized components would define a different experiment.
For discount-consistent potentials on an otherwise redundant trajectory, the
new raw residual is zero (up to floating-point roundoff), whereas the original
difference can remain nonzero after task-wise centering. This does not establish
that any remaining contextual residual uniquely measures semantic or causal
progress; it isolates this particular time-discount contribution.

## Audit outputs and acceptance tests

`outputs/future_progress/step-*.npz` and CSVs include `window_length`,
`endpoint_discount`, `future_credit_mode`, `undiscounted_raw`,
`discount_aligned_raw`, `time_passage_raw`, selected `raw_progress`,
`progress_normalized`, historical/future/episode applied credits and actor credit.
Metrics include `progress_discount_aligned`, `progress_gamma` and
`progress_decomposition_max_error`. The M5 edge correlation remains the original
one-step, undiscounted diagnostic reference.

CPU tests cover cancellation of pure discount-clock changes, gamma=1 parity,
clipped terminal windows, discounted telescoping, padding/LOO/penalty isolation,
actual trainer routing, masked PPO gradients and EP0/EP1 channel composition.
No training results are claimed by this preparation.

Launch from the repository root, after selecting available GPUs:
`bash experiments/m10-h2-noshrink-discount-aligned-ep0-alfworld-1.5b-2gpu-20260929/run.sh`.
