# M11 WebShop H2, κ=4

**Prepared only; not launched or queued.** Fresh Qwen2.5-1.5B-Instruct, two GPUs,
150 steps, seed 0, 8 rollouts per task. Recorded GPU IDs `2,3` are inherited
configuration placeholders, not a reservation. WebShop retains its 1,000-product
catalogue, original item-option scorer, binary-return credit target and
`mean_norm` training protocol.

Registry key: `future-progress-h2-kappa4` (now supports ALFWorld and WebShop).
Canonical ID: `ccpo-attncred-abl-fph2-kappa4-ws-1.5b`.

| Setting | Original M11 | M11 H2 control | This arm |
|---|---:|---:|---:|
| Prompt history length | 2 | 2 | 2 |
| Future horizon | 1 | 2 | 2 |
| κ | 2 | 2 | 4 |
| Context-stat vector weight | 1 | 1 | 1 |
| History / future weights | 1 / 1 | 1 / 1 | 1 / 1 |
| Episode / original-edge weights | 0 / 0 | 0 / 0 | 0 / 0 |

The sole method delta against the [prepared M11 H2 control](../m11-h2-ccpo-attncred-ctxadv-future-progress-webshop-1.5b-2gpu-20260918/NOTES.md)
is `ccpo_prior_kappa: 2 → 4`. Relative to original completed M11, this changes
**both** horizon and κ; that comparison cannot isolate the κ effect.

For target q in {Y, Z}, and each usable exact observation group:

```
B_s[q] = λ_s C_s[q] + (1 − λ_s) U_s[q]
λ_s = J_s / (J_s + 4)
H_t = Y_t − B_t[Y]
Z_s = γ^(T−s) R_episode,       V_s = B_s[Z]
F_t = z_task(V_min(t+2,T) − V_t)
A_t = H_t + F_t              (WebShop mean_norm; no extra task std scaling)
```

C is the context-weighted cross-trajectory readout, U the task prior, and J the
number of distinct supporting peer trajectories. The entire query trajectory
is excluded from both. κ=4 affects the history baseline and current/future
potential baselines. Each endpoint uses its own observation group and history
features. Missing exact peers retain contextual task fallback; terminal endpoints
retain fixed success/failure potential 10/0, without an estimated λ.

The support weight is smaller at fixed J: for J=1 it becomes 1/5 instead of 1/3;
for J=7 it becomes 7/11 instead of 7/9. This changes baseline shrinkage, not fusion.
Existing logging records history/current/future kernel estimates, priors, support,
λ, fallback levels, raw and normalized progress, and applied H/F components.

Files: [config](config.json), [delta from H2](config-diff-from-control.json),
[delta from original M11](config-diff-from-m11.json), [status](PREPARED.json),
[CPU validation](VALIDATION.json), [source hashes](prepared-source-sha256.json).
The local `run.sh` contains the exact recorded command and environment; it is
unexecuted and gitignored under the existing repository convention.

The shared runtime already implements this κ/horizon combination. This change
registers it for WebShop and verifies it under the WebShop component scaling.
Use the same current source, including verified reference-feature capture, for
the κ=2 H2 control and this arm. CPU checks do not establish GPU smoke-test success.

Validation: **33 future-progress CPU tests passed** in the WebShop virtual environment, including 3 new checks. The ablation registry guard, arm/document registry guard and generated shell syntax checks passed. No GPU was used.
