# M10-H2-NOCTX: two-step future progress with hidden-only similarity

**Prepared only; not launched or queued.** Fresh Qwen2.5-1.5B-Instruct, ALFWorld,
seed 0, 150 steps, two GPUs (0,1). GPU IDs are copied configuration, not a reservation.

Control: [M10 H2](../m10-h2-ccpo-attncred-ctxadv-future-progress-alfworld-1.5b-2gpu-20260918/NOTES.md).
The sole method change is `ccpo_ctx_w: 1 → 0`. Horizon remains **2**; κ remains **2**;
`ccpo_lk_fix` remains empty; history/future weights remain 1/1 and episode/edge weights
remain 0/0. This does not combine the no-shrinkage or κ=4 ablation with hidden-only
features. The existing queued M10-NOCTX is H1 and is a different experiment.

## Representation and math

The prepared config retains `ccpo_phi=hidden+ctx` and its original one-setting
delta. After the 2026-09-19 compatibility fix, `ccpo_phi=hidden` is also accepted
and numerically equivalent; frozen features remain mandatory.
`ACG_CCPO_CTX_W=0` skips context-statistics block construction/concatenation in
[core_ccpo.py](../../ccpo/core_ccpo.py:698). The resulting φ is the 1536-dimensional
frozen last-prompt-token hidden state after batch centering, removal of the top
three principal components, and L2 normalization. The control has 1573 dimensions.
The model prompt retains its existing two-turn text history; this is a removal of
the additional accumulated context-statistics vector, not of historical text.

The shared estimator uses that representation in the historical baseline B_t[Y]
and current/future nonterminal potential V_s=B_s[Z], with Z_s=γ^(T−s)R. Each
endpoint keeps its own exact `(task, observation)` group, excluding the entire
query trajectory. Kernel weights are `exp(-||φ_s-φ_j||₂/τ_group)`, followed by the
existing λₖ=J/(J+2) task-prior blend. J=0 retains the existing task-bucket fallback;
no usable cross-trajectory baseline remains masked. Terminal V_T is 10/0.

```
H_t = Y_t − B_t[Y]
F_t = z_task(V_min(t+2,T) − V_t)
A_t = mean_std_norm_task(H_t + F_t)
```

The existing [future-progress implementation](../../ccpo/future_progress.py:173)
reuses the processed historical feature matrix for the potential readout; no
separate feature transform or model pass is introduced. Context-statistics arrays
can still be logged for analysis, but do not feed similarity or advantages.

## Artifacts and validation

- [Config](config.json) and [exact difference from M10 H2](config-diff-from-m10-h2.json).
- [Preparation status](PREPARED.json), [source hashes](prepared-source-sha256.json), and [validation](VALIDATION.json).
- [Registry](../../ablations/ablations.py): `future-progress-h2-no-context-vector`.
- [Experiment ledger](../experiments.md) and [H2 regression tests](../../tests/test_future_progress_ablation.py).

The local generated `run.sh` invokes the recorded virtual environment and uses
isolated output/cache paths; it is gitignored under the normal repository
convention. No command was executed to launch training, and no controller or
chain entry was created. The prepared code includes verified reference capture;
use the same revision for the H2 control when evaluating the ablation.

Validation completed: **26 CPU regressions passed**, including the 1536/1573-dimensional representation comparison, t+2 hidden alignment, context-statistics invariance and retained κ=2 shrinkage. Registry guards and the complete arm/document guard on a clean source export passed. No GPU smoke test or training run was performed.
