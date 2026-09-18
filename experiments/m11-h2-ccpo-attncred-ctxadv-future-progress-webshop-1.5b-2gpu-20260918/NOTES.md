# M11-H2: contextual future-state progress — webshop

**Prepared only. Not launched or queued.** Fresh Qwen2.5-1.5B-Instruct, seed 0, 150 steps, two GPUs (2,3). Horizon 2; history coefficient 1; separately task-standardized future-progress coefficient 1; episode coefficient 0; original edge coefficient 0.

The control is `m5-ccpo-attncred-ctxadv-ret-ws-1.5b-2gpu-20260916`. All shared training, environment, optimization, validation and checkpoint settings match its recorded config. The only changed pre-existing method setting is `ccpo_edge_w: 1 → 0`; the new progress channel replaces it. The method-specific settings and inactive defaults added since that historical run are recorded in [config-diff-from-control.json](config-diff-from-control.json).

The primary experiment uses horizon 1. Horizon 2 is an explicitly separate prepared ablation; its raw value gain telescopes over the next two transitions, clipped at the recorded terminal endpoint. It does not divide by the window length.

## Method and math

Full definitions, terminal conventions, normalization, M5 equivalence and diagnostics: [FUTURE_PROGRESS.md](../../ccpo/FUTURE_PROGRESS.md).

`H_t = Y_t - B_t`, `P_t = V_context(t+2) - V_context(t)`, and `A_pre = H_t + task_standardize(P_t)`.

Current and future values use separate `(task, observation)` groups, the existing frozen reference prompt features and accumulated prefix statistics at each endpoint, whole-query-trajectory exclusion, and the production attention/credibility estimator. The potential target is the M3/M5 convention `Z_t = gamma^(T-t) R_episode`, without invalid-action penalties. Terminal potentials are 10 on success and 0 otherwise, including horizon exhaustion. No extra future model pass, gamma multiplier, immediate reward addition, or convex beta mix is used.

ALFWorld applies its existing final per-task mean/std normalization to the combined credit. WebShop applies no additional combined normalization. The future term itself is task-standardized in both benchmarks, exactly as the M3/M5 edge was.

## Artifacts when launched

Every training step will write scalar `ccpo/progress_*` metrics and `outputs/future_progress/step-NNNN.csv`, `.npz`, and `.metrics.json`: historical baseline/residual; unpenalized current/future potentials; both endpoint kernel/prior/support/shrinkage terms; raw and standardized progress; the original M5 edge diagnostic; applied history/future components; source and endpoint indices, features, context vectors and terminal flags. `plots/future_progress.png` exposes the corresponding curves.

Terminal future values use fixed sentinels and have no hidden vector or peer estimate; their corresponding snapshot fields are NaN and explicitly masked. Snapshots contain no pickle-dependent metadata. Full prompt strings are not needed for this variant's offline reproduction: both actual frozen hidden vectors and processed context features are saved.

CPU validation and replay findings will be recorded in [VALIDATION.md](../../experiments/future-progress-offline-20260918/VALIDATION.md). Prepared commands: [run.sh](run.sh); resolved settings: [config.json](config.json). No controller or chain entry was created.
