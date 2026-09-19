# M10-NOCTX: hidden-only future progress on ALFWorld

Prepared and queued on 2026-09-19 at the user's request. This is a fresh Qwen2.5-1.5B-Instruct, seed-0 experiment for 150 optimizer steps. The method changes only `ccpo_ctx_w: 1 → 0` relative to the original M10. The experiment identity is the only other configuration difference.

## Run settings

| Setting | Value |
|---|---|
| Control | M10 ALFWorld, 2026-09-18 |
| Backbone / initialization | Qwen2.5-1.5B-Instruct / fresh base model |
| Seed / training length | 0 / 150 optimizer steps |
| GPUs | Two A100 80GB, indices 0/1 |
| Training batch | 16 tasks × 8 trajectories |
| Evaluation | Existing 128 eval-in-distribution episodes, batches of 32, every 5 steps |
| Checkpoints | Every 5 steps; best, latest, and step-100 pin |
| Prompt history / maximum actions | 2 turns / 50 |
| Future progress horizon / weight | 1 / 1 |
| History / episode / original-edge weights | 1 / 0 / 0 |
| Context-statistics weight | 0 |
| Similarity representation | 1536-dimensional processed hidden state |
| Kernel / credibility | Exponential Euclidean-distance kernel / J/(J+2) |
| Final fusion normalization | M10's `mean_std_norm` |

## Representation and method

Let h be the frozen reference model's last-prompt-token hidden state. It still encodes the task, current observation, and existing two-turn text history. The feature is

\[
z_{i,t}=\operatorname{L2}\bigl(\operatorname{removeTop3PC}(h_{i,t}-\bar h)\bigr).
\]

The shared batch centering, removal of the top three principal directions, and L2 normalization remain. The accumulated trajectory-statistics block is omitted entirely: 1536 coordinates instead of the control's 1573. The compatibility flag `ccpo_phi=hidden+ctx` remains, but at `ccpo_ctx_w=0` the implementation does not concatenate statistics or zero-valued placeholder coordinates. There is no learned projection added.

This change applies to the historical baseline and both the current and future potential estimates. The future endpoint reuses its own processed hidden state. Exact-observation grouping remains separate at each endpoint, with the whole query trajectory excluded. The kernel remains

\[
w_{(i,s),(j,u)}=\exp\left(-\frac{\|z_{i,s}-z_{j,u}\|_2}{\tau_g}\right),\qquad
\tau_g=0.15\operatorname{median}(d_g),
\]

with the existing near-zero bandwidth fallback. The credibility coefficient remains \(\lambda=J/(J+2)\), where J counts distinct other trajectories. Unsupported observation groups retain the existing task-level kernel fallback.

History uses the existing penalized return-to-go Y; the potential channel uses unpenalized labels \(Z_{i,t}=0.95^{T_i-t}R_i\), with binary episode outcome \(R_i\in\{0,10\}\). Let B denote the unchanged kernel-and-shrinkage baseline operator:

\[
H_{i,t}=Y_{i,t}-B_{i,t}[Y],\qquad
\widehat V_{i,s}=B_{i,s}[Z].
\]

The terminal potential is the observed outcome. One-step future progress and fusion remain

\[
P_{i,t}=\widehat V_{i,t+1}-\widehat V_{i,t},\qquad
F_{i,t}=\frac{P_{i,t}-\mu_q(P)}{s_q(P)+10^{-6}},\qquad
A^{\mathrm{pre}}_{i,t}=H_{i,t}+F_{i,t}.
\]

There is no additional outer discount, reward bonus, adaptive gate, or 0.25 scaling. ALFWorld retains the final combined task normalization:

\[
A_{i,t}=\frac{A^{\mathrm{pre}}_{i,t}-\mu_q(A^{\mathrm{pre}})}{s_q(A^{\mathrm{pre}})+10^{-6}}.
\]

The implementation uses the existing supported/live-row rules, sample standard deviations, and padded-row restoration before final normalization. Features and advantages remain detached. Episode and original-edge terms are diagnostics only, with both fusion weights zero.

All existing history, potential, support, shrinkage, progress, and applied-component logging remains enabled. Context-statistics arrays remain available for diagnostics, but do not enter similarity or credit computation.

## Validation

[CPU replay](verify_features.py) of the original M10 step-100 snapshot passed on **2,450 canonical rows**:

- Reproduced original history advantage, current potential, raw progress, and processed features with zero maximum error.
- Confirmed the new representation is exactly the processed 1536-dimensional hidden vector and that the future endpoint uses its corresponding hidden feature.
- Perturbing all context-statistics vectors leaves history, both potentials, raw progress, and combined advantage exactly unchanged.
- Confirmed detached advantages and finite final normalized values. Applied history and future components sum to the normalized canonical-row advantage within 4.59e-7.
- Confirmed the configuration differs only in the context coefficient and experiment identity.

The replay verifies canonical rows; live training retains the control's padded-row restoration. It uses CPU only and starts no environment rollout. No estimator or trainer source code was changed.

Evidence: [VALIDATION.json](VALIDATION.json), [configuration diff](config-diff-from-m10.json), [config.json](config.json), [run.sh](run.sh).

## Persistent queue

The order is **current M10 ALFWorld → M11 WebShop hidden-only → this M10 ALFWorld hidden-only** on GPUs 0/1. The existing holder remains alive. The queue is detached from the interactive session.

The controller waits for successful predecessor completion and verifies its final step-150 validation and checkpoint shards. It then checks GPU identity/capacity, available process capacity, prepared-input hashes, and FlashAttention readiness before starting from the base model. It waits when GPUs are occupied and records training progress and exit status. Only recognized initialization-only memory failures have bounded retries; other failures stop the chain for inspection. Completion requires step 150, final validation, and complete checkpoint shards.

[Queue record](queue.json), [live controller state](../../.local/m10-noctx-chain-20260919/state.json), [manifest](../../.local/m10-noctx-chain-20260919/manifest.json), [controller log](../../.local/m10-noctx-chain-20260919/chain.log), [prelaunch verification](../../.local/m10-noctx-chain-20260919/PRELAUNCH_VALIDATION.json).

Queued does not mean training has started. Do not launch a second manual copy while the controller is active.

Control: [M10 method notes](../m10-ccpo-attncred-ctxadv-future-progress-alfworld-1.5b-2gpu-20260918/NOTES.md).

## 2026-09-19: finite padding-feature validation fix

The corrected [padding-feature patch](../future-progress-padded-phi.patch) was applied to the actual `ccpo/future_progress.py` at 2026-09-19T07:09:08.720316+00:00. Finite duplicate features use rtol=2^-7 and atol=1e-3; NaN/Inf in every row, including discarded padding copies, causes an error before estimation. Metadata remains exact. The estimator's first-occurrence readout and all advantage formulas are unchanged.

All 20 future-progress regression tests passed. This experiment's step-100 CPU replay also passed, including exact control parity and hidden-only context-invariance checks. Original launch manifests and validation records are preserved in `../../.local/padded-phi-apply-20260919/before/`.

This experiment is still queued. Its launch-input hashes have been updated after validation, so it will load the corrected implementation when the existing queue starts it.

Corrected source SHA-256: `dadf222ca39bb989b2be885b3fce5efc44132a0268fb84202c7d0a0a0615e288`.
