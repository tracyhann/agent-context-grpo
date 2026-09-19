# M11-NOCTX: hidden-only contextual future progress on WebShop

This ablation removes the accumulated trajectory-statistics vector from the similarity kernel used by M11. It applies to **historical credit and both current/future potential estimates**. The frozen reference model still encodes the task, current observation, and the existing two-turn text history.

Prepared and queued on 2026-09-19. The persistent controller waits for M10 to complete successfully, verifies its final checkpoint, and waits for GPU 0/1 capacity before starting. This is a **fresh training run from Qwen2.5-1.5B-Instruct**, not a continuation of the M11 checkpoint. Current queue state is linked below; being queued does not mean training has started.

| Setting | Value |
|---|---|
| Control | M11 contextual future progress, WebShop, 2026-09-18 |
| Backbone / seed | Qwen2.5-1.5B-Instruct / 0 |
| Training | 150 optimizer steps; 16 tasks × 8 trajectories |
| GPUs | Two A100 80GB, indices 0/1, after M10 |
| Catalogue | Existing 1,000-product WebShop dataset |
| Validation | Existing 256 episodes in two batches of 128; every 5 steps |
| Checkpoints | Every 5 steps; best, latest, and step-100 pin |
| Future horizon / fusion weight | 1 / 1 |
| History / episode / original-edge weights | 1 / 0 / 0 |
| Context-statistics coefficient | **0**, compared with **1** in M11 |
| Similarity feature dimension | **1536**, compared with **1573** in M11 |
| Prompt history | Existing two-turn text history |
| Other method and optimization settings | Identical to M11 |

## Exact feature change

Let \(h_{i,t}\) be the frozen reference model's last-prompt-token hidden vector. The retained hidden feature is

\[
x_{i,t}=\operatorname{L2}\left(\operatorname{removeTop3PC}(h_{i,t}-\bar h)\right).
\]

M11 concatenates this with a separately normalized 37-dimensional thermometer encoding of elapsed turns, unique observations, revisits, and observation novelty:

\[
\phi^{\mathrm{M11}}_{i,t}=\operatorname{L2}\left([x_{i,t},\operatorname{L2}(\operatorname{thermo}(c_{i,t}))]\right).
\]

This variant instead uses

\[
\boxed{\phi^{\mathrm{NOCTX}}_{i,t}=x_{i,t}}.
\]

The existing implementation already supports exactly this through `ccpo_ctx_w=0.0`. The configuration retains the compatibility selector `ccpo_phi=hidden+ctx`, but the code only constructs and concatenates the statistics block when the coefficient is strictly positive. At zero it returns the hidden feature directly; it does not append 37 zero-valued coordinates. No estimator or live-training source code was changed for this experiment.

The observation grouping and similarity kernel remain

\[
\mathcal P_{i,t}=\{(j,u):q_j=q_i,\ O_{j,u}=O_{i,t},\ j\ne i\},
\qquad
k_{(i,t),(j,u)}=\exp\left(-\frac{\|\phi_{i,t}-\phi_{j,u}\|_2}{\tau_g}\right),
\]

with the original group-median bandwidth rule \(\tau_g=0.15\operatorname{median}(d_g)\). The distances and therefore actual bandwidth values change with the representation, as intended. Whole-query-trajectory exclusion, task fallback, and credibility \(\lambda=J/(J+2)\) remain active.

## Returns and fusion

History retains its existing penalized return residual \(H_t=Y_t-B_t[Y]\). Potential labels retain the original unpenalized discounted binary outcome \(Z_{i,t}=0.95^{T_i-t}R_i\), with \(R_i\in\{0,10\}\). Both channels reuse the same processed hidden features. Current potential uses the current observation group; future potential uses the next observation group.

\[
P_t=\widehat V_{t+1}-\widehat V_t,\qquad
F_t=\frac{P_t-\mu_q(P)}{s_q(P)+10^{-6}},\qquad
\boxed{A_t=H_t+F_t}.
\]

Terminal potential is the observed binary outcome. There is no extra outer discount in the progress difference. WebShop retains M11's `mean_norm` protocol: the future term is task-standardized and the combined scalar is applied directly. No adaptive gate or fixed 0.25 coefficient is enabled. Advantages and reference features are detached.

The statistics remain in diagnostic snapshots (`current_context` and `future_context`) for analysis; their presence in a file does not mean they participate in similarity or credit computation. All history, current/future value, support, shrinkage, progress, and applied-component diagnostics remain enabled.

The original training and validation scorer is retained for the matched comparison. The separately explored item-option rescoring is not introduced into this ablation. Corrected scores should be compared only after rescoring both runs consistently.

## Verification

CPU replay of the existing M11 step-100 snapshot, 643 canonical rows, passed:

- Original M11 history, potential, raw progress, and processed features reproduced with zero maximum error.
- Hidden-only similarity has exactly 1536 dimensions and equals the retained whitened/normalized hidden vector.
- The future endpoint uses that endpoint's corresponding hidden-only feature.
- Replacing every context-statistics value with different values changes neither the historical credit nor current/future values, raw progress, or combined advantage.
- The combined scalar equals its logged history-plus-future components within 4.63e-7 and has no autograd connection.
- Configuration comparison has one method change, `ccpo_ctx_w: 1 → 0`; experiment identity and GPU indices are the only other differences.

Evidence: [VALIDATION.json](VALIDATION.json), [verification script](verify_features.py), [configuration diff](config-diff-from-m11.json), [configuration](config.json), [launch script](run.sh).

## Queue and monitoring

[Queue record](queue.json), [live controller state](../../.local/m11-noctx-chain-20260919/state.json), [queue manifest](../../.local/m11-noctx-chain-20260919/manifest.json), [controller log](../../.local/m11-noctx-chain-20260919/chain.log).

The queue is detached from the interactive session. It waits for successful M10 completion and final-checkpoint validation, then requires two consecutive GPU capacity checks. It also checks available process/thread capacity and prepared input hashes before launching. It does not stop M10 or external GPU jobs. The existing GPU-0/1 holder stays alive during the wait and through the new run; this placeholder is not an exclusive reservation enforced by a scheduler.

The controller records the actual training PID and exit status, monitors optimizer progress, and stops on an unrecognized training failure. Only recognized initialization-only memory failures can be archived and retried, with a bounded retry count. Final success requires step 150, final validation, and complete checkpoint shards. Runtime output will appear under `outputs/` after launch.

Control method: [M11 NOTES](../m11-ccpo-attncred-ctxadv-future-progress-webshop-1.5b-2gpu-20260918/NOTES.md). To reproduce the CPU check:

```bash
CUDA_VISIBLE_DEVICES='' .venv-webshop/bin/python experiments/m11-noctx-ccpo-attncred-hidden-future-progress-webshop-1.5b-2gpu-20260919/verify_features.py
```

Do not manually start a second copy of `run.sh` while the queue is active.

## 2026-09-19: finite padding-feature validation fix

The corrected [padding-feature patch](../future-progress-padded-phi.patch) was applied to the actual `ccpo/future_progress.py` at 2026-09-19T07:09:08.720316+00:00. Finite duplicate features use rtol=2^-7 and atol=1e-3; NaN/Inf in every row, including discarded padding copies, causes an error before estimation. Metadata remains exact. The estimator's first-occurrence readout and all advantage formulas are unchanged.

All 20 future-progress regression tests passed. This experiment's step-100 CPU replay also passed, including exact control parity and hidden-only context-invariance checks. Original launch manifests and validation records are preserved in `../../.local/padded-phi-apply-20260919/before/`.

This run started at 04:46 UTC with the original strict-equality implementation. The trainer was not restarted for this source update and retains the previously imported module. The refreshed VALIDATION.json is a CPU replay of the corrected code on disk, not evidence of a live-process upgrade.

Corrected source SHA-256: `dadf222ca39bb989b2be885b3fce5efc44132a0268fb84202c7d0a0a0615e288`.

## 2026-09-19: verified canonical reference capture

The actual trainer/reference-worker source and tracked overlays now use [verified canonical reference capture](../future-progress-verified-capture-20260919/NOTES.md). Source IDs are assigned before training padding; one reference feature/log-probability pair is retained per original row and restored by ID. Actual input fingerprints travel in the same tensor as hidden features through micro-batch reordering and DP collection. Identity errors, missing features, and nonfinite values still abort before optimization. Finite differences in temporary dispatch copies are logged separately.

CPU validation passed: 58 regressions in the ALFWorld environment and 10 capture regressions in the WebShop environment, including exact advantage/PPO-gradient parity with identical retained features. No real 8-H200 validation was performed. Advantage formulas and this experiment configuration are unchanged; canonical reference batching changes the floating-point execution layout, so comparisons to the older control include this implementation difference.

This already-running trainer was not restarted and retains its earlier imported source. The source update and refreshed prepared-input hashes do not constitute a live-process upgrade.

Previous runtime sources and queue hashes are preserved under `../../.local/phi-row-mapping-fix-20260919/before/`. Validation evidence is in [VALIDATION.json](../future-progress-verified-capture-20260919/VALIDATION.json).
