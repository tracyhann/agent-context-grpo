# M11 WebShop 1.5B: no-shrink, no-episode component ablations

Prepared on **2026-09-22**. These three separate variants use the
[main WebShop H2 no-shrink EP0 control](m11-h2-no-credit-shrinkage-webshop-1.5b-2gpu-20260919/NOTES.md).
All use Qwen2.5-1.5B-Instruct, 150 steps, seed 0, two GPUs, history-2 prompts,
15-turn WebShop, 16 tasks x 8 rollouts, and whole-query-trajectory LOO.
**Prepared only; none launched or queued.**

| Variant | H / F / episode weights | Context-stat weight | Sole method change from main |
|---|---|---:|---|
| [Without future](m11-h2-noshrink-history-only-webshop-1.5b-2gpu-20260922/NOTES.md) | 1 / 0 / 0 | 1 | `ccpo_progress_weight=0` |
| [Without history credit](m11-h2-noshrink-future-only-webshop-1.5b-2gpu-20260922/NOTES.md) | 0 / 1 / 0 | 1 | `ccpo_progress_history_weight=0` |
| [Without context-statistics vector](m11-h2-noshrink-noctx-webshop-1.5b-2gpu-20260922/NOTES.md) | 1 / 1 / 0 | 0 | `ccpo_ctx_w=0` |

Every variant keeps `ccpo_lk_fix=1`, `ccpo_lam_fix=1`, `ccpo_ep_w=0`,
`ccpo_edge_w=0`, `ccpo_loo=1`, and `ccpo_progress_horizon=2`.
The older WebShop history/future component variants use **EP1** and remain intact;
this set uses EP0 throughout. The existing ALFWorld EP0 registry entries now also
support WebShop with the correct benchmark overlay. No estimator or trainer
implementation change was needed.

## Method

Let C_t[q] denote the contextual kernel mean in the query's exact task/observation
group, excluding its entire trajectory. Retain the existing task-bucket fallback
when exact peers are unavailable. No shrinkage gives B_t[q]=C_t[q] on usable
readouts, including a group with only one peer trajectory. Kappa 2 remains a
recorded diagnostic parameter and cannot change a fixed lambda_k=1 readout.

\[
H_t=Y_t-C_t[Y],\qquad Z_t=\gamma^{T-t}R_{\rm episode},\qquad V_t=C_t[Z],
\]
\[
F_t=z_{\rm task}(V_{\min(t+2,T)}-V_t),\qquad
A_{t,\ell}=M_{t,\ell}(w_HH_t+w_FF_t),\quad \gamma=0.95.
\]

Y is the trainer's penalized discounted history target; Z is the unpenalized
potential label. Each nonterminal endpoint uses its own observation group;
terminal potential remains success 10 / failure 0. F is task-standardized;
WebShop's mean_norm path adds no combined standard-deviation normalization.
Only enabled channels contribute to the contextual support mask. All estimates
remain detached, with KL applied separately.

- **Without future:** w_H=1, w_F=0. H2 future values and credit remain diagnostic,
  with exactly zero applied future credit. Future-only support cannot affect the
  applied history channel. This removes future-progress credit; it retains
  episode-return labels in the historical residual.
- **Without history credit:** w_H=0, w_F=1. Historical-residual credit is zero in
  the actor update. Both actor and reference prompts still contain two history
  turns, and context statistics remain enabled in the potential readouts.
- **Without context-statistics vector:** w_H=w_F=1. The statistics block is
  removed from history/current/future similarity representations. Frozen hidden
  features retain their whitening/PCA processing, normalization and soft
  exponential kernel (tau scale 0.15). This is `hidden+ctx` with CTX_W=0,
  equivalent to the supported `hidden` mode. It is not a cosine-similarity arm.

Episode rewards still supply Y/Z labels. Raw episode advantage and original-edge
references remain logged, but both have **zero applied actor weight**. Disabled
H/F channels also remain diagnostic with zero applied contribution.

## Registry and reproduction

| Variant | Registry key | Canonical WebShop ID |
|---|---|---|
| No future | `future-progress-h2-no-credit-shrinkage-history-only` | `ccpo-attncred-abl-fph2-noshrink-history-only-ws-1.5b` |
| No history credit | `future-progress-h2-no-credit-shrinkage-future-only` | `ccpo-attncred-abl-fph2-noshrink-future-only-ws-1.5b` |
| No context statistics | `future-progress-h2-no-credit-shrinkage-no-context-vector` | `ccpo-attncred-abl-fph2-noshrink-noctx-ws-1.5b` |

```bash
# Prepare all three for a fresh date; does not launch.
python scripts/prepare_webshop_ep0_ablations.py --date YYYYMMDD

# Select one, also without launching.
python scripts/prepare_webshop_ep0_ablations.py --variant no-future --date YYYYMMDD
```

The preparer refuses to overwrite an existing experiment. Each folder includes
NOTES.md, config.json, run.sh, PREPARED.json, the exact EP0-control diff, preparation
command, source hashes and CPU VALIDATION.json. GPU IDs `2,3` are inherited
configuration, not reservations; the three experiments are not scheduled to run
concurrently. Checkpoint/evaluation frequency 5, 1K catalog and scorer settings
are inherited from the matched main protocol.

## Validation and diagnostics

History/current/future readouts, J/n_eff/lambda_k, raw and applied H/F/episode
terms, coefficients, support masks, actor identity and per-step snapshots are
retained. No-context snapshots have 1536-dimensional processed hidden features.
The existing future-progress plots show the actual applied component weights.

CPU checks cover prepared configs against the EP0 main control, actual trainer
PPO loss/gradients with disabled channels and episode perturbations, removal of
context-statistics influence, hidden-mode compatibility, and legacy EP1 behavior.
No GPU rollout, optimizer step or benchmark result is claimed by preparation.


Validation completed on 2026-09-22: **29 targeted CPU tests passed** in the
WebShop Python environment, plus registry/math guards, documentation consistency,
and all three shell syntax checks. Each folder contains VALIDATION.json and the
executed test/registry logs. Training output directories remain empty.
