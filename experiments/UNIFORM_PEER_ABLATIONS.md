# Uniform peer weighting: ALFWorld EP0 and WebShop EP1

Prepared on 2026-09-25; neither arm has been launched or queued.
Both use Qwen2.5-1.5B-Instruct, 150 training steps, two GPUs, history 2 / future 2,
full-strength usable baselines (lambda_u=lambda_k=1), and zero original-edge weight.

| Arm | H / F / EP weights | Turns | Prompt / response tokens | Experiment |
|---|---|---|---|---|
| M10 ALFWorld, uniform peers | 1 / 1 / 0 | 50 | 2048 / 512 | [Config and notes](m10-h2-noshrink-uniform-peers-alfworld-1.5b-2gpu-20260925/NOTES.md) |
| M11 WebShop, uniform peers + episode | 1 / 1 / 1 | 15 | 4096 / 512 | [Config and notes](m11-h2-noshrink-active-episode-uniform-peers-webshop-1.5b-2gpu-20260925/NOTES.md) |

## What changes

Each experiment changes only `ccpo_wmode: soft -> uniform` relative to its matched
no-shrink control, plus experiment identity. The old implicit LOO and history
coefficient defaults are recorded explicitly. This is an estimator ablation;
it does not change the prompt window, rollout protocol, targets or peer membership.

For occurrence i, let P_i contain occurrences j with the same task and exact
observation, excluding the **entire query trajectory**. If this pool is empty,
use other-trajectory occurrences of the same task. The existing unsupported-row
handling applies if that pool is also empty. Padding copies are deduplicated
before either readout.

The main method uses contextual similarity weights; this variant uses:

\[
  w_{ij}=1,\qquad
  C_i^{\mathrm{avg}}[q]=\frac{1}{|P_i|}\sum_{j\in P_i}q_j.
\]

Apply this rule to **history, current potential, future potential and task-level
fallback**. There is no radius or nearest-neighbor selection. Distinct real turns
are separate occurrences, including repeated visits from a peer trajectory.
For example, peer A's labels [2, 4] and peer B's label [9] produce a baseline of
5, not the equal-trajectory mean 6. J still counts distinct peer trajectories;
n_eff is still computed from their total occurrence masses and can be below J.
Neither diagnostic introduces shrinkage in these arms.

Frozen hidden and accumulated-context representations are still computed and
logged to preserve the matched execution/diagnostic path. With these exact
observation groups, they do not affect uniform weights or the resulting credit.
Feature finiteness and duplicate-row alignment checks remain active. Prompt
history still affects the actor's input.

## Credit and fusion

Let Y_i be the existing penalized return target, and let
Z_i=gamma^(T-t) R_episode be the existing unpenalized potential target, gamma=0.95.
Then:

\[
 H_t=Y_t-C_t^{\mathrm{avg}}[Y],\qquad
 V_t=C_t^{\mathrm{avg}}[Z],\qquad
 F_t=\mathcal N_{\mathrm{task}}
       [V_{\min(t+2,T)}-V_t].
\]

Each nonterminal endpoint uses its own observation-matched peer group with
whole-trajectory exclusion. Terminal potential remains success 10 / failure 0.
Future differences are standardized with per-task sample standard deviation
and epsilon 1e-6. Existing live/support masks remain in force.

ALFWorld:

\[
 A_t=\mathcal N_{\mathrm{task}}[H_t+F_t].
\]

WebShop:

\[
 A_t=H_t+F_t+A_{\mathrm{EP},t}.
\]

WebShop retains its existing mean_norm protocol: there is no final H+F
standardization, and episode advantage is added afterward with unit weight.
Its episode channel is the task-mean-centered row score, including the existing
local invalid-action penalty. Moments use turn rows, not equally weighted
trajectories. All policy advantages are detached and response-masked.
ALFWorld still uses episode rewards as return labels; its **episode advantage
actor coefficient is zero**. Episode diagnostics cannot affect its gradient.

## Why this differs from future-only

| Ablation | History credit | Future credit | Readout weighting | Question tested |
|---|---|---|---|---|
| Main method | on | on | contextual similarity | Joint method |
| Uniform peers (these arms) | on | on | arithmetic occurrence mean | Does contextual peer weighting help? |
| Future-only | off | on | contextual similarity | Does history credit help alongside future progress? |
| No context-statistics vector | on | on | hidden-state similarity | Does the explicit context summary help? |

Even under uniform weighting, H is generally nonzero, and potential differences
can remain nonzero because peer groups and return labels change across endpoints.
This variant also retains LOO and H2 progress, so it is not the self-inclusive
one-step M5 edge reference.

## Reproduction and validation

```bash
# From the project root; prepares both arms without launching.
python scripts/prepare_uniform_peer_ablations.py --date YYYYMMDD

# CPU estimator/config/actual-PPO checks.
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv-webshop/bin/python -m unittest discover -s tests -p test_uniform_peer_ablations.py -v
```

Registry keys:

- `future-progress-h2-no-credit-shrinkage-uniform-peers` (ALFWorld EP0).
- `future-progress-h2-no-credit-shrinkage-active-episode-uniform-peers` (WebShop EP1).

The launcher exports `ACG_CCPO_WMODE=uniform` to the shared core estimator.
All existing history/current/future support, baselines, raw/applied components
and snapshots remain available. `ccpo/progress_uniform_weighting=1` explicitly
identifies the mode. Kernel and uniform baseline diagnostics should coincide
up to floating-point arithmetic; their difference measures no similarity effect.

Each experiment includes its full config/Hydra command, matched-control delta,
source hashes, generation command, method notes and CPU validation record.
Tests cover arithmetic means with unequal repeat counts, exact/fallback pools,
trajectory exclusion, padding invariance, unsupported rows, feature/tau
invariance (with a soft-weighting positive control), distinct future-only credit,
both benchmarks' real trainer/PPO paths, and ALFWorld episode-gradient isolation.
No GPU smoke test or training result is implied.
