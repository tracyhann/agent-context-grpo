# M11 WebShop: no shrinkage, active episode advantage, peer/representation ablations

Prepared 2026-09-22. Three independent ablations of the
[M11 H2 no-shrink active-episode control](m11-h2-noshrink-active-episode-webshop-1.5b-2gpu-20260920/NOTES.md).
All use Qwen2.5-1.5B-Instruct, 150 training iterations, seed 0, 16 tasks x eight
rollouts, two GPUs, history 2 / future 2 and a 15-turn train/eval ceiling.
**Prepared only: no runs launched, queued or assigned GPUs.**

| Variant | Change from EP1 control | Self-trajectory peers | Context summary | Weighting | Notes |
|---|---|---|---|---|---|
| Control | none | excluded | on | soft exponential | [Control](m11-h2-noshrink-active-episode-webshop-1.5b-2gpu-20260920/NOTES.md) |
| No trajectory exclusion | `ccpo_loo: 1 -> 0` | included, including query | on | soft exponential | [NOTES](m11-h2-noshrink-active-episode-noloo-webshop-1.5b-2gpu-20260922/NOTES.md) |
| No context summary | `ccpo_ctx_w: 1 -> 0` | excluded | off | soft exponential | [NOTES](m11-h2-noshrink-active-episode-noctx-webshop-1.5b-2gpu-20260922/NOTES.md) |
| Cosine weights | `ccpo_wmode: soft -> cos` | excluded | on | clipped cosine | [NOTES](m11-h2-noshrink-active-episode-cosine-webshop-1.5b-2gpu-20260922/NOTES.md) |

Each full config differs from the same EP1 control only in its named setting and
experiment identity. The implicit legacy LOO/history-weight defaults of 1 are
recorded explicitly. All three keep history/future/episode weights **1/1/1** and
original-edge weight **0**. These changes are not combined into a single arm.

## Shared method and episode fusion

For the appropriate peer pool P_t and weights w_tj, define

\[
C_t[q]=\frac{\sum_{j\in P_t}w_{tj}q_j}{\sum_{j\in P_t}w_{tj}},\quad
H_t=Y_t-C_t[Y],\quad Z_t=\gamma^{T-t}R,\quad V_t=C_t[Z],
\]
\[
F_t=\operatorname{z}_{\mathrm{task}}(V_{\min(t+2,T)}-V_t),\qquad
A_{\mathrm{CC}}=H_t+F_t,\qquad
A_{t,\ell}=M_{t,\ell}(A_{\mathrm{CC}}+A_{\mathrm{EP},t}).
\]

Every usable readout uses full strength: lambda_u=lambda_k=1. Existing exact
observation/task grouping, task fallback and unsupported-row rules are retained.
Current and future endpoints have their own observation groups. Terminal potential
is 10 for success and 0 otherwise; gamma=0.95. The history return Y includes the
existing per-turn invalid-action penalty. Potential label Z is unpenalized.

WebShop mean_norm leaves the H+F sum unchanged. F alone is standardized within
each task before fusion. The existing episode helper uses
S_t=R_episode-0.1*invalid_t and A_EP,t=S_t-mean_task(S), with moments calculated
across turn rows rather than one record per trajectory. Episode weight is **1**;
there is no normalization after adding it. The episode channel changes the PPO
gradient. Estimator features and advantage targets remain detached.

## Exact interventions

**No trajectory exclusion:** permit matching turns from the query trajectory,
including the query itself, in all history/current/future readouts. Different
observations do not become exact peers. Padding copies are still removed before
peer selection. Support counts distinct represented trajectories, now including
self. A singleton exact group has H=0 and V=its own Z; future progress and episode
advantage can remain nonzero. Self/own-trajectory weight mass is logged. This
variant removes the protection against using the query's own outcome in its
contextual baseline. Episode normalization itself is unchanged.

**No context summary:** remove only the 37-dimensional accumulated statistics
block. Frozen hidden states still encode the current prompt and its two history
turns, then undergo centering, removal of three principal directions and L2
normalization. Representation width is 1536 on this backbone, instead of 1573.
The same hidden-only representation supplies history and both potentials. Raw
summary statistics may remain logged but cannot affect weights or actor credit.

**Cosine weights:** use w_ij=max(cos(z_i,z_j),0) on the same normalized
hidden-plus-statistics representation, then normalize weights over eligible peers.
If all weights are zero, the existing nearest-peer fallback supplies one nonzero
weight. Tau is inert. This reuses the
[existing September 20 cosine EP1 variant](m11-h2-noshrink-active-episode-cosine-webshop-1.5b-2gpu-20260920/NOTES.md);
there is no second cosine estimator. The new dated preparation records the same
learning protocol against the common control with current source provenance.
Earlier experiment artifacts and EP0 variants are preserved.

## Preparation and validation

From the project root, prepare all three for a fresh date without launching:

```bash
python scripts/prepare_webshop_ep1_ablations.py --date YYYYMMDD
```

Use `--variant no-loo`, `--variant no-context-vector`, or `--variant cosine` to
select a subset; the option may be repeated. Existing folders are never overwritten.
Per-folder config.json, run.sh, PREPARED.json, control diff, exact preparation
command, source hashes and VALIDATION.json record the configuration and checks.

The new tests load actual generated environments/configs and exercise the real
trainer/PPO path: active episode-gradient contribution, H+F+EP identity, full
baseline strength, self inclusion/exclusion in all three readouts, hidden-only
statistics invariance, independent cosine readout arithmetic and tau invariance.
Existing no-LOO tests retain padding deduplication and target-dependence coverage.
The 1K WebShop catalog, original scorer, 4096/512 token limits, learning rate
1e-6, sampling temperature 1.0, separate KL coefficient 0.01 and evaluation/save
every five iterations remain matched. No GPU validation or training result is
claimed by this preparation.

Validation completed: **45 scoped CPU tests passed**, plus registry/config/cosine
arithmetic guards, documentation resolution, shell syntax and trainer overlay parity.
Per-folder VALIDATION.json records commands and copied logs. No GPU used.
