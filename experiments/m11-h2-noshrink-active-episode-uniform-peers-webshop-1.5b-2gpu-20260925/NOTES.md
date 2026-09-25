# WEBSHOP H2 no shrinkage: uniform peer weighting

**Prepared only; not launched or queued.** Qwen2.5-1.5B-Instruct, 150 training
steps, seed 0, two GPUs (`2,3` configured, not reserved).
Registry key `future-progress-h2-no-credit-shrinkage-active-episode-uniform-peers`; canonical ID `ccpo-attncred-abl-fph2-noshrink-ep-uniform-peers-ws-1.5b`.

## Change and estimator

Change only `ccpo_wmode: soft -> uniform` from `experiments/m11-h2-noshrink-active-episode-webshop-1.5b-2gpu-20260920`.
[Full config diff](config-diff-from-control.json) also records experiment identity;
legacy implicit history weight and whole-trajectory LOO are made explicit.

For each query t, retain all other-trajectory occurrences in its exact task and
observation group. If none exist, use other-trajectory occurrences from the same
task. Every eligible occurrence has weight one: C_t[q] = mean(q_j, j in P_t).
Repeated real visits contribute separately; this is not a mean of trajectory
means. Padding copies are deduplicated before estimation. Whole-query-trajectory
exclusion remains on. No peers means the existing unsupported-credit behavior.
The rule applies to history, current and future potential, including fallback.
Usable baselines have full strength, lambda_u=lambda_k=1.

H_t = Y_t - C_t[Y], Z_t = gamma^(T-t) * R_episode, V_t = C_t[Z],
F_t = z_task(V_min(t+2,T) - V_t), gamma=0.95. Each endpoint uses its own
observation group. Terminal potential is success 10 / failure 0. Y retains the
existing local invalid-action penalty; Z excludes it. H/F/EP actor weights:
**1 / 1 / 1**. Original edge weight is zero.

**A = mask * (H + F + A_EP)**. ALFWorld uses the existing per-task contextual mean/sample-std
normalization. WebShop uses no final contextual normalization and adds the
mean-centered episode advantage at unit weight. Episode moments follow the
existing turn-row weighting, including the local invalid-action penalty.

This is distinct from future-only: both H and F still affect the PPO gradient.
Future-only sets H's actor weight to zero and keeps similarity-weighted potential
readouts. Here frozen hidden/context features remain computed for diagnostics,
but under exact grouping they cannot influence uniform weights or credit.
The actor still receives history-2 prompts. This also differs from no-context
vector, which keeps hidden-state similarity weighting.

## Protocol, logs and reproduction

16 tasks x 8 rollouts, 150 steps, 15-turn cap, save/evaluate every
5 steps. Prompt/response limits and all other optimization/environment settings
match the control: ALFWorld 2048/512; WebShop 4096/512 with its 1K catalog.
History/current/future baselines, support, lambda, H/F/EP raw/applied components,
feature diagnostics and actor-sum identity remain logged. The explicit metric
`ccpo/progress_uniform_weighting=1` identifies this weighting rule.

```bash
python scripts/prepare_uniform_peer_ablations.py --date YYYYMMDD
# Later launch from the project root:
bash experiments/m11-h2-noshrink-active-episode-uniform-peers-webshop-1.5b-2gpu-20260925/run.sh
```

[Shared equations and comparison](../UNIFORM_PEER_ABLATIONS.md).
[config.json](config.json), [PREPARED.json](PREPARED.json),
[prepare-command.json](prepare-command.json), [source hashes](prepared-source-sha256.json)
and [VALIDATION.json](VALIDATION.json) record preparation and CPU validation.
No training result or GPU smoke test is claimed.
