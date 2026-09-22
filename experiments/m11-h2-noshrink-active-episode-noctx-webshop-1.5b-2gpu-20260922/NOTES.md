# M11 WebShop H2 no shrinkage, active episode: no-context-vector

**Prepared only; not launched or queued.** Qwen2.5-1.5B-Instruct, 150 iterations,
seed 0, 16 tasks x 8 rollouts, two GPUs (`2,3` configured),
15-turn train/eval ceiling. Registry key `future-progress-h2-no-credit-shrinkage-active-episode-no-context-vector`; ID `ccpo-attncred-abl-fph2-noshrink-ep-noctx-ws-1.5b`.

## One-setting comparison

Control: `experiments/m11-h2-noshrink-active-episode-webshop-1.5b-2gpu-20260920`. [Full config diff](config-diff-from-control.json)
changes only the requested setting and experiment identity. Legacy implicit LOO
and history-weight defaults are explicitly resolved to 1. These are three
independent ablations, not one arm combining all three changes.

Set ccpo_ctx_w=0: remove the 37-dimensional accumulated-statistics summary from history and both potential representations. The frozen hidden representation remains centered, stripped of its top three principal directions and L2-normalized (1536 dimensions). Keep the two-turn prompt history, LOO and soft exponential weights.

H/F/episode weights are **1/1/1**; original-edge weight is 0. Both history and
future windows are 2. Every usable baseline uses full strength, lambda_u=lambda_k=1;
kappa stays recorded but has no shrinkage effect. Existing fallback/unsupported
handling remains, except self can supply an exact baseline in the no-LOO arm.

## Credit and episode fusion

For target q in {Y,Z}, C_t[q] is the weighted contextual readout under this arm's
peer and feature rules. H_t=Y_t-C_t[Y], Z_t=gamma^(T-t)*R_episode, V_t=C_t[Z],
F_t=z_task(V_min(t+2,T)-V_t), gamma=0.95. Each endpoint uses its own observation
group. Terminal V is 10 for success and 0 otherwise. Y includes the existing
per-turn -0.1 invalid-action penalty; Z excludes it.

WebShop retains mean_norm: **A=mask*(H+F+E)**, where E=S_t-mean_task(S),
S_t=R_episode-0.1*invalid_t. Episode moments use turn rows, including their existing
length weighting, not one equally weighted record per trajectory. Future progress
is standardized per task; there is no final H+F normalization or renormalization
after adding E. Episode credit has unit weight and affects the actual PPO gradient.
Frozen features/credit targets remain detached.

## Protocol and logging

The 1K WebShop catalog, original item-option scorer, binary 10/0 rewards,
4096/512 prompt/response token limits, learning rate 1e-6, temperature 1.0 and
separate KL coefficient 0.01 remain matched. Save/evaluate every 5 steps.
History/current/future features, baselines, J/n_eff/lambda_k, self/own-trajectory
mass, raw/applied H/F/E and the complete actor identity remain logged. Removing
context summary may leave raw summary statistics in diagnostics; they do not
enter the similarity representation. No original edge credit is added.

## Reproduction and evidence

Prepare a fresh dated set (or select one with --variant):

```bash
python scripts/prepare_webshop_ep1_ablations.py --date YYYYMMDD
```

For a later authorized launch:

```bash
bash experiments/m11-h2-noshrink-active-episode-noctx-webshop-1.5b-2gpu-20260922/run.sh
```

[config.json](config.json), [PREPARED.json](PREPARED.json),
[prepare-command.json](prepare-command.json), [source hashes](prepared-source-sha256.json)
and [VALIDATION.json](VALIDATION.json) record preparation and CPU checks.
[Shared comparison](../WEBSHOP_EP1_ABLATIONS.md) describes all three variants.
The cosine registry/estimator is the same as the September 20 cosine EP1 arm;
this dated set records current source/config provenance. Earlier artifacts are
preserved. No GPU smoke test or training result is claimed.

CPU validation completed: **45 tests passed**, plus registry arithmetic/config guards,
documentation resolution, shell syntax and overlay parity. See VALIDATION.json.
