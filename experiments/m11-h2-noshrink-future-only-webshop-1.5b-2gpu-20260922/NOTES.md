# M11 WebShop H2 no shrinkage, EP0: no-history

**Prepared only; not launched or queued.** Qwen2.5-1.5B-Instruct, 150 steps,
seed 0, two GPUs (`2,3` configured, not reserved), 15 turns.
Registry key `future-progress-h2-no-credit-shrinkage-future-only`; canonical ID `ccpo-attncred-abl-fph2-noshrink-future-only-ws-1.5b`.

## Method

History/future/episode coefficients: **0 / 1 / 0**.
Context-statistics weight: **1**. Original-edge weight 0.
Prompt history and diagnostic future horizon remain 2/2; LOO excludes the whole
query trajectory. All usable contextual readouts have lambda_u=lambda_k=1.

Only future progress F contributes to the actor. Historical residual H is diagnostic at weight 0. History-2 prompts still supply the representations used by the potential estimator.

Let C_t[q] be the kernel-weighted contextual readout of peers in the same task
and observation group, excluding the query trajectory. Retain existing task
fallback, unsupported-row handling and terminal success/failure potential 10/0.
H_t=Y_t-C_t[Y], Z_t=gamma^(T-t)*R_episode, V_t=C_t[Z],
F_t=z_task(V_min(t+2,T)-V_t), gamma=0.95. Y retains the historical invalid-action
penalty; Z is the unpenalized potential label. The WebShop actor advantage is
**A=mask*(F)**. F retains per-task standardization; the WebShop mean_norm
path does not add a combined standard-deviation normalization.

The policy and frozen reference keep two prompt-history turns. No-context means
no accumulated statistics block in similarity features; it does not erase prompt
history or bypass hidden-state processing, and it does not switch to cosine.
Episode rewards still label the returns. Raw episode advantage remains logged
but its applied contribution and original-edge contribution are exactly zero.

## Matched control and diagnostics

[Config diff](config-diff-from-control.json) changes one scientific setting from
`experiments/m11-h2-no-credit-shrinkage-webshop-1.5b-2gpu-20260919`, plus experiment identity. Both control and variant use EP0;
the earlier WebShop EP1 component arms are separate comparisons. Implicit legacy
history-weight/LOO defaults of 1 are made explicit in this preparation.

16 tasks x 8 rollouts; 1K WebShop catalog/scorer; evaluation and checkpoints every
5 steps. Frozen features, penalties, terminal handling and the standard main
protocol are retained. History/current/future raw readouts, J/n_eff/lambda_k,
actual coefficients, raw and applied H/F/episode terms, actor identity checks and
per-step snapshots remain enabled. Disabled channels have exactly zero applied
credit. [Shared equations and comparison table](../WEBSHOP_EP0_ABLATIONS.md).

## Reproduction

Prepare for a fresh date without training:

```bash
python scripts/prepare_webshop_ep0_ablations.py --variant no-history --date YYYYMMDD
```

Prepared launch command, to be scheduled separately:

```bash
bash experiments/m11-h2-noshrink-future-only-webshop-1.5b-2gpu-20260922/run.sh
```

[config.json](config.json), [prepare-command.json](prepare-command.json) and
[prepared-source-sha256.json](prepared-source-sha256.json) record the resolved
recipe and sources. [VALIDATION.json](VALIDATION.json) records CPU-only checks;
no GPU execution or benchmark training result is implied.
