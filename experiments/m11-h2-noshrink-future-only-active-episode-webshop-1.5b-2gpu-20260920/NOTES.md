# M11 H2 no shrinkage: future-only

**Prepared only; not launched or queued.** Qwen2.5-1.5B-Instruct, 150 steps, seed 0,
two GPUs (`2,3` configured, not reserved). Registry key `future-progress-h2-no-credit-shrinkage-future-only-active-episode`;
canonical ID `ccpo-attncred-abl-fph2-noshrink-future-only-ep-ws-1.5b`.

## Method

History / future / episode actor weights: **0 / 1 / 1**.
Original edge weight 0. All usable context baselines have lambda_k=1; task fallback
and unsupported-row behavior remain unchanged. Context-statistics vectors,
frozen reference features, soft similarity and history-2 prompts are retained.
H2 means the future endpoint is min(t+2,T); history length remains 2 independently.

Let H_t=Y_t-C_t[Y], V_t=C_t[Z], and F_t=z_task(V_min(t+2,T)-V_t).
Y includes the existing invalid-action penalty; Z is the unpenalized discounted
return target. Current/future potentials use their own observation groups with
whole-trajectory exclusion. Terminal potentials are success 10 / failure 0.
The masked policy advantage is **A = mask * (E + F)**. N_CC is the existing
ALFWorld per-task mean/sample-std normalization; WebShop has no additional
contextual normalization. Future F is task-standardized on both benchmarks.
E is the existing trainer episode channel, added after contextual normalization.
Episode rewards still provide return labels even when episode actor weight is 0.

Only enabled contextual channels contribute their support mask to normalization.
The disabled channel remains calculated and logged as a diagnostic; its weighted
and applied advantages are identically zero and cannot affect the PPO gradient.
Future-only removes H from credit, not the historical information used by V.
History-only keeps H2 future diagnostics at weight 0, with no future credit.

Full equations and interpretation: [HISTORY_FUTURE_ABLATIONS.md](../HISTORY_FUTURE_ABLATIONS.md).
Control: `experiments/m11-h2-noshrink-active-episode-webshop-1.5b-2gpu-20260920`. [Config diff](config-diff-from-control.json) changes only
one component coefficient and experiment identity (the old implicit history
weight 1 is made explicit). History/current/future raw and applied diagnostics,
actual weights, actor-sum identity and per-step snapshots remain enabled.

## Protocol and reproduction

16 tasks x 8 rollouts, 150 steps, turn ceiling 15, checkpoint/eval every
5 steps. Other learning, rollout, data and evaluation settings match the control.
WebShop retains its standard 15-turn protocol and 1K catalog.

Prepare these four arms for a new date without starting training:

```bash
python scripts/prepare_component_ablations.py --date YYYYMMDD
```

For a later authorized launch from the project root:

```bash
bash experiments/m11-h2-noshrink-future-only-active-episode-webshop-1.5b-2gpu-20260920/run.sh
```

[config.json](config.json) records the full config, environment and Hydra command;
[prepare-command.json](prepare-command.json) and [prepared-source-sha256.json](prepared-source-sha256.json)
record generation and source provenance. CPU validation is recorded separately
in [VALIDATION.json](VALIDATION.json); no GPU run is implied.
