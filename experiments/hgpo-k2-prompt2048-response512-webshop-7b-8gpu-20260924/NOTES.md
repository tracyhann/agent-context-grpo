# HGPO K=2: webshop, 7b

**Prepared only; not launched or queued.** Seed 0, 16 tasks x 8 trajectories,
150 training iterations. Training and validation use at most 15 turns.
Prompt cap **2048 tokens**, response cap **512 tokens per turn**.
These are separate limits; neither is the per-GPU dynamic minibatch token budget.

The official HGPO recipe at `20bd331bdbc9026a5668e11362178e10ab7400c8` supplies the estimator, trainer and
environment manager. Entry point: `recipe.hgpo.main_hgpo`; estimator:
`recipe.hgpo.core_hgpo.compute_hgpo_outcome_advantage`. The core is unchanged.
Exact observation-history groups, length weighting with alpha=1, and
`base_group=False` retain the official method. No separate episode advantage
is added. Grouping includes self and repeated visits, as upstream specifies.
Advantage mode is `mean_norm`. The shared collector and PPO workers use
this repository's benchmark/runtime patches; source hashes are recorded.

`trainer.total_training_steps=150` explicitly caps iterations independently of
parquet length. `trainer.total_epochs=150` supplies enough epochs. Fresh start,
no early stopping, evaluate/save every 5 steps, keep two actor checkpoints.
ALFWorld uses the seen validation split; WebShop uses the 1K catalog and existing
scorer. Validation processes the whole configured parquet in chunks of 64.
Learning rate 1e-6, gamma=.95, terminal reward 10/0, local invalid penalty .1,
training temperature 1.0, validation temperature .4, separate KL loss .01.

Original HGPO budgets were prompt 4096, response 512, 160 epochs, and
WebShop 30 turns. The prepared delta and actual runtime settings are in
[config.json](config.json). Historical WebShop CCPO/GiGPO configurations used
4096 input tokens, so this 2048-input variant needs a matching control for a
strict token-budget comparison. HGPO retains upstream left truncation.

[Method, comparisons and preparation commands](../HGPO_BASELINE.md).
For a later launch on allocated GPUs:

```bash
bash experiments/hgpo-k2-prompt2048-response512-webshop-7b-8gpu-20260924/run.sh
```

`PREPARED.json` records preparation, not runtime success. `VALIDATION.json`
records CPU checks separately; no GPU memory-fit or training result is claimed.

## Backbone comparison

Matched 1.5B control: [hgpo-k2-prompt2048-response512-webshop-1.5b-4gpu-20260924](../hgpo-k2-prompt2048-response512-webshop-1.5b-4gpu-20260924/NOTES.md).
Only model, model path, configured GPUs (4 to 8), rollout TP (1 to 2), and
experiment identity differ; see [config-diff-from-1.5b.json](config-diff-from-1.5b.json).
No additional trajectories, training steps or tokens are introduced by scaling.
