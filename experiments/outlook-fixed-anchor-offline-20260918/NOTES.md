# Fixed-anchor OUTLOOK: offline measurement preparation

Status: **historical offline preparation, superseded by the user-authorized training launches on 2026-09-18**. M8 ALFWorld and M9 WebShop now collect both reference feature sets and all scalar terms every step. The production future readout jointly encodes retained history and the observed continuation; the evaluator below was prepared for a future-only feature/context representation and must not be treated as the exact production estimator. Use each new run's `outputs/fixed_anchor/` artifacts and live diagnostics for that estimator.

The current and archived run outputs were inspected. The sampled OUTLOOK runs have `gdump=false`, `trainer.rollout_data_dir=null`, `trainer.validation_data_dir=null`, and `trainer.log_val_generations=0`. Their scalar CSVs contain bucket hashes, returns and readouts, but no hidden-state vectors, similarity matrices, observation/action text or tokenized prompts. Checkpoint `data.pt` is dataloader state, not saved rollout data. No reusable feature archive or Ray spill snapshot was found in the inspected locations.

The old endpoint-bootstrap advantage could be reconstructed using recorded scalar readouts. A fixed-anchor future readout requires new similarities between future contexts whose *current* observations match. Those similarities were never evaluated or saved, so the previous scalar reconstruction cannot identify this new estimator. No numerical correlation is being substituted from the old estimator or from synthetic fixtures.

| Benchmark | Frozen steps | Rows | Rows with exact cross-trajectory anchor peers | Required context features |
|---|---|---:|---:|---|
| ALFWorld | 10–14 | 29,767 | 26,889 (90.3%) | Missing |
| WebShop | 28–32 | 5,242 | 4,072 (77.7%) | Missing |

These coverage figures are measurable from the saved scalar records; they are not correlation measurements. Exact-anchor coverage matters because the strict future component should be zero where the current observation has no other-trajectory peer, instead of silently regrouping it at a future observation.

## Proposed signal

For the fixed set of other-trajectory occurrences sharing `(task, O_t)`, evaluate the production context kernel twice, once with history context and once with the next two transitions and accumulated statistics through their endpoint. Both readouts use returns measured from the corresponding anchor time, including the same local invalid-action penalty, and the same task prior and credibility rule `J/(J+2)`.

`F_fixed = B_future - B_history`

`H = Y - B_history = F_fixed + (Y - B_future)`

The proposed term reweights the part of historical credit explained by the realized continuation. It does not provide an unbiased-policy-gradient guarantee.

## Prepared evaluator

[../../scripts/analyse_fixed_anchor.py](../../scripts/analyse_fixed_anchor.py) is an isolated CPU-only tool. It calls the actual production value estimator for both readouts, keeps the anchor keys fixed, excludes the whole query trajectory, preserves the current local penalty exactly once, removes duplicate padding and restores chronological order. Task fallback is disabled for the proposed future component; an unsupported anchor gets zero added signal. Its old OUTLOOK comparator retains the production task fallback and endpoint lookup.

It reports Pearson and Spearman correlation with the original **pure edge component**, nonzero sign agreement, magnitudes, exact-anchor support, and separate nonterminal comparisons. Per-task standardized fixed-future credit is reported separately, with statistics over supported occurrences. The edge comparator uses all canonical task rows for its original normalization. This is scalar credit comparison, not measured policy-gradient similarity.

Input NPZ must contain `uid`, `traj_uid`, `turn_index`, `episode_lengths`, `anchor_obs`, `returns`, `immediate_rewards`, `episode_rewards`, `is_action_valid`, `history_hidden`, `future_hidden`, and `future_context`. Hidden vectors must come from the frozen reference model before PCA. `future_context` columns are turn, unique-observation count, revisit indicator and novelty fraction. Complete truncated terminal contexts must be explicitly encoded; missing terminal vectors must not be silently replaced by current/history vectors. A companion JSON should record the checkpoint, tasks, seed, encoder, prompts, truncation policy and rollout settings.

Usage after collecting the real inputs:

```bash
.venv/bin/python scripts/analyse_fixed_anchor.py /path/to/batch.npz \
  --output experiments/outlook-fixed-anchor-offline-20260918/results
```

[../../tests/test_fixed_anchor_analysis.py](../../tests/test_fixed_anchor_analysis.py) verifies zero signal under identical contexts, unchanged anchor support under changed future contexts, own-trajectory exclusion, shuffle/padding invariance, no future regrouping for unsupported anchors, explicit missing-feature failure, penalty accounting and serializable summaries. All eight CPU checks pass. Synthetic fixtures validate implementation only and are not benchmark evidence.

## Earlier collection proposal (superseded)

Collect one bounded inference-only batch for each benchmark from a completed checkpoint, with zero optimizer updates, preserving full trajectory order and both sets of reference features. Save inputs before computing comparisons. All eight GPUs were busy during inspection, so a new GPU collection must either share a running job's GPU or wait for an available device. The subsequent user instruction authorized pausing M6/M7 and launching M8/M9 instead; no separate inference-only collection is pending.
