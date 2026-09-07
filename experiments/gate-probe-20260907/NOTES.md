# gate-probe-20260907

**Not a training arm.** A 2-step diagnostic (steps 101-102, warm-started from
`ccpo-long-20260906` step 100) whose only purpose is to dump the raw inputs to the
anchor-state gate so candidate gates can be compared **offline**, without spending
a training run on each one.

`test_freq=0`, `save_freq=0` -> no validation, no checkpoints. Cost ~15 min on an
otherwise idle box; total footprint 1.3 MB.

## What it produced

`outputs/grouping.jsonl` -- 6,912 occurrences with full observation text, task uid,
trajectory uid, step index, the four `derive_context()` features, and both targets
(return-to-go and nextnode successor value).

Enabled by `gdump=True` -> `ACG_CCPO_GDUMP`. Off by default: ~100x the per-step
volume of `ccpo_samples.csv`, so it is for diagnostic arms only.

## What it settled

Run `scripts/analyse_gate.py experiments/gate-probe-20260907/outputs/grouping.jsonl`.

| gate | dead | R² (return-to-go) | R² (nextnode) |
|---|---|---|---|
| `(task, obs)` — GiGPO / G²PO | 0.073 | **0.4673** | 0.7458 |
| `+ progress band` | 0.141 | 0.4398 | 0.7125 |
| visited-set signature | 0.947 | 0.5263 | 0.7658 |
| task only, no obs gate | 0.000 | 0.4248 | 0.7512 |

* **H-J refuted** — progress banding loses under both targets. Its ICC looked 4.5x
  better; ICC is a signal-*share* statistic and ignores the noise of estimating a
  baseline from a smaller bucket. Judge gates on the LOO residual R² instead.
* **H-K opened** — the anchor gate is worth **+0.043** under return-to-go but
  **−0.005** under the nextnode target we actually run. G²PO's target and GiGPO's
  gate were adopted separately and appear to cancel.
* **H-L opened** — visited-set has the best R² but 94.7% dead; precise state
  identity works and simply has no support in a 128-trajectory batch.

See `experiments/hypothesis.md` for the full entries.
