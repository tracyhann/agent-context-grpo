# Qwen3.8-27B — full ALFWorld and WebShop, seed101

Status: **completed**. Updated 2026-09-21T15:49:33.633435+00:00.

One round, seed 101. Percent units. Final cells remain pending until the entire split finishes.

| ALFWorld split | Pick | Look | Clean | Heat | Cool | Pick2 | All |
|---|---:|---:|---:|---:|---:|---:|---:|
| alfworld-seen | 85.71 | 61.54 | 66.67 | 56.25 | 72.00 | 66.67 | 70.71 |
| alfworld-unseen | 95.83 | 66.67 | 61.29 | 60.87 | 80.95 | 82.35 | 73.88 |

| WebShop | Score | Succ. |
|---|---:|---:|
| webshop | 39.48 | 29.00 |

All is successes / tasks (micro-average). WebShop uses the original dense scorer and original option matching. All horizon failures remain in the denominator.

| Suite | Finished | Successful finished tasks | Calls | Output-capped calls | API cost USD |
|---|---:|---:|---:|---:|---:|
| alfworld-seen | 140/140 | 99/140 | 3454 | 0 | $0.000000 |
| alfworld-unseen | 134/134 | 99/134 | 3358 | 0 | $0.000000 |
| webshop | 500/500 | 145/500 | 6000 | 0 | $0.000000 |

**API usage-derived charge: $0.000000.**
Local Qwen has no API fee. GPU rental/electricity cost is not priced here; server timestamps are retained.

Exact task manifests, configuration, immutable source snapshots, dependency/data hashes, prompts, native reasoning, final responses, actions and token usage are retained in this folder. See NOTES.md for reproduction.
