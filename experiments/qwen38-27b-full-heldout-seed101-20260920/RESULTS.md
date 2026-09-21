# Qwen3.8-27B — full ALFWorld and WebShop, seed101

Status: **in_progress**. Updated 2026-09-21T06:58:22.399965+00:00.

One round, seed 101. Percent units. Final cells remain pending until the entire split finishes.

| ALFWorld split | Pick | Look | Clean | Heat | Cool | Pick2 | All |
|---|---:|---:|---:|---:|---:|---:|---:|
| alfworld-seen | 85.71 | 61.54 | 66.67 | 56.25 | 72.00 | 66.67 | 70.71 |
| alfworld-unseen | pending | pending | pending | pending | pending | pending | pending |

| WebShop | Score | Succ. |
|---|---:|---:|
| webshop | pending | pending |

All is successes / tasks (micro-average). WebShop uses the original dense scorer and original option matching. All horizon failures remain in the denominator.

| Suite | Finished | Successful finished tasks | Calls | Output-capped calls | API cost USD |
|---|---:|---:|---:|---:|---:|
| alfworld-seen | 140/140 | 99/140 | 3454 | 0 | $0.000000 |
| alfworld-unseen | 35/134 | 31/35 | 807 | 0 | $0.000000 |
| webshop | 323/500 | 101/323 | 3938 | 0 | $0.000000 |

**API usage-derived charge: $0.000000.**
Local Qwen has no API fee. GPU rental/electricity cost is not priced here; server timestamps are retained.

Exact task manifests, configuration, immutable source snapshots, dependency/data hashes, prompts, native reasoning, final responses, actions and token usage are retained in this folder. See NOTES.md for reproduction.
