# DeepSeek Flash — complete seed101 evaluation

One round: ALFWorld140 seen +134 unseen, WebShop500. Thinking high, output65536, history2, ALF50 / WebShop15 actions. Percent units.

| ALFWorld split | Pick | Look | Clean | Heat | Cool | Pick2 | All |
|---|---:|---:|---:|---:|---:|---:|---:|
| Seen (140) | 88.57 | 84.62 | 66.67 | 62.50 | 76.00 | 66.67 | 75.00 |
| Unseen (134) | 95.83 | 83.33 | 80.65 | 52.17 | 95.24 | 52.94 | 77.61 |

| WebShop | Score | Succ. |
|---|---:|---:|
| 500 test goals | 27.87 | 22.00 |

ALFWorld successes: seen 105/140; unseen 104/134. WebShop successes: 110/500.

All uses the micro-average. All horizon failures count. WebShop uses the original 1K-product-catalog scorer, without post-hoc item-option correction.

| Evaluation | Received-usage cost USD |
|---|---:|
| ALFWorld seen140 | $3.704741 |
| ALFWorld unseen134 (new) | $3.418961 |
| WebShop500 | $6.236102 |
| Total | **$13.359805** |

4 interrupted attempts lacked returned usage. Their charges are excluded above; conservative additional allowance **up to $0.157689**. This is an allowance assuming full output caps, not measured usage or an invoice.

[Unseen protocol and reproduction](NOTES.md); [seen/WebShop protocol and reproduction](../deepseek-flash-default-full-seed101-20260920/NOTES.md). Exact tasks, data/package/source snapshots and raw prompt/action/token ledgers are retained. API generation is not seedable; seed101 fixes the environment and task order. Model aliases can change over time.
