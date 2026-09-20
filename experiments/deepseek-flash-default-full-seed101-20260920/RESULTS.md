# DeepSeek Flash official-default full evaluation — seed 101

Status: **completed**. Updated 2026-09-20T13:58:42.868836+00:00.

Model `deepseek-flash`; native thinking/high; 65,536 max output tokens per action. 
History 2; ALFWorld 50 actions; WebShop 15 actions. Environment/task seed 101. 
API generation is not deterministically seeded. Raw prompts, responses, task IDs and usage are retained.

## Requested results

Percent units (0–100). Final scores appear only after every requested episode completes.

| Model | Pick | Look | Clean | Heat | Cool | Pick2 | All | WebShop Score | WebShop Succ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DeepSeek Flash | 88.57 | 84.62 | 66.67 | 62.50 | 76.00 | 66.67 | 75.00 | 27.87 | 22.00 |

ALFWorld All is total successes / 140, not the unweighted mean of task-type percentages. 
WebShop Score is 100 × mean original environment dense reward; Succ is 100 × fraction with reward exactly 1. 
Every action-limit failure remains in the denominator; unpurchased WebShop tasks receive zero.

## Progress and measured cost

| Benchmark | Finished | Successes among finished | API calls | Truncated calls | Cost USD |
|---|---:|---:|---:|---:|---:|
| alfworld | 140/140 | 105/140 | 2989 | 1 | $3.704741 |
| webshop | 500/500 | 110/500 | 6321 | 0 | $6.236102 |

**Total API usage-derived charge: $9.940843.**

Only this full seed-101 evaluation is included; earlier pilots and diagnostic replays are excluded. 
Costs use cache-hit input, cache-miss input and completion tokens; reasoning is already included in completion. 
Rates per million off-peak USD: 0.003 / 0.15 / 0.6. Peak doubles each component. 
All timestamps, pricing-window counts and backend fingerprints are in RESULTS.json. 
This is calculated from returned usage, rather than an account invoice.

API attempts without returned usage: 4. If nonzero, their possible charges are not included.

Missing-charge allowance: up to $0.157689 if each interrupted request used its full output ceiling. This uses each matching prompt’s input count at cache-miss price and the configured maximum output; it is an allowance, not a measured charge.

## Split and reproducibility

- ALFWorld: all 140 solvable `valid_seen` tasks, held out from training in seen environments. The distinct `valid_unseen` split has 134 tasks and is not evaluated here.
- WebShop: all 500 test goal positions (0–499) under seed-101 goal construction, on the existing 1,000-product catalog. Goal metadata and prices are saved; positions 500 onward are excluded.
- Existing original WebShop option matching/scoring is used; no post-hoc item-option score correction is applied to the headline table.
- Source snapshots, exact task plans, dependency versions, dataset hashes, and commands are in this experiment folder. Model alias/backend updates can prevent bit-identical API reproduction.
- The native reasoning field is logged separately and cannot supply executable actions. Existing benchmark prompts and action parsers otherwise apply.
- WebShop pre-action purchase state is retained because the environment auto-resets on terminal actions.

[Official API defaults](https://api-docs.deepseek.com/api/create-chat-completion/), 
[official pricing](https://api-docs.deepseek.com/quick_start/pricing/).
