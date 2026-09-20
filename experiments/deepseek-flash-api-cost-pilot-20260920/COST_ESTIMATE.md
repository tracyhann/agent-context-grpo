# DeepSeek Flash API: measured pilot and full-evaluation cost estimate

Actual pilot: six complete tasks per benchmark, seed 0, history 2, native thinking/high,
4,096 maximum output tokens per action. ALFWorld has a 50-action ceiling; WebShop has 15.
No local model or GPU is used. Prices below are USD; off-peak is half peak.

| Benchmark | Full tasks | Sample success | Mean turns | Mean input tokens/task | Mean output tokens/task | Pilot cost | Full off-peak | Full peak |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| alfworld | 140 | 4/6 | 28.83 | 12867 | 63161 | $0.2390 | $4.19 | $8.38 |
| webshop | 500 | 3/6 | 10.17 | 13122 | 14798 | $0.0651 | $5.42 | $10.85 |

**Combined full-evaluation estimate: $9.62 off-peak / $19.23 peak.**
Without cache savings: $9.62 / $19.23.
Planning allowance (2x no-cache estimate): $19.23 off-peak / $38.46 peak.
Actual pilot token charge at off-peak rates: $0.3040.

## Accounting

Per request: cost = (cache-hit input * 0.003 + cache-miss input * 0.15 + completion * 0.6) / 1,000,000.
Completion includes native reasoning. Peak rates double each component. ALFWorld is weighted by
the six task-type populations, since the pilot takes one task of each type; WebShop uses a random
six-goal sample and extrapolates its mean to all 500 goals.

[Official rate card](https://api-docs.deepseek.com/quick_start/pricing/).

## Completed examples

| Benchmark | Task / goal | Success | Score | Turns | Input tokens | Output tokens | Off-peak cost |
|---|---|---:|---:|---:|---:|---:|---:|
| alfworld | look_at_obj_in_light | 0 | 0.000 | 50 | 17972 | 166980 | $0.1029 |
| alfworld | pick_and_place_simple | 1 | 1.000 | 5 | 2185 | 3741 | $0.0026 |
| alfworld | pick_clean_then_place_in_recep | 1 | 1.000 | 7 | 4876 | 5267 | $0.0039 |
| alfworld | pick_cool_then_place_in_recep | 1 | 1.000 | 12 | 7329 | 24318 | $0.0157 |
| alfworld | pick_heat_then_place_in_recep | 1 | 1.000 | 49 | 25702 | 90771 | $0.0583 |
| alfworld | pick_two_obj_and_place | 0 | 0.000 | 50 | 19139 | 87888 | $0.0556 |
| webshop | 419 | 1 | 1.000 | 4 | 3840 | 3234 | $0.0025 |
| webshop | 459 | 0 | 0.000 | 15 | 24752 | 35214 | $0.0248 |
| webshop | 130 | 1 | 1.000 | 5 | 4664 | 7133 | $0.0050 |
| webshop | 431 | 1 | 1.000 | 7 | 7753 | 10336 | $0.0074 |
| webshop | 370 | 0 | 0.000 | 15 | 23317 | 15017 | $0.0125 |
| webshop | 26 | 0 | 0.000 | 15 | 14406 | 17852 | $0.0129 |

## Output-budget diagnostic

One truncated ALFWorld prompt was replayed at 8,192 tokens without stepping the environment: finish=stop, valid action tag=True, completion tokens=4638.
This additional request cost $0.00280; all pilot plus diagnostic calls cost $0.3068.
The probe shows that 4,096 tokens can cut off useful reasoning. It does not provide a full-benchmark cost estimate at 8,192 tokens; the estimates above remain conditional on the measured 4,096-token protocol.

## Generation truncation

Response token ceilings were reached on alfworld: 43/173 requests, webshop: 4/61 requests.
All of these requests and subsequent failed/horizon-limited trajectories are included in cost. The 4,096-token cap can prevent an action from being emitted; this pilot is not evidence of uncapped model accuracy. A full evaluation at 8,192 or more output tokens is a different cost condition.

## Limits

- Six episodes per benchmark: cost extrapolation, not a reliable performance estimate.
- ALFWorld valid_seen 140, WebShop test goals 0..499 on the existing 1000-product catalog.
- One seed fixes task/environment sampling. The API does not expose a deterministic generation seed.
- Completion tokens already include reasoning tokens; reasoning is not billed twice.
- Planning budget is 2x the no-cache estimate, not a statistical confidence bound.
- Future model versions, prompt/turn budgets and failures can change the cost.

Task identities, action traces, original item-option scorer state and API usage are saved under
`outputs/<benchmark>/episodes/` and `outputs/<benchmark>/api_calls/`. No API secret is written to these files.
The WebShop setup-only serialization failure was preserved separately and made no API calls.
