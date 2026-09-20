# Token cap and pilot task completion

This is a post-hoc analysis of the existing six ALFWorld and six WebShop episodes. No new API calls were made. A capped request is identified by `finish_reason == "length"`; the per-request output budget was 4,096 tokens.

| Benchmark | Tasks with any capped request: success | Tasks with no capped request: success |
|---|---:|---:|
| ALFWorld | 2/4 (50%) | 2/2 (100%) |
| WebShop | 0/2 (0%) | 3/4 (75%) |

| Benchmark | Task | Capped requests | Total actions | Final success | Final score |
|---|---|---:|---:|---|---:|
| alfworld | look_at_obj_in_light | 29 | 50 | False | 0.0 |
| alfworld | pick_cool_then_place_in_recep | 3 | 12 | True | 1.0 |
| alfworld | pick_heat_then_place_in_recep | 5 | 49 | True | 1.0 |
| alfworld | pick_two_obj_and_place | 6 | 50 | False | 0.0 |
| webshop | 459 | 3 | 15 | False | 0.0 |
| webshop | 370 | 1 | 15 | False | 0.0 |

All 47 capped responses lacked a complete valid action tag. The legacy parser nevertheless recovered two executable ALFWorld actions; the other 41 ALFWorld capped steps returned `Nothing happens.`. All four capped WebShop steps left the observation unchanged. These steps consumed the 50-action ALFWorld / 15-action WebShop budget.

The cooling task recovered after three consecutive capped requests and succeeded in 12 actions; heating succeeded in 49 actions despite five capped requests. Both failed capped WebShop episodes reached 15 actions without purchasing (no purchase confirmation in the trajectory, final score=0).

This small observational comparison cannot establish that truncation caused each task failure: harder tasks may both induce longer reasoning and be harder to solve. One WebShop task (goal 26) failed without any capped requests. The separate 8,192-token diagnostic produced a valid action for one previously truncated prompt but did not rerun or complete its task.
