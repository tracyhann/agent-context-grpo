# WebShop failure analysis — 2026-09-18

M11 is close to M5 at the end, but the policy has learned a brittle shopping routine. It usually buys the first search result without checking alternatives or verifying every requested option. There is also a confirmed option-scoring bug that rejects some fully correct purchases.

## Evidence and protocol

- Original training validation at step 150: M11 **179/256 = 69.92%** success, **84.33/100** task score; M5 **183/256 = 71.48%**, **84.66/100**. M11's final success gap is 1.56 percentage points. Mean validation success over steps 100–150 is 64.56% versus 67.40%.
- A new, evaluation-only replay of M11's step-150 FSDP checkpoint captured **256 complete episodes**, using the existing validation pipeline, 128 environments per batch, environment seed 1000, temperature 0.4, 15-turn limit, two-turn prompt history, and the 1,000-product synthetic-goal catalogue. No optimizer updates occurred.
- Replay result: **179/256 success**, **85.56/100 task score**, mean length **5.33**. This is a fresh validation draw, not a reconstruction of the original final validation episodes. Its success count happens to be identical.
- A paired M5 replay did **not** complete. Behavioral findings below are about M11; the comparison with M5 uses saved training metrics. They do not prove why one algorithm beats the other.

## What actually failed

Every episode reached a purchase. The 77 failures divide as follows:

| Primary observed failure | Episodes | Share of failures |
|---|---:|---:|
| Bought a different product from the goal's target and failed its constraints | 61 | 79.2% |
| Bought the target product but missed or misselected options | 9 | 11.7% |
| Bought the exact target and exact options, rejected by the option scorer | 7 | 9.1% |
| No purchase / ran out of turns | 0 | 0.0% |

Buying a different ASIN is not automatically a failure in WebShop; alternatives can satisfy the constraints. The first row contains purchases that actually scored below 1.0.

The strongest behavioral evidence:

- **254/256** episodes selected the first search result; the remaining two selected the second.
- **256/256** opened only one product. There was no comparison-shopping before purchase.
- **0/77 failures** opened a description, features, or attributes page.
- In **50/61 wrong-product failures**, the target ASIN was already on a search-results page the agent saw. These cannot be explained solely by search failing to retrieve the target.
- **35/77 failures** included an inadmissible action. Across all actions, format validity was **96.70%**, but actual admissibility was **94.13%**. The existing plotted `valid_action` metric only checks response formatting; it does not verify that a requested click exists.
- Under the original scorer, overlapping failure components were: options **59**, attributes **55**, product type **18**, price **7**. These overlap and should not be added.
- Only **27/77** failures scored at least 0.8. The remaining failures are not all small near misses.

## Concrete cases

Episode identifiers below are `(validation batch, environment slot)` in `failure-analysis-20260918/m11/episodes.jsonl`.

1. **Wrong product and invented option — (0, 7).** Requested a relaxed-fit navy-blue polo, 3XL. It bought `B09QQP3356` instead of target `B01MG1LTMS`, selected 3XL, attempted unavailable `click[navy blue]`, and bought anyway. Score **0.04**. Its response claimed the color was available despite the legal-action list disagreeing.
2. **Obvious category and price mismatch — (0, 21).** Requested women's open-toe pumps, size 10.5, below $60. It selected men's mesh sports shoes (`B09QXF3V3X`) and immediately bought, without selecting a size. Score **0.0625**. The response repeated the goal and said it needed to filter results, while the action was `buy now`.
3. **Correct item, missing option — (0, 35).** Found target kitchen mat `B09CQ45ZRB` and selected the requested color, but omitted the explicitly requested size `19.7x31.5in+19.7x47.2in` before buying. Score **0.8**; unchanged by the scorer correction.
4. **Repetitive response consumes an action — (0, 20).** Found the correct rug and chose size and shape, then generated a long repetitive response without a complete action. The projection used its last 20 characters as an invalid action; it then bought without selecting taupe. Score **0.8333**.
5. **Correct behavior falsely marked unsuccessful — (0, 66).** Bought target loafers `B082MT9162`, selected size `7` and exact requested color `2005brown`, and checked out. Original score **0.8**; corrected option matching gives **1.0** with identical actions.

The recurring reasoning text often restates the request or talks about filtering search results after the agent is already on a product page. These traces support a diagnosis of weak observation-grounded verification and premature checkout.

## Confirmed scorer bug

In `verl-agent/agent_system/environments/env_package/webshop/webshop/web_agent_site/engine/goal.py`, `get_reward` passes `goal['goal_options'].items()` to `get_option_reward`, while purchased options are passed as values. `normalize_color` therefore sees a tuple on the goal side and a string on the purchase side.

For example, purchased `2005brown` normalizes to `brown`; goal `('color', '2005brown')` does not normalize in the same way. The exact requested selection can fail fuzzy matching.

A CPU audit found **458/6,910 synthetic goals (6.63%)** whose exact requested options do not get full option credit. Passing goal **values** instead of dictionary **items** fixes all 458 exact-option self-matches in this audit. This is not a proof that all these goals are impossible under every alternative purchase.

On the replay's requested product/option combinations, **15/256** have this exact-option issue. M11 made fully correct purchases on **7** of those tasks; on the other 8 it also made policy mistakes.

Offline rescoring the actual captured purchases, changing only dictionary-item versus value handling:

| M11 replay scoring | Success | Task score /100 |
|---|---:|---:|
| Original scorer | 179/256 = 69.92% | 85.56 |
| Corrected option matching, identical actions | 186/256 = 72.66% | 86.45 |

This correction has **not** been applied to the shared runtime or training. Both M5 and M11 used the original scorer, so it cannot by itself explain the gap between them. Do not compare corrected M11 numbers against uncorrected historical M5 numbers as an algorithm improvement.

The oracle audit reproduces product/option identities, but independently regenerated price caps can differ from the live environment's caps. The actual-purchase rescoring uses the goal and price captured from the live episode and verifies that the original scorer reproduces its original task score exactly.

## Why the training signal may not repair this routine

Both M5 and M11 train on a binary purchase reward: 10 only for score exactly 1, otherwise 0. Partial task scores are diagnostic; `ACG_CCPO_TARGET=return` does not train on them. Thus a nearly correct purchase and an unrelated purchase share the same terminal reward, and the scorer bug also poisons some otherwise correct success labels.

Measured from all 150 saved M11 training snapshots:

- In steps 1–30, **257/480 task groups (53.54%)** had no successful rollout among their eight attempts. The future-value labels provide no positive success information for those groups.
- There were **247** task groups with exactly one successful trajectory across training. For that sole success, **all 1,932 nonterminal rows** had zero contextual value and zero raw progress: leave-one-trajectory-out excludes its own successful labels, leaving only failed peers. Task standardization made the future term negative on all those rows. This does **not** mean the total policy gradient was negative: history still supplies credit.
- In steps 100–150, mean nonterminal correlation between contextual future progress and the M5 edge diagnostic was only **0.155**. Mean all-row correlation was **0.440**; approximately **48.2%** of absolute standardized future credit was on terminal rows. These are averages of per-batch diagnostics.
- Late mean token entropy was **0.306** for M11 versus **0.607** for M5. Combined with the observed first-result routine, reduced behavioral diversity is a plausible contributor. Entropy alone does not establish causality.

The evidence supports sparse and sometimes incorrect supervision, plus weak verification at product selection and checkout. It does not isolate a causal effect of the future term: that requires a controlled intervention or completed paired behavioral comparison.

## Priorities before another method experiment

1. Correct dictionary-option matching with exact-option regression cases, then rescore **both** frozen checkpoints under the same scorer and task draw. Preserve original benchmark results separately.
2. Log executed actions, actual admissibility, selected options, and reward components in validation. Formatting validity alone concealed real action failures.
3. Target the observed bottleneck: distinguish candidate products and verify all requested constraints before purchase. A controlled partial-score/constraint-credit variant is motivated by these traces; do not claim it works before testing.
4. Test future-term changes only after fixing the reward defect. The current future estimator still inherits binary labels and loses the lone successful trajectory from its peer estimate.

## Reproduction and operational record

The compact results are versioned in `failure-analysis-20260918/summary.json`. Raw traces and per-purchase audit dumps remain local. Full local artifacts are in `failure-analysis-20260918/`: `m11/episodes.jsonl`, `m11/validation_metrics.json`, `trace_analysis.json`, `behavior_diagnostics.json`, `credit_diagnostics.json`, `oracle_targets.json`, and `option_rescoring.json`.

Scripts: `scripts/trace_webshop_eval.py`, `scripts/analyse_webshop_traces.py`, `scripts/analyse_webshop_credit_failures.py`, and local `audit_oracle.py` / `rescore_options.py`.

The original diagnostic wrapper unnecessarily instantiated training environments. M11 evaluation completed, but M5 startup exhausted container thread capacity and stalled M10's data loader after step 109. The diagnostic runtime was terminated. M10 was resumed from saved checkpoint 105, with uncheckpointed 106–109 metrics/diagnostics archived and both environment streams aligned. Recovery provenance is `.local/m10-recovery-20260918/recovery.json`. GPU holders were retained. The evaluation wrapper has since been changed to avoid the 128 unused training environments and to check thread headroom; that revised wrapper has only been syntax-checked, not rerun. No further diagnostic GPU run was launched during recovery.
