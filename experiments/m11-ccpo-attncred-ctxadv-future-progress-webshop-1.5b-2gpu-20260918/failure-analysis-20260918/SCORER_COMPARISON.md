# Item-option scorer comparison, checked 2026-09-19

There is an independently documented precedent for the same correction: **FedAgent** changes dictionary goal options from `.items()` to `.values()`. This correction is not universal. The currently fetched official WebShop, verl-agent, and G2PO reward files retain the original dictionary-items branch.

| Codebase and inspected version | Option handling | Evidence |
|---|---|---|
| Princeton WebShop, current `master` | Original `.items()` branch | [goal.py](https://github.com/princeton-nlp/WebShop/blob/master/web_agent_site/engine/goal.py#L238) |
| GiGPO/HGPO's verl-agent, current `master`, also local `20bd331bdbc9026a5668e11362178e10ab7400c8` | Original `.items()` branch | [goal.py](https://github.com/langfengQ/verl-agent/blob/master/agent_system/environments/env_package/webshop/webshop/web_agent_site/engine/goal.py#L238) |
| G2PO, current `main`, also local `b8ffaf56ad241ce73507908e6ed30209cafd324d` | Original `.items()` branch | [goal.py](https://github.com/Nala-YN/G2PO/blob/main/agent_system/environments/env_package/webshop/webshop/web_agent_site/engine/goal.py#L238) |
| FedAgent, current `main` | Same dictionary-values correction as our offline rescoring | [goal.py](https://github.com/sunblaze-ucb/FedAgent/blob/main/fedagent/envs/webshop/engine/webshop/web_agent_site/engine/goal.py#L247), [regression tests](https://github.com/sunblaze-ucb/FedAgent/blob/main/tests/test_webshop_option_reward.py) |
| Agent-R1, current `main`, full-WebShop reward path | Converts dictionary goals to option values before color normalization and fuzzy matching | [engine.py](https://github.com/AgentR1/Agent-R1/blob/main/recipes/webshop/env/engine.py#L204) |

The official WebShop, current verl-agent/G2PO files, both local reference copies, and our active training copy all have SHA-256 `9703c8583244e2041182eb14856186b998145ab16ce2ba9aef8cb680877c8ed5`. Thus our active scorer is byte-identical to these reference reward files at inspection time. Current main/master URLs may change later; the pinned local commit IDs above identify the versions used for our comparisons.

## Exact external precedent

FedAgent's [2026-07-25 bugfix entry](https://github.com/sunblaze-ucb/FedAgent/blob/main/fedagent/docs/bugfixes.md?plain=1#L1268) identifies the same mismatch between selected option values and target `(key, value)` pairs. Its documentation reports 458 affected goals in the 6,910-goal pool and successful oracle scoring after the fix. This matches our independently saved count of 458 affected goals. Its fix also adds rejection of non-string inputs to color normalization, whereas our counterfactual rescoring only changes the goal-option argument shape. The two patches therefore share the relevant scoring correction, but are not identical in every defensive check.

Agent-R1 is additional evidence for value-based matching, not evidence that its entire scorer equals ours. The inspected full-WebShop path has its own surrounding implementation, and its non-full reward path uses a different scoring formula. We do not treat those results as numerically interchangeable with the M11 protocol.

## Why the shape change matters

The original scorer compares selected values against dictionary key/value tuples. Color normalization performs a substring test on a string but an element-membership test on a tuple, so the two sides may be transformed differently even when the user selected the exact required option. Our offline replay instead supplies goal values, retaining the existing normalization, fuzzy threshold, and reward formula.

The problematic branch depends on goal representation. Dictionary-based synthetic goals take it; goals already stored as value lists bypass it. Presence of the source-code branch alone does not establish the magnitude of the effect in every published experiment or dataset.

## Consequence for reporting our runs

- M11's original-protocol final success remains **179/256 = 69.921875%**.
- The saved independent replay, rescored with goal values, gives **186/256 = 72.65625%**, task score **86.450825/100**. This should be labeled **option-fixed offline rescoring**, not silently substituted for the original benchmark result.
- The main M3/M5/M10/M11 comparisons retain their original scoring protocol. A corrected comparison needs all compared trajectories/checkpoints evaluated under the same corrected protocol.
- The queued M11-NOCTX ablation retains the original scorer to isolate the context-statistics change. This audit changes no training, queue, or environment code.

Local evidence: [our correction](rescore_options.py), [our rescored outcomes](option_rescoring.json), [public-source audit](upstream-option-scorer-audit-20260919.json).
