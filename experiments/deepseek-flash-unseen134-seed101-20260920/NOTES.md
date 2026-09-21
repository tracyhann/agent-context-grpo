# DeepSeek Flash: remaining unseen ALFWorld134, seed101

Requested September20,2026 as an extension of the [completed seen140 + WebShop500 evaluation](../deepseek-flash-default-full-seed101-20260920/RESULTS.md). This run evaluates **all 134 distinct held-out `valid_unseen` games exactly once**. It does not repeat the completed seen or WebShop tasks.

Live/final requested **Pick | Look | Clean | Heat | Cool | Pick2 | All** table and API cost: [RESULTS.md](RESULTS.md). Per-task results, exact prompt/action trajectories, native reasoning, final responses and API token-usage records are retained. API cost is calculated from returned usage and cache pricing; missing-usage attempts are reported separately, not silently treated as free.

## Protocol

Exactly the previous DeepSeek protocol: model `deepseek-flash`; native thinking on, reasoning effort `high`, maximum output65536; default thinking sampling (no unsupported temperature override); same history-2 prompt, action parser and ALFWorld50-action horizon. Only API final content can supply an action; native reasoning is logged separately. Seed101 controls task order and environment; this API has no supported generation-seed parameter. Model aliases/backend changes mean bit-identical future reproduction is not guaranteed.

`benchmark_census.py` selects the actual **eval_out_of_distribution** environment, not the seen split. Every episode asserts its reset gamefile equals the planned task. Counts: Pick24, Look18, Clean31, Heat23, Cool21, Pick2 17; total134. `PREFLIGHT.json` records a real unseen-environment reset before paid requests. Exclusions (unsolvable/movable/sliced) match the existing benchmark loader. All successes/134 is the micro-average, and action-limit failures remain in its denominator.

32 CPU workers; no GPUs. The transport wrapper retries HTTP disconnects and malformed/incomplete JSON through the existing retry/error ledger; all returned API responses are persisted before environment actions. Retries do not regenerate any previously returned action.

## Reproduction

Exact executed command and PID are in `RUN_STATE.json`. To reproduce into a new output directory:

```bash
cd /workspace/agent-context-grpo
.venv/bin/python scripts/deepseek_split_eval.py \
  --split unseen --seed 101 --workers 32 \
  --output experiments/NEW-EXPERIMENT/outputs/alfworld-unseen
```

The explicitly authorized credential is read from `baselines/deepseek/DEEPSEEK-API-KEY.txt`; it is neither copied nor hashed. Preserve it locally and do not commit it. Copy the experiment's `suites.json` to the new experiment root for aggregation. The durable `scripts/watch_census_eval.py` records progress and produces `FINAL_AUDIT.json` after the evaluator exits.

```bash
.venv/bin/python scripts/report_census_eval.py experiments/deepseek-flash-unseen134-seed101-20260920
```

`task-manifest.json` fixes all134 task paths. `source/` and SHA256 manifests freeze the environment, prompts, parsers, adapters and relevant data; dependency inventories and Git state document the executable setup. `outputs/alfworld-unseen/config.json` pins settings and runner hashes. Raw outputs stay local; summaries are versionable.

## Cost

This September20 run is off-peak (Sunday). DeepSeek Flash USD/million: cached input0.003, uncached input0.15, output0.6; reasoning is already included in output. Weekday UTC01–04/06–10 peak rates double; the reporter applies time-of-request rates if an evaluation crosses a pricing boundary. Resulting costs are usage-derived estimates, not an account invoice.

The previous seen140 + WebShop500 run cost **$9.940843356** in received usage, plus a separate conservative allowance of **$0.15768915** for four interrupted requests without returned usage. This unseen run's incremental and combined costs are updated in RESULTS.md/JSON. Earlier pilots are excluded.

[Official API defaults](https://api-docs.deepseek.com/api/create-chat-completion/), [official pricing](https://api-docs.deepseek.com/quick_start/pricing/).

## Combined final table

After both datasets complete, `finalize.py` verifies the frozen source/data inputs and writes [COMBINED_RESULTS.md](COMBINED_RESULTS.md), joining seen140, unseen134 and WebShop500 with separate and total costs. `REPLAY_VALIDATION.json` records one successful and one horizon-failure unseen episode replayed locally with exact prompts, actions, observations and scores, without API calls.

## Original interruption: insufficient API balance (resolved September 21)

The first invocation completed **132/134** tasks with **104 successes** and stopped with HTTP402 on episodes100 (46 saved actions) and123 (25 saved actions). The provider balance endpoint confirmed `is_available=false`. Neither interrupted task has been counted as a completed failure; the full134 score remains pending. All returned responses are retained. Received usage cost is **$3.307043799**; the two HTTP402 rejections are tracked separately from unknown-usage transport interruptions.

After topping up, resume only those errored episodes with:

```bash
cd /workspace/agent-context-grpo
.venv/bin/python scripts/resume_deepseek_census.py \
  experiments/deepseek-flash-unseen134-seed101-20260920/outputs/alfworld-unseen
```

The helper checks balance first. It rebuilds the exact unseen environment, replays the71 saved actions/responses, checks their prompt hashes, then requests only the remaining continuation. Originals are archived and completed tasks are untouched. Its insufficient-credit path was checked and issued zero generation requests. Source snapshots include the recovery helper. After completion, rerun the reporter, update the final audit and run `finalize.py` to generate the combined table.

`RESUME_VALIDATION.json` independently confirms both interrupted prefixes (46 +25 =71 actions) replay exactly, including their next-request hashes; zero API calls were made.

## September 21 recovery with replacement credential

The user supplied `baselines/deepseek/DEEPSEEK-API-KEY2.txt` and authorized finishing
the two interrupted tasks. Recovery started at 2026-09-21 05:36:40 UTC with two
CPU workers. The runtime credential override leaves the original config, old
credential file, 132 completed trajectories and saved responses intact. No key
contents or key hashes are recorded.

```bash
.venv/bin/python scripts/resume_deepseek_census.py \
  experiments/deepseek-flash-unseen134-seed101-20260920/outputs/alfworld-unseen \
  --workers 2 --key-file baselines/deepseek/DEEPSEEK-API-KEY2.txt
```

The helper replays episode100's 46 and episode123's 25 saved actions with prompt
hash validation; API calls begin only after each saved prefix is exhausted.
Model alias, thinking/high, 65,536 token cap, history2, seed101 and 50-action
ceiling are retained. Separate recovery metadata records the credential source
path and counts of replayed versus new responses.

[key2-recovery-20260921/BEFORE.json](key2-recovery-20260921/BEFORE.json) records
all 132 completed trajectory hashes, the original config hash, prefix lengths
and pre-recovery cost. The two interrupted trajectories and their API ledgers
are also archived. [INPUT_VERIFICATION.json](key2-recovery-20260921/INPUT_VERIFICATION.json)
checks 892 original pinned inputs. The only changed input revisions are the two
credential-override helpers and the unrelated Qwen scheduling entrypoint; each
old/new hash and reason is retained. All other benchmark/source/data inputs
match. The finalizer accepts only those exact verified revisions.

Three CPU regression tests passed: cached turns make no new API calls, the key
override does not mutate the original config, replay mismatch rejects new calls,
and completed tasks are skipped. Logs and recovery source snapshots are retained
in `key2-recovery-20260921/`. Current official Flash rates were rechecked and
recorded in [PRICING.json](key2-recovery-20260921/PRICING.json); they match the
existing usage reporter. New continuation cost is separated from the original
unseen and combined seen/unseen/WebShop totals in the final recovery report.

### Recovery completed and audited

All **134/134** unseen tasks completed; **104/134 = 77.61%**. The new key funded 29 new requests costing **$0.111918**. All 132 prior completed trajectories and all 71 cached actions/responses are unchanged. [Recovery audit and costs](key2-recovery-20260921/RESULTS.md); [combined final results](COMBINED_RESULTS.md).
