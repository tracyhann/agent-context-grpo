# DeepSeek Flash full held-out evaluation — seed 101

The user authorized one full evaluation of ALFWorld's 140 held-out tasks and
WebShop's 500 held-out goal positions, using DeepSeek's official thinking
defaults. This is a separate experiment from the seed-0 cost pilot. Results and
usage-derived API cost are refreshed in [RESULTS.md](RESULTS.md) and
[RESULTS.json](RESULTS.json); per-task outcomes are in
[per-task-results.json](per-task-results.json).

## Completed result

All **140 ALFWorld + 500 WebShop** tasks completed. ALFWorld: **105/140, 75.00%**.
WebShop: **Score 27.8671 / Succ 22.00% (110/500)**, in 0–100 units.
The full requested per-task-type table is in [RESULTS.md](RESULTS.md).

All **9,310 received API responses** reconcile one-to-one with saved action turns.
Usage-derived cost is **$9.940843**. Four interrupted requests did not return usage;
a separate full-ceiling allowance for those is **up to $0.157689**, making the
accounted cost plus that allowance **$10.098533**. This allowance is not claimed
to have been billed. Every priced response occurred off-peak.

Only **one ALFWorld response** hit 65,536 tokens; WebShop had no output-cap hits.
The four transport-failed tasks resumed from their saved responses and all
finished. The initial supervisor/metrics reflected those errors; their original
versions are retained as `RUN_STATE_INITIAL.json` and per-benchmark
`metrics-before-recovery.json`, with final status reconciled to the audited records.

[FINAL_AUDIT.json](FINAL_AUDIT.json) verifies complete unique task coverage,
all received usage, goals/prices/scores, and 492 source/data hashes. Five local
trajectory replays also reproduced every prompt, transition and reward, with
zero API calls. No credential value appears in the experiment artifacts.

## Pinned evaluation protocol

| Parameter | Value |
|---|---|
| API endpoint | `https://api.deepseek.com/chat/completions` |
| Requested model | `deepseek-flash` |
| Documented model version at launch | DeepSeek-V4.1-Flash |
| Native thinking | enabled |
| Reasoning effort | high, official default |
| `max_tokens` | 65,536 per action, explicitly pinning the documented 64K thinking default |
| Temperature / top-p / API seed | omitted, provider defaults; no deterministic generation seed is supported |
| Environment / task ordering seed | 101 |
| Python hash seed | 101 |
| Observation history | 2 steps |
| ALFWorld action limit | 50 |
| WebShop action limit | 15 |
| Local execution | CPU only, 32 independent worker processes per benchmark |
| HTTP timeout | 900 seconds per attempt |
| Transient retry policy | at most 5 attempts, only network/429/5xx failures, exponential backoff capped at 30s |
| Response-cap retries | none; capped responses remain part of the trajectory |

Thinking and the output ceiling are explicitly set to the documented defaults,
rather than left mutable if the provider changes its defaults later. The model
alias itself is not revision-pinned by the service. Every response retains its
returned model name, backend fingerprint, request ID and timestamp. Exact
bit-identical API generation is not promised by seed 101.

## Task populations and requested table

ALFWorld uses all **140 solvable `valid_seen` / `eval_in_distribution`** games,
once each. These are held-out tasks in seen environments. The distinct
`valid_unseen` split contains 134 local games and is not part of this run.
Selection uses the same solvability and task filters as the upstream loader.
The exact game is installed in a one-game environment; reset identity is asserted.

| Report label | Environment task type | Denominator |
|---|---|---:|
| Pick | pick_and_place_simple | 35 |
| Look | look_at_obj_in_light | 13 |
| Clean | pick_clean_then_place_in_recep | 27 |
| Heat | pick_heat_then_place_in_recep | 16 |
| Cool | pick_cool_then_place_in_recep | 25 |
| Pick2 | pick_two_obj_and_place | 24 |
| All | all six types | 140 |

Each ALFWorld column is success percentage. All is the total number of successful
tasks divided by 140, not an unweighted average of the six columns.

WebShop uses every test position **0–499**, once each, under seed-101 goal
construction. The runtime has 6,910 synthetic goals and the existing
**1,000-product catalog/index**. Positions 500 onward are excluded. This is the
local benchmark's 1K-product configuration, not the full million-product catalog.
The 500 exact goals and all product prices are saved in
[webshop-goals-seed101.json](webshop-goals-seed101.json). A goal index alone is
insufficient to reproduce this environment because goal shuffling and price
construction depend on the seed.

WebShop Score = 100 × mean original environment terminal dense score;
Succ = 100 × fraction with terminal score exactly 1.0. Non-purchase action-limit
failures score zero. Headline metrics retain the original item-option scorer;
no post-hoc option-scoring correction is applied. Purchase product/options are
saved for any later separately labeled analysis.

## Reproducibility fixes validated before API calls

1. **Native reasoning cannot execute actions.** The older cost pilot could feed
   `reasoning_content` into the legacy parser when final content was empty. This
   evaluator keeps it only in the log. If final content lacks a thinking block,
   an empty compatibility block is prepended. The final content remains the sole
   source of action tags. Existing environment prompts/parsers are otherwise reused.
2. **Seeded WebShop construction is independent of process reuse.** A preflight
   found that two initializations with seed 101 could produce different sampled
   prices and price bounds. The evaluator now explicitly seeds and rebuilds
   product prices and synthetic goals after all lazy initialization, then applies
   the seed-101 goal shuffle. Independent fresh/reused environments matched.
   Each episode checks its entire 500-goal hash against the saved manifest before
   requesting an API action. This change is confined to the API evaluator;
   training environment files are unchanged.
3. **Terminal WebShop state is captured before auto-reset.** The environment
   resets itself before returning the terminal observation. Each turn therefore
   records pre-action state, and a terminal purchase retains that state. The
   returned dense reward remains authoritative.
4. **Every received paid response is logged before stepping the environment.**
   Usage, native reasoning, final content, parsed action, prompt, observations,
   reward, done flag and environment info are retained. Network failures without
   usage are logged separately, so their possible unobserved charges are disclosed.

The benchmark prompt still asks for a visible `<think>...</think><action>...</action>`
response. Native thinking is an additional API behavior. Each current prompt is
sent as a single user message; there is no tokenizer-based input truncation.
The inherited WebShop prompt builder can drop history when its constructed prompt
exceeds 13,000 characters; exact actual prompts are recorded. No reward, goal
answer, selected target product, or privileged state is added to the model input
beyond the existing benchmark prompt.

## Accounting

Per-response off-peak USD charge:

`(cache_hit_input × 0.003 + cache_miss_input × 0.15 + completion × 0.6) / 1e6`.

Completion tokens already include reasoning tokens. Peak prices double all three
components. Official peak windows are Monday–Friday 01:00–04:00 and 06:00–10:00
UTC, excluding Chinese holidays. This run starts Sunday, 2026-09-20, off-peak.
The report classifies each response timestamp and retains pricing-window counts.
Any future reuse of the report on a Chinese weekday holiday must apply that
holiday's off-peak exception explicitly.

Cost includes every received response in the run's API ledger, including capped
responses and unsuccessful tasks. It excludes the earlier seed-0 pilot and the
single 8K diagnostic. Attempts without returned usage cannot be fully priced and
are listed separately. Figures are usage-derived charges, not an account invoice.

## Reproduce

The active source checkout at launch is branch `adb`, commit
`f05157f613d571fe63e2b3edaa5c8b29d606f1f0`, with working-tree changes. Commit alone
is therefore insufficient: use the exact files under `source/` at their original
relative paths and verify `source-sha256.json` and `dataset-sha256.json`.
The two Python package sets are recorded in `environment-alfworld.json` and
`environment-webshop.json`. Existing local runtime assets and Java setup are
those described in `.local/WEBSHOP_READY.md`.

Run from the repository root, with the API key in the separately supplied ignored
credential file. No key value is copied into this experiment folder.

```bash
python3 experiments/deepseek-flash-default-full-seed101-20260920/run_evals.py \
  --output-root experiments/deepseek-flash-default-full-seed101-repeat
```

The launcher refuses to overwrite an existing evaluation. It starts both
benchmark workers, saves exact commands/PIDs in `RUN_STATE.json`, and updates the
requested result table every 30 seconds. The actual task plans and settings are
also saved in `outputs/<benchmark>/config.json`.

To recompute the table and cost without API calls:

```bash
python3 scripts/report_deepseek_full_eval.py \
  experiments/deepseek-flash-default-full-seed101-20260920
```

Raw records: `outputs/<benchmark>/episodes/` and `api_calls/`; process logs:
`outputs/alfworld.log`, `outputs/webshop.log`, `outputs/supervisor.log`.
The outputs directory is gitignored and must be archived separately to preserve
full replay evidence. Configuration/source/data manifests and summarized results
are outside that ignored directory.

[Official API defaults](https://api-docs.deepseek.com/api/create-chat-completion/),
[thinking mode](https://api-docs.deepseek.com/guides/thinking_mode/),
[official pricing](https://api-docs.deepseek.com/quick_start/pricing/).

## Transport recovery and local replay

Some requests ended with `IncompleteRead` or `RemoteDisconnected`. Their
previously completed actions are replayed from the existing response ledger,
with prompt hashes and transitions checked, before making the next new request.
No saved response is regenerated, and no task is dropped or replaced. The
original errored episode is retained under `outputs/<benchmark>/recovery/`.
The recovery wrapper converts incomplete HTTP/JSON responses into retryable
transport errors; model settings, prompts, environments and scoring are unchanged.

To recover transport-failed episodes after the main processes have finished:

```bash
PYTHONHASHSEED=101 .venv/bin/python scripts/recover_deepseek_eval.py \
  experiments/deepseek-flash-default-full-seed101-20260920/outputs/alfworld
PYTHONHASHSEED=101 .venv-webshop/bin/python scripts/recover_deepseek_eval.py \
  experiments/deepseek-flash-default-full-seed101-20260920/outputs/webshop
```

Use the virtual-environment executable path as written, without resolving its
symlink to the system interpreter. An initial recovery setup attempt used the
system interpreter, failed before any API calls, and was restored from the
archived original episode. These setup failures are retained in the audit trail.

To verify a saved trajectory locally with **zero API calls**:

```bash
PYTHONHASHSEED=101 .venv-webshop/bin/python scripts/replay_deepseek_episode.py \
  experiments/deepseek-flash-default-full-seed101-20260920/outputs/webshop/episodes/0002.json
```

Initial replay checks passed for two ALFWorld tasks (success and action-limit
failure), and three WebShop tasks (success, partial-credit purchase, and
action-limit failure). Every prompt, parsed action, next observation, reward
and done flag matched. The partial purchase reproduced score 5/7 exactly.

For interrupted requests without usage, the report also supplies a conservative
missing-charge allowance: matching retry prompt tokens charged entirely as cache
misses, plus the full 65,536-output-token ceiling for each interrupted attempt.
This allowance is separate from measured usage; it is not asserted to have been
billed. The reporting-only update is snapshotted; the launch report version is
preserved in `source-revisions/`.

To rerun the final offline audit:

```bash
python3 scripts/audit_deepseek_full_eval.py \
  experiments/deepseek-flash-default-full-seed101-20260920
```
