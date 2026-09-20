# DeepSeek Flash API cost pilot — ALFWorld and WebShop

The user authorized a few real complete examples per benchmark, followed by an
estimate for one full evaluation seed: ALFWorld 140 tasks and all WebShop test
positions. This pilot runs six episodes per benchmark, using CPU environments
and the official DeepSeek endpoint. Full evaluations are not launched.

## Completed result

All 12 pilot episodes finished. API token charges were $0.30404 for the pilots,
plus $0.00280 for one 8,192-token diagnostic replay: **$0.30684 total**.
ALFWorld succeeded on 4/6 tasks; WebShop on 3/6. These are small pilot outcomes,
not full benchmark scores. Response caps were reached on 43/173 ALFWorld and
4/61 WebShop requests. Full-evaluation extrapolation at this measured budget is
**$9.62 off-peak / $19.23 peak**; a 2x no-cache planning allowance rounds to
**$20 off-peak / $40 peak**, and is not a guaranteed upper bound.

## Protocol and task coverage

- Model: `deepseek-flash`, currently documented as DeepSeek-V4.1-Flash.
- Native thinking enabled, effort `high`; 4,096 maximum completion tokens per request.
  Native reasoning is included in the API's completion-token count.
- Task/environment seed 0. No unsupported API generation-seed parameter is sent.
- ALFWorld: the actual `valid_seen` / `eval_in_distribution` split has 140 solvable
  games. The pilot chooses one game per task type using seed 0. The exact selected
  game file is installed in the environment and its reset identity is asserted.
  This is a full-split census target, distinct from the training logger's 128 sampled
  validation episodes. The local unseen split has 134 game files and is not used.
- WebShop: a seed-0 shuffle selects six distinct goals from the full test range
  0..499. Full evaluation means all 500 goals once, retaining the existing
  1,000-product catalogue and original item-option scorer. This differs from the
  training logger's 256 sampled resets, which can repeat a goal across batches.
- History 2 in both; maximum 50 ALFWorld actions / 15 WebShop actions per episode.
- The existing environment prompts, action parsers and scores are reused. Each
  prompt is sent as a single user message. Native reasoning is wrapped in a
  response-side thinking block when needed by the legacy parser; only an action
  emitted in the normal content can provide a valid action block. Truncated
  generations and their executed invalid actions remain recorded in the pilot.
- API prompt tokens are counted by the service; no Qwen tokenizer or silent
  prompt truncation is used. This is a separate API compute budget from the
  local model baseline's 512-token output cap. Temperature is not sent in native
  thinking mode, where the official API says it has no effect.
- Two CPU processes per benchmark. No model weights, CUDA device or Ray training
  cluster are allocated for this evaluation.

The API key is read from the user-specified file at request time and sent only
in the official endpoint's Authorization header. It is not copied into source,
configs, traces, hashes or this report; only its source path is recorded.

## Cost accounting

Record each successful API response's `usage`, request ID, backend fingerprint,
latency, finish reason, content and native reasoning. The per-request ledger is
written before stepping the environment so a paid request is not lost if an
environment action fails. Episode records are saved after each turn.

Off-peak USD per million: cache-hit input 0.003, cache-miss input 0.15,
completion 0.6. Peak prices are twice these. Reasoning is part of completion,
not an extra term to add again. This pilot is running on Sunday, 2026-09-20,
entirely in the off-peak window.

The ALFWorld extrapolation weights each task type by its population (13 light,
35 simple placement, 27 clean, 25 cool, 16 heat, 24 two-object tasks). WebShop
uses the random sample mean times 500. The report also recomputes input cost
without cache savings and provides a planning allowance of twice that estimate.
This allowance is not a confidence interval. Six examples cannot establish
benchmark accuracy or tightly bound costs on long/failing trajectories.

See [COST_ESTIMATE.md](COST_ESTIMATE.md) and [COST_ESTIMATE.json](COST_ESTIMATE.json)
for final results when both pilots finish. Raw per-turn episodes and request
ledgers are under ignored `outputs/<benchmark>/episodes/` and `api_calls/`.

## Reproduction

Use a new output path for each launch; automatic paid reruns/resumption are
intentionally not performed. With the existing virtual environments:

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python scripts/deepseek_api_eval.py --benchmark alfworld --episodes 6 --workers 2 --output experiments/deepseek-new-pilot/outputs/alfworld
CUDA_VISIBLE_DEVICES='' .venv-webshop/bin/python scripts/deepseek_api_eval.py --benchmark webshop --episodes 6 --workers 2 --output experiments/deepseek-new-pilot/outputs/webshop
python3 scripts/report_deepseek_api_cost.py experiments/deepseek-new-pilot
```

`--prepare-only` writes task/settings manifests without contacting the API.
`--episodes 140` / `--episodes 500` selects the complete corresponding split;
these full runs have not been started. Raw process logs and the initial WebShop
setup-only serialization failure are retained in `outputs/`. That failure was
fixed before WebShop sent any API requests; sets in its session state are now
converted to JSON lists.

[Official pricing](https://api-docs.deepseek.com/quick_start/pricing/),
[thinking-mode behavior](https://api-docs.deepseek.com/guides/thinking_mode/),
[usage schema](https://api-docs.deepseek.com/api/create-chat-completion/).
