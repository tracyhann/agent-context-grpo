# Qwen3.8-27B full held-out evaluation, seed 101

Requested September 20, 2026: start after the active M10-NOCTX ALFWorld run completes, then evaluate one round of all **140 seen + 134 unseen ALFWorld tasks and 500 WebShop test goals**. This is prompt-only evaluation of the frozen base checkpoint, with no training or benchmark-specific adaptation.

Live/final tables: [RESULTS.md](RESULTS.md), machine-readable [RESULTS.json](RESULTS.json). The durable controller writes [RUN_STATE.json](RUN_STATE.json) and [FINAL_AUDIT.json](FINAL_AUDIT.json). Final cells are withheld until the full corresponding split is complete. The currently running training validation (128 sampled episodes) is a different population from this census.

## Model and inference protocol

- `Qwen/Qwen3.8-27B`, pinned revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, local BF16 weights, tensor parallelism 2. No quantization.
- Native thinking **enabled**, effort **xhigh** (model's default); official thinking sampling: temperature 1.0, top-p 0.95, top-k 20, min-p 0, presence/frequency penalties 0, repetition penalty 1.0.
- **65,536 output tokens per action is our explicitly chosen evaluation cap**, matching the DeepSeek experiment's budget. It is not claimed to be a universal Qwen default. Context limit 131,072; prompt + output overflow raises an error instead of silently truncating.
- Checkpoint-native chat template, `preserve_thinking=True`; server `/tokenize` produces exact token IDs, then `/v1/completions` generates. The benchmark's history-2 prompt is supplied as the current user message; native reasoning is not added to future environment prompts.
- The native template prefills `<think>`. Only the text **after the first `</think>`** is passed to the environment action parser; incomplete native reasoning produces an empty action, even if it quotes an action tag. Both native reasoning and final content are logged. An empty compatibility think block satisfies the existing parser when final content contains only an action tag.
- Server seed 101. Per-action sampling seed = `101 + suite_offset + episode_id*1000 + zero_based_turn`; suite offsets seen=0, unseen=1,000,000, WebShop=2,000,000. Environment/task-order seed 101. These seeds do not guarantee bit-identical outputs across different inference software/hardware/batching.
- Same benchmark prompts/scorers as DeepSeek: history 2, **ALFWorld 50 actions / WebShop 15 actions**, no extra retries of benchmark tasks, one census each. Local transport retries are permitted; no API fees apply.

Official references: [model card](https://huggingface.co/Qwen/Qwen3.8-27B), [vLLM recipe](https://recipes.vllm.ai/Qwen/Qwen3.8-27B). Dependency versions actually used are frozen in `environment-qwen-server.json`, including vLLM 0.20.2 / Transformers 5.8.1.

## Task populations and scores

`task-manifests.json` contains every exact gamefile / goal position. ALFWorld skips unsolvable, movable and sliced variants exactly as the environment split does. Seen counts: Pick35, Look13, Clean27, Heat16, Cool25, Pick2 24. Unseen counts: Pick24, Look18, Clean31, Heat23, Cool21, Pick2 17. The two populations are disjoint.

WebShop evaluates goal positions 0–499 once, on the existing **1,000-product catalog**. `webshop-goals-seed101.json` freezes the same generated goals and prices used by the completed DeepSeek seed101 evaluation. Every worker reseeds data generation after imports, and checks its actual initial goal and manifest hash.

Report ALFWorld **Pick | Look | Clean | Heat | Cool | Pick2 | All**, separately for seen and unseen. All is successes divided by task count, not the unweighted mean of categories. WebShop **Score | Succ.** are 100×mean original environment score and 100×success fraction. Original item-option matching is used; no post-hoc correction. Horizon failures remain in the denominator. Pre-purchase state is saved because terminal WebShop actions auto-reset the environment.

## Queue and reproduction

`chain.json` contains exact server and evaluator commands. `scripts/chain_qwen_census.py` waits for the existing training controller to report successful completion, verifies final step150 validation and both ranks of final/pinned/best checkpoints, checks that the original trainer identity has exited, then checks GPUs **0,1** are idle twice. It preserves the existing GPU holder. The controller verifies all 889 pinned source/data files before launching. It does not preempt training.

Validated setup: two idle A10080GB GPUs, BF16, max context131072, max sequences8, memory utilization0.85, eager execution. Actual one-action ALFWorld and WebShop tests both generated valid final actions with completed native reasoning. The temporary test server on GPUs4,5 was stopped afterward; diagnostics are in `preflight/`. Test requests are not counted in the full census.

To reproduce, use the pinned snapshot and recorded environments/data; choose a **new output folder**, then run the exact `server_command` and each `evaluations[].command` in `chain.json`, changing only physical GPU selection, port, and output paths as needed. The suite order is seen140 → unseen134 → WebShop500. Do not rerun the active controller in the same folder. Each evaluator refuses to overwrite an existing configuration.

```bash
cd /workspace/agent-context-grpo
.venv/bin/python scripts/report_census_eval.py experiments/qwen38-27b-full-heldout-seed101-20260920
```

`source/` holds immutable snapshots of the adapters, prompts, parsers and scorers. `input-sha256.json`, `source-sha256.json`, `dataset-sha256.json`, package inventories and Git revision record the exact local state, including uncommitted changes. Raw `outputs/<suite>/episodes/*.json` and `api_calls/*.jsonl` hold prompts, observations, actions, responses, seeds, finish reasons and token usage. Raw outputs stay local; summaries remain versionable. Local inference has **$0 API fee**; GPU rental/electricity is not estimated, and server start/end timestamps are retained.

## Acceleration applied September 21, 2026

The original sequential schedule was amended after seen140 completed. WebShop500
now runs concurrently with the ongoing unseen134 evaluation, using a second
BF16 server on GPUs2,3, port8019, with compilation/CUDA graphs enabled and 16
workers. The original unseen server and trajectories remain uninterrupted.
Model, xhigh thinking, seed formula, generation budget and scoring are retained.
The old controller joins the managed WebShop execution instead of repeating it.
Serving configuration differs between the ALFWorld and WebShop suites and is
recorded explicitly; bit-identical generation across batching/kernels is not
claimed. [Acceleration commands, measurements, tests and provenance](acceleration-20260921/NOTES.md).
Parallel WebShop process status: [STATE.json](acceleration-20260921/STATE.json).
