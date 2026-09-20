# Qwen3.8-27B prompt-only benchmark baseline

Evaluate the frozen **Qwen/Qwen3.8-27B** checkpoint on the same ALFWorld and WebShop environments, observation/history prompts, action parsers, and original reward functions used by the local M10/M11 runs. This is inference only; it does not compute CCPO advantages or update weights.

The implementation is versioned at `qwen_baseline/`. In this workspace, `baselines/qwen` links here because the parent `baselines/` is an ignored legacy-directory link. Both entry points work. New checkouts should use `qwen_baseline/` directly.

## Model and environments

The public checkpoint is pinned in `model.json` to revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`. All **18 safetensors shards** were downloaded on 2026-09-20, totaling **55,586,114,863 bytes** with repository metadata/tokenizer files. File sizes and the shard index were verified. The download manifest is `.local/qwen38-model.json` and the snapshot is:

```
hf/hub/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
```

Serving uses a separate `.venv-qwen` with **vLLM 0.20.2 / Transformers 5.8.1**. Benchmark processes use the existing `.venv` for ALFWorld and `.venv-webshop` for WebShop. They run on CPU and send token IDs to the local model server. This avoids upgrading dependencies in the training environments or starting another Ray cluster.

To reproduce the serving setup from the project root:

```bash
python3 -m venv .venv-qwen
.venv-qwen/bin/python -m pip install -r qwen_baseline/requirements-server.txt
.venv-qwen/bin/python qwen_baseline/download_model.py
```

Existing benchmark assets are required: `alfworld_data/`, the two benchmark virtual environments, and WebShop's existing Java/Lucene installation and 1,000-item catalog. `ACG_JAVA_HOME` can override the workspace's Java 11 path. The model download is public and does not read `hf/hf-token.md`.

## Start the model server

Choose available GPUs; **2,3 below is an example**, not a reservation. The launcher checks for busy devices before loading. Two 80-GB GPUs are the initial serving configuration; actual inference memory and throughput have not yet been measured. A single GPU can be selected with `--gpus 2` when available.

```bash
cd /workspace/agent-context-grpo
python3 baselines/qwen/serve.py --gpus 2,3 --dry-run
python3 baselines/qwen/serve.py --gpus 2,3
```

Run the server in a separate terminal or tmux session. It listens on `127.0.0.1:8018`, serves BF16 weights, uses tensor parallelism across the selected GPUs, limits context to 8,192 tokens, and disables multimodal input for these text benchmarks. Inspect startup output; `/v1/models` becomes available when ready:

```bash
curl -fsS http://127.0.0.1:8018/v1/models
```

The served model ID includes the full pinned revision. The evaluator rejects endpoints advertising a different ID. The launcher records its exact command and GPU selection in `.local/qwen38-server-8018.json`. `--enforce-eager` is available for diagnosing CUDA graph compatibility. Neither model execution nor evaluation has been launched on occupied GPUs.

## Run evaluation

Use the same server for both benchmarks. The wrapper selects the correct benchmark Python environment. Prepared full-evaluation plans already exist in the folders below, so start them with `--resume`:

```bash
python3 baselines/qwen/run.py alfworld \
  --output experiments/qwen38-27b-alfworld-20260920 --resume

python3 baselines/qwen/run.py webshop \
  --output experiments/qwen38-27b-webshop-20260920 --resume
```

For new output folders omit `--resume`. `--prepare-only` writes the complete protocol and episode plan without contacting a model server or constructing environments. Run a small **actual model** smoke evaluation before the full jobs:

```bash
python3 baselines/qwen/run.py alfworld \
  --episodes 2 --workers 2 --output .local/qwen38-real-smoke/alfworld
python3 baselines/qwen/run.py webshop \
  --episodes 2 --workers 2 --output .local/qwen38-real-smoke/webshop
```

`--episodes` selects a prefix of the full episode plan. CPU execution concurrency (`--workers`, default 4) does not change task sampling. Use a new output directory if changing sampling, prompt settings, checkpoint, adapter/parser/scorer code, or episode count. Resume retries error episodes and skips completed episodes; differing configurations and episode identities are rejected.

## Comparison protocol

| Setting | ALFWorld | WebShop |
|---|---|---|
| Evaluation episodes | 128 | 256 |
| Split | `eval_in_distribution` / valid_seen | test goals 0–499 |
| Logical validation batch size | 32 | 128 |
| Environment seed | 1000 + worker slot | 1000 + worker slot |
| Recent observation/action history | 2 steps | 2 steps |
| Episode horizon | 50 actions | 15 actions |
| Prompt token cap | 2,048 | 4,096 |
| New tokens per action | 512 | 512 |
| Sampling | temperature 0.4, top_p 1, top_k -1 | same |
| Native Qwen thinking | disabled | disabled |
| Environment/scorer | current text ALFWorld worker | current 1,000-item WebShop worker |

These settings follow the local M10/M11 configurations. WebShop's actual `envdata/webshop_data/text/test.parquet` has **256 rows**, despite the historical launcher config's `val_data_size=128` field; ALFWorld's corresponding file has 128 rows. With WebShop, each of the two batches samples 128 goals without replacement; the same goal can occur across batches. These are not 256 distinct goals or the complete 500-goal test set.

`--eval-round 0` reproduces the initial validation sampling stream. Later rounds advance the environment reset stream (`--eval-round N`). A historical training step is not itself an eval-round number: initial validation, frequency, resume alignment and restarts determine the mapping. ALFWorld also depends on the local game-file order. Traces record actual game paths/goal identities; compare those before claiming an exact paired comparison with a historical checkpoint. Sampling seeds are deterministic, but generation is not guaranteed bit-identical across different GPU/batching arrangements.

The native Qwen chat template wraps each current environment prompt as a single user message. The prompt still requests the benchmark's `<think>...</think><action>...</action>` format; disabling **native thinking** controls Qwen's template and does not rewrite this instruction. `/v1/completions` receives the exact token IDs, preserving the generated continuation instead of losing reasoning text to a chat API parser. Prompt overflow is recorded as an error rather than silently truncating observations or history.

The existing WebShop parser requires response-side `<think>` tags when computing its **format-valid flag**. Qwen's non-thinking template prefills an empty thinking block, so a correct `<action>` continuation may receive `legacy_format_valid=false` while its action still executes. We keep this historical parser behavior and separately log `action_tag_valid`; neither field changes the environment's task score. The ALFWorld parser already accepts Qwen's prefill convention.

For an explicit native-thinking comparison, use a **new output folder** and pass `--thinking --reasoning-effort low --max-tokens 2048`. Increase the server's context cap if prompt cap + output cap exceeds it. This is a separate compute-budget condition, not the matched 512-token baseline.

## Outputs and scoring

- `config.json`: checkpoint, settings, fixed episode identities, source/data hashes, and protocol fingerprint.
- `runtime.json`: benchmark package versions, execution concurrency, and server model advertisement.
- `episodes/NNNN.json`: atomic per-episode records for resumption, prompts, responses, actions, scores, token usage, finish reasons, and task identity.
- `episodes.jsonl`: ordered combined episode/turn trace after completion.
- `metrics.json`: success rate, score, episode length, action-format diagnostics, truncation rate, completion/error counts.

Success means ALFWorld's `won` or a completed WebShop purchase with original reward exactly 1. WebShop score retains the original graded task reward in [0,1]; ALFWorld score is binary. Metrics are fractions, so multiply by 100 for percentages. No item-option scoring correction is applied to the primary result. WebShop traces preserve goal, selected options, final product and price so purchases can be rescored offline under a separately labeled correction.

Incomplete evaluations have `success_rate=null` and `score=null`. Partial aggregates use `observed_*` fields; HTTP/environment failures are counted as errors and never silently dropped into a misleading final result. Per-episode traces and model weights stay ignored by Git; aggregate results and experiment notes can be versioned.

## Validation and current execution status

The checkpoint download, serving dependency imports, exact server CLI arguments, and real environment reset/action checks are validated. CPU regression tests and the complete synthetic HTTP integration test can be run with:

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m unittest discover \
  -s qwen_baseline/tests -p 'test_*.py' -v
CUDA_VISIBLE_DEVICES='' .venv/bin/python qwen_baseline/tests/integration_smoke.py
```

The integration test uses real ALFWorld/WebShop environments and the actual Qwen tokenizer, but **synthetic action responses**. Its metrics are not Qwen results. It checks two episodes/two actions per benchmark, HTTP token-ID requests, saved traces, resume skipping, and changed-config rejection. Artifacts remain under `.local/qwen38-integration-smoke/`.

**Real 27B generation and full benchmark scores are pending an available GPU.** All eight A100s were busy during setup on 2026-09-20; existing runs were left running. Passing CPU checks does not validate the model's GPU kernels or final inference memory use.

Model architecture and template source: [official Qwen model card](https://huggingface.co/Qwen/Qwen3.8-27B). Serving reference: [official vLLM Qwen3.8-27B recipe](https://recipes.vllm.ai/Qwen/Qwen3.8-27B).
