# HGPO baseline: 2048 prompt tokens, 150 updates, WebShop 15 turns

Prepared on 2026-09-24; **no training launched or queued**.

The baseline runs the official HGPO recipe through `recipe.hgpo.main_hgpo`.
The estimator is `verl-agent/recipe/hgpo/core_hgpo.py`, pinned to upstream
`langfengQ/verl-agent` commit `20bd331bdbc9026a5668e11362178e10ab7400c8`.
The preparation script verifies its SHA-256 before writing a configuration.
This is the HGPO method with adjusted budgets, not a reproduction of the
published 30-turn WebShop budget.

## Prepared protocol

The current interpretation of **token budget 2048 is the input prompt limit**.
Generation retains the existing 512-token cap per turn. They have separate CLI
flags, so a 2048-token generation variant can be prepared explicitly if intended.
Neither limit refers to the dynamic minibatch token budget per GPU.

| Setting | ALFWorld | WebShop |
|---|---|---|
| Models | Qwen2.5-1.5B-Instruct and Qwen2.5-7B-Instruct | Same |
| Training iterations | **150** | **150** |
| Prompt / response tokens per turn | **2048 / 512** | **2048 / 512** |
| Maximum turns, train and validation | 50 | **15** |
| History length | 2 | 2 |
| HGPO weights / alpha | length / 1.0 | Same |
| Episode base group | False | False |
| Group advantage normalization | mean/std, population std | mean-centering |
| Tasks x trajectories per step | 16 x 8 | 16 x 8 |
| Learning rate / discount | 1e-6 / 0.95 | Same |
| Terminal reward / current invalid penalty | 10 or 0 / 0.1 | Same |
| Training / validation temperature | 1.0 / 0.4 | Same |
| Separate KL coefficient | 0.01 | 0.01 |
| Validation / checkpoint frequency | Every 5 steps | Every 5 steps |
| Validation chunk size | 64; full 128-row parquet | 64; full 256-row parquet |
| Configured GPUs / rollout TP | 1.5B: 4 / 1; 7B: 8 / 2 | Same |
| Benchmark data | ALFWorld, seen validation | WebShop 1K catalog |

`trainer.total_training_steps=150` is explicit, with an epoch ceiling of 150.
Thus increasing the number of training batches per epoch cannot accidentally
turn this into more than 150 updates. These are fresh runs, with no resume or
early stopping. GPU IDs are configuration only; no GPU reservation or memory
smoke test has been performed.

## What is retained from HGPO

For each task, the recipe groups turn occurrences by exact observation suffixes
of lengths k=1,...,min(H+1,t+1), including the current observation. At H=2 this
means up to three consecutive observations. Actions appear in the agent prompt
history but are not part of the grouping key. Groups are self-inclusive and
retain repeated visits; there is no CCPO trajectory exclusion or context kernel.

For a group g, the target is the discounted environment return minus the current
turn's invalid-action penalty. Group credit is Y-mean_g(Y), additionally divided
by population std_g(Y)+1e-6 in ALFWorld. Singleton groups contribute no credit.
The recipe aggregates nonzero group credits using weights proportional to
(k+1)^alpha, alpha=1, with its existing epsilon denominator. This follows the
actual upstream implementation, including its zero-credit filtering and indexing.
`base_group=False` excludes the episode-advantage group from that aggregation.
The resulting detached scalar is broadcast across generated response tokens.

The HGPO trainer computes returns and advantages before padding and balancing
the rollout batch. Its own trainer and environment manager are retained; shared
PPO workers, rollout collection, prompt templates and environment packages use
the same local verl-agent overlay as the other methods. Source hashes are stored
with each preparation. `ACG_ADV_ESTIMATOR=hgpo` disables CCPO frozen-feature capture.
No memory compaction, observation repair or original edge/future credit is added.

## Differences to account for in comparisons

The official HGPO scripts use prompt 4096 / response 512, 160 epochs, and
WebShop 30 turns. This preparation changes prompt to 2048, sets an explicit
150-update cap, and changes WebShop to 15 turns. History remains at the official
K=2 default; the generator also supports K=4.

Historical local ALFWorld CCPO/GiGPO settings already use prompt 2048 / response
512. Historical WebShop settings use prompt 4096 / response 512. Therefore the
new WebShop baseline is only a strict token-budget match to controls explicitly
set to prompt 2048. It should not be described as identical to those older runs.
HGPO retains upstream left truncation; the existing CCPO launcher uses error on
overlength input. A 2048 cap may truncate observation/history on longer WebShop
pages. Full configuration and runtime deltas are recorded in each `config.json`.

Runtime changes from the published launcher include dynamic token batching,
validation chunks of 64, local paths, rollout TP=1 for 1.5B, vLLM memory fraction
0.25, evaluate/save every 5 steps, and retaining the two latest actor checkpoints.
The source is unchanged, but these preparations do not claim identical runtime
or validation draws to historical checkpoints.

## Prepare and run

From the project root, with the existing benchmark environment installed:

```bash
python scripts/prepare_hgpo_baseline.py --backbone 1.5b --date YYYYMMDD
python scripts/prepare_hgpo_baseline.py --backbone 7b --date YYYYMMDD
```

Together these prepare both benchmarks at both model sizes and refuse to overwrite existing folders.
It always invokes the launcher with `--dry-run`; it does not acquire GPUs or
start Ray. Optional selections:

```bash
python scripts/prepare_hgpo_baseline.py --benchmark webshop --gpus 0,1 --date YYYYMMDD
python scripts/prepare_hgpo_baseline.py --backbone 7b --gpus 0,1,2,3,4,5,6,7 --date YYYYMMDD
python scripts/prepare_hgpo_baseline.py --history-length 4 --date YYYYMMDD
# If 2048 means generated tokens, set that independently and choose the prompt cap:
python scripts/prepare_hgpo_baseline.py --prompt-tokens 2048 --response-tokens 2048 --date YYYYMMDD
```

This dated preparation contains all four runs: ALFWorld/WebShop x 1.5B/7B.
The 7B configurations use eight GPUs and TP=2; 1.5B uses four GPUs and TP=1.
Within each benchmark only model, resolved model path, GPU allocation, TP and
experiment identity differ between sizes. The 150-step training and evaluation
protocol remains the same. Each 7B folder records `config-diff-from-1.5b.json`.
A later authorized launch uses the generated `run.sh` for the selected folder.
The existing setup scripts install the runtime; `scripts/sync_patches.sh` applies
the shared overlay. For a fresh checkout, obtain the above pinned upstream
revision before preparation; a changed HGPO core fails its provenance check.

| Benchmark | 1.5B | 7B |
|---|---|---|
| ALFWorld | [NOTES](hgpo-k2-prompt2048-response512-alfworld-1.5b-4gpu-20260924/NOTES.md) | [NOTES](hgpo-k2-prompt2048-response512-alfworld-7b-8gpu-20260924/NOTES.md) |
| WebShop | [NOTES](hgpo-k2-prompt2048-response512-webshop-1.5b-4gpu-20260924/NOTES.md) | [NOTES](hgpo-k2-prompt2048-response512-webshop-7b-8gpu-20260924/NOTES.md) |

Validation covers real Hydra schema composition for both model sizes and K=2/4,
actual HGPO trainer dispatch, hand-calculated hierarchical credit, PPO gradients,
detached targets, invalid-action penalties, and unchanged existing estimator
routes. See each folder's `VALIDATION.json`. GPU training remains untested.
