# Search-R1: a third benchmark for the rebuttal

**Prepared tooling only. Nothing here has been launched, and none of it has run on
a GPU.** The launcher, the estimator guard and the prepare script are covered by CPU
tests; the environment setup below is upstream's and is unverified on this
project's hosts.

Search-R1 (verl-agent's `search` env) answers the reviewer's concern that ALFWorld
and WebShop have recurring observations and small action spaces. Actions are
free-form search queries, and the observation after a search is the retrieved
passages, so a later turn repeats across rollouts only when two rollouts issued the
same query.

## Arms

`scripts/prepare_search_r1_ccpo.py` prepares four runs that share one resolved
config and differ in the estimator alone:

| Arm | Estimator | Notes |
|---|---|---|
| `ccpo` | main method, EP0 | H2, no shrinkage, `A_CC` only: the ALFWorld configuration |
| `ccpo-ep1` | main method, EP1 | adds the episode advantage, `A_CC + A_EP` |
| `grpo` | GRPO | |
| `gigpo` | GiGPO | similarity grouping at 0.9, as its published script sets |

The CCPO arms take every estimator key from the registry entry
`future-progress-h2-no-credit-shrinkage` and are checked against the recorded
ALFWorld control `m10-h2-no-credit-shrinkage-alfworld-1.5b-2gpu-20260919`. The
estimator's math is unchanged: exact task/observation peers, the task-bucket
fallback, hidden-state plus context-statistics features, soft weights at 0.15,
whole-trajectory exclusion, history and two-step future credit at 1/1. Its one
addition is a check that stops a Search-R1 run whose returns are off the 10/0
scale (see Protocol).

## Protocol

GiGPO's published `examples/gigpo_trainer/run_search.sh` (upstream `20bd331`),
applied by `scripts/exp_run.py` whenever `env_name=search`:

| Setting | Search-R1 | ALFWorld |
|---|---|---|
| Questions per step x rollouts | 256 x 5 | 16 x 8 |
| Turn ceiling / prompt-history turns | 4 / 4 | 50 / 2 |
| Prompt budget, truncation | 4096, left | 2048, error |
| KL coefficient / LR warmup / entropy | 0.001 / 0.1 / 0 | 0.01 / 0 / 0.001 |
| Mini-batch | 512 | 256 |
| Validation decoding | greedy | T=0.4 |
| Advantage mode | `mean_std_norm` | `mean_std_norm` |
| Steps | 662 (one epoch of 169,615 questions) | 150 |
| Validate / save every | 50 / 50 | 5 / 5 |

`tests/test_search_protocol.py` compares the generated command with that script,
override by override. Three things depart from it:

- **Reward scale.** The env's exact-match reward is 1/0. The estimator's success
  potential is fixed at 10, so at 1/0 every terminal potential reads as failure and
  a successful final turn is credited as a loss. `SearchEnvironmentManager` therefore
  multiplies the reward by `ACG_SEARCH_REWARD_SCALE=10`, which the launcher sets for
  every arm. The invalid-action penalty stays at 0.1, the reference's 0.01 times
  ten, so GRPO and GiGPO see the same standardized advantages as under the
  reference. `val/test_score` is ten times exact match; `success_rate` is exact
  match.
- **Resources.** `gpu_memory_utilization` and dynamic batching keep this tree's
  defaults. Set them with `--set` for the host.
- **Step cap.** The parquet holds real questions, so `total_epochs` is a step budget
  here and also caps `trainer.total_training_steps`.

The prompt-history window is 4 because the agent answers from what it retrieved;
the future horizon stays 2. This is the one place the "history 2 / future 2" pairing
of the main method does not carry over.

## Setup

One-time, from the repository root. These follow upstream's README with this
project's paths.

```bash
bash scripts/sync_patches.sh
```

The reward scale lives in the env-manager overlay. The prepare script refuses a
stale overlay, and the CCPO estimator stops at the first successful episode whose
return is not 10.

Trainer environment (`.venv`): the search env needs `gym`, `requests` and
`func_timeout` importable, the dependencies of the skyrl-gym package bundled with
verl-agent. Install whichever is missing.

Questions (written to `envdata/searchr1/{train,test}.parquet`, the launcher's
defaults):

```bash
PYTHONPATH=verl-agent .venv/bin/python verl-agent/examples/data_preprocess/preprocess_search_r1_dataset.py --local_dir envdata/searchr1
```

Retriever: a separate environment with `faiss-gpu`, the e5 index and the wiki-18
corpus, as described under "3. Search" in verl-agent's README. Start it before any
run and leave it up; the launcher expects `http://127.0.0.1:8000/retrieve`
(`--set search_url=...` to change). It holds GPU memory of its own, so keep it off
the training devices or lower `gpu_mem_util`.

Check the host before preparing. The second file drives the real environment
against a stub retriever, so it needs neither the index nor a GPU:

```bash
.venv/bin/python tests/test_search_protocol.py
```

```bash
.venv/bin/python tests/test_search_r1_ccpo.py
```

## Prepare and launch

```bash
python3 scripts/prepare_search_r1_ccpo.py --backbone 7b --gpus 0,1,2,3,4,5,6,7
```

That writes four `experiments/search-*-<date>/` folders with `config.json`,
`run.sh`, `PREPARED.json` and `NOTES.md`, and starts nothing. Launch each with
`bash experiments/<folder>/run.sh`.

Budget and resources are the caller's choice and are recorded as departures from
the protocol:

- `--steps N` shortens the run. All arms must use the same value.
- `--backbone 3b` or `1.5b`. GiGPO's published Search-R1 numbers are 3B and 7B.
- `--seed N` sets `env.seed` and names the folder. Question order comes from the
  dataloader, so replicates differ through sampling only.
- `--set key=value` for anything else, such as `gpu_mem_util` or `test_freq`.
  Estimator keys (`ccpo_*`) are refused.

Validation covers the whole file every 50 steps, and the full test split is 51,713
questions. For periodic validation, write a per-dataset subset and score the final
checkpoint on the full file:

```bash
python3 scripts/subset_search_val.py --per-source 500
python3 scripts/prepare_search_r1_ccpo.py --backbone 7b --gpus 0,1,2,3,4,5,6,7 --val-files envdata/searchr1/test_500_per_source.parquet
```

## What to report

- Exact match overall and per dataset: `val/success_rate` and
  `val/<dataset>_success_rate` in `outputs/metrics.jsonl`.
- Observation recurrence for the CCPO arms: `progress_current_exact_frac` and
  `progress_current_backoff_frac` in `outputs/future_progress/step-*.metrics.json`,
  the share of turns with an exact peer and on the task fallback. The paper reports
  about 8% fallback on ALFWorld and WebShop late in training; this is the
  comparable number.

## Known limits

- Context statistics are thermometer-coded against 30 turns and 25 distinct
  observations. With at most four turns the turn and count codes barely move, so
  the context block separates peers mainly by the revisit bit and the novelty
  ratio. Left unchanged so the method is the one in the paper.
- Similarity grouping is not available to the main method: the future-progress
  path requires exact observation groups. GiGPO's 0.9 grouping applies to the
  GiGPO arm only.
- Resuming skips the env-draw alignment used on ALFWorld (questions come from the
  dataloader). Resume has not been exercised on this benchmark.
- Hidden-state snapshots (`.npz`) are written every 50 steps instead of every step;
  per-step CSV and metrics are unchanged.
