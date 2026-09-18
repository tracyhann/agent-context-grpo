# baseline-repo — agentic-RL baselines on ALFWorld and WebShop

**Destinations**

| What | Where |
|---|---|
| This repo | https://github.com/tracyhann/agent-context-grpo — branch **`baselines`** |
| Checkpoints | https://huggingface.co/tracyhan816/ccpo-variants — one folder per run |
| 1.5B backbone | https://huggingface.co/latent-artist/66c200392ff148279a5995ef1455d3de |
| 7B backbone | https://huggingface.co/latent-artist/f142c54d3ea54565a5ce95f881497e41 |

Run outputs (`runs/`) are **not** committed — a single 7B checkpoint is ~81 GB. Push
weights to the HuggingFace repo above instead, keeping the local layout:
`<run-name>/<checkpoint-name>/actor/...` plus `data.pt`. Upload both the FSDP shards (for
resuming) and a merged `huggingface/` copy (for loading on any GPU count).

Reproducible launch scripts, environment setup and logging for four baseline methods,
on two benchmarks, at 150 steps each. Every host-specific deviation is collected in one
file (`scripts/common.sh`) so no two runs differ by accident.

## The matrix

| Method | 7B | Tree | Entrypoint |
|---|---|---|---|
| GRPO  | ✅ | `baselines/verl-agent` | `verl.trainer.main_ppo` |
| GiGPO | ✅ | `baselines/verl-agent` | `verl.trainer.main_ppo` |
| HGPO  | ✅ | `baselines/verl-agent` | `recipe.hgpo.main_hgpo` |
| G2PO  | ✅ | `baselines/G2PO`       | `verl.trainer.main_ppo` |

Four configurations × two benchmarks (ALFWorld, WebShop) = **8 runs**, 150 steps each.

**1.5B is out of scope here** — those runs are being carried out in the main workspace
(`/workspace/experiments`), not through this repo. `train.sh` still accepts `1.5b` for
debugging, but `run_all.sh` and the matrix cover 7B only, so nothing is duplicated.
**Backbones (required)** — blinded re-uploads, resolved automatically by `train.sh`:

| | Repo |
|---|---|
| 1.5B | `latent-artist/66c200392ff148279a5995ef1455d3de` |
| 7B | `latent-artist/f142c54d3ea54565a5ce95f881497e41` |

Both are public and architecturally identical to the 1.5B/7B instruct models. Override
with `MODEL_1_5B` / `MODEL_7B` only for debugging. Their `config.json` still declares
`Qwen2ForCausalLM`, so the masking is repo-name level, not artifact level.

## How many experiments

| | Count |
|---|---|
| Training runs | **8** — 4 methods × 2 benchmarks, 7B, 150 steps each |
| Held-out evaluations | **40** — each finished run scored on 5 seeds (997 / 101 / 3173 / 869 / 2917) |

A 150-step 7B run takes roughly 12–24 h here depending on the neighbour's GPU load; each
evaluation is ~10–20 min. The 8 runs are serialized (one fits the host at a time), so the
full matrix is on the order of a week of wall-clock.

## Workflow, step by step

**1. Set up the benchmark.** ALFWorld works already; WebShop does not exist yet.

```bash
env/setup_alfworld.sh                     # data + parquets (already done on this host)
env/setup_webshop.sh                      # READ ITS HEADER FIRST — separate venv, Java 11
```

**2. Check the backbones are cached.** Both are required and resolved automatically:

```bash
du -sh /workspace/hf/hub/models--latent-artist--*      # expect ~2.9G (1.5B) and ~15G (7B)
```

**3. Dry-run the config you intend to launch.** Prints the assembled command and exits.

```bash
DRY_RUN=1 scripts/train.sh hgpo 7b alfworld
# [run] tree=... entry=recipe.hgpo.main_hgpo prompt=4096 trunc=left max_steps=50 mini=256 micro=8
```

Confirm `mini` is 256 on ALFWorld and 64 on WebShop, and that `entry` matches the method.

**4. Launch.** One run, or the whole matrix serialized:

```bash
scripts/train.sh hgpo 7b alfworld 0,1,2,3 150     # one
scripts/run_all.sh                                # all 8, waits for idle GPUs between
```

**Launch detached, never from inside a monitor or watcher** — a run started that way dies
silently when its parent session ends. `run_all.sh` already uses `setsid`. After launching
by hand, verify the trainer's parent is PID 1:

```bash
ps -o pid,ppid,sid -p $(cat runs/<run>/outputs/train.pid)
```

**5. Start the metrics mirror** (`run_all.sh` does this for you):

```bash
logging/mirror_metrics.sh runs/<run> /tmp/ray_<method>_<size>_<bench> <pid> &
```

Without it there are no metrics at all — verl prints them inside the ray actor, and `/tmp`
is volatile. Never delete a ray tmpdir before mirroring.

**6. Watch it.** Progress and held-out scores land in `runs/<run>/outputs/metrics.jsonl`:

```bash
python -c "import json;[print(json.loads(l)['step'], round(100*json.loads(l).get('val/success_rate',0),1))
           for l in open('runs/<run>/outputs/metrics.jsonl') if 'val/success_rate' in l]"
logging/plot_metrics.py --exp runs/<run>
```

**7. Pin the checkpoints you care about**, before rotation deletes them:

```bash
logging/pin_checkpoint.sh runs/<run>/outputs/checkpoints 100 step100-budget
```

**8. Score the final checkpoint on five seeds.** This is the number you report:

```bash
scripts/evaluate.sh hgpo 7b alfworld runs/<run>/outputs/checkpoints/global_step_150 0,1,2,3
```

It prints a per-seed table and the mean ± std. Seeds default to
`997 101 3173 869 2917`; override with `SEEDS="..."` (e.g. the original three, to
reproduce an earlier number exactly). If you cannot supply the run's original GPU
count, merge the checkpoint first — the script tells you so and refuses rather than
producing a wrong number.

**9. Record it** in `RESULTS.md` as `mean ± std`, with the step count beside it. Never
quote a single evaluation: see §3 of `experiments/experiments.md` for why.

Runs land in `runs/<method>-<size>-<bench>-<date>/outputs/`.

## Environment setup

| Benchmark | Script | State |
|---|---|---|
| ALFWorld | `env/setup_alfworld.sh` | **working** — data in `/workspace/alfworld_data`, parquets in `/workspace/envdata` |
| WebShop  | `env/setup_webshop.sh`  | **not yet installed** — read the header before running |

WebShop is the harder one and has three traps, all documented in that script:

1. **It needs its own virtualenv.** Its `requirements.txt` pins torch 2.6 / transformers
   4.51 / numpy 1.26; the shared `.venv` runs torch 2.8 and every trainer uses it.
   Installing the pins into the shared venv would break running jobs and probably drop
   Blackwell (sm_120) support, which needs torch ≥ 2.7.
2. **It needs Java 11** for pyserini's Lucene index, and there is no conda on this host.
3. **Comparability.** Earlier work here used a BM25 retriever instead of pyserini;
   `docs/method.html` records that those numbers are not comparable to published WebShop
   results. Also check the catalogue: verl-agent's copy defaults to the 1,000-product
   files, not the full ~1.18M.

## What differs by benchmark

Taken from the upstream scripts and left unchanged:

| | ALFWorld | WebShop |
|---|---|---|
| `env.env_name` | `alfworld/AlfredTWEnv` | `Webshop` |
| `env.max_steps` | 50 | 15 (HGPO: 30) |
| `max_prompt_length` | 2048 (HGPO: 4096) | 4096 |
| `truncation` | `error` (HGPO: `left`) | `error` (HGPO: `left`) |
| micro-batch (7B) | 8 | 2 |

Shared: `train_batch_size=16`, `group_size=8`, `val_data_size=128`, `history_length=2`,
`lr=1e-6`, `gamma=0.95`, KL 0.01 `low_var_kl`, invalid-action penalty 0.1, val T=0.4.

## Host deviations (all in `scripts/common.sh`)

These are forced by this machine, not choices. Each is annotated in the file.

| Setting | Upstream | Here | Why |
|---|---|---|---|
| `VLLM_ATTENTION_BACKEND` | XFORMERS | `TRITON_ATTN` | no sm_120 xformers kernel |
| `gpu_memory_utilization` | 0.6 | **0.3** | at 0.6 a 1.5B run allocated ~99 GB on a 96 GB card and died in vLLM's `cumem` wake_up |
| `data.val_batch_size` | 128 | **64** | one ray actor per val env; 128 train + 128 val exceed the 8192-pid ceiling. Same 128 games, two chunks |
| `ray_init.num_cpus` | unset | **64** | Ray sizes from nproc (256) and prestarts a worker per CPU |
| `tensor_model_parallel_size` | 2–4 | **1** | TP shards the model; with N GPUs, TP=1 gives N data-parallel ranks |
| `default_local_dir` | relative | **explicit** | several upstream scripts cd into their own tree and write checkpoints inside the baselines checkout |
| logger | wandb | **console** | metrics then land in the ray WORKER log — see Logging |

## Logging

verl's console backend prints inside the ray actor, so **metrics never reach the driver's
`train.log`** — only the tqdm bar does. `logging/mirror_metrics.sh` copies the worker log
into the run directory and parses it:

```bash
logging/mirror_metrics.sh <run_dir> <ray_tmpdir> [pid]   # loops every 5 min while pid lives
```

Produces `outputs/worker_metrics.log` and `outputs/metrics.jsonl`, then plots via
`logging/plot_metrics.py --exp <run_dir>`.

`/tmp` is volatile and ray sessions are deleted on cleanup — **mirror before you delete a
ray tmpdir**, or the run's metrics are unrecoverable.

Checkpoint pinning: `logging/pin_checkpoint.sh <ckpt_dir> <step> <name>` hard-links a
checkpoint so the trainer's rotation cannot delete it (hard links cost no disk until the
trainer removes its own copy).

## Gotchas learned the hard way

- **Resume needs the original GPU count.** FSDP shards are saved as
  `model_world_size_<N>_rank_*.pt` and only load on N ranks. A 2-GPU run cannot resume on
  4. To evaluate on a different GPU count, merge first:
  `verl-agent/scripts/model_merger.py merge --backend fsdp --local_dir <ckpt>/actor --target_dir <out>`.
- **Never launch a run from inside a monitor/watcher.** A run started that way dies with
  no error in its log when the parent session is torn down. Launch with `setsid` from a
  plain shell and verify the process's parent is PID 1.
- **Rotation does not know about checkpoints from a previous attempt.** After a resume,
  `max_actor_ckpt_to_keep` only counts saves made by the *current* process, so the
  checkpoint you resumed from is orphaned and never reclaimed. Delete it by hand.
- **Evaluation noise is large.** A single 128-game evaluation at T=0.4 varies by several
  points; the same frozen checkpoint scored 83.6 / 86.7 / 89.1 across three seeds. Quote a
  multi-seed mean ± std, or a window mean over the last few evaluations — never one point.
- **A shared host is the main failure mode.** Another tenant filling GPU memory mid-run
  killed a 7B run at step 37. `run_all.sh` waits for GPUs to be idle on memory *and*
  utilization for three consecutive minutes before launching.

## Reference results

`RESULTS.md` holds the numbers already measured on this host, with 3-seed error bars where
available, and the published targets. `experiments/experiments.md` is the experiment
protocol: matrix, reporting rules, metric inventory and hyperparameters.
`related-work/` holds the four papers these methods come from.
