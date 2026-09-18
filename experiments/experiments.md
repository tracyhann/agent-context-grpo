# Baseline experiments

Eight runs: **four methods × two benchmarks**, 7B backbone, 150 steps each.

**1.5B is deliberately excluded.** Those arms are being run in the main workspace
(`/workspace/experiments`) rather than through this repo; measured values are carried in
§2 for comparison only. These are
reproductions of published baselines, not new methods — every arm runs its authors' own
estimator, and the only departures from the upstream scripts are the host deviations in
`../scripts/common.sh`, each annotated there with the failure that forced it.

| Method | Size | Defined in |
|---|---|---|
| GRPO  | 7B | `../related-work/grpo_2402.03300.pdf` |
| GiGPO | 7B | `../related-work/gigpo_2505.10978.pdf` |
| HGPO  | 7B | `../related-work/hgpo_2602.22817.pdf` |
| G2PO  | 7B | `../related-work/g2po_2606.22995.pdf` |

Benchmarks: **ALFWorld** and **WebShop**.

---

## 0. Protocol — identical in every run

| | |
|---|---|
| steps | **150** (`total_epochs=150`, early stopping off) |
| seed | 0 for training; held-out scoring uses **997 / 101 / 3173 / 869 / 2917** (5 seeds) |
| eval | every 5 steps (`test_freq=5`), T=0.4 with sampling, 128 games |
| checkpoints | every 20 steps, one rolling copy (`keep=1`) plus pinned `step100-budget` and `stepbest` |
| group / batch | `env.rollout.n=8`, `train_batch_size=16` → 128 episodes per step |
| mini-batch | **256 on ALFWorld, 64 on WebShop** (upstream value; sets gradient updates per batch) |
| history | `history_length=2` (K=2) |
| optimiser | lr 1e-6, KL 0.01 `low_var_kl`, γ 0.95, invalid-action penalty 0.1 |

**Backbones — required.** Not the public Qwen repos:

| | Repo |
|---|---|
| 7B (used here) | `latent-artist/f142c54d3ea54565a5ce95f881497e41` |
| 1.5B (main workspace) | `latent-artist/66c200392ff148279a5995ef1455d3de` |

Both are public, architecturally identical to the 1.5B/7B instruct models (1.5B: hidden
1536, 28 layers, vocab 151936; 7B: hidden 3584, 28 layers, vocab 152064). `train.sh`
resolves them automatically; override with `MODEL_1_5B` / `MODEL_7B` only for debugging.
Note their `config.json` still declares `Qwen2ForCausalLM`, so the masking is at the
repo-name level — anyone loading the artifact can identify the family.

---

## 1. What separates the four estimators

All four optimise the same policy on the same rollouts. They differ only in how an
advantage is assigned to a step.

**GRPO** — trajectory-level only. One advantage per rollout, standardised across the group
of rollouts sharing a prompt; every token of that rollout gets it. No step term.

**GiGPO** — episode advantage **plus** a step advantage, where steps are grouped by the
*current observation*. Combined as `A = A_ep + w · A_step`, `w = step_advantage_w = 1.0`.

**G2PO** — same two-term shape, but the step-grouping key is an anchor/graph over states
rather than the raw observation, so two steps group together only if their anchors match.

**HGPO** — **step-level only**, and that term is a hierarchy. Each step joins one group per
history depth `k = 1 … K+1`, keyed on its last `k` observations in order; every group with
≥2 members standardises the γ-discounted return-to-go; the per-depth advantages are then
combined with a length weight `w_k ∝ (k+1)^α` (α = 1.0, so depths 1/2/3 weigh 2:3:4).
There is **no separate episode term** — `base_group=False` in every published ALFWorld
script — so the trajectory signal reaches a step only through the discounted return
`R_t = r_t + γ·R_{t+1}`. With γ=0.95 over 50 turns, an early step in a successful episode
sees ~8% of the terminal reward.

---

## 2. Run matrix

| # | Method | Benchmark | GPUs | Status |
|---|---|---|---|---|
| B1 | GRPO  | ALFWorld | 4 | to run |
| B2 | GiGPO | ALFWorld | 4 | to run |
| B3 | HGPO  | ALFWorld | 4 | to run |
| B4 | G2PO  | ALFWorld | 4 | to run |
| B5 | GRPO  | WebShop  | 4 | to run — install the environment first |
| B6 | GiGPO | WebShop  | 4 | to run — install the environment first |
| B7 | HGPO  | WebShop  | 4 | to run — install the environment first |
| B8 | G2PO  | WebShop  | 4 | to run — install the environment first |

**Nothing here has been run yet.** This repo is the procedure; fill the status column in as
runs complete on this machine. Prior numbers from a *different* host appear in
`../RESULTS.md` strictly as sanity checks — they are not results of this repo, were
produced at different step budgets (160/150/100, not 150), and must never be copied into a
results table as if they came from here.

Launch: `../scripts/train.sh <method> <size> <benchmark> [gpus] [steps]`, or
`../scripts/run_all.sh` for the whole matrix, serialized.

---

## 3. Reporting

**Never quote a single evaluation.** A 128-game evaluation at T=0.4 varies by several
points, and the same *frozen* checkpoint scored 83.6 / 86.7 / 89.1 across the first three
seeds — a 5.5-point spread with no training variance at all. Five seeds are used because
three leave the mean itself uncertain by roughly ±1.6 points on a run with that spread. Every headline number is either

- **5-seed mean ± std** on the final checkpoint (997 / 101 / 3173 / 869 / 2917), or
- a **window mean** over the last four evaluations of the run.

The cost of ignoring this is on record: HGPO 7B's single step-160 evaluation read 92.2,
while the 3-seed mean of that same checkpoint is **96.09 ± 0.78**. The endpoint understated
the run by 3.9 points and would have hidden that it beats its published figure.

**State the budget.** Endpoints are not comparable across methods unless the step count
matches. At 100 steps our 1.5B runs read G2PO 91.4, HGPO 79.7, GiGPO 76.6; by 160 steps
HGPO's 11.7-point deficit to G2PO has closed entirely.

**Per-task columns are mostly noise.** Each of Pick / Look / Clean / Heat / Cool / Pick2
covers a handful of the 128 games and swings ±10 between seeds while `All` moves ±1–3.
Print them with that caveat or not at all.

### ALFWorld table

```
| Method | Size | Steps | Pick | Look | Clean | Heat | Cool | Pick2 | All (3-seed) |
```

### WebShop table

WebShop reports **task score** and **success rate**; quote both, and state the catalogue
(verl-agent's copy defaults to the 1,000-product files) and the retriever — this repo's
earlier WebShop work used BM25 rather than pyserini, which makes those numbers
incomparable to published ones.

---

## 4. Metrics logged

verl's console backend prints inside the ray actor, so **nothing lands in the driver's
`train.log` except the tqdm bar**. `../logging/mirror_metrics.sh` copies the ray worker log
into the run directory every 5 minutes and parses it to `outputs/metrics.jsonl`.

Logged per training step:

| Group | Keys |
|---|---|
| outcome | `episode/success_rate`, `episode/reward/{mean,max,min}`, per-task `episode/<task>_success_rate` |
| behaviour | `episode/length/{mean,max,min}`, `episode/valid_action_ratio`, `response_length/*`, `prompt_length/*` |
| optimisation | `actor/{pg_loss,kl_loss,entropy_loss,grad_norm,pg_clipfrac,ppo_kl,lr}` |
| advantages | `critic/advantages/{mean,max,min}`, `critic/returns/*`, `critic/score/*` |
| throughput | `timing_s/{step,gen,ref,old_log_prob,update_actor,adv}`, `perf/{throughput,max_memory_allocated_gb,cpu_memory_used_gb}` |
| held-out (every 5 steps) | `val/success_rate`, per-task `val/<task>_success_rate`, `val/text/test_score` |

**Not logged by the upstream trees:** `val/episode_length` — turns per held-out episode.
That metric exists only in this repo's patched `verl-agent` overlay
(`/workspace/verl-agent`), not in `baselines/verl-agent` where HGPO's `recipe/hgpo` lives.
To report turns, either evaluate through the patched overlay from merged weights, or port
the six-line patch (collect `episode_lengths` / `episode_rewards` per `traj_uid` in
`_validate`, take one value per trajectory, and emit mean overall and mean over successes).

**Mirror before deleting a ray tmpdir.** `/tmp` is volatile and the worker log is the only
copy of the metrics; deleting a ray session before mirroring loses them irrecoverably.

---

## 5. Training hyperparameters

Identical across methods except where the upstream scripts differ.

All 7B. (`train.sh` still carries the 1.5B micro-batches — 32 ALFWorld / 8 WebShop — for
debugging runs.)

### 5.1 ALFWorld

| | 7B |
|---|---|
| `max_prompt_length` | 2048 (HGPO 4096) |
| `truncation` | `error` (HGPO `left`) |
| `env.max_steps` | 50 |
| `ppo_mini_batch_size` | 256 |
| `ppo_micro_batch_size_per_gpu` | 8 |
| `gpu_memory_utilization` | 0.3 |
| offloading | none (96 GB cards hold ~30 GB of FSDP state per rank) |

### 5.2 WebShop

| | 7B |
|---|---|
| `max_prompt_length` | 4096 |
| `env.max_steps` | 15 (HGPO 30) |
| `ppo_mini_batch_size` | **64** (not 256 — differs from ALFWorld in every upstream script) |
| `ppo_micro_batch_size_per_gpu` | 2 |

---

## 6. Checkpoints, disk and GPUs

- **A checkpoint only loads on the GPU count that wrote it** — shards are
  `model_world_size_<N>_rank_*.pt`. To evaluate on a different count, merge first with
  `verl-agent/scripts/model_merger.py merge --backend fsdp`.
- **Sizes**: 1.5B ≈ 19 GB per checkpoint, 7B ≈ 81 GB, mostly optimizer state. Merged
  inference-only weights are 3.4 GB and 15 GB.
- **After a resume, rotation forgets the previous attempt's checkpoints**, so the one you
  resumed from is orphaned and never reclaimed. Delete it by hand.
- **Pin what matters**: `../logging/pin_checkpoint.sh <ckpt_dir> <step> <name>` hard-links a
  checkpoint so rotation cannot remove it, at no disk cost until the trainer deletes its own.
- **One run at a time.** A run uses ~7,500 of this container's 8,192 pids and ~200 GB of
  host RAM; two cannot coexist.
- **This host is shared.** A neighbouring tenant taking GPU memory mid-run killed a 7B run
  at step 37 inside vLLM's `cumem` wake-up. `run_all.sh` waits for the GPUs to be idle on
  memory *and* utilization for three consecutive minutes before launching.
- **Never launch from inside a monitor or watcher.** A run started that way dies with no
  error in its log when the parent session is torn down — this cost a G2PO continuation at
  step 116. Launch with `setsid` from a plain shell and check the process's parent is PID 1.
