# Handoff — 2026-09-09

State after a server move. Everything below is committed; nothing lives only in a
running process. **The move destroyed every checkpoint** — see the next section. Full reasoning for every entry is in `experiments/hypothesis.md`.

## RESULT — H-AD is NULL, and what it leaves

**FINAL (2026-09-10), paired:** the same-hardware comparator `ccpo-global-fa` finished at a
converged mean of **74.9%** against the anchor arm's **73.0%** (-1.9, t -0.8). Two runs of
the *same* config (`global-fa` vs Blackwell) differ by +1.8 converged and by up to 21 points
at matched steps, so the anchor effect is inside run-to-run noise. **H-AD is null.** The
cross-hardware figures below are superseded by this paired result.


`ccpo-anchor-2gpu-20260909` ran 100 steps with both anchor flags. **Twenty paired
evaluations vs `ccpo-global-20260907`: mean +1.72 pts, sd 6.36, SE 1.42.** Not
significant. Converged windows: **-0.11** (steps>=70) and **+1.56** (steps>=80);
trimmed of the base arm's three erratic draws, **+0.78**. Step-100 endpoint 82.8% vs
79.7%.

**The anchor repair does not change held-out accuracy** — despite `n_buckets` +47%,
mean node size 8.44 -> 5.51 and singletons 0.298 -> ~0.40. Second time a large grouping
change has bought nothing (`ccpo-hardedge` was the first).

**82.8% is not a result.** Single 128-game draw, SE +/-3.3, and the run maximum. The
series ran 60.9 / 76.6 / 70.3 / 78.9 / 79.7 / 82.8 over its last six evaluations.

**All of it is cross-hardware** (base: 4x Blackwell/Triton; this arm: 2x A100/FLASH_ATTN).
`ccpo-global-fa-20260909` is now running the base config on the same GPUs and backend —
the first properly paired comparison this project will have, and the second seed Open #4
has wanted since the checkpoints were destroyed. **Treat H-AD's verdict as provisional
until it finishes (~10 h).**

**What this leaves:** config, eval protocol, split, training length, context
conditioning, memory and now the anchor are all excluded. The estimator itself is what
remains, so **H-AB (G2PO on our harness) is the only way left to attribute the gap.**

## SERVER MOVE 2026-09-09 — read before trusting anything below

We are on a **new box**. The container rebuild everyone planned did **not** happen; a
host move did. Four things below this section are now wrong.

**1. Hardware changed: 4x A100-SXM4-80GB (sm_80), not 6x Blackwell (sm_120).**
`scripts/setup_env.sh` needed **no** changes — python 3.10 matches the cp310 flash-attn
wheel, and flash-attn 2.8.3 has native sm_80 kernels (A100 is its best-supported arch),
so its Blackwell prose is stale but harmless. Built clean: torch 2.8.0+cu128,
vllm 0.11.0, transformers 4.57.1, ray 2.50.0, flash-attn 2.8.3.post1.
`VLLM_ATTENTION_BACKEND=TRITON_ATTN` is still hardcoded in `exp_run.py` and justified
there as a Blackwell choice — **on A100 the FLASH_ATTN backend is likely faster and is
untested here.** Left alone deliberately: it keeps the rollout engine matched to the
79.7% arm. Revisit only if step time is the binding constraint.

**2. Only TWO GPUs are ours.** GPUs 0 and 1 carry another tenant's live job (18.4 GB
and 17.4 GB pinned, utilisation bursting to ~50% over a 15 s sample). GPUs 2-3 are
idle and are what we took. Do not grab 0-1 — that is the "do not interfere with other
users" case, and this box has only 4 cards total.

**3. Nothing outside git survived the move.** No `.venv`, no Qwen2.5-1.5B weights, no
ALFWorld data, and **no checkpoints anywhere on disk**. Consequence:
**Open item #4's checkpoint re-scoring is dead** — `step100-last`, `step100-best`,
`step105-best`, `step145-last` no longer exist, so `scripts/eval_checkpoints.sh` has
nothing to score. Re-scoring is only possible by retraining. The second-seed check is
unaffected.

**4. Container limits on THIS box.** `pids.max` is **8,192** and `memory.max`
**256 GiB**. The "rebuild to 32768" plan in the section below belonged to the *old*
server — it does not describe anything pending here. Host RAM is 1,607 GB available.

**The 2-GPU pid estimate is confirmed.** The handoff estimated ~6,500-7,000 pids for a
2-GPU arm. Measured with the arm below running: **7,075** against the 8,192 cap. The
estimate was good, and the margin is ~14% — do not start anything alongside this arm.

Minor: the raylet logs `/tmp/ray_acg ... is over 95% full` every 10 s. It is a
percentage trigger on a 10 TB filesystem with 459 GB free; object spilling has room.
Noise, not a fault.

## Correction: what `ccpo-anchor-20260909` actually ran

The text under H-AD below says that arm "ran `obs_repair=1` alone". **That describes
only its first launch.** Git shows two:

* `e58532a` — launched with `ACG_OBS_REPAIR=1` alone.
* `9a3bd6b` — **step-1 diagnostics recorded from that obs_repair-only launch**
  (`n_buckets` 797, `bucket_singleton_frac` 0.227, `success_rate` 0.0547).
* `35ba3a2` — **relaunched with BOTH** `ACG_OBS_REPAIR=1` and `ACG_ANCHOR_AFF=1`,
  then stopped before step 1 completed.

So the recommendation below (relaunch with both flags) was already acted on, and the
`run.sh` in that directory already carries both. **But the step-1 table below is an
obs_repair-only measurement, not a both-flags one** — the singleton drop from 0.335 to
0.227 is attributable to repair (B) alone. The both-flags relaunch truncated
`metrics.jsonl` back to empty, which is why that directory looks bare; the step-1 row
is preserved in git at `9a3bd6b` and nowhere else.

## The live arm: `ccpo-anchor-2gpu-20260909`

Launched 2026-09-09 on **GPUs 2,3**, fresh (no checkpoint existed to resume from).
Both anchor flags on. `tests/test_obs_repair.py` passed 9/9 against G2PO's transcribed
loop, plus the live-path stub, before launch.

Config diff against the stopped `ccpo-anchor-20260909` `run.sh` is **exactly two lines**:
`trainer.experiment_name` and `trainer.n_gpus_per_node=4` -> `2`. Every ACG_* flag,
batch size, and hyperparameter is identical.

```
python3 scripts/exp_run.py --name ccpo-anchor-2gpu --arm ccpo \
  --set gpus=2,3 --set obs_repair=1 --set anchor_aff=1 \
  --set total_epochs=100 --set compact_budget=0 --set ccpo_gate=global \
  --set ccpo_phi=hidden+ctx --set ccpo_edge_w=1.0 --set ccpo_rho=0.59 \
  --set ccpo_tau=0.15 --set ccpo_target=nextnode --set ccpo_shrink=eb \
  --set ccpo_whiten=3 --set early_stop_min_steps=40 --set early_stop_patience=8
```

**Kill rule is unchanged and still applies:** run to 50 unless held-out at step 20 is
below ~11% (2 SE under base's 21.1%); judge at 50 vs base's 50.0%, at 100 vs 79.7%.
Judge on held-out, **not** on grouping metrics — `ccpo-hardedge` moved `effect_rel` a
hundredfold and held-out by -0.01.

### Step 1 (both flags): `anchor_aff` REVERSES the singleton gain

560 s/step, ETA ~15.4 h for 100 steps — better than the ~25 h the 2-GPU estimate feared.

| metric | base (`ccpo-global`) | +obs_repair only | +BOTH flags |
|---|---|---|---|
| `n_buckets` | 758 | 797 | **1,149** |
| `bucket_singleton_frac` | 0.335 | **0.227** | **0.339** |
| `bucket_size_mean` | — | 8.03 | 5.35 |
| `bucket_size_p90` | 16 | 20 | 12 |
| `effect_rel` | 0.119 | 0.116 | 0.180 |

Folding admissible actions into the node key splits nodes so much finer that singletons
return to the base level. The two halves pull in **opposite directions**: obs_repair
merges failure turns back onto their true state (singletons 0.335 -> 0.227); anchor_aff
then re-splits on the 42.8% of observations that map to >1 admissible set
(0.227 -> 0.339). Net effect on the fraction of occurrences with no leave-one-out
baseline is roughly nil; what changed is *which* occurrences are pooled.

**This makes the later ablation of the two halves mandatory, not optional.** If this arm
moves held-out, the handoff's earlier reasoning — that the singleton drop was the
mechanism — cannot be the explanation, because the pair does not preserve that drop.

**`bucket_singleton_frac` is a fraction of NODES, not of occurrences.**
`ray_trainer.py:453` computes `(sizes <= 1).mean()` over the node list. A singleton node
holds exactly one occurrence, so the occurrence-level share is
`n_buckets * singleton_frac / (n_buckets * bucket_size_mean)`:

| arm | nodes | mean size | occurrences | singleton nodes | % nodes | **% occurrences** |
|---|---|---|---|---|---|---|
| base `ccpo-global` | 758 | 8.44 | 6,400 | 254 | 33.5% | **4.0%** |
| +obs_repair | 797 | 8.03 | 6,400 | 181 | 22.7% | **2.8%** |
| +both flags | 1,149 | 5.35 | 6,144 | 390 | 33.9% | **6.3%** |

The earlier entry read the 10.8-point node move as "~11% of occurrences". The effect is
real and in the claimed direction but **about nine times smaller than recorded**: 1.2
points of occurrences, not 11. Base and +obs_repair share 6,400 occurrences exactly,
which is the paired-hardware signature; the both-flags row has 6,144 and is not paired.

**The step-1 rows are NOT paired.** `episode/success_rate` is 0.0547 (obs_repair-only,
4 GPUs) vs 0.1094 (both flags, 2 GPUs) and `episode/reward/mean` 0.547 vs 1.094, so the
rollouts differ. **The cause is not sharding** — `tensor_model_parallel_size=1` in both
arms, so every GPU holds a whole copy of the 1.5B model and nothing is split. It is
data-parallel width: verl runs one independent vLLM engine per rank
(`distributed_executor_backend="external_launcher"`), and the 128 rollouts split across
2 ranks instead of 4. Confirmed in the metrics — `global_seqlen/mean` went 898,233 ->
1,846,313 (x2.06) while `perf/total_num_tokens` held at 3.59M -> 3.69M (x1.03): same
total work, half the workers, double each.

Two mechanisms then make identical prompts emit different tokens. (i) **RNG position** —
`SamplingParams` carries no per-request seed (verified from the live log: no `seed` key),
so each engine draws from one generator seeded 0 in scheduling order; the same prompt
sits at a different position in that stream when an engine serves 64 sequences instead
of 32, and training temperature is 1.0. (ii) **Batch-dependent floating point** — larger
per-engine batches select different GEMM/attention tile shapes and reduction orders, and
float addition is not associative, so logits differ in the last bits. Either one flips a
token; a flipped token changes the action, which changes the next observation and every
prompt after it, so a 50-step agentic rollout amplifies it into a different trajectory.

**The task draw is NOT the confound.** `build_alfworld_envs` takes
`(seed, train_batch_size=16, group_n=8)` and has no GPU-count dependence, so both arms
play the same 128 games. That also supports the "arms share their evaluation draw"
property surviving the move — still worth confirming at step 5, but the code says it holds.

7 successes vs 14 out of 128 is ~2.3 sd at p~0.08; it needs no bug to explain. The clean
ON/OFF property the step-1 check relied on last time (`success_rate` identical to 4 dp)
**does not hold across a GPU-count change**, so the grouping deltas above are confounded
by a different rollout batch. The direction of the
`n_buckets`/singleton move is far too large to be batch noise, but do not quote these as
a controlled ablation.

**Two caveats the GPU count introduces, both to check at the first evaluation (step 5):**

* The comparator (79.7%) ran on **4 Blackwell GPUs**; this runs on **2 A100s**. The
  gradient math is unchanged — `train_batch_size=16` divides 2, so only gradient
  accumulation differs — but this is no longer a same-hardware comparison.
* "Arms share their evaluation draw" is asserted below for *fresh* arms. The 128
  validation environments are Ray actors and are **not** per-GPU, so the draw should
  still match; **this is an assumption, not a measurement.** Confirm at step 5 that
  `val/success_rate` sits in the expected band before treating the comparison as paired.

Expect a longer step than the 4-GPU Blackwell figure of **407 s/step** (measured at
`9a3bd6b`). The handoff's ~25 h estimate for 100 steps at 2 GPUs assumed Blackwell, so
treat it as a floor.

## READ FIRST if you are rebuilding the container

**More GPUs did not buy concurrency, and raising `pids.max` alone will not fix that.**
Measured with one 4-GPU arm running:

| limit | one arm uses | two arms need | current ceiling |
|---|---|---|---|
| pids | 7,738 | ~15,500 | **8,192** |
| cgroup memory | 105 GiB anon | ~210 GiB | 256 GiB (fits, tight) |
| **host RAM** | 105 GiB | ~210 GiB | **224 GB available** (was 61 — see below) |

**UPDATE 2026-09-09: host RAM freed up.** It was 61 GB available; something outside this
container released ~160 GB and it is now **224 GB**. Two arms need ~210 GiB anon, so
host RAM is **no longer the blocker** — `pids.max=8192` now is, and that is exactly what
the rebuild fixes.

So a rebuild with `--pids-limit 32768` should genuinely deliver two concurrent 4-GPU
arms. Also consider raising the cgroup `memory.max` from 256 GiB to ~384 GiB: two arms
at ~210 GiB fits, but with little margin.

Recheck `free -g` and `/sys/fs/cgroup/pids.current` before relying on this — the 224 GB
belongs to the host and is not ours to assume.

### Recommended container limits for TWO concurrent 4-GPU arms

Measured, not estimated: one arm peaked at **7,738 pids** and **130 GiB**
(105 anon + 25 file cache); a second Ray cluster pushed pids to **8,133** against the
8,192 cap and died silently one line after `Started a local Ray instance`.

| setting | current | recommended | why |
|---|---|---|---|
| `--pids-limit` | 8,192 | **32,768** | 2 x 7,738 ~ 15,500, with 2x margin |
| `--memory` | 256 GiB | **384 GiB** | 2 x 130 GiB ~ 260 GiB — **exceeds the current 256** |
| `--shm-size` | 64 GB | 64 GB (keep) | stayed at 64 G free with one arm running |
| `--ulimit nofile` | 524,288 | keep | nowhere near binding |

```
docker run --pids-limit 32768 --memory 384g --shm-size 64g \
           --ulimit nofile=524288:524288 --gpus all ...
```

**`memory.max` at 256 GiB is the trap**: it is *just under* what two arms need, so
raising only the pid limit clears the first wall and OOMs at the second.

**Host RAM is not ours to control and moves.** It was 61 GB available, then 224, then
217. Two arms need ~210 GiB anon. **Check `free -g` immediately before launching the
second arm; under ~230 GB available, run them serially instead.**

**Asymmetric split (4 GPUs + 2 GPUs) costs almost the same.** A 2-GPU arm is NOT half a
4-GPU arm: the 128 env workers dominate and are identical either way; only the per-GPU
`WorkerDict` processes scale (~10.5 GB RSS each).

| | 4-GPU arm | 2-GPU arm | total |
|---|---|---|---|
| pids | **7,738** (measured) | ~6,500-7,000 (est) | ~14,300-14,700 |
| anon RAM | **105 GiB** (measured) | ~90 GiB (est) | ~195 GiB |
| cgroup mem incl. cache | **130 GiB** (measured) | ~115 GiB (est) | ~245 GiB |

Same budget as the symmetric case: `--pids-limit 32768 --memory 384g`. The 4-GPU
figures are measured; **the 2-GPU figures are estimates** -- no clean 2-GPU run was ever
obtained (the eval attempt died on the pid cap before stabilising), so treat them
as +/-15%.

`train_batch_size=16` divides 2 cleanly, so a 2-GPU arm stays comparable (gradient
accumulation changes, the math does not); expect ~25 h for 100 steps against ~13 h.

Cheaper alternative if the host will not cooperate: the 128 env workers dominate both
pids and RAM and that footprint barely depends on GPU count, so halving them roughly
halves both — at the cost of a smaller rollout batch, which breaks comparability with
the 79.7% arm. Raising the limits is the better path.

GPUs 4–5 are genuinely idle (4 MiB), so nobody else is training — but someone may hold
that RAM, which bears on the "do not interfere with other users" instruction.

**The better fix is to shrink the per-run footprint, not raise the ceiling.** The 128
env workers dominate *both* pids and RAM, and that footprint is nearly independent of
GPU count — which is exactly why 6 GPUs bought nothing. Halving the env workers roughly
halves both.

**Failure signature to recognise:** a run that dies silently one line after
`Started a local Ray instance`, with no traceback, is pid exhaustion — not a config or
CUDA fault. Check `cat /sys/fs/cgroup/pids.current` *before* launching anything
alongside a running arm.

## Where the numbers stand

| arm | steps | held-out | note |
|---|---|---|---|
| **`ccpo-global-20260907`** | 100 | **79.7%** | **best result; the comparator for everything** |
| `ccpo-global-ext-20260908` | 145 | 79.7% mean | 150-step probe: **+0.00** over step 100 |
| `ccpo-long-20260906` | 100 | 70.3% | hard gate, no edge term |
| `ccpo-localstd-20260908` | 80 (stopped) | 64.1% | coherent standardisation, H-V |
| `ccpo-hardedge-20260907` | 50 (paused) | 41.4% | CCPO ON/OFF ablation |
| `ccpo-gatedmem-20260908` | 20 (killed) | 15.6% @20 | stall-gated digest, H-Y |
| `ccpo-memory-20260907` | 22 (stopped) | 9.4% @20 | ungated digest, H-W |
| `ccpo-cheapmem-20260908` | 20 (stopped) | 16.4% @20 | digest at 192 tok / replace; cost fixed, mechanism null, H-AC |
| `ccpo-anchor-20260909` | 1 (stopped) | — | anchor repair; stopped for the host move, no checkpoint |
| `ccpo-anchor-2gpu-20260909` | 100 | 82.8% @100 | H-AD/H-AE, both anchor flags. **NULL: converged 73.0% vs paired comparator 74.9% (-1.9, t -0.8).** |
| `ccpo-global-fa-20260909` | 100 | 82.0% @100 | base config, 2xA100/FLASH — the paired comparator. **Converged 74.9%; the 79.7 config reproduces.** |
| `ccpo-return-hard-20260910` | **RUNNING** | — | **ladder step 1: hard gate + return + uniform (rho=0), raw anchor. The live arm.** |

Published at the same protocol (Qwen2.5-1.5B, ALFWorld, 100 iters, 3 seeds):
**G²PO 95.0**, GiGPO 86.7 (at *150* iters), HGPO 92.77 (at 160), GRPO 72.8, RLOO 69.7,
PPO 54.4.

**Every number above is a single seed.** Including the headline 79.7%.

## The four results that matter

**1. Our config matches G²PO's published script exactly (H-AB).** Diffed against
`baselines/G2PO/examples/g2po_trainer/run_alfworld.sh`: lr 1e-6, kl_loss_coef 0.01,
low_var_kl, gamma 0.95, `use_invalid_action_penalty` True at coef 0.1, max_steps 50,
group 8, train_batch 16, prompt/response lengths, val temperature 0.4, test_freq 5,
mini/micro batch sizes, and the `eval_in_distribution` split all identical. The
"differing" keys are paths, names, logger, GPU count, TP size, memory utilisation.
`val_batch_size` 128 vs our 64 is *not* real — `test.parquet` is 128 rows in both and
ours is evaluated in two chunks (`val/success_rate` = 51/64 = 102/128 exactly).
**Our held-out numbers are directly comparable to the published ones, and n really is
128.**

**2. CCPO's context conditioning is inert.** `ccpo-hardedge` vs `ccpo-global` is
context conditioning fully OFF (λ=0.011, effect_rel 0.001) vs fully ON (λ=1,
effect_rel 0.119), everything else identical. Mean difference over 9 paired
evaluations: **−0.01 points**. Corroborated four ways — `phi_rel_corr` 0.011
(R² 0.00013), H-H's ICC ceiling of 6.1%, H-Q showing φ-weighting inside a bucket is
*worse* than uniform, and this ablation.

**3. The 79.7% came from G²PO's edge term, not from CCPO.** `edge_w` had defaulted to
0.0 while we were trying to beat G²PO. Turning it on coincided with 70.3 → 79.7.

**4. 150 steps buys nothing (H-Z).** `ccpo-global-ext` warm-started at step 100 and ran
to 145: nine evaluations, mean 79.69, sd 4.97, internal trend −0.14 pts/5 steps. The
mean over 50 extra steps equals the step-100 value to two decimals. GiGPO's 86.7 is a
150-iteration number and we now have one at ~80, so the gap is not training length.

**Taken together, the 15-point gap is excluded from:** configuration, eval protocol,
eval split, training length, context conditioning, memory, and the estimator in the
sense H-O measured (r ≈ 0.90 with G²PO's advantage). See "Open" below.

## Where the 15 points actually are (H-Y')

Pooling `ccpo-global-ext`'s nine converged evaluations:

| task type | mean | SE |
|---|---|---|
| look_at_obj_in_light | 65.6 | 6.8 |
| **pick_two_obj_and_place** | **69.6** | **2.1** |
| pick_cool_then_place_in_recep | 78.2 | 2.6 |
| pick_clean_then_place_in_recep | 80.4 | 2.6 |
| pick_heat_then_place_in_recep | 87.1 | 2.8 |
| pick_and_place | 87.6 | 2.6 |

The gap is **not** spread evenly — two types are already near 87. Reaching 95 overall
needs ~25 points on two types while holding the other four, not +15 everywhere.

Caveat: per-type rates are a **mean of the two batch-level rates**, not a pooled
proportion, so they are not exact proportions over 128.

## Do not quote a `stepN-best` number

Best-checkpoint selection takes the maximum of a noisy series and is biased upward by
~1.5 sd. `ccpo-global-ext`: mean 79.69, sd 4.97, and `best.json` reads 85.9 — almost
exactly the expected maximum of 9 draws. Report the mean over evaluations, or
re-evaluate on a fresh draw. `scripts/eval_checkpoints.sh` does exactly that and is
ready to run (needs pid headroom — see top).

## Measurement corrections that change how to read everything

* **Arms share their evaluation draw.** Validation workers are seeded identically and
  iterate in lockstep, so at step *k* every fresh arm sees the same 128 games.
  Matched-step comparisons are **paired**, delta sd ~5.3 — *not* the ±13
  single-evaluation band.
* **A RESUMED run used to restart its evaluation draw — now FIXED.** Checkpoints hold
  no env state and `make_envs()` runs before `_load_checkpoint()`. `ray_trainer.py` now
  burns `(global_steps/test_freq) * len(val_dataloader)` resets on resume, controlled by
  `ACG_ALIGN_VAL_ON_RESUME` (default 1). **`ccpo-long` and `ccpo-global-ext` predate the
  fix and are offset** from fresh runs. Set it to **0** when comparing checkpoints saved
  at *different* steps, or each is scored on a different draw.
* **The evaluation set is not fixed across steps.** TextWorld shuffles per worker seed
  and iterates, so every validation draws different games.
* **Read per-type from `val/*`, never `episode/*`.** The latter is the 16-task training
  draw.

## Open, ranked

0. **H-AD — G²PO repairs the grouping anchor; we never did. THE BEST LEAD.**
   The full harness diff against the G²PO checkout is now done and yields **exactly one
   substantive difference**: `prompts/alfworld.py` is byte-identical, `rollout_loop.py`
   differs only in `enable_thinking=False` (verified a no-op on Qwen2.5 — the chat
   template never references it, rendered prompt byte-identical), and `env_manager.py`
   contains an anchor repair we lack. `git log -S` shows it was never in our tree, and
   GiGPO's original lacks it too, so it is a G²PO contribution.

   It rewrites **`anchor`**, not the prompt — and `anchor` feeds `anchor_obs` into
   `ccpo_step_advantage`/`derive_context` as the state identity for node grouping.
   Two repairs: **(B)** a failed action no longer advances the anchor (without it every
   failure batch-wide collapses to the single anchor `"Nothing happens."`), and
   **(A)** observations after heat/cool/clean/turn-on are concatenated with their
   predecessor, since ALFWorld does not restate those results.

   Caveat recorded honestly: per-type deficits do **not** support (A) — pattern-affected
   types average 77.8 vs unaffected 78.6. (B) is untested by that comparison and is the
   more plausible half.

   Implemented behind **`ACG_OBS_REPAIR`** (default 0). `tests/test_obs_repair.py`
   transcribes G²PO's loop literally; ours matches **9/9 steps**.

   **LAUNCHED as `ccpo-anchor-20260909`** (GPUs 0–3, 100 steps, ~421 s/step). Config
   diff against the 79.7% arm: `obs_repair` is the ONLY real difference.

   **Step-1 check PASSED — the repair is live and the ablation is clean:**

   | metric | base | anchor |
   |---|---|---|
   | `n_buckets` | 758 | **797** |
   | `bucket_singleton_frac` | 0.335 | **0.227** |
   | `bucket_size_p90` | 16 | 20 |
   | `episode/success_rate` | 0.0547 | **0.0547** (identical → rollouts unchanged) |

   Singletons fell 10.8 points **of NODES, not of occurrences** — see the correction
   below; the occurrence-level move was 4.0% -> 2.8%, about 1.2 points. That is repair (B) — failures now anchor to their true
   state instead of the shared `"Nothing happens."` string. (I had predicted singletons
   would RISE; being wrong is what identified (B) as the dominant half.)

   **Caution that must stay attached:** structural gain ≠ accuracy. `ccpo-hardedge` vs
   `ccpo-global` moved `effect_rel` a hundredfold and moved held-out success by −0.01.
   Judge this arm on held-out at steps 20/50/100, not on grouping metrics.

   **Kill rule:** run to 50 unless held-out at step 20 is below ~11% (2 SE under base's
   21.1%). Then judge at 50 vs base's 50.0%, at 100 vs 79.7%.

   **H-AE — a SECOND anchor difference, found after the arm launched.** G²PO also
   folds the admissible-action list *into* the anchor, so its node identity is
   observation-text + admissible-actions; ours is observation text alone
   (`key = (task, str(anchor_obs[i]))`, core_ccpo.py:409). This tree already carries
   that list as `aff` — added 2026-09-03 because "42.8% of observations map to >1
   admissible set" — but `aff` only ever reaches the diagnostic CSV (core_ccpo.py:795).
   **We measured the ambiguity, built the disambiguator, and never grouped with it.**
   On our own number that leaves 42.8% of observations conflated, a larger population
   than the repair touches. Implemented as `ACG_ANCHOR_AFF`.

   **So the relaunch needs BOTH flags** — `ccpo-anchor-20260909` ran `obs_repair=1`
   alone, which is only half of G²PO's anchor. Ablate the halves later, and only if the
   pair moves the number.

   **Run `/workspace/.venv/bin/python tests/test_obs_repair.py` first** (5 s, no GPU).
   It checks ours against a literal transcription of G²PO's loop AND drives the real
   `AlfWorldEnvironmentManager.reset/step` against a stub env — added after my first
   version of this change parsed cleanly but would have raised `NameError` eight
   minutes into a GPU run.

   **To relaunch after the rebuild (fresh, not resumed — the stopped arm never reached
   its first checkpoint at step 5):**
   ```
   python3 scripts/exp_run.py --name ccpo-anchor --arm ccpo \
     --set gpus=0,1,2,3 --set obs_repair=1 --set anchor_aff=1 \
     --set total_epochs=100 --set compact_budget=0 --set ccpo_gate=global \
     --set ccpo_phi=hidden+ctx --set ccpo_edge_w=1.0 --set ccpo_rho=0.59 \
     --set ccpo_tau=0.15 --set ccpo_target=nextnode --set ccpo_shrink=eb \
     --set ccpo_whiten=3 --set early_stop_min_steps=40 --set early_stop_patience=8
   ```

1. **H-AB — G²PO on our harness.** *Declined twice by the user*, so not run. It is now
   the only remaining way to attribute our 79.7%, because the config is proven
   identical. If G²PO scores ~80 here, our method is competitive and 95.0 does not
   transfer to this stack; if ~95, the gap is in code we can diff line by line, since
   both trees are on disk. Our tree has **no `g2po` estimator** — porting it is
   `baselines/G2PO/g2po/core_g2po.py` (246 lines) plus an enum, a trainer branch, and
   `algorithm.g2po.*` keys. `core_ccpo.py` already vendors their group-aggregation
   machinery.
2. **H-AC — `ccpo-cheapmem`, RUNNING.** Digest at budget 192 in `replace` mode.
   Kill rule fixed in advance: stop at step 20 if train mean over steps 10–20 < 0.130
   (base 0.160) or held-out at 20 < 15%. Continue only on the pre-registered per-type
   concentration, never on overall success alone.
3. **H-S — epistemic weighting.** `w_u = J_u/(J_u+c)` on the step term. **`localstd`'s
   −2.08 does not refute it**: localstd divided by σ and damped *high-variance*
   neighbourhoods, H-S damps *poorly-sampled* ones. They disagree exactly where variance
   and support are both high. Small predicted effect; needs multiple seeds or an offline
   criterion first.
4. **Two cheap checks needing no training run:** re-score `step100-last` /
   `step100-best` / `step105-best` / `step145-last` on one common draw
   (`scripts/eval_checkpoints.sh`), and a second seed of `ccpo-global`.

## Do not re-run without changing something

* Progress banding the gate (H-J) — loses under both targets.
* Observation-derived memory features in φ (H-T route A) — restates `progress`, +0.003 R².
* Hard gate with λ=1 (H-Q) — φ-weighting inside a bucket is 2–10 R² points worse than uniform.
* Per-node standardisation on reliability grounds (H-U) — that evidence was an artifact.
* **The digest as originally built (H-AA).** Paired over 15 steps on a shared seed it
  made responses *longer* at 15/15 (+6.3 tok, p=6e−5) and KL *lower* at 15/15. Its
  stated purpose — sparing the agent from re-deriving state in `<think>` — predicts
  shorter responses and is refuted. Cost driver is the **budget**: 512 tokens against a
  466-token base prompt more than doubles it whenever the digest fires.

## Resuming

Every arm is resumable:

```
python3 scripts/exp_run.py --name X --arm ccpo \
  --set resume_from=experiments/<arm>/outputs/checkpoints/global_step_N \
  --set total_epochs=<ABSOLUTE target>
```

`total_epochs` is absolute, not an increment. Safe here because the LR is constant with
zero warmup. Match the 79.7% arm with:
`ccpo_gate=global ccpo_phi=hidden+ctx ccpo_edge_w=1.0 ccpo_rho=0.59 ccpo_tau=0.15
ccpo_target=nextnode ccpo_shrink=eb ccpo_whiten=3 early_stop_min_steps=40
early_stop_patience=8 total_epochs=100` (defaults differ: `total_epochs` defaults to 20
and `ccpo_target` to `return`).

Eval-only: `--set val_only=1 --set align_val_on_resume=0 --set resume_from=<ckpt>`.

`train_batch_size` must divide the GPU count: 16 works with 1/2/4/8, **not 6**.

**Kill by PID from `outputs/train.pid`, never `pkill -f <pattern>`** — the pattern
matches the inline shell running the kill. This has caused two incidents.

## Method note for this arm

`experiments/ccpo-anchor-2gpu-20260909/docs/method.html` — architecture, the three
repairs with figures, the 9-step anchor demo, the refinement argument, and the
confound table. Written in the house style of `docs/method.html`.
