# Handoff — 2026-09-09

State for a container rebuild. Everything below is committed; nothing lives only in a
running process. Full reasoning for every entry is in `experiments/hypothesis.md`.

## READ FIRST if you are rebuilding the container

**More GPUs did not buy concurrency, and raising `pids.max` alone will not fix that.**
Measured with one 4-GPU arm running:

| limit | one arm uses | two arms need | current ceiling |
|---|---|---|---|
| pids | 7,738 | ~15,500 | **8,192** |
| cgroup memory | 105 GiB anon | ~210 GiB | 256 GiB (fits, tight) |
| **host RAM** | 105 GiB | ~210 GiB | **61 GB available** |

Raise `--pids-limit` to 32768, but **host RAM is the binding constraint**: 755 GB total
with only ~61 GB free, and ~500 GB is allocated outside this container and invisible
from inside it (`free` reports host-wide; `ps` shows only our processes). A second arm
needs ~105 GiB anon and would OOM before it hit any pid limit.

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
| `ccpo-anchor-20260909` | **RUNNING** | — | **G²PO anchor repair, H-AD. The live arm.** |

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

   Singletons fell 10.8 points: ~11% of occurrences moved from having NO leave-one-out
   baseline to having neighbours. That is repair (B) — failures now anchor to their true
   state instead of the shared `"Nothing happens."` string. (I had predicted singletons
   would RISE; being wrong is what identified (B) as the dominant half.)

   **Caution that must stay attached:** structural gain ≠ accuracy. `ccpo-hardedge` vs
   `ccpo-global` moved `effect_rel` a hundredfold and moved held-out success by −0.01.
   Judge this arm on held-out at steps 20/50/100, not on grouping metrics.

   **Kill rule:** run to 50 unless held-out at step 20 is below ~11% (2 SE under base's
   21.1%). Then judge at 50 vs base's 50.0%, at 100 vs 79.7%.

   **To resume after the rebuild:**
   ```
   python3 scripts/exp_run.py --name ccpo-anchor2 --arm ccpo \
     --set resume_from=experiments/ccpo-anchor-20260909/outputs/checkpoints/global_step_N \
     --set total_epochs=100 --set obs_repair=1 --set ccpo_gate=global \
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
