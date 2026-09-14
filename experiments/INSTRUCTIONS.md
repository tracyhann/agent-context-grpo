# Instructions — `attncred` across two backbones and two benchmarks

Four cells: {Qwen2.5-1.5B-Instruct, Qwen2.5-7B-Instruct} x {ALFWorld, WebShop}, each
150 steps, each keeping the step-100, best and last checkpoints.

**What the arm is.** `attncred` = the hard-gate return arm plus three flags: the
phi-attention readout with `lam` pinned to 1 (`ccpo_lam_fix=1.0`), the credibility
prior `lam_k = J/(J+2)` (`ccpo_prior_kappa=2.0`), and the J=0 task-level fallback
(`ccpo_backoff_task=1`). On ALFWorld at 1.5B over 100 steps it is a **null**: window
70-100 75.56 vs the control's 76.34 (t -0.76), while `effect_rel` ran 0.24-0.35 — the
estimator moved a quarter to a third of the step credit and held-out did not follow.
See `hypothesis.md` H-AL and `ccpo-attncred-20260912/docs/method.html`.

These four cells ask whether that null is specific to *ALFWorld at 1.5B over 100
steps*. Each cell is a single seed, so each answers "does the arm behave differently
here", not "is the arm better".

---

## Status: what can start today

| cell | state | blocker |
|---|---|---|
| 1.5B / ALFWorld | **running** as `ccpo-attncred-150-20260913` (GPUs 4,5) | none — finishes ~2026-09-15 03:00 UTC |
| 1.5B / WebShop | ready to launch | 2 free GPUs; host RAM for 256 env workers |
| 7B / ALFWorld | **blocked** | weights not downloaded; disk; 4 GPUs |
| 7B / WebShop | **blocked** | the above, plus WebShop's RAM on top |

Three hard constraints to settle before promising all four:

1. **Qwen2.5-7B-Instruct is not on this box.** `hf/hub/` holds only the 1.5B model and
   `e5-base-v2`. Download is ~15 GB.
2. **Disk.** A 1.5B checkpoint measures **25 GB**; a 7B one is ~**124 GB**. Free space
   is **134 GB**. A 7B arm cannot hold a rolling checkpoint *and* a pinned step-100 at
   the same time, let alone a best. Decide the 7B checkpoint policy before launching
   (options in cell 3).
3. **GPUs.** All eight cards are busy; two of them are ours and hold the running arm.
   A 7B arm wants 4. These four cells are a queue, not a batch.

---

## Invariants — identical in all four cells

```
--arm ccpo
--set ccpo_lam_fix=1.0 --set ccpo_prior_kappa=2.0 --set ccpo_backoff_task=1
--set ccpo_jweight_c=0.0 --set ccpo_phi=hidden+ctx
--set ccpo_gate=hard --set ccpo_target=return --set ccpo_rho=0.0
--set total_epochs=150 --set pin_steps=100 --set keep_ckpts=1
```

Everything else stays at `exp_run.py` defaults, which are the published reference
protocol (lr 1e-6, KL 0.01 low-var, gamma 0.95, group 8, train batch 16, invalid-action
penalty 0.1, seed 0, `test_freq=5`, `save_freq=5`).

**Checkpoints.** `pin_steps=100` hardlinks `global_step_100` to `step100-pin` at the
step-100 save, so the rolling window cannot prune it. `step<N>-best` and `step<N>-last`
are maintained automatically. `keep_ckpts=1` keeps disk down; raise it only if you
intend to resume often. To resume from or score a pin, verl asserts on the path name:

```
cp -al <ckpts>/step100-pin <ckpts>/global_step_100
```

**Do not change the seed between cells.** Every published number in this project is
seed 0, and cross-cell comparisons are already weak enough.

---

## Cell 1 — 1.5B / ALFWorld  (running)

Launched as `ccpo-attncred-150-20260913` on GPUs 4,5:

```
python3 scripts/exp_run.py --name ccpo-attncred-150 --arm ccpo \
  --set gpus=4,5 --set total_epochs=150 --set pin_steps=100 --set keep_ckpts=1 \
  --set ccpo_lam_fix=1.0 --set ccpo_prior_kappa=2.0 --set ccpo_backoff_task=1 \
  --set ccpo_jweight_c=0.0 --set ccpo_phi=hidden+ctx \
  --set ccpo_gate=hard --set ccpo_target=return --set ccpo_rho=0.0
```

**Comparator:** `ccpo-return-hard-20260910` (same config minus the three flags, 100
steps). Steps 1-100 of this run also re-run `ccpo-attncred-20260912` exactly, so they
are a free same-config replicate — at step 40 the two differ by a mean of 3.5 points
per evaluation, which is the noise floor every claim here must clear.

**The question:** does the readout separate from the control *after* step 100, where
no arm in this project has ever run? Judge on the 120-150 window, not on single
evaluations.

**Comparators (ALFWorld, Qwen2.5-1.5B):** GiGPO 86.7 ±1.7 @150, G²PO 95.0 ±0.8 @100,
HGPO 92.77 ±1.08 (K=2) @160, GRPO 72.8. Our best arm to date is 78.2 on the window.

**Cost:** ~896 s/step, ~37 h total.

---

## Cell 2 — 1.5B / WebShop

```
python3 scripts/exp_run.py --name ccpo-attncred-ws-150 --arm ccpo \
  --set env_name=Webshop --set max_steps=15 --set gpus=<pair> \
  --set total_epochs=150 --set pin_steps=100 --set keep_ckpts=1 \
  --set ccpo_lam_fix=1.0 --set ccpo_prior_kappa=2.0 --set ccpo_backoff_task=1 \
  --set ccpo_jweight_c=0.0 --set ccpo_phi=hidden+ctx \
  --set ccpo_gate=hard --set ccpo_target=return --set ccpo_rho=0.0
```

**Preconditions.**
* `env_name=Webshop` switches `exp_run.py` to `/workspace/.venv-webshop` and sets
  `JAVA_HOME` for the per-worker Lucene JVM automatically. Do not run it in `.venv`.
* Keep `webshop_use_small=1` (the default). GiGPO's published 67.4% is on the
  1,000-product subset — `ppo_trainer.yaml` sets `use_small: True` and
  `run_webshop.sh` never overrides it. The full catalogue peaks at 19 GB per worker
  and is not runnable here.
* `search_engine/indexes` must be the 1,000-doc index; `init_search_engine` always
  selects `indexes` regardless of catalogue.
* **Stop the retrieval server if it is up** — it holds ~66 GB of anon memory, and
  WebShop starts 128 + 128 workers against a 256 GiB cgroup.

**`max_steps` must be changed for WebShop: add `--set max_steps=15`.** The default is
50, which is ALFWorld's cap. GiGPO and G²PO both run WebShop at **15** environment
steps (`gigpo:1242`, `g2po:1377`); HGPO uses 30 (`hgpo:518`). Leaving it at 50 would
make our episodes 3x longer than every number we compare against, change the
return-to-go target, and inflate step time — a protocol break, not a knob.

**Grouping is expected to work here, but verify at step 1.** G²PO measured WebShop's
state groups directly: mean group size ~5, and only **11.9% of steps sit in a group of
1** against 8.1% on ALFWorld (`g2po:1244-1247`, `:1305-1314`). So exact-match grouping
is not degenerate on WebShop — though HGPO shows that *history-conditioned* groups thin
out much faster there (step utilisation 0.92/0.64/0.44 for 0/1/2-context, vs ALFWorld's
0.97/0.75/0.52, `hgpo:567-573`). Our gate is state-only, so this cell should behave.
Check at step 1 anyway: if `ccpo/bucket_singleton_frac` > 0.9 or `effect_rel` < 0.02,
the estimator is inert, and that is worth reporting after one step rather than 150.
ALFWorld reference values: mean bucket 7.2, singleton fraction 0.33, `live_frac` 1.000.

**Comparators (Qwen2.5-1.5B, Score / Success %):**

| | Score | Success | iters |
|---|---|---|---|
| GiGPO (w/o std) | 83.5 ±1.8 | **67.4** ±4.5 | 150 |
| G²PO | 85.1 ±1.4 | 71.2 ±2.6 | 100 |
| HGPO K=4 | 90.64 ±1.05 | 78.12 ±2.06 | 160 |
| GRPO | 75.8 ±3.5 | 56.8 ±3.8 | — |
| PPO / RLOO | 73.8 / 73.9 | 51.5 / 52.1 | — |

**Report both Score and Success.** WebShop's Score is partial credit and Success is the
binary; every published table gives both, and quoting only one is not comparable.

Ours will sit far below these, as on ALFWorld. The comparison that matters is against
our own control, which **does not exist on WebShop** — so either run `ret-hard` there
first, or state plainly that this cell has no paired control and is descriptive only.

**Cost:** unmeasured. Time steps 1-5 and re-estimate before assuming 150 steps fit.

---

## Cell 3 — 7B / ALFWorld

**Not launchable as written.** Three things to settle first.

1. **Get the weights** (~15 GB):
   ```
   HF_HOME=/workspace/hf /workspace/.venv/bin/python3 -c \
     "from huggingface_hub import snapshot_download; \
      snapshot_download('Qwen/Qwen2.5-7B-Instruct')"
   ```
   Then `--set model=Qwen/Qwen2.5-7B-Instruct`.

2. **Decide the checkpoint policy.** At ~124 GB per checkpoint against 134 GB free,
   "step-100 + best + last" is not affordable as three independent copies. Options, in
   the order I would take them:
   * `save_freq=10` and `keep_ckpts=1` — halves save traffic, still leaves rolling +
     pin ~248 GB at the moment the pin materialises. **Needs more disk regardless.**
   * Drop `hf_model` from `checkpoint.contents` (~15 GB/checkpoint) — helps, not enough.
   * Free disk first: ~250 GB would make the pin plan work; ~400 GB makes it
     comfortable. The 64 GB e5 index is already gone.
   * Or accept **best + last only**, dropping the step-100 pin, and say so in NOTES.
3. **GPUs:** 4, not 2. Keep `tensor_model_parallel_size=1` if the memory allows;
   otherwise raise `gpu_mem_util` deliberately rather than by trial.

**Comparators (ALFWorld, Qwen2.5-7B):** GiGPO **90.8** ±1.3 @150, HGPO **95.44** ±0.62
(K=2) @160, G²PO **96.9** ±1.3 @100, GRPO 77.6. Note HGPO re-runs GiGPO in its own
harness and reports 93.29 for it, against GiGPO's own 90.8 — do not cross-compare cells
between papers.

**Cost:** at ~2.5-3x the 1.5B step time, 150 steps is **4-5 days**. Decide whether that
is worth one seed of a null before starting it.

---

## Cell 4 — 7B / WebShop

Everything in cell 3, plus everything in cell 2, plus: the 7B rollout engine and
WebShop's 256 workers compete for the same host RAM. **Do not attempt until cells 2 and
3 have each run**, and only if cell 2 showed the gate is not degenerate on WebShop.
Flags are cell 3's plus `--set env_name=Webshop --set max_steps=15`.

**Comparators (WebShop, Qwen2.5-7B, Score / Success %):** GiGPO 86.2 ±2.6 / **75.2**
±3.8 @150, G²PO 89.8 ±1.7 / **78.3** ±0.6 @100, HGPO K=2 88.96 ±1.04 / 78.51 ±1.40
@160, GRPO 79.3 / 66.1, PPO 81.4 / 68.7. Same `max_steps=15` correction as cell 2.

---

## Report metrics — the same block for every cell

Produce it with:

```
python3 scripts/report_results.py --exp experiments/<exp-id> --rows best,last,100,150
python3 scripts/report_results.py --exp experiments/<exp-id> --rows best,last,100,150 --markdown
```

**1. Held-out success by task type**, at best / last / step 100 / step 150, in the
published-table layout (Pick, Look, Clean, Heat, Cool, Pick2, All). WebShop reports a
single score, so the type columns will be empty there — keep `All`.

**2. Behaviour at the same steps** (training rollouts, not held-out): turns per
episode, train success, valid-action %, admissible %, response tokens, truncation %.
`Adm%` populates for runs launched after commit `4bbd026`; `NoOp%` and `Revisit%`
remain blank until the env-side instrumentation is written.

**3. Window means, which are the statistic this project trusts:** 70-100 and 120-150,
plus the mean over all evaluations. Single evaluations carry SE ~±3.3 and a best-step
figure is biased up ~1.5 sd.

**4. Paired comparison against the cell's control**, if one exists: mean difference,
sd, se, t, and "ahead at k/n" over the shared evaluation steps. State the control by
exp-id. If there is no control (cells 2 and 4 today), say so instead of comparing to a
published number as though it were paired.

**5. Estimator diagnostics — the evidence that the arm was live at all:**
`effect_rel`, `live_frac`, `lam_k_mean`, `n_eff_mean`, `bucket_size_mean`,
`bucket_singleton_frac`, `n_buckets`, `phi_rel_corr`, `lam_eb_obs`. Report them by
phase (1-50 / 51-100 / 101-150). An arm with `effect_rel` ~0 changed nothing and its
success number says nothing about the method.

**6. Cost and provenance:** step time, wall clock, peak disk, GPUs used, exp-id, git
sha, seed, and the resolved paths of the three checkpoints (`step100-pin`,
`step<N>-best`, `step<N>-last`).

**7. The caveat, stated once per cell:** single seed. The same-config replicate spread
on ALFWorld is 1.79 points on a window mean and up to 15 points on a single step. No
difference smaller than that is a result.

---

## Decision gates — fixed in advance

* **Step 1, every cell:** if `bucket_singleton_frac` > 0.9 or `effect_rel` < 0.02, the
  estimator is inert on this benchmark. Stop and report; do not spend the budget.
* **Step 20, ALFWorld:** abort if held-out is below ~11% (more than 2 SE under the
  control's trajectory).
* **Step 50:** compare to the control at the same step. A deficit larger than 10 points
  is worth stopping for; anything smaller is inside this setup's noise.
* **Step 100:** the pin lands here. Everything from 101-150 is new ground for this
  project — no arm has run past 100 steps, so there is no prior for what the curve does.
