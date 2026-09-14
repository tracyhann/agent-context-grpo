# Instructions — `attncred` across two backbones and two benchmarks

Four experiments to run: {Qwen2.5-1.5B-Instruct, Qwen2.5-7B-Instruct} x {ALFWorld,
WebShop}. Each runs **150 steps** and keeps the **step-100, best and last**
checkpoints. One seed each (seed 0).

**The arm.** `attncred` is the hard-gate return estimator plus three flags: the
phi-attention readout with `lam` pinned to 1 (`ccpo_lam_fix=1.0`), the credibility
prior `lam_k = J/(J+kappa)`, `kappa=2` (`ccpo_prior_kappa=2.0`), and the J=0 task-level
fallback (`ccpo_backoff_task=1`). Background: `hypothesis.md` H-AL,
`ccpo-attncred-20260912/docs/method.html`.

**What these four ask.** On ALFWorld at 1.5B over 100 steps the arm is a null: window
70-100 of 75.56, inside the same-config noise band, while `effect_rel` ran 0.24-0.35 —
the estimator moved a quarter to a third of the step credit and held-out success did
not follow. These runs test whether that null is specific to that backbone, that
benchmark, and that step budget.

---

## Host requirements

| | 1.5B | 7B |
|---|---|---|
| GPUs | ≥ 4 GPUs | 8 GPUs |
| checkpoint size | ~25 GB each | ~124 GB each |

Peak disk assumes `keep_ckpts=1` and the step-100 pin: rolling + in-flight save +
best + pin. If the host cannot hold that for 7B, drop `pin_steps` and keep best + last
only, and record that choice in the arm's NOTES.md.

**Before launching:**

* Weights: `Qwen/Qwen2.5-1.5B-Instruct`, `Qwen/Qwen2.5-7B-Instruct` under `HF_HOME`.
* ALFWorld data at `ALFWORLD_DATA`; `scripts/setup_env.sh` builds the main venv.
* WebShop needs its own venv at `.venv-webshop` (`web_agent_site` is imported
  in-process) and data at `envdata/webshop_data`. `exp_run.py` selects that venv and
  sets `JAVA_HOME` for the per-worker Lucene JVM when `env_name=Webshop`.
* WebShop runs the **1,000-product subset** (`webshop_use_small=1`, the default) —
  that is what the published numbers are on. `search_engine/indexes` must be the
  1,000-doc index, because `init_search_engine` always selects `indexes` regardless of
  which catalogue loaded. The full catalogue needs ~19 GB per worker and 256 workers
  start concurrently.

---

## Invariants — identical in all four runs

```
--arm ccpo
--set ccpo_lam_fix=1.0 --set ccpo_prior_kappa=2.0 --set ccpo_backoff_task=1
--set ccpo_jweight_c=0.0 --set ccpo_phi=hidden+ctx
--set ccpo_gate=hard --set ccpo_target=return --set ccpo_rho=0.0
--set total_epochs=150 --set pin_steps=100 --set keep_ckpts=1
```

Everything else stays at `exp_run.py` defaults, which are the published reference
protocol: lr 1e-6, KL 0.01 low-var, gamma 0.95, group 8, train batch 16,
invalid-action penalty 0.1, seed 0, `test_freq=5`, `save_freq=5`.

**Do not change the seed between runs.** Cross-run comparisons are weak enough already.

**Checkpoints.** `pin_steps=100` hardlinks `global_step_100` to `step100-pin` at the
step-100 save so the rolling window cannot prune it; `step<N>-best` and `step<N>-last`
are maintained automatically. To resume from or score a pin, copy it back first — verl
asserts the path contains `global_step_`:

```
cp -al <ckpts>/step100-pin <ckpts>/global_step_100
```

---

# Backbone: Qwen2.5-1.5B-Instruct

## 1. ALFWorld — 150 steps

```
python3 scripts/exp_run.py --name ccpo-attncred-150 --arm ccpo \
  --set gpus=<a,b> --set total_epochs=150 --set pin_steps=100 --set keep_ckpts=1 \
  --set ccpo_lam_fix=1.0 --set ccpo_prior_kappa=2.0 --set ccpo_backoff_task=1 \
  --set ccpo_jweight_c=0.0 --set ccpo_phi=hidden+ctx \
  --set ccpo_gate=hard --set ccpo_target=return --set ccpo_rho=0.0
```

**Published, Qwen2.5-1.5B:** GiGPO 86.7 ±1.7 @150, G²PO 95.0 ±0.8 @100, HGPO 92.77
±1.08 (K=2) @160, GRPO 72.8.

**The question:** what does the curve do *after* step 100? No arm in this project has
run past 100 steps, so steps 101-150 are unmeasured ground. Judge on the 120-150 window
mean, not on single evaluations.

## 2. WebShop — 150 steps

```
python3 scripts/exp_run.py --name ccpo-attncred-ws-150 --arm ccpo \
  --set env_name=Webshop --set max_steps=15 --set gpus=<a,b> \
  --set total_epochs=150 --set pin_steps=100 --set keep_ckpts=1 \
  --set ccpo_lam_fix=1.0 --set ccpo_prior_kappa=2.0 --set ccpo_backoff_task=1 \
  --set ccpo_jweight_c=0.0 --set ccpo_phi=hidden+ctx \
  --set ccpo_gate=hard --set ccpo_target=return --set ccpo_rho=0.0
```

**`--set max_steps=15` is required.** The default is 50, which is ALFWorld's cap.
GiGPO and G²PO both run WebShop at 15 environment steps; HGPO uses 30. At 50 the
episodes are three times longer than everything the numbers are compared against, and
the return-to-go target changes with them.

**Grouping should hold here, but verify at step 1.** G²PO measured WebShop's state
groups: mean size ~5, only 11.9% of steps in a group of 1 (ALFWorld: 8.1%). Exact-match
grouping is therefore not degenerate on WebShop. Confirm with `ccpo/bucket_size_mean`,
`ccpo/bucket_singleton_frac` and `effect_rel` at step 1 — ALFWorld reference values are
mean bucket 7.2, singleton fraction 0.33, `live_frac` 1.000.

**Published, Qwen2.5-1.5B (Score / Success %):**

| | Score | Success | iters |
|---|---|---|---|
| GiGPO (w/o std) | 83.5 ±1.8 | 67.4 ±4.5 | 150 |
| G²PO | 85.1 ±1.4 | 71.2 ±2.6 | 100 |
| HGPO K=4 | 90.64 ±1.05 | 78.12 ±2.06 | 160 |
| GRPO | 75.8 ±3.5 | 56.8 ±3.8 | — |
| PPO / RLOO | 73.8 / 73.9 | 51.5 / 52.1 | — |

---

# Backbone: Qwen2.5-7B-Instruct

Add `--set model=Qwen/Qwen2.5-7B-Instruct` and four GPUs to the commands above. Keep
`tensor_model_parallel_size=1` if memory allows; if it does not, raise `gpu_mem_util`
deliberately and record the value rather than tuning it per run.

## 1. ALFWorld — 150 steps

```
python3 scripts/exp_run.py --name ccpo-attncred-7b-150 --arm ccpo \
  --set model=Qwen/Qwen2.5-7B-Instruct --set gpus=<a,b,c,d> \
  --set total_epochs=150 --set pin_steps=100 --set keep_ckpts=1 \
  --set ccpo_lam_fix=1.0 --set ccpo_prior_kappa=2.0 --set ccpo_backoff_task=1 \
  --set ccpo_jweight_c=0.0 --set ccpo_phi=hidden+ctx \
  --set ccpo_gate=hard --set ccpo_target=return --set ccpo_rho=0.0
```

**Published, Qwen2.5-7B:** GiGPO 90.8 ±1.3 @150, HGPO 95.44 ±0.62 (K=2) @160, G²PO
96.9 ±1.3 @100, GRPO 77.6. HGPO re-runs GiGPO in its own harness and reports 93.29 for
it against GiGPO's own 90.8 — do not cross-compare cells between papers.

## 2. WebShop — 150 steps

```
python3 scripts/exp_run.py --name ccpo-attncred-7b-ws-150 --arm ccpo \
  --set model=Qwen/Qwen2.5-7B-Instruct --set env_name=Webshop --set max_steps=15 \
  --set gpus=<a,b,c,d> \
  --set total_epochs=150 --set pin_steps=100 --set keep_ckpts=1 \
  --set ccpo_lam_fix=1.0 --set ccpo_prior_kappa=2.0 --set ccpo_backoff_task=1 \
  --set ccpo_jweight_c=0.0 --set ccpo_phi=hidden+ctx \
  --set ccpo_gate=hard --set ccpo_target=return --set ccpo_rho=0.0
```

**Published, Qwen2.5-7B (Score / Success %):** GiGPO 86.2 ±2.6 / 75.2 ±3.8 @150, G²PO
89.8 ±1.7 / 78.3 ±0.6 @100, HGPO K=2 88.96 ±1.04 / 78.51 ±1.40 @160, GRPO 79.3 / 66.1,
PPO 81.4 / 68.7.

All WebShop preconditions from the 1.5B run apply, including `max_steps=15` and the
1,000-product subset.

---

# Report metrics — the same block for every run

```
python3 scripts/report_results.py --exp experiments/<exp-id> --rows best,last,100,150
python3 scripts/report_results.py --exp experiments/<exp-id> --rows best,last,100,150 --markdown
```

**1. Held-out success by task type** at best / last / step 100 / step 150, in the
published-table layout: Pick, Look, Clean, Heat, Cool, Pick2, All. WebShop has no task
types — report `All`, and report **both Score and Success**, since every published
WebShop table gives both and one alone is not comparable.

**2. Behaviour at the same steps** (training rollouts, not held-out): turns per
episode, train success, valid-action %, admissible %, response tokens, truncation %.

**3. Window means — the statistic to trust:** 70-100 and 120-150, plus the mean over
all evaluations. A single evaluation carries SE ~±3.3 points; a best-step figure is the
max of a noisy series and is biased up by roughly 1.5 sd.

**4. Estimator diagnostics — the evidence the arm was live at all**, by phase
(1-50 / 51-100 / 101-150): `effect_rel`, `live_frac`, `lam_k_mean`, `n_eff_mean`,
`bucket_size_mean`, `bucket_singleton_frac`, `n_buckets`, `phi_rel_corr`,
`lam_eb_obs`. An arm with `effect_rel` ~0 changed nothing, and its success number says
nothing about the method.

**5. Cost and provenance:** step time, wall clock, peak disk, GPU count, exp-id, git
sha, seed, and the resolved paths of the three checkpoints (`step100-pin`,
`step<N>-best`, `step<N>-last`).

**6. The caveat, once per run:** single seed. On ALFWorld the same-config replicate
spread is 1.79 points on a window mean and up to 15 points on a single step. No
difference smaller than that is a result.

---

# Decision gates — fixed in advance

* **Step 1, every run:** if `bucket_singleton_frac` > 0.9 or `effect_rel` < 0.02, the
  estimator is inert on this benchmark. Stop and report it — that is a finding about
  where the method applies, and it costs one step instead of the whole budget.
* **Step 20, ALFWorld:** abort if held-out is below ~11%. Prior 1.5B arms reach
  15-23% by then.
* **Step 50:** on ALFWorld at 1.5B, prior arms sit at 49-57% held-out here. More than
  ~10 points below that is worth stopping for; anything smaller is inside this setup's
  noise.
* **Step 100:** the pin lands here, and everything past it is new ground — no arm in
  this project has run beyond 100 steps, so there is no prior for the curve's shape.
