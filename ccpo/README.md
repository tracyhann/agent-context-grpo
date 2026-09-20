# `ccpo/` — estimator and method arms

The selected main method (2026-09-20) is **H2, no credit shrinkage, no episode
advantage**, for both M10/ALFWorld and M11/WebShop. It uses the existing registry
key `future-progress-h2-no-credit-shrinkage` in `ablations/ablations.py`, with
full-strength context baselines and soft exponential weights. See the
[current definition and configurations](../experiments/MAIN_METHOD.md).

The definitions below describe the historical arms and their equations. Their
credit-shrinkage and episode-on settings are retained for reproducibility.

## Historical arm definitions

Two methods, each on two benchmarks and two backbones (`experiments/experiments.md`):

| method | delta from the published arm |
|---|---|
| `attncred` | — |
| `attncred-context-adv-only` | `ccpo_ep_w=0` — the episode advantage removed |

```bash
python3 official-repo/ccpo/run.py --list
python3 official-repo/ccpo/run.py --method attncred --benchmark alfworld --backbone 1.5b --gpus 0,1,2,3
python3 official-repo/ccpo/run.py --method attncred --benchmark webshop  --backbone 7b   --gpus 0,1,2,3,4,5,6,7
python3 official-repo/ccpo/run.py --method attncred-context-adv-only --benchmark alfworld --backbone 1.5b --gpus 0,1,2,3
```

Eight runs in all. Arguments after the fixed flags go to `scripts/exp_run.py` verbatim
and a later `--set` wins, so `--set val_batch_size=64` adapts to a host without editing
the arm. `--dry-run` writes `config.json` and `run.sh` without launching.

## The advantage

```
A[i,t] = ep_w · A_EP[i,t] + step_w · z_task_live(A_CC_i) · mask[i,t]

A_CC_i = ( TGT_i − (J·b_loo + κ·b_task)/(J+κ) ) + z_task( V(next_i) − V(cur_i) )
b_loo  = Σ_b exp(−‖φ_i−φ_b‖/τ)·TGT_b / Σ_b exp(−‖φ_i−φ_b‖/τ),   b over OTHER trajectories
φ      = [ whiten(hidden, top-3 removed) ; thermo(t, n_unique, progress, revisit) ]
τ      = 0.15 · median pairwise distance in the bucket
κ = 2,  λ pinned to 1 (ccpo_lam_fix), gate = exact (task_uid, observation)
```

`attncred-context-adv-only` sets `ep_w = 0`: `A_EP` is plain GRPO on the trajectory
return and is the term attncred *shares* with the baselines it is measured against, so
dropping it leaves only the context-conditioned step credit and the edge term. HGPO
ships this shape and reports that adding the episode term hurt.

## What each overlay changes

**Benchmark.** ALFWorld: `max_steps=50`, `ccpo_target=return`. WebShop: `max_steps=15`
(GiGPO and G²PO both publish at 15 turns, and the return-to-go target changes with the
horizon) and **`ccpo_target=score`** — experiments.md asks for the dense score, because
verl-agent overwrites WebShop's reward with a binary 10/0 and 30–69% of task groups
then score zero on *every* rollout, where a group-relative estimator computes exactly
zero advantage. The dense target is the discounted return-to-go of `info['task_score']`,
scaled to the binary reward's units, and it changes the **step channel only** — the
episode term keeps the published binary reward.

The rest of the WebShop protocol (prompt 4096, `mean_norm`, mini-batch 64, val batch
128, 0.05 CPU/worker) arrives from `exp_run.WEBSHOP_PROTOCOL`, which is G²PO's
`run_webshop.sh` verbatim and is asserted by `tests/test_webshop_protocol.py`. Note
`mean_norm` changes the update: on WebShop **neither** advantage term is standardised,
both stay in reward units; on ALFWorld both are unit-variance.

**Backbone.** 1.5B needs ≥4 GPUs, 7B needs ≥8 (`run.py` warns below the floor; the ids
are yours to pass). `tensor_model_parallel_size` stays 1. The token budgets in
`exp_run.DEFAULTS` were memory-calibrated for 1.5B and are **not** raised here, because
no 7B run has been measured on the target host — smoke-test one step and adjust with
`--set ppo_max_token_len_per_gpu=…` rather than trusting an unverified number.

## Protocol

150 steps, three checkpoints, one seed (0). The three checkpoints come out of the
retention logic in `ray_trainer.py:1533-1578`: `step<N>-best` (hardlinked at each new
best eval), `step100-pin` (`pin_steps=100`), `step<N>-last` (symlink to the newest
rolling dir, kept by `keep_ckpts=1`). To resume or score a pin, copy it back first —
verl asserts the path contains `global_step_`:

```bash
cp -al <ckpts>/step100-pin <ckpts>/global_step_100
```

**Early stopping is off** (`early_stop_patience=0`). This is the one intended
difference from the historical control `ccpo-attncred-150-20260913`, which ran with
patience 8; experiments.md says every run goes to 150 steps.

## Two flags that `experiments/INSTRUCTIONS.md` omits

Its command blocks list eight `ccpo_*` invariants but not `ccpo_tau` or `ccpo_edge_w`,
whose `exp_run` defaults are **1.0** and **0.0** — while every run ever called attncred
used **0.15** and **1.0**. Running those commands verbatim gives a 6.7× wider kernel and
no edge term: a different estimator under the same name. Both are pinned in `BASE`, as
is every other method key, so no arm here inherits its method from a mutable default.

## Flash attention (required)

The target is A100 (sm_80) / H100 (sm_90), and FA2 runs on both sides:
`vllm_attn_backend=FLASH_ATTN` (rollout) and `remove_padding=True` (trainer's packed
varlen path). Neither is inferred — `exp_run.build_env` *derives* `VERL_ATTN_IMPL` from
whether the venv can `import flash_attn` and falls back to `sdpa` plus `docker/fa_stub`
on `PYTHONPATH`, which is right on the sm_120 research box and wrong here: packing off,
~2.6× per step, and a `config.json` that still reads like a normal run.

So `run.py` preflights the training interpreter (`.venv`, or `.venv-webshop` for
WebShop) and refuses to launch without a real build, asking three things an import
check does not: that `flash_attn_2_cuda` is compiled, that `flash_attn_varlen_func`
actually runs on this card, and — by stripping `fa_stub` from `PYTHONPATH` before
probing — that a stub is not shadowing a missing build. `--dry-run` reports without
blocking; `--allow-sdpa` overrides, and a run made with it is not comparable.

## Guard

```bash
python3 official-repo/ccpo/test_arms.py     # CPU, ~2 s
```

Checks that every key an arm sets is a real `exp_run` key, that
`attncred/alfworld/1.5b` reproduces the control's `config.json` with only the documented
delta, that each overlay moves exactly the keys it claims, that attention is pinned and
the preflight refuses fake builds, and that the trainer still reads `ACG_CCPO_EP_W`,
builds the dense target, and receives `ACG_CCPO_LK_FIX`.
