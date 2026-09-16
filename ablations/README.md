# `ablations/` — the five CCPO-ATTNCRED ablations

Each removes one component of the main arm, on both benchmarks, 1.5B, 150 steps
(`experiments/experiments.md`). The control is the main arm of the **same** benchmark —
so a WebShop ablation inherits `ccpo_target=score` and stays paired with what it ablates.

```bash
python3 official-repo/ablations/run.py --list
python3 official-repo/ablations/run.py --ablation hard-gate --benchmark alfworld --gpus 0,1,2,3
python3 official-repo/ablations/run.py --ablation cosine    --benchmark webshop  --gpus 0,1,2,3
```

| ablation | key | removes | asks |
|---|---|---|---|
| `hard-gate` | `ccpo_wmode=hard` | the soft kernel | ordering neighbours, or only restricting them? |
| `no-task-baseline` | `ccpo_prior_kappa=0` | the credibility prior `b_task` | does leaning on the task mean buy anything? |
| `even-blend` | `ccpo_lk_fix=0.5` | the *support weighting* of the prior | the credibility form, or any fixed ratio? |
| `no-context-vector` | `ccpo_phi=hidden` | the trajectory-context block of φ | does whole-episode context earn its place? |
| `cosine` | `ccpo_wmode=cos` | the exponential kernel | does `exp(−d/τ)` beat a plain similarity? |

Ten runs. Each is a **single-key** delta from its control, asserted by the guard.

## What each one does to the math

Baseline in the main arm, per credited occurrence *i*:

```
b_loo = Σ_b w_b·TGT_b / Σ_b w_b      w_b = exp(−‖φ_i−φ_b‖/τ),  b over OTHER trajectories
base  = (J·b_loo + κ·b_task)/(J+κ)   = λ_k·b_loo + (1−λ_k)·b_task,  λ_k = J/(J+κ), κ=2
```

**`hard-gate`** → `w_b = 1[d ≤ τ]`, and with 0/1 weights `b_loo` is the plain average of
the survivors. Same τ, so the same neighbourhood is selected and only the weighting
inside it differs. If τ admits nobody, the nearest trajectory is kept, so no row loses
its baseline. On ALFWorld this measured `E[w]=0.011` — ~1 neighbour in 90 clears
τ=0.15·median, so the arm runs close to a nearest-trajectory baseline, and `effect_rel`
rose to 0.70 against soft's 0.28.

**`no-task-baseline`** → κ=0, so the blend never fires and `b_task` is not even computed:
`base = b_loo`, context only, exactly as the spec words it. `lam_k_mean` goes to 1.000.
Note `ccpo_backoff_task` stays 1 — the level-1 task *bucket* is a different mechanism
from `b_task` (it recomputes `b_loo` over the task, it does not mix in a prior) and it
is what keeps `live_frac` at 1.0. To ablate that instead, add
`--set ccpo_backoff_task=0` and expect `live_frac` to fall by ~8% on ALFWorld and ~20%
on WebShop.

**`even-blend`** → `λ_k = 0.5` on every level-0 occurrence, whatever its support:
`base = ½·b_loo + ½·b_task`. κ no longer enters the weight. Paired with
`no-task-baseline` this separates *the prior exists* from *the prior is weighted by
evidence* — the two things the Bühlmann form does at once.

**`no-context-vector`** → φ is the whitened last-prompt-token hidden state alone. The
prompt carries `step_count` plus the most recent 2 turns, so it can separate "step 5
from step 15" but not "has this agent already searched here twice" — `n_unique`,
`revisit` and `progress` summarise the whole episode and appear nowhere in the prompt.
Watch `phi_rel_corr`: it rose 0.01 → 0.24 over training on ALFWorld with the block in,
and stayed near 0.02 on WebShop, so this ablation may cost little there and a lot here.

**`cosine`** → `w_b = max(cos(φ_i, φ_b), 0)`, no τ at all: weight falls off linearly in
the angle instead of exponentially in the distance, so distant neighbours keep far more
weight. Negative cosines clip to zero — a weighted mean needs non-negative weights, and
a neighbour pointing the other way is evidence of nothing, not evidence against. If
every neighbour clips away, the most similar one is kept.

## Two of these needed new code

`ccpo_wmode=cos` and `ccpo_lk_fix` did not exist. Both were added to `ccpo/core_ccpo.py`
(the estimator the runs actually import) behind env knobs that default to current
behaviour, so no existing arm changes:

* `ACG_CCPO_WMODE=cos` — `core_ccpo.py:802-816`, beside the `hard` branch.
* `ACG_CCPO_LK_FIX` — `core_ccpo.py:316-323` and `:952-959`. Setting it also makes
  `B_TASK` be computed at κ=0, since the blend needs it; otherwise the ablation would
  silently depend on κ staying non-zero.

`scripts/exp_run.py` plumbs `ccpo_lk_fix` → `ACG_CCPO_LK_FIX` so it lands in
`config.json` like every other knob.

## Guard

```bash
.venv/bin/python3 official-repo/ablations/test_ablations.py     # CPU, ~3 s
```

Needs numpy; torch is stood in for if absent, so it runs outside the training venv too.
It checks that each ablation differs from its control in exactly its declared key on
both benchmarks, that the WebShop ones keep the dense-score target, and — numerically,
against the real estimator — that `wmode=cos` weights by clipped cosine (verified
against a hand-computed mean, with the all-clipped fallback exercised) and that
`lk_fix` pins λ_k, including at κ=0, while the derived path still gives `J/(J+κ)`.
