# `ablations/` — the CCPO-ATTNCRED ablations

**Current main method (2026-09-20):** `future-progress-h2-no-credit-shrinkage`
on ALFWorld and WebShop. This existing registry entry is promoted to the main
method for reporting and paired comparisons; its key and configurations are
retained. See [the main-method definition](../experiments/MAIN_METHOD.md).

Each entry declares its implementation parent and exact delta on the same
benchmark. The H2 main method uses `ccpo_target=return` on both benchmarks;
legacy WebShop `attncred` entries inherit the dense-score target. The table and
baseline equations immediately below describe the original ablation matrix.
Later sections record the newer variants; compare their final configurations
against the selected main method as documented in the experiment plan.

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
| `no-edge` | `ccpo_edge_w=0` | G²PO's value-gain term | baseline or edge — which carries the effect? |
| `no-edge-return-ws` | `ccpo_edge_w=0` + `ccpo_target=return` | the edge term **and** the dense-score adaptation | WebShop only: does the baseline carry anything with nothing borrowed? |

Thirteen runs. Each is a single-key delta from its control — except
`no-edge-return-ws`, which declares two and runs on WebShop alone. The guard asserts
each arm moves exactly the keys it declares, on exactly the benchmarks it declares.

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


## H2 full-strength baselines with active episode credit

Registry key `future-progress-h2-no-credit-shrinkage-active-episode` runs on both ALFWorld and WebShop, using
`attncred-context-future-progress-h2` as parent and exactly two method changes:
`ccpo_lk_fix=1` and `ccpo_ep_w=1`. The existing H2 no-shrink arm differs only in
its episode coefficient (0). Context-stat weight stays 1 and original-edge weight
stays 0. The total actor advantage is E+N_CC(H+F), with episode credit added after
the benchmark's existing step-channel normalization.

The [experiment plan](../experiments/experiments.md) links both prepared 1.5B,
150-step, two-GPU configurations and their full math/validation. No run is queued
or launched. The existing generic registry CLI retains its separate GPU minimum;
the prepared configs reproduce the recorded two-GPU local protocol.


## M11 H2 WebShop: 30-turn no-shrink variants

These WebShop-only registry entries use Qwen2.5-1.5B-Instruct for 150 training
steps. Both set `max_steps=30` for training and validation, keep prompt history
and future horizon at 2, and retain context statistics.

| Registry key | Delta from M11 H2 | Episode weight |
|---|---|---:|
| `future-progress-h2-no-credit-shrinkage-30turn` | `ccpo_lk_fix=1`, `max_steps=30` | 0 |
| `future-progress-h2-no-credit-shrinkage-active-episode-30turn` | `ccpo_lk_fix=1`, `ccpo_ep_w=1`, `max_steps=30` | 1 |

Relative to their respective 15-turn no-shrink controls, only `max_steps` changes.
The two new arms differ from each other only in `ccpo_ep_w`. Their advantage
formulas remain H+F and E+H+F under WebShop mean_norm; full method definitions,
paired config diffs and the two-GPU launch preparation are in the
[no-episode notes](../experiments/m11-h2-noshrink-30turn-webshop-1.5b-2gpu-20260920/NOTES.md)
and [active-episode notes](../experiments/m11-h2-noshrink-active-episode-30turn-webshop-1.5b-2gpu-20260920/NOTES.md).
Prepared only; neither arm has been launched or queued.


## M10/M11 H2: full-strength cosine context baselines

Both benchmarks now have 1.5B, 150-step variants with the same context
representations and `ccpo_wmode=cos` (clipped cosine instead of the exponential kernel):

| Registry key | Delta from H2 parent |
|---|---|
| `future-progress-h2-no-credit-shrinkage-cosine` | `ccpo_lk_fix=1`, `ccpo_wmode=cos` |
| `future-progress-h2-no-credit-shrinkage-active-episode-cosine` | `ccpo_lk_fix=1`, `ccpo_wmode=cos`, `ccpo_ep_w=1` |

History and both potential readouts use cosine on processed frozen hidden states
plus context statistics. Prompt history/future horizon stay 2/2. The paired
no-shrink controls differ only in similarity mode; the episode pair differs only
in episode weight. Standard turn ceilings stay 50 for ALFWorld and 15 for WebShop.
The [experiment plan](../experiments/experiments.md#m10m11-h2-no-shrinkage-with-cosine-context-similarity-2026-09-20)
links all four prepared two-GPU configurations and documents the math and checks.
Prepared only; no runs launched or queued.


## M10/M11: history 1 / future 1, no shrinkage, no episode advantage

`future-progress-history1-future1-no-credit-shrinkage` is available for both
ALFWorld and WebShop, using 1.5B for 150 training steps. Its H1 parent is
`attncred-context-future-progress`, with exactly `history_length=1` and
`ccpo_lk_fix=1` added. Episode weight stays 0. Context statistics and standard
soft exponential weights remain enabled; accumulated context is not truncated.

Relative to H2 no-shrink, only prompt history and future horizon change from
2/2 to 1/1. Two-GPU configs retain turn ceilings of 50 (ALFWorld) and 15
(WebShop). The [experiment plan](../experiments/experiments.md#m10m11-history-1-future-1-no-shrinkage-no-episode-advantage-2026-09-20)
links both folders and records the math and validation. Prepared only; not
launched or queued.

## 7B main method and active-episode pair

The registry and CLI support `--backbone 7b` (Qwen2.5-7B-Instruct); omitted
backbone still means 1.5B. Programmatic callers can use
`build(key, benchmark, backbone="7b")`; the third positional `extra` argument
remains compatible. Control names and experiment IDs use the selected backbone.

The two H2 no-shrink keys, with and without active episode credit, have
[eight-GPU/150-step preparations for both benchmarks](../experiments/MAIN_METHOD_7B.md).
Run `python ablations/run.py --list --backbone 7b` to inspect IDs. The prepared
full configs retain the matched 1.5B benchmark settings and use rollout TP2.
No training is launched or queued by preparation.


## WebShop H2 no-shrink EP0 components

`future-progress-h2-no-credit-shrinkage-history-only`,
`future-progress-h2-no-credit-shrinkage-future-only`, and
`future-progress-h2-no-credit-shrinkage-no-context-vector` now accept WebShop
as well as ALFWorld. They retain episode weight 0. The active-episode component
keys retain episode weight 1. [Three prepared WebShop EP0 variants](../experiments/WEBSHOP_EP0_ABLATIONS.md)
include configs, math and reproducible preparation commands; no runs launched.


## M10 ALFWorld window allocation

`future-progress-noshrink-history{h}-future{f}` is available for (h,f) in
(4,0), (0,4), (3,1), (1,3), (3,3), on ALFWorld. These arms use no shrinkage,
EP0, context statistics and whole-trajectory LOO. Zero history removes prior
prompt turns and history credit; zero future disables progress with an explicit
zero horizon/weight. Positive advantage coefficients remain 1.
[Five prepared 1.5B / 150-step configurations](../experiments/ALFWORLD_WINDOW_ABLATIONS.md)
include full method notes and reproduction commands. None launched or queued.
