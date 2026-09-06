# Experiment plan

Target: beat the published ALFWorld numbers on Qwen2.5-1.5B-Instruct. The bar,
from HGPO Table 1 (in-distribution / out-of-distribution success):

| method | In | Out |
|---|---|---|
| GRPO | 72.8 | 70.1 |
| GiGPO (K=2) | 90.16 | 84.76 |
| **HGPO (K=2)** | **92.77** | **90.16** |

HGPO K=2 is what "SOTA" means at our `history_length=2`.

## Phase 1 — validate the harness and get a fair three-way comparison

Three arms, 2 GPUs each, launched together so they see identical wall-clock
conditions. Configs differ **only** in `--arm`.

| exp | arm | question |
|---|---|---|
| `gigpo-repro` | gigpo | **The gating question.** Does this harness reproduce published GiGPO (~90/85)? If not, no estimator comparison run here means anything, and the gap is in the harness. |
| `grpo-base` | grpo | The baseline every paper reports. Expect ~73/70. |
| `ccpo-base` | ccpo | CCPO as it stands, with the advantage-scale fix. Expect it near GiGPO — the two agree on 99.7% of sign decisions — so this arm mainly proves the CCPO path trains correctly and gives the ablation reference for Phase 2. |

Phase 1 is not optional and cannot be skipped for speed: without it, a CCPO number
has nothing trustworthy to be compared against.

## Phase 2 — the CCPO changes that could actually move the number

Chosen after Phase 1, one variable at a time, each against `ccpo-base`:

| exp | change | rationale |
|---|---|---|
| `ccpo-nextnode` | `ccpo_target=nextnode` | Credit G2PO's successor node value instead of the step's own return-to-go. A return-to-go carries every downstream accident of one trajectory, so conditioning the *baseline* cannot remove noise living in the *target*. Largest single lever available. |
| `ccpo-gate` | `ccpo_sim=0.95`, `ccpo_sim_backoff=0.8` | ~44% of buckets are singletons and take `A_CC = 0`. GiGPO already ships the similarity gate; the backoff borrows HGPO's "never waste a sample" idea. |
| `ccpo-full` | both, if each helps alone | Only run if the ablations justify it. |

## Choosing the CCPO arm's configuration — and a fairness trap in it

The default `eb` shrinkage measured **λ = 0.000** on a real batch, which makes
`A_CC` exactly the uniform baseline: that arm would be numerically identical to
`gigpo-repro` and ten hours would buy nothing. So the CCPO arm must use
`ccpo_shrink=eb_pooled`.

The similarity gate is a harder call. It lifts effective neighbourhood size from
`n_eff` 2.1 to 7.0 with full coverage, which the estimator plainly needs — but
GiGPO has the *same* gate natively (`algorithm.gigpo.enable_similarity`), and
`gigpo-repro` runs with exact match because that is the published setting. So
**CCPO(sim) vs GiGPO(exact) would confound the context conditioning with the
gate**, and the obvious criticism of any win is that the gate did the work.

Order of preference:

1. **CCPO with `eb_pooled` and the exact gate** (`ccpo_sim=0`). Differs from
   `gigpo-repro` in exactly one thing — the context-conditioned baseline — so a
   win is attributable. Risk: with `n_eff` ≈ 2.1 the estimator may still be inert.
2. If it is inert, **add the similarity gate** — and then a GiGPO arm with
   `enable_similarity=True` becomes mandatory, not optional, for the comparison to
   mean anything.

**Decide this in 30 minutes, not 10 hours.** `lam_pooled_obs`, `effect_rel`,
`n_eff_mean` and `r_vs_gigpo` are logged every step, so run option 1 for ~5 steps
and read them: `effect_rel` ≈ 0 or `r_vs_gigpo` ≈ 1.0 means inert, and the arm is
restarted under option 2 having cost half an hour.

## Phase 3 — if a winner emerges

Repeat the winning arm on 2 further seeds. Held-out evaluation here is stochastic
by design (T=0.4, 128 episodes, ~±0.048 measured), so a single run is not a result.

## Measured budget

From `gigpo-repro-20260906`, after enabling flash-attn and `use_remove_padding`:
**333 s per step** (was 851 s padded), of which `gen` is 177 s and is now the
floor — 50 turns over 128 environments at ~3.3 s each, which is vLLM plus
TextWorld, not something a config change reaches. A validation pass runs every 5
steps. **100 steps is roughly 10 h per arm.**

Arms cannot run in parallel here, and the packing speedup does not change that:
the binding constraint is ~192 ALFWorld ray actors per arm against the container's
8192-pid ceiling, and the actor count is set by `train_batch_size × group_size`,
not by GPU count. Two idle GPUs do not buy a second arm. So the schedule is
sequential and Phase 1's three arms would be ~30 h.

**Realistic allocation:** two arms. `gigpo-repro` (running) then `ccpo-pooled`.
That yields the harness validation, the strongest baseline this box can reproduce,
and the method measured against it on an identical config. GRPO is covered by the
published 72.8 / 70.1 on exactly this model and protocol, and is the arm to run
third if time allows. If the budget is
tighter than that, the order to cut is:

1. `gigpo-repro` — never cut. Without it no number here means anything.
2. `ccpo-*` — the method. One arm, the best-justified configuration.
3. `grpo-base` — the published GRPO figure (72.8 / 70.1) is on exactly this model
   and config, so this arm is the most substitutable by the literature. Run it
   only if the first two leave room.

Comparing arms at a **common step count** is fair even if that count is below 100;
comparing a 100-step arm against a 40-step one is not.

## Stopping rules

`early_stop_patience=6` evaluations without a new best, never before step 30
(`early_stop_min_steps`). At `test_freq=5` that is 30 steps of no improvement.
Any arm that has not cleared GRPO's published 70.1 by step 60 is not going to
reach SOTA and its GPUs are better spent on the next variant.
