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

## Phase 3 — if a winner emerges

Repeat the winning arm on 2 further seeds. Held-out evaluation here is stochastic
by design (T=0.4, 128 episodes, ~±0.048 measured), so a single run is not a result.

## Stopping rules

`early_stop_patience=6` evaluations without a new best, never before step 30
(`early_stop_min_steps`). At `test_freq=5` that is 30 steps of no improvement.
Any arm that has not cleared GRPO's published 70.1 by step 60 is not going to
reach SOTA and its GPUs are better spent on the next variant.
