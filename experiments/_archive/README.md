# Archived partial runs

Kept for the record, not as results. Each was stopped deliberately once it had
answered something, and the reason is in the file name and below.

## `gigpo-hardened-prompt-metrics.jsonl`

11 steps of `gigpo-repro` under the **Qwen3-hardened ALFWorld prompt** — the one
that told the model to reason "concisely" and gave a one-shot example with a
~25-token think block. Held out **0.195 at step 5, 0.211 at step 10** against a
published GiGPO (K=2) figure of 90.16, with `response_length/mean` stuck at 54
tokens.

Stopped when the prompt was identified as a deviation from the reference wording:
the hardening existed for Qwen3, which could not reliably emit action tags, while
Qwen2.5-1.5B-Instruct parses at 99.6% valid untrained. Kept because it is the only
measurement of what that prompt costs, and the comparison against the reverted
prompt is informative on its own.

## `ccpo-base-inert-metrics.jsonl`

2 steps of CCPO with `shrink=eb_hier`, reference prompt, memory off. Stopped
because the estimator is **provably inert**, and every further step would have
trained something numerically identical to the baseline:

| | step 1 | step 2 |
|---|---|---|
| `lam_u_mean` | 0.000 | 0.000 |
| `effect_mean` / `effect_rel` | 0.000 / 0.000 | 0.000 / 0.000 |
| `r_vs_gigpo` | +0.977 | +0.977 |
| `r_vs_g2po` | +0.393 | +0.334 |
| `n_eff_mean` | 4.93 | 4.81 |
| `bucket_size_mean` | 8.44 | — |
| `bucket_singleton_frac` | 0.33 | 0.35 |
| `live_frac` | 0.90 | 0.90 |

**Not a support problem.** ~5 effective neighbours, mean bucket size 8.4, 90% of
samples credited, 758 buckets, `phi_is_hidden=1`, `E_w=0.376`. Every condition the
method needs is met.

**Not a shrinkage-rule problem.** All three rules agree: `eb_hier` gives 0 because
`tau^2 = max(0, E[d^2] - E[Var(d)])` is 0, `eb` fires on 2% of samples then 0.3%,
`eb_pooled` gives 0. The disagreement between the phi-weighted and the uniform
baseline is entirely explained by sampling noise.

**Mechanism.** `d = b_obs - b_LOO` reweights a handful of the *same* neighbour
returns. Return-to-go on ALFWorld is near-binary (0 or 10), so
`Var(d) = s^2 (1/n_eff - 1/J)` is large, and reweighting five high-variance numbers
cannot move their mean beyond that noise. Conditioning the *baseline* cannot
remove noise that lives in the *target* -- which is the argument for `nextnode`.

Also the first measurement of CCPO against G2PO's real estimator: **+0.39**, against
+0.98 for GiGPO. The two references are genuinely different quantities, which is
what the attribution correction claimed.

## `ccpo-nextnode-step1-metrics.jsonl`

1 step of CCPO with `target=nextnode`, `shrink=eb_hier`. Kept because it settles a
hypothesis, read against one step of `ccpo-base`:

| | `target=return` | `target=nextnode` |
|---|---|---|
| `lam_u_mean` | 0.000 | **0.000** |
| `effect_rel` | 0.0000 | **0.0000** |
| `r_vs_gigpo` | +0.977 | **+0.530** |
| `r_vs_g2po` | +0.393 | **+0.719** |
| `acc_len_corr` | +0.005 | -0.011 |

The target change did what it was designed to do — the credit is now genuinely
different from GiGPO and much closer to G2PO — but **lambda is still 0**. So "the
target is too noisy for the conditioning to register" is refuted: with the pooled,
far lower variance successor value, tau^2 is still 0.

Everything that moved came from the target and the leave-one-out exclusion. The
context conditioning contributed nothing under either target, which points at phi
rather than the target. With lambda = 0 this arm is G2PO's component 1 with
leave-one-out instead of self-inclusive — a real variant, but not a
context-conditioned one.

## memory arm, first attempt (reversed digest) — not kept as metrics

Step 1 with `compact_budget=512` solved **0 of 128 episodes**, where memory-off
arms solved 7-16 on the same seed. Actions were still 124/128 parser-valid and 120
admissible: legal moves, no progress. `lambda` was 0.000 again, but that step is
degenerate for judging the estimator — with every episode reward 0 all node values
are 0, which is why `r_vs_gigpo` and `r_vs_g2po` both read ~0.97.

Cause: the digest was rendered in its **eviction** order (informative before
no-effect, recent before old), so the agent read its own history backwards, most
recent action first. Fixed in `agent_system/memory/compact.py` — eviction still
decides what survives the budget, then the kept lines are re-sorted chronologically.

Worth recording as a near-miss: the reversed order had been in the codebase since
compaction was written, and "frozen phi + memory" was nonetheless the only arm the
prior work found beating GRPO. Whatever that result was, it was obtained with the
history inverted.
