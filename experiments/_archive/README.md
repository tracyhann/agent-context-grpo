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
