# CCPO hypotheses — brainstorm and tracker

Running list of what we believe, what we have tested, and what is worth testing
next. Add a hypothesis here **before** running the arm that tests it, so the
prediction is on record and cannot be rewritten after the fact.

**Legend** — `[ ]` open · `[~]` in flight · `[x]` settled · `[!]` refuted

Each settled entry carries: the arm that tested it, the numbers, and what follows.

---

## Open — ranked by expected value

### [ ] H-A. Uncertainty should measure *grouping relevance*, not penalise within-bucket variance

**The inversion.** `λ_i = τ² / (τ² + ρ²·s²·var_gain)`. Within-bucket variance `s²`
sits in the **denominator**, so **high within-bucket variance → lower λ → less
conditioning**. But the method's premise is that high within-bucket return variance
*is the evidence of conflation* — distinct states sharing one observation — which
is exactly when conditioning should matter most. The rule suppresses conditioning
precisely where the thesis says it is needed.

It is defensible as pure bias-variance reasoning (a noisy neighbourhood gives a
noisy LOO estimate), but it cannot distinguish:

* **`s²` is noise** — nothing to condition on, back off. Correct behaviour.
* **`s²` is structure φ could explain** — conflation, condition harder. Currently
  penalised identically.

**Also: λ tests the wrong thing for judging φ.** `d = b_obs − b_LOO` only detects
whether φ shifts the *mean* of the baseline. A φ that correctly orders neighbours
by similarity, but whose reweighting happens not to move that mean, reads as
exactly zero signal. So "λ = 0" is weaker evidence about φ than it looks.

**Proposed replacement.** Decompose `s²` into the part φ explains and the part it
does not, and let *that* drive trust:

```
R²_bucket = 1 − Var(target | phi-weighted) / Var(target)
lambda    driven by R², not by s² alone
```

**First step, already shipped:** `ccpo/phi_rel_corr` — within-bucket correlation
between φ-distance and |target difference|, over the pairs that actually feed the
estimator. Diagnostic only; never fed back into the weights, so the reward-free
leave-one-out argument is untouched.

**Predictions.**
* `phi_rel_corr > 0` while `λ = 0` ⇒ φ carries signal the rule is discarding, and
  the fix is in the rule. **Act on it.**
* `phi_rel_corr ≈ 0` ⇒ the premise itself does not hold on ALFWorld; no rule
  change repairs that. **Stop pursuing conditioning.**

Status: measured for the first time by `ccpo-mem-20260906`.

---

### [~] H-B. Compaction makes φ informative — and has never actually been evaluated

Inside a bucket the observation is **constant by construction**, and without a
digest the prompt carries only `step_count` plus the most recent `history_length`
(2) turns. So φ can separate "step 5 from step 15" but not "has this agent already
searched here twice". `n_unique`, `revisit` and `progress` summarise the whole
episode and appear nowhere in the prompt unless compaction is on.

Prior work found **frozen φ + memory was the only arm beating plain GRPO**
(0.625 vs 0.604) — but see B-3 below: that result was obtained with the digest
rendered backwards, so compaction has never been evaluated as designed.

Independent support from outside: HGPO's `K=2 → K=4` (more prompt history) bought
**more than their entire estimator contribution** (+6.77 vs +5.40 out-of-distribution).

Arm: `ccpo-mem-20260906`. Watch `phi_rel_corr`, `λ`, then `episode/length/mean`
(the mechanism claim is that remembering where it searched shortens episodes:
58.7% of turns revisit an already-seen observation, failures 2.68× as often).

---

### [ ] H-C. The learned successor-feature φ

`ccpo/learned.py` — `LearnedPhi` + action-conditioned successor features +
learned affinity, TD-trained reward-free. This is §02's actual method and **has
never been wired in**. Cost: plumbing plus TD training inside the rollout loop.
Prior evidence is weak (Phase 1 found no advantage over a null control), so this
ranks below H-A and H-B despite being the headline method.

### [ ] H-D. Similarity gate raises bucket occupancy

`ACG_CCPO_SIM=0.95` — GiGPO ships this (`are_similar`, SequenceMatcher). Would
raise `n_eff` and cut the 33% singleton rate. **Deprioritised**: support is not the
binding constraint (see H-2), and GiGPO has the same gate natively, so a
CCPO-with-gate vs GiGPO-without comparison would confound the gate with the
estimator.

### [ ] H-E. G²PO's edge term

`ACG_CCPO_EDGE_W>0` adds `V(next) − V(current)`, standardised per task. Untested.
Note `V(g)` carries trajectory-length information through `γ^(T−t)`, so watch
`acc_len_corr` for a length bias.

---

## Settled — refuted

### [!] H-1. Context conditioning improves credit assignment on ALFWorld

The central hypothesis. **λ = 0.000 in every variant tried**, ~1,500 credited
samples per step:

| arm | φ | target | λ | effect_rel | r_vs_gigpo |
|---|---|---|---|---|---|
| `ccpo-base` | hidden | return | 0.000 | 0.0000 | +0.977 |
| `ccpo-nextnode` | hidden | nextnode | 0.000 | 0.0000 | +0.530 |
| (mis-gated, ran bow) | bow + explicit ctx | nextnode | 0.000 | 0.0000 | +0.523 |
| `ccpo-ctx` | hidden+ctx | nextnode | 0.000 | 0.0000 | +0.505 |

`Var(d) = s²(1/n_eff − 1/J)` is **exactly the permutation null** — verified
numerically at 1.0005 ± 0.0155 over 400 configurations × 4000 shuffles. So λ asks
"does the actual φ-weighting shift the baseline more than a *random* reweighting of
the same neighbours?" and answers no, every time.

This is the production-scale test of the write-up's own **E5**, previously a null
on 15–22 pairs and explicitly not a refutation. It is now one — **for the mean
shift**. See H-A: the ordering question is still open.

### [!] H-2. The inertness is thin bucket support

Refuted. `n_eff` 4.86–4.93 of a maximum 7, mean bucket size 8.4, 90% of samples
credited, 758 buckets per step. Every condition the method needs is met.

**Correction:** the `n_eff ≈ 2.1` figure quoted earlier came from a smoke run with
`group_size=4`, where J ≤ 3 by construction — 2.1 was 70% of the ceiling, not thin
support. Never quote a support number from a reduced-group run.

### [!] H-3. The inertness is the shrinkage rule

Refuted. `eb`, `eb_pooled` and `eb_hier` independently agree. Two of the three were
built specifically to rule this out:

* `eb_pooled` — James-Stein, one λ per batch. Fixed the ~2-degrees-of-freedom
  problem in `eb`; still 0.
* `eb_hier` — `τ² = max(0, E[d²] − E[Var(d)])` from the batch, then
  `λ_i = τ²/(τ² + ρ²Var_i)` per occurrence. Keeps ensemble power *and* per-occurrence
  adaptivity (λ sd 0.196 over 0.34–1.00 on a synthetic fixture, against `eb_pooled`'s
  0.000); still 0 on real data.

### [!] H-4. The inertness is noise in the target

Refuted, and this one was mine. Reasoning was: return-to-go on ALFWorld is
near-binary (0 or 10), so `Var(d)` swamps any reweighting — conditioning the
*baseline* cannot remove noise living in the *target*. So switch to `V(next node)`,
pooled over every sibling visit and far lower variance.

The target change **worked as designed** — `r_vs_gigpo` +0.977 → +0.530, `r_vs_g2po`
+0.393 → +0.719 — and **λ was still 0**. The target was not the obstacle.

---

## Settled — confirmed (corrections, not hypotheses)

### [x] C-1. The reference estimator was mislabelled

`G[i] − mean(G[bucket])` is **GiGPO's** step advantage in `mean_norm` mode, not
G²PO's. So the repo's headline "corr(A_CC, A_G2PO) = 0.956" says CCPO tracks
**GiGPO**; against G²PO's actual estimator it had never been compared. Now
measured: **+0.39**. The two references are genuinely different quantities.
G²PO port verified against the published implementation to ≤2.1e-07
(`tests/test_g2po_port.py`).

### [x] C-2. The two advantage terms were on mismatched scales

Episode term mean-centred in reward units (|A| ≈ 2.5–7.5 on ALFWorld), step term
always standardised (|A| ≈ 1). So `step_advantage_w = 1` was in effect a ~5×
**down-weight** of the entire context contribution. Nothing chose this; it was two
defaults meeting. Both now follow one `mode`, defaulting to `mean_std_norm` — what
the G²PO reference sets, and where verl's GRPO arm already is. Measured
`|A_EP|/|A_CC|` = 1.06–1.17 after the fix.

**Candidate explanation for "the context term is inert" in §05c that does not
require the estimator to be wrong.**

### [x] C-3. The compaction digest was rendered backwards

The digest was shown in its **eviction** order (informative before no-effect,
recent before old), so the agent read its own history most-recent-first —
inverting the causality of a sequential plan, which is what the digest exists to
convey. Caught when a memory-on arm solved **0 of 128** episodes at step 1 where
memory-off solved 7–16, with actions still 124/128 valid and 120 admissible.

Eviction priority and display order are different concerns. Fixed.

**This ordering has been in the codebase since compaction was written**, so the
"frozen φ + memory beats GRPO" result was obtained with the history inverted.
Compaction has never been evaluated as designed.

### [x] C-4. φ never saw the trajectory context

E4 says φ must be a function of both observation and context. The `bow` path
thermometer-codes `{t, n_unique, revisit, progress}`; the `hidden` path computes
`derive_context()` and **never consults it**. Added `ACG_CCPO_PHI=hidden+ctx`,
with `tests/test_phi_context.py` measuring `effect_rel` 0.0007 → 0.0169 → 0.1058
(hidden → hidden+ctx → bow) on a fixture where returns depend on context.

### [x] C-5. The ALFWorld prompt had drifted from the reference

Ours told the model to reason "concisely" and gave a one-shot example with a
~25-token think block — added for Qwen3, which could not reliably emit action tags.
Reverted to reference wording. Response length 54 → 96 tokens.

**But the argument for reverting was wrong**: I expected longer reasoning to help;
train success went *down*. And the 0.843 valid-action ratio I worried about
self-corrected to 0.996 within three steps — RL fixed the format unaided.

---

## Measurement notes — things that will mislead you

* **`E_w` is not evidence about φ.** `τ_b` is the bucket's *median* distance, so
  `E_w ≈ exp(−1) ≈ 0.37` for essentially any distance distribution. It is
  scale-invariant by construction and absorbs the collapse it was meant to detect.
* **`perf/max_memory_reserved_gb` is an aggregate across pools.** It read 103.8 GB
  against a 102.6 GB card. Sample `nvidia-smi` during the update instead.
* **Never quote `n_eff` from a reduced-`group_size` run** — it is capped by `J`.
* **A degenerate batch fakes agreement.** With every episode reward 0, all node
  values are 0 and `r_vs_gigpo`/`r_vs_g2po` both read ~0.97. Check `reward_sum`
  before reading any correlation.
* **`phi_is_hidden` had an exact-match bug** (`== "hidden"` vs `hidden+ctx`), which
  silently ran `bow` and made the guard metric report a φ failure that was really a
  gating failure. Same bug in the capture gate in `fsdp_workers.py`.
