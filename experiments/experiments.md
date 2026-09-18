# CCPO experiments

Twenty-three runs: **10 main** (4 variants × 2 backbones, plus one WebShop-only
variant × 2 backbones) and **13 ablations** (6 variants × 2 benchmarks, plus one
WebShop-only variant, 1.5B throughout).

Every method variant below has a name, a location in the repo, and the one equation
that separates it from the base estimator in §1. Nothing else differs between any two
arms in this document — each is a single-delta change, asserted by the guards.

---

## 0. Protocol — identical in every run

| | |
|---|---|
| steps | **150** (`total_epochs=150`, early stopping off) |
| checkpoints | **3**: `step<N>-best`, `step100-pin`, `step<N>-last` (`pin_steps=100`, `keep_ckpts=1`) |
| seed | 0, one seed per arm |
| eval | every 5 steps, T=0.4 with sampling |
| attention | FA2 on both sides — vLLM `FLASH_ATTN`, trainer `remove_padding=True`; A100/H100 |

To resume or score the pin, copy it back first (verl asserts the path contains
`global_step_`): `cp -al <ckpts>/step100-pin <ckpts>/global_step_100`.

---

## 1. Notation and the base estimator

**Indices.** `i` is one occurrence — one turn of one rollout. `j(i)` is its trajectory,
`B(i)` its bucket: the exact `(task_uid, anchor_obs)` match, the same gate GiGPO and
G²PO use. Neighbours are **cross-trajectory only**:

```
N(i) = { b ∈ B(i) : j(b) ≠ j(i) }          J = |{ j(b) : b ∈ N(i) }|
```

Leaving *own trajectory* out — not just occurrence `i` — is what keeps the baseline
valid: a revisit later in the same rollout is downstream of the action being scored.

**Target.** What the step credit predicts:

```
TGT_i = Σ_{t ≥ i, same trajectory} γ^(t−i) · r_t          γ = 0.95
```

**Affinity features.** `h_i` is the frozen reference policy's last-prompt-token hidden
state (free — that forward pass already runs for the KL term):

```
φ_i = [ whiten₃(h_i) ; w_ctx · thermo(t, n_unique, progress, revisit) ]     both blocks L2-normalised
```

`whiten₃` centres, removes the top 3 principal directions and L2-normalises, batch-wide.
The context block carries what the prompt cannot: the prompt holds `step_count` plus the
last 2 turns, so a hidden state separates "step 5 from step 15" but not "has this agent
already searched here twice".

**Kernel and baseline.**

```
d_ib  = ‖φ_i − φ_b‖₂
τ_B   = 0.15 · median{ d_ab : a < b ∈ B }
w_b   = exp( −d_ib / τ_B )                                            ← (K)
b_loo = Σ_{b ∈ N(i)} w_b · TGT_b  /  Σ_{b ∈ N(i)} w_b                 ← (L)
```

**Credibility prior.** `b_task,i` is the leave-own-trajectory-out mean of `TGT` over the
whole task — the Bühlmann form, so a thinly supported node leans on it and a
well-supported one keeps its own evidence:

```
λ_k   = J / (J + κ)                     κ = 2                         ← (C)
base_i = λ_k · b_loo + (1 − λ_k) · b_task,i                           ← (B)
```

**Step advantage.** The second term is G²PO's value gain, standardised per task, with
`V(g)` the group-aggregated node value `mean over visits of γ^(T−t)·R`:

```
A_CC,i = ( TGT_i − base_i ) + z_task( V(next_i) − V(cur_i) )          ← (S)
```

**Total.**

```
A[i,t] = ep_w · A_EP[i,t] + step_w · Z( A_CC,i ) · mask[i,t]          ← (T)

  A_EP    group-relative episode advantage over the task's 8 sibling rollouts (plain GRPO)
  ep_w = step_w = 1
  Z       per-task standardisation over the estimator's live rows on ALFWorld
          (adv_mode=mean_std_norm); the IDENTITY on WebShop (mean_norm), where
          neither term is standardised and both stay in reward units
```

Two notes that apply everywhere. The mixing weight λ between `b_loo` and the uniform
mean `b_obs` is **pinned to 1** (`ccpo_lam_fix=1.0`) — the empirical-Bayes rule measures
λ = 0.000 on every real batch, so without the pin φ is computed and discarded. And rows
whose node holds no sibling trajectory are credited against a task-wide bucket at
level 1 (`ccpo_backoff_task=1`), where (C)–(B) do **not** apply; this keeps `live_frac`
at 1.0 (~8% of rows on ALFWorld, ~20% on WebShop).

---

## 2. Main methods

Run code: **`official-repo/ccpo/`** — definition in `arms.py` (`BASE`, `BENCHMARK`,
`METHODS`), launcher `run.py`, guard `test_arms.py`.

```bash
python3 official-repo/ccpo/run.py --list
```

| # | Name | Δ from §1 | benchmark |
|---|---|---|---|
| M1 | `CCPO-ATTNCRED` | — (the base estimator) | ALFWorld |
| M2 | `CCPO-ATTNCRED-WS` | (D) dense target | WebShop |
| M3 | `CCPO-ATTNCRED-CTXADV` | (T) `ep_w = 0` | ALFWorld |
| M4 | `CCPO-ATTNCRED-CTXADV-WS` | (T) `ep_w = 0` **and** (D) | WebShop |
| M5 | `CCPO-ATTNCRED-CTXADV-RETURN-WS` | (T) `ep_w = 0`, **without** (D) | WebShop |

### M1 · `CCPO-ATTNCRED` — ALFWorld

The base estimator of §1, unchanged. `TGT` is the γ-discounted return-to-go of the
binary 10/0 environment reward.

```bash
python3 official-repo/ccpo/run.py --method attncred --benchmark alfworld \
    --backbone 1.5b --gpus <4+ ids>       # Experiment 1: Qwen2.5-1.5B, ≥4 GPUs
python3 official-repo/ccpo/run.py --method attncred --benchmark alfworld \
    --backbone 7b   --gpus <8+ ids>       # Experiment 2: Qwen2.5-7B,   ≥8 GPUs
```

exp-ids: `ccpo-attncred-alfworld-1.5b`, `ccpo-attncred-alfworld-7b`.

### M2 · `CCPO-ATTNCRED-WS` — WebShop, dense score

#### The WebShop reward, and which channel sees what

Both GiGPO and the released HGPO recipe train WebShop on **terminal success**. Their
shared environment wrapper (`verl-agent/agent_system/environments/env_package/webshop/envs.py`,
upstream — we do not patch it) converts WebShop's graded score into

```
      r_T = 10   if the episode ends with a perfect score (score == 1.0)
      r_T = 0    otherwise, partial matches included
```

and retains the original graded score as `info['task_score']` for reporting. We inherit
that unchanged: `env_manager.py:683-686` reads `info['won']` into `success_rate` and
`info['task_score']` into the reported score, and the reward manager writes the episode
reward, unaltered, onto the last response token.

*Verified on `ccpo-attncred-ws-20260914`, 84 steps:* `episode/reward/mean` equals
`10 × episode/success_rate` to four decimals (mean absolute difference **0.0000**),
against **3.2214** for the graded-score hypothesis, and the reward takes only the values
0 and 10 (`reward/min` 0.0, `reward/max` 10.0 at every step). The binary channel is what
reaches the trainer.

That is also the problem: 30–69% of task groups score zero on **every** rollout under
this reward, and a group-relative estimator computes exactly zero advantage there. So
this arm points the **step channel** at the dense score, which still varies inside those
groups:

```
(D)   TGT_i = Σ_{t ≥ i} γ^(t−i) · ( 10 · score_t )  −  0.1 · 1[action invalid]
```

The ×10 keeps the dense target in the binary reward's units, so the invalid-action
penalty and the episode term keep their relative sizes.

**Step channel only.** `A_EP` in (T) keeps the published binary 10/0, `V` in (S) is
still built from it, and the reported Success and Score both come from the environment
manager — so the outcome signal and the reported numbers stay comparable to GiGPO and
HGPO. What deviates is what the credit-assignment channel regresses on, and nothing else.

For the strictly-comparable arm — binary on both channels, as the published runs are —
add `--set ccpo_target=return`. That is what `ccpo-attncred-ws-20260914` ran.

Horizon 15 turns, not ALFWorld's 50. `Z` in (T) is the identity here (see §1).

```bash
python3 official-repo/ccpo/run.py --method attncred --benchmark webshop \
    --backbone 1.5b --gpus <4+ ids>       # Experiment 1: Qwen2.5-1.5B, ≥4 GPUs
python3 official-repo/ccpo/run.py --method attncred --benchmark webshop \
    --backbone 7b   --gpus <8+ ids>       # Experiment 2: Qwen2.5-7B,   ≥8 GPUs
```

exp-ids: `ccpo-attncred-ws-1.5b`, `ccpo-attncred-ws-7b`.

### M3 · `CCPO-ATTNCRED-CTXADV` — context advantage only, ALFWorld

The standard episode advantage is ablated from the total. `A_EP` is plain GRPO on the
trajectory return and is precisely what this method **shares** with the baselines it is
measured against; what remains is the normalised, context-conditioned, credit-adapted
term — plus the edge term, which (S) already folds in.

```
(T′)  A[i,t] = step_w · Z( A_CC,i ) · mask[i,t]                      ep_w = 0
```

HGPO ships this shape and reports that *adding* the trajectory-level advantage hurt, so
this is also a direct test of that claim on our harness.

```bash
python3 official-repo/ccpo/run.py --method attncred-context-adv-only \
    --benchmark alfworld --backbone 1.5b --gpus <4+ ids> # Experiment 1: 1.5B, ≥4 GPUs
python3 official-repo/ccpo/run.py --method attncred-context-adv-only \
    --benchmark alfworld --backbone 7b   --gpus <8+ ids> # Experiment 2: 7B,   ≥8 GPUs
```

exp-ids: `ccpo-attncred-ctxadv-alfworld-1.5b`, `ccpo-attncred-ctxadv-alfworld-7b`.

**Watch `ccpo/live_frac`.** With `A_EP` gone, an uncredited occurrence contributes
advantage exactly 0 — no signal at all for those tokens. `ccpo_backoff_task=1` is what
keeps that from happening; if `live_frac` drops below 1.0 the arm is training on a
subset of its batch.

### M4 · `CCPO-ATTNCRED-CTXADV-WS` — context advantage only, WebShop

(T′) **and** (D) together: no episode term, dense step target.

```bash
python3 official-repo/ccpo/run.py --method attncred-context-adv-only \
    --benchmark webshop --backbone 1.5b --gpus <4+ ids>  # Experiment 3: 1.5B, ≥4 GPUs
python3 official-repo/ccpo/run.py --method attncred-context-adv-only \
    --benchmark webshop --backbone 7b   --gpus <8+ ids>  # Experiment 4: 7B,   ≥8 GPUs
```

exp-ids: `ccpo-attncred-ctxadv-ws-1.5b`, `ccpo-attncred-ctxadv-ws-7b`.

### M5 · `CCPO-ATTNCRED-CTXADV-RETURN-WS` — context advantage only, binary target, WebShop

M4 on WebShop's **own** reward. Same context-advantage-only shape, but the step channel
predicts the published binary return-to-go instead of the dense score:

```
(T′)   A[i,t] = step_w · Z( A_CC,i ) · mask[i,t]            ep_w = 0
(D⁻)   TGT_i  = Σ_{t ≥ i} γ^(t−i) · r_t  −  0.1·1[invalid]  r = the binary 10/0
```

So M5 borrows nothing from this project's WebShop adaptation. Against **M4** it isolates
the dense target under the no-episode-term condition — the only place in the main set
where `score` and `return` are compared on the same shape, same benchmark, same
backbone. Against **M3** it asks whether the ALFWorld reading survives the benchmark
change once the target is held fixed.

Note this is *context advantage only*, not *ours only*: `A_CC` still carries G²PO's edge
term (`ccpo_edge_w=1.0`), as M3 and M4 do. A6-R is the arm that drops both.

**WebShop only.** On ALFWorld `ccpo_target` is already `return`, so the delta would
collapse to M3 and the run would be a duplicate; `run.py` refuses `--benchmark alfworld`
for it.

**Watch the zero-advantage population.** Under the binary reward 30–69% of task groups
score zero on *every* rollout, where a group-relative estimator computes exactly zero
advantage — and with `ep_w = 0` there is no episode term to cover those rows either.
Read `ccpo/live_frac` and `ccpo/effect_rel` at step 1 before trusting anything later.

```bash
python3 official-repo/ccpo/run.py --method attncred-context-adv-only-return \
    --benchmark webshop --backbone 1.5b --gpus <4+ ids>  # Experiment 1: 1.5B, ≥4 GPUs
python3 official-repo/ccpo/run.py --method attncred-context-adv-only-return \
    --benchmark webshop --backbone 7b   --gpus <8+ ids>  # Experiment 2: 7B,   ≥8 GPUs
```

exp-ids: `ccpo-attncred-ctxadv-ret-ws-1.5b`, `ccpo-attncred-ctxadv-ret-ws-7b`.

---

## 3. Ablations

Run code: **`official-repo/ablations/`** — definitions in `ablations.py` (`ABLATIONS`),
launcher `run.py`, guard `test_ablations.py`.

```bash
python3 official-repo/ablations/run.py --list
```

All thirteen are **Qwen2.5-1.5B, ≥4 GPUs, 150 steps**. Each runs on both benchmarks, and the
control is the main arm of the *same* benchmark — so a WebShop ablation inherits (D)
and stays paired with what it ablates.

| # | Name | Δ from §1 | one key |
|---|---|---|---|
| A1 | `CCPO-ATTNCRED-HARDGATE` | (K) → binary threshold | `ccpo_wmode=hard` |
| A2 | `CCPO-ATTNCRED-NOTASK` | (B) → `b_loo` only | `ccpo_prior_kappa=0` |
| A3 | `CCPO-ATTNCRED-EVENBLEND` | (C) → constant ½ | `ccpo_lk_fix=0.5` |
| A4 | `CCPO-ATTNCRED-NOCTX` | φ → hidden state only | `ccpo_phi=hidden` |
| A5 | `CCPO-ATTNCRED-COS` | (K) → cosine, no kernel | `ccpo_wmode=cos` |
| A6 | `CCPO-ATTNCRED-NOEDGE` | (S) → node term only | `ccpo_edge_w=0` |
| A6-R | `CCPO-ATTNCRED-NOEDGE-RETURN-WS` | (S) → node term only, **and** (D) undone | `ccpo_edge_w=0` + `ccpo_target=return` |

### A1 · `CCPO-ATTNCRED-HARDGATE` — binary hard gating

```
(K₁)  w_b = 1[ d_ib ≤ τ_B ]          (if nothing survives, the nearest trajectory is kept)
```

With 0/1 weights (L) becomes the plain average of the survivors. τ is unchanged, so both
modes select the same neighbourhood and differ only in how they weight inside it.
**Asks:** does the gain come from *ordering* neighbours by φ-distance, or only from
*restricting* which ones count?

**Watch** `ccpo/E_w`, now the survival fraction: on ALFWorld it measured 0.011 — about 1
neighbour in 90 clears `τ = 0.15·median` — so the arm runs close to a
nearest-trajectory baseline, and `effect_rel` rose to 0.70 against soft's 0.28.

```bash
python3 official-repo/ablations/run.py --ablation hard-gate --benchmark alfworld --gpus <4+ ids>   # 1.5B, ≥4 GPUs
python3 official-repo/ablations/run.py --ablation hard-gate --benchmark webshop  --gpus <4+ ids>   # 1.5B, ≥4 GPUs
```

### A2 · `CCPO-ATTNCRED-NOTASK` — without task baseline fallback

```
(B₂)  base_i = b_loo                                      κ = 0, b_task not used
```

λ stays pinned at 1, so this is the context baseline alone, and `b_task` is not even
computed. **Asks:** does leaning a thinly supported node on the task mean buy anything?
**Watch** `ccpo/lam_k_mean` → 1.000.

`ccpo_backoff_task` stays 1: the level-1 task *bucket* is a different mechanism (it
recomputes `b_loo` over the task rather than mixing in a prior) and is what holds
`live_frac` at 1.0. To ablate that instead, add `--set ccpo_backoff_task=0`.

```bash
python3 official-repo/ablations/run.py --ablation no-task-baseline --benchmark alfworld --gpus <4+ ids>   # 1.5B, ≥4 GPUs
python3 official-repo/ablations/run.py --ablation no-task-baseline --benchmark webshop  --gpus <4+ ids>   # 1.5B, ≥4 GPUs
```

### A3 · `CCPO-ATTNCRED-EVENBLEND` — without evidential-support shrinkage

```
(C₃)  λ_k = ½       ⇒     base_i = ½ · b_loo + ½ · b_task,i
```

The prior stays; its *support weighting* goes. κ no longer enters. **Asks:** is the
Bühlmann credibility form doing the work, or merely the presence of a task prior at some
fixed ratio? Paired with A2 this separates *the prior exists* from *the prior is weighted
by evidence*. **Watch** `ccpo/lam_k_mean` pinned to 0.500, against `J/(J+2)` in M1.

```bash
python3 official-repo/ablations/run.py --ablation even-blend --benchmark alfworld --gpus <4+ ids>   # 1.5B, ≥4 GPUs
python3 official-repo/ablations/run.py --ablation even-blend --benchmark webshop  --gpus <4+ ids>   # 1.5B, ≥4 GPUs
```

### A4 · `CCPO-ATTNCRED-NOCTX` — without the context summary vector

```
(φ₄)  φ_i = whiten₃(h_i)                     no context block concatenated
```

Only the hidden states of the recent obs-action pairs remain. **Asks:** does
whole-episode context earn its place, or does the prompt already carry it? **Watch**
`ccpo/phi_rel_corr`: it rose 0.01 → 0.24 over training on ALFWorld with the block in and
stayed near 0.02 on WebShop, so this ablation may cost little there and a lot here.

```bash
python3 official-repo/ablations/run.py --ablation no-context-vector --benchmark alfworld --gpus <4+ ids>   # 1.5B, ≥4 GPUs
python3 official-repo/ablations/run.py --ablation no-context-vector --benchmark webshop  --gpus <4+ ids>   # 1.5B, ≥4 GPUs
```

### A5 · `CCPO-ATTNCRED-COS` — cosine similarity instead of the attention kernel

```
(K₅)  w_b = max( cos(φ_i, φ_b), 0 )          no τ   (if all clip to 0, the nearest is kept)
```

Weight falls off linearly in the angle instead of exponentially in the distance, so
distant neighbours keep far more weight. Negative cosines clip to zero — a weighted mean
needs non-negative weights, and a neighbour pointing the other way is evidence of
nothing, not evidence against. **Asks:** does `exp(−d/τ)` earn its place over a plain
similarity score? **Watch** `ccpo/E_w` rise sharply; if `effect_rel` collapses, the
kernel's sharpness was the mechanism.

```bash
python3 official-repo/ablations/run.py --ablation cosine --benchmark alfworld --gpus <4+ ids>   # 1.5B, ≥4 GPUs
python3 official-repo/ablations/run.py --ablation cosine --benchmark webshop  --gpus <4+ ids>   # 1.5B, ≥4 GPUs
```

### A6 · `CCPO-ATTNCRED-NOEDGE` — without the edge advantage

```
(S₆)  A_CC,i = TGT_i − base_i                    the value-gain term dropped
```

The second half of (S), `z_task( V(next_i) − V(cur_i) )`, is G²PO's edge term: `V` is
the group-aggregated node value, so it scores **where the action moved the agent**
rather than the value of where it stood. Removing it leaves the node term alone — the
context-conditioned baseline and nothing else.

**Asks:** how much of the step credit's effect is the context-conditioned baseline, and
how much is the edge term it is summed with? Every arm in this project has carried
`ccpo_edge_w=1.0`, so no run to date separates the two.

**Watch** `ccpo/edge_cov` → 0 and `ccpo/adv_cc_absmean`. The edge term is standardised
per task, so it contributes about one unit of scale that the node term no longer has
beside it. On ALFWorld `A_CC` is standardised again per task afterwards (`Z` in (T)), so
the arm mostly reweights the two halves; on WebShop `Z` is the identity, so the step
channel gets absolutely smaller as well as differently shaped.

This is **`ccpo_edge_w`, not `ccpo_target=nextnode`.** Both involve the successor node
and they are different knobs: `edge_w` weights the value-*gain* term added to the step
credit, while `target=nextnode` would change what the node term *predicts* — from the
return-to-go to `V(next(u))`. This ablation leaves the target alone.

```bash
python3 official-repo/ablations/run.py --ablation no-edge --benchmark alfworld --gpus <4+ ids>   # 1.5B, ≥4 GPUs
python3 official-repo/ablations/run.py --ablation no-edge --benchmark webshop  --gpus <4+ ids>   # 1.5B, ≥4 GPUs
```

### A6-R · `CCPO-ATTNCRED-NOEDGE-RETURN-WS` — no edge term, binary target, WebShop

A third run in the A6 family, and the only **two-key** ablation in the set:

```
(S₆)  A_CC,i = TGT_i − base_i                       edge term dropped
(D⁻)  TGT_i  = Σ_{t ≥ i} γ^(t−i) · r_t              r = the published binary 10/0
      A[i,t] = A_EP[i,t] + Z(A_CC,i) · mask[i,t]    the episode term stays
```

Two keys because the question needs both: this is the arm in which **nothing is
borrowed** — not G²PO's value-gain term, and not the dense-score adaptation this
project added on top of the published protocol. What remains is the standard
`A_EP + A_CC` shape with `A_CC` built only from the exact-observation gate, the
φ-attention readout and the credibility prior, on WebShop's own reward.

**Asks:** on WebShop, with the published binary reward and no edge term, does the
context-conditioned baseline carry anything on its own?

**WebShop only.** On ALFWorld `ccpo_target` is already `return`, so this delta would
collapse to plain A6 and the run would be a duplicate. `run.py` refuses
`--benchmark alfworld` for it.

**Watch the zero-advantage population — it is the risk.** Under the binary reward
30–69% of task groups score zero on *every* rollout, and a group-relative estimator
computes exactly zero advantage there; that is why (D) exists. Read `ccpo/effect_rel`
and `ccpo/live_frac` beside the success curve. A flat curve here **with `effect_rel`
above zero** says the target was the binding constraint, not the estimator — which is
the finding that justifies the dense-score arms.

```bash
python3 official-repo/ablations/run.py --ablation no-edge-return-ws --benchmark webshop --gpus <4+ ids>   # 1.5B, ≥4 GPUs
```

---

## 4. Run matrix

| variant | ALFWorld 1.5B | ALFWorld 7B | WebShop 1.5B | WebShop 7B |
|---|---|---|---|---|
| | **≥4 GPUs** | **≥8 GPUs** | **≥4 GPUs** | **≥8 GPUs** |
| `CCPO-ATTNCRED` | M1·E1 | M1·E2 | M2·E1 | M2·E2 |
| `CCPO-ATTNCRED-CTXADV` | M3·E1 | M3·E2 | M4·E3 | M4·E4 |
| `CCPO-ATTNCRED-CTXADV-RETURN-WS` | — | — | M5·E1 | M5·E2 |
| `CCPO-ATTNCRED-HARDGATE` | A1 | — | A1 | — |
| `CCPO-ATTNCRED-NOTASK` | A2 | — | A2 | — |
| `CCPO-ATTNCRED-EVENBLEND` | A3 | — | A3 | — |
| `CCPO-ATTNCRED-NOCTX` | A4 | — | A4 | — |
| `CCPO-ATTNCRED-COS` | A5 | — | A5 | — |
| `CCPO-ATTNCRED-NOEDGE` | A6 | — | A6 | — |
| `CCPO-ATTNCRED-NOEDGE-RETURN-WS` | — | — | A6-R | — |

**Every run must be given its devices explicitly** — `--gpus` is required and there is
no default. `run.py` warns when the count is below the floor for that backbone: **≥4
GPUs for 1.5B, ≥8 for 7B**. All thirteen ablation runs are 1.5B, so all thirteen need ≥4.

`trainer.n_gpus_per_node` is derived from the number of ids you pass, and verl asserts
`train_batch_size × rollout.n % n_gpus == 0` — 16 × 8 = 128, so 4 and 8 both divide it
and 6 does not.

## 5. Reporting

One command per run. `--paper-table` prints the table below; without it you get the
per-type block plus the training-behaviour block.

```bash
python3 scripts/report_results.py --exp experiments/<exp-id> --paper-table [--markdown]
python3 scripts/report_results.py --exp experiments/<exp-id> --rows best,last,100,150   # full block
```

Rows are **step 100**, **best** (the highest held-out evaluation) and **final**
(step 150), each reported for both splits:

* **train** — that step's rollout batch, 16 tasks × 8 rollouts. It is *not* a
  checkpoint evaluation, and per-type training cells are noisy by construction: 128
  episodes spread over 6 task types is ~20 per type on a good step and 0 on some.
* **held-out** — the 128-episode validation draw at that step, T=0.4 with sampling.

### ALFWorld — example

Filled from `ccpo-attncred-150-20260913` to show the format; success rates in %,
turns in environment steps.

| checkpoint | split | Pick | Look | Clean | Heat | Cool | Pick2 | All | Turns |
|---|---|---|---|---|---|---|---|---|---|
| step 100 | train | 100.0 | 100.0 | 93.8 | 84.4 | 100.0 | 87.5 | 93.8 | 14.7 |
| step 100 | held-out | 89.2 | 50.0 | 86.4 | 80.0 | 72.5 | 76.4 | 78.9 | 15.2 |
| best @150 | train | 81.2 | 100.0 | 87.5 | 100.0 | 100.0 | 95.8 | 94.5 | 14.5 |
| best @150 | held-out | 100.0 | 100.0 | 100.0 | 100.0 | 95.8 | 71.7 | 94.5 | 13.8 |
| final @150 | train | 81.2 | 100.0 | 87.5 | 100.0 | 100.0 | 95.8 | 94.5 | 14.5 |
| final @150 | held-out | 100.0 | 100.0 | 100.0 | 100.0 | 95.8 | 71.7 | 94.5 | 13.8 |

The held-out **Turns** cells are illustrative: `val/length/mean` landed with this
revision, so runs predating it print `--` there and every new run fills it.

### WebShop — example

Both **Success** and **Score** are reported; every published WebShop table gives both
and one alone is not comparable. Filled from `ccpo-attncred-ws-20260914`, which stopped
at step 80 — hence the `n/a` row, which is what an incomplete run looks like.

| checkpoint | split | Success | Score | Turns |
|---|---|---|---|---|
| step 100 | train | n/a | n/a | n/a |
| step 100 | held-out | n/a | n/a | n/a |
| best @80 | train | 43.8 | 60.2 | 8.2 |
| best @80 | held-out | 45.7 | 67.7 | 8.5 |
| final @80 | train | 43.8 | 60.2 | 8.2 |
| final @80 | held-out | 45.7 | 67.7 | 8.5 |

### Where each cell comes from

Every column is a metric the trainer already writes to
`experiments/<exp-id>/outputs/metrics.jsonl`; nothing is derived after the fact.

| column | train | held-out |
|---|---|---|
| Pick … Pick2 | `episode/<type>_success_rate` | `val/<type>_success_rate` |
| All / Success | `episode/success_rate` | `val/success_rate` |
| Score (WebShop) | `episode/webshop_task_score (not success_rate)` | `val/webshop_task_score (not success_rate)` |
| Turns | `episode/length/mean` | `val/length/mean` |

`val/length/mean` is averaged over unique **trajectories**, not rows — the batch is
expanded by step, so a row-wise mean would weight long episodes by their own length.

### Alongside the table

**Window means are the statistic to trust:** 70–100 and 120–150, plus the mean over all
evaluations. A single evaluation carries SE ≈ ±3.3 points, and `best` is the maximum of
a noisy series, biased up by roughly 1.5 sd. One seed per arm: on ALFWorld the
same-config replicate spread is 1.79 points on a window mean and up to 15 on a single
step — no difference smaller than that is a result.

**Estimator diagnostics, by phase (1–50 / 51–100 / 101–150):** `effect_rel`,
`live_frac`, `lam_k_mean`, `n_eff_mean`, `bucket_size_mean`, `bucket_singleton_frac`,
`n_buckets`, `phi_rel_corr`. An arm with `effect_rel ≈ 0` changed nothing, and its
success number says nothing about the method.

**Cost and provenance:** step time, wall clock, peak disk, GPU count, exp-id, git sha,
seed, and the resolved paths of the three checkpoints.

## 6. Plots — written while the run trains

`scripts/exp_run.py` starts `scripts/plot_metrics.py --watch` on every launch (pass
`--no-plot` to suppress it), so both figures refresh during training and are on disk
whether or not the run finishes:

```
experiments/<exp-id>/plots/progress.png   training and held-out curves
experiments/<exp-id>/plots/ccpo.png       the estimator's own terms
```

`progress.png` carries the reported columns as curves — **held-out success rate**,
**train success rate**, **held-out and train success by task type** (ALFWorld, one
labelled line per type), **WebShop task score** (train and held-out), and **turns per
episode** (train and held-out on the same axes) — beside return, KL, entropy, response
length, truncation, valid-action ratio, grad norm, clip fraction, step time and the
time breakdown. Panels with no data are skipped, so each benchmark draws only its own.

`ccpo.png` is the panel that says whether the method did anything: `lam`, `n_eff`,
bucket occupancy and singleton fraction, `live_frac` / `lvl1_frac`, `E[w]`,
`phi_rel_corr`, `effect_mean` / `effect_rel`, correlation against the GiGPO and G²PO
references, and the ratio of the two advantage terms.

To refresh by hand, or to put two arms side by side:

```bash
python3 scripts/plot_metrics.py --exp experiments/<exp-id>
python3 scripts/plot_metrics.py --compare experiments/<a> experiments/<b> -o plots/compare.png
```

**Decision gate, step 1 of every run:** if `bucket_singleton_frac > 0.9` or
`effect_rel < 0.02`, the estimator is inert on that benchmark. Stop and report it — that
is a finding about where the method applies, and it costs one step instead of the budget.

---

## 7. Training hyperparameters

Resolved values, one table per benchmark, with the hydra key each one sets so any cell
can be checked against a run's `config.json` or `run.sh`. Both tables are identical
except where marked — the WebShop column of §7.2 lists only what *differs*, and the
differences are the published WebShop protocol (`exp_run.WEBSHOP_PROTOCOL`, G²PO's
`run_webshop.sh` verbatim, asserted by `tests/test_webshop_protocol.py`).

### 7.1 ALFWorld

| | value | hydra key |
|---|---|---|
| **Batch geometry** | | |
| tasks per step | 16 | `data.train_batch_size` |
| rollouts per task (group) | 8 | `env.rollout.n` |
| episodes per step | **128** | 16 × 8 |
| PPO mini-batch | 256 | `actor_rollout_ref.actor.ppo_mini_batch_size` |
| dynamic batching | on | `actor_rollout_ref.actor.use_dynamic_bsz` |
| token budget / GPU, update | 12288 | `actor_rollout_ref.actor.ppo_max_token_len_per_gpu` |
| token budget / GPU, log-prob | 24576 | `actor_rollout_ref.{ref,rollout}.log_prob_max_token_len_per_gpu` |
| micro-batch / GPU, update | 32 | `actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu` |
| micro-batch / GPU, log-prob | 64 | `actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu` |
| **Sequence lengths** | | |
| max prompt | 2048 tokens | `data.max_prompt_length` |
| max response | 512 tokens | `data.max_response_length` |
| over-long prompts | filtered; truncation is an error | `data.filter_overlong_prompts`, `data.truncation` |
| **Optimisation** | | |
| learning rate | 1e-6 | `actor_rollout_ref.actor.optim.lr` |
| critic warmup | 0 | `trainer.critic_warmup` |
| gradient checkpointing | on | `actor_rollout_ref.model.enable_gradient_checkpointing` |
| sequence packing | on | `actor_rollout_ref.model.use_remove_padding` |
| clip ratio, entropy coef, weight decay, LR schedule | **not overridden — verl defaults** | `actor_rollout_ref.actor.clip_ratio`, … |
| **Objective** | | |
| KL loss | on, coef 0.01, `low_var_kl` | `actor.use_kl_loss`, `actor.kl_loss_coef`, `actor.kl_loss_type` |
| KL in reward | off | `algorithm.use_kl_in_reward` |
| discount γ | 0.95 | `algorithm.gamma` |
| step advantage weight | 1.0 | `algorithm.gigpo.step_advantage_w` |
| advantage normalisation | `mean_std_norm` | `algorithm.gigpo.mode` |
| invalid-action penalty | 0.1 | `actor_rollout_ref.actor.invalid_action_penalty_coef` |
| **Sampling** | | |
| train | T=1.0, top-p 1.0, top-k −1 | `actor_rollout_ref.rollout.{temperature,top_p,top_k}` |
| eval | T=0.4, top-p 1.0, top-k −1, sampling on | `actor_rollout_ref.rollout.val_kwargs.*` |
| **Environment** | | |
| env | `alfworld/AlfredTWEnv` | `env.env_name` |
| max turns per episode | 50 | `env.max_steps` |
| history in prompt | 2 turns | `env.history_length` |
| seed | 0 | `env.seed` |
| eval split | `eval_in_distribution` (valid_seen) | `env.alfworld.eval_dataset` |
| **Schedule** | | |
| steps | 150 | `trainer.total_epochs` |
| evaluate / checkpoint every | 5 steps | `trainer.test_freq`, `trainer.save_freq` |
| held-out episodes | 128, in chunks of 64 | `data.val_batch_size` |
| early stopping | off | `ACG_EARLY_STOP_PATIENCE=0` |
| **Runtime** | | |
| vLLM memory fraction | 0.25 | `actor_rollout_ref.rollout.gpu_memory_utilization` |
| tensor parallel | 1 | `actor_rollout_ref.rollout.tensor_model_parallel_size` |
| attention | FA2 both sides | `VLLM_ATTENTION_BACKEND=FLASH_ATTN`, `use_remove_padding` |

### 7.2 WebShop

Everything above holds except these, which arrive with the published WebShop protocol:

| | ALFWorld | **WebShop** | hydra key |
|---|---|---|---|
| max prompt | 2048 | **4096** tokens — pages are long | `data.max_prompt_length` |
| max turns per episode | 50 | **15** | `env.max_steps` |
| PPO mini-batch | 256 | **64** | `actor.ppo_mini_batch_size` |
| micro-batch / GPU, update | 32 | **8** | `actor.ppo_micro_batch_size_per_gpu` |
| micro-batch / GPU, log-prob | 64 | **16** | `rollout.log_prob_micro_batch_size_per_gpu` |
| advantage normalisation | `mean_std_norm` | **`mean_norm`** | `algorithm.gigpo.mode` |
| held-out chunk size | 64 | **128** | `data.val_batch_size` |
| CPU per env worker | 0.1 | **0.05** | `env.resources_per_worker.num_cpus` |
| step target | return-to-go of the binary reward | **dense task score** (D) | `ACG_CCPO_TARGET=score` |
| catalogue | — | 1,000-product subset | `env.webshop.use_small=True` |
| venv | `.venv` | **`.venv-webshop`** | selected by `exp_run` |
| JVM | — | one per env worker, pinned | `JAVA_TOOL_OPTIONS`, set by `exp_run` |

`mean_norm` is not bookkeeping: it changes the update. With it, `_remove_std` is true, so
**neither** advantage term is standardised on WebShop and both stay in reward units,
where on ALFWorld both are unit-variance. That is `Z` in equation (T) of §1.

The micro-batch rows are inert while `use_dynamic_bsz` is on — the token budgets govern.
They are kept at the reference values so that turning dynamic batching off reproduces
the published script exactly.

## 8. Checkpoints, disk and GPUs

Each run keeps three checkpoints: `step<N>-best`, `step100-pin`, `step<N>-last`.

| backbone | checkpoint | GPUs | peak disk per run |
|---|---|---|---|
| Qwen2.5-1.5B-Instruct | **~25 GB** | **≥4** | ~100 GB |
| Qwen2.5-7B-Instruct | **~125 GB** | **≥8** | ~500 GB |

A checkpoint holds model, optimizer, extra state and an HF copy
(`actor_rollout_ref.actor.checkpoint.contents`), which is why it is far larger than the
weights alone.

Peak disk is four distinct step-copies: the rolling one (`keep_ckpts=1`), the in-flight
save, the best, and the pin. `step<N>-best` and `step100-pin` are made with `cp -al`, so
they cost nothing while the rolling copy at that step still exists and only become
distinct once it is pruned — the peak above is the worst case, not the steady state.

**If a 7B host cannot hold ~500 GB**, drop `--set pin_steps=` (keeping best and last,
~375 GB) and record that choice in the run's `NOTES.md`. Do not lower `keep_ckpts`: 1 is
already the minimum that can resume.

Thirteen ablation runs × ~100 GB run sequentially on one 4-GPU host; the ten main runs need a
host per backbone. `scripts/exp_status.py` reports what is running and what it has used.


### Context outlook variant (ALFWorld and WebShop)

`CCPO-ATTNCRED-CTXADV-OUTLOOK` and `CCPO-ATTNCRED-CTXADV-OUTLOOK-WS`
are the `attncred-context-outlook` main-method variant. They retain our historical
context-conditioned credit with weight 0.75 and add a two-step outlook estimate
with weight 0.25. Both use the binary return target, `ccpo_ep_w=0`,
`ccpo_edge_w=0`, `ccpo_outlook_horizon=2`, and `ccpo_outlook_beta=0.25`.
See [the estimator, terminal/penalty conventions, and ordering caveat](../ccpo/OUTLOOK.md).

Canonical experiment IDs: `ccpo-attncred-ctxadv-outlook-alfworld-1.5b`,
`ccpo-attncred-ctxadv-outlook-alfworld-7b`,
`ccpo-attncred-ctxadv-outlook-ws-1.5b`,
`ccpo-attncred-ctxadv-outlook-ws-7b`.

The requested initial runs use 1.5B with two GPUs per benchmark, the existing
M3/M5-NOEDGE batch/memory settings, seed 0 and 150 training steps.


### Fixed-anchor gain: history plus two observed future steps (2026-09-18)

`CCPO-ATTNCRED-CTXADV-FIXED-ANCHOR` and `CCPO-ATTNCRED-CTXADV-FIXED-ANCHOR-WS`
are registered under `attncred-context-fixed-anchor`. IDs are
`ccpo-attncred-ctxadv-fixed-anchor-alfworld-1.5b`,
`ccpo-attncred-ctxadv-fixed-anchor-alfworld-7b`,
`ccpo-attncred-ctxadv-fixed-anchor-ws-1.5b`, and
`ccpo-attncred-ctxadv-fixed-anchor-ws-7b`.

The authorized M8/M9 runs use 1.5B and two GPUs each. Their advantage is
`H + (B_joint - B_history)`, where `H = Y - B_history`. Both readouts retain
the exact current `(task, observation)` anchor and exclude the whole query
trajectory. The joint readout encodes the original historical prompt plus
the next two observed action/observation transitions, including terminal
observations, and both historical and future accumulated statistics.
No exact peer means zero added gain; the historical task fallback remains.
The original edge and episode advantage coefficients are zero.

Both use binary return targets, gamma .95, gain coefficient 1, kappa 2, and
the existing ALFWorld/WebShop outer normalization conventions. No optimizer
state or weights are resumed from the preempted OUTLOOK runs: these are fresh
seed-0 comparisons from Qwen2.5-1.5B-Instruct. Scalar metrics, canonical CSV,
raw/processed feature NPZ and exact joint-prompt tokens are saved each step.
See `ccpo/fixed_anchor.py`, `tests/test_fixed_anchor.py`, and the new run NOTES.

### Contextual future-state progress

`attncred-context-future-progress` replaces M3/M5's edge with the task-standardized difference of contextual values at the next and current observations. `attncred-context-future-progress-h2` is the separate two-step ablation. Both retain history coefficient 1, future coefficient 1, episode coefficient 0, return targets and each benchmark's original final normalization. No fixed-anchor or endpoint-bootstrap OUTLOOK term is mixed in.

| Variant | Method | Registry experiment names (1.5B / 7B) |
|---|---|---|
| CCPO-ATTNCRED-CTXADV-FUTURE-PROGRESS | attncred-context-future-progress | ccpo-attncred-ctxadv-future-progress-alfworld-1.5b / ccpo-attncred-ctxadv-future-progress-alfworld-7b |
| CCPO-ATTNCRED-CTXADV-FUTURE-PROGRESS-WS | attncred-context-future-progress | ccpo-attncred-ctxadv-future-progress-ws-1.5b / ccpo-attncred-ctxadv-future-progress-ws-7b |
| CCPO-ATTNCRED-CTXADV-FUTURE-PROGRESS-TWO-STEP | attncred-context-future-progress-h2 | ccpo-attncred-ctxadv-future-progress-h2-alfworld-1.5b / ccpo-attncred-ctxadv-future-progress-h2-alfworld-7b |
| CCPO-ATTNCRED-CTXADV-FUTURE-PROGRESS-TWO-STEP-WS | attncred-context-future-progress-h2 | ccpo-attncred-ctxadv-future-progress-h2-ws-1.5b / ccpo-attncred-ctxadv-future-progress-h2-ws-7b |

The prepared M10/M11 local configs clone the recorded two-GPU 1.5B M3/M5 protocol. Horizon 1 is primary; optional H2 configs change only the future horizon. **Primary M10/M11 launched at 04:14 UTC on 2026-09-18 by explicit user request, on GPUs 0–1 and 2–3. M8/M9 are paused with saved checkpoints pinned at steps 10/25. The separate H2 variants remain unlaunched.** Full math and terminal/grouping conventions are in [FUTURE_PROGRESS.md](../ccpo/FUTURE_PROGRESS.md).
