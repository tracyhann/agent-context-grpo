# CCPO — Context-Conditioned Policy Optimization

Credit assignment for multi-turn agent RL. CCPO replaces the *step* term of the
GRPO family (GiGPO, G²PO) with a context-conditioned, uncertainty-shrunk
leave-one-out baseline, leaving the episode term and everything else identical —
so a head-to-head against GRPO isolates the estimator and nothing else.

Built on [verl-agent](https://github.com/langfengQ/verl-agent) (itself built on
[verl](https://github.com/volcengine/verl)); benchmark is ALFWorld.

---

## The estimator

For a trajectory *i* at step *t*, the advantage is

```
A = A_EP + A_CC
```

`A_EP` is the standard group-relative episode advantage over the 8 siblings of a
task. `A_CC` is ours:

1. **Gate.** Entries are bucketed by `(task_uid, observation_text)` — the same
   anchor-state grouping G²PO and GiGPO use.
2. **Context.** Each entry gets a frozen feature vector φ. The default is the
   **reference policy's last-prompt-token hidden state**, whitened across the
   batch (corpus mean plus top-3 principal directions removed). The reference
   model is frozen by definition — which is what the estimator assumes — and
   already runs every step for the KL term, so φ costs no extra forward pass.
   `ACG_CCPO_PHI=bow` selects the original hashed bag-of-words features.
3. **Soft weights.** Siblings are weighted `w(u,v) = exp(−d(φ_u, φ_v) / τ_b)`
   with `τ_b` the bucket's median distance — near siblings count more.
4. **Leave-one-out baseline.** `b_LOO = Σ w·G / Σ w` over *other* trajectories.
   G²PO's node value includes the trajectory being scored; ours does not.
5. **Shrinkage.** `λ* = max(0, 1 − Var(d)/(ρ·d)²)` where `d = b_obs − b_LOO` is
   the realised disagreement between the conditioned and uniform baselines and
   `Var(d) = s²(1/n_eff − 1/J)`. It fires only where the disagreement exceeds
   its own sampling noise. ρ ∈ [0,1] is confidence in the metric, calibrated
   from the conflation AUC as `ρ = 2(AUC − 0.5)`; ρ→0 gives λ=0, an exact
   fallback to the uniform G²PO baseline.

   The earlier MSE form `λ* = (ρB)²/((ρB)² + s²(1/n_eff − 1/J))` is kept behind
   `ACG_CCPO_SHRINK=mse`. Writing the bias prior in sd units (`B = 0.30σ`) makes
   σ² cancel outright, so it reduces to a pure function of `n_eff` and `J` — the
   n_eff rule its own comment set out to replace. Measured over 305k samples it
   correlated **−0.239** with `|d|`: it shrank hardest exactly where the
   correction was largest. The replacement correlates **+0.212**.

Implementation: [`ccpo/core_ccpo.py`](ccpo/core_ccpo.py).

## Two supporting components

Both default to **off**, so setting `ACG_COMPACT_BUDGET=0 ACG_FORCE_BUDGET=0`
reproduces the stock G²PO protocol exactly and each becomes a one-flag ablation.

**Compaction** ([`compact.py`](patches/verl-agent/agent_system/memory/compact.py)) —
the stock prompt shows only the last 2 turns, so the agent cannot remember where
it has already searched. Compaction prepends a budgeted digest of the *entire*
history: action→outcome pairs, deduplicated, failures collapsed. It is
domain-agnostic: a failed action is detected as `obs_{t+1} == obs_t` (the world
did not change), with no per-benchmark regex.

**Budget forcing** ([`sidecar_client.py`](patches/verl-agent/verl/workers/rollout/sidecar_client.py)) —
responses that overrun the token budget lose their trailing `<action>` tag and
produce no usable action. Forcing caps the think block, then injects
`</think><action>` and generates a short action, so an unparseable response
becomes structurally impossible and length is capped by construction.

---

## A correctness defect, and what it invalidated

In the per-sample loop, `other` held **local** positions into the bucket. The
`b_LOO` aggregation mapped them back correctly via `G[idx[b]]`; two lines below,
`b_obs` and `s²` did not, and so were read off rows `0…len(idx)−1` of the whole
batch — arbitrary trajectories unrelated to the bucket. `b_obs` carries weight
`(1−λ)` on every sample and is the *entire* estimator wherever `λ=0`.

Measured over all 304,858 dumped samples: `b_obs` matched the correct
leave-one-trajectory-out mean on **56.5%** of rows, correlation **+0.051**, mean
absolute error 0.627. Every diagnostic taken before the fix was measured through
it:

| quantity | reported before | recomputed correctly |
|---|---|---|
| `corr(A_CC, A_G²PO)` | 0.64–0.68 | **0.956** |
| sign disagreement with G²PO | 23.7% | **0.3%** |
| mean \|effect\| | 0.229 | **0.027** |
| mean signed effect | +0.62 | **+0.0004** |
| mean `A_CC` (must be ~0) | +0.386 | **−0.026** |

The retraction that matters: **the "materially distinct estimator" claim was the
bug.** Correctly implemented, and with the bag-of-words φ, CCPO agrees with the
G²PO step credit on 99.7% of sign decisions — it is the baseline in disguise.
Fixed in `core_ccpo.py`; the mean-zero property a group-relative advantage must
have is restored.

## What has been measured

On ALFWorld with Qwen3-1.7B, 144 episodes/step (18 tasks × 8 rollouts):

| question | measurement | reading |
|---|---|---|
| Does anchor-state grouping conflate distinct states? | **42.8%** of observations map to >1 admissible-action set | yes — and G²PO/GiGPO inherit it, since their node key is the same raw text |
| Does that conflation cost anything? | within-bucket sibling pairs differ **1.56–1.69×** more in return when their hidden state differs (n ≈ 10M pairs) | yes; the bias prior B=0.30 is measured, not assumed |
| Does the bag-of-words φ correct it? | conflation AUC **0.483 / 0.492** (random-policy corpus), **0.568** (trained-policy); affinity weights 0.389 vs 0.391, `E[w]` pinned at 0.394–0.400 every step | **no.** Inside a bucket the observation block is constant, leaving history scalars that cannot see inventory |
| Does a policy hidden state correct it? | **0.795** — last-prompt-token, whitened (top-3 PCs removed), trained-policy corpus | **yes.** Raw 0.659; mean-pooled only 0.505, so the pooling choice is load-bearing |
| Is that leakage from action names? | context rebuilt from observations only, no action strings: **0.655–0.688** | no — both halves clear the bar independently and the full context beats either by ~0.09 |
| Does A_CC reward verbosity? | `corr(A_CC, response length)` = **0.007** | no — length inflation comes from the shared objective |
| Why do long-horizon tasks fail? | **58.7%** of turns revisit an already-seen observation; failures revisit **2.68×** as often as successes | search inefficiency — what compaction targets |

Two of these deserve emphasis. First, the corpus distribution mattered more than
the encoder: moving from an admissible-random walk to the trained policy lifted
even the *unchanged* bag-of-words φ from 0.492 to 0.568, so **every earlier
negative result measured on a random-policy corpus is suspect** and is being
re-scored. Second, a discriminating φ is necessary but not sufficient — the step
term still acts only where a bucket holds ≥2 trajectories with disagreeing
returns, and ~44% of buckets are singletons.

**No valid GRPO baseline exists yet.** An attempted one was contaminated: a
supervisor relaunch passed the tag but not the estimator, so `adv_estimator=ccpo`
ran from step 8 to 38 under the GRPO run's name. Only steps 1–7 were GRPO. A
guard now refuses any launch whose tag and estimator disagree. No claim of
superiority over GRPO is made anywhere in this repository.

Evaluation is stochastic by design (the reference protocol samples at T=0.4), and
two evaluations of the *same* checkpoint returned 0.094 and 0.188 — a single
128-episode score carries roughly **±0.048**. Report the mean of ≥3 seeds; do not
read single points.

---

## Method write-up

[`docs/method-changes.html`](docs/method-changes.html) records the current
revision: the indexing defect and what it invalidated, the change ledger, the
estimator compared column-by-column against GRPO and G²PO/GiGPO, and the
training-configuration diff against the G²PO reference script.

[`docs/method.html`](docs/method.html) is the original standalone page (open it in a browser)
covering the estimator's derivation, worked examples of how the grouping differs
from GRPO and G²PO, per-mechanism figures for the context-conditioned credit
term, the full training-configuration comparison against published methods, and
the measurement log — including the negative results.

## Layout

```
ccpo/                     the estimator (core_ccpo.py) and a learned-φ variant
patches/verl-agent/       files that replace their verl-agent counterparts
scripts/                  training, eval, generation-sidecar and supervisor scripts
tests/                    guards on shipped behaviour (see below)
docs/                     method write-up + the revision record
docker/                   container build + flash-attn import stub (sm_120)
```

`patches/` mirrors verl-agent's tree: copy each file over the corresponding path
in a verl-agent checkout. The modified files carry inline comments explaining
each change and why it was needed.

## Running

```bash
# CCPO
scripts/run_verl_ccpo.sh 0,1,2,3,4,5 ccpo verl_ccpo_alfworld

# GRPO baseline — identical config, one flag
scripts/run_verl_ccpo.sh 0,1,2,3,4,5 grpo verl_grpo_alfworld

# from INSIDE the trainer container (no docker CLI needed)
scripts/run_local_ccpo.sh ccpo
scripts/sidecar_status.sh

# evaluate a checkpoint's HF export on either split
scripts/run_eval_split.sh <hf_dir> eval_out_of_distribution my_tag
```

Key environment variables (all with defaults in the launcher):

| variable | default | effect |
|---|---|---|
| `ACG_COMPACT_BUDGET` | 512 | compaction digest token budget; 0 disables |
| `ACG_FORCE_BUDGET` | 512 | think-token cap before an action is forced; 0 disables |
| `ACG_FORCE_TAIL` | 32 | token budget for the forced action |
| `ACG_KL_COEF` | 0.01 | KL loss coefficient |
| `ACG_EPOCHS` | 75 | training steps |
| `ACG_CCPO_DUMP` | — | path for per-sample diagnostics CSV |
| `ACG_CCPO_PHI` | `hidden` | affinity metric: reference-policy hidden state, or `bow` |
| `ACG_CCPO_WHITEN` | 3 | principal directions removed before distances |
| `ACG_CCPO_RHO` | 0.59 | confidence in the metric, `2(AUC−0.5)`; 0 ⇒ exact G²PO fallback |
| `ACG_CCPO_SHRINK` | `eb` | shrinkage rule; `mse` reproduces the superseded form |
| `ACG_EVAL_SPLIT` | `eval_in_distribution` | `valid_seen` (reference default) or `eval_out_of_distribution` |
| `ACG_VAL_TEMP` | 0.4 | validation temperature, matching the reference script |

## Notes on the hardware path

Developed on RTX PRO 6000 Blackwell (sm_120), where flash-attn has no build and
older vLLM produced incoherent generations. Consequences visible in this repo:

- `docker/fa_stub/` — an import-only flash-attn stub; attention runs on sdpa,
  and `use_remove_padding` is therefore off.
- Generation runs in **separate vLLM 0.28 containers** with
  `VLLM_ATTENTION_BACKEND=TRITON_ATTN`, talking HTTP to the trainer
  (`scripts/sidecar_launch.sh`, `scripts/sidecar_sync.sh`). This isolates a hard
  version conflict — vLLM 0.28 needs torch 2.13 / transformers 5.x while the
  trainer runs torch 2.7 / transformers 4.51 — and measured **6.3× faster
  generation** than in-process HF generate. The client falls back to HF
  automatically on any sidecar failure, so the fast path cannot corrupt a batch.
- `hf_rollout.py` removes stale `position_ids` from `generate()`; under
  transformers 4.51 caching these caused rotary drift and degenerate output
  (parser-valid responses went from ~13% to 100% when removed).

## Tests

Both run on CPU except where noted; they guard behaviour that is easy to break
silently.

```bash
python tests/test_phi_hidden.py     # needs 1 GPU for the second half
python tests/test_digest_trim.py
```

`test_phi_hidden.py` checks that whitening recovers a state direction hidden
beneath higher-variance nuisance directions, that a discriminating φ removes the
state component from the credit, that **ρ=0 collapses λ to exactly 0** (the
fallback guarantee — an earlier revision of the shrinkage rule dropped ρ and
silently lost it), and that the affinity vector is pooled at the last *prompt*
token, `seqlen − response_length − 1`, verified against `hidden_states[-1]`. An
off-by-one there would misattribute every sample's features to a different
sample without any visible error.

`test_digest_trim.py` checks the prompt-overflow path: that the digest
delimiters hold and that trimming never touches the task description, the
current observation or the admissible-action list.

## Operational note

Scripts that run on the **host** (`sidecar_sync.sh`, `sidecar_launch.sh`,
`run_exec_ccpo.sh`, `auto_resume.sh`, `container_launch.sh`) invoke the docker
CLI. The trainer container deliberately has **no docker socket mounted** —
`/var/run/docker.sock` in a container is equivalent to host root. For the same
reason, host-executed scripts should not live in a directory the container can
write; keep them outside the bind-mounted project tree. `run_local_ccpo.sh` and
`sidecar_status.sh` are the in-container counterparts and need no docker.

## Attribution

`patches/verl-agent/verl/**` derives from [verl](https://github.com/volcengine/verl)
(Apache 2.0); `patches/verl-agent/agent_system/**` from
[verl-agent](https://github.com/langfengQ/verl-agent). Both are included here only
as modified copies so the changes are reviewable; upstream licenses and copyright
apply to those files.
