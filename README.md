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
2. **Context.** Each entry gets a frozen feature vector φ built from the
   observation and the trajectory's history scalars (turn index, distinct states
   seen, progress, revisit flag), optionally including a compacted memory digest.
3. **Soft weights.** Siblings are weighted `w(u,v) = exp(−d(φ_u, φ_v) / τ_b)`
   with `τ_b` the bucket's median distance — near siblings count more.
4. **Leave-one-out baseline.** `b_LOO = Σ w·G / Σ w` over *other* trajectories.
   G²PO's node value includes the trajectory being scored; ours does not.
5. **Shrinkage.** `λ* = (ρB)² / ((ρB)² + s²(1/n_eff − 1/J))` interpolates between
   `b_LOO` (low bias, higher variance) and the uniform baseline `b_obs`
   (G²PO-style). With ρ=1 and B=0.30 this reduces to a function of how
   concentrated the affinity weights are.

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

## What has been measured

On ALFWorld with Qwen3-1.7B, 144 episodes/step (18 tasks × 8 rollouts),
evaluated on `valid_unseen`:

| question | measurement | reading |
|---|---|---|
| Is A_CC distinguishable from G²PO's credit? | `corr(A_ours, A_G²PO)` = **0.64** mean, range 0.45–0.94, same batch | materially distinct; never approaches the r≈1 indistinguishability line |
| Does anchor-state grouping conflate distinct states? | **42.8%** of observations map to >1 admissible-action set | yes — and G²PO/GiGPO inherit it, since their node key is the same raw text |
| Does that conflation cost anything? | within-bucket sibling pairs differ **1.56–1.69×** more in return when their hidden state differs (n ≈ 10M pairs) | yes; the bias prior B=0.30 is measured, not assumed |
| Does φ correct it? | AUC **0.483**; affinity weights 0.389 vs 0.391 | **no.** Inside a bucket the observation block is constant, leaving history scalars that cannot see inventory |
| Does A_CC reward verbosity? | `corr(A_CC, response length)` = **0.007** | no — length inflation comes from the shared objective |
| Why do long-horizon tasks fail? | **58.7%** of turns revisit an already-seen observation; failures revisit **2.68×** as often as successes | search inefficiency — what compaction targets |

The fourth row is a negative result and is stated as one: the method's premise
(anchor-state grouping false-merges, at a real cost) is confirmed, but the frozen
random-projection φ does not deliver the correction it was designed for. A φ that
encodes inventory is the open work.

**No GRPO baseline has been run yet**, so no claim of superiority over GRPO is
made anywhere in this repository.

---

## Method write-up

[`docs/method.html`](docs/method.html) is a standalone page (open it in a browser)
covering the estimator's derivation, worked examples of how the grouping differs
from GRPO and G²PO, per-mechanism figures for the context-conditioned credit
term, the full training-configuration comparison against published methods, and
the measurement log — including the negative results.

## Layout

```
ccpo/                     the estimator (core_ccpo.py) and a learned-φ variant
patches/verl-agent/       files that replace their verl-agent counterparts
scripts/                  training, eval, generation-sidecar and supervisor scripts
docs/                     method write-up (method.html)
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

## Attribution

`patches/verl-agent/verl/**` derives from [verl](https://github.com/volcengine/verl)
(Apache 2.0); `patches/verl-agent/agent_system/**` from
[verl-agent](https://github.com/langfengQ/verl-agent). Both are included here only
as modified copies so the changes are reviewable; upstream licenses and copyright
apply to those files.
