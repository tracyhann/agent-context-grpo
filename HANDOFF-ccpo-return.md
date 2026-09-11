# Handoff — `ccpo-return` (the H-K test arm)

Self-contained instructions for running `ccpo-return` in a **different container**.
Written 2026-09-10. Background reasoning is in `experiments/hypothesis.md` (H-K, H-AD,
H-AH); the full method note for the preceding arm is
`experiments/ccpo-anchor-2gpu-20260909/docs/method.html`.

## TL;DR

One flag differs from the 79.7% configuration: `ccpo_target=nextnode` → `return`.
The prediction is that this **restores the value of the grouping machinery**, which
`nextnode` measurably disables. Judge it on held-out success against a comparator run
**on the same hardware**, over the converged window — never on a single step.

---

## 0. BLOCKING — the code this needs is not on `main`

Get it from the **`ccpo-return`** branch (this file only exists there):

```bash
git clone https://github.com/tracyhann/agent-context-grpo
cd agent-context-grpo && git checkout ccpo-return
```

Two changes made on 2026-09-09/10 are required and were uncommitted when this was
written. **A fresh clone of `main` cannot run the launch command below**:

| file | change | without it |
|---|---|---|
| `scripts/exp_run.py` | new config key `vllm_attn_backend` (was hardcoded `TRITON_ATTN`) | launch dies: `unknown config key: vllm_attn_backend` |
| `ccpo/core_ccpo.py` | dump columns `g_cur`, `g_next` + `Counter` import | runs, but H-AH's offline analysis has no data |

Check before anything else:

```bash
grep -c '"vllm_attn_backend"' scripts/exp_run.py   # must be >= 1
grep -c 'g_next' ccpo/core_ccpo.py                  # must be >= 1
```

If either prints `0`, you are on `main` — run `git checkout ccpo-return`.

---

## 1. What this tests, and why it is expected to matter

**H-K, measured offline** on `experiments/gate-probe-20260907/outputs/grouping.jsonl`
(6,912 occurrences; the dump carries both targets on the same rows), with the
leave-one-*trajectory*-out rule the estimator actually uses:

| target | Var resid, task only | Var resid, + anchor gate | R² gain |
|---|---|---|---|
| return-to-go `G` (GiGPO's) | 7.0379 | 6.5922 | **+0.0633** |
| nextnode `V(next)` (**current**) | 1.9521 | 2.0184 | **−0.0339** |

Under `nextnode` the anchor gate makes the baseline *worse than ignoring the
observation*. Every member of a successor node shares the same target, so a
within-group comparison has nothing left to discriminate.

This explains two measured nulls: `ccpo-hardedge` (gate ON vs OFF, −0.01 over 9 paired
evals) and `ccpo-anchor-2gpu` (anchor repaired vs raw, −0.1 on the converged mean over
20 evals, −0.50 paired over 11). **Both changed a component `nextnode` had disabled.**

The design error it points at: we took GiGPO's use of the group as a *comparison set*
and G²PO's *successor target*, the one pairing neither paper validated. GiGPO pairs
state-grouping with return-to-go; G²PO pairs node-values with `V(next)` plus an edge
term. `ccpo-return` restores GiGPO's self-consistent pairing.

---

## 2. The method

```
A = A_EP  +  λ·A_CC  +  edge_w · std_per_task( V(next) − V(cur) )
                └─ TGT = G = Σ γ^(k−t) r_k      <- the only change
```

| component | `ccpo-global` (79.7%) | `ccpo-return` |
|---|---|---|
| episode term | group-relative outcome | same |
| **step target** | **`V(next(u))`** | **return-to-go `G`** |
| step baseline | φ-weighted leave-one-out over the task bucket | same |
| edge term `V(next)−V(cur)` | on, w=1.0 | **still on** — computed from `VAL` regardless of target |
| gate / λ / φ / σ | global, λ≡1, hidden+ctx, per-task | same |

**This is a new cell, not a revert.** `target=return` with `edge_w=1.0` has probably
never been run: the 70.3 → 79.7 jump is credited to turning the edge term on, which
happened alongside the switch to `nextnode`.

Verified to execute (2026-09-10, port-test fixture, `gate=global`): finite advantages,
`edge_cov=1.0`, advantages differ from `nextnode` on 23/23 rows, and the edge term still
moves 23/23 rows under `return`.

---

## 3. Environment, from bare

```bash
bash scripts/setup_env.sh          # venv, verl-agent + patches, vllm 0.11.0, flash-attn,
                                   # ALFWorld data, Qwen2.5-1.5B-Instruct. ~15 min.
```

Built clean on Python 3.10 / A100 with torch 2.8.0+cu128, vllm 0.11.0,
transformers 4.57.1, ray 2.50.0, flash-attn 2.8.3.post1. The flash-attn wheel is
cp310 — **a different Python version needs a different wheel.**

`baselines/*/` is gitignored and does not survive a move. The tests need it:

```bash
git clone --depth 1 https://github.com/Nala-YN/G2PO.git baselines/G2PO     # pin b8ffaf5
git -C baselines/G2PO fetch --depth 1 origin b8ffaf5 && git -C baselines/G2PO checkout b8ffaf5
```

Without it `tests/test_g2po_port.py` prints `test 1 SKIPPED` — silently, not a failure.

### Attention backend — pick by architecture

| GPU | `vllm_attn_backend` | why |
|---|---|---|
| A100 / sm_80 | **`FLASH_ATTN`** | measured: generation −12.2% / −9.8% vs Triton, held-out identical |
| RTX PRO 6000 Blackwell / sm_120 | `TRITON_ATTN` (the default) | why it was pinned originally |
| anything else | measure first | 2-step A/B, see `experiments/attn-*-20260909` |

**The comparator must use the same backend as this arm.** A kernel swap alone
decorrelates rollouts (measured: identical config, Triton vs Flash, all 13 step-1
metrics differed).

---

## 4. Pre-flight — five minutes, no GPU

```bash
for t in tests/test_*.py; do ./.venv/bin/python $t | tail -1; done
```

All seven should pass; `test_g2po_port.py` should read `test 1 PASS | test 2 PASS`.

Then confirm the arm differs from the comparator by exactly one flag:

```bash
./.venv/bin/python scripts/exp_run.py --name DRY --arm ccpo --dry-run <the ccpo-return flags>
diff <(grep -oE "ACG_[A-Z_]+=[^ ]*" experiments/DRY-*/run.sh | grep -vE "DUMP|EXP_DIR|METRICS" | sort) \
     <(grep -oE "ACG_[A-Z_]+=[^ ]*" experiments/ccpo-global-fa-20260909/run.sh | grep -vE "DUMP|EXP_DIR|METRICS" | sort)
rm -rf experiments/DRY-*
```

Expected output: a single line pair, `ACG_CCPO_TARGET=return` vs `=nextnode`.

---

## 5. Launch

### 5a. The arm

```bash
./.venv/bin/python scripts/exp_run.py --name ccpo-return --arm ccpo \
  --set gpus=0,1 --set vllm_attn_backend=FLASH_ATTN \
  --set total_epochs=100 --set compact_budget=0 --set ccpo_gate=global \
  --set ccpo_phi=hidden+ctx --set ccpo_edge_w=1.0 --set ccpo_rho=0.59 \
  --set ccpo_tau=0.15 --set ccpo_target=return --set ccpo_shrink=eb \
  --set ccpo_whiten=3 --set early_stop_min_steps=40 --set early_stop_patience=8
```

Adjust `gpus=` to whatever cards are free there. **`train_batch_size=16` must divide
the GPU count: 1, 2, 4 or 8 — not 3 or 6.**

### 5b. The comparator — needed unless the hardware is identical

`ccpo-global-fa-20260909` is the paired baseline, but **only if the new container has
the same GPU model, the same GPU count (2), and the same backend.** If any of those
differ, run the comparator there too — same command with `--set ccpo_target=nextnode`
and `--name ccpo-global-cmp`. Otherwise you are back to a cross-hardware comparison,
where the delta sd was 6.36 and a 3-point effect is undetectable.

Run them **one at a time** unless §6 says two fit.

---

## 6. Resource limits — check before launching

```bash
cat /sys/fs/cgroup/pids.max /sys/fs/cgroup/pids.current
nproc; cat /proc/loadavg; nvidia-smi
```

* **pids is the binding constraint, not GPUs.** One 2-GPU arm holds ~7,300 pids. The
  original container capped at 8,192, so **two arms cannot run concurrently there**.
  Exhaustion is silent: the run dies one line after `Started a local Ray instance`,
  no traceback. Two arms need ~15,000.
* **Memory:** ~130 GiB per arm including page cache; GPU reserve peaks near the full
  80 GB card during training, even when `nvidia-smi` reads ~28 GB between phases.
* **Shared hosts:** another tenant drove host load to 180 on 96 cores and doubled step
  time (480 → 1000+ s). Held-out results are unaffected; wall clock is not.
* **Kill by PID** from `outputs/train.pid`. **Never `pkill -f` or `pgrep -f`** — the
  pattern matches the shell running it. This has now caused three incidents,
  including one on 2026-09-10 that killed a chain script it was meant to protect.

---

## 7. Validity checks while it runs

**Step 1, if paired on identical hardware:** at step 1 no gradient has been applied, so
the target cannot affect rollouts. Compare against the comparator's step 1:

| metric | expected |
|---|---|
| `episode/success_rate` | **identical** — `ccpo-global-fa` read 0.140625 |
| `episode/length/mean` | **identical** — 47.0234375 |
| `ccpo/n_buckets` | **identical** — 760 (the target does not change the anchor) |
| token counts, `valid_action_ratio` | may differ in the 4th decimal — sampling wobble |

If `success_rate` differs at step 1, the two runs are not paired and every later delta
is cross-hardware. Stop and find out why before trusting anything.

From step 2 onward the arms diverge — that is the ablation working.

**Also at step 1:** `ccpo/edge_cov` should be `1.0`.

---

## 8. How to judge it

* **Held-out only.** Read `val/success_rate`, never `episode/success_rate` — the latter
  is the 16-task training draw, resampled every step at temperature 1.0.
* **Report the converged mean** (evaluations at step ≥ 70), with its n and sd. **Do not
  quote a `stepN-best`** — best-of-a-noisy-series is biased up by ~1.5 sd, which was
  ~10 points here. The anchor arm's 82.8% at step 100 was its maximum; its converged
  mean was 73.0%.
* **Paired deltas are still noisy.** Sharing hardware, seed and step-1 rollouts did
  *not* shrink the variance much (paired sd 5.8–6.5 vs 6.4 unpaired) — the policies
  diverge through their gradients. A difference under ~2 points is not resolvable.
* **Early points mislead.** The comparator ran 6 points under the original arm for 15
  steps, then caught up exactly. Early positive deltas have repeatedly decayed.
* **Kill rule:** stop at step 20 if held-out < ~11%.

### Pre-registered prediction

H-K predicts `ccpo-return` > `ccpo-global` on the converged mean. **If it does not**,
check the edge-term interaction first (§2) before concluding H-K is wrong — H-K is a
measurement about R², and stands regardless of how this arm lands. The informative
follow-up would be `target=return, edge_w=0.0`.

---

## 9. Reference numbers

Held-out success, 128 unseen tasks, `eval_in_distribution`:

```
step          5    10    15    20    25    30    35    40    45    50    55
global-fa   3.9   6.2  10.2  18.8  21.1  34.4  34.4  43.8  46.1  46.9  53.9
anchor      9.4  14.1  18.0  19.5  21.1  21.9  33.6  38.3  45.3  43.8  49.2
blackwell  10.2  10.9  17.2  21.1  21.1  30.5  35.9  22.7  41.4  50.0  35.2
ret-hard    3.9   8.6  18.0  20.3  22.7  32.8  31.2  37.5  36.7  49.2  50.0

step         60    65    70    75    80    85    90    95   100
global-fa  50.8  65.6  71.9  70.3  74.2  75.8  75.8  74.2  82.0
anchor     49.2  57.0  61.7  60.9  76.6  70.3  78.9  79.7  82.8
blackwell  41.4  47.7  63.3  68.0  71.9  75.8  75.0  78.1  79.7
ret-hard   58.6  62.5  69.5  64.1  80.5  76.6  78.1  81.2  84.4
```

* `global-fa` = `ccpo-global-fa-20260909`: the comparator. 2×A100, FLASH_ATTN.
  **Finished: converged mean 74.9% (sd 3.7), step 100 82.0%.** This is the number
  `ccpo-return` is judged against.
* `ret-hard` = `ccpo-return-hard-20260910`: **the same config as `ccpo-return` except
  hard gate + uniform weighting (`ccpo_gate=hard`, `ccpo_rho=0.0`) instead of global gate
  + phi.** Converged mean **76.3%**, step 100 84.4%. **`ccpo-return` vs this arm is the
  phi-isolating comparison** -- valid only on matched hardware (2xA100, FLASH_ATTN).
* **Same-config spread:** `global-fa` and `blackwell` run the identical config, yet differed
  by **-3.9 to +21.1 points** at matched steps >= 20 (+1.8 on the converged mean). Treat any
  single-run difference under ~5 points as unresolved.
* `anchor` = `ccpo-anchor-2gpu-20260909`: 2×A100, FLASH_ATTN. Converged mean 73.0%.
* `blackwell` = `ccpo-global-20260907`: the original 79.7% arm. 4× Blackwell, Triton.
  Converged mean 73.1%. Its dips at steps 40 and 55 did not reproduce in either later
  arm — treat them as bad draws.

Published, Qwen2.5-1.5B, ALFWorld: G²PO 95.0 (100 iters, 128 tasks), GiGPO 86.7 (150
iters), HGPO 92.77 (160 iters, **512** tasks). Every published row is 3 seeds; every
row here is 1.

---

## 10. What to bring back

* `experiments/ccpo-return-*/outputs/metrics.jsonl`, `config.json`, `run.sh`
* `outputs/ccpo_samples.csv` — carries `g_cur,g_next`, so H-AH's target-support
  analysis can be re-run on the `return` regime
* Record the result in `experiments/hypothesis.md` under H-K, and add a row to
  `HANDOFF.md`'s arms table.
