# M11 WebShop: history 1, future 1

**Prepared only; not launched or queued.** Fresh Qwen2.5-1.5B-Instruct, two GPUs,
150 steps, seed 0, 8 rollouts per task. Recorded GPU IDs `2,3` are inherited
configuration placeholders, not a reservation.

Registry key: `future-progress-history1-future1`; WebShop only.
Variant: `CCPO-ATTNCRED-FUTURE-PROGRESS-HISTORYONE-FUTUREONE-WS`.
Canonical ID: `ccpo-attncred-abl-hist1-fut1-ws-1.5b`.
Control: [original M11 H1](../m11-ccpo-attncred-ctxadv-future-progress-webshop-1.5b-2gpu-20260918/NOTES.md).

| Setting | Original M11 | This arm |
|---|---:|---:|
| Prompt history length | 2 | **1** |
| Future horizon | 1 | **1** |
| κ | 2 | **2** |
| Context-stat vector weight | 1 | 1 |
| History / future weights | 1 / 1 | 1 / 1 |
| Episode / original-edge weights | 0 / 0 | 0 / 0 |

The only method delta is `history_length: 2 → 1`. κ remains the M11 default 2;
this arm is independent of the H2/κ=4 ablation.

## Meaning of the ±1 window

The prompt at t contains task/instructions, step count, current observation O_t,
and at most the immediately preceding (O_(t−1), a_(t−1)) pair. At the first turn
there is no historical pair. This same prompt is used by the actor and frozen
reference feature extraction. WebShop's existing overlong-prompt fallback is
unchanged and can omit history entirely.

Future progress uses the next endpoint V_(t+1). Its frozen representation is
computed from that endpoint's own causal prompt (one past pair plus its current
observation); future text is never added to the acting policy's prompt at t.

The shorthand ±1 describes the local prompt history and future endpoint.
**Accumulated context statistics remain enabled and still summarize the whole
prefix; step count and return-to-go targets also remain. This is not a strict
local-only estimator or an estimator-only context intervention.** It changes
both the policy's historical input and the frozen representation. No additional
one-step return truncation, statistics truncation or actor/reference split is
introduced. This is the working interpretation of the requested window.

```
h_s^(1) = frozen_hidden(prompt with one preceding observation-action pair)
φ_s = normalize(concat(whiten_and_normalize(h_s^(1)), normalize(ctx_stats_s)))
B_s[q] = λ_s C_s[q; φ] + (1 − λ_s) U_s[q],  λ_s = J_s / (J_s + 2)
H_t = Y_t − B_t[Y]
Z_s = γ^(T−s) R_episode,       V_s = B_s[Z]
F_t = z_task(V_min(t+1,T) − V_t)
A_t = H_t + F_t              (WebShop mean_norm)
```

Grouping, whole-trajectory exclusion, task fallback, terminal 10/0 potentials,
1,000-product catalogue, original scorer and binary-return credit targets are
inherited unchanged. Existing history/current/future diagnostics remain enabled.

Files: [config](config.json), [exact delta](config-diff-from-control.json),
[status](PREPARED.json), [CPU validation](VALIDATION.json),
[source hashes](prepared-source-sha256.json). Local `run.sh` is prepared,
unexecuted and gitignored. [CPU tests](../../tests/test_future_progress_webshop_ablation.py)
exercise the real WebShop prompt builder and memory fetch without starting a JVM
or environment, including removal of older history and retention of current
observation, task, actions and the full stored memory.

For a controlled training comparison, use the same source revision and verified
reference-feature capture for this arm and history-length-2 M11. CPU checks do
not establish GPU smoke-test success.

Validation: **33 future-progress CPU tests passed** in the WebShop virtual environment, including 3 new checks. The ablation registry guard, arm/document registry guard and generated shell syntax checks passed. No GPU was used.
