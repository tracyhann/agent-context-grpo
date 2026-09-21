# M10 ALFWorld: history 1, future 1, no shrinkage, no episode advantage

**Prepared only; not launched or queued.** Fresh Qwen2.5-1.5B-Instruct,
150 training steps, two GPUs, seed 0, 8 rollouts/task. GPU IDs `0,1`
are inherited configuration placeholders, not a reservation.

Registry key: `future-progress-history1-future1-no-credit-shrinkage`.
Canonical ID: `ccpo-attncred-abl-hist1-fut1-noshrink-alfworld-1.5b`.

| Setting | Value |
|---|---|
| Previous prompt-history turns | 1 |
| Future-progress horizon | 1 |
| Usable history/current/future baseline lambda_k | 1 |
| Episode / original-edge coefficients | 0 / 0 |
| History / future coefficients | 1 / 1 |
| Representation | processed frozen hidden state + accumulated context statistics |
| Similarity weights | standard soft exponential kernel |
| Train/eval maximum actions per episode | 50 / 50 |

Compared with the [H2 no-shrink control](../m10-h2-no-credit-shrinkage-alfworld-1.5b-2gpu-20260919/NOTES.md), only
`history_length: 2 -> 1`, `ccpo_progress_horizon: 2 -> 1` and experiment ID change.
Context statistics remain enabled (`ccpo_ctx_w=1`); the full representation and
standard kernel are retained. No cosine or 30-turn intervention is combined here.

History 1 supplies the previous observation/action pair plus the current
observation to the policy and frozen reference. It does not truncate accumulated
context statistics or the discounted-return target. Future 1 uses the next
endpoint, clipped at actual termination. For usable contextual baselines, B=C
at lambda_k=1, including one-peer groups. Existing task-bucket fallback remains.

H = Y - C[Y]; V = C[unpenalized discounted outcome];
F = z_task(V_min(t+1,T) - V_t). Terminal potential remains 10/0.
The actor advantage is z_task(H+F), with the response mask applied.
Episode advantage is still logged, but its applied contribution is exactly zero.
Original-edge weight also stays zero. See the
[full method and comparison](../experiments.md#m10m11-history-1-future-1-no-shrinkage-no-episode-advantage-2026-09-20).

Existing history/current/future baselines, support, effective support, lambda,
features, future progress, raw episode and applied-component diagnostics remain
available. No estimator or trainer runtime change was needed.

## Preparation and validation

The launcher generated `config.json` and local gitignored `run.sh` in dry-run
mode. Once separately scheduled, the prepared command is:

```bash
bash experiments/m10-history1-future1-noshrink-alfworld-1.5b-2gpu-20260920/run.sh
```

See [preparation](PREPARED.json), [paired delta](config-diff-from-h2-noshrink.json),
[validation](VALIDATION.json) and [source hashes](prepared-source-sha256.json).
No queue entry, GPU reservation or GPU preflight was performed.

CPU validation passed: 39 future-progress tests; registry-delta guards; H1
endpoint and full-strength baseline checks; the actual trainer's episode-zero
gradient-isolation test under no shrinkage; actual ALFWorld/WebShop prompt checks;
config/Hydra/environment checks; shell syntax; and six scoped arm/documentation
guards. GPU execution remains untested. See VALIDATION.json for details.
