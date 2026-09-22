# M10 ALFWorld no shrinkage, EP0: history 3 / future 3

**Prepared only; not launched or queued.** Qwen2.5-1.5B-Instruct, 150 steps,
seed 0, two GPUs (`0,1` configured, not reserved), 50-turn cap.
Registry key `future-progress-noshrink-history3-future3`; canonical ID `ccpo-attncred-abl-noshrink-hist3-fut3-alfworld-1.5b`.

## Window and method

Prompt-history length: **3 previous observation/action pairs**.
Future-progress horizon: **3 steps**. These are window lengths, not numerical
advantage multipliers. History/future/episode actor weights are
**1 / 1 / 0**.

At history=0, both actor and frozen reference see the task goal, current
observation and action instructions, with no previous observation/action pairs;
the historical residual has zero actor weight. At future=0, the future horizon
and actor weight are both zero; current endpoints produce zero raw/normalized
future credit and no future support. Canonicalization, verified feature capture,
history/value diagnostics and component snapshots remain enabled in that mode.

The accumulated context-statistics block remains enabled for every arm, even at
history=0. Prompt windows do not truncate rollout memory or return labels.
Processed frozen hidden + context features, soft exponential kernel (tau 0.15),
three-direction whitening and whole-query-trajectory LOO are retained. Usable
readouts are full strength, lambda_u=lambda_k=1, with existing task fallback.

C_t[q] is the contextual readout over matching peers in the same task/observation
group, excluding the query trajectory. H_t=Y_t-C_t[Y],
Z_t=gamma^(T-t)*R_episode, V_t=C_t[Z], gamma=0.95.
For f>0, F_t=z_task(V_min(t+f,T)-V_t); for f=0, F_t=0.
Terminal potential is success 10 / failure 0. Each nonterminal future endpoint
uses its own observation group. Y retains the invalid-action penalty, Z does not.
The ALFWorld actor advantage is **A=mask*N_CC(H + F)**, where N_CC is the
existing per-task mean/sample-std normalization over enabled contextual support.

Episode rewards still supply return labels. The separate episode-advantage and
original-edge coefficients are 0. Raw H, episode and original-edge diagnostics
remain available when their actor contribution is disabled. No extra rollout
steps are generated to fill a window; future endpoints clip at termination.

## Control and reproduction

[Exact config diff](config-diff-from-control.json) compares with
`experiments/m10-h2-no-credit-shrinkage-alfworld-1.5b-2gpu-20260919`. The H2 main control has two prompt-history turns and two future
steps. Only the two window lengths, zero-side coefficients where needed, and
experiment identity change. Learning/evaluation settings stay matched: 16 tasks
x 8 rollouts, 150 steps, evaluation/checkpoints every 5, 2048/512 prompt/response
limits, 50 train/eval turns. Four-turn prompts keep the same token limits.

[Shared plan and equations](../ALFWORLD_WINDOW_ABLATIONS.md).
Prepare for a fresh date without training:

```bash
python scripts/prepare_alfworld_window_ablations.py --window h3f3 --date YYYYMMDD
```

Prepared launch command, for separate scheduling:

```bash
bash experiments/m10-hist3-fut3-noshrink-alfworld-1.5b-2gpu-20260922/run.sh
```

[config.json](config.json), [prepare-command.json](prepare-command.json),
[prepared-source-sha256.json](prepared-source-sha256.json), and
[VALIDATION.json](VALIDATION.json) record reproduction and CPU verification.
No GPU run or benchmark result is implied by preparation.
