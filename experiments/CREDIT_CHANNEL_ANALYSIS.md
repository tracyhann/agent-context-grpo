# Offline credit-channel analysis

**Tooling only. It has not been run on a real training log.**

`scripts/analyse_credit_channels.py` reads the per-turn logs every future-progress
run already writes, `outputs/future_progress/step-NNNN.csv`, and needs no training,
model forward or GPU. It addresses two reviewer questions about the
[main method](MAIN_METHOD.md):

- **A.** Historical credit already uses the discounted return. What distinct
  information does future credit provide?
- **B.** How does `V(t+K) - V(t)` compare with simpler signals derived from the same
  outcomes: the episode-level reward, one-step value differences, a TD residual?

Every number is a statistic of credit signals on the rollouts of the logged policy.
It says how different two signals are. It does not say which one trains a better
policy; that takes a training run, such as
[the discount-aligned arm](DISCOUNT_ALIGNED_FUTURE_RESIDUAL.md).

## Run

On the host that holds the run's `outputs/`, from the repository root:

```bash
.venv/bin/python tests/test_credit_channel_analysis.py
```

```bash
.venv/bin/python scripts/analyse_credit_channels.py experiments/m10-h2-no-credit-shrinkage-alfworld-1.5b-2gpu-20260919
```

Discount, future horizon and future-credit mode come from the run's `config.json`.
`--steps 1-150` restricts the logs, `--phases` sets the step ranges that are averaged
(default `1-30,31-120,121-150`, plus the whole range), and a bare directory of logs
works with `--gamma` and `--horizon`. Output goes to
`<run>/outputs/credit_channels/`: `credit_channels.md` (the tables),
`credit_channels.json` (everything, per step and per range) and
`credit_channels_per_step.csv`.

The September main runs log every column the analysis reads.

Before any statistic, each step's logged future credit is recomputed from the logged
contextual values with the training code's own function. If the eligible rows, the
window lengths or the raw credit do not match, the analysis stops.

## A. Historical credit versus future credit

Two versions of each channel are compared: `applied`, its contribution to the actor
advantage after the trainer's task-wise normalization, and `raw`,
`H = Y - C[Y]` and `F = V(t+K) - V(t)`, where zero means "as good as the peers" and
"the contextual value did not change". Rows are turns that have both channels, in
tasks whose future credit is not flat (a task in which every rollout failed has none;
the count of such rows is reported). Each statistic is also given for non-terminal
windows alone, where neither endpoint value uses the turn's own outcome.

| Statistic | Meaning |
|---|---|
| Pearson, Spearman | correlation of H and F across turns |
| sign disagreement | share of turns where H and F have opposite signs |
| positive in failed | share of turns of failed trajectories with positive credit, in tasks with both outcomes |
| negative in successful | share of turns of successful trajectories with negative credit, same tasks |
| trajectory share | fraction of a channel's within-task variance explained by which trajectory the turn is in |
| outcome share | the same, explained by whether that trajectory succeeded |
| chance | the share expected from labels that carry no information, (groups - 1) / (rows - 1) |

Shares are averaged over tasks and do not change under task-wise centering and
scaling, so they are reported once.

The claim under test is that H mostly tells trajectories apart by their outcome,
while F tells turns apart inside a trajectory:

- Supported if H's trajectory and outcome shares are high and F's are close to
  chance, and a sizeable share of turns in failed trajectories has positive F.
- Not supported if the two channels correlate strongly and F's shares are as high as
  H's. Complementarity then rests on the trained history-only and future-only
  ablations alone.

On ALFWorld a failed trajectory has return zero, so its raw H cannot be positive.
That entry is zero by construction; the entry for F is the measurement.

## B. Future credit versus simpler signals

Each signal is recomputed from the logged values on the same rollouts and
standardized within task exactly as future credit is, then compared with the run's
own future credit by Pearson, Spearman and sign agreement.

| Signal | Definition |
|---|---|
| `one_step_difference` | `V(t+1) - V(t)` |
| `one_step_td_residual` | `gamma V(t+1) - V(t)` |
| `k_step_td_residual` | `gamma^w V(t+w) - V(t)` over the run's window; the discount-aligned arm's signal |
| `discounted_final_reward` | `gamma^(T-t) R`, with R the success potential or zero |
| `episode_outcome` | the trajectory's return, the same for all of its turns |

Terminal endpoints use the success potential, as in training. A correlation near one
means the two signals rank turns the same way, and a trained comparison between them
is then unlikely to separate them. It does not show that either is the better
training signal.

## Verification

[Tests](../tests/test_credit_channel_analysis.py), CPU only: the statistics against
hand-computed values and an independent one-way ANOVA; the alternative signals
against hand-computed values; rejection of logs whose credit does not follow from
their values, of wrong settings and of incomplete trajectories; and logs written by
the real estimator and log writer, read back and compared with the arrays that were
logged, on both the `difference` and the `discount_aligned` modes. No real training
log was available on the machine the tool was written on, and no result is claimed.
