# Context-conditioned future outlook

`attncred-context-outlook` is a main-method variant for ALFWorld and WebShop.
It mixes 75% historical Monte Carlo advantage with 25% two-step outlook credit.
Both benchmarks use the binary environment return; edge and episode advantages
are disabled. The policy remains Qwen2.5-1.5B-Instruct for the requested runs.

For history H at turn t, V(H_t) is our existing cross-trajectory kernel readout,
using frozen reference-policy features plus accumulated trajectory statistics.
The same estimator at t+2 sees the intervening two turns in its history and
statistics accumulated only through t+2. It uses the same credibility blend
J/(J+2); empirical-Bayes mixing remains pinned at 1.

```
A_history(t) = G_t - V_penalised(H_t)
A_outlook(t) = r_t + gamma*r_(t+1) + gamma^2*V_env(H_(t+2))
               - current_invalid_action_penalty - V_penalised(H_t)
A(t) = 0.75*A_history(t) + 0.25*A_outlook(t)
```

Here r is the unpenalised immediate environment reward. Existing runs subtract
an invalid-action penalty only at the current turn, rather than accumulating all
future penalties into G. To preserve that convention, the future bootstrap
predicts unpenalised environment return with the SAME neighbour weights and
credibility prior. The current penalty is retained once in both targets.
At termination (including the rollout horizon), remaining value is zero and
rewards are included only through the end of the episode. An unsupported future
endpoint falls back to the historical advantage. No new environment rollouts or
reference-model forward passes are required.

ALFWorld applies its existing per-task normalization after mixing. WebShop keeps
the mixture in reward units. No separately standardized edge term is added.

## Temporal order and padding

Rollout rows now carry `ccpo_turn_index` before batch adjustment. The outlook
estimator restores chronological order, removes padding copies from reference
pools, and maps credits back to every training row. It rejects incomplete
trajectories rather than treating an adjacent balanced row as the next turn.

This also corrects history/statistics reconstruction for the outlook variant:
the older estimator traverses the post-balancing row order and counts padding
copies. Existing methods retain their previous behavior. Consequently, a clean
follow-up ablation of outlook itself should set `ccpo_outlook_horizon=2` and
`ccpo_outlook_beta=0` to retain the same ordering/padding correction. Comparisons
against the already-running NOEDGE jobs include that implementation difference.

## Usage

```
python3 ccpo/run.py --method attncred-context-outlook --benchmark alfworld --backbone 1.5b --gpus 0,1
python3 ccpo/run.py --method attncred-context-outlook --benchmark webshop --backbone 1.5b --gpus 2,3
```

The local launch manifests retain the proven two-GPU memory and batch settings
from M3/M5-NOEDGE. Runs start from the base model, use seed 0, train for 150 steps,
and evaluate/save every 5 steps. This is a bootstrapped estimator and may add bias
when its value estimates are inaccurate; improvement is an experimental question.

Diagnostics include `ccpo/outlook_delta_absmean`, component magnitudes, component
correlation, endpoint fallback and terminal fractions, and unique/padded row
counts. `ccpo/effect_rel` and estimator correlations use the final mixed credit.
The existing `ccpo_samples.csv` retains the historical pre-mix credit; scalar
outlook diagnostics in `metrics.jsonl` describe the mixture.

References: [GAE](https://arxiv.org/abs/1506.02438),
[future-dependent baseline caveats](https://proceedings.mlr.press/v139/nota21a.html).
