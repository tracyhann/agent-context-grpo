# CCPO: Context-Conditioned Policy Optimization

CCPO assigns credit to each agent turn using a context-conditioned historical
return residual and a future potential increase. This package contains the
**no-shrinkage main method**, with episode advantage either disabled or added
with unit weight. It is a standalone advantage estimator for PPO, independent
of the surrounding research repository.

## Install and run

Python 3.10+ and NumPy are sufficient for the core. Run from this directory:

```bash
pip install -e .
python examples/minimal.py
python -m unittest discover -s tests -v
```

For the PyTorch/verl adapter and its tests, install `pip install -e '.[torch]'`.
The example uses synthetic trajectories and downloads nothing.

## Method

For a task q and trajectory i, let t=0,...,T_i-1 index action turns. The current
prompt contains the task, current observation and two preceding observation/action
pairs. The default future horizon is h=2.

### Context representation

Take the final-layer hidden state at the **last prompt token** from a frozen
reference policy. Center these states across the batch, remove the top three
principal directions, and L2-normalize. Append an independently L2-normalized
37-dimensional summary of the observed trajectory prefix:

| Statistic | Meaning | Encoding |
|---|---|---|
| t | Current zero-based turn | 12 thermometer coordinates, range 0–30 |
| u_t | Number of distinct observations through t | 12 coordinates, range 0–25 |
| u_t/(t+1) | Observation novelty ratio | 12 coordinates, range 0–1 |
| revisit_t | Current observation has appeared before | One binary coordinate |

Observation equality is exact string equality. The novelty ratio is not task
completion. Thermometer encoding sets the first floor(12 clip(x/range,0,1))
coordinates to one. Counts saturate at their range limits. Normalize the final
concatenation to obtain z_t; both blocks have coefficient 1. The reference is
frozen, while the centering/PCA basis is recomputed per batch without reward labels.

### Contextual baselines

For each query (i,t), use matching observations from other trajectories of the
same task:

\[
\mathcal P_{i,t}=\{(j,u):q_j=q_i,\ O_{j,u}=O_{i,t},\ j\ne i\}.
\]

Use the exponential distance kernel

\[
w_{ab}=\exp(-\|z_a-z_b\|_2/\tau_g),\qquad
\tau_g=0.15\,\operatorname{median}_{a<b\in g}\|z_a-z_b\|_2,
\qquad C_a[Q]=\frac{\sum_{b\in\mathcal P_a}w_{ab}Q_b}{\sum_{b\in\mathcal P_a}w_{ab}}.
\]

Bandwidth uses the full observation bucket, before trajectory exclusion. A
near-zero median is replaced by 1. If no other trajectory matches the observation,
back off to all turns of the same task outside the query trajectory and recompute
the kernel within that task bucket. If no such peers exist, contextual credit is
unsupported and zero. Repeated visits retain their occurrence weights; support
is counted by distinct trajectories. Every usable baseline is applied at full
strength: **no shrinkage toward a uniform baseline or task prior**.

### History and future credit

The environment has terminal reward R_i in {0,10}. With gamma=0.95 and local
invalid-action penalty p_it=0.1 1[invalid_it], the two target channels are

\[
Y_{i,t}=\gamma^{T_i-1-t}R_i-p_{i,t},\qquad
Z_{i,t}=\gamma^{T_i-t}R_i.
\]

The API accepts Y as `returns`, the unpenalized R as `episode_rewards`, and
R-p as `episode_scores`; the penalty is not reapplied by the estimator.
Y discounts environment rewards, then subtracts only the current turn's penalty.
Potential targets Z exclude invalid-action penalties and intentionally have one
additional discount, matching the original edge convention.

\[
H_{i,t}=Y_{i,t}-C_{i,t}[Y],\quad V_{i,t}=C_{i,t}[Z],\qquad
F_{i,t}=\operatorname{z}_{q_i}\left(V_{i,\min(t+h,T_i)}-V_{i,t}\right).
\]

Each endpoint uses its own observation group and its own ordinary prompt/history
representation. At a terminal endpoint V_i,T_i=R_i, including 0 for unsuccessful
horizon exhaustion. Future differences require supported current and endpoint
values; unsupported differences are zero and excluded from their task moments.
The task z-score uses sample standard deviation plus 1e-6; fewer than two eligible
rows give zero future credit. There is no extra gamma^h multiplier, immediate
reward term, or original edge bonus. This is a progress signal, not a TD residual
or a claim of unbiased causal attribution to a single action.

### Episode modes and normalization

\[
A_{\mathrm{CC}}=\mathcal N_q(H+F),\qquad
A=A_{\mathrm{CC}}+\eta A_{\mathrm{EP}},\qquad \eta\in\{0,1\}.
\]

| Convention | ALFWorld | WebShop |
|---|---|---|
| Future term F | Task z-score | Task z-score |
| Final contextual transform N | Task z-score over supported turn rows | Identity |
| Episode advantage | Task z-score of episode_scores | Task mean-centering of episode_scores |
| Episode fusion | Add after N; no further normalization | Add after N; no further normalization |

Episode statistics use **turn rows**, preserving the existing length weighting;
they do not count each trajectory just once. A task with one episode-score row
uses mean=0 and std=1. The combined contextual transform leaves a single supported
row unchanged. Episode rewards still define Y and Z when eta=0; only the separate
episode-advantage channel is disabled.

## API and PPO integration

```python
from ccpo import CCPOConfig, compute_advantages

config = CCPOConfig(benchmark="alfworld", episode_weight=0)  # or 1
result = compute_advantages(batch, config)  # RolloutBatch: see examples/minimal.py
```

`RolloutBatch` has one row per observed action turn: task/trajectory IDs,
observation strings, integer turn indices and episode lengths, frozen hidden
states, and the three reward channels above. All trajectories must be complete,
IDs must be globally unique per trajectory, and features must align with those
rows. Hidden width is backbone-dependent; the output width is hidden width + 37.
Input order may be arbitrary; output order is preserved.

Compute advantages on the **globally gathered rollout batch**, before scattering
PPO minibatches. Padding copies are removed before feature processing, peer
selection and future-term normalization. Duplicate rows must agree exactly,
including their hidden states: extract once per source turn, then copy, rather
than independently re-encoding training padding. Final contextual and episode
moments follow restored trainer rows, preserving existing padded-batch semantics.
Prefer computing before padding when the trainer permits it.

For a verl-agent `DataProto` containing aligned `ccpo_phi_feats` and explicit
`ccpo_turn_index` metadata:

```python
from ccpo import CCPOConfig
from ccpo.torch import compute_verl_advantages

advantages, components = compute_verl_advantages(
    data, CCPOConfig(benchmark="webshop", episode_weight=1)
)
data.batch["advantages"] = advantages
data.batch["returns"] = advantages  # outcome-advantage PPO convention; no critic
```

The adapter consumes the existing `step_rewards`, `token_level_rewards`,
`response_mask`, `uid`, `traj_uid`, `anchor_obs`, `episode_rewards`,
`episode_lengths` and `ccpo_turn_index` fields. It does not apply another penalty,
normalize the fused sum, or change the trainer configuration. Supply the frozen
reference's final hidden states via `last_prompt_hidden`; both padded and packed
layouts are supported. When distributed workers reorder rows, carry source IDs
with the features and verify the correspondence before calling the estimator.

One detached scalar is broadcast to **all generated response tokens**, including
`<think>` and `<action>` content. Prompt and padding positions have no direct
policy loss. The existing clipped PPO objective and separate KL regularization
remain in the trainer. Frozen feature extraction is in evaluation/no-grad mode;
the actor itself remains trainable.

## Benchmark training defaults

| Setting | ALFWorld | WebShop |
|---|---|---|
| Backbone | Qwen2.5-1.5B / 7B-Instruct | Same |
| Training iterations | 150 | 150 |
| Tasks x trajectories per iteration | 16 x 8 | 16 x 8 |
| Prompt history / future horizon | 2 / 2 | 2 / 2 |
| Maximum turns | 50 | 15 |
| Prompt / response token limits | 2048 / 512 | 4096 / 512 |
| Sampling temperature / learning rate | 1.0 / 1e-6 | Same |
| Discount / KL loss coefficient | 0.95 / 0.01 | Same |

These are rollout/trainer settings, not hidden side effects of the estimator.
This release contains source, tests and a synthetic usage example only. It does
not bundle training infrastructure, model weights, datasets or experiment outputs.
