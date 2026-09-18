# M7 context outlook — webshop

Qwen2.5-1.5B-Instruct, GPUs 2,3, seed 0, 150 steps.
Fresh base-model initialization; no checkpoint resume.

Two-step outlook mixed at beta=0.25 with our historical context advantage.
Edge and episode weights are zero. Binary environment-return target, gamma=0.95,
credibility prior kappa=2, lambda fixed at 1; all batch and memory settings match
[the NOEDGE control](../m5-noedge-ccpo-attncred-ctxadv-ret-ws-1.5b-2gpu-20260916/config.json).

See [method details](../../ccpo/OUTLOOK.md), especially the temporal-order/padding
correction and the need for a beta=0 ordered control to isolate outlook itself.
The two added config knobs and experiment ID are the only configuration changes.

The user prioritized immediate launch on 2026-09-17: the corresponding NOEDGE
run was preempted with existing checkpoints preserved. Both outlook runs retain
their two GPU slots, and M5-NOCTX remains held.

## Current OUTLOOK method

Implementation snapshot: 2026-09-17. This describes the running M6-OUTLOOK
(ALFWorld) and M7-OUTLOOK (WebShop) arms, both using
Qwen2.5-1.5B-Instruct. The implemented estimator mixes a historical Monte Carlo
advantage with a two-step bootstrapped outlook advantage:

$$
A_i^{\mathrm{mix}}
=(1-\beta)A_i^{\mathrm{hist}}+\beta A_i^{\mathrm{out}},
\qquad \beta=0.25.
$$

The two terms share a context-conditioned, cross-trajectory kernel value
estimator. Its local baseline is shrunk toward a task prior according to the
number of supporting trajectories. Episode-advantage and G2PO edge-advantage
coefficients are both zero.

**Trajectory and reward notation.** Let $i=(q,j,t)$ identify a task $q$, a sampled
trajectory $j$, and a zero-based environment turn $t<T_j$. Write $o_i$ for its
anchor observation, $a_i$ for its generated response/action, and $r_i$ for its
immediate environment reward. Both arms train on the binary environment reward
channel, with success reward 10 and failure reward 0. WebShop partial-credit
scores are evaluation diagnostics, not this arm's training target.

Define the raw discounted return and the current-turn validity penalty by

$$
R_{j,t}=\sum_{u=t}^{T_j-1}\gamma^{u-t}r_{j,u},\qquad
p_{j,t}=0.1\,\mathbf 1\{\texttt{is\_action\_valid}_{j,t}=0\},
\qquad \gamma=0.95,
$$

and the historical target by

$$
Y_{j,t}=R_{j,t}-p_{j,t}.
$$

The existing harness subtracts the penalty at the current turn only. It does
not subtract the discounted sum of all future invalid-action penalties. This
convention is retained by OUTLOOK. The validity flag is the environment's
recorded flag; it is not a new penalty for every action absent from an admissible
list.

**Context representation.** The prompt $H_{j,t}$ contains the task, current
observation, and up to two preceding observation/action turns. Obtain $h_i$ from
the frozen reference model's final hidden state at the last prompt token. The
causal prompt representation does not include the current generated response.
The reference forward pass already needed for KL supplies these features.
WebShop's existing long-prompt fallback can omit textual history; OUTLOOK uses
the actual captured prompt features in that case as well.

From the chronological observation prefix, compute

$$
U_{j,t}=|\{o_{j,0},\ldots,o_{j,t}\}|,\qquad
v_{j,t}=\mathbf 1\{o_{j,t}\in\{o_{j,0},\ldots,o_{j,t-1}\}\},
\qquad f_{j,t}=\frac{U_{j,t}}{t+1}.
$$

Here $f$ measures observation novelty per turn, not task completion. Concatenate
12-bin thermometer encodings of turn index, unique-observation count and novelty
fraction, followed by the revisit indicator:

$$
c_i=[\operatorname{th}_{[0,30]}(t);\,
       \operatorname{th}_{[0,25]}(U_i);\,
       \operatorname{th}_{[0,1]}(f_i);\,v_i].
$$

A thermometer vector has its first
$\lfloor12\,\operatorname{clip}((x-l)/(u-l),0,1)\rfloor$ entries set to one.
Center the hidden features over the canonical rollout batch, remove its top
three principal directions, and normalize. With $P_3$ containing those
principal directions and $\operatorname{norm}$ denoting L2 normalization with a
numerical floor,

$$
\widetilde h_i=\operatorname{norm}((I-P_3P_3^\top)(h_i-\bar h)),\qquad
\phi_i=\operatorname{norm}([\widetilde h_i;\operatorname{norm}(c_i)]).
$$

The hidden and context blocks have equal configured weight. This preprocessing
is centering and principal-component removal, rather than full covariance
whitening. The accumulated statistics use all observations through the current
turn, even though textual history is limited to two turns.

**Cross-trajectory value readout.** The exact observation bucket and its
reference set for query $i$ are

$$
\mathcal B_i=\{\ell:q_\ell=q_i,\ o_\ell=o_i\},\qquad
\mathcal N_i=\{\ell\in\mathcal B_i:j_\ell\ne j_i\}.
$$

All occurrences from the query's own trajectory are excluded from the return
averages. For reference occurrences, use

$$
d_{i\ell}=\|\phi_i-\phi_\ell\|_2,\qquad
w_{i\ell}=\exp(-d_{i\ell}/\tau_{\mathcal B_i}),\qquad
\tau_{\mathcal B}=0.15\operatorname{median}_{u<v,\,u,v\in\mathcal B}d_{uv},
$$

with a positive fallback when the median distance vanishes. The median is over
all distinct row pairs in the bucket; the own-trajectory exclusion applies to
the reference return averages.

Use one linear readout for either target channel $z\in\{Y,R\}$:

$$
b_i^{\mathrm{node}}[z]
=\frac{\sum_{\ell\in\mathcal N_i}w_{i\ell}z_\ell}
       {\sum_{\ell\in\mathcal N_i}w_{i\ell}},\qquad
b_i^{\mathrm{task}}[z]
=\frac{\sum_{\ell\in\mathcal P_i}z_\ell}{|\mathcal P_i|},\qquad
\mathcal P_i=\{\ell:q_\ell=q_i,j_\ell\ne j_i\}.
$$

The implementation accumulates weights within reference trajectories and then
combines them, which is algebraically the occurrence-weighted expression above.
It does not give each trajectory equal total mass. The task prior also averages
occurrences. Distinct trajectories determine the support count

$$
J_i=|\{j_\ell:\ell\in\mathcal N_i\}|,
\qquad \lambda_{k,i}=\frac{J_i}{J_i+\kappa},\qquad\kappa=2.
$$

For a supported exact bucket, the applied value is

$$
\boxed{
\widehat V_i[z]=\lambda_{k,i}b_i^{\mathrm{node}}[z]
 +(1-\lambda_{k,i})b_i^{\mathrm{task}}[z].
}
$$

Define $\widehat V_i^{\mathrm{pen}}=\widehat V_i[Y]$ and
$\widehat V_i^{\mathrm{env}}=\widehat V_i[R]$. Both channels use the same features,
neighbors, kernel weights and credibility coefficient. They differ only in the
return values being averaged.

When the exact bucket has no other trajectory, the reference pool expands to
other trajectories of the same task and uses the context-weighted task-pool
readout. No additional node-to-task blend is applied on this fallback level. If
no cross-trajectory pool is available, the row receives zero credit. The logged
`ccpo/lam_k_mean` averages the applied coefficients, including coefficient 1 on
task-backoff rows.

The separate attention-versus-uniform mixing coefficient $\lambda$ is fixed at
1. Its empirical-Bayes diagnostics do not choose the current baseline. The
support-dependent coefficient above is the active credibility shrinkage. The
estimated `kappa_hat` is diagnostic only; applied $\kappa$ remains 2. There is no
additional multiplication of the advantage by $J/(J+c)$ in these arms.

**Historical and future outlook advantages.** Historical credit is

$$
A_{j,t}^{\mathrm{hist}}=Y_{j,t}-\widehat V_{j,t}^{\mathrm{pen}}.
$$

For horizon $n=2$, set $k=\min(2,T_j-t)$ and define

$$
Z_{j,t}^{(2)}
=\sum_{u=0}^{k-1}\gamma^u r_{j,t+u}
 +\mathbf 1\{t+k<T_j\}\gamma^k\widehat V_{j,t+k}^{\mathrm{env}}
 -p_{j,t},
$$

$$
A_{j,t}^{\mathrm{out}}=Z_{j,t}^{(2)}-\widehat V_{j,t}^{\mathrm{pen}}.
$$

The future endpoint uses the same context encoder evaluated at $t+2$. Its
prompt normally contains the intervening two observation/action turns and its
current observation $o_{t+2}$; its accumulated statistics are updated through
$t+2$. Thus the outlook uses realized future context in a bootstrapped target.
It does not concatenate that future into the baseline at time $t$, and the
acting policy never receives future observations when choosing an action.

At a terminal endpoint, including the recorded rollout horizon, the bootstrap
is zero. If a nonterminal endpoint has no supported value estimate, set
$A^{\mathrm{out}}=A^{\mathrm{hist}}$ for that query. Only supported current rows
receive nonzero credit.

For an ordinary nonterminal two-step endpoint, the implemented mixture is

$$
\boxed{
A_{j,t}^{\mathrm{mix}}
=0.75\big[R_{j,t}-p_{j,t}-\widehat V_{j,t}^{\mathrm{pen}}\big]
+0.25\big[r_{j,t}+0.95r_{j,t+1}
 +0.95^2\widehat V_{j,t+2}^{\mathrm{env}}
 -p_{j,t}-\widehat V_{j,t}^{\mathrm{pen}}\big].
}
$$

Equivalently, OUTLOOK adds a bootstrapped-tail correction to historical credit:

$$
A_{j,t}^{\mathrm{mix}}-A_{j,t}^{\mathrm{hist}}
=0.25\,\gamma^2\big(\widehat V_{j,t+2}^{\mathrm{env}}-R_{j,t+2}\big).
$$

This identity applies when the two-step endpoint is nonterminal and supported.
For rows reaching termination within two steps, the outlook and historical
advantages agree, up to floating-point precision. Value estimation can introduce
bootstrap bias; no unbiasedness guarantee is claimed.

**Normalization and policy optimization.** ALFWorld standardizes the mixed
advantage over supported rows of the same task, using sample standard deviation:

$$
\overline A_i=
\begin{cases}
(A_i^{\mathrm{mix}}-\mu_{q_i})/(s_{q_i}+10^{-6}),&\text{ALFWorld},\\
A_i^{\mathrm{mix}},&\text{WebShop}.
\end{cases}
$$

Tasks with fewer than two supported rows skip standardization. This outer
normalization occurs after canonical credits are mapped back to the training
batch. WebShop's `mean_norm` mode does not apply an additional centering here;
the residual already subtracts its estimated baseline.

Broadcast the turn advantage to its valid generated-response tokens, with mask
$m_{i,l}$ and stop-gradient operator $\operatorname{sg}$:

$$
A^{\mathrm{actor}}_{i,l}=m_{i,l}\operatorname{sg}(\overline A_i),
\qquad w_{\mathrm{episode}}=w_{\mathrm{edge}}=0,
\qquad w_{\mathrm{step}}=1.
$$

The episode advantage remains logged for comparison but has zero coefficient
in actor advantages. Returns, reference features, kernel weights, credibility
weights and future bootstrap values are detached from policy backpropagation.
They influence the update numerically through $A^{\mathrm{actor}}$; gradients
flow through the actor's token probabilities and its regularization terms.
There is no learned value head or auxiliary critic-fitting loss.

For token ratio
$\rho_{i,l}(\theta)=\pi_\theta(a_{i,l}\mid H_i,a_{i,<l})/
\pi_{\mathrm{old}}(a_{i,l}\mid H_i,a_{i,<l})$, define

$$
g_{i,l}=\min\left(\rho_{i,l}\overline A_i,
\operatorname{clip}(\rho_{i,l},0.8,1.2)\overline A_i\right),
\qquad
g^{\mathrm{dual}}_{i,l}=
\begin{cases}
\max(g_{i,l},3\overline A_i),&\overline A_i<0,\\
g_{i,l},&\overline A_i\geq0.
\end{cases}
$$

The actor minimizes the masked token-mean loss, with the harness's microbatch
and gradient-accumulation weighting:

$$
\mathcal L(\theta)
=-\langle g^{\mathrm{dual}}\rangle_m
 +0.01\langle\widehat D_{\mathrm{KL}}\rangle_m
 -0.001\langle\mathcal H(\pi_\theta)\rangle_m.
$$

The reference policy is frozen. The implemented low-variance KL surrogate is
$\widehat D_{\mathrm{KL}}=\operatorname{clip}(e^d-d-1,-10,10)$ with
$d=\operatorname{clip}(\log\pi_{\mathrm{ref}}-\log\pi_\theta,-20,20)$.
KL is an actor-loss term, not a reward-shaping term in these runs.

**Batch integrity and current protocol.** The outlook estimator reconstructs
chronological trajectories using explicit turn indices, removes duplicate
padding rows from its value-estimation pools, and restores credits to all
training rows afterward. Incomplete trajectories fail validation. These
ordering/padding corrections also affect historical credit relative to the
older NOEDGE implementation. A control that isolates the outlook mixture itself
must retain `horizon=2` and set `beta=0`.

Both runs start from the base 1.5B model with seed 0, use 16 tasks with 8
trajectories per task per rollout batch, learning rate $10^{-6}$, 150 training
steps, and evaluation/checkpointing every 5 steps. Each run uses two GPUs. The
outlook reuses collected trajectories and existing reference features; it adds
no environment rollouts or language-model forward passes.

Implementation: [outlook estimator](../../ccpo/outlook.py),
[context and value readout](../../ccpo/core_ccpo.py),
[trainer integration](../../patches/verl-agent/verl/trainer/ppo/ray_trainer.py),
[actor loss](../../patches/verl-agent/verl/workers/actor/dp_actor.py).
Run definitions: [M6 config](../m6-ccpo-attncred-ctxadv-outlook-alfworld-1.5b-2gpu-20260917/config.json)
and [M7 config](config.json).

## History and outlook diagnostics

`outputs/metrics.jsonl` logs both component mean absolute advantages as
`ccpo/history_adv_absmean` and `ccpo/outlook_adv_absmean`, their correlation
(`ccpo/outlook_history_corr`), the actual change from historical credit
(`ccpo/outlook_delta_absmean`), beta, horizon, endpoint usage/terminal/fallback
fractions, and canonical/padded row counts.

[The OUTLOOK dashboard](plots/outlook.png) plots the raw component magnitudes,
their magnitudes after applying `(1-beta)` and `beta`, raw and weighted ratios,
component correlation, the mixture's change from history, and endpoint coverage.
Weighted magnitudes and ratios are derived by the plotter from logged values;
they are not additional fields written into the training log. All component
magnitudes are measured before outer task normalization on canonical supported
rows. The separately logged `ccpo/adv_cc_absmean` is measured after the trainer's
normalization and restoration of padded rows, so it is not directly overlaid.

The component summaries do not record each row's reward prefix, penalized
current baseline, unpenalized future bootstrap, signed component means, or
component standard deviations separately. `ccpo_samples.csv` retains historical
pre-mixture credit; it does not contain a full per-row outlook decomposition.

Offline same-sample comparison: [OUTLOOK versus the pure edge term](../OUTLOOK_EDGE_COMPARISON.md).
