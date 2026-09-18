# M11: contextual future-state progress — WebShop

Launched 2026-09-18T04:14:39+00:00 from fresh Qwen2.5-1.5B-Instruct, seed 0, for 150 optimizer steps on GPUs 2,3. This document describes the active **one-step** method.

The matched control is M5: `m5-ccpo-attncred-ctxadv-ret-ws-1.5b-2gpu-20260916`. All shared training, environment, optimization, validation and checkpoint settings match its recorded config. The only changed pre-existing method setting is `ccpo_edge_w: 1 → 0`; the new progress channel replaces it. Additional method flags and inactive defaults are recorded in [config-diff-from-control.json](config-diff-from-control.json).

| Run setting | Value |
|---|---|
| Backbone | Qwen2.5-1.5B-Instruct |
| GPUs | 2,3 (two GPUs) |
| Initialization | Fresh base model; seed 0; no checkpoint resume |
| Training | 150 steps; batch of 16 tasks; 8 trajectories per task |
| Environment | At most 15 actions; history length 2 |
| Discount / learning rate | 0.95 / 1e-06 |
| Validation / save frequency | Every 5 / 5 steps |
| Future horizon / weight | 1 / 1 |
| History / episode / original-edge weights | 1 / 0 / 0 |
| Credibility | kappa 2; dynamic J/(J+2) |
| Final fusion mode | `mean_norm` |

## Method overview

This arm adds context-conditioned future-state progress to our unchanged historical advantage. It follows the original M3/M5 edge structure: estimate the current and next endpoint values, subtract them, standardize that difference within each task, and add it to historical credit. The endpoint value estimator changes; the additive fusion and its benchmark-specific normalization match the control.

The active horizon is **one step**. The separate H2 experiment is unlaunched. This run does not use M8/M9's fixed-anchor gain, which compares two estimates of the return at the same source observation after revealing a continuation.

## 1. Trajectories and two distinct return targets

For trajectory \(i\) of task \(q_i\), let \(O_{i,t}\) be the anchor observation before action \(a_{i,t}\), and \(T_i\) its number of recorded actions. Episode outcome \(R_i\) is 10 on success and 0 otherwise; \(\gamma=0.95\).

The historical channel keeps the existing penalized return-to-go:

\[
Y_{i,t}=\sum_{u=t}^{T_i-1}\gamma^{u-t}r_{i,u}
       -0.1\,\mathbf 1[\text{invalid action}_{i,t}].
\]

The endpoint potential channel instead uses the original edge's unpenalized labels:

\[
Z_{i,t}=\gamma^{T_i-t}R_i.
\]

The exponent in \(Z\) deliberately has one more discount than ordinary sparse terminal return-to-go at action \(t\), matching `g2po_node_values`. The invalid-action penalty belongs to \(Y\), not \(Z\). Both channels use binary environment outcomes. WebShop's dense task score is an evaluation metric, not a replacement training target.

## 2. Context representation at each endpoint

The normal frozen reference-model pass provides the last-prompt-token hidden vector \(h_{i,s}\) at each observed turn \(s\). Its prompt contains the task/current observation and the policy's two-turn text history. Accumulated prefix statistics are

\[
c_{i,s}=\left(s,\ n_{\mathrm{unique},i,s},\ \mathrm{revisit}_{i,s},\
                 \frac{n_{\mathrm{unique},i,s}}{s+1}\right).
\]

The final statistic measures observation novelty, not distance to the goal. The existing thermometer encoding maps these statistics to 37 coordinates. The processed feature is

\[
z_{i,s}=\operatorname{L2}\!\left([
 \operatorname{L2}(\operatorname{removeTop3PC}(h_{i,s}-\bar h)),\
 \operatorname{L2}(\operatorname{thermometer}(c_{i,s}))]\right).
\]

The context coefficient is 1. Removing the top three principal directions retains the hidden vector's 1536 dimensions; the concatenated feature has 1573 dimensions. Batch-wide centering and principal-direction removal are performed once; historical and potential estimates reuse the exact same processed feature matrix.

The current endpoint uses \(z_{i,t}\); the future endpoint uses \(z_{i,t+1}\), including that endpoint's own history and accumulated statistics. We reuse the frozen representations already collected for those turns. There is no additional future-prompt model pass or concatenation of the whole continuation into a source-anchored query.

## 3. Peer groups, contextual averaging, and credibility shrinkage

For each endpoint separately, peers are visits to the same anchor observation in the same task, excluding the entire query trajectory:

\[
\mathcal P_{i,s}=\{(j,u):q_j=q_i,\ O_{j,u}=O_{i,s},\ j\ne i\}.
\]

Thus current peers are grouped by \((q_i,O_{i,t})\), and future peers by \((q_i,O_{i,t+1})\). Future peers may have arrived from different source observations. The groups need not share peers; we neither intersect them nor require an identical continuation sequence. All revisits from trajectory \(i\) are excluded from reward averaging.

Context similarity weights are

\[
w_{(i,s),(j,u)}=\exp\!\left(-\frac{\|z_{i,s}-z_{j,u}\|_2}{\tau_g}\right),
\qquad
\tau_g=0.15\,\operatorname{median}_{a<b\in g}\|z_a-z_b\|_2.
\]

A zero/near-zero median is replaced by 1 before multiplying by 0.15. Bandwidth estimation uses the group's features; label averaging excludes the query trajectory.

For labels \(X\), where \(X=Y\) for history or \(X=Z\) for potentials, define

\[
K_{i,s}[X]=
\frac{\sum_{(j,u)\in\mathcal P_{i,s}}w_{(i,s),(j,u)}X_{j,u}}
     {\sum_{(j,u)\in\mathcal P_{i,s}}w_{(i,s),(j,u)}},
\qquad
\overline X^{\mathrm{task}}_{-i}
=\operatorname{mean}_{(j,u):q_j=q_i,\,j\ne i}X_{j,u}.
\]

For a supported exact-observation group, the shared baseline operator is

\[
\mathcal B_{i,s}[X]
=\lambda_{i,s}K_{i,s}[X]
 +(1-\lambda_{i,s})\overline X^{\mathrm{task}}_{-i},
\qquad
\lambda_{i,s}=\frac{J_{i,s}}{J_{i,s}+2}.
\]

\(J_{i,s}\) counts distinct other trajectories, not visits. The task prior is an unweighted average over eligible visits; kernel averaging also retains the existing visit weights. Effective support is recorded at trajectory level. Current and future endpoints can have different support and credibility.

`ccpo_lam_fix=1` fixes the separate attention-versus-uniform blend at full attention. It does **not** fix credibility \(\lambda_{i,s}\): `ccpo_lk_fix=''` and `ccpo_prior_kappa=2` leave \(J/(J+2)\) active and logged. No additional J-based scaling of the advantage is applied.

If there are no exact-observation peers, the estimator uses its existing **context-weighted whole-task fallback**, still excluding trajectory \(i\), without further observation-group-to-task shrinkage. A missing exact group does not automatically imply a zero potential. If even task-level peers are absent, that potential is unsupported and its progress credit is zero.

## 4. Historical credit and one-step future progress

The historical baseline and residual are

\[
B^-_{i,t}=\mathcal B_{i,t}[Y],
\qquad
H_{i,t}=Y_{i,t}-B^-_{i,t}.
\]

This is the unchanged historical estimator from the control. Contextual nonterminal potentials are

\[
\widehat V_{i,s}=\mathcal B_{i,s}[Z].
\]

Let \(e=\min(t+1,T_i)\). For a recorded terminal endpoint,

\[
\widehat V_{i,T_i}=
\begin{cases}
10,&R_i=10,\\
0,&\text{otherwise}.
\end{cases}
\]

Unsuccessful rollout-horizon exhaustion uses the failure sentinel, following M3/M5. No terminal hidden vector or peer estimate is required.

Raw future progress is

\[
P_{i,t}=\widehat V_{i,e}-\widehat V_{i,t}.
\]

There is no outer \(\gamma\) multiplier and no additional immediate-reward sum. This reproduces the implemented edge-difference convention, rather than defining a discounted TD residual. Positive raw progress means the reached situation has a higher estimated potential.

Normalize progress separately within each task, across supported rows in the current rollout batch:

\[
E_{i,t}=\frac{P_{i,t}-\mu_q(P)}{s_q(P)+10^{-6}},
\]

where \(s_q\) is the sample standard deviation (`ddof=1`). Fewer than two supported rows give zero progress credit. Unsupported rows remain zero before final combined normalization. A positive raw increase may receive negative normalized credit if it is below the task's average increase.

## 5. Fusion and the actor advantage for this benchmark

History and future weights are both 1:

\[
A^{\mathrm{pre}}_{i,t}=H_{i,t}+E_{i,t}.
\]

**M11 uses WebShop's M5 protocol (`mean_norm`).** Use the combined credit directly, with no additional combined centering or standardization:

\[
A^{\mathrm{actor}}_{i,t}=A^{\mathrm{pre}}_{i,t}=H_{i,t}+E_{i,t}.
\]

The history residual stays in its existing return units. The future term has already been task-standardized, exactly as the original M5 edge was, even though the benchmark's mode is named `mean_norm`. Applied history and future components are simply \(H\) and \(E\).

The separately computed episode-advantage coefficient is **0**, and the original edge coefficient is **0** because contextual progress replaces it. Fixed-anchor gain and the earlier OUTLOOK term are disabled. There is no convex beta blend or 0.25 multiplier. History is not independently standardized before this addition.

The resulting turn scalar is broadcast over valid response tokens and used by the existing PPO clipped policy loss. PPO clipping, the KL regularizer, and optimizer settings are unchanged. Reference features, baselines, potentials, and constructed advantages are detached: no gradient is backpropagated through this credit estimator. The credit still affects the policy gradient through its weight on the PPO loss. Episode outcomes supply return labels, but the separately plotted episode advantage has no coefficient in that loss.

## 6. Exact relationship to the original edge

The control's original endpoint readout is a uniform, self-inclusive visit average:

\[
V^{\mathrm{old}}(q,O)=
\operatorname{mean}_{(j,u):q_j=q,\,O_{j,u}=O}\gamma^{T_j-u}R_j,
\qquad
E_t^{\mathrm{old}}=\operatorname{Norm}_q\bigl(V^{\mathrm{old}}_{t+1}-V^{\mathrm{old}}_t\bigr).
\]

This arm preserves the labels, terminal convention, one-step subtraction, task normalization, coefficients, and benchmark-specific fusion. It changes the value readout through context similarity, whole-query-trajectory exclusion, and credibility shrinkage. It is a contextual edge variant; equality with the original values is not expected.

The low-level `readout='m5'` reference mode substitutes the original uniform/self-inclusive values and reproduces the original edge and combined pre-normalization credit to numerical tolerance on the same canonical batch. The training arm always selects `readout='context'`.

The separate, unlaunched H2 ablation uses \(e=\min(t+2,T_i)\) and \(P_t^{(2)}=\widehat V_e-\widehat V_t\). For full two-step windows, this telescopes into two raw consecutive value increases. Their sum is standardized afterward, with no division by window length. H2 is not active in this run.

## 7. Ordering, diagnostics, and reproducibility

Before estimation, rows are ordered by task, trajectory, and explicit turn index. Duplicate padding votes are removed, complete trajectories and duplicate consistency are checked, and credits are restored to trainer order afterward. ALFWorld's final combined normalization retains the trainer's existing handling of restored padded rows.

Each completed training step records:

- Historical targets, baselines, residuals, and applied historical credit.
- Current/future potential labels and estimates; kernel readouts, uniform readouts, task priors, distinct/effective support, actual credibility, and fallback levels at both endpoints.
- Raw progress, its task mean/std, normalized progress, and the weighted/applied future component.
- Combined pre-normalization and actual actor advantages, plus the numerical error in `history_applied + future_applied = combined_applied`.
- Original edge diagnostics and future-versus-edge correlations, including a nonterminal comparison; diagnostic episode advantage remains excluded from fusion.
- Source/destination indices, actual window length, terminal/support masks, frozen hidden vectors, processed features, and accumulated context statistics.

Scalar metrics are saved in `outputs/metrics.jsonl` under `ccpo/progress_*`, alongside existing historical and episode diagnostics. Per-step numeric snapshots are `outputs/future_progress/step-NNNN.csv`, `.npz`, and `.metrics.json`. Terminal future feature/peer fields are NaN and explicitly masked. Snapshots require no pickle-dependent metadata. The plotter supports `plots/future_progress.png`; both feature matrices needed for offline replay are saved without needing full prompt strings.

The CPU checks passed 12 focused tests in each benchmark environment, including original-edge reference parity, grouping and trajectory exclusion, discount/terminal conventions, both fusion paths, padding/order invariance, logging identities, and episode-gradient isolation. Existing OUTLOOK, fixed-anchor, and episode-isolation regression suites also passed. Validation details, including the optional unavailable legacy fixture, are in [VALIDATION.md](../future-progress-offline-20260918/VALIDATION.md).

An offline replay on one saved batch per benchmark found one-step future-versus-edge correlations of 0.663 on ALFWorld and 0.687 on WebShop. These are signal comparisons, not this run's validation success rates. WebShop's nonterminal correlation was only 0.127 on that batch, so substantial full-batch agreement occurred near terminal windows. This auxiliary credit is not claimed to be an unbiased policy-gradient advantage.

Implementation: [future_progress.py](../../ccpo/future_progress.py); shared estimator: [core_ccpo.py](../../ccpo/core_ccpo.py); tests: [test_future_progress.py](../../tests/test_future_progress.py); shared method reference: [FUTURE_PROGRESS.md](../../ccpo/FUTURE_PROGRESS.md).

## Run records

Launch metadata: [LAUNCH.json](LAUNCH.json); command: [run.sh](run.sh); resolved configuration: [config.json](config.json). Controller state is maintained in `.local/future-progress-webshop-chain-20260918/state.json`. M8/M9 were paused with their saved checkpoints pinned at steps 10/25; GPU holders covered the handoff and released after the first completed optimizer step in each new run. The separate H2 experiments remain unlaunched.

Method documentation expanded on 2026-09-18T04:58:15+00:00. Training code, configs, commands and running processes were not changed by this documentation update.
