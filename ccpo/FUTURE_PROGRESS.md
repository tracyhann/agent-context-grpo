# Contextual future-state progress

Status: primary one-step M10 (ALFWorld, GPUs 0–1) and M11 (WebShop, GPUs 2–3) launched on 2026-09-18 at 04:14 UTC after explicit user authorization. Both use fresh Qwen2.5-1.5B-Instruct. M8/M9 fixed-anchor jobs are paused, with saved checkpoints pinned at steps 10/25. GPU holders covered the handoff and remain until each new trainer completes its first optimizer step. Separate two-step ablations remain prepared and unlaunched. Preparation-time source snapshots remain archived under `.local/future-progress-prepare-20260918/source-before/`; launch records are in `.local/future-progress-launch-20260918/`.

## Purpose

M3 (ALFWorld) and M5 (WebShop) add a standardized source-to-destination value increase to our historical credit. The earlier fixed-anchor variant instead re-estimated the return at the same source observation after seeing a continuation. This implementation restores the former structure while conditioning the endpoint values on context.

Let a trajectory be `(O_0,a_0,...,O_{T-1},a_{T-1},terminal)` with task `q`, episode reward `R` (10 on success, 0 otherwise), and discount `gamma=.95`. The trajectory index is `i`. Every value below is detached credit-estimation data, not a learned critic trained by the PPO loss.

## History channel: unchanged estimator

The penalized target remains the M3/M5 return channel:

\[
Y_{i,t}=\sum_{u=t}^{T_i-1}\gamma^{u-t}r_{i,u}
           -0.1\,\mathbf 1[\text{invalid action}_{i,t}].
\]

Our existing conditional baseline estimates this target from other trajectories. Denote it by `B^-_{i,t}`. Historical credit is

\[
H_{i,t}=Y_{i,t}-B^-_{i,t}.
\]

Both benchmarks keep the binary-return target used by their actual M3/M5 controls. WebShop's dense task score remains an evaluation metric; it is not substituted into this arm's value labels or history target.

## Features and groups

At each observed turn `s`, the normal frozen reference pass already produces the last-prompt-token hidden vector `h_{i,s}` from the policy's task/current observation and two-turn text history. The accumulated prefix statistics are

\[
c_{i,s}=(s,\;n_{\mathrm{unique},i,s},\;\mathrm{revisit}_{i,s},\;
                n_{\mathrm{unique},i,s}/(s+1)).
\]

The last quantity measures observation novelty, not measured distance to the goal. The existing thermometer encoding produces 37 context coordinates. The production feature is

\[
z_{i,s}=\operatorname{L2}\left([
 \operatorname{L2}(\operatorname{removeTop3PC}(h_{i,s}-\bar h)),
 \operatorname{L2}(\operatorname{thermometer}(c_{i,s}))]\right).
\]

The context coefficient is 1. Removing three principal directions retains the hidden vector's original 1536 dimensions; the combined feature has 1573 dimensions. Whitening/centering is batch-wide and performed once; both history and potential readouts use the exact same processed matrix.

Hidden-only ablations may use either `ccpo_phi=hidden` or
`ccpo_phi=hidden+ctx` with `ccpo_ctx_w=0`. Both produce the same 1536-dimensional
processed hidden representation in history and both potential readouts. The
future-progress entry point accepts both spellings and still requires finite
`phi_feats` from the frozen reference; `bow` is unsupported. The launcher rejects
unsupported modes before preparing files or starting model probes/training.
This compatibility fix does not change feature capture or padding checks.

For each endpoint separately, define

\[
\mathcal P_{i,s}=\{(j,u):q_j=q_i,\ O_{j,u}=O_{i,s},\ j\ne i\}.
\]

The future group is keyed by the reached observation `O_{i,t+h}`. Peers may arrive there from different source observations. We do not intersect it with the current group or hard-group the entire future sequence. The entire query trajectory, including its revisits, is excluded.

Within a group, use the existing soft kernel

\[
w_{(i,s),(j,u)}=\exp(-\|z_{i,s}-z_{j,u}\|_2/\tau_g),\qquad
\tau_g=0.15\,\operatorname{median}_{a<b\in g}\|z_a-z_b\|_2.
\]

As in the existing estimator, a zero/near-zero median uses 1 before multiplying by .15. Kernel bandwidth uses the group's features; the reward averaging excludes the query trajectory.

## Unpenalized endpoint potentials

To preserve the actual M3/M5 edge convention, the potential labels are

\[
Z_{j,u}=\gamma^{T_j-u}R_j.
\]

This exponent has one more discount than the ordinary sparse terminal return-to-go at turn `u`; it intentionally matches `g2po_node_values`, including the final jump to the terminal sentinel. These labels contain no invalid-action penalty.

For a supported exact observation group,

\[
V^{\mathrm{node}}_{i,s}
 =\frac{\sum_{(j,u)\in\mathcal P_{i,s}}w_{(i,s),(j,u)}Z_{j,u}}
        {\sum_{(j,u)\in\mathcal P_{i,s}}w_{(i,s),(j,u)}},
\]

\[
\widehat V_{i,s}
 =\lambda_{i,s}V^{\mathrm{node}}_{i,s}
 +(1-\lambda_{i,s})V^{\mathrm{task}}_{-i},
\qquad \lambda_{i,s}=\frac{J_{i,s}}{J_{i,s}+2}.
\]

`J` counts distinct other trajectories. The task prior is the unweighted mean of `Z` over all visits in the same task outside trajectory `i`. The attention-versus-uniform blend is fixed at 1, as in M3/M5. Kernel averages retain the existing visit weights; `J` and effective support are calculated at trajectory level.

If no other trajectory visits the exact observation, preserve the existing estimator's **context-weighted whole-task backoff**. It uses other trajectories and has no further node-to-task shrinkage. It is not an automatic zero-value endpoint. If no other trajectory exists even at task level, the contextual potential is unsupported and its progress credit is zero.

Current and future groups can have different `J`, effective support and credibility. Unlike fixed-anchor gain, they are not required to share peers or priors at the observation-group level.

## Future progress and normalization

For horizon `h` equal to 1 (primary) or 2 (ablation), set `e=min(t+h,T_i)`. At a recorded terminal endpoint,

\[
\widehat V_{i,T_i}=
\begin{cases}10,&R_i=10,\\0,&\text{otherwise}.\end{cases}
\]

This includes treating rollout horizon exhaustion as the failure sentinel, as M3/M5 did. No terminal hidden-state embedding is needed. Terminal future feature/peer fields are explicitly missing in snapshots.

Raw future progress is

\[
P^{(h)}_{i,t}=\widehat V_{i,e}-\widehat V_{i,t}.
\]

There is **no extra `gamma^h` multiplier**, no additional immediate-reward sum, and no convex beta blend. This matches the implemented M3/M5 edge-difference convention; it is not the conventional discounted TD residual `r+gamma V_next-V_current`.

Standardize raw progress separately within each task, over supported rows:

\[
E^{(h)}_{i,t}=\frac{P^{(h)}_{i,t}-\mu_q(P^{(h)})}
                       {s_q(P^{(h)})+10^{-6}},
\]

where `s_q` is the sample standard deviation (`ddof=1`). Fewer than two supported rows produce zero. Unsupported rows stay zero before any final combined normalization. With the normal eight-trajectories-per-task protocol, task backoff supplies nonterminal potentials.

Combine the terms with coefficients 1:

\[
A^{\mathrm{pre}}_{i,t}=H_{i,t}+E^{(h)}_{i,t}.
\]

The original edge coefficient is 0 because this channel replaces it. In the base M10/M11 arms, the episode coefficient is 0, so episode advantage and the original edge are diagnostic-only. The separately named H2 NOSHRINK + ACTIVE-EPISODE ablation below enables the episode channel.

- **ALFWorld / M3 protocol:** apply the existing final per-task mean/sample-std normalization to the combined `A_pre`.
- **WebShop / M5 protocol:** use `A_pre` directly, with no further combined normalization. The future term was already standardized, just as the original edge was.

Broadcast the resulting scalar over valid response tokens. PPO clipping and the original KL regularizer remain unchanged. Applied history/future diagnostics share the combined divisor on ALFWorld and sum to the step-channel advantage (the full actor advantage when episode weight is zero); they are not independently standardized a second time.

A positive raw value increase can become negative standardized credit if it is below the task's mean increase. This is also true of M3/M5's edge normalization.

## One-step identity and two-step extension

With uniform self-inclusive node means instead of contextual leave-trajectory-out values, the primary method exactly recovers the original edge on the same canonical batch:

\[
V^{\mathrm{M5}}(q,O)=\operatorname{mean}_{(j,u):q_j=q,O_{j,u}=O}
                      \gamma^{T_j-u}R_j,\qquad
E^{(1)}=\operatorname{Normalize}_{q}(V^{\mathrm{M5}}_{t+1}-V^{\mathrm{M5}}_t).
\]

`readout='m5'` is available in the estimator for CPU parity checks; the trainer and registered arms always select the contextual readout. In the contextual method, affinity weighting, trajectory exclusion and shrinkage are deliberate changes, so equality with the historical edge is not expected.

Using the same contextual value estimator throughout a trajectory gives, for nontruncated two-step windows,

\[
P^{(2)}_t=(\widehat V_{t+1}-\widehat V_t)
          +(\widehat V_{t+2}-\widehat V_{t+1}).
\]

Normalize the sum afterward. It is not the sum of separately normalized edges, and it assigns continuation-level credit to the action at `t`. There is no claim that this auxiliary signal is an unbiased policy-gradient advantage or is empirically better before training.

## Temporal ordering and diagnostics

Rows are ordered by task, trajectory and explicit turn index before value estimation. Duplicate padding votes are removed; complete trajectories and duplicate consistency are checked; credits are restored to trainer order afterward. Reference-mode parity is defined on this canonical batch. The final ALFWorld normalization retains the trainer's existing handling of restored padded rows.

Every step records the history baseline/residual, value labels, current/future potentials, raw/standardized progress, its per-task mean/std, original edge, applied components, endpoint indices and window lengths, both endpoint kernel/uniform/task-prior readouts, distinct/effective support, actual credibility, fallback levels, terminal flags, hidden vectors and processed features/context. Scalar summaries go to `metrics.jsonl`; row-level CSV and NPZ snapshots go to `outputs/future_progress/`. A separate `future_progress.png` plots these terms. Future peer summaries exclude terminal endpoints, which have fixed values.

Implementation: [future_progress.py](future_progress.py). Validation: [test_future_progress.py](../tests/test_future_progress.py). Offline replay and CPU check results: [VALIDATION.md](../experiments/future-progress-offline-20260918/VALIDATION.md).


## H2 no-shrinkage + active episode variant (2026-09-20)

`future-progress-h2-no-credit-shrinkage-active-episode` retains context statistics
and sets `ccpo_lk_fix=1`, `ccpo_ep_w=1` on the H2 parent. Usable history and both
potential baselines use the full kernel estimate. Existing fallback and terminal
rules remain. The final advantage is E+N_CC(H+F): episode credit is added after
step-channel scaling, without normalizing the fused sum again. Episode credit is
standardized on ALFWorld and mean-centered on WebShop using the existing helper.
Its inputs include the original per-turn invalid-action penalty, and its task
moments use turn rows rather than deduplicating trajectories. Endpoint potentials
continue to use the separate unpenalized episode return. No estimator gradient
passes through frozen features; the nonzero episode term does affect PPO gradients.

The finalizer accepts nonzero episode weight with explicit episode/mask/actor
tensors and verifies their complete token-level identity. `combined_applied`
stays the H+F channel; new `episode_adv`, `episode_applied`, `actor_applied` and
`progress_actor_identity_error` distinguish diagnostics from actual actor credit.
The episode coefficient and applied contribution are plotted. EP=0 remains
supported with unchanged actor advantages and gradients.

Full math, protocol and paired controls:
[M10 ALFWorld](../experiments/m10-h2-noshrink-active-episode-alfworld-1.5b-2gpu-20260920/NOTES.md),
[M11 WebShop](../experiments/m11-h2-noshrink-active-episode-webshop-1.5b-2gpu-20260920/NOTES.md).
Both are prepared and unlaunched; GPU execution has not been validated.
