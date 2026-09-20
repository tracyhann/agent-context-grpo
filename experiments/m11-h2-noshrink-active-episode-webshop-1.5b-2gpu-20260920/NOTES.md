# M11 H2: no shrinkage + active episode advantage (webshop)

**Prepared only; not launched or queued.** Fresh Qwen2.5-1.5B-Instruct,
150 steps, seed 0, 8 rollouts per task, two GPUs. GPU IDs `2,3` are copied
protocol settings, not reservations. Prompt history and future horizon are 2.

Registry key: `future-progress-h2-no-credit-shrinkage-active-episode`.
Variant: `CCPO-ATTNCRED-FUTURE-PROGRESS-TWO-STEP-NOSHRINK-ACTIVE-EPISODE`.
Canonical ID: `ccpo-attncred-abl-fph2-noshrink-ep-ws-1.5b`.

## Controls and exact deltas

| Setting | H2 control | H2 no-shrink control | This arm |
|---|---:|---:|---:|
| lambda_k on usable exact groups | J/(J+2) | 1 | **1** |
| Episode coefficient | 0 | 0 | **1** |
| Context-stat coefficient | 1 | 1 | 1 |
| Future horizon | 2 | 2 | 2 |
| History / future coefficients | 1 / 1 | 1 / 1 | 1 / 1 |
| Original edge coefficient | 0 | 0 | 0 |

Against [H2](../m11-h2-ccpo-attncred-ctxadv-future-progress-webshop-1.5b-2gpu-20260918/NOTES.md), the only method changes are
`ccpo_lk_fix: "" -> 1.0` and `ccpo_ep_w: 0 -> 1.0`.
Against [H2 no-shrink](../m11-h2-no-credit-shrinkage-webshop-1.5b-2gpu-20260919/NOTES.md), only `ccpo_ep_w` changes.
That second comparison isolates the episode channel under full-strength context
baselines. Comparing against H1 or an H2 shrinkage control changes multiple factors.
This arm retains the context-statistics vector; it is separate from NOSHRINK-NOCTX.

A pre-preparation audit of 34 parseable configs under local `experiments/` and
`legacy-experiments/` found no full-strength-baseline config with active episode
weight. The three existing explicit no-shrink configs had episode weight 0 and
no training metrics/checkpoints. This is a local-record audit, not a claim about
external jobs whose artifacts are absent here. Earlier offline no-shrink analyses
are not trained active-episode results.

## Math and episode-reward interpretation

For target q in {Y,Z}, let C_s[q] be the existing kernel-weighted baseline
excluding the entire query trajectory. Every usable history/current/future
readout uses **B_s[q]=C_s[q]**, with lambda_k=1, including exact groups with J=1.
Kappa=2 stays recorded but cannot change the fixed weight. Existing task-bucket
fallback, no-peer unsupported rows, and fixed terminal 10/0 potentials remain.
Context statistics stay enabled (`ccpo_ctx_w=1`, `ccpo_phi=hidden+ctx`).

\[
H_{i,t}=Y_{i,t}-C_{i,t}[Y],\qquad
Z_{i,s}=\gamma^{T_i-s}R_i,\qquad V_{i,s}=C_{i,s}[Z],
\]
\[
F_{i,t}=\operatorname{z}_{\mathrm{task}}
 \left(V_{i,\min(t+2,T_i)}-V_{i,t}\right).
\]

Each endpoint uses its own observation group. History predicts the existing
penalized return Y; potentials predict the unpenalized label Z. There is no added
original edge term and no change to the source/destination grouping.

The active episode channel reuses the actual trainer's `episode_norm_reward`.
Let S_it be the sum of `token_level_rewards` on that response. Under these
prepared configs, rewards contain the episode's binary outcome (10/0) and the
existing **per-turn invalid-action penalty -0.1**; KL remains a separate loss.
Thus S_it=R_i-0.1*1[invalid action at t]. No dense WebShop score is substituted.
The current helper computes episode-channel mean/std across **turn rows within
the task** (`compute_mean_std_cross_steps=True`), not once per distinct trajectory.
This preserves the existing length weighting rather than introducing a new
trajectory-uniform normalization experiment.

\[
E_{i,t}=\begin{cases}
(S_{i,t}-\mu_q^{EP})/(\sigma_q^{EP}+10^{-6}),&\mathrm{ALFWorld},\\
S_{i,t}-\mu_q^{EP},&\mathrm{WebShop}.
\end{cases}
\]
\[
A_{i,t,\ell}=M_{i,t,\ell}
\left[E_{i,t}+\mathcal N_{CC}(H_{i,t}+F_{i,t})\right],
\quad
\mathcal N_{CC}=\begin{cases}
\text{existing per-task mean/sample-std over live step rows},&\mathrm{ALFWorld},\\
\text{identity},&\mathrm{WebShop}.
\end{cases}
\]

**Episode advantage is added after H+F normalization**, with coefficient 1.
There is no final renormalization of E+N_CC(H+F). The episode channel applies to
all valid response tokens, independent of contextual-baseline support. Advantage
targets and frozen features stay detached; the nonzero episode coefficient
changes the actor's PPO policy gradient. Weight 0 remains diagnostic-only.

## Runtime, logging and launch preparation

The old future-progress logging guard rejected nonzero episode weight. The
runtime now permits finite nonnegative episode weights and validates the complete
masked actor tensor **after fusion**. It still rejects competing OUTLOOK modes,
original edge credit, invalid features and unsupported step scaling. The
checked-in trainer overlay and live checkout are synchronized; no running
process was restarted.

Snapshots and scalar logs contain `episode_adv`, `episode_applied`, and
`actor_applied`, alongside all existing history/current/future baselines, support,
lambda and component diagnostics. `combined_applied` retains its historical
meaning N_CC(H+F); `actor_applied` includes the episode channel. The logger checks
both H_applied+F_applied=combined_applied and the complete token-masked
H_applied+F_applied+E_applied=actor advantage. The future-progress plot now includes
the applied episode magnitude, total actor magnitude, actual episode coefficient,
and full actor identity error. Raw episode diagnostics remain available at weight 0.

Files: [resolved config](config.json), [H2 delta](config-diff-from-h2.json),
[no-shrink delta](config-diff-from-noshrink.json), [preparation status](PREPARED.json),
[CPU validation](VALIDATION.json), [source hashes](prepared-source-sha256.json).
The local gitignored `run.sh` is generated but unexecuted. On this host, it
reproduces the copied two-GPU protocol; launch only when scheduling is authorized:

```bash
bash experiments/m11-h2-noshrink-active-episode-webshop-1.5b-2gpu-20260920/run.sh
```

The generic registry CLI remains available via `ablations/run.py --list`; its
existing GPU-floor guard differs from the copied two-GPU experiment protocol.
For another host, regenerate from this full config and check local runtime assets.

[Gradient and configuration regressions](../../tests/test_future_progress_active_episode.py)
exercise both benchmark normalization modes, exact real episode normalization,
full-strength readouts (including J=1), active-episode changes to PPO gradients,
zero-weight isolation, response masks, snapshot identities and config/runtime
parity. CPU checks do not replace the eventual GPU smoke test. No new result is
reported for this prepared arm.

Validation completed: **67 distinct CPU unittest cases passed**, including 39 future-progress tests also repeated successfully in the WebShop environment (**106 total test executions**). Registry/config/overlay/document guards, the negative FlashAttention probe under its intended system Python, generated shell syntax, Python compilation, diff formatting and synthetic plot rendering passed. No GPU was used. See VALIDATION.json for commands/evidence and probe-environment details.
