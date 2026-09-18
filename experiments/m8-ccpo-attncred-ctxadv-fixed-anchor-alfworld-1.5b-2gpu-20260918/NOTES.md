# M8-FIXED-ANCHOR: fixed-anchor history-plus-future gain

Fresh Qwen2.5-1.5B-Instruct, seed 0, 150 steps, GPUs 0,1. User-authorized replacement of m6-ccpo-attncred-ctxadv-outlook-alfworld-1.5b-2gpu-20260917; the prior run and its completed pause checkpoint remain preserved.

## Applied method

Let Y be binary environment return-to-go (gamma .95) minus the current invalid-action penalty, H=Y-B_history, and F=B_joint-B_history. The actor credit before benchmark normalization is **A=H+F**, with both coefficients 1. The episode and original edge coefficients are 0; their diagnostics never enter this arm's actor advantage.

B_history uses the existing frozen-reference historical prompt embedding and 37-dimensional accumulated context. B_joint uses an extra frozen-reference pass over that same historical prompt followed by the next two observed projected actions and resulting observations. The acting policy never receives those future observations. Short terminal windows include their last action and final observed observation; rollout truncation is labeled as the recorded trajectory end.

The joint feature is L2([center/remove-top-3-PCs/L2(joint hidden), L2([L2(history context), L2(future context)])]). Historical features retain their original hidden-plus-context construction. Both baseline readouts use the exact current (task, observation) anchor, leave the entire query trajectory out, and average peer **anchor-time Y targets**, with the production exp(-distance/(.15*median-distance)) kernel and J/(J+2) task-prior shrinkage. The joint readout never regroups at the future observation. If no exact anchor peer exists, F=0 and historical task fallback is preserved.

ALFWorld uses the existing combined per-task mean/std normalization; WebShop keeps return units. Separate components are never independently normalized. Logged applied components use the same combined divisor and their own means so they sum to the actor step advantage.

Gain weight 1 is the chosen experiment setting, not an empirically optimized coefficient. This is an intentional emphasis of continuation-explained credit: H=F+(Y-B_joint). It is not a claim of independent additional evidence or an unbiased policy gradient.

## Future encoding and complete diagnostics

Joint reference prompts have an 8192-token cap. If needed, both ends of history and future text are preserved within balanced token budgets; truncation rate and exact encoded tokens are saved. Textual policy history remains two turns; both accumulated context vectors retain their complete prefix statistics.

Every step writes:

- `outputs/metrics.jsonl`: history and future baseline means/std/magnitudes; kernel, uniform and task-prior readouts; applied credibility, J and n_eff for both; H, F, residual, combined pre-normalization and applied component summaries; pure-edge correlations with/without terminal windows; coefficients, coverage, prompt length/truncation and decomposition checks.
- `outputs/fixed_anchor/step-NNNN.csv`: all canonical per-row scalar terms, support and flags, before/after normalization.
- `outputs/fixed_anchor/step-NNNN.npz`: scalar terms, observation identities, chronology, targets/rewards/validity, response lengths, both raw hidden embeddings, both actual processed feature matrices, and both accumulated context vectors.
- `outputs/fixed_anchor/step-NNNN.prompts.jsonl.gz`: original history text, observed future transitions, terminal/window flags and exact joint prompt token IDs.
- `plots/fixed_anchor.png`: both readouts, kernel/prior terms, shrinkage, peer support, component means/std/magnitudes, applied components, correlations, coverage, encoding time and identity errors. Existing progress and CCPO plots remain.

Raw per-row future gain is zero on unsupported anchors. Component centering during ALFWorld task normalization can assign a nonzero centered future component to those rows; the sum still exactly matches combined normalization. Future baseline summaries include the applied historical fallback; `measured_future_baseline` and eligibility flags distinguish actual joint readouts.

## Validation and reproducibility

The CPU suite checks the credit identity, shared anchor support/shrinkage, exclusion of own-trajectory returns, chronological/padding invariance, complete terminal contexts, separate aligned reference capture, complete saved fields, both benchmark normalization modes, and unchanged PPO gradients when the zero-weight episode diagnostic is perturbed. Old OUTLOOK and episode-isolation regressions and registry/protocol guards are retained.

Run config: [config.json](config.json). Method: [fixed_anchor.py](../../ccpo/fixed_anchor.py). Tests: [test_fixed_anchor.py](../../tests/test_fixed_anchor.py). Controller: `.local/fixed-anchor-alfworld-chain-20260918` with SHA-256-pinned inputs.

## Launch verification

Both fixed-anchor suites passed (11 checks per Python environment), plus 15 old OUTLOOK checks, 2 episode-gradient isolation checks, and the protocol defaults checks. The arm registry checks passed; the broad vendored-path scan reports pre-existing absolute paths in generated caches, temporary environments and historical run scripts. GPU FlashAttention compiled-kernel probes passed on each benchmark environment before launch. Live step 1 verified: 6093 canonical rows, 165 fixed-anchor scalar metrics, 38 CSV fields, 45 snapshot fields, and 6093 exact prompt records. Both gain identities, shared peers/shrinkage, zero episode/edge coefficients, positive future-reference timing, and unchanged source hashes passed.

Summary masks: history kernel/prior/support/credibility summaries cover all live history rows, including task fallback. Corresponding future summaries cover exact-anchor-supported rows. Their plotted means can differ because their populations differ; per-row peer counts, task priors and credibility are checked equal on the exact-supported rows. `eligible` in each snapshot identifies that matched comparison. Removing the top three principal directions retains the 1536-dimensional hidden vector; processed history and joint features have dimensions 1573 and 1610 respectively.

## Paused for future-progress experiments

Paused on 2026-09-18T04:16:36+00:00 at the user's request. Last completed metric step: 12; saved resume checkpoint: step 10, pinned at `outputs/checkpoints/step10-paused-future-progress`. M10/M11 start fresh from the base model. See `outputs/preemption-future-progress.json`.
