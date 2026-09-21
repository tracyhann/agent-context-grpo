# M10 / M11 H2 no shrinkage, no episode: no-LOO comparison

Prepared on **2026-09-21**, at the user's request. Two Qwen2.5-1.5B-Instruct
experiments, 150 steps, seed 0, two GPUs each. **Neither launched nor queued.**
The selected [main method](MAIN_METHOD.md) retains whole-trajectory exclusion.
These variants change one method flag: `ccpo_loo=0` / `ACG_CCPO_LOO=0`.

| Arm | Benchmark | Prepared experiment | Matched EP0 control |
|---|---|---|---|
| M10 | ALFWorld | [Config and notes](m10-h2-noshrink-noloo-alfworld-1.5b-2gpu-20260921/NOTES.md) | [Main H2 no-shrink](m10-h2-no-credit-shrinkage-alfworld-1.5b-2gpu-20260919/NOTES.md) |
| M11 | WebShop | [Config and notes](m11-h2-noshrink-noloo-webshop-1.5b-2gpu-20260921/NOTES.md) | [Main H2 no-shrink](m11-h2-no-credit-shrinkage-webshop-1.5b-2gpu-20260919/NOTES.md) |

Registry key: `future-progress-h2-no-credit-shrinkage-no-loo`.
Variant: `CCPO-ATTNCRED-FUTURE-PROGRESS-TWO-STEP-NOSHRINK-NOLOO`.
IDs: `ccpo-attncred-abl-fph2-noshrink-noloo-alfworld-1.5b` and
`ccpo-attncred-abl-fph2-noshrink-noloo-ws-1.5b`.

## Peer selection and math

Interpretation of **no LOO**: include the query occurrence itself AND matching
revisits from the same trajectory. Removing only the trajectory exclusion while
retaining an occurrence-level exclusion would be a different ablation.

For query occurrence u=(i,t), define its exact peer set:

\[
\mathcal P_u^{\rm all}=\{v=(j,s):\operatorname{task}(v)=\operatorname{task}(u),\ O_v=O_u\}.
\]

The main method additionally requires j != i. This variant removes that condition,
so u is an eligible peer with distance zero. Same-trajectory turns with different
observations are not exact peers. Padding replicas sharing the same trajectory
and turn ID are canonicalized away **before** feature processing and estimation;
they never count as additional peers.

Let x_u be the processed frozen reference hidden state plus context-statistics
representation used by the main method. Retain its whitening, block normalization,
context weight and soft kernel:

\[
w_{uv}=\exp\{-\|x_u-x_v\|_2/\tau_b\},\qquad
p_{uv}=\frac{w_{uv}}{\sum_{a\in\mathcal P_u^{\rm all}}w_{ua}},\qquad
C_u^{\rm all}[q]=\sum_{v\in\mathcal P_u^{\rm all}}p_{uv}q_v.
\]

The bucket bandwidth is the existing 0.15 times median off-diagonal distance
(with existing degenerate-bucket fallback); self distance is exactly zero.
Bandwidth construction is unchanged. The reduction first accumulates kernel mass
within trajectories, then combines their weighted means; equivalently, C is the
occurrence-weighted average above, not an equal-weight average of trajectories.

No shrinkage means **lambda_u=lambda_k=1** on usable groups, for BOTH return
channels. The history target Y is the existing penalized return supplied by the
trainer. The potential target Z excludes the invalid-action penalty:

\[
H_{i,t}=Y_{i,t}-C_{i,t}^{\rm all}[Y],\qquad
Z_{i,s}=\gamma^{T_i-s}R_i,\qquad V_{i,s}=C_{i,s}^{\rm all}[Z],\quad\gamma=0.95,
\]
\[
F_{i,t}=z_{\rm task}\!\left(V_{i,\min(t+2,T_i)}-V_{i,t}\right),\qquad
A_{i,t,\ell}=M_{i,t,\ell}\,N_{\rm CC}(H+F)_{i,t}.
\]

At a nonterminal future endpoint, V uses **that endpoint's own observation group**
and permits that endpoint and its matching same-trajectory peers. This change
does not switch to fixed-anchor grouping. Terminal V stays 10 for success and 0
for failure. F retains per-task standardization. N_CC is the existing task
mean/sample-std normalization on ALFWorld and identity on WebShop. History/future
actor coefficients are 1/1. Episode and original-edge coefficients are **0/0 on
both benchmarks**. Episode rewards still supply return labels, and raw episode
advantage is logged; the separate episode channel cannot enter the PPO gradient.

## Support and interpretation

With the default min_traj=2, LOO needs at least one other trajectory. No-LOO needs
at least one represented trajectory and the query's own trajectory can supply it.
More generally, no-LOO requires max(1,min_traj-1) represented trajectories. J counts
distinct trajectories in the peer set, including the query trajectory; repeated
visits from one trajectory do not each increment J. If alpha_j is its total
kernel mass, n_eff=(sum alpha_j)^2/sum alpha_j^2.

Every valid query now has an exact match to itself. Under this prepared protocol,
exact singleton buckets no longer need task backoff: C_u[q]=q_u, H_u=0 and V_u=Z_u.
Future progress can still be nonzero because a later discounted return or terminal
sentinel differs. The task fallback machinery remains available for other minimum
support settings. Its task prior follows the same no-LOO policy; it is diagnostic
at lambda_k=1 and does not dilute the context baseline.

Where the main LOO peer set is nonempty, write m_u as total kernel mass from the
query trajectory. With the same fixed features and bandwidth,
C_all=(1-m_u) C_LOO + m_u C_own. Thus the ablation can change both magnitude and
relative allocation of credit. If self is the only own-trajectory match,
H_all=(1-p_uu) H_LOO before downstream normalization. Matching revisits can change
the direction as well. The query has kernel weight 1 before normalization, so its
share can be large with the retained narrow kernel; the experiment measures this.

All features and advantage estimates stay detached, but detaching does not remove
the statistical dependence on the query's own outcome. This ablation intentionally
removes the main method's whole-trajectory exclusion protection; it is not an
unbiasedness claim or a replacement of the main method by designation.

## Diagnostics and verification

The shared core reports `loo`, and per-query `self_mass_values`,
`same_traj_mass_values`, and `peer_rows_values`. Future-progress snapshots expose
`history_`, `current_`, and `future_` versions of `self_mass`, `same_traj_mass`,
and `peer_rows`, together with J, n_eff, lambda_k, kernel/prior values, raw and
applied advantages. Masses refer to the normalized **kernel readout**; in these
arms its applied baseline weight is 1. Future terminal peer metrics are NaN and
excluded from summary statistics because terminal sentinels have no peer readout.

Metrics include `ccpo/progress_loo=0`, and
`ccpo/progress_{history,current,future}_{self_mass,same_traj_mass}_mean`.
The future-progress figure has query-mass and query-trajectory-mass panels.
Existing component weights, actor identity checks, and padding diagnostics remain.

CPU tests check hand-computed weighted means and support counts, matching within
task/observation, self-target dependence, singleton behavior, task/auxiliary priors,
legacy LOO parity, shuffled padded batches and continued strict feature validation.
Tests also run the actual trainer advantage function and PPO loss for both
benchmark normalization modes: no-LOO changes the gradient while perturbing only
the diagnostic episode advantage at EP0 does not. Config tests compare the full
prepared protocols against their respective **EP0** main controls.

## Reproduction

```bash
# Prepare both, without launching. Use a new date to avoid overwriting these records.
python scripts/prepare_no_loo_ablations.py --date YYYYMMDD

# CPU regression suite for this variant:
CUDA_VISIBLE_DEVICES='' PYTHONPATH=tests:. .venv/bin/python -m unittest test_future_progress_no_loo -v
```

Each folder includes the full config/env/Hydra recipe, run.sh, method notes,
PREPARED.json, exact control diff, preparation command and source hashes.
VALIDATION.json records the executed CPU checks. Configs retain the main protocol:
16 tasks x 8 rollouts, 150 steps, evaluation/checkpoints every 5 steps, ALFWorld
50-turn cap, WebShop 15-turn cap and 1K catalog/scorer. The GPU identifiers 0,1 and
2,3 are inherited launch settings, not reservations. No training scores exist for
these newly prepared variants.


Validation completed on 2026-09-21: **10 no-LOO tests + 79 existing regressions
passed**, including fresh-process activation from both prepared environments.
Registry/math guards, documentation/registry consistency and both shell syntax
checks passed. Each experiment's VALIDATION.json and validation logs contain the
commands, scope and source hashes. No GPU training or benchmark rollout was run.
