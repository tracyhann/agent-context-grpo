# M11 H2 WebShop: no shrinkage, episode advantage disabled, 30 turns

**Prepared only; not launched or queued.** Fresh Qwen2.5-1.5B-Instruct,
150 training steps, two GPUs, seed 0, 8 rollouts per task. Recorded GPU IDs
`2,3` are copied configuration values, not a reservation.

Registry key: `future-progress-h2-no-credit-shrinkage-30turn`.
Canonical ID: `ccpo-attncred-abl-fph2-noshrink-30turn-ws-1.5b`.

## Paired comparisons

The only configuration change from the
[15-turn control](../m11-h2-no-credit-shrinkage-webshop-1.5b-2gpu-20260919/NOTES.md) is **`max_steps: 15 -> 30`**,
plus the experiment ID. The limit applies to both training and validation.
The [other 30-turn arm](../m11-h2-noshrink-active-episode-30turn-webshop-1.5b-2gpu-20260920/NOTES.md) differs only in the episode
coefficient; it isolates episode supervision at this longer budget.

| Setting | This arm |
|---|---:|
| Maximum actions per episode, training / validation | **30 / 30** |
| Prompt-history turns / future-progress horizon | 2 / 2 |
| Usable history/current/future baseline credibility lambda_k | 1 |
| Context-statistics weight | 1 |
| History / future coefficients | 1 / 1 |
| Episode coefficient | 0 |
| Original-edge coefficient | 0 |
| Training steps / rollouts per task | 150 / 8 |

An episode still stops when the environment reports done; 30 is a ceiling.
The two-step future endpoint clips at the actual episode end. No extra future
steps are generated for an episode that has already ended. Neither H2 nor
prompt history becomes 30.

## Method

Let C_s[q] be the existing kernel-weighted contextual baseline, excluding the
entire query trajectory, for target q. Usable readouts take **B_s[q]=C_s[q]**
with lambda_k=1, including exact groups with one peer. Kappa=2 is recorded but
cannot change this fixed weight. Existing contextual task fallback, unsupported
rows and fixed 10/0 terminal potentials retain their conventions. Features retain
frozen hidden states plus the accumulated context-statistics vector.

\[
H_{i,t}=Y_{i,t}-C_{i,t}[Y],\qquad
Z_{i,s}=\gamma^{T_i-s}R_i,\qquad V_{i,s}=C_{i,s}[Z],\quad\gamma=0.95,
\]
\[
F_{i,t}=\operatorname{z}_{\mathrm{task}}
\left(V_{i,\min(t+2,T_i)}-V_{i,t}\right),
\]
\[
A_{i,t,\ell}=M_{i,t,\ell}
\left[H_{i,t}+F_{i,t}+0\,E_{i,t}\right].
\]

Y is the existing penalized discounted return; Z is the unpenalized potential
label. Current and future potentials use their respective observation groups.
F is standardized per task. WebShop's mean_norm path applies no additional
normalization to H+F or to the final sum.

The episode channel E is the existing helper's mean-centered response reward:
S_it=R_i-0.1*1[invalid action at t], E_it=S_it-mean_task_turn_rows(S).
It preserves the 10/0 outcome, per-turn invalid-action penalty and turn-row
weighting (not one vote per trajectory). KL remains a separate loss.
Episode weight 0 keeps E diagnostic-only; it does not contribute to the actor advantage. Frozen features and advantage estimates stay detached;
nonzero applied channels determine the actor's PPO policy gradient.

## Turn-budget implementation and comparison scope

`max_steps=30` becomes `env.max_steps=30` in the generated Hydra command.
Both training and validation call the same trajectory collector, whose rollout
loop iterates `range(self.config.env.max_steps)` and stops early when all
trajectories are done. The existing estimator supports variable episode lengths;
no estimator or rollout-loop modification is needed for this variant.

This matches the turn limit in the released
[HGPO WebShop launcher](https://github.com/langfengQ/verl-agent/blob/20bd331bdbc9026a5668e11362178e10ab7400c8/recipe/hgpo/run_qwen2.5_1.5b_webshop_train.sh).
All other M11 settings retain the paired 15-turn protocol: two-turn history,
1,000-item catalogue, original item-option scorer, mean_norm, and a fresh
150-step 1.5B run. This does not reproduce HGPO's K=4 setting. The generated
reference metadata explicitly records the deviation from the G2PO 15-turn limit.
Use a common source revision for paired training; extending the budget may
change trajectories, reward, support and resource use.

## Preparation, logging and launch

The generated `config.json` and local gitignored `run.sh` contain the exact
command and environment. Once scheduled on available GPUs, the prepared launch is:

```bash
bash experiments/m11-h2-noshrink-30turn-webshop-1.5b-2gpu-20260920/run.sh
```

No queue/controller entry was created. Existing history/current/future baseline,
support, lambda, raw progress and applied-component diagnostics remain available.
Raw episode advantage is logged in both variants; its applied contribution is
zero when episode weight is zero. The full actor identity check includes the
weighted episode term.

Files: [configuration](config.json), [15-turn delta](config-diff-from-15turn.json),
[episode-on/off delta](config-diff-from-30turn-peer.json),
[preparation status](PREPARED.json), [validation](VALIDATION.json),
[source hashes](prepared-source-sha256.json).

CPU validation passed: 39 future-progress tests, exact ablation-delta guards,
six scoped arm/documentation guards, generated Hydra/environment checks, shell
syntax and a 30-turn H2 endpoint check across the previous cutoff. Each paired
15-turn config changes only max_steps and experiment ID; the 30-turn pair changes
only episode weight and experiment ID. No GPU execution or training is claimed.

The full repository portability guard remains unclean because it scans
provisioned caches and generated local scripts for absolute paths. This includes
pre-existing files and the generated local launchers; see VALIDATION.json.
