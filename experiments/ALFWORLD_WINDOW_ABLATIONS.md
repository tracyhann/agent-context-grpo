# M10 ALFWorld 1.5B: no-shrink, EP0 history/future window ablations

Prepared **2026-09-22** against the [H2 main control](m10-h2-no-credit-shrinkage-alfworld-1.5b-2gpu-20260919/NOTES.md).
All five use Qwen2.5-1.5B-Instruct, 150 steps, seed 0, two GPUs, no credit
shrinkage, episode weight 0, context statistics enabled and whole-trajectory LOO.
**Prepared only; none launched or queued.** The requested windows replace the
parent's two-history/two-future allocation; they are not additional H2 windows.

| Prompt history / future steps | History / future actor weights | Prepared arm |
|---|---|---|
| 4 / 0 | 1 / 0 | [H4 F0](m10-hist4-fut0-noshrink-alfworld-1.5b-2gpu-20260922/NOTES.md) |
| 0 / 4 | 0 / 1 | [H0 F4](m10-hist0-fut4-noshrink-alfworld-1.5b-2gpu-20260922/NOTES.md) |
| 3 / 1 | 1 / 1 | [H3 F1](m10-hist3-fut1-noshrink-alfworld-1.5b-2gpu-20260922/NOTES.md) |
| 1 / 3 | 1 / 1 | [H1 F3](m10-hist1-fut3-noshrink-alfworld-1.5b-2gpu-20260922/NOTES.md) |
| 3 / 3 | 1 / 1 | [H3 F3](m10-hist3-fut3-noshrink-alfworld-1.5b-2gpu-20260922/NOTES.md) |

## Meaning of history, future and zero

History h is the number of **previous observation/action pairs** in the policy
and frozen-reference prompt, in addition to the current observation and normal
action instructions. It is not a return horizon or a coefficient multiplying
historical advantage. At h=0, historical-residual advantage also has actor weight
0, and the prompt contains no previous turns. The task goal remains present.

Future f is the endpoint offset along the sampled trajectory, clipped to actual
termination. At f=0, both `ccpo_progress_horizon=0` and `ccpo_progress_weight=0` are
explicit. The estimator returns zero raw/normalized future credit, zero future
support, and a history-only actor advantage. It does not use a hidden positive
future horizon for this arm. Positive sides have actor weight 1, not h or f.

The accumulated context-statistics vector remains enabled in every arm, including
h=0. Thus history=0 removes explicit past turns and historical-residual actor
credit; it does not remove every summary of the past. Stored rollout memory and
return labels are not truncated to the prompt window. Current and future
potentials keep their respective observation groups, with whole-query-trajectory
exclusion. No extra rollout steps are generated to fill a future window.

## Equations

Let x_t^(h) be the main method's processed frozen reference hidden state under
the h-turn prompt, concatenated with accumulated context statistics. Keep the
three-direction whitening, block normalization, soft exponential kernel with
tau scale 0.15, exact task/observation gate, whole-trajectory LOO, and existing
task-bucket fallback. Let C_t^(h)[q] denote that contextual weighted mean.
All usable readouts have lambda_u=lambda_k=1; kappa 2 remains recorded but does
not shrink these baselines.

\[
H_t^{(h)}=Y_t-C_t^{(h)}[Y],\qquad
Z_t=\gamma^{T-t}R_{\mathrm{episode}},\qquad
V_t^{(h)}=C_t^{(h)}[Z],\quad\gamma=0.95.
\]
\[
F_t^{(h,f)}=\begin{cases}
z_{\mathrm{task}}\!\left(V_{\min(t+f,T)}^{(h)}-V_t^{(h)}\right),& f>0,\\
0,&f=0,
\end{cases}
\]
\[
A_{t,\ell}=M_{t,\ell}\,N_{\mathrm{CC}}\!\left(
\mathbf1[h>0]H_t^{(h)}+\mathbf1[f>0]F_t^{(h,f)}\right).
\]

Y retains the existing invalid-action penalty; Z excludes it. Terminal potential
is success 10 / failure 0. N_CC is ALFWorld's existing per-task mean/sample-std
normalization over the enabled channels' support. There is no added reward sum
or outer gamma multiplier on the future difference. Multi-step differences
telescope over the same potential function. Advantages and features remain
detached; KL stays a separate loss.

The episode reward still labels returns, but the separate episode-advantage and
original-edge actor weights are 0. Disabled H or episode credit remains available
as a raw diagnostic, with applied contribution exactly zero.

## Runtime changes and diagnostics

- The endpoint estimator, launch validator and live/overlay trainer now support
  horizons 3 and 4, as well as existing horizons 1 and 2.
- `(horizon=0, future_weight=0)` explicitly selects the history-only component
  path. It retains canonicalization, verified frozen-feature capture before
  padding, support masks, component identity checks and snapshots. Legacy
  `(horizon=0, future_weight=1)` remains the ordinary non-progress CCPO path.
- Horizon 0 inside the future-progress estimator requires future weight 0;
  endpoints are the current row, window length and future credit are zero, and
  future eligibility is false. `progress_future_active=0` makes this explicit.
  `progress_enabled=1` means that the component logging/estimator path is active.
- ALFWorld's existing zero-history prompt omitted the task goal after the initial
  turn. The new current-only template preserves the goal and current observation
  while omitting previous turns. Initial prompts and positive-history formatting
  retain their existing behavior. Live files and checked-in overlays match.

Logs retain actual horizon/weights, history/current/future baselines and features,
J/n_eff/lambda_k, eligibility, terminal clipping, raw/applied H/F/episode terms and
full actor-sum identity. Horizon-zero future statistics with no eligible samples
are undefined rather than evidence of active future credit.

## Protocol, preparation and interpretation

All five preserve the matched main settings: 16 tasks x 8 rollouts, 50 training
and evaluation turns, 150 steps, evaluation/checkpoints every 5 steps, and
2048/512 prompt/response limits. A four-turn prompt keeps the same token budget;
no new truncation or rollout budget is introduced. GPU IDs `0,1` are configuration
placeholders, not reservations or a concurrent five-run schedule.

```bash
# Prepare all five for a fresh date; does not launch or queue.
python scripts/prepare_alfworld_window_ablations.py --date YYYYMMDD

# Or choose one; --window may be repeated.
python scripts/prepare_alfworld_window_ablations.py --window h0f4 --date YYYYMMDD

# CPU tests of the five prepared arms:
CUDA_VISIBLE_DEVICES='' PYTHONPATH=tests:. .venv/bin/python -m unittest test_alfworld_windows -v
```

Each folder contains config.json, run.sh, NOTES.md, PREPARED.json, the exact main
control diff, preparation command, source hashes, and CPU validation records.
The preparer refuses to overwrite existing experiments. GPU rollouts and optimizer
steps have not been run, so this preparation contains no training results.

The 4/0 and 0/4 arms change both prompt length and the enabled credit channel;
they test the requested allocation, not a coefficient-only intervention. The
3/1, 1/3 and 3/3 arms keep both coefficients at 1 and change only the two window
lengths relative to the main control.


Validation completed on 2026-09-22: **9 window-specific tests + 76 existing
regressions passed (85 total)**, plus registry/math guards, documentation
consistency and all five shell syntax checks. Each experiment includes the
executed logs and VALIDATION.json. Live trainer/environment files match their
checked-in overlays. Training output directories remain empty.
