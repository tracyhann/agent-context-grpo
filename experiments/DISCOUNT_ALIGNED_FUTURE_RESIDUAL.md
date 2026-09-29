# Historical + discount-aligned future residual

This reviewer ablation tests whether the future channel helps after removing
the explicit time-discount contribution in `V(t_plus)-V(t)`. It uses
`z_task(gamma^(t_plus-t)*V(t_plus)-V(t))`, standardized only after this subtraction.
The actual clipped endpoint distance is used, including the final one-step window.
Historical credit, contextual values, peer groups and normalization are unchanged.
The residual is not by itself proof of semantic or causal progress.

| Prepared arm | Matching control | H/F/EP |
| --- | --- | --- |
| [M10 ALFWorld](m10-h2-noshrink-discount-aligned-ep0-alfworld-1.5b-2gpu-20260929/NOTES.md) | M10 H2 no shrink, EP0 | 1/1/0 |
| [M11 WebShop](m11-h2-noshrink-discount-aligned-ep1-webshop-1.5b-2gpu-20260929/NOTES.md) | M11 H2 no shrink, EP1 | 1/1/1 |

Both use Qwen2.5-1.5B-Instruct, history 2, future horizon 2, gamma 0.95,
lambda_u=lambda_k=1, task-observation matching, soft contextual weights and
whole-trajectory LOO. These are prepared configurations; no runs were launched.
Each run records the exact control diff, runtime environment, source hashes,
launch command and per-turn decomposition diagnostics.

Reproduce preparation: `python scripts/prepare_discount_aligned_ablations.py --date 20260929`.
Tests: `.venv/bin/python -m unittest discover -s tests -p test_discount_aligned_future.py`.

Validation: 102 related CPU regression tests passed in `.venv`; all 8 focused
tests also passed in `.venv-webshop`. The ablation registry/kernel guards,
launcher shell syntax, exact control diffs, prepared source hashes and active
trainer/overlay equality passed. Each prepared folder contains `VALIDATION.json`.
These checks exercise the actual trainer credit path and masked PPO gradients;
no GPU training or benchmark performance evaluation was run.
