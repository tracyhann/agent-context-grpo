# Future-progress hidden-only compatibility fix

The reported failure is reproducible on adb HEAD `8c39f4c`: composing the generic
`no-context-vector` delta (`ccpo_phi=hidden`) with H1 or H2 future progress raises
`Future progress requires frozen history-plus-context features` before estimation.

The separate prepared `future-progress-h2-no-context-vector` arm in this working
tree instead changes `ccpo_ctx_w=0`, leaving `ccpo_phi=hidden+ctx`. That arm already
passed the guard, but its registry entry was not in HEAD. Local preparation and
committed availability must not be conflated.

## Fix

- `ccpo/future_progress.py` accepts both frozen-reference modes, `hidden` and
  `hidden+ctx`. Their hidden-only forms (`hidden`, or `hidden+ctx` with `ctx_w=0`)
  produce identical history/current/future features, baselines and advantages.
- `phi_feats` remains mandatory. Shape, finiteness, metadata and duplicate-padding
  checks remain in force. This is not a relaxation of the FPH2 padding checks.
- `scripts/exp_run.py` checks the feature mode and horizon before preparing files
  or starting model probes/training. Programmatic `build_command` also validates.
  This guard covers this local launcher; remote schedulers must apply the patch
  before submission or use their own configuration preflight.
- The dedicated prepared H2-NOCTX configuration retains its original one-setting
  `ccpo_ctx_w=0` delta. No experiment configuration or running process was changed.

## Validation and delivery

- Reproduced the original hidden-only failure by executing the committed HEAD
  estimator on a CPU fixture.
- **30 future-progress tests passed** in the ALFWorld environment, including all
  existing strict nonfinite/padding tests and the H2 ablations.
- **4 compatibility tests passed** in the WebShop environment.
- Generic NOCTX delta + H1/H2 passes; equivalence includes 1536-dimensional
  features, context-stat perturbation invariance, baselines and endpoint credit.
- The actual trainer `compute_advantage` path is tested for both horizons and
  both `mean_norm` / `mean_std_norm`; returned advantages are finite and detached.
- Unsupported feature modes fail before the launcher's first subprocess or
  experiment-directory creation. Ablation registry guards pass.
- [fix.patch](fix.patch) was applied to copies of the relevant files from HEAD;
  the resulting files exactly matched the tested working tree. It contains the
  runtime fix, launcher checks, method documentation and regression test, not the
  separate uncommitted H2 registry/experiment additions.

**Delivery status: working tree only; not committed or pushed.** Applying or
committing this patch is required before relying on the fix in a HEAD checkout.
Do not reapply it to this already-patched working tree.

The two changed source pins for the already-authorized, running local M10-NOCTX
controller were refreshed only after validation, with the old manifest backed up.
No process was restarted and no GPU job was launched. Historical completed-run
and preparation-time source manifests remain historical records.

Evidence: [VALIDATION.json](VALIDATION.json), [ALFWorld tests](tests-alfworld.log),
[WebShop tests](tests-webshop.log), [registry guards](ablation-guards.log),
[controller pin refresh](controller-pin-refresh.json).
