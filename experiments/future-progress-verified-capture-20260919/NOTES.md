# Verified reference features for future progress

Implemented in the actual local trainer/workers on 2026-09-19, with matching tracked overlays. See [implementation](../../ccpo/phi_capture.py), [regression tests](../../tests/test_phi_capture.py), and [validation evidence](VALIDATION.json).

## What the failure establishes

The supplied FPH2-WS traceback reports **finite duplicate-feature disagreement**, not the NaN/Inf branch: 9 rows exceeded tolerance, with a worst-coordinate difference of 1.75 versus an allowed 0.123. It does not distinguish numerical dependence on forward packing from a row-ordering error. A successful 4-GPU step and a failed 8-GPU step on different batches do not establish that 8 GPUs are necessary to reproduce it.

Training copies come from `adjust_batch`, whose text-batch divisor is `lcm(W*m_ref, W*m_rollout, W*m_actor)`. Generation-time `pad_dataproto_to_divisor` copies are removed after generation. In particular, 1598 unique rows need 2 additional rows to reach 1600 for both W=4 and W=8; total training padding also depends on the configured micro-batch sizes.

## Implemented repair

1. Assign a tensor source-row ID and retain the original row count **before** training padding. These survive `adjust_batch` and length balancing.
2. Before the reference RPC, validate complete source coverage and verify that all copies with the same ID have identical model inputs. Retain one row per source in the balanced batch's first-occurrence order.
3. Pad only that canonical reference request to the reference worker count. This needs at most W-1 transport copies. They still participate in the forward as required by dispatch; only the original canonical outputs are retained.
4. At each actual micro-batch forward, validate the last-prompt-token index against padded/packed token geometry. Compute SHA-256 over that row's input IDs, attention mask, position IDs, responses, and source ID. Concatenate the hash bytes with the captured hidden state in **one float32 tensor before inverse permutation or distributed gathering**.
5. On the driver, verify every returned fingerprint against the independently computed request fingerprints. Reject missing features, malformed coverage, NaN/Inf, and identity mismatches. Finite differences in transport copies are measured separately; they are never averaged into the retained source feature.
6. Restore both reference log probabilities and features to the existing training batch using the source mapping. Every training copy now receives exactly the same retained feature. The existing future-progress consistency assertion remains enabled, with its existing tolerance.

Feature-validation errors on a worker are deferred until its reference micro-batch forwards finish, so a rank-local validation failure does not strand the other ranks in their next FSDP forward collective. The failing call still aborts before advantage computation or any optimizer update. Model-forward failures themselves retain the runtime's existing failure handling.

This is automatic for CCPO arms with `ACG_CCPO_PROGRESS_HORIZON > 0`, including horizons 1 and 2. It requires a standalone frozen reference, text inputs, and sequence parallel size 1, matching the current M10/M11 configurations. Unsupported capture fails rather than silently switching to bag-of-words features. Other arms retain their previous capture path.

## Method and numerical scope

History credit, future-progress math, grouping, shrinkage, episode/edge weights, PPO training padding, and final advantage normalization are unchanged. The reference batch layout changes, and a source's reference log probabilities now come from the same canonical forward as its features. Thus floating-point trajectories need not reproduce an old run bit for bit, even though the estimator and PPO gradients agree exactly when their retained numerical inputs agree.

This fixes the repeated-readout/row-mapping path and makes input correspondence verifiable. It does **not** prove that the remote error was harmless BF16 drift, validate FlashAttention's internal attention computation, or certify historical ALFWorld results. A same-input 8-H200 reproduction is still needed to settle the remote numerical cause. No remote submission or GPU training was performed for this repair.

## Logged metrics

All names have prefix `ccpo/`:

- `phi_verified_capture`: 1 confirms the new driver path actually ran.
- `phi_source_rows`, `phi_training_padding_rows`, `phi_transport_padding_rows`: distinguish experience from both kinds of padding.
- `phi_transport_duplicate_max_diff`, `phi_transport_duplicate_differing_rows`: variation in the independently forwarded transport copies.
- `phi_input_identity_mismatches`, `phi_nonfinite_rows`: zero on successful batches; failures abort instead of reaching training metrics.

The existing `progress_phi_duplicate_*` metrics describe the restored **training** rows and should now show zero feature difference. Their zero value alone does not establish numerical invariance across GPUs; inspect the transport-copy metrics too.

## Validation and deployment

CPU tests exercise the actual actor forward, reference worker, dynamic micro-batch inverse permutation, and verl DP split/collect functions, using a deterministic tiny model with 1/4/8 simulated shards. Packed-token helper operations are represented by CPU equivalents; no real distributed GPU collectives or FlashAttention kernels are exercised. Coverage includes deliberate input/packet swaps, missing rows, NaN/Inf, a 1.75 transport-copy perturbation, short batches, 1598-row batches, hook cleanup, deferred errors, old capture compatibility, and exact estimator/PPO-gradient parity for both normalization modes and horizons.

Results: [58 CPU regressions in the ALFWorld environment](cpu-alfworld-tests.txt); [10 capture regressions in the WebShop environment](cpu-webshop-tests.txt).

The incremental [verified-capture patch](../future-progress-verified-capture.patch) includes the new module, tests, and all three tracked worker/trainer overlays. It is a separate repair after the older [finite-tolerance patch](../future-progress-padded-phi.patch). The older artifact is retained as history.

For a matching remote checkout, review and apply the new patch, then synchronize the overlays:

```bash
git apply --check experiments/future-progress-verified-capture.patch
git apply experiments/future-progress-verified-capture.patch
bash scripts/sync_patches.sh
CUDA_VISIBLE_DEVICES='' PYTHONPATH=tests:verl-agent:. .venv-webshop/bin/python -m unittest test_phi_capture -v
```

The patch is already applied to the local working tree, so do not apply it there again. New worker processes are required to load the change. The existing local M11-NOCTX trainer was not restarted and still uses its earlier imported code. The queued M10-NOCTX run will load the repaired source when it starts; its prepared-input hashes have been refreshed with a backup of the preceding records. No experiment configuration or GPU allocation was changed.
