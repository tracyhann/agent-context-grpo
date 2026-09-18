# M3-NOEDGE · ALFWorld · Qwen2.5-1.5B-Instruct

Authorized on 2026-09-17 as the next training run after M3.

This isolates our context-conditioned advantage: `ccpo_ep_w=0`, `ccpo_edge_w=0`, `ccpo_ctx_w=1`, `ccpo_phi=hidden+ctx`, and `ccpo_target=return`.

The control is [M3](../m3-ccpo-attncred-ctxadv-alfworld-1.5b-2gpu-20260916/NOTES.md). The configuration differs only in experiment ID and `ccpo_edge_w` (1 → 0); see [the exact diff](config-diff-from-m3.json).

Start from the same original Qwen2.5-1.5B-Instruct checkpoint with seed 0. Run 150 optimizer steps on GPUs 0–1, with 16 tasks × 8 rollouts, validation batch 32 over 128 examples, evaluation/checkpointing every 5 steps, and early stopping disabled. Retain M3's Ray and attention settings. This is a fresh training run, not a continuation of M3's trained checkpoint.

The detached controller at `.local/m3-noedge-chain-20260917/runner.py` waits for M3's original process to exit, then verifies step 150, finite final validation, and the final/best/step-100 checkpoint shards. It also requires GPUs 0–1 to remain idle across six samples before preflight and checks capacity again before launch.

M3-NOEDGE has priority over the existing WebShop queue. The WebShop controller waits until M3-NOEDGE records its first optimizer step, then resumes waiting for GPUs 2–3 before running M5-NOEDGE → M5-NOCTX. Existing failed WebShop startup records and retry accounting are preserved.

Controller output: `.local/m3-noedge-chain-20260917/chain.log`; status: `.local/m3-noedge-chain-20260917/state.json`. Training outputs will appear under this experiment's `outputs/`. Input fingerprints protect the prepared configuration and training code. Only recognized startup memory failures before any training output may be retried; training failures stop the chain.
