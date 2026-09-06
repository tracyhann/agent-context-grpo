# gigpo-repro-20260906

**Question.** Does this harness reproduce published GiGPO on ALFWorld with
Qwen2.5-1.5B-Instruct? This is the gating result for the whole project: until it
holds, a CCPO-vs-baseline number measures the harness, not the estimator.

**Arm.** `--arm gigpo`; every other setting is the shared default, which mirrors
`baselines/G2PO/examples/g2po_trainer/run_alfworld.sh`. GiGPO's own
`examples/gigpo_trainer/run_alfworld.sh` agrees with it on the settings that
matter (`mode=mean_std_norm`, group 8, response length 512, val T=0.4).
`config.json` → `reference_protocol` lists what is matched and the
hardware-forced deltas.

**Target.** HGPO Table 1 reports GiGPO (K=2) at **90.16 in-distribution / 84.76
out-of-distribution**. We evaluate on `eval_in_distribution` (valid_seen, the
reference default), so 90.16 is the number to approach. Reaching ~0.85+ validates
the harness; landing near the ~0.60 the earlier Qwen3 runs produced would say the
gap is in the harness rather than the estimator.

**Watch.** `plots/progress.png`: held-out success rate first, then
`episode/valid_action_ratio` (the first untrained turn already parses 127/128
valid, 124 admissible), `response_length/clip_ratio` for truncation, and the
step-time breakdown.

**Observed cadence.** ~3.2 s per rollout turn over 128 environments, so a 50-turn
rollout is ~2.7 min; warm-cache startup ~1 min. The untrained policy already
completes episodes — 12 successes out of 128 in step 1.

**Run history.** Five launches; the first four died on environment limits, each
fixed and documented in `experiments/README.md`:
1. cgroup pid ceiling — 258 env actors at ~115 threads each against 8192 pids.
2. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments` versus vLLM's memory pool.
3. killed deliberately: `ppo_micro_batch_size_per_gpu=8` made `update_actor` the
   long pole, four times below the reference 32.
4. OOM in the backward pass at 32 — the reference can use 32 only because it packs
   sequences; we pad to 2560 tokens.
5. this run: 16 for the update, 32 for the forward-only log-prob passes,
   `gpu_memory_utilization` 0.35.

**Result.** _pending_

**Reading.** _pending_
