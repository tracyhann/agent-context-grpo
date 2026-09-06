# gigpo-repro-20260906

**Question.** Does this harness reproduce published GiGPO on ALFWorld with
Qwen2.5-1.5B-Instruct? This is the gating result for the whole project: until it
holds, a CCPO-vs-baseline number measures the harness, not the estimator.

**Arm.** `--arm gigpo`; every other setting is the shared default, which mirrors
`baselines/G2PO/examples/g2po_trainer/run_alfworld.sh` (GiGPO's own
`examples/gigpo_trainer/run_alfworld.sh` agrees with it on `mode=mean_std_norm`,
group 8, response length 512, val T=0.4). `config.json` → `reference_protocol`
lists the matched settings and the hardware-forced deltas: sdpa instead of
flash-attn (no sm_120 build, so `use_remove_padding=False`), Triton attention in
vLLM, 4 GPUs at tp=1 rather than 8 at tp=2.

**Target.** HGPO Table 1 reports GiGPO (K=2) at **90.16 in-distribution / 84.76
out-of-distribution**. We evaluate on `eval_in_distribution` (valid_seen, the
reference default), so 90.16 is the number to approach. Reaching ~0.85+ validates
the harness; landing near the ~0.60 the earlier Qwen3 runs produced would say the
gap is in the harness rather than the estimator.

**Watch.** `plots/progress.png`: held-out success rate first, then
`episode/valid_action_ratio` (format compliance — the first untrained turn already
parses 127/128 valid, 124 admissible) and `response_length/mean` for inflation.
`plots/progress.png` also carries the step-time breakdown.

**Run history.** Launched three times. The first two died on environment limits
(cgroup pid ceiling; `expandable_segments` versus vLLM's memory pool) — see
`experiments/README.md`. The third was killed deliberately at step 1 once the
timing showed `update_actor` dominating with `ppo_micro_batch_size_per_gpu=8`,
four times below the reference 32. This run uses the reference value.

**Observed cadence.** ~3.2 s per rollout turn over 128 environments, so a 50-turn
rollout is ~2.7 min; warm-cache startup ~1 min. The untrained policy already
completes episodes (two successes by turn 11 of step 1).

**Result.** _pending_

**Reading.** _pending_
