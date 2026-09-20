# Qwen3.8-27B prompt-only alfworld evaluation

Status: **prepared; no real model inference or benchmark score yet**. The weights have been downloaded and the separate serving environment installed. GPUs were busy during setup on 2026-09-20. Synthetic integration-test metrics are not model results.

Model: `Qwen/Qwen3.8-27B`, revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, BF16, no fine-tuning. Method and launch instructions: [qwen_baseline/README.md](../../qwen_baseline/README.md).

This condition uses the existing benchmark's environment prompts/action parser and original reward. It evaluates 128 episodes with a 50-action horizon, a 2048-token prompt cap, 512 new tokens/action, temperature 0.4, top_p 1, top_k -1, recent history length 2, native thinking disabled, and initial evaluation environment seed 1000. Episode identities are fixed independently of CPU worker count. `config.json` contains the complete plan and source fingerprints.

After starting the model server on available GPUs:

```bash
python3 baselines/qwen/run.py alfworld --output experiments/qwen38-27b-alfworld-20260920 --resume
```

Results will be written to `metrics.json` and local per-episode traces. Do not compare a partial `observed_success_rate` as if it were a completed result. Use the recorded task identities to establish paired comparisons with historical runs; matching environment settings alone does not prove identical historical samples.
