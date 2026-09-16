# M3 · CCPO-ATTNCRED-CTXADV · ALFWorld · Qwen2.5-1.5B

Launched on 2026-09-16 on GPUs 0,1, concurrently with M5 WebShop on GPUs 2,3.
The user originally called this "M4 on ALFWorld", then confirmed proceeding after clarification that the repository calls the ALFWorld context-advantage-only method M3. The requested benchmark is ALFWorld.

Canonical method: episode advantage weight 0, binary return-to-go step target, context-conditioned step credit with the edge term retained. Seed 0; 150 training steps; evaluate/save every 5; preserve best, step-100, and final checkpoints.

Hardware/runtime adjustments:
- Exactly two A100s, as requested, overriding the repository's >=4-GPU recommendation.
- Validation batch 32 instead of the launcher's default 64: 128 total validation rows are processed in four chunks. This changes validation concurrency and random sampling batches; the training batch remains 16 tasks x 8 rollouts.
- Ray CPU slots 24 instead of 64, unused worker prestarts disabled, GCS thread pools bounded, and a private Ray cluster and compilation cache.

Reason: the existing M5 Ray version uses about 20,000 of the container's 25,000 process/thread slots. Smaller validation concurrency leaves room for both requested runs. All estimator, optimization, training-batch, and evaluation-temperature settings are retained.

An initial startup that did not yet include the intended Ray limits was stopped before any training update. Its configuration and logs are preserved under outputs/startup-before-ray-limits. The current config.json/run.sh describe the active launch.

Watch ccpo/live_frac, ccpo/effect_rel, actor gradient norm, and held-out success; no episode term can cover uncredited rows.

A subsequent startup hit the cgroup process limit during Gloo initialization, before any training update; its records are in outputs/startup-thread-limit. To recover, 48 confirmed idle, prestarted Ray workers belonging to M5 were terminated, reclaiming 2,450 threads. M5's active actors and driver were left running and its optimizer progress continued. M3 was then relaunched with the same configuration and seed.

Startup verified: optimizer metrics are finite, ccpo/live_frac=1.0, and the driver remains running. At verification, logged step=1; PID=36490. Full setup verification is in .local/two-run-startup-verified.json.
