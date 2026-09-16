# M5 · CCPO-ATTNCRED-CTXADV-RETURN-WS · Qwen2.5-1.5B

Launched on 2026-09-16 at the user's request, using exactly two A100 GPUs (2,3).
This is the canonical M5 method: episode advantage weight 0, binary return-to-go step target, context-conditioned step credit with the edge term retained.

150 training steps; evaluate/save every 5; preserve best, step-100, and final checkpoints. Seed 0. Existing 1,000-product WebShop protocol and all method/batch settings retained. The requested two-GPU allocation overrides the repository's >=4-GPU recommendation.

Runtime uses .venv-webshop, the restored Java 11 and local Qwen snapshot, a private Ray cluster, and per-run compilation caches. scripts/exp_run.py was updated to isolate concurrent runs; the source diff is saved beside this record.

Watch ccpo/live_frac, ccpo/effect_rel, and the zero-advantage population: this variant intentionally uses WebShop's binary reward, with no episode term.

Concurrent-run housekeeping: 48 confirmed idle, prestarted Ray workers (2,450 threads) were terminated to free process slots for M3. The M5 driver, model workers, and environment actors were untouched, and training continued without a restart. The PID inventory is in outputs/idle-worker-reclamation.json.

Startup verified: optimizer metrics are finite, ccpo/live_frac=1.0, and the driver remains running. At verification, logged step=4; PID=3190. Full setup verification is in .local/two-run-startup-verified.json.
