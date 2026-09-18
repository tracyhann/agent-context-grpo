# Future-progress validation and offline comparison

**No new training was launched or queued.** All checks and replays below used CPU. These are measurements on already saved M8/M9 rollouts, not validation returns for the new method.

## Method checks

- 12 focused tests passed in each benchmark Python environment (24 executions).
- One-step uniform/self-inclusive reference mode reproduces M3/M5 edge credit and combined pre-normalization advantage to float32 tolerance.
- Context mode passes destination-group sharing, whole-trajectory exclusion, unpenalized value labels, terminal sentinel, two-step telescoping, chronology/padding, invalid configuration, actual PPO episode-gradient isolation, both normalization paths, and complete numeric snapshot checks.
- Existing OUTLOOK (15), fixed-anchor (11), and episode-isolation (2) suites pass. WebShop protocol checks and original G2PO port fidelity checks pass.
- The G2PO test that requires an externally supplied pre-revision core file was skipped because that fixture is not configured; all active port checks passed.
- Registry/config-key/overlay/trainer-wiring/documentation guards pass. The broad pre-existing vendored-path scan was not rerun; earlier failures came from generated caches and historical files.
- Prepared M10/M11 configs retain every shared M3/M5 setting except the experiment name and original edge coefficient (1 to 0). The new future-progress coefficient is 1. Horizon 2 configs are separate optional ablations.

## Real saved-batch comparison

Pearson correlations use exactly the same rows for the old fixed-anchor gain and new progress term. The mask is the intersection of their supported rows. The nonterminal comparison excludes rows whose original two-step future window reaches the recorded terminal endpoint; the same mask is used for h=1 and h=2. Edge credit is recomputed from the original M5 formula and checked against the saved diagnostic. Historical credit is also checked unchanged.

| Source batch | Horizon | Common rows | Old gain vs edge | New progress vs edge | Old nonterminal | New nonterminal |
|---|---:|---:|---:|---:|---:|---:|
| alfworld step 7 | 1 | 5558 | 0.0415 | 0.6629 | 0.0081 | 0.5360 |
| alfworld step 7 | 2 | 5558 | 0.0415 | 0.3875 | 0.0081 | 0.2724 |
| webshop step 14 | 1 | 942 | 0.1178 | 0.6865 | 0.0278 | 0.1274 |
| webshop step 14 | 2 | 942 | 0.1178 | 0.4838 | 0.0278 | 0.0787 |

The one-step variant is closer to M5 on these two batches than the two-step variant. WebShop's nonterminal correlation remains modest (0.1274); much of the full-batch agreement is near terminal windows. One batch per benchmark establishes a concrete implementation check, not a stable estimate across training or evidence of improved success rates.

## Reproduction and artifacts

Run `scripts/analyse_future_progress.py SNAPSHOT --source-config CONFIG --output DIRECTORY` with the benchmark Python environment. It runs both horizons, preserves source hashes, and writes CSV/NPZ/full scalar diagnostics under each `h1/` and `h2/` subdirectory. It does not start Ray, an environment, inference or an optimizer. The synthetic log step 1 in those replay folders is an artifact index; the actual source training step is recorded in `comparison.json`.

- [ALFWorld source/measurements](alfworld-step7/comparison.json)
- [WebShop source/measurements](webshop-step14/comparison.json)
- [Method and math](../../ccpo/FUTURE_PROGRESS.md)

Plots rendered from replay diagnostics are offline visual checks, not learning curves. Prepared runs have no metrics file, training PID, checkpoint, or controller.
