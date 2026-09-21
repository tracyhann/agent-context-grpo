# Qwen evaluation acceleration — 2026-09-21

Authorized by the user's request to accelerate the running census. Applied while
ALFWorld unseen was already in progress; completed seen trajectories are retained.

## Active execution

| Suite | Serving | GPUs | Workers | State |
|---|---|---|---:|---|
| ALFWorld seen 140 | original eager BF16 server, port 8018 | 0,1 | 8 | completed: 99/140 = 70.71% |
| ALFWorld unseen 134 | original eager BF16 server, port 8018 | 0,1 | 8 | continues without interruption |
| WebShop 500 | compiled BF16 server with CUDA graphs, port 8019 | 2,3 | 16 | started 2026-09-21 04:41:19 UTC, concurrently with unseen |

The original server used `--enforce-eager`, which disabled torch.compile and
CUDA graphs. The new server omits that flag and uses `--max-num-seqs 16`.
Logs confirm graph capture and compilation succeeded. The two added GPUs were
idle before launch; other jobs and the existing GPU holders were preserved.
The supervisor stops its own optimized server after WebShop finishes.

**The original unseen evaluation remains on the original server.** No live
requests, partial trajectories or completed task results were interrupted or
replayed to perform this acceleration.

## Protocol preserved

The exact Qwen/Qwen3.8-27B revision, BF16 precision (no quantization), native
xhigh thinking, sampling parameters, seed101 task/action seed formula, history2
prompts, 65,536 output cap, 131,072 context limit, 15 WebShop actions, frozen
500-goal/price manifest, 1K catalog, and original scorer are unchanged.
The output cap was not reduced. Compiled kernels and different batching can
change floating-point execution and sampled trajectories; bit-identical output
across serving configurations is not claimed. Serving configuration is recorded
per suite for reproduction.

The throughput diagnostic is separate from benchmark evaluation. It uses fixed
256-token outputs, a different diagnostic seed, and ignores EOS to measure
serving speed. Those diagnostic calls are not in the scored task ledgers.

## Measured speed

| Concurrent requests | Optimized diagnostic aggregate output tokens/sec |
|---:|---:|
| 1 | 47.58 |
| 8 | 312.63 |
| 16 | 585.41 |

The original production server was observed at about 12.1 tokens/sec with one
request and 91.6 tokens/sec with eight. These are production-versus-diagnostic
comparisons, not a controlled speedup measurement or a promised end-to-end ETA.
After launch the new server logged **531–584 tokens/sec with 16 real WebShop
requests**. The first 98 received actions averaged 15.9 seconds/request and
563.9 output tokens; all 98 completed native reasoning, and 81 contained an
explicit action tag. These are early runtime checks, not final success scores.

## Scheduling and duplicate prevention

The existing controller remains alive and updates the shared RESULTS files for
all suites. Its RUN_STATE still identifies unseen as its current sequential
suite; parallel WebShop's process state is in [STATE.json](STATE.json).
When that controller reaches its original WebShop command, the evaluator finds
`outputs/webshop/ACCELERATED_EXECUTION.json` and joins the managed execution.
It does not start new environment workers or repeat tasks. An exclusive lock
protects the owner, and the join checks the protocol, model, full count, unique
planned task IDs and actual task identities before returning success. The
original final census audit still reconciles every received request and turn.

New helper: `scripts/qwen_eval_coordination.py`.
Supervisor: `scripts/run_qwen_parallel_webshop.py`.
The evaluator's ordinary refusal to overwrite output remains in effect for
unmanaged directories. Failed/incomplete managed runs are not called completed.

## Reproduction and provenance

- [PLAN.json](PLAN.json): exact optimized server command and GPU selection.
- [supervisor-command.json](supervisor-command.json): durable supervisor command.
- [STATE.json](STATE.json): exact WebShop command, server/evaluator identities and timestamps.
- [THROUGHPUT.json](THROUGHPUT.json), `throughput-benchmark.py`: serving diagnostic.
- [input-verification.json](input-verification.json): all 889 original inputs checked;
  the evaluator scheduling entrypoint is the sole allowed changed pinned input.
- `source/` and [source-sha256.json](source-sha256.json): code actually present when
  the parallel supervisor launched; original experiment snapshots are retained.
- `validated-source/` and [validated-source-sha256.json](validated-source-sha256.json):
  tested code including a subsequent defensive cleanup guard for absent PID identity.
  Running processes were not reloaded by that guard.
- [VALIDATION.json](VALIDATION.json), `cpu-tests.log`: 11 passing CPU tests covering
  ownership, joining, protocol drift, missing/duplicate/wrong tasks, incomplete
  runs, command preservation, PID identity, native reasoning and census integrity.

To reproduce this serving condition use the commands in PLAN.json and STATE.json
with new output directories and available GPU IDs/ports. Preserve the pinned
model, data, generation settings and task seed formula. Do not restart these
commands in the active output directory.
