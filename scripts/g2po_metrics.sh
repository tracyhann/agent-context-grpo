#!/bin/bash
# Pull the G2PO reference run's metrics out of Ray's per-worker log.
#
# Their tree logs via verl's console backend, which print()s to stdout inside the
# TaskRunner ray actor. Ray does NOT forward that to the driver's stdout -- only the
# tqdm progress bar (stderr) reaches our train.log. The metrics land in
#   /tmp/ray_g2po/ray/session_latest/logs/worker-*.out
# which is also volatile, so this mirrors it into the experiment directory.
set -u
D=/workspace/experiments/g2po-ref-20260909/outputs
SRC=$(grep -rl "^step:[0-9]" /tmp/ray_g2po/ray/session_latest/logs/worker-*.out 2>/dev/null | head -1)
[ -z "${SRC:-}" ] && { echo "no worker log with metrics yet"; exit 0; }
cp -f "$SRC" "$D/worker_metrics.log" 2>/dev/null
python3 /workspace/scripts/parse_g2po_log.py "$D/worker_metrics.log" "$D/metrics.jsonl"
