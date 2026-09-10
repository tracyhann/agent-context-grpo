#!/bin/bash
# Mirror a reference-codebase run's metrics out of Ray's per-worker log, then plot.
# Generalises scripts/g2po_metrics.sh (G2PO-only) for any tree that logs via verl's
# console backend: those print()s land in <ray_tmp>/ray/session_latest/logs/worker-*.out,
# not in train.log, and /tmp is volatile, so copy them into the experiment directory.
#
# Usage: ref_metrics.sh <exp_dir> <ray_tmpdir>        (one pass)
#        ref_metrics.sh <exp_dir> <ray_tmpdir> <pid>  (loop every 5 min while pid lives)
set -u
E=$1; R=$2; P=${3:-}
once() {
  local src; src=$(grep -rl "^step:[0-9]" "$R"/ray/session_latest/logs/worker-*.out 2>/dev/null | head -1)
  [ -z "${src:-}" ] && return 0
  cp -f "$src" "$E/outputs/worker_metrics.log" 2>/dev/null
  python3 /workspace/scripts/parse_g2po_log.py "$E/outputs/worker_metrics.log" "$E/outputs/metrics.jsonl" >/dev/null
  /workspace/.venv/bin/python3 /workspace/scripts/plot_metrics.py --exp "$E" >/dev/null 2>&1 || true
}
if [ -z "$P" ]; then once; exit 0; fi
while kill -0 "$P" 2>/dev/null; do once; sleep 300; done
once   # final pass after exit
