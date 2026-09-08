#!/bin/bash
# Re-score saved checkpoints on ONE COMMON validation draw.
#
# Why: best-checkpoint selection takes the max of a noisy series, so it is biased
# upward by ~1.5 sd (H-Z). ccpo-global-ext reported mean 79.69 / sd 4.97 over nine
# evaluations and best.json reads 85.9 -- almost exactly the expected maximum of nine
# draws. This asks whether step105-best is actually a better policy than step100-last,
# or just the luckiest draw.
#
# align_val_on_resume=0 is REQUIRED. The alignment burn advances the draw by
# global_steps/test_freq resets, so checkpoints saved at different steps would each be
# scored on a different draw -- the exact confound this exists to remove.
set -u
cd /workspace
declare -a ARMS=(
  "evalck-g100last:/workspace/experiments/ccpo-global-20260907/outputs/checkpoints/step100-last"
  "evalck-g100best:/workspace/experiments/ccpo-global-20260907/outputs/checkpoints/step100-best"
  "evalck-x105best:/workspace/experiments/ccpo-global-ext-20260908/outputs/checkpoints/step105-best"
  "evalck-x145last:/workspace/experiments/ccpo-global-ext-20260908/outputs/checkpoints/step145-last"
)
for entry in "${ARMS[@]}"; do
  name="${entry%%:*}"; ckpt="${entry#*:}"
  echo "=== $name  <- $ckpt"
  python3 scripts/exp_run.py --name "$name" --arm ccpo --no-plot \
    --set gpus=4,5 --set val_only=1 --set align_val_on_resume=0 \
    --set resume_from="$ckpt" --set total_epochs=1 \
    --set ccpo_gate=global --set ccpo_phi=hidden+ctx --set ccpo_edge_w=1.0 \
    --set ccpo_rho=0.59 --set ccpo_tau=0.15 --set ccpo_target=nextnode \
    --set compact_budget=0 --set compact_stall=0 2>&1 | tail -3
  d="/workspace/experiments/${name}-$(date +%Y%m%d)"
  for i in $(seq 1 240); do
    [ -f "$d/outputs/train.pid" ] || { sleep 5; continue; }
    kill -0 "$(cat $d/outputs/train.pid)" 2>/dev/null || break
    sleep 15
  done
  echo "--- $name done; result:"
  python3 -c "
import json,glob
f='$d/outputs/metrics.jsonl'
try:
    rs=[json.loads(l) for l in open(f) if l.strip()]
    v=[r for r in rs if 'val/success_rate' in r]
    print('   val/success_rate =', v[-1]['val/success_rate'] if v else 'NONE LOGGED')
except Exception as e: print('   no metrics:', e)"
done
echo "ALL EVAL RUNS COMPLETE"
