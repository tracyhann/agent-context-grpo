#!/bin/bash
# Launch the anchor_aff arm as soon as g2po-harness frees its memory.
#
# WHY QUEUED, not concurrent: the running arm holds ~113 GiB and the host has ~107 GB
# free. A second arm needs ~90-105 GiB. Overcommitting invites the OOM killer, which
# already cost 7 hours on the G2PO reference run today.
#
# WHAT IT TESTS (H-AJ): G2PO's anchor is `observation + admissible actions`; ours is
# observation only. Measured on 264k dumped occurrences, the admissible set explains
# 29.1% of within-node variance in V(next) -- 5x the 6.1% ICC of context conditioning.
# This arm turns that refinement on, with their estimator, in our harness.
set -u
P=/workspace/experiments/g2po-harness-20260910/outputs/train.pid
while kill -0 "$(cat $P 2>/dev/null)" 2>/dev/null; do sleep 300; done
sleep 120                                    # let memory actually drain
A=$(free -g | awk '/^Mem:/{print $7}')
echo "g2po-harness finished; host RAM ${A}G"
[ "${A:-0}" -lt 100 ] && { echo "ABORT: only ${A}G free, need ~100G"; exit 1; }
cd /workspace
rm -rf experiments/g2po-aff-20260910
python3 scripts/exp_run.py --name g2po-aff --arm g2po \
  --set gpus=0,1,2,3 --set total_epochs=100 --set compact_budget=0 \
  --set obs_repair=1 --set anchor_aff=1 \
  --set early_stop_min_steps=40 --set early_stop_patience=8 2>&1 | tail -2
echo "LAUNCHED g2po-aff-20260910 (their estimator + our generic anchor refinement)"
