#!/bin/bash
# Wait for g2po-harness to reach its step-5 checkpoint, pause it, then launch the
# CORRECTED CCPO.
#
# THE DESIGN ERROR (H-AK / root-cause note): ACG_CCPO_GATE=global uses context
# conditioning to REPLACE state grouping -- one bucket per task, phi expected to recover
# structure. phi is inert (r=0.011), so the kernel concentrates on arbitrary neighbours.
#
# THE FIX: context REFINES the partition instead of replacing it.
#   ccpo_gate=hard   -> bucket = (task, anchor), the partition that works
#   anchor_aff=1     -> anchor carries the admissible-action set: 29.1% of within-node
#                       value variance, 5x phi's 6.1% ICC
#   obs_repair=1     -> failures no longer collapse to one shared anchor
# CCPO's estimator (phi-weighted leave-one-out) then operates WITHIN that refined
# partition rather than in place of it.
#
# ccpo_shrink=one is REQUIRED here, and it is not cosmetic. Under the hard gate the
# empirical-Bayes lambda collapses to ~0.011 because tau^2 measures 0.00e+00: the
# estimator finds no between-bucket variance for context to explain and shrinks the term
# away. With shrink=eb this arm would put 98.9% of its weight on the plain
# observation-only baseline and would NOT be a context-conditioned method at all -- it
# would be g2po-affonly with a vestigial 1% term. `one` forces lambda=1 so the question
# "does context conditioning help on a GOOD partition?" can actually be asked.
#
# PRIOR, on record before the result: H-Q measured phi-weighting inside a bucket at 2-10
# R^2 points WORSE than uniform, and the refinement makes buckets smaller still
# (45,978 vs 17,607), so phi has less room, not more. Expect harm.
set -u
D=/workspace/experiments/g2po-harness-20260910/outputs
# wait for the step-5 checkpoint (save_freq=5) so the pause is resumable
for i in $(seq 1 90); do
  # gate on the COMPLETION marker, not the directory: the directory appears when the
  # save STARTS, so gating on it sent SIGTERM mid-save and left global_step_5 without
  # its HF export, data.pt and marker (recorded in g2po-harness NOTES).
  [ "$(cat $D/checkpoints/latest_checkpointed_iteration.txt 2>/dev/null)" = "5" ] && break
  kill -0 "$(cat $D/train.pid 2>/dev/null)" 2>/dev/null || break
  sleep 20
done
echo "checkpoint state: $(ls $D/checkpoints 2>/dev/null | tr '\n' ' ')"
P=$(cat $D/train.pid 2>/dev/null)
if kill -0 "$P" 2>/dev/null; then
  echo "pausing g2po-harness (pid $P)"
  kill -TERM "$P"
  for j in $(seq 1 20); do kill -0 "$P" 2>/dev/null || break; sleep 3; done
  # the trainer IGNORES SIGTERM (two recorded occurrences): escalate, and take the
  # Ray tree with it or the raylet and workers are orphaned holding GPU memory.
  if kill -0 "$P" 2>/dev/null; then
    echo "SIGTERM ignored; SIGKILL trainer + ray tree"
    KIDS=$(pgrep -P "$P"); RL=$(pgrep -P "$P" -f raylet)
    kill -9 "$P" $KIDS $( [ -n "$RL" ] && pgrep -P "$RL" ) 2>/dev/null
  fi
fi
sleep 90
echo "host RAM after pause: $(free -g | awk '/^Mem:/{print $7}')G"
cd /workspace
rm -rf experiments/ccpo-refined-20260910
python3 scripts/exp_run.py --name ccpo-refined --arm ccpo \
  --set gpus=0,1,2,3 \
  --set ccpo_gate=hard --set anchor_aff=1 --set obs_repair=1 \
  --set ccpo_phi=hidden+ctx --set ccpo_edge_w=1.0 --set ccpo_rho=0.59 \
  --set ccpo_tau=0.15 --set ccpo_target=nextnode --set ccpo_shrink=one \
  --set ccpo_whiten=3 --set compact_budget=0 --set total_epochs=100 \
  --set early_stop_min_steps=40 --set early_stop_patience=8 2>&1 | tail -3
echo "LAUNCHED ccpo-refined-20260910: context REFINES the partition instead of replacing it"
