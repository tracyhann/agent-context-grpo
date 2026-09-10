#!/bin/bash
# Post-ccpo-refined queue: one arm at a time on GPUs 0-3, each only after the previous
# has exited AND host RAM has drained (a second concurrent arm OOMs this host).
#
#  1. g2po-harness-resume  paused diagnostic, resumed from global_step_5 via EXPLICIT
#                          path (that checkpoint lacks its marker, so auto-resume fails).
#                          G2PO estimator + our plain anchor: estimator vs harness.
#  2. g2po-aff             + obs_repair=1 anchor_aff=1
#  3. g2po-affonly         + anchor_aff=1 only -- fully generic, no borrowed PATTERNS
#  4. hgpo-ref             HGPO from baselines/verl-agent, 160 iterations
#
# Names use a fixed --date so the pidfiles waited on are deterministic across midnight.
set -u
cd /workspace
D8=20260910
wait_done() {   # $1 pidfile: wait until it exists and its process has exited
  while [ ! -f "$1" ]; do sleep 60; done
  while kill -0 "$(cat "$1" 2>/dev/null)" 2>/dev/null; do sleep 300; done
  sleep 120
}
wait_ram() {    # wait (up to 2h) for >=100G free; abort the queue rather than skip an arm
  for i in $(seq 1 24); do
    a=$(free -g | awk '/^Mem:/{print $7}')
    [ "${a:-0}" -ge 100 ] && { echo "host RAM ${a}G ok"; return 0; }
    echo "host RAM ${a}G < 100, waiting"; sleep 300
  done
  echo "ABORT: host RAM never drained"; exit 1
}
launch() {      # $1 name, rest = exp_run args
  local n=$1; shift
  rm -rf "experiments/${n}-${D8}"
  python3 scripts/exp_run.py --name "$n" --date "$D8" "$@" 2>&1 | tail -2
  echo "LAUNCHED ${n}-${D8}"
}
COMMON=(--arm g2po --set gpus=0,1,2,3 --set total_epochs=100 --set compact_budget=0
        --set early_stop_min_steps=40 --set early_stop_patience=8)

wait_done experiments/ccpo-refined-${D8}/outputs/train.pid
echo "ccpo-refined finished"

wait_ram
launch g2po-harness-resume "${COMMON[@]}" --set obs_repair=0 --set anchor_aff=0 \
  --set resume_from=/workspace/experiments/g2po-harness-${D8}/outputs/checkpoints/global_step_5
wait_done experiments/g2po-harness-resume-${D8}/outputs/train.pid

wait_ram
launch g2po-aff "${COMMON[@]}" --set obs_repair=1 --set anchor_aff=1
wait_done experiments/g2po-aff-${D8}/outputs/train.pid

wait_ram
launch g2po-affonly "${COMMON[@]}" --set obs_repair=0 --set anchor_aff=1
wait_done experiments/g2po-affonly-${D8}/outputs/train.pid

wait_ram
GPUS=0,1,2,3 bash experiments/hgpo-ref-${D8}/run.sh &
echo $! > experiments/hgpo-ref-${D8}/outputs/train.pid
echo "LAUNCHED hgpo-ref-${D8} (published 92.77 @160 iters)"
