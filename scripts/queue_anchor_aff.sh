#!/bin/bash
# Post-GiGPO queue: one arm at a time on GPUs 0-3, each only after the previous has
# exited AND host RAM / GPUs 0-3 / disk are free (a second concurrent arm OOMs this host,
# and one reference arm alone uses ~14k of the 20k pids).
#
#  1. g2po-aff      G2PO estimator + obs_repair=1 anchor_aff=1 (our state-grouping fixes)
#  2. g2po-affonly  + anchor_aff=1 only -- fully generic, no borrowed PATTERNS
#
# History: ccpo-refined (done, FAILED @50) and hgpo-ref-4gpu (done, 93.0 @160) ran ahead of
# this. g2po-harness-resume was stopped at step 6 on 2026-09-11 at the user's request, and
# gigpo-ref-20260911 took its GPUs; this queue waits on it.
#
# Names use a fixed --date so the pidfiles waited on are deterministic across midnight.
set -u
cd /workspace
D8=20260911
wait_done() {   # $1 pidfile: wait until it exists and its process has exited
  while [ ! -f "$1" ]; do sleep 60; done
  while kill -0 "$(cat "$1" 2>/dev/null)" 2>/dev/null; do sleep 300; done
  sleep 120
}
wait_ram() {    # wait for >=100G host RAM, GPUs 0-3 genuinely free AND >=80G disk, then return.
  # GPU check added 2026-09-10: earlier that day another tenant grabbed GPUs 0/1/2/4 the
  # moment our run released them, and GPUs 4-5 carry a static footprint (12/24 GB) from
  # an unidentified tenant. RAM alone cannot tell us whether 0-3 are ours to take.
  # (An earlier version of this comment attributed the 4-5 footprint to the HGPO run in
  # another container. Unverified and likely wrong: the footprint predates that handoff
  # and sits near 0% utilization, unlike a live training run.)
  # Up to 6h: waiting is cheap, and an abort drops every remaining arm.
  for i in $(seq 1 288); do
    a=$(free -g | awk '/^Mem:/{print $7}')
    g=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0,1,2,3 | awk '$1>=80000' | wc -l)
    d=$(df -B1G --output=avail /workspace | tail -1 | tr -d ' ')
    if [ "${a:-0}" -ge 100 ] && [ "${g:-0}" -ge 4 ] && [ "${d:-0}" -ge 80 ]; then
      echo "resources ok: host RAM ${a}G, GPUs 0-3 free, disk ${d}G"; return 0
    fi
    echo "waiting: host RAM ${a}G (need 100), GPUs 0-3 free ${g}/4, disk ${d}G (need 80)"; sleep 300
  done
  echo "ABORT: resources never drained in 24h"; exit 1
}
launch() {      # $1 name, rest = exp_run args
  local n=$1; shift
  rm -rf "experiments/${n}-${D8}"
  python3 scripts/exp_run.py --name "$n" --date "$D8" "$@" 2>&1 | tail -2
  echo "LAUNCHED ${n}-${D8}"
}
COMMON=(--arm g2po --set gpus=0,1,2,3 --set total_epochs=100 --set compact_budget=0
        --set early_stop_min_steps=40 --set early_stop_patience=8)

# 2026-09-11 22:27: g2po-aff was killed externally at step 16 (no error, no OOM; a bash
# watcher died with it). It was relaunched by hand as g2po-aff-resume-20260911, resumed
# from its own global_step_15 (which has data.pt, so the dataloader state is intact).
# This queue now waits on THAT arm.
# 2026-09-11 22:33: the resume died in vLLM's sleep() with "Memory usage increased after
# sleeping" -- ANOTHER TENANT had taken GPUs 0-3 (36-38 GB each at 72-90% util) while our
# rollout was measuring its own memory. Nothing of ours was on the GPUs. So this queue now
# WAITS for 0-3 to be genuinely free (>=80 GB each) and relaunches the resume itself.
wait_ram
launch g2po-aff-r2 "${COMMON[@]}" --set obs_repair=1 --set anchor_aff=1 \
  --set resume_from=/workspace/experiments/g2po-aff-20260910/outputs/checkpoints/global_step_15
wait_done experiments/g2po-aff-r2-${D8}/outputs/train.pid
echo "g2po-aff-r2 finished"

wait_ram
launch g2po-affonly "${COMMON[@]}" --set obs_repair=0 --set anchor_aff=1
echo "queue complete: last arm (g2po-affonly) launched"
