#!/bin/bash
# Post-ccpo-refined queue: one arm at a time on GPUs 0-3, each only after the previous
# has exited AND host RAM has drained (a second concurrent arm OOMs this host).
#
#  0. hgpo-ref-4gpu        HGPO reference (recipe/hgpo, unmodified), 160 iters, GPUs 0-3.
#                          Added 2026-09-10 at the user's request, FIRST after ccpo-refined.
#                          Not exp_run.py (their tree), so metrics are mirrored by
#                          scripts/ref_metrics.sh from the Ray worker log.
#  1. g2po-harness-resume  paused diagnostic, resumed from global_step_5 via EXPLICIT
#                          path (that checkpoint lacks its marker, so auto-resume fails).
#                          G2PO estimator + our plain anchor: estimator vs harness.
#  2. g2po-aff             + obs_repair=1 anchor_aff=1
#  3. g2po-affonly         + anchor_aff=1 only -- fully generic, no borrowed PATTERNS
#  (hgpo-ref removed 2026-09-10: running in another container)
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
wait_ram() {    # wait for >=100G host RAM, GPUs 0-3 genuinely free AND >=80G disk, then return.
  # GPU check added 2026-09-10: earlier that day another tenant grabbed GPUs 0/1/2/4 the
  # moment our run released them, and GPUs 4-5 carry a static footprint (12/24 GB) from
  # an unidentified tenant. RAM alone cannot tell us whether 0-3 are ours to take.
  # (An earlier version of this comment attributed the 4-5 footprint to the HGPO run in
  # another container. Unverified and likely wrong: the footprint predates that handoff
  # and sits near 0% utilization, unlike a live training run.)
  # Up to 6h: waiting is cheap, and an abort drops every remaining arm.
  for i in $(seq 1 72); do
    a=$(free -g | awk '/^Mem:/{print $7}')
    g=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0,1,2,3 | awk '$1>=80000' | wc -l)
    d=$(df -B1G --output=avail /workspace | tail -1 | tr -d ' ')
    if [ "${a:-0}" -ge 100 ] && [ "${g:-0}" -ge 4 ] && [ "${d:-0}" -ge 80 ]; then
      echo "resources ok: host RAM ${a}G, GPUs 0-3 free, disk ${d}G"; return 0
    fi
    echo "waiting: host RAM ${a}G (need 100), GPUs 0-3 free ${g}/4, disk ${d}G (need 80)"; sleep 300
  done
  echo "ABORT: resources never drained in 6h"; exit 1
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

H=experiments/hgpo-ref-4gpu-${D8}
wait_ram
GPUS=0,1,2,3 setsid nohup bash "$H/run.sh" >/dev/null 2>&1 < /dev/null &
echo $! > "$H/outputs/train.pid"
echo "LAUNCHED hgpo-ref-4gpu-${D8} (pid $(cat "$H/outputs/train.pid"))"
setsid nohup bash scripts/ref_metrics.sh "/workspace/$H" /tmp/ray_hgpo4 "$(cat "$H/outputs/train.pid")" \
  > "$H/outputs/ref_metrics.log" 2>&1 < /dev/null &
wait_done "$H/outputs/train.pid"
echo "hgpo-ref-4gpu finished"

wait_ram
launch g2po-harness-resume "${COMMON[@]}" --set obs_repair=0 --set anchor_aff=0 \
  --set resume_from=/workspace/experiments/g2po-harness-${D8}/outputs/checkpoints/global_step_5
wait_done experiments/g2po-harness-resume-${D8}/outputs/train.pid

wait_ram
launch g2po-aff "${COMMON[@]}" --set obs_repair=1 --set anchor_aff=1
wait_done experiments/g2po-aff-${D8}/outputs/train.pid

wait_ram
launch g2po-affonly "${COMMON[@]}" --set obs_repair=0 --set anchor_aff=1
echo "queue complete: last arm (g2po-affonly) launched"
