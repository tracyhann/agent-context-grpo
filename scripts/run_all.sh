#!/bin/bash
# Run the full baseline matrix, ONE AT A TIME.
#
# Serialized deliberately: a single run uses ~7.5k of this container's 8192 pids and
# ~200 GB of host RAM, so two concurrent runs cannot fit. Each waits for its GPUs to be
# genuinely idle (memory AND utilization) before starting -- a neighbouring tenant on
# this host cycles between phases, and launching into one of their troughs is what killed
# an earlier reference run inside vLLM's sleep().
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
GPUS=${GPUS:-0,1,2,3}
STEPS=${STEPS:-150}
BENCHES=${BENCHES:-"alfworld webshop"}
# 7B only: the 1.5B arms are run in the main workspace, not through this repo.
MATRIX=${MATRIX:-"grpo:7b gigpo:7b hgpo:7b g2po:7b"}

wait_gpus() {   # all requested GPUs idle for 3 consecutive minutes
  local need ok=0
  need=$(awk -F, '{print NF}' <<<"$GPUS")
  while :; do
    local n
    n=$(nvidia-smi --query-gpu=memory.free,utilization.gpu --format=csv,noheader,nounits -i "$GPUS" \
        | awk -F', *' '$1>=80000 && $2<15' | wc -l)
    if [ "$n" -ge "$need" ]; then
      ok=$((ok+1)); [ "$ok" -ge 3 ] && { echo "[queue] GPUs $GPUS idle, launching"; return 0; }
    else ok=0; fi
    sleep 60
  done
}

for bench in $BENCHES; do
  for spec in $MATRIX; do
    m=${spec%%:*}; s=${spec##*:}
    run="${m}-${s}-${bench}-$(date +%Y%m%d)"
    d=/workspace/baseline-repo/runs/$run
    if [ -f "$d/outputs/train.pid" ] && kill -0 "$(cat $d/outputs/train.pid)" 2>/dev/null; then
      echo "[queue] $run already running, skipping"; continue; fi
    wait_gpus
    echo "[queue] starting $run at $(date +%T)"
    RUN_NAME="$run" setsid -f bash "$HERE/train.sh" "$m" "$s" "$bench" "$GPUS" "$STEPS"
    sleep 45
    pid=$(ps -eo pid,cmd | awk '$2=="/workspace/.venv/bin/python3" && /main_(ppo|hgpo)/ {print $1; exit}')
    [ -z "${pid:-}" ] && { echo "[queue] $run FAILED to start"; continue; }
    echo "$pid" > "$d/outputs/train.pid"
    echo "[queue] $run pid $pid"
    nohup bash /workspace/baseline-repo/logging/mirror_metrics.sh "$d" "/tmp/ray_${m}_${s}_${bench}" "$pid" \
      > "$d/outputs/mirror.log" 2>&1 &
    while kill -0 "$pid" 2>/dev/null; do sleep 120; done
    echo "[queue] $run finished at $(date +%T)"
  done
done
echo "[queue] matrix complete $(date +%T)"
