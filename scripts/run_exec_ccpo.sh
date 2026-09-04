#!/bin/bash
# Restart the trainer INSIDE the persistent container via docker exec — bypasses
# the dockerd create/destroy queue entirely. Usage: run_exec_ccpo.sh <ccpo|grpo> <tag>
# Generation hygiene: each trainer runs as its own setsid process group (PGID
# recorded in /tmp/acg_trainer.pgid); cleanup kills the whole group atomically,
# then ray stop --force as belt-and-suspenders, then verifies the container is
# actually clean before launching the next generation.
EST=${1:-ccpo}; TAG=${2:-verl_${EST}_alfworld}; GPUS=${3:-0,1,2,3,4,5}   # container-local indices (all six; ocean released GPUs 0-3 on 09-02)
C=acg_persist
docker exec $C bash -c '
  [ -f /tmp/acg_trainer.pgid ] && kill -9 -- -"$(cat /tmp/acg_trainer.pgid)" 2>/dev/null
  ray stop --force 2>/dev/null; pkill -9 -f main_ppo 2>/dev/null
  # Env workers are plain-named processes that survive ray stop and the PGID kill.
  # Each restart otherwise orphans ~270 of them holding TextWorld state (~50MB each);
  # 2803 accumulated by step 25 and OOM-killed the trainer twice. Reap them here.
  pkill -9 -f "[A]lfworldWorker" 2>/dev/null; pkill -9 -f "[a]lfworld.*worker" 2>/dev/null
  sleep 3; rm -rf /tmp/ray_acg /dev/shm/* /tmp/acg_trainer.pgid 2>/dev/null
  n=$(ps -e -o comm= | grep -cE "ray::|raylet|gcs_server|Alfworld" || true)
  if [ "$n" -gt 3 ]; then
    echo "[exec-clean] $n ray procs survived; second sweep"; ray stop --force 2>/dev/null; sleep 3
    n=$(ps -e -o comm= | grep -cE "ray::|raylet|gcs_server|Alfworld" || true)
  fi
  echo "[exec-clean] residual ray procs: $n"'
docker exec $C bash -c "command -v python >/dev/null || ln -sf \$(command -v python3) /usr/local/bin/python"
L=/DATA/tracy/agentic-context-grpo/experiments/08-27/logs/$TAG.log
[ -s "$L" ] && mv "$L" "${L%.log}.gen$(date +%m%d%H%M).log"
docker exec -d $C setsid bash -c "echo \$\$ > /tmp/acg_trainer.pgid; stdbuf -oL -eL bash /DATA/tracy/agentic-context-grpo/experiments/08-27/run_verl_ccpo.sh $GPUS $EST $TAG \
  > /DATA/tracy/agentic-context-grpo/experiments/08-27/logs/$TAG.log 2>&1"
echo "trainer (re)started inside $C via exec (own process group)"
