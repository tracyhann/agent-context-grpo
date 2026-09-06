#!/bin/bash
# In-container trainer launcher. Same contract as run_exec_ccpo.sh, minus the
# `docker exec` wrapper -- use this when you are ALREADY inside acg_persist.
#
#   run_local_ccpo.sh <ccpo|grpo> [tag] [gpus]
#
# Generation hygiene matches run_exec_ccpo.sh: the trainer runs as its own setsid
# process group (PGID in /tmp/acg_trainer.pgid) so cleanup kills the whole group
# atomically; env workers are plain-named processes that survive `ray stop` and
# orphan ~270 per restart, so they are reaped explicitly.
set -u
EST=${1:-ccpo}; TAG=${2:-verl_${EST}_alfworld}; GPUS=${3:-0,1,2,3,4,5}
ACG=/DATA/tracy/agentic-context-grpo
HERE=$ACG/experiments/08-27

# Arm guard: the GRPO baseline silently ran as CCPO from step 8 to 38 because a
# caller passed the tag but not the estimator. Refuse rather than repeat that.
case "$TAG:$EST" in
  *grpo*:ccpo|*ccpo*:grpo)
    echo "[local] REFUSING: tag '$TAG' and estimator '$EST' disagree" >&2; exit 2 ;;
esac

if [ -f /.dockerenv ] || grep -qa docker /proc/1/cgroup 2>/dev/null; then :; else
  echo "[local] WARNING: this does not look like the container. Inside acg_persist," \
       "use this script; from the host use run_exec_ccpo.sh." >&2
fi

# ---- clean any previous generation ----------------------------------------
[ -f /tmp/acg_trainer.pgid ] && kill -9 -- -"$(cat /tmp/acg_trainer.pgid)" 2>/dev/null
ray stop --force >/dev/null 2>&1; pkill -9 -f main_ppo 2>/dev/null
pkill -9 -f "[A]lfworldWorker" 2>/dev/null; pkill -9 -f "[a]lfworld.*worker" 2>/dev/null
sleep 3; rm -rf /tmp/ray_acg /dev/shm/* /tmp/acg_trainer.pgid 2>/dev/null
n=$(ps -e -o comm= | grep -cE "ray::|raylet|gcs_server|Alfworld" || true)
if [ "$n" -gt 3 ]; then
  echo "[local] $n ray procs survived; second sweep"; ray stop --force >/dev/null 2>&1; sleep 3
  n=$(ps -e -o comm= | grep -cE "ray::|raylet|gcs_server|Alfworld" || true)
fi
echo "[local] residual ray procs: $n"
command -v python >/dev/null || ln -sf "$(command -v python3)" /usr/local/bin/python

# ---- sidecar reachability --------------------------------------------------
# NetworkMode=host means 127.0.0.1:810K in here is the same loopback the sidecar
# containers publish on, so we can CHECK them from inside. Starting them is a
# host operation on purpose -- mounting the docker socket into this container
# would hand it host root. sidecar_sync.sh (host) owns their lifecycle: it
# creates them on first sync, restarts them on each new checkpoint and re-points
# the served path. sidecar_client falls back to HF generation on any failure, so
# a run without them is still correct -- just ~6x slower to generate.
UP=0
for K in 1 2 3 4 5; do
  curl -s -m 2 "http://127.0.0.1:810$K/v1/models" >/dev/null 2>&1 && UP=$((UP+1))
done
if [ "$UP" -eq 5 ]; then
  echo "[local] sidecars: 5/5 up (Triton generation active)"
else
  echo "[local] sidecars: $UP/5 up -- HF fallback for the missing ranks (~6x slower gen)."
  echo "[local]   on the HOST, once per run:  setsid bash $HERE/sidecar_sync.sh $TAG \\"
  echo "[local]                                 >> $HERE/logs/sidecar_sync_$TAG.log 2>&1 &"
fi

L=$HERE/logs/$TAG.log
[ -s "$L" ] && mv "$L" "${L%.log}.gen$(date +%m%d%H%M).log"
setsid bash -c "echo \$\$ > /tmp/acg_trainer.pgid; stdbuf -oL -eL bash $HERE/run_verl_ccpo.sh $GPUS $EST $TAG > $L 2>&1" &
sleep 1
echo "[local] trainer started (PGID $(cat /tmp/acg_trainer.pgid 2>/dev/null)) -> $L"
echo "[local] tail -f $L"
