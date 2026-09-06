#!/bin/bash
# Weight-sync loop for the vLLM sidecars. Watches latest_checkpointed_iteration.txt;
# when a new step with a complete HF export appears: repoint sidecar_hf, restart the
# sidecars, wait for health, then publish {"step": N} to sidecar_state.json (the
# freshness handshake read by sidecar_client in the trainer).
set -u
ACG=/DATA/tracy/agentic-context-grpo
TAG=${1:-verl_ccpo_alfworld}
RES=$ACG/experiments/08-27/results/$TAG
STATE=$RES/sidecar_state.json
ITER=$RES/latest_checkpointed_iteration.txt
LAST=-1

while true; do
  N=$(cat "$ITER" 2>/dev/null || echo -1)
  HF=$RES/global_step_$N/actor/huggingface
  if [ "$N" != "$LAST" ] && [ -f "$HF/config.json" ] && ls "$HF"/*.safetensors >/dev/null 2>&1; then
    # results dir is root-owned (written by the trainer container) -> write via exec
    docker exec acg_persist ln -sfn "$HF" "$RES/sidecar_hf"
    # first sync of a run: the containers do not exist yet (a step-0 start has no
    # checkpoint to serve), so create them; afterwards a restart picks up new weights.
    # docker restart replays a container's ORIGINAL command, so a container created
    # for a different run keeps serving that run's (now archived) path forever.
    # Recreate whenever the served path does not match this run's.
    SERVING=$(docker inspect acg_vllm_side1 --format '{{range .Args}}{{.}} {{end}}' 2>/dev/null | grep -c "$RES/sidecar_hf" || true)
    if ! docker inspect acg_vllm_side1 >/dev/null 2>&1 || [ "${SERVING:-0}" -eq 0 ]; then
      bash "$ACG/experiments/08-27/sidecar_launch.sh" "$TAG" >/dev/null 2>&1
    else
      for K in 1 2 3 4 5; do docker restart "acg_vllm_side$K" >/dev/null 2>&1 & done
      wait
    fi
    OK=0
    for _ in $(seq 1 60); do
      OK=0
      for K in 1 2 3 4 5; do
        curl -s -m 2 "http://127.0.0.1:810$K/v1/models" >/dev/null 2>&1 && OK=$((OK+1))
      done
      [ "$OK" -eq 5 ] && break
      sleep 5
    done
    if [ "$OK" -eq 5 ]; then
      docker exec acg_persist bash -c "echo '{\"step\": $N}' > $STATE"
      echo "$(date +%m%d-%H:%M:%S) sidecars serving step $N"
      LAST=$N
    else
      echo "$(date +%m%d-%H:%M:%S) WARN only $OK/5 sidecars healthy for step $N (state not published)"
    fi
  fi
  sleep 10
done
