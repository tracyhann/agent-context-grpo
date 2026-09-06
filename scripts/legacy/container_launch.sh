#!/bin/bash
# Create (or recreate) the acg_persist trainer container.
#
#   container_launch.sh [--recreate]
#
# NO docker socket is mounted, deliberately. /var/run/docker.sock in a container
# is equivalent to host root: anything inside can `docker run --privileged
# -v /:/host` and take the whole machine, including other users' containers and
# data. The sidecar lifecycle is handled instead by sidecar_sync.sh running on
# the HOST, which creates, restarts and re-points the sidecars by itself -- so
# nothing inside ever needs docker.
#
# Gotchas that cost three attempts:
#   * the image ENTRYPOINT is /bin/bash, so `sleep infinity` as the command
#     yields `bash: infinity: No such file or directory`. Override the
#     entrypoint and pass only `infinity`.
#   * with --entrypoint sleep, passing `sleep infinity` yields `sleep sleep
#     infinity` -> "cannot execute binary file".
#   * --memory-swap must EQUAL --memory: that is what disables swap. Leaving it
#     unset gives the container 2x memory in swap, which is the thing that got
#     us in trouble on this box.
#   * host GPUs 4 and 5 are another project's (ocean). Never add them.
set -eu
NAME=acg_persist
IMAGE=acg-verl:cu128
GPUS='"device=0,1,2,3,6,7"'
ACG=/DATA/tracy/agentic-context-grpo

if docker inspect "$NAME" >/dev/null 2>&1; then
  if [ "${1:-}" != "--recreate" ]; then
    echo "$NAME exists ($(docker inspect -f '{{.State.Status}}' $NAME))."
    echo "Start it with:   docker start $NAME"
    echo "Replace it with: $0 --recreate"
    exit 0
  fi
  echo "recreating $NAME (checking it is idle first)"
  if docker exec "$NAME" pgrep -f '[m]ain_ppo' >/dev/null 2>&1; then
    echo "REFUSING: a trainer is running inside $NAME" >&2; exit 1
  fi
  docker rm -f "$NAME" >/dev/null
fi

docker run -d --name "$NAME" \
  --entrypoint sleep \
  --gpus "$GPUS" \
  --network host --ipc host --shm-size 64g \
  --memory 200g --memory-swap 200g \
  -w /vllm-workspace \
  -e ACG_VERL_BIN=/usr/local/bin \
  -e HF_HUB_OFFLINE=1 \
  -v "$ACG":"$ACG" \
  -v /DATA/tracy/hf:/DATA/tracy/hf \
  "$IMAGE" infinity >/dev/null

sleep 3
docker ps --filter "name=$NAME" --format '{{.Names}}  {{.Status}}'
echo -n "gpus inside: "; docker exec "$NAME" nvidia-smi --query-gpu=index --format=csv,noheader | tr '\n' ' '; echo
docker exec "$NAME" python3 -c "import torch,transformers;print('torch',torch.__version__,'transformers',transformers.__version__)"
