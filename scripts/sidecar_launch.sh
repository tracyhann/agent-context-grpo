#!/bin/bash
# Launch the 6 vLLM sidecar generation servers (one per training GPU).
# Serves the symlink results/verl_ccpo_alfworld/sidecar_hf, which sidecar_sync.sh
# repoints at the newest global_step_N/actor/huggingface before each restart.
set -u
ACG=/DATA/tracy/agentic-context-grpo
RES=$ACG/experiments/08-27/results/verl_ccpo_alfworld
# No sidecar on host GPU 0: it already hosts the rank-0 worker AND verl's
# TaskRunner driver actor — a sidecar there OOMs compute_log_prob (crash1).
# Ranks 0 and 1 share the GPU-1 engine instead (ACG_SIDECAR_PORTS lists 8101 twice).
HOST_GPUS=(- 1 2 3 6 7)
IMG=vllm/vllm-openai:v0.28.0-cu129

for K in 1 2 3 4 5; do
  docker rm -f "acg_vllm_side$K" >/dev/null 2>&1
  docker run -d --name "acg_vllm_side$K" \
    --gpus "\"device=${HOST_GPUS[$K]}\"" \
    --memory 24g --memory-swap 24g --cpus 8 \
    -p "127.0.0.1:810$K:8000" \
    -e VLLM_ATTENTION_BACKEND=TRITON_ATTN \
    -v /DATA/tracy/hf:/DATA/tracy/hf:ro \
    -v "$ACG":"$ACG" \
    "$IMG" "$RES/sidecar_hf" \
    --served-model-name acg \
    --dtype bfloat16 --max-model-len 2816 \
    --gpu-memory-utilization 0.18 \
    --enable-prefix-caching --max-num-batched-tokens 16384
done
echo "sidecars launched: acg_vllm_side0..5 on host GPUs ${HOST_GPUS[*]} ports 8100-8105"
