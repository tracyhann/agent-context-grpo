#!/bin/bash
# Shared launch environment for every baseline run on this host.
#
# Each variable below is a DEVIATION from the upstream scripts, forced by this machine.
# They are collected here so no individual run script silently differs from another.
#
#  1. VLLM_ATTENTION_BACKEND=TRITON_ATTN   upstream uses XFORMERS, which has no sm_120
#     (Blackwell) kernel on these RTX PRO 6000 cards.
#  2. The flash-attn stub in docker/fa_stub is kept OFF PYTHONPATH: real flash-attn 2.8.3
#     is installed and its unpad_input is pure PyTorch, so use_remove_padding=True works.
#  3. HOME=/home/claude              `env -i` clears it and /root is not writable here.
#  4. RAY_* thread caps + ray_init.num_cpus=64
#     Ray otherwise sizes itself from nproc (256) and prestarts a worker per CPU. This
#     container allows 8192 pids; 128 train env actors + GPU workers already use ~7.5k.
#  5. VAL_BATCH=128 (upstream 128)   RESTORED. It was lowered to 64 because 128 train +
#     128 val ray actors exceeded the pid ceiling -- but the 128 train actors came from
#     passing env.rollout.n=8 to a val_only run, and validation envs are ALWAYS built
#     with group_n=1 (env_manager.py:813/850), so those actors were never used. The eval
#     scripts now pass env.rollout.n=1 and the pressure is gone.
#     The old note claimed halving this "changes evaluation batching only, never the
#     objective". That was wrong. ALFWorld hands games out PER WORKER -- each worker
#     shuffles its own copy of the pool and reset() takes the next one -- so 128 workers
#     playing one game each and 64 workers playing two each are different samples, not
#     two chunks of one. That is where the repeat-weighting came from: a 140-episode
#     pass covering only 92 distinct games, counting some up to 4 times.
#  6. GPU_MEM_UTIL=0.3 (upstream 0.6)  upstream's 0.6 assumes the actor is offloaded or
#     sharded across a bigger tensor-parallel group. At 0.6 a 1.5B reference allocated
#     ~99 GB on a 96 GB card and died inside vLLM's cumem wake_up.
#  7. default_local_dir is always passed EXPLICITLY. Several upstream scripts use a
#     relative path and cd into their own tree, which silently writes checkpoints inside
#     the baselines checkout.
#  8. Logging is console-only; verl's console backend prints inside the ray actor, so
#     metrics land in the RAY WORKER log, not the driver's. logging/mirror_metrics.sh
#     copies them out and parses them to metrics.jsonl.
set -u
REPO_ROOT=${REPO_ROOT:-/workspace}
VENV=${VENV:-$REPO_ROOT/.venv/bin/python3}
ALFWORLD_DATA=${ALFWORLD_DATA:-$REPO_ROOT/alfworld_data}
DATA_DIR=${DATA_DIR:-$REPO_ROOT/envdata/verl_data/text}
HF_HOME=${HF_HOME:-$REPO_ROOT/hf}
VAL_BATCH=${VAL_BATCH:-128}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.3}
RAY_CPUS=${RAY_CPUS:-64}

# Assemble the `env -i` prefix used by every run. RAY_TMPDIR must be unique per run so
# concurrent runs never share a ray session.
build_env() {   # $1 gpus  $2 ray_tmpdir  $3 tree(PYTHONPATH)
  echo env -i \
    ALFWORLD_DATA="$ALFWORLD_DATA" \
    CUDA_VISIBLE_DEVICES="$1" \
    HF_HOME="$HF_HOME" \
    HOME=/home/claude \
    MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    PATH=/usr/local/cuda/bin:/usr/bin:/bin \
    PYTHONPATH="$3" \
    RAYON_NUM_THREADS=1 \
    RAY_TMPDIR="$2" \
    RAY_event_stats=0 RAY_num_grpc_internal_threads=1 RAY_num_prestart_python_workers=4 \
    RAY_num_server_call_thread=1 RAY_object_manager_rpc_threads_num=1 \
    RAY_start_python_gc_manager_thread=0 \
    TOKENIZERS_PARALLELISM=false \
    TORCHINDUCTOR_CACHE_DIR="$REPO_ROOT/.cache/inductor" \
    TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas \
    VERL_ATTN_IMPL=flash_attention_2 \
    VLLM_ATTENTION_BACKEND=TRITON_ATTN \
    VLLM_CACHE_ROOT="$REPO_ROOT/.cache/vllm" \
    VLLM_USE_FLASHINFER_SAMPLER=0
}
