#!/bin/bash
# HGPO REFERENCE, Qwen2.5-7B, 4 GPUs (0-3), ALFWorld — runs recipe/hgpo from baselines/verl-agent.
# Source script: baselines/verl-agent/recipe/hgpo/run_qwen2.5_7b_alfworld_train.sh
# Launched from the hgpo-ref session's container (pids.max 8192), not the 20k-pid one.
#
# DEVIATIONS from their script, each with its reason:
#   1. VLLM_ATTENTION_BACKEND XFORMERS -> TRITON_ATTN. No sm_120 xformers kernel.
#   2. docker/fa_stub NOT on PYTHONPATH; real flash-attn 2.8.3 is used (as hgpo-ref-4gpu).
#   3. HOME=/home/claude. env -i clears it; /root is unwritable here.
#   4. param_offload / optimizer_offload True -> False. Offloading a 7B actor puts ~90 GB of
#      fp32 master weights + Adam state in HOST RAM, on a host with ~170 GB free and swap
#      full. On 96 GB cards the FSDP-sharded state is ~30 GB/GPU, so it stays on GPU.
#      Memory placement only, not the objective. Fallback if it OOMs: optimizer_offload=True.
#   5. gpu_memory_utilization 0.6 -> 0.3. Their 0.6 assumes the actor is offloaded; with the
#      actor resident, 0.3 (~28 GB: 3.8 GB of TP=4 weights + KV cache) leaves room for it.
#   6. data.val_batch_size 128 -> 64. HGPO builds one ray actor per validation env
#      (env_num = val_batch_size); 128 train + 128 val actors exceed this container's 8192
#      pids. All 128 validation games are still played, in two chunks of 64.
#   7. ray_init.num_cpus=64. Ray otherwise sizes itself from nproc (256) and prestarts a
#      worker per CPU, which also runs into the pid ceiling.
#   8. logger console only (no wandb); metrics land in the ray worker log.
#   9. data_preprocess.prepare SKIPPED so the validation set matches every number we hold.
#  10. save_freq 40 (theirs) with max_actor_ckpt_to_keep=2: a 7B checkpoint with optimizer
#      state is ~90+ GB, so at most two exist. default_local_dir set explicitly.
#
# UNCHANGED (everything that affects the objective): adv_estimator hgpo, weight_type length,
# length_weight_alpha 1.0, base_group False, mode mean_std_norm, lr (their default; the 7B
# script does not set it), kl_loss_coef 0.01 low_var_kl, gamma 0.95, invalid-action penalty
# 0.1, max_steps 50, group 8, train_batch 16, history_length 2, max_prompt_length 4096,
# truncation left, ppo_mini_batch 256, micro batch 8, TP 4, val temperature 0.4,
# test_freq 5, total_epochs 160.
set -u
D=/workspace/experiments/hgpo-7b-ref-20260911
GPUS="${GPUS:-0,1,2,3}"
NG=$(awk -F, '{print NF}' <<<"$GPUS")
mkdir -p "$D/outputs"

cd /workspace/baselines/verl-agent
env -i \
  ALFWORLD_DATA=/workspace/alfworld_data \
  CUDA_VISIBLE_DEVICES="$GPUS" \
  HF_HOME=/workspace/hf \
  HOME=/home/claude \
  MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PATH=/usr/local/cuda/bin:/usr/bin:/bin \
  PYTHONPATH=/workspace/baselines/verl-agent \
  RAYON_NUM_THREADS=1 \
  RAY_TMPDIR=/tmp/ray_hgpo7b \
  RAY_event_stats=0 RAY_num_grpc_internal_threads=1 RAY_num_prestart_python_workers=4 \
  RAY_num_server_call_thread=1 RAY_object_manager_rpc_threads_num=1 \
  RAY_start_python_gc_manager_thread=0 \
  TOKENIZERS_PARALLELISM=false \
  TORCHINDUCTOR_CACHE_DIR=/workspace/.cache/inductor \
  TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas \
  VERL_ATTN_IMPL=flash_attention_2 \
  VLLM_ATTENTION_BACKEND=TRITON_ATTN \
  VLLM_CACHE_ROOT=/workspace/.cache/vllm \
  VLLM_USE_FLASHINFER_SAMPLER=0 \
  /workspace/.venv/bin/python3 -m recipe.hgpo.main_hgpo \
    algorithm.adv_estimator=hgpo \
    algorithm.hgpo.weight_type=length \
    algorithm.hgpo.mode=mean_std_norm \
    algorithm.hgpo.length_weight_alpha=1.0 \
    algorithm.hgpo.base_group=False \
    data.train_files=/workspace/envdata/verl_data/text/train.parquet \
    data.val_files=/workspace/envdata/verl_data/text/test.parquet \
    data.train_batch_size=16 \
    data.val_batch_size=64 \
    data.max_prompt_length=4096 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='left' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-7B-Instruct \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    algorithm.use_kl_in_reward=False \
    algorithm.gamma=0.95 \
    env.env_name=alfworld/AlfredTWEnv \
    env.resources_per_worker.num_cpus=0.1 \
    env.seed=0 \
    env.history_length=2 \
    env.max_steps=50 \
    env.rollout.n=8 \
    ${RAY_INIT_OVERRIDE:-ray_init.num_cpus=64} \
    trainer.critic_warmup=0 \
    "trainer.logger=[console]" \
    trainer.project_name=hgpo_reference \
    trainer.experiment_name=hgpo-7b-ref-20260911 \
    trainer.n_gpus_per_node="$NG" \
    trainer.nnodes=1 \
    trainer.save_freq=40 \
    trainer.max_actor_ckpt_to_keep=2 \
    trainer.test_freq=5 \
    trainer.total_epochs=160 \
    trainer.default_local_dir="$D/outputs/checkpoints" \
    trainer.val_only=False \
    trainer.val_before_train=False \
  >> "$D/outputs/train.log" 2>&1
