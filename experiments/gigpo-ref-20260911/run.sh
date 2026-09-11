#!/bin/bash
# GiGPO REFERENCE — runs verl-agent's own trainer (baselines/verl-agent is the GiGPO repo).
# Published: 86.7 in-distribution at 150 iterations, Qwen2.5-1.5B (GiGPO paper; also the
# value G2PO's and HGPO's tables report).
# Source script: baselines/verl-agent/examples/gigpo_trainer/run_alfworld.sh
#
# DEVIATIONS, all forced by this box or by project rules, none touching the objective
# (same list as experiments/hgpo-ref-4gpu-20260910/run.sh):
#   1. VLLM_ATTENTION_BACKEND XFORMERS -> TRITON_ATTN. No sm_120 xformers kernel.
#   2. docker/fa_stub NOT on PYTHONPATH (real flash-attn 2.8.3 is installed).
#   3. HOME=/home/claude. env -i clears it; /root is unwritable here.
#   4. gpu_memory_utilization 0.6 -> 0.3. At 0.6 the G2PO reference hit ~99 GB on a 96 GB
#      card and died in vLLM's cumem wake_up. Sizes the KV cache, not results.
#   5. tensor_model_parallel_size 2 -> 1, 4 GPUs (0-3) instead of 2: same GPU count as
#      our arms, the G2PO and the HGPO references. 4 data-parallel ranks; train_batch 16
#      and val_batch 128 divide 4.
#   6. logger console only (no wandb); metrics mirrored from the Ray WORKER log by
#      scripts/ref_metrics.sh.
#   7. data_preprocess.prepare SKIPPED: regenerating the parquets could draw a different
#      validation set. Same test.parquet (128 games) as every run.
#   8. save_freq -1 -> 25 with max_actor_ckpt_to_keep=2. Theirs writes no checkpoints
#      at all; 25 hits both 100 (budget-matched) and 150 (their endpoint), and the trainer
#      rotates so at most 2 weight sets exist. scripts/snapshot_ckpt.sh preserves step 100
#      as step100-budget. default_local_dir set EXPLICITLY (theirs is relative).
#
# UNCHANGED (everything that affects the objective): adv_estimator gigpo,
# step_advantage_w 1.0, mode mean_std_norm, lr 1e-6, kl_loss_coef 0.01 low_var_kl,
# gamma 0.95, invalid-action penalty 0.1, max_steps 50, group 8, train_batch 16,
# history_length 2 (config default; their script does not override it),
# max_prompt_length 2048, truncation error, micro batch 32, val temperature 0.4,
# test_freq 5, total_epochs 150, val_before_train True.
set -u
D=/workspace/experiments/gigpo-ref-20260911
GPUS="${GPUS:-0,1,2,3}"
NG=$(awk -F, '{print NF}' <<<"$GPUS")

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
  RAY_TMPDIR=/tmp/ray_gigpo \
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
  /workspace/.venv/bin/python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gigpo \
    data.train_files=/workspace/envdata/verl_data/text/train.parquet \
    data.val_files=/workspace/envdata/verl_data/text/test.parquet \
    data.train_batch_size=16 \
    data.val_batch_size=128 \
    data.max_prompt_length=2048 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-1.5B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    algorithm.use_kl_in_reward=False \
    algorithm.gamma=0.95 \
    algorithm.gigpo.step_advantage_w=1.0 \
    algorithm.gigpo.mode=mean_std_norm \
    env.env_name=alfworld/AlfredTWEnv \
    env.seed=0 \
    env.max_steps=50 \
    env.rollout.n=8 \
    env.resources_per_worker.num_cpus=0.1 \
    trainer.critic_warmup=0 \
    "trainer.logger=[console]" \
    trainer.project_name=gigpo_reference \
    trainer.experiment_name=gigpo-ref-20260911 \
    trainer.n_gpus_per_node="$NG" \
    trainer.nnodes=1 \
    trainer.save_freq=25 \
    trainer.max_actor_ckpt_to_keep=2 \
    trainer.test_freq=5 \
    trainer.total_epochs=150 \
    trainer.default_local_dir="$D/outputs/checkpoints" \
    trainer.val_before_train=True \
  >> "$D/outputs/train.log" 2>&1
