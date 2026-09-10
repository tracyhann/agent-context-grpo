#!/bin/bash
# HGPO REFERENCE — runs recipe/hgpo from baselines/verl-agent (HGPO's official repo).
# Published: 92.77 in-distribution at 160 iterations, Qwen2.5-1.5B, K=2.
#
# HGPO ships INSIDE verl-agent (the GiGPO repo), not standalone: recipe/hgpo has its own
# ray trainer, env manager, config and run scripts. Source script:
#   baselines/verl-agent/recipe/hgpo/run_qwen2.5_1.5b_alfworld_train.sh
#
# DEVIATIONS, all forced by this box, each with its reason:
#   1. VLLM_ATTENTION_BACKEND XFORMERS -> TRITON_ATTN. No sm_120 xformers kernel.
#   2. docker/fa_stub NOT on PYTHONPATH. Real flash-attn 2.8.3 is installed and its
#      unpad_input is pure PyTorch, so use_remove_padding=True works on Blackwell. The
#      stub shadows it and every call raises.
#   3. HOME=/home/claude. env -i clears it; /root is unwritable here.
#   4. gpu_memory_utilization 0.6 -> 0.3. THEIR 0.6 ASSUMES 2 GPUs AT TP=2 as one
#      tensor-parallel group; run at 0.6 the G2PO reference allocated ~99 GB on a 96 GB
#      card and died in vLLM's cumem wake_up at step 37. Sizes the KV cache, not results.
#   5. tensor_model_parallel_size 2 -> 1. TP shards the model; with 2 GPUs TP=1 gives
#      2 data-parallel ranks. train_batch_size 16 divides 2. Changes gradient
#      accumulation, not the objective.
#   6. logger console only (no wandb). scripts/g2po_metrics.sh-style parsing; verl's
#      console backend prints inside the ray actor, so metrics land in the WORKER log.
#   7. data_preprocess.prepare SKIPPED — it regenerates the parquets and could draw a
#      different validation set, destroying comparability with every number we hold.
#   8. save_freq 40 -> 40 (kept) and default_local_dir set EXPLICITLY, because their
#      default is relative and would write inside the baselines checkout.
#
# UNCHANGED (everything that affects the objective): adv_estimator hgpo, weight_type
# length, length_weight_alpha 1.0, base_group False, mode mean_std_norm, lr 1e-6,
# kl_loss_coef 0.01 low_var_kl, gamma 0.95, invalid-action penalty 0.1, max_steps 50,
# group 8, train_batch 16, history_length 2, max_prompt_length 4096, truncation left,
# micro batch 16, val temperature 0.4, test_freq 5, total_epochs 160.
set -u
D=/workspace/experiments/hgpo-ref-20260910
GPUS="${GPUS:-4,5}"
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
  RAY_TMPDIR=/tmp/ray_hgpo \
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
    data.val_batch_size=128 \
    data.max_prompt_length=4096 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='left' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-1.5B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
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
    trainer.critic_warmup=0 \
    "trainer.logger=[console]" \
    trainer.project_name=hgpo_reference \
    trainer.experiment_name=hgpo-ref-20260910 \
    trainer.n_gpus_per_node="$NG" \
    trainer.nnodes=1 \
    trainer.save_freq=40 \
    trainer.test_freq=5 \
    trainer.total_epochs=160 \
    trainer.default_local_dir="$D/outputs/checkpoints" \
    trainer.val_only=False \
    trainer.val_before_train=False \
  >> "$D/outputs/train.log" 2>&1
