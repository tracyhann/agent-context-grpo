#!/bin/bash
# G2PO REFERENCE REPRODUCTION — runs the baselines/G2PO tree itself, not our port.
#
# Question: does G2PO reach its published 95.0 on THIS hardware and stack?
#   ~95 -> the gap is ours, and both trees run here so it is diffable.
#   ~80 -> our 79.7 is competitive and the published number does not transfer.
#
# Faithful to baselines/G2PO/examples/g2po_trainer/run_alfworld.sh except where this
# box forces a change. Every deviation is listed:
#
#   1. VLLM_ATTENTION_BACKEND: XFORMERS -> TRITON_ATTN. XFORMERS has no sm_120 kernel;
#      Blackwell requires Triton. Same backend our arms use, so this is not a handicap.
#      NOTE: docker/fa_stub must NOT be on PYTHONPATH. The real flash_attn (2.8.3.post1)
#      is installed and its unpad_input is pure PyTorch, so use_remove_padding=True works
#      on sm_120; the stub shadows it and raises. Our own arms never load the stub.
#   2. data paths -> /workspace/envdata/verl_data/text. Their script REGENERATES the
#      parquets via examples.data_preprocess.prepare; that step is DELIBERATELY SKIPPED.
#      Regenerating could draw a different validation set and destroy comparability with
#      every number we have. The existing test.parquet is 128 rows, the size their script
#      asks for.
#   3. logger console only (no wandb here). Metrics are parsed from the console log by
#      scripts/parse_g2po_log.py.
#   4. 2 GPUs, tensor_model_parallel_size=1 (theirs: 8 GPUs, TP=2). train_batch_size 16
#      divides 2, so this is valid; TP and GPU count change gradient accumulation, not
#      the objective.
#   5. gpu_memory_utilization 0.6 -> 0.3. THEIR 0.6 IS TUNED FOR 8 GPUs AT TP=2, where
#      each rank carries a quarter of what it carries here on 2 GPUs. Run at 0.6 the
#      first attempt allocated ~99 GB on a ~96 GB card (perf/max_memory_allocated_gb
#      98.8 mean / 99.4 peak, against 39.5 for our 4-GPU arms) and died at step ~37 with
#      `CUDA Error: out of memory` inside vLLM's cumem wake_up -- it sleeps and wakes the
#      KV pool every step, so sitting at 99% of the card is a coin flip each time.
#      This sizes the KV cache, not the result.
#   6. save_freq -1 -> 20. Theirs writes NO checkpoints, which meant the OOM at step 37
#      destroyed 7 hours with nothing to resume from. 20 gives resume points at a cost of
#      ~50 GB (verl keeps the last two); disk has 283 GB free.
#   6. RAY_* thread caps and single-thread BLAS, as in our arms: this box has a pid
#      budget and Ray's defaults exhaust it.
#
# UNCHANGED, deliberately: lr, kl_loss_coef/type, gamma, invalid-action penalty and coef,
# max_steps, group size, train_batch_size, prompt/response lengths, val temperature and
# sampling, test_freq, mini/micro batch sizes, save_freq=-1, total_epochs.
set -euo pipefail
D=/workspace/experiments/g2po-ref-20260909
GPUS="${GPUS:-3,5}"
NG=$(awk -F, '{print NF}' <<<"$GPUS")

cd /workspace/baselines/G2PO
env -i \
  ALFWORLD_DATA=/workspace/alfworld_data \
  CUDA_VISIBLE_DEVICES="$GPUS" \
  HF_HOME=/workspace/hf \
  HOME=/home/claude \
  MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PATH=/usr/local/cuda/bin:/usr/bin:/bin \
  PYTHONPATH=/workspace/baselines/G2PO \
  RAYON_NUM_THREADS=1 \
  RAY_TMPDIR=/tmp/ray_g2po \
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
    algorithm.adv_estimator=g2po \
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
    algorithm.g2po.step_advantage_w=1.0 \
    algorithm.g2po.mode=mean_std_norm \
    env.env_name=alfworld/AlfredTWEnv \
    env.seed=0 \
    env.max_steps=50 \
    env.rollout.n=8 \
    env.resources_per_worker.num_cpus=0.05 \
    trainer.critic_warmup=0 \
    "trainer.logger=[console]" \
    trainer.project_name=g2po_reference \
    trainer.experiment_name=g2po-ref-20260909 \
    trainer.n_gpus_per_node="$NG" \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.total_epochs=100 \
    trainer.val_before_train=False \
  >> "$D/outputs/train.log" 2>&1
