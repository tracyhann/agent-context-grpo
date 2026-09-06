#!/bin/bash
# Evaluate a checkpoint's HF export on one ALFWorld split, no training.
#
#   run_eval_split.sh <hf_model_dir> <eval_in_distribution|eval_out_of_distribution> <tag>
#
# Loads the exported HF weights as the model path (so no world-size-6 sharded
# resume is needed), runs val_before_train, prints the metrics, then exits.
# eval_in_distribution -> valid_seen (what G2PO's default reports on)
# eval_out_of_distribution -> valid_unseen (our training-time metric)
set -x
source /DATA/tracy/agentic-context-grpo/env.sh
HFDIR=${1:?need hf model dir}; SPLIT=${2:-eval_in_distribution}; TAG=${3:-evalsplit}
GPUS=0,1,2,3,4,5; NGPU=6
VENV=${ACG_VERL_BIN:-/usr/bin}
export PYTHONPATH="$ACG_ROOT/verl-agent:$ACG_ROOT/docker-acg/fa_stub"
export VERL_ATTN_IMPL=sdpa
export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
export CUDA_VISIBLE_DEVICES=$GPUS
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export RAY_TMPDIR=/tmp/ray_eval
mkdir -p "$RAY_TMPDIR"
DATA=$ACG_ROOT/envdata/verl_data
cd "$ACG_ROOT/verl-agent"
"$VENV/python" -m verl.trainer.main_ppo \
    algorithm.adv_estimator=ccpo \
    data.train_files=$DATA/text/train.parquet \
    data.val_files=$DATA/text/test.parquet \
    data.train_batch_size=18 \
    data.val_batch_size=128 \
    data.max_prompt_length=2048 \
    data.max_response_length=768 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    +data.apply_chat_template_kwargs.enable_thinking=False \
    actor_rollout_ref.model.path=$HFDIR \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.ppo_mini_batch_size=240 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=10 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=10 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=hf \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.7 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.8 \
    actor_rollout_ref.rollout.val_kwargs.top_k=20 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=10 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    algorithm.use_kl_in_reward=False \
    algorithm.gamma=0.95 \
    env.env_name=alfworld/AlfredTWEnv \
    env.seed=0 \
    env.max_steps=50 \
    env.rollout.n=8 \
    env.alfworld.eval_dataset=$SPLIT \
    env.resources_per_worker.num_cpus=0.1 \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.project_name='acg_eval' \
    trainer.experiment_name=$TAG \
    trainer.n_gpus_per_node=$NGPU \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.default_local_dir=/tmp/acg_eval_$TAG \
    trainer.test_freq=1 \
    trainer.total_epochs=1 \
    trainer.val_before_train=True
