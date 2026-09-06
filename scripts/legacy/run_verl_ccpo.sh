#!/bin/bash
# G2PO-scale run of OUR estimator inside verl-agent: free generation (<=512 think
# tokens + action text), CoT trained by the RL gradient, batch 16 tasks x group 8
# = 128 episodes/step, minibatch 256, 150 steps -- mirroring
# examples/gigpo_trainer/run_alfworld.sh with algorithm.adv_estimator=ccpo.
# Usage: run_verl_ccpo.sh <gpus e.g. 0,3> <estimator ccpo|grpo> <tag>
#
# Eval protocol matches g2po_official/examples/g2po_trainer/run_alfworld.sh, which
# is what their reported numbers come from:
#   val temperature 0.4, top_p 1.0, top_k unrestricted, do_sample True, 128 episodes
#   eval_dataset unset -> eval_in_distribution (valid_seen)
# We had been running temperature 0.7 / top_p 0.8 / top_k 20 on valid_unseen. The
# split is now ACG_EVAL_SPLIT (default eval_in_distribution) so both can be run.
# top_k is 0 rather than -1 because ours is the `hf` rollout, where 0 is the
# unrestricted value (-1 is the vllm spelling; G2PO runs vllm).
# NOTE eval is stochastic by design here, so a single 128-episode score carries
# ~+/-0.048 (measured from two evaluations of the same checkpoint, 0.094 vs 0.188).
# Report the mean of >=3 seeds per checkpoint, do not read single points.
set -x
source /DATA/tracy/agentic-context-grpo/env.sh
GPUS=${1:-0,1,2,3,6,7}; EST=${2:-ccpo}; TAG=${3:-verl_${EST}_alfworld}
NGPU=$(echo "$GPUS" | awk -F, '{print NF}')
# verl asserts episodes (train_batch_size*n) % n_gpus == 0 and splits the PPO
# minibatch across GPUs; pick the closest matched-scale numbers that divide.
if [ "$NGPU" = 5 ]; then TB=15; MINI=240; MICRO=24  # 120 eps/step (G2PO: 128); 5 clean GPUs beat 6 with a straggler
elif [ "$NGPU" = 6 ]; then TB=18; MINI=240; MICRO=10  # 20 OOMed at step 26 once responses grew; same math, more accumulation      # 144 eps/step (G2PO: 128), mini 240 (G2PO: 256)
elif [ "$NGPU" = 4 ]; then TB=16; MINI=256; MICRO=32    # exact G2PO numbers
elif [ "$NGPU" = 2 ]; then TB=18; MINI=240; MICRO=10    # ocean-coexist fallback: identical batch math to the 6-GPU config, just slower
else TB=16; MINI=256; MICRO=32; fi
VENV=${ACG_VERL_BIN:-$ACG_ROOT/.venv-verl/bin}   # in-container: ACG_VERL_BIN=/usr/bin
export PYTHONPATH="$ACG_ROOT/verl-agent:$ACG_ROOT/docker-acg/fa_stub"
export VERL_ATTN_IMPL=sdpa
# Blackwell (sm_120): xformers has no kernels here — let vllm's V1 engine
# auto-select its attention backend (FlashInfer/Triton JIT support sm_120).
# sm_120: vllm 0.9.1's bundled FA fork lacks kernels; Triton JIT-compiles per-arch
export VLLM_ATTENTION_BACKEND=TRITON_ATTN   # only affects vllm; trainer uses sdpa. Sidecars set this in their own env too.
export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas   # triton 3.3.0 bundles a pre-sm_120 ptxas; use the CUDA 12.8 one
unset VLLM_USE_V1
export VLLM_USE_FLASHINFER_SAMPLER=0
export CUDA_VISIBLE_DEVICES=$GPUS
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # counter long-run fragmentation
export RAY_TMPDIR=/tmp/ray_acg
mkdir -p "$RAY_TMPDIR"
# Length control: REVERTED 2026-09-03 (user: keep comparability to the baseline).
# The reward-side penalty is off by default. It was also structurally weak here:
# group-relative advantage mean-centres over the 8 siblings, so any penalty common to
# the whole group cancels -- only differential length survives, which is why steps
# 43-45 arrested growth (374/372/374) without reversing it. Length control, if we
# reintroduce it, belongs at decode time (budget forcing) or in the context, not the reward.
export ACG_LEN_PENALTY=${ACG_LEN_PENALTY:-0.0}
export ACG_LEN_SOFT=${ACG_LEN_SOFT:-512}
export ACG_LEN_HARD=${ACG_LEN_HARD:-768}
# ACG sidecar generation (vllm 0.28 servers on host loopback; container is host-net).
# hf_rollout falls back to local HF generate whenever these are absent/stale.
export ACG_SIDECAR_HOST=127.0.0.1   # vLLM 0.28 + TRITON_ATTN generation containers; client falls back to HF on any failure
export ACG_SIDECAR_PORTS=8101,8101,8102,8103,8104,8105   # rank0 shares GPU-1's engine (no sidecar on GPU 0)
export ACG_SIDECAR_MODEL=acg
export ACG_SIDECAR_STATE=$ACG_ROOT/experiments/08-27/results/$TAG/sidecar_state.json
export ACG_SIDECAR_ITER=$ACG_ROOT/experiments/08-27/results/$TAG/latest_checkpointed_iteration.txt
# Comprehensive CCPO diagnostics: per-sample rows (level, lam, b_loo, b_obs,
# effect, our advantage against BOTH reference estimators, affordance label).
# Response-length pairs go to $ACG_CCPO_DUMP.len.csv -- they used to be appended
# to this same file under a different schema, which made it unparseable.
export ACG_CCPO_DUMP=$ACG_ROOT/experiments/08-27/results/$TAG/ccpo_samples.csv
# ---- CCPO estimator knobs (all default to the pre-2026-09-06 behaviour) -----
# target=nextnode swaps the step credit's TARGET from the step's own discounted
# return-to-go to G2PO's successor node value V(next(u)) -- the quantity G2PO
# actually credits. This is the largest single change available here and it is
# UNMEASURED; leave it at `return` unless you are running the comparison.
export ACG_CCPO_TARGET=${ACG_CCPO_TARGET:-return}
# sim>0 replaces the byte-exact observation gate with GiGPO's SequenceMatcher
# clustering (their default is 0.95). sim_backoff>0 gives occurrences whose
# level-0 bucket is a singleton a second chance in a looser cluster.
export ACG_CCPO_SIM=${ACG_CCPO_SIM:-0.0}
export ACG_CCPO_SIM_BACKOFF=${ACG_CCPO_SIM_BACKOFF:-0.0}
export ACG_CCPO_BACKOFF_RHO=${ACG_CCPO_BACKOFF_RHO:-0.5}
# weight on G2PO's edge term V(next)-V(current), standardised per task
export ACG_CCPO_EDGE_W=${ACG_CCPO_EDGE_W:-0.0}
export ACG_CCPO_INVALID_PEN=${ACG_CCPO_INVALID_PEN:-0.1}
# Budget forcing (decode-time): cap the think block, then force </think><action>
# on any response that lacks one. Truncation-without-action becomes impossible.
# Validated offline: parseable 5/8 -> 8/8, max length 659 -> 313.
export ACG_FORCE_BUDGET=${ACG_FORCE_BUDGET:-512}
export ACG_FORCE_TAIL=${ACG_FORCE_TAIL:-32}
# Context compaction: budgeted digest of the FULL history prepended to the
# recent-2 window. Motivation measured: 58.7% of turns revisit an already-seen
# observation; failures revisit 2.68x as often as successes.
export ACG_COMPACT_BUDGET=${ACG_COMPACT_BUDGET:-512}
mkdir -p $ACG_ROOT/experiments/08-27/results/$TAG
# ---- comparability with the G2PO reference -------------------------------
# g2po_official/examples/g2po_trainer/run_alfworld.sh, which is where their
# published ALFWorld numbers come from. Already identical here: val_batch_size
# 128, group 8, max_prompt_length 2048, lr 1e-6, kl 0.01 low_var_kl, gamma 0.95,
# step_advantage_w 1.0, invalid-action penalty 0.1, env.max_steps 50, env.seed 0,
# eval split eval_in_distribution (valid_seen), val temp 0.4 / do_sample.
#
# ACG_G2PO_COMPAT=1 aligns the three that still differ by choice. What it CANNOT
# align is hardware-forced: base model (Qwen3-1.7B vs their Qwen2.5-1.5B-Instruct),
# rollout engine (hf + vLLM sidecar vs in-process vllm), use_remove_padding
# (no flash-attn on sm_120), and the batch/GPU geometry. Those stay differences,
# so published G2PO numbers remain regime context, not a parity claim.
if [ "${ACG_G2PO_COMPAT:-0}" = 1 ]; then
  ACG_MAX_RESP=${ACG_MAX_RESP:-512}       # theirs: 512 (ours had 768)
  ACG_TRAIN_TOPP=${ACG_TRAIN_TOPP:-1.0}   # theirs: unset -> 1.0 (ours 0.95, Qwen3 card)
  ACG_TRAIN_TOPK=${ACG_TRAIN_TOPK:-0}     # theirs: unset -> unrestricted (ours 20)
  ACG_EPOCHS=${ACG_EPOCHS:-100}           # theirs: 100 (ours 75)
fi
MAX_RESP=${ACG_MAX_RESP:-768}
TRAIN_TOPP=${ACG_TRAIN_TOPP:-0.95}
TRAIN_TOPK=${ACG_TRAIN_TOPK:-20}
# Normalisation convention, applied to the episode AND step terms alike. The
# G2PO reference script sets mean_std_norm, and verl's GRPO arm standardises by
# default (norm_adv_by_std_in_grpo=True) -- so this is the only setting under
# which all three arms carry advantages of the same scale and step_advantage_w=1
# means what it says. verl-agent's own default is mean_norm.
ADV_MODE=${ACG_ADV_MODE:-mean_std_norm}
MODEL=/DATA/tracy/hf/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e
DATA=$ACG_ROOT/envdata/verl_data
cd "$ACG_ROOT/verl-agent"
if [ ! -f "$DATA/text/train.parquet" ]; then
  "$VENV/python" -m examples.data_preprocess.prepare --mode text \
    --train_data_size $TB --val_data_size 128 --local_dir "$DATA"
fi
# use_remove_padding needs a working flash-attn; detect it live (Blackwell has none)
RP=False; "$VENV/python" -c "import flash_attn_2_cuda" 2>/dev/null && RP=True
"$VENV/python" -m verl.trainer.main_ppo \
    algorithm.adv_estimator=$EST \
    data.train_files=$DATA/text/train.parquet \
    data.val_files=$DATA/text/test.parquet \
    data.train_batch_size=$TB \
    data.val_batch_size=128 \
    data.max_prompt_length=2048 \
    data.max_response_length=$MAX_RESP \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    +data.apply_chat_template_kwargs.enable_thinking=False \
    actor_rollout_ref.model.path=$MODEL \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=$RP \
    actor_rollout_ref.actor.ppo_mini_batch_size=$MINI \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$MICRO \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=${ACG_KL_COEF:-0.01} \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$MICRO \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=hf \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.top_p=$TRAIN_TOPP \
    actor_rollout_ref.rollout.top_k=$TRAIN_TOPK \
    actor_rollout_ref.rollout.val_kwargs.temperature=${ACG_VAL_TEMP:-0.4} \
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_k=0 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$MICRO \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    algorithm.use_kl_in_reward=False \
    algorithm.gamma=0.95 \
    algorithm.gigpo.step_advantage_w=1.0 \
    algorithm.gigpo.mode=$ADV_MODE \
    env.env_name=alfworld/AlfredTWEnv \
    env.seed=0 \
    env.max_steps=50 \
    env.rollout.n=8 \
    env.alfworld.eval_dataset=${ACG_EVAL_SPLIT:-eval_in_distribution} \
    env.resources_per_worker.num_cpus=0.1 \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.project_name='acg_verl' \
    trainer.experiment_name=$TAG \
    trainer.n_gpus_per_node=$NGPU \
    trainer.nnodes=1 \
    trainer.save_freq=1 \
    "actor_rollout_ref.actor.checkpoint.contents=[model,optimizer,extra,hf_model]" \
    trainer.max_actor_ckpt_to_keep=2 \
    trainer.default_local_dir=$ACG_ROOT/experiments/08-27/results/$TAG \
    trainer.test_freq=5 \
    trainer.total_epochs=${ACG_EPOCHS:-75} \
    trainer.val_before_train=True
