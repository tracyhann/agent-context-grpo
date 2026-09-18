#!/bin/bash
# Launch one baseline run.
#
#   scripts/train.sh <method> <size> <benchmark> [gpus] [steps]
#   scripts/train.sh hgpo 7b alfworld 0,1,2,3 150
#
# method    grpo | gigpo | hgpo | g2po
# size      1.5b | 7b
# benchmark alfworld | webshop
# gpus      comma list, default 0,1,2,3   (must equal the checkpoint's world size on resume)
# steps     default 150
#
# WHERE EACH METHOD LIVES
#   grpo, gigpo   baselines/verl-agent            -m verl.trainer.main_ppo
#   hgpo          baselines/verl-agent            -m recipe.hgpo.main_hgpo   (own ray trainer)
#   g2po          baselines/G2PO                  -m verl.trainer.main_ppo   (separate tree)
#
# WHAT DIFFERS BY BENCHMARK (from the upstream scripts, unchanged here)
#   alfworld  env alfworld/AlfredTWEnv, max_steps 50, prompt 2048 (hgpo 4096)
#   webshop   env Webshop,              max_steps 15 (hgpo 30),   prompt 4096
#   both      train_data_size 16, group 8, val_data_size 128, history_length 2
#
# Deviations forced by this host live in common.sh, not here.
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/common.sh"
BASE=${BASE:-/workspace/baselines}
OUT_ROOT=${OUT_ROOT:-/workspace/baseline-repo/runs}

METHOD=${1:?method: grpo|gigpo|hgpo|g2po}
SIZE=${2:?size: 1.5b|7b}
BENCH=${3:?benchmark: alfworld|webshop}
GPUS=${4:-0,1,2,3}
STEPS=${5:-150}
NG=$(awk -F, '{print NF}' <<<"$GPUS")

case "$SIZE" in
  # REQUIRED backbones for these experiments -- blinded re-uploads, not the public Qwen
  # repos. Architecturally identical (1.5B: hidden 1536 / 28 layers / vocab 151936;
  # 7B: hidden 3584 / 28 layers / vocab 152064). Override with MODEL_1_5B / MODEL_7B.
  1.5b) MODEL=${MODEL_1_5B:-latent-artist/66c200392ff148279a5995ef1455d3de} ;;
  7b)   MODEL=${MODEL_7B:-latent-artist/f142c54d3ea54565a5ce95f881497e41} ;;
  *) echo "size must be 1.5b or 7b"; exit 1 ;;
esac

# --- per-method tree, entrypoint and estimator-specific flags -------------------------
case "$METHOD" in
  grpo)  TREE=$BASE/verl-agent; ENTRY="verl.trainer.main_ppo"
         ALG=(algorithm.adv_estimator=grpo) ;;
  gigpo) TREE=$BASE/verl-agent; ENTRY="verl.trainer.main_ppo"
         ALG=(algorithm.adv_estimator=gigpo algorithm.gigpo.step_advantage_w=1.0
              algorithm.gigpo.mode=mean_std_norm) ;;
  hgpo)  TREE=$BASE/verl-agent; ENTRY="recipe.hgpo.main_hgpo"
         ALG=(algorithm.adv_estimator=hgpo algorithm.hgpo.weight_type=length
              algorithm.hgpo.mode=mean_std_norm algorithm.hgpo.length_weight_alpha=1.0
              algorithm.hgpo.base_group=False) ;;
  g2po)  TREE=$BASE/G2PO;       ENTRY="verl.trainer.main_ppo"
         ALG=(algorithm.adv_estimator=g2po algorithm.g2po.step_advantage_w=1.0) ;;
  *) echo "method must be grpo|gigpo|hgpo|g2po"; exit 1 ;;
esac

# --- per-benchmark settings ----------------------------------------------------------
case "$BENCH" in
  alfworld) ENV_NAME="alfworld/AlfredTWEnv"; MAX_STEPS=50
            if [ "$METHOD" = hgpo ]; then PROMPT=4096; TRUNC=left; else PROMPT=2048; TRUNC=error; fi ;;
  webshop)  ENV_NAME="Webshop";              MAX_STEPS=15
            [ "$METHOD" = hgpo ] && MAX_STEPS=30
            PROMPT=4096; TRUNC=error; [ "$METHOD" = hgpo ] && TRUNC=left ;;
  *) echo "benchmark must be alfworld|webshop"; exit 1 ;;
esac

# --- batch geometry ------------------------------------------------------------------
# ppo_mini_batch_size is 256 on ALFWorld and 64 on WebShop in EVERY upstream script
# (grpo/gigpo/g2po examples and hgpo's recipe alike). It sets how many gradient updates
# each collected batch is split into, so it affects the optimisation, not just memory --
# it must track the benchmark, never be fixed.
MINI=256; [ "$BENCH" = webshop ] && MINI=64
# micro-batch is a pure memory knob: 7B needs a smaller one, and webshop prompts are 2x
# alfworld's. Values follow the upstream scripts (hgpo's 7B webshop script uses 2).
if [ "$SIZE" = 7b ]; then MB=8; [ "$BENCH" = webshop ] && MB=2; else MB=32; [ "$BENCH" = webshop ] && MB=8; fi

RUN=${RUN_NAME:-${METHOD}-${SIZE}-${BENCH}-$(date +%Y%m%d)}
D=$OUT_ROOT/$RUN
RAY_TMP=/tmp/ray_${METHOD}_${SIZE}_${BENCH}
mkdir -p "$D/outputs"
echo "[run] $RUN  method=$METHOD size=$SIZE bench=$BENCH gpus=$GPUS steps=$STEPS"
echo "[run] tree=$TREE entry=$ENTRY prompt=$PROMPT trunc=$TRUNC max_steps=$MAX_STEPS mini=$MINI micro=$MB"
echo "[run] out=$D"
[ "${DRY_RUN:-0}" = 1 ] && { echo "[run] dry run, not launching"; exit 0; }

rm -rf "$RAY_TMP"
cd "$TREE"
$(build_env "$GPUS" "$RAY_TMP" "$TREE") \
  "$VENV" -m "$ENTRY" \
    "${ALG[@]}" \
    data.train_files="$DATA_DIR/train.parquet" \
    data.val_files="$DATA_DIR/test.parquet" \
    data.train_batch_size=16 \
    data.val_batch_size="$VAL_BATCH" \
    data.max_prompt_length="$PROMPT" \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation="$TRUNC" \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path="$MODEL" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size="$MINI" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$MB" \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="$MB" \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization="$GPU_MEM_UTIL" \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="$MB" \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    algorithm.gamma=0.95 \
    env.env_name="$ENV_NAME" \
    env.resources_per_worker.num_cpus=0.1 \
    env.seed="${SEED:-0}" \
    env.history_length=2 \
    env.max_steps="$MAX_STEPS" \
    env.rollout.n=8 \
    ray_init.num_cpus="$RAY_CPUS" \
    trainer.critic_warmup=0 \
    "trainer.logger=[console]" \
    trainer.project_name="baselines" \
    trainer.experiment_name="$RUN" \
    trainer.n_gpus_per_node="$NG" \
    trainer.nnodes=1 \
    trainer.save_freq="${SAVE_FREQ:-20}" \
    trainer.max_actor_ckpt_to_keep="${KEEP_CKPT:-1}" \
    trainer.test_freq=5 \
    trainer.total_epochs="$STEPS" \
    trainer.default_local_dir="$D/outputs/checkpoints" \
    trainer.val_only=False \
    trainer.val_before_train=False \
    ${RESUME_FROM:+trainer.resume_mode=resume_path trainer.resume_from_path=$RESUME_FROM} \
  >> "$D/outputs/train.log" 2>&1
