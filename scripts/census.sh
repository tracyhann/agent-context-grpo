#!/bin/bash
# Full-coverage evaluation: play every held-out task exactly once, dump per-episode.
#
#   scripts/census.sh <method> <size> <benchmark> <checkpoint> [gpus] [decode_seed]
#   scripts/census.sh hgpo 7b alfworld runs/hgpo-7b-alfworld-.../checkpoints/global_step_150 0,1,2,3
#
# WHAT THIS REPLACES. evaluate.sh scores a SAMPLE of the split and repeats the
# seed to average the sampling noise away. This plays the whole split once, so
# sampling variance is exactly zero and there is nothing to average. The 128-task
# draws evaluate.sh would have produced are then reproduced OFFLINE from the dump
# at no GPU cost:
#
#   census/census_draw.py <dump> --seeds 997 101 3173 869 2917
#   census/census_draw.py <dump> --draws 20000        # the whole sampling distribution
#
# WHICH SEED MATTERS. With coverage pinned there is no draw left for env.seed to
# move, so it is FIXED at 0 here and the decoding seed is the one that varies. It
# needs a '+' because rollout.seed is absent from verl's config schema -- without
# the prefix Hydra hard-errors, and any other way of setting it silently leaves
# vLLM on seed 0 (which is what evaluate.sh does today: its five seeds all decode
# identically, so the spread it measures is task draw, not temperature).
#
# Repeat with different decode seeds to put an error bar on the census itself:
#   for s in 101 202 303; do scripts/census.sh ... "$s"; done
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/common.sh"
BASE=${BASE:-/workspace/baselines}
OUT_ROOT=${OUT_ROOT:-/workspace/baseline-repo/runs}

METHOD=${1:?method: grpo|gigpo|hgpo|g2po}; SIZE=${2:?size}; BENCH=${3:?benchmark}
CKPT=${4:?checkpoint dir}; GPUS=${5:-0}; DECODE_SEED=${6:-101}
NG=$(awk -F, '{print NF}' <<<"$GPUS")
[ -d "$CKPT" ] || { echo "no such checkpoint: $CKPT"; exit 1; }

if ls "$CKPT"/*.safetensors >/dev/null 2>&1; then MODEL="$CKPT"; RESUME=""
else
  shards=$(ls "$CKPT"/actor/model_world_size_*_rank_0.pt 2>/dev/null | head -1)
  ws=$(sed -E 's/.*world_size_([0-9]+)_.*/\1/' <<<"${shards:-}")
  [ -z "${ws:-}" ] && { echo "cannot tell the world size of $CKPT"; exit 1; }
  [ "$ws" != "$NG" ] && { echo "checkpoint is world_size_$ws but you gave $NG GPU(s)."; exit 1; }
  case "$SIZE" in 1.5b) MODEL=${MODEL_1_5B:-latent-artist/66c200392ff148279a5995ef1455d3de} ;;
                  7b)   MODEL=${MODEL_7B:-latent-artist/f142c54d3ea54565a5ce95f881497e41} ;; esac
  RESUME="$CKPT"
fi

case "$METHOD" in
  grpo)  TREE=$BASE/verl-agent; ENTRY=verl.trainer.main_ppo; ALG=(algorithm.adv_estimator=grpo) ;;
  gigpo) TREE=$BASE/verl-agent; ENTRY=verl.trainer.main_ppo; ALG=(algorithm.adv_estimator=gigpo) ;;
  hgpo)  TREE=$BASE/verl-agent; ENTRY=recipe.hgpo.main_hgpo; ALG=(algorithm.adv_estimator=hgpo) ;;
  g2po)  TREE=$BASE/G2PO;       ENTRY=verl.trainer.main_ppo; ALG=(algorithm.adv_estimator=g2po) ;;
  *) echo "method must be grpo|gigpo|hgpo|g2po"; exit 1 ;;
esac
python3 "$HERE/../census/apply_census_patch.py" --check "$TREE" >/dev/null 2>&1 || {
  echo "tree $TREE is not patched; run: census/apply_census_patch.py $TREE"; exit 1; }

# Prompt budget and turn cap must match how the method was TRAINED -- these differ
# by method on purpose, which is also why the cross-method table is not a
# matched-budget comparison. See README.
case "$BENCH" in
  alfworld) ENV_NAME="alfworld/AlfredTWEnv"; MAX_STEPS=50; CENSUS_VAR=ACG_ALF_CENSUS
            VB=140; SRC=$DATA_DIR
            if [ "$METHOD" = hgpo ]; then PROMPT=4096; TRUNC=left; else PROMPT=2048; TRUNC=error; fi ;;
  webshop)  ENV_NAME="Webshop"; MAX_STEPS=15; [ "$METHOD" = hgpo ] && MAX_STEPS=30
            CENSUS_VAR=ACG_WS_CENSUS; VB=125; SRC=${WEBSHOP_DATA_DIR:?set WEBSHOP_DATA_DIR}
            PROMPT=4096; TRUNC=error; [ "$METHOD" = hgpo ] && TRUNC=left ;;
  *) echo "benchmark must be alfworld|webshop"; exit 1 ;;
esac
if [ "$SIZE" = 7b ]; then MB=8; [ "$BENCH" = webshop ] && MB=2; else MB=32; [ "$BENCH" = webshop ] && MB=8; fi

CENSUS_DATA=${CENSUS_DATA:-/workspace/baseline-repo/census_data/$BENCH}
[ -f "$CENSUS_DATA/test.parquet" ] || \
  "$VENV" "$HERE/../census/make_val_parquet.py" "$BENCH" "$SRC" "$CENSUS_DATA"

TAG=${TAG:-census-$METHOD-$SIZE-$BENCH-d$DECODE_SEED-$(date +%Y%m%d)}
D=$OUT_ROOT/$TAG; rm -rf "$D"; mkdir -p "$D/outputs"
DUMP=$D/outputs/episodes.jsonl
RT=/tmp/ray_census_${METHOD}_${SIZE}_${BENCH}_$$; rm -rf "$RT"
echo "== $TAG | model=$MODEL ${RESUME:+resume=$RESUME} | gpus=$GPUS | decode seed $DECODE_SEED"
cd "$TREE"
# env.rollout.n=1: validation envs are ALWAYS built with group_n=1
# (env_manager.py:813/850), so a larger value only builds train_batch_size x n
# TRAINING actors that a val_only run never touches -- 128 of them at n=8. That is
# what pushes this container against its pid ceiling; with n=1 the full 140/125
# validation envs fit in one reset.
$(build_env "$GPUS" "$RT" "$TREE") "$CENSUS_VAR"=1 ACG_VAL_DUMP="$DUMP" \
  "$VENV" -m "$ENTRY" "${ALG[@]}" \
    data.train_files="$CENSUS_DATA/train.parquet" data.val_files="$CENSUS_DATA/test.parquet" \
    data.train_batch_size=16 data.val_batch_size="$VB" \
    data.max_prompt_length="$PROMPT" data.max_response_length=512 \
    data.filter_overlong_prompts=True data.truncation="$TRUNC" data.return_raw_chat=True \
    actor_rollout_ref.model.path="$MODEL" actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$MB \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$MB \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$MB \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization="$GPU_MEM_UTIL" \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    +actor_rollout_ref.rollout.seed="$DECODE_SEED" \
    algorithm.gamma=0.95 env.env_name="$ENV_NAME" env.resources_per_worker.num_cpus=0.1 \
    env.seed=0 env.history_length=2 env.max_steps="$MAX_STEPS" env.rollout.n=1 \
    ray_init.num_cpus="$RAY_CPUS" trainer.critic_warmup=0 "trainer.logger=[console]" \
    trainer.project_name=baselines_census trainer.experiment_name="$TAG" \
    trainer.n_gpus_per_node="$NG" trainer.nnodes=1 \
    trainer.save_freq=-1 trainer.test_freq=1 trainer.total_epochs=1 \
    trainer.default_local_dir="$D/outputs/checkpoints" \
    trainer.val_only=True trainer.val_before_train=True \
    ${RESUME:+trainer.resume_mode=resume_path trainer.resume_from_path=$RESUME} \
  >> "$D/outputs/train.log" 2>&1

# Kill only THIS run's ray session. Matching /ray::/ box-wide (as evaluate.sh does)
# also kills every other container's actors on a shared pid namespace.
for p in $(ps -eo pid,args --no-headers | grep -F "$RT" | grep -v grep | awk '{print $1}'); do
  kill -9 "$p" 2>/dev/null
done
rm -rf "$RT"

n=$(grep -c '' "$DUMP" 2>/dev/null || echo 0)
echo "== episodes dumped: $n (expected $( [ "$BENCH" = alfworld ] && echo 140 || echo 500 ))"
[ "$n" -gt 0 ] && "$VENV" "$HERE/../census/census_draw.py" "$DUMP" --n 128
