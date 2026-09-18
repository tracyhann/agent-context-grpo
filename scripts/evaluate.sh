#!/bin/bash
# Score a finished checkpoint on the held-out split across three seeds.
#
#   scripts/evaluate.sh <method> <size> <benchmark> <checkpoint> [gpus]
#   scripts/evaluate.sh hgpo 7b alfworld runs/hgpo-7b-alfworld-20260918/outputs/checkpoints/global_step_150
#
# WHY FIVE SEEDS. One 128-game evaluation at T=0.4 is noisy: the same FROZEN checkpoint
# scored 83.6 / 86.7 / 89.1 on the first three of these -- a 5.5-point spread with no
# training variance at all. Every headline number in RESULTS.md is a multi-seed mean ± std,
# and a single evaluation must never be quoted. A training run's own final score is the
# worst case of this: HGPO 7B logged 92.2 at step 160, while the 3-seed mean of that exact
# checkpoint is 96.09 ± 0.78.
#
# Seeds: 997 101 3173 (the original three) + 869 2917. Five seeds narrows the standard
# error of the mean by ~30% against three, at a cost of ~20 min per extra seed per run.
# Override with SEEDS="..." -- e.g. SEEDS="997 101 3173" to reproduce an earlier 3-seed
# number exactly.
#
# GPU COUNT. FSDP shards only load on the world size that wrote them. If you cannot give
# this the original GPU count, merge first and pass the merged directory instead:
#   verl-agent/scripts/model_merger.py merge --backend fsdp \
#     --local_dir <ckpt>/actor --target_dir <ckpt>-hf
# A merged model loads on any number of GPUs. This script auto-detects which you gave it.
#
# Runs sequentially: each evaluation builds ~7k processes against this container's 8192.
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/common.sh"
BASE=${BASE:-/workspace/baselines}
SEEDS=${SEEDS:-"997 101 3173 869 2917"}

METHOD=${1:?method}; SIZE=${2:?size}; BENCH=${3:?benchmark}; CKPT=${4:?checkpoint dir}; GPUS=${5:-0}
NG=$(awk -F, '{print NF}' <<<"$GPUS")
[ -d "$CKPT" ] || { echo "no such checkpoint: $CKPT"; exit 1; }

# merged HF directory (model.safetensors present) vs raw FSDP shards
if ls "$CKPT"/*.safetensors >/dev/null 2>&1; then MODEL="$CKPT"; RESUME=""
else
  shards=$(ls "$CKPT"/actor/model_world_size_*_rank_0.pt 2>/dev/null | head -1)
  ws=$(sed -E 's/.*world_size_([0-9]+)_.*/\1/' <<<"${shards:-}")
  [ -z "${ws:-}" ] && { echo "cannot tell the world size of $CKPT"; exit 1; }
  [ "$ws" != "$NG" ] && { echo "checkpoint is world_size_$ws but you gave $NG GPU(s)."; \
    echo "Give $ws GPUs, or merge it first (see header)."; exit 1; }
  case "$SIZE" in 1.5b) MODEL=${MODEL_1_5B:-latent-artist/66c200392ff148279a5995ef1455d3de} ;;
                  7b)   MODEL=${MODEL_7B:-latent-artist/f142c54d3ea54565a5ce95f881497e41} ;; esac
  RESUME="$CKPT"
fi

case "$METHOD" in
  grpo)  TREE=$BASE/verl-agent; ENTRY=verl.trainer.main_ppo;  ALG=(algorithm.adv_estimator=grpo) ;;
  gigpo) TREE=$BASE/verl-agent; ENTRY=verl.trainer.main_ppo;  ALG=(algorithm.adv_estimator=gigpo) ;;
  hgpo)  TREE=$BASE/verl-agent; ENTRY=recipe.hgpo.main_hgpo;  ALG=(algorithm.adv_estimator=hgpo) ;;
  g2po)  TREE=$BASE/G2PO;       ENTRY=verl.trainer.main_ppo;  ALG=(algorithm.adv_estimator=g2po) ;;
  *) echo "method must be grpo|gigpo|hgpo|g2po"; exit 1 ;;
esac
# must match train.sh -- the policy has to be scored under the prompts it trained on
case "$BENCH" in
  alfworld) ENV_NAME="alfworld/AlfredTWEnv"; MAX_STEPS=50
            if [ "$METHOD" = hgpo ]; then PROMPT=4096; TRUNC=left; else PROMPT=2048; TRUNC=error; fi ;;
  webshop)  ENV_NAME="Webshop"; MAX_STEPS=15; [ "$METHOD" = hgpo ] && MAX_STEPS=30
            PROMPT=4096; TRUNC=error; [ "$METHOD" = hgpo ] && TRUNC=left ;;
esac
if [ "$SIZE" = 7b ]; then MB=8; [ "$BENCH" = webshop ] && MB=2; else MB=32; [ "$BENCH" = webshop ] && MB=8; fi

TAG=${TAG:-eval-$METHOD-$SIZE-$BENCH-$(date +%Y%m%d)}
echo "== $TAG | model=$MODEL ${RESUME:+resume=$RESUME} | gpus=$GPUS | seeds: $SEEDS"
for SEED in $SEEDS; do
  D=/workspace/baseline-repo/runs/$TAG/seed-$SEED; rm -rf "$D"; mkdir -p "$D/outputs"
  RT=/tmp/ray_eval_${METHOD}_${SIZE}_${BENCH}_$SEED; rm -rf "$RT"
  echo "EVAL seed=$SEED start $(date +%T)"
  cd "$TREE"
  $(build_env "$GPUS" "$RT" "$TREE") "$VENV" -m "$ENTRY" "${ALG[@]}" \
      data.train_files="$DATA_DIR/train.parquet" data.val_files="$DATA_DIR/test.parquet" \
      data.train_batch_size=16 data.val_batch_size="$VAL_BATCH" \
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
      algorithm.gamma=0.95 env.env_name="$ENV_NAME" env.resources_per_worker.num_cpus=0.1 \
      env.seed="$SEED" env.history_length=2 env.max_steps="$MAX_STEPS" env.rollout.n=8 \
      ray_init.num_cpus="$RAY_CPUS" trainer.critic_warmup=0 "trainer.logger=[console]" \
      trainer.project_name=baselines_eval trainer.experiment_name="$TAG-s$SEED" \
      trainer.n_gpus_per_node="$NG" trainer.nnodes=1 \
      trainer.save_freq=-1 trainer.test_freq=1 trainer.total_epochs=1 \
      trainer.default_local_dir="$D/outputs/checkpoints" \
      trainer.val_only=True trainer.val_before_train=True \
      ${RESUME:+trainer.resume_mode=resume_path trainer.resume_from_path=$RESUME} \
    >> "$D/outputs/train.log" 2>&1
  sc=$(grep -rhoE "'val/success_rate': np\.float64\(([0-9.]+)\)" "$RT"/ray/session_*/logs/worker-*.out 2>/dev/null | grep -oE '[0-9.]+' | tail -1)
  [ -z "${sc:-}" ] && sc=$("$HERE/../logging/extract_val.py" "$D/outputs/train.log" 2>/dev/null)
  if [ -n "${sc:-}" ]; then echo "RESULT seed=$SEED $(awk "BEGIN{printf \"%.2f\", 100*$sc}")%"
  else echo "RESULT seed=$SEED PARSE-FAILED (see $D/outputs/train.log)"; fi
  for p in $(ps -eo pid,args | awk '/[r]ay::|[r]aylet|[g]cs_server/ {print $1}'); do kill -9 "$p" 2>/dev/null; done
  rm -rf "$RT"; sleep 20
done
"$HERE/../logging/summarise_eval.py" "/workspace/baseline-repo/runs/$TAG"
