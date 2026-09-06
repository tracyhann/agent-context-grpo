#!/bin/bash
# Build the training environment from a bare box. Idempotent; safe to re-run.
#
# Everything lands under the repo root (the persistent volume), not $HOME, so a
# container restart does not cost another hour of downloads.
#
# Blackwell (sm_120) notes:
#   * flash-attn has no sm_120 wheel, so the trainer runs sdpa and
#     use_remove_padding stays False. docker/fa_stub/ satisfies the import.
#   * vLLM runs its Triton attention backend (VLLM_ATTENTION_BACKEND=TRITON_ATTN),
#     which JIT-compiles per architecture.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="$ROOT/.venv/bin/python3"
PIP="$ROOT/.venv/bin/pip"

if [ ! -x "$PY" ]; then
  python3 -m venv .venv
  "$PIP" install -q --upgrade pip setuptools wheel
fi

# vLLM pins torch; install it first so nothing downgrades the CUDA build.
"$PY" -c 'import vllm' 2>/dev/null || "$PIP" install vllm==0.11.0
# verl-agent needs transformers<=4.57.3 and ray<=2.50.0; vllm pulls newer ones.
"$PIP" install -q "transformers==4.57.1" "ray[default]==2.50.0"

# runnable overlay = upstream verl-agent + patches/
if [ ! -d "$ROOT/verl-agent" ]; then
  [ -d "$ROOT/baselines/verl-agent" ] || git clone --depth 1 \
    https://github.com/langfengQ/verl-agent.git "$ROOT/baselines/verl-agent"
  cp -r "$ROOT/baselines/verl-agent" "$ROOT/verl-agent"
  rm -rf "$ROOT/verl-agent/.git"
fi
"$ROOT/scripts/sync_patches.sh"
"$PIP" install -q -e "$ROOT/verl-agent"

# ALFWorld
"$PY" -c 'import alfworld' 2>/dev/null || \
  "$PIP" install "gymnasium==0.29.1" "stable-baselines3==2.6.0" alfworld
export ALFWORLD_DATA="$ROOT/alfworld_data"
mkdir -p "$ALFWORLD_DATA"
[ -d "$ALFWORLD_DATA/json_2.1.1" ] || "$ROOT/.venv/bin/alfworld-download"

# base model
export HF_HOME="$ROOT/hf"
"$ROOT/.venv/bin/hf" download Qwen/Qwen2.5-1.5B-Instruct >/dev/null

# verl parquet shards (they only carry modality and dataset size)
if [ ! -f "$ROOT/envdata/verl_data/text/train.parquet" ]; then
  mkdir -p "$ROOT/envdata/verl_data"
  (cd "$ROOT/verl-agent" && "$PY" -m examples.data_preprocess.prepare --mode text \
      --train_data_size 16 --val_data_size 128 --local_dir "$ROOT/envdata/verl_data")
fi

"$PY" - <<'PYCHECK'
import torch, vllm, transformers, ray
print(f"torch {torch.__version__}  cuda={torch.cuda.is_available()} "
      f"devices={torch.cuda.device_count()}")
print(f"vllm {vllm.__version__}  transformers {transformers.__version__}  ray {ray.__version__}")
PYCHECK
echo "environment ready"
