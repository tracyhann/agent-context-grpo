# cu128 stack for RTX PRO 6000 Blackwell (sm_120): vllm 0.9.1 + torch 2.7 base.
# Flaky-network hardening: mirror-first indexes, small cached layers, a
# persistent pip download cache shared across retries (BuildKit cache mount),
# and --no-build-isolation so sdist compiles stop spawning network-hungry
# isolated envs (build tools preinstalled instead).
FROM vllm/vllm-openai:v0.10.1
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential cmake git curl && rm -rf /var/lib/apt/lists/*
ENV PIP_DEFAULT_TIMEOUT=120 PIP_RETRIES=10 \
    PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
    PIP_EXTRA_INDEX_URL=https://pypi.org/simple
COPY wheels /wheels
ENV PIP_FIND_LINKS=/wheels
RUN ln -sf "$(command -v python3)" /usr/local/bin/python
RUN --mount=type=cache,target=/root/.cache/pip pip install "ray[default]"
RUN --mount=type=cache,target=/root/.cache/pip pip install \
      "tensordict==0.6.2" hydra-core codetiming torchdata \
      accelerate peft pyarrow pybind11 wandb uvicorn fastapi dill pandas gymnasium datasets
RUN --mount=type=cache,target=/root/.cache/pip pip install \
      setuptools wheel cython ninja   # cmake comes from apt (3.22, sufficient for fast-downward)
RUN --mount=type=cache,target=/root/.cache/pip pip install --no-build-isolation alfworld
RUN pip install --no-deps /wheels/transformers-4.51.1-py3-none-any.whl
ENV PYTHONDONTWRITEBYTECODE=1
ENTRYPOINT ["/bin/bash"]
