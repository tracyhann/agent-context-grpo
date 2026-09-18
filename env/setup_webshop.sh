#!/bin/bash
# WebShop environment setup.  NOT YET RUN ON THIS HOST -- see "status" below.
#
# STATUS: WebShop has never been installed here. ALFWorld works out of the box
# (env/setup_alfworld.sh); WebShop needs everything below. Budget an hour, mostly download.
#
# WHY A SEPARATE VIRTUALENV -- do not skip this
#   WebShop's requirements.txt pins torch==2.6.0, transformers==4.51.1, numpy==1.26.4.
#   The shared /workspace/.venv has torch 2.8.0 / transformers 4.57.1 / numpy 2.2.6, and
#   every training run uses it. Installing WebShop's pins into it would (a) break running
#   jobs, and (b) likely drop Blackwell (sm_120) support, which needs torch >= 2.7.
#   So: install WebShop's *environment* deps into their own venv, and keep the trainer on
#   the shared one. The trainer talks to the env through ray actors, not imports.
#
# WHAT THIS NEEDS THAT ALFWORLD DOES NOT
#   - Java 11          pyserini's Lucene index is a JVM library
#   - ~5 GB of product data from Google Drive (gdown)
#   - a built Lucene index (search_engine/run_indexing.sh)
#   - spaCy models en_core_web_lg / en_core_web_sm
#
# COMPARABILITY WARNING
#   This repo's earlier WebShop work (experiments/08-26) used a BM25 retriever instead of
#   pyserini/Lucene, and docs/method.html states those success rates are NOT comparable to
#   published WebShop numbers. Use the pyserini path below if you intend to compare.
#   Also check which catalogue you are on: verl-agent's copy defaults to the 1,000-product
#   files (items_shuffle_1000.json), not the full ~1.18M catalogue.
set -eu
WS=${WS:-/workspace/baselines/verl-agent/agent_system/environments/env_package/webshop/webshop}
VENV_WS=${VENV_WS:-/workspace/.venv-webshop}
STEP=${1:-all}

echo "== 1. python venv (WebShop needs python <= 3.10; this box has $(python3 -V 2>&1))"
[ -d "$VENV_WS" ] || python3 -m venv "$VENV_WS"
"$VENV_WS/bin/pip" install -q --upgrade pip

echo "== 2. java 11 (pyserini/Lucene). No conda here, so install a JDK directly."
if ! command -v java >/dev/null; then
  echo "   java not found. Install one of:"
  echo "     apt-get install -y openjdk-11-jdk      # needs root"
  echo "     or unpack a JDK tarball and set JAVA_HOME"
  echo "   pyserini 0.17.0 expects Java 11."
fi

echo "== 3. WebShop python deps (into \$VENV_WS, NOT the shared venv)"
echo "   Installing unpinned where the pin conflicts with this box:"
"$VENV_WS/bin/pip" install -q \
  beautifulsoup4 cleantext Flask gdown gym==0.24.0 pyserini==0.17.0 \
  rank_bm25 requests rich scikit_learn selenium spacy thefuzz tqdm Werkzeug \
  || { echo "   pip install failed -- see notes above about pins"; exit 1; }
"$VENV_WS/bin/python" -m spacy download en_core_web_lg || true
"$VENV_WS/bin/python" -m spacy download en_core_web_sm || true

echo "== 4. product data (~5 GB, Google Drive via gdown; needs outbound network)"
mkdir -p "$WS/data"; cd "$WS/data"
dl() { [ -f "$2" ] || "$VENV_WS/bin/gdown" "https://drive.google.com/uc?id=$1" -O "$2"; }
dl 1EgHdxQ_YxqIQlvvq5iKlCrkEKR6-j0Ib items_shuffle_1000.json     # 1k products
dl 1IduG0xl544V_A_jv3tHXC0kyFi7PnyBu items_ins_v2_1000.json
dl 14Kb5SPBk_jfdLZ_CDBNitW98QLDlKR5O items_human_ins.json
if [ "${FULL_CATALOGUE:-0}" = 1 ]; then
  dl 1A2whVgOO0euk5O13n2iYDM0bQRkkRduB items_shuffle.json        # full ~1.18M
  dl 1s2j6NgHljiZzQNL3veZaAiyW_qDEgBNi items_ins_v2.json
fi

echo "== 5. build the Lucene index"
cd "$WS/search_engine"
mkdir -p resources resources_100 resources_1k resources_100k indexes
echo "   run: bash run_indexing.sh   (uses \$VENV_WS/bin/python -m pyserini.index.lucene)"

echo "== 6. smoke test"
cat <<'PY' > /tmp/webshop_smoke.py
import sys; sys.path.insert(0, "WS_PATH")
from web_agent_site.envs import WebAgentTextEnv
import gym
env = gym.make('WebAgentTextEnv-v0', observation_mode='text', num_products=None)
obs, info = env.reset(session=0)
print("reset OK, obs chars:", len(obs), "| actions:", len(env.get_available_actions()))
PY
sed -i "s|WS_PATH|$WS|" /tmp/webshop_smoke.py
echo "   run: \$VENV_WS/bin/python /tmp/webshop_smoke.py"
echo
echo "== MEMORY WARNING"
echo "   Each ray env actor builds its own WebAgentTextEnv (own product table + search"
echo "   engine). A run uses 128 train + $((${VAL_BATCH:-64})) val actors. Measure host RAM on a"
echo "   SHORT run before launching 150 steps: ALFWorld already uses ~200 GB here."
