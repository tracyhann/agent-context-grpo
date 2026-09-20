# CCPO — Context-Conditioned Policy Optimization

Credit assignment for multi-turn agent RL. The current main method is **M10/M11
H2 with no credit shrinkage and no episode advantage**: full-strength contextual
history baselines plus two-step contextual future progress. It retains frozen
hidden-state + context-statistics representations and soft exponential weights.
The [main-method definition](experiments/MAIN_METHOD.md) records its math, exact
configurations and ablations. Earlier shrinkage and episode-on arms remain
historical comparators.

* Benchmarks: **ALFWorld**, **WebShop** · Backbones: **Qwen2.5-1.5B / 7B-Instruct**
* Target hardware: **A100 (sm_80)** and **H100 (sm_90)**
* Checkpoint backups: **https://huggingface.co/tracyhan816/ccpo-variants**

**The spec is [`experiments/experiments.md`](experiments/experiments.md)** — the selected
main method, historical matrix, later ablations, hyperparameters and reporting format.

---

## 1. Quickstart

Clone the **`amz-a100-h100`** branch and run commands from that checkout:

```bash
git clone --branch amz-a100-h100 https://github.com/tracyhann/agent-context-grpo.git
cd agent-context-grpo
```

For **WebShop / Qwen2.5-1.5B**, on Linux x86_64 with Python 3.10 and an NVIDIA
GPU runtime (see §2–3 for prerequisites):

```bash
scripts/setup_webshop.sh                 # venv, Java, catalogue/index, data, model
scripts/setup_webshop.sh --check         # CPU validation of the prepared runtime

# Main method: H2, no shrinkage, episode advantage off (1.5B); use allocated GPU IDs
python3 ablations/run.py --ablation future-progress-h2-no-credit-shrinkage \
  --benchmark webshop --gpus 0,1,2,3

python3 scripts/exp_status.py
python3 scripts/report_results.py --exp experiments/<exp-id> --paper-table
```

For **ALFWorld**, use its separate setup and launcher selection:

```bash
scripts/setup_env.sh
python3 ablations/run.py --ablation future-progress-h2-no-credit-shrinkage \
  --benchmark alfworld --gpus 0,1,2,3
```

| Benchmark | Setup command | Python environment | Training inputs |
| --- | --- | --- | --- |
| WebShop | `scripts/setup_webshop.sh` | `.venv-webshop/` | `envdata/webshop_data/` + local JDK |
| ALFWorld | `scripts/setup_env.sh` | `.venv/` | `alfworld_data/` + `envdata/verl_data/` |

The WebShop setup works independently of the ALFWorld setup. Both environments share
this checkout's patched `verl-agent/` source and `hf/` model cache. The launchers
choose the matching venv automatically; shell activation is optional.

Add `--dry-run` to a launch to write `config.json` and `run.sh` without starting the
trainer. `--gpus` is required: the spec recommends at least four devices for 1.5B and
eight for 7B. Archived two-GPU runs used explicit resource overrides (§5).

## 2. Container

The optional `docker/Dockerfile` starts from **`vllm/vllm-openai:v0.10.1`** and adds
trainer and ALFWorld dependencies, including a pinned `transformers` wheel. The
benchmark setup scripts in §3 install their own stacks in separate virtual environments.

```bash
cd docker
mkdir -p wheels                       # the build context expects it
#   place transformers-4.51.1-py3-none-any.whl here (the Dockerfile installs it
#   with --no-deps); any other local wheels are picked up via PIP_FIND_LINKS
DOCKER_BUILDKIT=1 docker build -t ccpo:cu128 .
cd ..
```

Three things to know before building:

* **`COPY wheels /wheels` is not optional.** The build fails if `docker/wheels/` does
  not exist, and the last layer installs `transformers-4.51.1-py3-none-any.whl` from it.
* **The pip index is mirrored** (`PIP_INDEX_URL` → Tsinghua, with pypi.org as the extra
  index) for flaky-network hardening. Drop or repoint those `ENV` lines if that mirror
  is slow from your location.
* **FlashAttention is provisioned per venv.** ALFWorld uses `scripts/setup_env.sh`,
  which fetches `flash_attn-2.8.3.post1+cu12torch2.8cxx11abiFALSE-cp310` — a cu12 /
  torch 2.8 build that covers sm_80 and sm_90. This matters: the launchers **refuse to
  start without a working FA2 build** (§5), because the fallback disables sequence
  packing. WebShop uses its separate torch 2.6 / FlashAttention 2.7.4 build (§3).

The image provides a host runtime. Mount the checkout into it, then run the setup
command for the benchmark you want; the image alone does not provision WebShop.
WebShop's setup requires a Python 3.10 interpreter even if the container's default
Python has another version (`--python /path/to/python3.10`).

```bash
docker run --gpus all -it --shm-size=32g \
  -v "$PWD:/workspace/agent-context-grpo" -w /workspace/agent-context-grpo ccpo:cu128
```

## 3. Environment

### ALFWorld

`scripts/setup_env.sh` builds the ALFWorld runtime under the checkout and reuses
existing downloaded assets:

1. `.venv/` with the training stack, pinned against verl-agent's requirements;
2. `verl-agent/` — upstream cloned into `baselines/`, copied out, then
   `scripts/sync_patches.sh` overlays `patches/verl-agent/` onto it;
3. flash-attn (see §2);
4. `alfworld_data/`, `hf/` (Qwen weights) and `envdata/verl_data/` (the parquets).

**`patches/verl-agent/` is the source of truth for every file we changed in the
trainer.** Edit there and re-run `scripts/sync_patches.sh`; never edit `verl-agent/`
directly — it is gitignored and gets overwritten.

### WebShop

Run the separate WebShop setup from the checkout root. All assets default to this
project's directory, including when it is a subdirectory:

```bash
scripts/setup_webshop.sh                 # provision and validate; includes Qwen 1.5B
scripts/setup_webshop.sh --check         # validate existing assets without downloads
# For environment-only preparation (weights checked/downloaded separately):
scripts/setup_webshop.sh --skip-model
```

Requires **Linux x86_64, Python 3.10 with venv support**, and `libgomp.so.1` (Ubuntu:
`libgomp1`). Training requires an NVIDIA driver supporting CUDA 12.4 and the A100/H100
hardware used by these runs. Setup downloads prebuilt CUDA/FlashAttention wheels;
no system CUDA toolkit or system Java installation is required. Allow roughly
25 GB for the environment, model and caches on a cold install.

The script provisions:

* `.venv-webshop/`: torch **2.6.0 / CUDA 12.4**, vLLM **0.8.4**, transformers
  **4.51.1**, Ray **2.46.0**, tensordict **0.7.2**, FlashAttention **2.7.4.post1**;
* `verl-agent/`: upstream revision `20bd331bdbc9026a5668e11362178e10ab7400c8`
  with this project's `patches/verl-agent/` overlay;
* `jdk/jdk-11.0.32.1+1/`: Temurin Java 11 for Pyserini/Lucene;
* `envdata/webshop_data/data/`: the pinned **1,000-product** catalogue and attributes;
* `envdata/webshop_data/search_engine/indexes/`: a locally built Lucene index, linked
  into the runtime alongside the data;
* `envdata/webshop_data/text/{train,test}.parquet`: 16 / 256 modality placeholders,
  generated locally; actual goals come from WebShop;
* `hf/`: the pinned `Qwen/Qwen2.5-1.5B-Instruct` snapshot; `--skip-model` omits it.

[requirements/webshop.txt](requirements/webshop.txt) lists the direct dependencies;
[webshop-lock.txt](requirements/webshop-lock.txt) pins their resolved transitive
versions. [webshop-assets.json](requirements/webshop-assets.json) records source,
data and model revisions plus SHA-256 checksums for source, data, Java and wheels.
The pinned source is made importable through a `.pth` file: upstream `setup.py` has
a tensordict constraint incompatible with this torch 2.6 runtime, so setup does not
use `pip install -e verl-agent`.

Existing environments, data and indexes are reused. Setup **does not upgrade an
existing venv or overwrite differing runtime patches/data**. A checksum or patch
mismatch stops with the conflicting path. A new venv uses spaCy 3.7.5 / weasel 0.4.1
and compatible NumPy/OpenCV/CuPy versions to resolve metadata conflicts in the
migrated environment, while preserving the M5 training core. The migrated venv's
known `pip check` conflicts are reported without changing its packages; newly
provisioned environments must pass `pip check`. The exact seven legacy conflicts
and their fresh-install resolutions are listed in [requirements/README.md](requirements/README.md).

Checks exercise trainer imports, the compiled FlashAttention extension, all 1,000
catalogue/index identifiers, both spaCy models, parquet inputs, **6,910 synthetic
goals**, and scripted purchases yielding score 1.0 → reward 10 and score 0.75 →
reward 0. They also read the local model configuration, tokenizer and weight shards.
The report is `.cache/webshop-setup/report.json`. Setup checks run on CPU;
`exp_run.py` separately tests the attention kernel on the selected GPU at launch.
These scripted purchase scores are environment checks, not model evaluation results.

`exp_run.py` automatically selects `.venv-webshop` and the local JDK for WebShop.
`--python /path/to/python3.10` chooses the interpreter for a new venv;
`--venv /path/to/new-venv` supports isolated installation checks (training launchers
continue to use `.venv-webshop`). Downloads cache under `.cache/webshop-setup/` and
pip under `.cache/pip/`. Failed new installations can be rerun to finish provisioning.
The default setup supplies 1.5B weights; 7B weights can be cached separately:

```bash
HF_HOME="$PWD/hf" .venv-webshop/bin/hf download Qwen/Qwen2.5-7B-Instruct
```

## 4. Layout

```
ccpo/            the estimator AND the main-method arms
  core_ccpo.py     CCPO itself: gate, phi kernel, leave-one-out baseline, credibility prior
  arms.py          BASE / BENCHMARK / BACKBONE / METHODS — every arm is BASE plus a delta
  run.py           launcher for the main-method arms        test_arms.py   guards
ablations/       the six ablations (plus one WebShop-only variant)
  ablations.py     run.py     test_ablations.py
requirements/    WebShop direct dependencies, resolved lock and pinned asset manifest
scripts/         exp_run.py (training entry point), setup_env.sh, setup_webshop.sh,
                 check_webshop.py, sync_patches.sh,
                 report_results.py, chain_eval.py, plot_metrics.py, exp_status.py,
                 resume_guard.py, pin_watch.py, hf_backup.py,
                 hf_verify_and_delete.py
patches/verl-agent/   our trainer overlay: advantage assembly (ray_trainer.py), the phi
                      hook (dp_actor.py), anchors (env_manager.py), task_scores
                      (rollout_loop.py), episode rewards (reward_manager/)
docker/          Dockerfile, and fa_stub/ — an import-only flash_attn stub for
                 architectures with no FA2 build
tests/           guards on the estimator and on protocol fidelity
experiments/     experiments.md (the spec), the control's config.json, and where every
                 run writes metrics, checkpoints and plots
```

Provisioned at deploy time and deliberately absent: `verl-agent/`, `.venv`,
`.venv-webshop`, `alfworld_data/`, `envdata/`, `hf/`, `jdk/`. Portable source files
resolve project paths from the checkout root. Machine-specific launch scripts,
local controllers and runtime caches are generated locally and excluded from Git.

## 5. Running experiments

```bash
python3 ccpo/run.py --list                 # the 10 main runs
python3 ablations/run.py --list            # the 13 ablation runs

python3 ccpo/run.py --method attncred --benchmark webshop --backbone 7b --gpus 0,1,2,3,4,5,6,7
python3 ablations/run.py --ablation cosine --benchmark alfworld --gpus 0,1,2,3
```

`--gpus` is required and has no default; `run.py` warns below the floor (≥4 for 1.5B,
≥8 for 7B). Arguments after the fixed flags go to `scripts/exp_run.py` verbatim and a
later `--set` wins, so `--set val_batch_size=64` adapts to a host without editing an arm.

M5 is `--method attncred-context-adv-only-return --benchmark webshop`: it sets
`ccpo_target=return` and `ccpo_ep_w=0`. Its independent ablations keep those settings:
add `--set ccpo_edge_w=0` for no edge term, or `--set ccpo_phi=hidden` for no context
vector. Keep the credibility shrinkage defaults (`ccpo_prior_kappa=2`, empty
`ccpo_lk_fix`); `ccpo_lam_fix=1` controls a different feature-readout weight.
Each ablation starts from the base model. Use a distinct `--name` for each run.

The dated M3/M5 records in `experiments/` document two-A100 runs, including their
resource overrides. Use their `config.json` and `NOTES.md` for the exact setup;
they do not change the default GPU recommendation. Concurrent launches use separate
Ray clusters and compilation caches, with bounded worker/thread settings.

Both launchers **preflight flash attention** on the training interpreter and refuse to
start unless `flash_attn_2_cuda` is compiled and `flash_attn_varlen_func` actually runs
on the card — an import check is not enough, because `docker/fa_stub` imports cleanly
and raises only when a kernel is called. `--allow-sdpa` overrides; a run made with it is
not comparable to the others.

Each run writes to `experiments/<exp-id>/`: `config.json` and `run.sh` (exactly what was
launched), `outputs/metrics.jsonl`, `outputs/train.log`, `outputs/checkpoints/`, and
`plots/`. Generated `run.sh` files contain machine-specific paths; regenerate them
with the launcher on a new machine.

## 6. Results, plots and checkpoints

```bash
python3 scripts/report_results.py --exp experiments/<exp-id> --paper-table [--markdown]
```

prints success by task type for **both** training rollouts and the held-out split at
step 100 / best / final, plus turns per episode — the format `experiments.md` §5 defines.
`plot_metrics.py --watch` is started automatically by every launch, so
`plots/progress.png` (success, per-type success, turns, KL, entropy, …) and
`plots/ccpo.png` (the estimator's own terms) refresh while the run trains.

For WebShop, **Success** is the fraction of purchases with perfect task reward;
**Score** is the mean graded task score (reported on a 0–100 scale). M5 trains on
binary returns while still logging both evaluation metrics.

Committed configs, notes, metrics and plots are snapshots at commit time. They do
not provide live process status on another machine. Checkpoints, raw logs, PID
files, startup debug records and `.local/` queue controllers stay on the training
host; notes may reference those local records.

Each run keeps three checkpoints — `step<N>-best`, `step100-pin`, `step<N>-last` —
at ~25 GB each for 1.5B and ~125 GB for 7B. To resume from or re-score one, copy it back
to a name verl accepts first:

```bash
cp -al experiments/<exp-id>/outputs/checkpoints/step100-pin \
       experiments/<exp-id>/outputs/checkpoints/global_step_100
python3 scripts/exp_run.py --name evalck-<tag> --arm ccpo --no-plot \
  --set gpus=<ids> --set val_only=1 --set align_val_on_resume=0 --set total_epochs=1 \
  --set resume_from=experiments/<exp-id>/outputs/checkpoints/global_step_100
```

`align_val_on_resume=0` is required, or checkpoints from different steps are each scored
on a different validation draw. `scripts/chain_eval.py` drives that loop for a finished
arm and repeats it for a multi-draw estimate.

## 7. Checkpoint backups — Hugging Face

**https://huggingface.co/tracyhan816/ccpo-variants**

Checkpoints are archived there under one folder per run, so that local disk can be
reclaimed without losing a result. Log in once (`hf auth login`, a write token), then:

```bash
# wait for the run to finish, then upload best + step100-pin + final, and verify
python3 scripts/hf_backup.py --exp experiments/<exp-id> [--steps 150]

# only after that: prove each copy is resumable and byte-identical, then delete locally
python3 scripts/hf_verify_and_delete.py --exp experiments/<exp-id>
```

The backup uploads exactly the three checkpoints worth keeping, deletes nothing, and
verifies file count and total bytes against the local tree before reporting success.
`hf_verify_and_delete.py` is the only thing that removes a local copy, and only after
three gates pass: the FSDP shard set is complete and resumable, every `.pt` /
`.safetensors` opens locally, and the local sha256 matches the LFS sha256 Hugging Face
reports for every file. Anything short of all three leaves the local copy in place.

To pull one back:

```bash
hf download tracyhan816/ccpo-variants --include "<exp-id>/step100-pin/*" \
   --local-dir experiments/<exp-id>/outputs/checkpoints
```

Both scripts default to the repo id above and to the historical
`ccpo-attncred-150-20260913`; pass `--exp` for any other run and `--repo` for a
different destination.

## 8. Guards

```bash
python3 ccpo/test_arms.py              # arms, doc/code agreement, self-containment
.venv-webshop/bin/python ablations/test_ablations.py # ablation deltas and estimator paths
python3 tests/test_webshop_setup.py    # provisioning checksum and no-overwrite guards
python3 tests/test_webshop_protocol.py # 37 overrides against the published script
python3 tests/test_g2po_port.py        # our port of G2PO's node values
python3 tests/test_phi_context.py      # phi must be a function of observation AND context
```

Use the benchmark's venv for tests that import NumPy or torch (`.venv/` for
ALFWorld, `.venv-webshop/` for WebShop). The arm and ablation guards are CPU-only;
`tests/test_phi_hidden.py` and `test_phi_packed.py` require a GPU. The arm guard's
path scan is intended for the source checkout; host-generated launch scripts and
caches can contain absolute paths. Run the arm and ablation guards after changing
an arm to check its declared delta and agreement with the spec. The published-protocol
comparison additionally needs the local reference file
`baselines/G2PO/examples/g2po_trainer/run_webshop.sh`; that historical checkout is
not included in Git or required by WebShop setup itself.

## 9. Workspace and historical records

`/workspace/agent-context-grpo` is the active project on this host. Run setup,
training and Git commands from this subdirectory. The older `/workspace` checkout
is retained for history; it is not required by a fresh clone of this repository.

Local `legacy-experiments/` and `baselines/` links expose historical artifacts and
reference checkouts. They, `.local/`, environments, model weights and datasets are
excluded from Git. New runs belong in this project's `experiments/`. A fresh clone
can prepare WebShop using the setup command in §1 without those historical assets.
