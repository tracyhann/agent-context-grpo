# CCPO — Context-Conditioned Policy Optimization

Credit assignment for multi-turn agent RL. CCPO replaces the *step* term of the GRPO
family (GiGPO, G²PO) with a context-conditioned, credibility-shrunk leave-one-out
baseline, leaving the episode term and everything else identical — so a head-to-head
isolates the estimator and nothing else.

* Benchmarks: **ALFWorld**, **WebShop** · Backbones: **Qwen2.5-1.5B / 7B-Instruct**
* Target hardware: **A100 (sm_80)** and **H100 (sm_90)**
* Checkpoint backups: **https://huggingface.co/tracyhan816/ccpo-variants**

**The spec is [`experiments/experiments.md`](experiments/experiments.md)** — 21 runs,
each with its variant name, its equations, its GPU floor and the command that launches
it, plus the hyperparameter tables and the results-table format.

---

## 1. Quickstart

```bash
# 1. container  (see §2 if you need to build the image)
docker run --gpus all -it --shm-size=32g -v /path/to/ccpo:/workspace -w /workspace <image> 

# 2. environment: venv, verl-agent + our overlay, flash-attn, data, weights (~1 h cold)
scripts/setup_env.sh

# 3. a run — --gpus is required; 1.5B needs >=4 devices, 7B needs >=8
python3 ccpo/run.py --method attncred --benchmark alfworld --backbone 1.5b --gpus 0,1,2,3

# 4. watch it
python3 scripts/exp_status.py
python3 scripts/report_results.py --exp experiments/<exp-id> --paper-table
```

Add `--dry-run` to any launch to write `config.json` and `run.sh` without starting.

## 2. Container

`docker/Dockerfile` builds the image this project runs in: **vLLM 0.10.1** as the base
(CUDA 12.8, torch 2.8), plus Ray, tensordict 0.6.2, hydra, accelerate/peft, datasets,
gymnasium and ALFWorld, with `transformers` pinned by wheel.

```bash
cd docker
mkdir -p wheels                       # the build context expects it
#   place transformers-4.51.1-py3-none-any.whl here (the Dockerfile installs it
#   with --no-deps); any other local wheels are picked up via PIP_FIND_LINKS
DOCKER_BUILDKIT=1 docker build -t ccpo:cu128 .
```

Three things to know before building:

* **`COPY wheels /wheels` is not optional.** The build fails if `docker/wheels/` does
  not exist, and the last layer installs `transformers-4.51.1-py3-none-any.whl` from it.
* **The pip index is mirrored** (`PIP_INDEX_URL` → Tsinghua, with pypi.org as the extra
  index) for flaky-network hardening. Drop or repoint those `ENV` lines if that mirror
  is slow from your location.
* **No flash-attn in the image.** It is installed per-venv by `scripts/setup_env.sh`,
  which fetches `flash_attn-2.8.3.post1+cu12torch2.8cxx11abiFALSE-cp310` — a cu12 /
  torch 2.8 build that covers sm_80 and sm_90. This matters: the launchers **refuse to
  start without a working FA2 build** (§5), because the fallback silently disables
  sequence packing and costs ~2.6× per step.

The image is a runtime only — mount this repository into it. Nothing in the image is
project state.

## 3. Environment

`scripts/setup_env.sh` is idempotent and builds everything under the repo root, not
`$HOME`, so a container restart costs nothing:

1. `.venv/` with the training stack, pinned against verl-agent's requirements;
2. `verl-agent/` — upstream cloned into `baselines/`, copied out, then
   `scripts/sync_patches.sh` overlays `patches/verl-agent/` onto it;
3. flash-attn (see §2);
4. `alfworld_data/`, `hf/` (Qwen weights) and `envdata/verl_data/` (the parquets).

**`patches/verl-agent/` is the source of truth for every file we changed in the
trainer.** Edit there and re-run `scripts/sync_patches.sh`; never edit `verl-agent/`
directly — it is gitignored and gets overwritten.

**WebShop needs its own venv**, `.venv-webshop` (it imports `web_agent_site` in
process), plus `envdata/webshop_data` and a JDK for the Lucene search index. `exp_run`
selects that venv and sets `JAVA_HOME` automatically when `env_name=Webshop`.

## 4. Layout

```
ccpo/            the estimator AND the main-method arms
  core_ccpo.py     CCPO itself: gate, phi kernel, leave-one-out baseline, credibility prior
  arms.py          BASE / BENCHMARK / BACKBONE / METHODS — every arm is BASE plus a delta
  run.py           launcher for the four main variants        test_arms.py   guards
ablations/       the six ablations (plus one WebShop-only variant)
  ablations.py     run.py     test_ablations.py
scripts/         exp_run.py (training entry point), setup_env.sh, sync_patches.sh,
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
`.venv-webshop`, `alfworld_data/`, `envdata/`, `hf/`. Everything else resolves inside
this directory — `ccpo/test_arms.py` fails if any file names a path outside it.

## 5. Running experiments

```bash
python3 ccpo/run.py --list                 # the 8 main runs
python3 ablations/run.py --list            # the 13 ablation runs

python3 ccpo/run.py --method attncred --benchmark webshop --backbone 7b --gpus 0,1,2,3,4,5,6,7
python3 ablations/run.py --ablation cosine --benchmark alfworld --gpus 0,1,2,3
```

`--gpus` is required and has no default; `run.py` warns below the floor (≥4 for 1.5B,
≥8 for 7B). Arguments after the fixed flags go to `scripts/exp_run.py` verbatim and a
later `--set` wins, so `--set val_batch_size=64` adapts to a host without editing an arm.

Both launchers **preflight flash attention** on the training interpreter and refuse to
start unless `flash_attn_2_cuda` is compiled and `flash_attn_varlen_func` actually runs
on the card — an import check is not enough, because `docker/fa_stub` imports cleanly
and raises only when a kernel is called. `--allow-sdpa` overrides; a run made with it is
not comparable to the others.

Each run writes to `experiments/<exp-id>/`: `config.json` and `run.sh` (exactly what was
launched), `outputs/metrics.jsonl`, `outputs/train.log`, `outputs/checkpoints/`, and
`plots/`.

## 6. Results, plots and checkpoints

```bash
python3 scripts/report_results.py --exp experiments/<exp-id> --paper-table [--markdown]
```

prints success by task type for **both** training rollouts and the held-out split at
step 100 / best / final, plus turns per episode — the format `experiments.md` §5 defines.
`plot_metrics.py --watch` is started automatically by every launch, so
`plots/progress.png` (success, per-type success, turns, KL, entropy, …) and
`plots/ccpo.png` (the estimator's own terms) refresh while the run trains.

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
python3 ablations/test_ablations.py    # ablation deltas + the two new estimator paths
python3 tests/test_webshop_protocol.py # 37 overrides against the published script
python3 tests/test_g2po_port.py        # our port of G2PO's node values
python3 tests/test_phi_context.py      # phi must be a function of observation AND context
```

The first two are CPU-only and take seconds; `tests/test_phi_hidden.py` and
`test_phi_packed.py` want a GPU. Run the first two after any change to an arm — they are
what keeps each ablation a single-key delta from its control, and the spec in step with
the code.

## 9. Relationship to the research tree

This directory is the single source of truth for the code. The research checkout it grew
out of points at it — `ccpo/`, `scripts/`, `patches/`, `docker/` and `tests/` there are
symlinks into this tree, so there is one copy of the estimator and no opportunity for
the two to drift. A run launched from the research tree keeps that tree's root (its
venvs, data and `experiments/`); a run launched from here keeps this one's. `ACG_ROOT`
overrides the root for either.
