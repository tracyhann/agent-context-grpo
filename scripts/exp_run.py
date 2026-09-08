#!/usr/bin/env python3
"""Launch one experiment, with its configuration recorded beside its results.

Every run materialises   experiments/<name>-<YYYYMMDD>/
    config.json     the FULL resolved configuration: hydra overrides, ACG_* env,
                    git commit, package versions. Enough to rerun the experiment
                    without this script.
    run.sh          the exact command, regenerated from config.json
    outputs/        train.log, metrics.jsonl, resolved_config.json, ccpo_samples.csv,
                    checkpoints/  (stepN-best and stepN-last only)
    plots/          refreshed while the run is in flight
    NOTES.md        what this arm is for and what to look at

Arms differ ONLY in the fields named on the command line, so a comparison is a
diff of two config.json files.

Usage:
    scripts/exp_run.py --name ccpo-alfworld --arm ccpo [--set k=v ...] [--dry-run]
"""
import argparse
import datetime
import json
import os
import shlex
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# Defaults. These mirror the G2PO reference script
# (baselines/G2PO/examples/g2po_trainer/run_alfworld.sh) wherever the two can
# agree, so every arm is comparable to the published baselines by construction.
# Anything that differs from that reference is flagged in `_REFERENCE_DELTA`.
# ---------------------------------------------------------------------------
DEFAULTS = {
    "arm": "ccpo",                      # ccpo | grpo | gigpo
    "model": "Qwen/Qwen2.5-1.5B-Instruct",
    # Four, not six: verl asserts train_batch_size * rollout.n % n_gpus == 0 and the
    # reference batch of 128 does not divide by 6. Two GPUs idle is the price of
    # keeping the batch the published numbers were produced with.
    "gpus": "0,1,2,3",
    "seed": 0,

    # batch geometry -- G2PO uses 16 tasks x 8 rollouts = 128 episodes/step
    "train_batch_size": 16,
    "group_size": 8,
    "ppo_mini_batch_size": 256,
    # Two separate knobs. The reference can use 32 for both because it packs
    # sequences (use_remove_padding=True); without flash-attn we pad to 2560
    # tokens, so the same 32 OOMs in the BACKWARD pass while being fine for the
    # forward-only log-prob passes. Measured: 32 OOMs in update_actor
    # ("Tried to allocate 9.27 GiB"), 32 is fine for old_log_prob and ref.
    # ON, and every arm must keep it on or the comparison is not fair.
    #
    # With packing, a micro-batch counted in SEQUENCES has unbounded token count:
    # as responses lengthen during training the same 32 sequences carry more
    # tokens, and memory grows with them. Sampling nvidia-smi directly during the
    # update measured a per-GPU peak of 86.5 GB of 95.6 GiB -- 88% -- five steps in,
    # with responses still at their initial ~54 tokens. Budgeting TOKENS instead
    # bounds the peak whatever the responses do, which is what use_dynamic_bsz is
    # for. (verl's own perf/max_memory_reserved_gb is an aggregate across pools and
    # read 103.8 against a 102.6 GB card, so it cannot be used for this judgement;
    # the nvidia-smi peak watch in experiments/README.md can.)
    "dynamic_bsz": True,
    "ppo_max_token_len_per_gpu": 12288,        # update: forward + backward
    "log_prob_max_token_len_per_gpu": 24576,   # forward only
    # These ARE the values gigpo-repro-20260906 ran with. Defaults, not flags, so a
    # later arm launched without arguments cannot silently differ from it.
    "ppo_micro_batch_size_per_gpu": 32,        # update: forward + backward
    "log_prob_micro_batch_size_per_gpu": 64,   # forward only
    "val_data_size": 128,
    # Evaluation still covers all 128 reference episodes; verl iterates the whole
    # val dataloader. Splitting it into chunks matters because verl-agent creates
    # ONE ray actor per environment and keeps the train and validation pools alive
    # together: 128 + 128 actors put ~8165 threads against the 8192-pid cgroup
    # ceiling and every worker aborted. 128 train + 64 validation fits.
    "val_batch_size": 64,

    # lengths and sampling
    "max_prompt_length": 2048,
    "max_response_length": 512,         # G2PO reference
    "train_temperature": 1.0,
    "train_top_p": 1.0,                 # G2PO leaves these at the vllm defaults
    "train_top_k": -1,
    "val_temperature": 0.4,             # G2PO reference eval protocol
    "val_top_p": 1.0,
    "val_top_k": -1,

    # optimisation
    # Pure memory/compute trade, no effect on the math: a 1.5B model on 96 GB does
    # not need activation checkpointing, and paying ~30% extra compute for it makes
    # the update phase the long pole of a step.
    # flash-attn 2.8.3 DOES have sm_120 kernels (measured: varlen 0.10 ms/call on
    # this card), contrary to the long-standing assumption in this repo -- that was
    # true of the older flash-attn/torch pair, not this one. With it,
    # use_remove_padding packs the batch instead of padding every sequence to
    # max_prompt_length + max_response_length. Measured on step 1: prompts average
    # 549 tokens (max 1120) and responses 54, against 2560 padded -- a >4x waste,
    # and perf/mfu/actor read 0.0.
    "remove_padding": True,
    "grad_ckpt": True,
    # vLLM reserves this fraction of the card up front and holds it for the whole
    # run. It only needs KV cache for ~32 concurrent generations of <=512 tokens;
    # everything else is better left to the trainer's backward pass.
    "gpu_mem_util": 0.25,
    "lr": 1e-6,
    "kl_loss_coef": 0.01,
    "kl_loss_type": "low_var_kl",
    "gamma": 0.95,
    "step_advantage_w": 1.0,
    "adv_mode": "mean_std_norm",        # both advantage terms, same convention
    # 20, not the reference's 100-150. Arms cannot run in parallel here, so the
    # budget buys one long arm or several comparable short ones. A 2x2 over
    # estimator x memory is worth more than one finished baseline, because the
    # memory component has prior evidence of a win and has never been separated
    # from the estimator. See experiments/PLAN.md, including what a 20-step budget
    # can and cannot resolve (+/-0.048 per evaluation).
    "total_epochs": 20,
    "test_freq": 5,
    "save_freq": 5,

    # Warm-start from a previous arm's checkpoint: pass a .../global_step_N path.
    # The step counter CONTINUES from N, so total_epochs is an absolute target,
    # not an increment -- resume_from=.../global_step_20 with total_epochs=120
    # runs 100 further steps. Every other knob must match the run that produced
    # the checkpoint or the warm start is not a continuation of anything.
    "resume_from": "",

    # Dump raw grouping inputs for offline gate analysis (large; diagnostic arms only).
    "gdump": False,

    # environment
    # Ray must not size itself from nproc. This box reports 256 CPUs but the
    # cgroup allows only 8192 pids, and Ray prestarts one python worker per CPU;
    # at 256 workers thread creation fails with EAGAIN and every worker aborts
    # ("Unhandled exception: ... thread: Resource temporarily unavailable").
    # This is a SCHEDULING quantity only -- per-actor thread counts are controlled
    # by the RAY_* knobs in build_env, not by this. 64 leaves headroom for the 128
    # train and 128 validation env actors at 0.1 CPU each plus the GPU workers.
    "ray_num_cpus": 64,
    "env_name": "alfworld/AlfredTWEnv",
    "max_steps": 50,
    "history_length": 2,
    "eval_split": "eval_in_distribution",   # = valid_seen, the reference default

    # early stopping -- evaluations without a new best, never before min_steps
    "early_stop_patience": 6,
    "early_stop_min_steps": 30,

    # CCPO estimator knobs (see ccpo/core_ccpo.py)
    "ccpo_phi": "hidden",
    "ccpo_rho": 0.59,
    "ccpo_shrink": "eb",
    "ccpo_whiten": 3,
    "ccpo_target": "return",
    "ccpo_sim": 0.0,
    "ccpo_sim_backoff": 0.0,
    "ccpo_backoff_rho": 0.5,
    "ccpo_edge_w": 0.0,
    # "hard" = the GiGPO/G2PO exact (task, observation) gate. "global" makes the
    # task one bucket and lets exp(-d/tau) gate softly; ccpo_tau is then the
    # kernel width as a multiple of the median phi-distance and is the parameter
    # that matters. Under "global" the shrinkage is bypassed (lam = 1) because
    # b_obs degenerates to the uniform task mean, which measures WORSE than the
    # hard gate -- see H-M in experiments/hypothesis.md.
    "ccpo_gate": "hard",
    "ccpo_tau": 1.0,
    # "task" = divide the step credit by the per-task sd in the trainer (default).
    # "local" = divide by the phi-weighted sd over the same soft neighbourhood that
    # produced the baseline. Set ccpo_step_norm=none alongside it or the advantage is
    # standardised twice. See H-V in experiments/hypothesis.md.
    "ccpo_std": "task",
    "ccpo_std_floor": 0.25,
    "ccpo_step_norm": "mode",

    # supporting components, both off = stock protocol
    "compact_budget": 0,
    # Gate the digest on the agent being stuck: inject it only when `stall` (turns
    # since the last NEW observation) reaches this. 0 = never gate, i.e. every turn,
    # which is what ccpo-memory-20260907 ran and what cost it action validity.
    "compact_stall": 0,
    # prepend (default) stacks the digest ON the recent window; replace substitutes it.
    # Prepend duplicates -- build_digest already covers the window's turns -- and cost
    # +194 prompt AND +6 response tokens at 15/15 paired steps (H-AA). See
    # tests/test_compact_mode.py for the budget/coverage sweep behind the chosen value.
    "compact_mode": "prepend",
    "force_budget": 0,
    "force_tail": 32,
}

_REFERENCE_DELTA = [
    "base model: Qwen2.5-1.5B-Instruct matches the G2PO/GiGPO/HGPO reference",
    "attention: flash-attn 2.8.3 DOES build for sm_120; trainer runs flash_attention_2\n     with use_remove_padding=True (packed). Falls back to sdpa if the import fails.",
    "rollout: vllm with VLLM_ATTENTION_BACKEND=TRITON_ATTN (Blackwell)",
    "tensor_model_parallel_size=1 on 6 GPUs vs their 8 with tp=2",
]

ENV_KEYS = {
    "ccpo_phi": "ACG_CCPO_PHI", "ccpo_rho": "ACG_CCPO_RHO",
    "ccpo_shrink": "ACG_CCPO_SHRINK", "ccpo_whiten": "ACG_CCPO_WHITEN",
    "ccpo_target": "ACG_CCPO_TARGET", "ccpo_sim": "ACG_CCPO_SIM",
    "ccpo_sim_backoff": "ACG_CCPO_SIM_BACKOFF",
    "ccpo_backoff_rho": "ACG_CCPO_BACKOFF_RHO", "ccpo_edge_w": "ACG_CCPO_EDGE_W",
    "ccpo_gate": "ACG_CCPO_GATE", "ccpo_tau": "ACG_CCPO_TAU",
    "ccpo_std": "ACG_CCPO_STD", "ccpo_std_floor": "ACG_CCPO_STD_FLOOR",
    "ccpo_step_norm": "ACG_CCPO_STEP_NORM",
    "compact_budget": "ACG_COMPACT_BUDGET", "compact_stall": "ACG_COMPACT_STALL",
    "compact_mode": "ACG_COMPACT_MODE", "force_budget": "ACG_FORCE_BUDGET",
    "force_tail": "ACG_FORCE_TAIL",
    "early_stop_patience": "ACG_EARLY_STOP_PATIENCE",
    "early_stop_min_steps": "ACG_EARLY_STOP_MIN_STEPS",
}


def _coerce(v):
    for cast in (int, float):
        try:
            return cast(v)
        except (TypeError, ValueError):
            pass
    if isinstance(v, str) and v.lower() in ("true", "false"):
        return v.lower() == "true"
    return v


def _have_flash_attn(python):
    """A real flash-attn with kernels for this card, not the import stub."""
    try:
        r = subprocess.run([python, "-c", "import flash_attn, sys; sys.exit(0)"],
                           capture_output=True, timeout=120)
        return r.returncode == 0
    except Exception:                                        # noqa: BLE001
        return False


def git_state():
    def _run(*a):
        try:
            return subprocess.run(a, cwd=ROOT, capture_output=True, text=True).stdout.strip()
        except OSError:
            return ""
    return {"commit": _run("git", "rev-parse", "HEAD"),
            "branch": _run("git", "rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(_run("git", "status", "--porcelain"))}


def versions(python):
    code = ("import json,torch,transformers,vllm,ray;"
            "print(json.dumps({'torch':torch.__version__,'cuda':torch.version.cuda,"
            "'transformers':transformers.__version__,'vllm':vllm.__version__,'ray':ray.__version__}))")
    try:
        out = subprocess.run([python, "-c", code], capture_output=True, text=True, timeout=180)
        return json.loads(out.stdout.strip().splitlines()[-1])
    except Exception as e:                                   # noqa: BLE001
        return {"error": str(e)}


def build_command(cfg, exp_dir):
    ngpu = len(cfg["gpus"].split(","))
    est = {"ccpo": "ccpo", "grpo": "grpo", "gigpo": "gigpo"}[cfg["arm"]]
    ckpt = os.path.join(exp_dir, "outputs", "checkpoints")
    args = [
        "python3", "-m", "verl.trainer.main_ppo",
        f"algorithm.adv_estimator={est}",
        f"data.train_files={cfg['data_dir']}/text/train.parquet",
        f"data.val_files={cfg['data_dir']}/text/test.parquet",
        f"data.train_batch_size={cfg['train_batch_size']}",
        f"data.val_batch_size={cfg['val_batch_size']}",
        f"data.max_prompt_length={cfg['max_prompt_length']}",
        f"data.max_response_length={cfg['max_response_length']}",
        "data.filter_overlong_prompts=True",
        "data.truncation=error",
        "data.return_raw_chat=True",
        f"actor_rollout_ref.model.path={cfg['model_path']}",
        f"actor_rollout_ref.actor.optim.lr={cfg['lr']}",
        f"actor_rollout_ref.model.use_remove_padding={cfg['remove_padding']}",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={cfg['ppo_mini_batch_size']}",
        f"actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu={cfg['ppo_micro_batch_size_per_gpu']}",
        f"actor_rollout_ref.actor.use_dynamic_bsz={cfg['dynamic_bsz']}",
        f"actor_rollout_ref.actor.ppo_max_token_len_per_gpu={cfg['ppo_max_token_len_per_gpu']}",
        f"actor_rollout_ref.ref.log_prob_max_token_len_per_gpu={cfg['log_prob_max_token_len_per_gpu']}",
        f"actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu={cfg['log_prob_max_token_len_per_gpu']}",
        "actor_rollout_ref.actor.use_kl_loss=True",
        f"actor_rollout_ref.actor.kl_loss_coef={cfg['kl_loss_coef']}",
        f"actor_rollout_ref.actor.kl_loss_type={cfg['kl_loss_type']}",
        f"actor_rollout_ref.model.enable_gradient_checkpointing={cfg['grad_ckpt']}",
        "actor_rollout_ref.actor.fsdp_config.param_offload=False",
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload=False",
        f"actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu={cfg['log_prob_micro_batch_size_per_gpu']}",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
        "actor_rollout_ref.rollout.name=vllm",
        f"actor_rollout_ref.rollout.gpu_memory_utilization={cfg['gpu_mem_util']}",
        "actor_rollout_ref.rollout.enable_chunked_prefill=False",
        "actor_rollout_ref.rollout.enforce_eager=False",
        "actor_rollout_ref.rollout.free_cache_engine=False",
        f"actor_rollout_ref.rollout.temperature={cfg['train_temperature']}",
        f"actor_rollout_ref.rollout.top_p={cfg['train_top_p']}",
        f"actor_rollout_ref.rollout.top_k={cfg['train_top_k']}",
        f"actor_rollout_ref.rollout.val_kwargs.temperature={cfg['val_temperature']}",
        f"actor_rollout_ref.rollout.val_kwargs.top_p={cfg['val_top_p']}",
        f"actor_rollout_ref.rollout.val_kwargs.top_k={cfg['val_top_k']}",
        "actor_rollout_ref.rollout.val_kwargs.do_sample=True",
        f"actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu={cfg['log_prob_micro_batch_size_per_gpu']}",
        "actor_rollout_ref.ref.fsdp_config.param_offload=True",
        "actor_rollout_ref.actor.use_invalid_action_penalty=True",
        "actor_rollout_ref.actor.invalid_action_penalty_coef=0.1",
        "algorithm.use_kl_in_reward=False",
        f"algorithm.gamma={cfg['gamma']}",
        f"algorithm.gigpo.step_advantage_w={cfg['step_advantage_w']}",
        f"algorithm.gigpo.mode={cfg['adv_mode']}",
        f"env.env_name={cfg['env_name']}",
        f"env.seed={cfg['seed']}",
        f"env.max_steps={cfg['max_steps']}",
        f"env.history_length={cfg['history_length']}",
        f"env.rollout.n={cfg['group_size']}",
        f"env.alfworld.eval_dataset={cfg['eval_split']}",
        "env.resources_per_worker.num_cpus=0.1",
        f"ray_init.num_cpus={cfg['ray_num_cpus']}",
        "trainer.critic_warmup=0",
        "trainer.logger=[console,jsonl]",
        "trainer.project_name=ccpo_alfworld",
        f"trainer.experiment_name={cfg['exp_id']}",
        f"trainer.n_gpus_per_node={ngpu}",
        "trainer.nnodes=1",
        f"trainer.save_freq={cfg['save_freq']}",
        "actor_rollout_ref.actor.checkpoint.contents=[model,optimizer,extra,hf_model]",
        f"trainer.default_local_dir={ckpt}",
        f"trainer.test_freq={cfg['test_freq']}",
        f"trainer.total_epochs={cfg['total_epochs']}",
        "trainer.val_before_train=False",
    ]
    if cfg.get("resume_from"):
        args += [
            "trainer.resume_mode=resume_path",
            f"trainer.resume_from_path={cfg['resume_from']}",
        ]
    else:
        # 'auto' would silently pick up a checkpoint left in this run's own dir,
        # which turns a rerun of a named arm into an undeclared continuation.
        args.append("trainer.resume_mode=disable")
    return args


def build_env(cfg, exp_dir):
    env = {
        "CUDA_VISIBLE_DEVICES": cfg["gpus"],
        "ALFWORLD_DATA": cfg["alfworld_data"],
        "HF_HOME": cfg["hf_home"],
        # docker/fa_stub only satisfies the import when no real flash-attn is
        # installed; it must NOT shadow a working one, so it is appended only as a
        # fallback. PYTHONPATH precedes site-packages.
        "PYTHONPATH": (f"{ROOT}/verl-agent:{ROOT}" if _have_flash_attn(cfg["venv_python"])
                       else f"{ROOT}/verl-agent:{ROOT}:{ROOT}/docker/fa_stub"),
        # Blackwell (sm_120): flash-attn 2.8.3 does build here and the packed path
        # is ~2.6x faster per step, so prefer it; sdpa remains the fallback and
        # vllm runs its Triton attention kernels, which JIT per-arch.
        "VERL_ATTN_IMPL": ("flash_attention_2" if _have_flash_attn(cfg["venv_python"]) else "sdpa"),
        "VLLM_ATTENTION_BACKEND": "TRITON_ATTN",
        "TRITON_PTXAS_PATH": "/usr/local/cuda/bin/ptxas",
        "VLLM_USE_FLASHINFER_SAMPLER": "0",
        # NOT expandable_segments: vLLM's memory pool (sleep/wake between rollout
        # and training) asserts against it -- "Expandable segments are not
        # compatible with memory pool", pytorch#147851. That setting came from the
        # old HF-rollout path, where it countered long-run fragmentation.
        "RAY_TMPDIR": "/tmp/ray_acg",
        "TOKENIZERS_PARALLELISM": "false",
        # Persist vLLM's torch.compile / CUDA-graph cache on the data volume. Cold,
        # the first generate() call compiles and captures graphs for every engine
        # and takes minutes with the GPUs pinned at 100% and nothing in the log --
        # which reads exactly like a hang. Warm, engine start is ~16s. Measured
        # generation itself is 6127 tok/s per GPU on TRITON_ATTN, so throughput was
        # never the problem.
        "VLLM_CACHE_ROOT": os.path.join(ROOT, ".cache", "vllm"),
        "TORCHINDUCTOR_CACHE_DIR": os.path.join(ROOT, ".cache", "inductor"),
        # belt and braces against the pid ceiling described at ray_num_cpus
        "RAY_num_prestart_python_workers": "4",
        # verl-agent gives every ALFWorld environment its own ray actor, so a
        # reference-protocol batch is 128 train + 128 validation actors. Untuned,
        # each carries ~115 threads of gRPC and asio pools sized from nproc (256
        # here) -- about 29k threads against a cgroup ceiling of 8192, which shows
        # up as every worker aborting with "thread: Resource temporarily
        # unavailable". These three knobs take an actor to ~24 threads (measured);
        # the rest of the vars stop numpy/BLAS/tokenizers opening their own pools.
        "RAY_num_server_call_thread": "1",
        "RAY_num_grpc_internal_threads": "1",
        "RAY_object_manager_rpc_threads_num": "1",
        "RAY_event_stats": "0",
        "RAY_start_python_gc_manager_thread": "0",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "RAYON_NUM_THREADS": "1",
        "ACG_ADV_ESTIMATOR": cfg["arm"],
        "ACG_EXP_DIR": exp_dir,
        "ACG_METRICS_JSONL": os.path.join(exp_dir, "outputs", "metrics.jsonl"),
        "ACG_CCPO_DUMP": os.path.join(exp_dir, "outputs", "ccpo_samples.csv"),
        # Raw gate inputs (full observation text) for offline re-bucketing. Off by
        # default: ~100x the per-step volume of the CSV. Set gdump=True for a short
        # diagnostic arm whose only purpose is to characterise the grouping.
        **({"ACG_CCPO_GDUMP": os.path.join(exp_dir, "outputs", "grouping.jsonl")}
           if cfg.get("gdump") else {}),
    }
    for k, e in ENV_KEYS.items():
        env[e] = str(cfg[k])
    return env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--arm", default=None, choices=["ccpo", "grpo", "gigpo"])
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--date", default=datetime.date.today().strftime("%Y%m%d"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-plot", action="store_true")
    a = ap.parse_args()

    cfg = dict(DEFAULTS)
    if a.arm:
        cfg["arm"] = a.arm
    for kv in a.set:
        k, _, v = kv.partition("=")
        if k not in cfg:
            sys.exit(f"unknown config key: {k}\nknown: {', '.join(sorted(cfg))}")
        cfg[k] = _coerce(v)

    cfg["exp_id"] = f"{a.name}-{a.date}"
    exp_dir = os.path.join(ROOT, "experiments", cfg["exp_id"])
    cfg.update({
        "data_dir": os.environ.get("ACG_DATA_DIR", "/workspace/envdata/verl_data"),
        "alfworld_data": os.environ.get("ALFWORLD_DATA", "/workspace/alfworld_data"),
        "hf_home": os.environ.get("HF_HOME", "/workspace/hf"),
        "venv_python": os.path.join(ROOT, ".venv", "bin", "python3"),
    })
    snap = os.path.join(cfg["hf_home"], "hub",
                        "models--" + cfg["model"].replace("/", "--"), "snapshots")
    cfg["model_path"] = (os.path.join(snap, sorted(os.listdir(snap))[0])
                         if os.path.isdir(snap) else cfg["model"])

    for sub in ("outputs", "plots"):
        os.makedirs(os.path.join(exp_dir, sub), exist_ok=True)

    argv = build_command(cfg, exp_dir)
    env = build_env(cfg, exp_dir)
    record = {
        "experiment": cfg["exp_id"], "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "config": cfg, "hydra_overrides": argv[3:], "env": env,
        "git": git_state(), "versions": versions(cfg["venv_python"]),
        "reference_protocol": {
            "source": "baselines/G2PO/examples/g2po_trainer/run_alfworld.sh",
            "matched": ["val temperature 0.4 + do_sample", "val set 128 episodes (in chunks of val_batch_size)",
                        "eval split eval_in_distribution (valid_seen)",
                        "group size 8", "train_batch_size 16", "max_prompt_length 2048",
                        "max_response_length 512", "lr 1e-6", "kl 0.01 low_var_kl",
                        "gamma 0.95", "step_advantage_w 1.0", "mode mean_std_norm",
                        "invalid-action penalty 0.1", "env.max_steps 50", "env.seed 0",
                        "history_length 2"],
            "deltas": _REFERENCE_DELTA,
        },
    }
    with open(os.path.join(exp_dir, "config.json"), "w") as fh:
        json.dump(record, fh, indent=2, sort_keys=True)

    envline = " \\\n  ".join(f"{k}={shlex.quote(str(v))}" for k, v in sorted(env.items()))
    runsh = ("#!/bin/bash\n# Regenerated from config.json by scripts/exp_run.py -- do not hand-edit.\n"
             "set -x\ncd %s\nexport \\\n  %s\nexec %s %s\n"
             % (shlex.quote(os.path.join(ROOT, "verl-agent")), envline,
                shlex.quote(cfg["venv_python"]), " \\\n  ".join(shlex.quote(x) for x in argv[1:])))
    run_path = os.path.join(exp_dir, "run.sh")
    with open(run_path, "w") as fh:
        fh.write(runsh)
    os.chmod(run_path, 0o755)

    notes = os.path.join(exp_dir, "NOTES.md")
    if not os.path.exists(notes):
        tmpl = os.path.join(ROOT, "experiments", "_template", "NOTES.md")
        body = open(tmpl).read() if os.path.exists(tmpl) else "# <exp-id>\n"
        with open(notes, "w") as fh:
            fh.write(body.replace("<exp-id>", cfg["exp_id"], 1))

    print(f"[exp] {cfg['exp_id']}  arm={cfg['arm']}  gpus={cfg['gpus']}")
    print(f"[exp] dir     {exp_dir}")
    print(f"[exp] config  {exp_dir}/config.json")
    if a.dry_run:
        print("[exp] dry run; not launching")
        return

    log = os.path.join(exp_dir, "outputs", "train.log")
    full = dict(os.environ, **env)
    with open(log, "ab", buffering=0) as fh:
        # start_new_session detaches the run from the launching shell's process
        # group, so a training run that takes hours is not killed by whatever
        # started it going away.
        p = subprocess.Popen([cfg["venv_python"]] + argv[1:], cwd=os.path.join(ROOT, "verl-agent"),
                             env=full, stdout=fh, stderr=subprocess.STDOUT,
                             start_new_session=True)
    with open(os.path.join(exp_dir, "outputs", "train.pid"), "w") as fh:
        fh.write(str(p.pid))
    print(f"[exp] launched pid {p.pid}, log -> {log}")
    if not a.no_plot:
        # Kill any watcher left over from an earlier launch. They are long-lived and
        # accumulate one per launch; twelve of them once helped exhaust the
        # container's pid budget mid-run, which looks like a training crash.
        subprocess.run(["pkill", "-9", "-f", f"plot_metrics.py --exp {exp_dir}"],
                       capture_output=True)
        subprocess.Popen([cfg["venv_python"], os.path.join(ROOT, "scripts", "plot_metrics.py"),
                          "--exp", exp_dir, "--watch"],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         start_new_session=True)
        print("[exp] plot watcher started")


if __name__ == "__main__":
    main()
