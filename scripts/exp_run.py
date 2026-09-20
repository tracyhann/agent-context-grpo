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
import hashlib
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
    # vLLM tensor parallelism. The published scripts use tp=2 on 8 GPUs, which is a
    # parallelism choice, not a learning hyperparameter: at tp=1 each GPU holds a
    # whole 1.5B replica and the rollout runs n_gpus generation streams instead of
    # one. On two GPUs tp=2 would halve the number of replicas, so tp=1 stays the
    # default and the delta is recorded in _REFERENCE_DELTA.
    "tp_size": 1,
    # vLLM's attention backend. TRITON_ATTN was pinned for Blackwell (sm_120);
    # on A100 (sm_80) FLASH_ATTN is the mature path and generation is ~50% of the
    # step, so this is worth measuring rather than assuming. Configurable so the
    # two can be A/B'd as a single-flag ablation, like every other option here.
    "vllm_attn_backend": "TRITON_ATTN",
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
    # Evaluate resume_from and exit. Pair with align_val_on_resume=0.
    "val_only": 0,
    # 0 disables the resume validation-draw alignment; required when comparing
    # checkpoints saved at different steps on one common draw.
    "align_val_on_resume": 1,
    # 0 disables the resume TRAINING-draw alignment. On resume the env workers
    # replay their game list from the start, so a resumed run re-trains on games it
    # has already seen; the alignment burns one env reset per completed step, which
    # is what the rollout loop consumes. Leave at 1 -- set 0 only to reproduce a run
    # made before this existed, and say so in the arm's NOTES.
    "align_train_on_resume": 1,

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
    # CPU units reserved per env worker. ALFWorld runs at verl-agent's default 0.1;
    # WebShop's published script lowers it to 0.05 because it starts
    # train_batch_size*group_n + val_batch_size = 256 workers at once.
    "env_num_cpus": 0.1,
    "env_name": "alfworld/AlfredTWEnv",
    # Search-augmented QA (Search-R1 suite via verl-agent's `search` env). Selected by
    # env_name=search; the keys below are ignored for every other benchmark. GiGPO's
    # published protocol (examples/gigpo_trainer/run_search.sh): group 5, 4 turns,
    # history 4, prompt 4096, KL 0.001, invalid-action penalty 0.01, and -- critically --
    # similarity grouping at 0.9, because retrieved-passage observations never repeat
    # verbatim and exact-match grouping is degenerate on this benchmark.
    "search_train_files": os.path.join(ROOT, "envdata", "searchr1", "train.parquet"),
    "search_val_files": os.path.join(ROOT, "envdata", "searchr1", "test.parquet"),
    "search_url": "http://127.0.0.1:8000/retrieve",
    "search_topk": 3,

    # WebShop (verl-agent's `Webshop` env), selected by env_name=Webshop; ignored
    # elsewhere. GiGPO's published protocol (examples/gigpo_trainer/run_webshop.sh):
    # group 8, 15 turns, batch 16, val 128, gamma 0.95, XFORMERS attention. Products
    # come from the full 1.18M-item BM25 index; the parquet only encodes modality and
    # data size, so WebShop gets its own dir prepared at 16 train / 256 val.
    "webshop_data_dir": os.path.join(ROOT, "envdata", "webshop_data"),
    # use_small picks the 1,000-product catalogue. It is verl-agent's default and what
    # run_webshop.sh actually runs, so GiGPO's published 67.4% (Qwen2.5-1.5B) is on this
    # subset. It is also the only feasible setting here: load_products json.loads the
    # whole 5.2 GB items_shuffle.json before truncating, so a full-catalogue worker peaks
    # at 19.0 GB regardless of num_products, and WebShop starts train_batch_size*group_n
    # + val_batch_size workers concurrently (128 + 128) against a 256 GiB cgroup.
    # NOTE: num_products is hardcoded None upstream, so init_search_engine always selects
    # `indexes`; that directory must therefore hold the SAME product set as use_small.
    "webshop_use_small": 1,
    "webshop_human_goals": 0,
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
    # What the STEP channel predicts. "return" = gamma-discounted return-to-go of
    # the env reward (the shipped default); "nextnode" = G2PO's successor value;
    # "score" = WebShop's dense partial score, which only that benchmark supplies
    # (info['task_score']). "score" exists because WebShop's reward reaching the
    # trainer is binary 10/0, leaving 30-69% of task groups with zero advantage on
    # every turn -- see ccpo/core_ccpo.dense_step_returns. It changes the target
    # only: the gate, the attention readout, the credibility prior, the J=0
    # fallback and the episode term are all unchanged.
    "ccpo_target": "return",
    # Mixing ratio for the context block when phi=hidden+ctx: both blocks are
    # L2-normalised first, so 1.0 is a true 50/50 and 0.0 leaves the whitened hidden
    # state alone. 0.0 is the designed null control for the context features
    # (n_unique, revisit, progress) -- the episode-level signal the prompt cannot
    # carry. Only meaningful with the readout ON (ccpo_lam_fix=1.0); under the EB
    # rule phi is computed and discarded, so this is a no-op there.
    "ccpo_ctx_w": 1.0,
    "ccpo_sim": 0.0,
    "ccpo_sim_backoff": 0.0,
    "ccpo_backoff_rho": 0.5,
    "ccpo_backoff_task": 0,     # 1: hard-gate rows with no sibling fall back to the task baseline
    "ccpo_jweight_c": 0.0,      # >0: step term *= J/(J+c); c=1 is the derived value. 0 = off
    "ccpo_prior_kappa": 0.0,    # >0: hard-gate node baseline shrunk to the task mean, weight kappa/(J+kappa)
    # Pin the credibility weight instead of deriving it from support J. "" keeps
    # lam_k = J/(J+kappa); "0.5" blends node baseline and task prior evenly on every
    # occurrence. Separates "the prior exists" from "the prior is weighted by evidence".
    "ccpo_lk_fix": "",
    "ccpo_lam_fix": "",
    # neighbour weighting: "soft" = phi kernel exp(-d/tau); "hard" = 0/1 gate at the
    # same tau. Ablates ordering-by-similarity against mere neighbourhood restriction.
    # Weight on the episode-level advantage A^EP. 1.0 ships (GiGPO/G2PO shape);
    # 0.0 removes the trajectory-level term, leaving the step term -- which already
    # carries the edge contribution -- as the whole advantage. That is HGPO's shape.
    "ccpo_ep_w": 1.0,
    "ccpo_wmode": "soft",         # "1.0": use the phi-weighted (attention) readout instead of the EB-estimated lam
    "ccpo_edge_w": 0.0,
    "ccpo_progress_horizon": 0,  # 1/2 enables contextual endpoint value progress
    "ccpo_progress_weight": 1.0,
    "ccpo_progress_history_weight": 1.0,  # 0 isolates future progress; progress_weight=0 isolates history
    "ccpo_progress_snapshot_every": 1,
    "ccpo_fixed_anchor": 0,
    "ccpo_fixed_horizon": 2,
    "ccpo_fixed_gain_weight": 1.0,
    "ccpo_fixed_max_prompt": 8192,
    "ccpo_fixed_snapshot_every": 1,
    "ccpo_outlook_horizon": 0,   # 0 preserves the existing estimator
    "ccpo_outlook_beta": 0.0,    # convex weight on the n-step outlook advantage
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
    # G2PO's anchor repair (H-AD). 0 = this tree's historical behaviour, which is also
    # verl-agent/GiGPO's. 1 = reproduce G2PO. The anchor is the state identity used for
    # node grouping in the step term, NOT the prompt.
    "obs_repair": 0,
    # Fold the admissible-action list into the anchor, as G2PO does. This tree has
    # carried that list as `aff` since 2026-09-03 -- because 42.8% of observations map
    # to >1 admissible set -- but `aff` only ever reached the diagnostic CSV, never the
    # node key. Set BOTH this and obs_repair to reproduce G2PO's anchor exactly.
    "anchor_aff": 0,
    "force_budget": 0,
    "force_tail": 32,
    # rolling global_step checkpoints kept after each save (best-* is kept separately)
    "keep_ckpts": 2,
    # comma-separated global steps to pin against rolling pruning, e.g. "100". Each is
    # hardlinked to step<N>-pin at save time (same cp -al trick as best-*), so it costs
    # no extra disk until the rolling copy is pruned. To resume or evaluate from one:
    #   cp -al <ckpts>/step100-pin <ckpts>/global_step_100
    # (the resume path must contain "global_step_"; verl asserts on it).
    "pin_steps": "",
}

_REFERENCE_DELTA = [
    "base model: Qwen2.5-1.5B-Instruct matches the G2PO/GiGPO/HGPO reference",
    "attention: flash-attn 2.8.3 DOES build for sm_120; trainer runs flash_attention_2\n     with use_remove_padding=True (packed). Falls back to sdpa if the import fails.",
    "rollout: vllm, VLLM_ATTENTION_BACKEND set by the vllm_attn_backend key",
    "tensor_model_parallel_size=1 on this box's GPUs vs their 8 with tp=2",
]

# ---------------------------------------------------------------------------
# Per-benchmark protocol. DEFAULTS above are the ALFWorld reference
# (baselines/G2PO/examples/g2po_trainer/run_alfworld.sh). WebShop's published
# script differs in more than the env name, and those differences are protocol
# rather than taste, so env_name=Webshop applies them automatically -- unless the
# command line set the key, which always wins.
# Source: baselines/G2PO/examples/g2po_trainer/run_webshop.sh (identical on every
# one of these to GiGPO's examples/gigpo_trainer/run_webshop.sh).
# ---------------------------------------------------------------------------
WEBSHOP_PROTOCOL = {
    # WebShop pages are long: the reference doubles the prompt budget.
    "max_prompt_length": 4096,
    # val_data_size=128 in the reference; the parquet is prepared at 2x that.
    "val_batch_size": 128,
    # 15 turns, not ALFWorld's 50.
    "max_steps": 15,
    # mean_norm on WebShop vs mean_std_norm on ALFWorld. This one is an estimator
    # setting, so it changes the update, not just the budget.
    "adv_mode": "mean_norm",
    "ppo_mini_batch_size": 64,
    # Inert while dynamic_bsz is on (the token budgets below govern), kept at the
    # reference values so turning dynamic_bsz off reproduces the published script.
    "ppo_micro_batch_size_per_gpu": 8,
    "log_prob_micro_batch_size_per_gpu": 16,
    "env_num_cpus": 0.05,
}

_WEBSHOP_DELTA = [
    "WebShop protocol from run_webshop.sh: prompt 4096, val 128, 15 turns,\n     mean_norm, mini-batch 64, 0.05 CPU/worker -- see WEBSHOP_PROTOCOL",
    "gpu_memory_utilization stays 0.25 (theirs 0.6): they hold 8 cards with tp=2,\n     this box trains and generates on the same two, and the actor peaked at 86.5 GB\n     of 95.6 GiB on ALFWorld",
    "use_dynamic_bsz=True replaces their fixed micro-batches; the token budgets are\n     memory-calibrated for this box and bound peak memory as prompts lengthen",
    "1,000-product catalogue (env.webshop.use_small=True) -- verl-agent's default and\n     what the published scripts run, so 67.4% (GiGPO, Qwen2.5-1.5B) is on this subset",
]

ENV_KEYS = {
    "ccpo_phi": "ACG_CCPO_PHI", "ccpo_rho": "ACG_CCPO_RHO",
    "ccpo_shrink": "ACG_CCPO_SHRINK", "ccpo_whiten": "ACG_CCPO_WHITEN",
    "ccpo_target": "ACG_CCPO_TARGET", "ccpo_sim": "ACG_CCPO_SIM",
    "ccpo_sim_backoff": "ACG_CCPO_SIM_BACKOFF",
    "ccpo_backoff_rho": "ACG_CCPO_BACKOFF_RHO", "ccpo_edge_w": "ACG_CCPO_EDGE_W",
    "ccpo_backoff_task": "ACG_CCPO_BACKOFF_TASK", "ccpo_jweight_c": "ACG_CCPO_JWEIGHT_C",
    "ccpo_prior_kappa": "ACG_CCPO_PRIOR_KAPPA", "keep_ckpts": "ACG_KEEP_CKPTS",
    "pin_steps": "ACG_PIN_STEPS",
    "ccpo_lam_fix": "ACG_CCPO_LAM_FIX", "ccpo_wmode": "ACG_CCPO_WMODE", "ccpo_ep_w": "ACG_CCPO_EP_W", "ccpo_ctx_w": "ACG_CCPO_CTX_W",
    "ccpo_lk_fix": "ACG_CCPO_LK_FIX",
    "ccpo_gate": "ACG_CCPO_GATE", "ccpo_tau": "ACG_CCPO_TAU",
    "ccpo_std": "ACG_CCPO_STD", "ccpo_std_floor": "ACG_CCPO_STD_FLOOR",
    "ccpo_step_norm": "ACG_CCPO_STEP_NORM",
    "ccpo_progress_horizon": "ACG_CCPO_PROGRESS_HORIZON",
    "ccpo_progress_weight": "ACG_CCPO_PROGRESS_WEIGHT",
    "ccpo_progress_history_weight": "ACG_CCPO_PROGRESS_HISTORY_WEIGHT",
    "ccpo_progress_snapshot_every": "ACG_CCPO_PROGRESS_SNAPSHOT_EVERY",
    "ccpo_fixed_anchor": "ACG_CCPO_FIXED_ANCHOR",
    "ccpo_fixed_horizon": "ACG_CCPO_FIXED_HORIZON",
    "ccpo_fixed_gain_weight": "ACG_CCPO_FIXED_GAIN_WEIGHT",
    "ccpo_fixed_max_prompt": "ACG_CCPO_FIXED_MAX_PROMPT",
    "ccpo_fixed_snapshot_every": "ACG_CCPO_FIXED_SNAPSHOT_EVERY",
    "ccpo_outlook_horizon": "ACG_CCPO_OUTLOOK_HORIZON",
    "ccpo_outlook_beta": "ACG_CCPO_OUTLOOK_BETA",
    "compact_budget": "ACG_COMPACT_BUDGET", "compact_stall": "ACG_COMPACT_STALL",
    "compact_mode": "ACG_COMPACT_MODE", "obs_repair": "ACG_OBS_REPAIR", "anchor_aff": "ACG_ANCHOR_AFF",
    "align_val_on_resume": "ACG_ALIGN_VAL_ON_RESUME",
    "align_train_on_resume": "ACG_ALIGN_TRAIN_ON_RESUME", "force_budget": "ACG_FORCE_BUDGET",
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


def validate_future_progress_config(cfg):
    """Reject unsupported feature modes before setup, model probes or launch."""
    if cfg.get("arm", "ccpo") != "ccpo":
        return
    horizon = cfg.get("ccpo_progress_horizon", 0)
    if horizon not in (0, 1, 2):
        raise ValueError("ccpo_progress_horizon must be 0, 1 or 2")
    import math
    for key in ("ccpo_progress_history_weight", "ccpo_progress_weight"):
        value = float(cfg.get(key, 1.0))
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{key} must be finite and nonnegative")
    if not horizon and float(cfg.get("ccpo_progress_history_weight", 1.0)) != 1.0:
        raise ValueError("History component ablation requires ccpo_progress_horizon=1 or 2")
    if horizon:
        episode_weight = float(cfg.get("ccpo_ep_w", 1.0))
        if not math.isfinite(episode_weight) or episode_weight < 0:
            raise ValueError("Future progress requires finite nonnegative ccpo_ep_w")
        if float(cfg.get("step_advantage_w", 1.0)) != 1.0:
            raise ValueError("Prepared future-progress arms require step_advantage_w=1")
    if horizon and str(cfg.get("ccpo_phi", "hidden")).lower() not in ("hidden", "hidden+ctx"):
        raise ValueError("Future progress requires ccpo_phi=hidden or hidden+ctx; "
                         "hidden-only also supports hidden+ctx with ccpo_ctx_w=0")


def build_command(cfg, exp_dir):
    validate_future_progress_config(cfg)
    ngpu = len(cfg["gpus"].split(","))
    est = {"ccpo": "ccpo", "grpo": "grpo", "gigpo": "gigpo"}[cfg["arm"]]
    _is_search = "search" in str(cfg["env_name"]).lower()
    _is_webshop = "webshop" in str(cfg["env_name"]).lower()
    if _is_search:
        _train_f, _val_f = cfg["search_train_files"], cfg["search_val_files"]
    elif _is_webshop:
        _train_f = f"{cfg['webshop_data_dir']}/text/train.parquet"
        _val_f = f"{cfg['webshop_data_dir']}/text/test.parquet"
    else:
        _train_f = f"{cfg['data_dir']}/text/train.parquet"
        _val_f = f"{cfg['data_dir']}/text/test.parquet"
    ckpt = os.path.join(exp_dir, "outputs", "checkpoints")
    args = [
        "python3", "-m", "verl.trainer.main_ppo",
        f"algorithm.adv_estimator={est}",
        f"data.train_files={_train_f}",
        f"data.val_files={_val_f}",
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
        f"actor_rollout_ref.rollout.tensor_model_parallel_size={cfg['tp_size']}",
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
        *([f"env.search.search_url={cfg['search_url']}",
           f"env.search.topk={cfg['search_topk']}"]
          if _is_search
          else [f"env.webshop.use_small={bool(int(cfg['webshop_use_small']))}",
                f"env.webshop.human_goals={bool(int(cfg['webshop_human_goals']))}"] if _is_webshop
          else [f"env.alfworld.eval_dataset={cfg['eval_split']}"]),
        f"env.resources_per_worker.num_cpus={cfg['env_num_cpus']}",
        f"ray_init.num_cpus={cfg['ray_num_cpus']}",
        "trainer.critic_warmup=0",
        "trainer.logger=[console,jsonl]",
        f"trainer.project_name=ccpo_{'search' if _is_search else 'webshop' if _is_webshop else 'alfworld'}",
        f"trainer.experiment_name={cfg['exp_id']}",
        f"trainer.n_gpus_per_node={ngpu}",
        "trainer.nnodes=1",
        f"trainer.save_freq={cfg['save_freq']}",
        "actor_rollout_ref.actor.checkpoint.contents=[model,optimizer,extra,hf_model]",
        f"trainer.default_local_dir={ckpt}",
        f"trainer.test_freq={cfg['test_freq']}",
        f"trainer.total_epochs={cfg['total_epochs']}",
        f"trainer.val_before_train={bool(cfg.get('val_only'))}",
    ]
    if cfg.get("val_only"):
        # Evaluate a checkpoint and exit -- no training. Used to re-score a
        # `stepN-best` on a FRESH draw, since best-checkpoint selection takes the
        # max of a noisy series and is biased upward by ~1.5 sd (H-Z).
        #
        # ACG_ALIGN_VAL_ON_RESUME must be 0 for these runs. The alignment burn
        # advances the validation draw by global_steps/test_freq resets, so
        # checkpoints saved at different steps would each be scored on a DIFFERENT
        # draw -- reintroducing exactly the confound the comparison exists to remove.
        args.append("trainer.val_only=True")
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
        "VLLM_ATTENTION_BACKEND": cfg["vllm_attn_backend"],
        "TRITON_PTXAS_PATH": "/usr/local/cuda/bin/ptxas",
        "VLLM_USE_FLASHINFER_SAMPLER": "0",
        # NOT expandable_segments: vLLM's memory pool (sleep/wake between rollout
        # and training) asserts against it -- "Expandable segments are not
        # compatible with memory pool", pytorch#147851. That setting came from the
        # old HF-rollout path, where it countered long-run fragmentation.
        # Separate Ray clusters per run, including runs with different Ray versions.
        # Keep the directory short enough for Ray's Unix socket paths.
        "RAY_TMPDIR": "/tmp/ray_acg_" + hashlib.sha256(exp_dir.encode()).hexdigest()[:8],
        "RAY_ADDRESS": "local",
        "TOKENIZERS_PARALLELISM": "false",
        # Persist vLLM's torch.compile / CUDA-graph cache on the data volume. Cold,
        # the first generate() call compiles and captures graphs for every engine
        # and takes minutes with the GPUs pinned at 100% and nothing in the log --
        # which reads exactly like a hang. Warm, engine start is ~16s. Measured
        # generation itself is 6127 tok/s per GPU on TRITON_ATTN, so throughput was
        # never the problem.
        "VLLM_CACHE_ROOT": os.path.join(ROOT, ".cache", "vllm", os.path.basename(exp_dir)),
        "TORCHINDUCTOR_CACHE_DIR": os.path.join(ROOT, ".cache", "inductor", os.path.basename(exp_dir)),
        # belt and braces against the pid ceiling described at ray_num_cpus
        "RAY_num_prestart_python_workers": "4",
        # Avoid idle worker pools competing with another experiment's env actors.
        "RAY_enable_worker_prestart": "0",
        "RAY_prestart_worker_first_driver": "0",
        "RAY_gcs_server_rpc_server_thread_num": "1",
        "RAY_gcs_server_rpc_client_thread_num": "1",
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
        # Historical recorded configs predate new knobs; use their runtime defaults.
        env[e] = str(cfg.get(k, DEFAULTS[k]))
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
    explicit = set()
    for kv in a.set:
        k, _, v = kv.partition("=")
        if k not in cfg:
            sys.exit(f"unknown config key: {k}\nknown: {', '.join(sorted(cfg))}")
        cfg[k] = _coerce(v)
        explicit.add(k)

    # Benchmark protocol, applied only where the command line was silent.
    if "webshop" in str(cfg["env_name"]).lower():
        for k, v in WEBSHOP_PROTOCOL.items():
            if k not in explicit:
                cfg[k] = v

    try:
        validate_future_progress_config(cfg)
    except ValueError as error:
        ap.error(str(error))

    cfg["exp_id"] = f"{a.name}-{a.date}"
    exp_dir = os.path.join(ROOT, "experiments", cfg["exp_id"])
    cfg.update({
        "data_dir": os.environ.get("ACG_DATA_DIR", os.path.join(ROOT, "envdata", "verl_data")),
        "alfworld_data": os.environ.get("ALFWORLD_DATA", os.path.join(ROOT, "alfworld_data")),
        "hf_home": os.environ.get("HF_HOME", os.path.join(ROOT, "hf")),
        "venv_python": os.path.join(ROOT, ".venv", "bin", "python3"),
    })
    snap = os.path.join(cfg["hf_home"], "hub",
                        "models--" + cfg["model"].replace("/", "--"), "snapshots")
    cfg["model_path"] = (os.path.join(snap, sorted(os.listdir(snap))[0])
                         if os.path.isdir(snap) else cfg["model"])

    for sub in ("outputs", "plots"):
        os.makedirs(os.path.join(exp_dir, sub), exist_ok=True)

    if "webshop" in str(cfg.get("env_name", "")).lower():
        cfg["venv_python"] = os.path.join(ROOT, ".venv-webshop", "bin", "python3")
    argv = build_command(cfg, exp_dir)
    env = build_env(cfg, exp_dir)
    if "webshop" in str(cfg.get("env_name", "")).lower():
        # WebShop's product search is BM25/Lucene via pyserini, which starts a JVM
        # inside every Ray env worker; without JAVA_HOME jnius fails on "Unable to
        # find javac" before the env can load.
        _jh = os.environ.get("ACG_JAVA_HOME", os.path.join(ROOT, "jdk", "jdk-11.0.32.1+1"))
        env["JAVA_HOME"] = _jh
        env["PATH"] = _jh + "/bin:" + env.get("PATH", os.environ.get("PATH", ""))
        # One JVM per env worker, and WebShop starts train_batch_size*group_n +
        # val_batch_size of them at once (128 + 128 here). The JVM sizes its GC and JIT
        # thread pools from the VISIBLE core count -- 96 on this box -- so the default
        # costs 88.3 PIDs per worker against a cgroup pids.max of 20,000. Measured on a
        # 16-worker probe: Ray itself takes 3,543, and 256 workers project to 26,152,
        # which is exactly how the first launch died ("pthread_create failed (EAGAIN)",
        # java.lang.OutOfMemoryError: unable to create native thread, at 19,941 PIDs).
        # Pinning the JVM to one processor with the serial collector cuts it to 18.6 per
        # worker and the projection to 8,311. Heap is irrelevant to the thread count;
        # 512m is ample for a BM25 search over the 1,000-product index.
        env["JAVA_TOOL_OPTIONS"] = ("-XX:ActiveProcessorCount=1 -XX:+UseSerialGC "
                                    "-Xss512k -Xms32m -Xmx512m")
    record = {
        "experiment": cfg["exp_id"], "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "config": cfg, "hydra_overrides": argv[3:], "env": env,
        "git": git_state(), "versions": versions(cfg["venv_python"]),
        "reference_protocol": {
            "source": ("baselines/G2PO/examples/g2po_trainer/run_webshop.sh"
                       if "webshop" in str(cfg["env_name"]).lower()
                       else "baselines/G2PO/examples/g2po_trainer/run_alfworld.sh"),
            # Read off the resolved config, not hardcoded, so a benchmark whose
            # protocol differs (WebShop: prompt 4096, mean_norm, 15 turns) records
            # what it actually ran rather than ALFWorld's numbers.
            "matched": [f"val temperature {cfg['val_temperature']} + do_sample",
                        f"val batch {cfg['val_batch_size']}",
                        *([f"eval split {cfg['eval_split']} (valid_seen)"]
                          if "alfworld" in str(cfg["env_name"]).lower() else
                          [f"webshop use_small={bool(int(cfg['webshop_use_small']))}"]
                          if "webshop" in str(cfg["env_name"]).lower() else []),
                        f"group size {cfg['group_size']}",
                        f"train_batch_size {cfg['train_batch_size']}",
                        f"max_prompt_length {cfg['max_prompt_length']}",
                        f"max_response_length {cfg['max_response_length']}",
                        f"lr {cfg['lr']}", f"kl {cfg['kl_loss_coef']} {cfg['kl_loss_type']}",
                        f"gamma {cfg['gamma']}",
                        f"step_advantage_w {cfg['step_advantage_w']}",
                        f"mode {cfg['adv_mode']}",
                        "invalid-action penalty 0.1",
                        *([f"env.max_steps {cfg['max_steps']}"]
                          if "webshop" not in str(cfg["env_name"]).lower()
                          or cfg["max_steps"] == WEBSHOP_PROTOCOL["max_steps"] else []),
                        f"env.seed {cfg['seed']}",
                        f"history_length {cfg['history_length']}"],
            "deltas": _REFERENCE_DELTA + (_WEBSHOP_DELTA
                      if "webshop" in str(cfg["env_name"]).lower() else []) + (
                      [f"env.max_steps {cfg['max_steps']} instead of the reference "
                       f"{WEBSHOP_PROTOCOL['max_steps']}: applies to both training and validation"]
                      if "webshop" in str(cfg["env_name"]).lower()
                      and cfg["max_steps"] != WEBSHOP_PROTOCOL["max_steps"] else []) + (
                      [f"val_batch_size {cfg['val_batch_size']} not the reference 128: the val"
                       f" set (256 rows) is evaluated in {256 // int(cfg['val_batch_size'])} chunks"
                       f" instead of 2, same episodes. WebShop holds one Ray actor per"
                       f" concurrent env and each costs ~65 threads, so 128 train + 128 val"
                       f" actors reach 19,800 of the cgroup's 20,000 pids.max before step 1."]
                      if "webshop" in str(cfg["env_name"]).lower()
                      and int(cfg["val_batch_size"]) != 128 else []),
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
