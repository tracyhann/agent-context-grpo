#!/usr/bin/env python3
"""The CCPO-ATTNCRED arms: one definition, four benchmark x backbone builds.

Every arm in `official-repo/` is this file's BASE plus a small delta. Nothing is
inherited from `exp_run.DEFAULTS` that the method depends on -- an arm that takes
its method from a mutable default is not a reproducible arm, and two of those
defaults (`ccpo_tau`, `ccpo_edge_w`) do NOT match what the runs called attncred
actually executed. See the note on BASE.

  BASE          the estimator, identical in every run
  BENCHMARK     alfworld | webshop -- env, horizon, and what the step credit predicts
  BACKBONE      1.5b | 7b -- weights and the GPU floor
  METHODS       attncred, attncred-context-adv-only

Protocol, from experiments/experiments.md: 150 steps, three checkpoints kept
(best, step 100, last), one seed.
"""
import argparse
import json
import os
import subprocess
import sys

# ccpo/ -> the repo root. Everything the arms need is vendored under it: the
# estimator (ccpo/core_ccpo.py), the launcher (scripts/), the trainer overlay
# (patches/), the FA2 stub (docker/). ACG_ROOT overrides, for a checkout whose
# runtime (verl-agent/, .venv/, data) lives somewhere else.
ROOT = os.environ.get("ACG_ROOT") or os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))
EXP_RUN = os.path.join(ROOT, "scripts", "exp_run.py")
VENV_PYTHON = os.path.join(ROOT, ".venv", "bin", "python3")
WEBSHOP_VENV = os.path.join(ROOT, ".venv-webshop", "bin", "python3")

# The run whose config.json defines the method on ALFWorld. Guards compare against it.
CONTROL = "ccpo-attncred-150-20260913"

# Historical runs the documentation cites by name -- as the control, or to fill an
# example results table with real numbers. They are not arms this repo builds, so the
# doc/code guard accepts them without expecting a registry entry.
HISTORICAL = {CONTROL, "ccpo-attncred-ws-20260914"}

# ---------------------------------------------------------------------------
# BASE -- the estimator. Identical in all four main runs and in every ablation.
#
# experiments/INSTRUCTIONS.md lists the invariants as eight ccpo_* flags, but its
# command blocks omit `ccpo_tau` and `ccpo_edge_w`, whose exp_run defaults (1.0 and
# 0.0) are NOT what any run called attncred executed (0.15 and 1.0). Running those
# commands verbatim would give a 6.7x wider kernel and no edge term -- a different
# estimator under the same name. Both are pinned here.
# ---------------------------------------------------------------------------
BASE = {
    # -- gate: exact (task_uid, observation) bucket, as GiGPO/G2PO -----------
    "ccpo_gate": "hard",
    "ccpo_sim": 0.0,
    "ccpo_sim_backoff": 0.0,
    "ccpo_backoff_task": 1,      # J=0 rows fall back to the task bucket; keeps live_frac at 1
    "ccpo_backoff_rho": 0.5,
    # -- affinity metric ----------------------------------------------------
    "ccpo_phi": "hidden+ctx",    # whitened last-prompt-token hidden state + trajectory context
    "ccpo_ctx_w": 1.0,
    "ccpo_whiten": 3,
    "ccpo_tau": 0.15,            # kernel width as a multiple of the bucket's median distance
    "ccpo_wmode": "soft",        # exp(-d/tau)
    # -- baseline -----------------------------------------------------------
    "ccpo_lam_fix": 1.0,         # use the phi-attention readout; do not estimate lam
    "ccpo_rho": 0.0,             # inert under lam_fix -- recorded so config.json says so
    "ccpo_shrink": "eb",         # ditto
    "ccpo_prior_kappa": 2.0,     # credibility weight lam_k = J/(J+kappa)
    "ccpo_lk_fix": "",           # derive lam_k from support, do not pin it
    "ccpo_jweight_c": 0.0,
    # -- scale --------------------------------------------------------------
    "ccpo_std": "task",
    "ccpo_std_floor": 0.25,
    "ccpo_step_norm": "mode",    # follow adv_mode; see the webshop note below
    "ccpo_edge_w": 1.0,          # G2PO's value-gain term, folded into the step credit
    "ccpo_ep_w": 1.0,            # weight on the episode term A^EP
    # -- attention: FA2 on both sides, pinned, never inferred ---------------
    "vllm_attn_backend": "FLASH_ATTN",
    "remove_padding": True,
    # -- protocol: 150 steps, three checkpoints -----------------------------
    # best (step<N>-best), step 100 (step100-pin) and last (step<N>-last) are all
    # hardlinked/symlinked outside the rolling window, so keep_ckpts=1 is enough.
    # Early stopping OFF: experiments.md says every run goes to 150.
    "total_epochs": 150,
    "pin_steps": 100,
    "keep_ckpts": 1,
    "early_stop_patience": 0,
    "save_freq": 5,
    "test_freq": 5,
    "seed": 0,
}

BENCHMARK = {
    "alfworld": {
        "env_name": "alfworld/AlfredTWEnv",
        "max_steps": 50,
        # gamma-discounted return-to-go of the binary 10/0 env reward
        "ccpo_target": "return",
    },
    "webshop": {
        "env_name": "Webshop",
        # 15, not ALFWorld's 50: GiGPO and G2PO both publish WebShop at 15 turns and
        # the return-to-go target changes with the horizon.
        "max_steps": 15,
        # experiments.md: "use the dense-score from Webshop tasks (avoid zero advantage
        # situation)". verl-agent overwrites WebShop's reward with a binary 10/0 and
        # keeps the env's partial score in info['task_score']; 30-69% of task groups
        # score zero on EVERY rollout, where a group-relative estimator computes
        # exactly zero advantage. This target is the discounted return-to-go of the
        # dense score instead, scaled to the binary reward's units.
        # STEP CHANNEL ONLY -- the episode term keeps the published binary reward.
        "ccpo_target": "score",
        # The rest of the WebShop protocol (prompt 4096, mean_norm, mini-batch 64,
        # val batch 128, 0.05 CPU/worker) comes from exp_run.WEBSHOP_PROTOCOL, which
        # is G2PO's run_webshop.sh verbatim and is asserted by
        # tests/test_webshop_protocol.py. Not repeated here so there is one source.
        #
        # NOTE adv_mode=mean_norm arrives with that protocol and changes the update:
        # neither term is standardised on WebShop, both stay in reward units.
    },
}

BACKBONE = {
    "1.5b": {"model": "Qwen/Qwen2.5-1.5B-Instruct", "min_gpus": 4},
    # The token budgets in exp_run.DEFAULTS (ppo_max_token_len_per_gpu 12288,
    # log_prob 24576) were memory-calibrated for 1.5B. They are not raised here
    # because no 7B run has been measured on the target host: smoke-test one step
    # and adjust with --set rather than trusting a number nobody has seen hold.
    "7b": {"model": "Qwen/Qwen2.5-7B-Instruct", "min_gpus": 8},
}

METHODS = {
    # The published method.
    "attncred": {},
    # experiments.md, "Main method experiments": the episode advantage removed, so the
    # only signal is the normalised, context-conditioned, credit-adapted step term.
    # A^EP is plain GRPO on the trajectory return and is what attncred SHARES with the
    # baselines; dropping it leaves context + edge. (HGPO ships this shape and reports
    # that adding the episode term hurt.)
    "attncred-context-adv-only": {"ccpo_ep_w": 0.0},
}

# Short tags for experiment ids.
_TAG = {"attncred": "attncred", "attncred-context-adv-only": "attncred-ctxadv"}

# Canonical variant names. These are what experiments/experiments.md calls each arm;
# the guard asserts the two agree, so the doc cannot drift from the code.
VARIANT = {
    ("attncred", "alfworld"): "CCPO-ATTNCRED",
    ("attncred", "webshop"): "CCPO-ATTNCRED-WS",
    ("attncred-context-adv-only", "alfworld"): "CCPO-ATTNCRED-CTXADV",
    ("attncred-context-adv-only", "webshop"): "CCPO-ATTNCRED-CTXADV-WS",
}


def variant_name(method, benchmark):
    return VARIANT[(method, benchmark)]


def build(method, benchmark, backbone, extra=None):
    """Resolve one arm to (exp_name, {config key: value})."""
    if method not in METHODS:
        raise KeyError(f"unknown method {method!r}; known: {', '.join(METHODS)}")
    if benchmark not in BENCHMARK:
        raise KeyError(f"unknown benchmark {benchmark!r}; known: {', '.join(BENCHMARK)}")
    if backbone not in BACKBONE:
        raise KeyError(f"unknown backbone {backbone!r}; known: {', '.join(BACKBONE)}")
    cfg = dict(BASE)
    cfg.update(BENCHMARK[benchmark])
    cfg.update({"model": BACKBONE[backbone]["model"]})
    cfg.update(METHODS[method])
    cfg.update(extra or {})
    bench_tag = "alfworld" if benchmark == "alfworld" else "ws"
    return f"ccpo-{_TAG[method]}-{bench_tag}-{backbone}", cfg


def flags(cfg):
    """--set arguments for exp_run.py, in a stable order."""
    out = []
    for k, v in cfg.items():
        out += ["--set", f"{k}={v}"]
    return out


def venv_for(benchmark):
    return WEBSHOP_VENV if benchmark == "webshop" else VENV_PYTHON


# ---------------------------------------------------------------------------
# Flash attention preflight.
#
# exp_run.build_env DERIVES VERL_ATTN_IMPL from whether `import flash_attn` succeeds
# in the venv, and appends docker/fa_stub to PYTHONPATH when it does not. That is
# right on the sm_120 research box and wrong on the A100/H100 deployment target: a
# silent fall back to sdpa turns off packing, costs ~2.6x per step, and still writes
# a config.json that reads like a normal run.
#
# Two ways the derivation goes wrong here:
#   1. flash-attn is genuinely missing -> sdpa, silently.
#   2. docker/fa_stub is already on the launching shell's PYTHONPATH. The probe
#      inherits it, `import flash_attn` succeeds against the STUB, and the run gets
#      VERL_ATTN_IMPL=flash_attention_2 with no kernels behind it -- every callable
#      in the stub raises, so the first compute_log_prob dies.
# So this asks for a compiled, CALLABLE build, not a successful import.
# ---------------------------------------------------------------------------

_PROBE = r"""
import json, os, sys
out = {}
try:
    import flash_attn
    out["version"] = getattr(flash_attn, "__version__", "?")
    out["file"] = getattr(flash_attn, "__file__", "") or ""
except Exception as e:
    out["import_error"] = "%s: %s" % (type(e).__name__, e)
try:
    import flash_attn_2_cuda                    # compiled extension; the stub has none
    out["compiled"] = True
except Exception as e:
    out["compiled"] = False
    out["compiled_error"] = "%s: %s" % (type(e).__name__, e)
try:
    import torch
    out["torch"] = torch.__version__
    if torch.cuda.is_available():
        out["n_gpu"] = torch.cuda.device_count()
        out["devices"] = sorted({
            "%s (sm_%d%d)" % ((torch.cuda.get_device_name(i),) + torch.cuda.get_device_capability(i))
            for i in range(torch.cuda.device_count())})
        try:
            from flash_attn import flash_attn_varlen_func
            q = torch.randn(4, 2, 16, dtype=torch.bfloat16, device="cuda")
            cu = torch.tensor([0, 4], dtype=torch.int32, device="cuda")
            flash_attn_varlen_func(q, q, q, cu, cu, 4, 4, causal=True)
            out["varlen_ok"] = True
        except Exception as e:
            out["varlen_ok"] = False
            out["varlen_error"] = "%s: %s" % (type(e).__name__, e)
except Exception as e:
    out["torch_error"] = "%s: %s" % (type(e).__name__, e)
print(json.dumps(out))
"""


def flash_attn_status(python):
    """(ok, reason, detail) for the trainer-side flash-attn build in `python`.

    Probed with docker/fa_stub stripped from PYTHONPATH, so a stub on the launching
    shell's path cannot make a missing build look present.
    """
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in env.get("PYTHONPATH", "").split(os.pathsep)
        if p and "fa_stub" not in p)
    try:
        r = subprocess.run([python, "-c", _PROBE], capture_output=True, text=True,
                           timeout=300, env=env)
    except Exception as e:                                   # noqa: BLE001
        return False, f"could not run {python}: {e}", {}
    try:
        d = json.loads(r.stdout.strip().splitlines()[-1])
    except Exception:                                        # noqa: BLE001
        return False, f"probe produced no result: {(r.stderr or r.stdout).strip()[-400:]}", {}

    if "import_error" in d:
        return False, "flash-attn is not installed (%s)" % d["import_error"], d
    if "fa_stub" in d.get("file", ""):
        return False, "PYTHONPATH resolves flash_attn to docker/fa_stub, the import-only stub", d
    if not d.get("compiled"):
        return False, "flash_attn imports but flash_attn_2_cuda does not (%s)" % d.get(
            "compiled_error", "no compiled extension"), d
    if d.get("varlen_ok") is False:
        return False, "the varlen kernel does not run on this card (%s)" % d.get("varlen_error", ""), d
    return True, "flash-attn %s, compiled%s" % (
        d.get("version", "?"), ", varlen kernel verified on GPU" if d.get("varlen_ok") else ""), d


def preflight(python, fatal=True):
    """Refuse to launch an arm that would silently run on sdpa."""
    ok, reason, detail = flash_attn_status(python)
    for dev in detail.get("devices", []):
        print(f"[arm] gpu        {dev}")
    print(f"[arm] flash-attn {'OK: ' if ok else 'MISSING: '}{reason}")
    if ok:
        return True
    print("[arm] these arms are defined for A100 (sm_80) / H100 (sm_90) with FA2:\n"
          "      without it exp_run falls back to VERL_ATTN_IMPL=sdpa, packing is off,\n"
          "      and the run is neither comparable to the others nor affordable.\n"
          "      Install flash-attn into the venv, or pass --allow-sdpa to run anyway.",
          file=sys.stderr)
    if fatal:
        sys.exit(2)
    return False


def add_common_args(ap):
    ap.add_argument("--benchmark", required=True, choices=sorted(BENCHMARK))
    ap.add_argument("--gpus", required=True,
                    help="CUDA device ids for this run, e.g. 0,1,2,3")
    ap.add_argument("--name", default=None, help="override the generated experiment name")
    ap.add_argument("--allow-sdpa", action="store_true",
                    help="launch without a working flash-attn build (NOT comparable)")
    return ap


def launch(name, cfg, benchmark, gpus, passthrough, allow_sdpa=False, label=""):
    """Preflight, then hand the arm to scripts/exp_run.py."""
    if not os.path.isfile(EXP_RUN):
        sys.exit(f"scripts/exp_run.py not found under {ROOT}; set ACG_ROOT to the "
                 "research checkout that holds the launcher")
    cfg = dict(cfg, gpus=gpus)
    print(f"[arm] {label or name}")
    print(f"[arm] benchmark  {benchmark}   gpus {gpus}")
    dry = "--dry-run" in passthrough
    preflight(venv_for(benchmark), fatal=not (dry or allow_sdpa))

    argv = ([sys.executable, EXP_RUN, "--name", name, "--arm", "ccpo"]
            + flags(cfg) + passthrough)
    print("[arm] " + " ".join(argv[1:]))
    return subprocess.call(argv)


def check_gpus(gpus, backbone):
    n = len([g for g in str(gpus).split(",") if g.strip() != ""])
    need = BACKBONE[backbone]["min_gpus"]
    if n < need:
        print(f"[arm] WARNING: {backbone} is specified for >= {need} GPUs, got {n} "
              f"({gpus}). experiments.md sets that floor; a smaller host may OOM or "
              f"fail verl's train_batch_size * rollout.n % n_gpus assertion.",
              file=sys.stderr)
    return n
