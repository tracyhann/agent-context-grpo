#!/usr/bin/env python3
"""Verify env_name=Webshop reproduces the published WebShop protocol.

DEFAULTS in scripts/exp_run.py are the ALFWorld reference. WebShop's published
script (baselines/G2PO/examples/g2po_trainer/run_webshop.sh, identical on every
value below to GiGPO's examples/gigpo_trainer/run_webshop.sh) differs in six
learning-relevant settings -- prompt 4096, val 128, 15 turns, mean_norm,
mini-batch 64, 0.05 CPU per env worker. Getting any of them wrong makes our
WebShop numbers incomparable to the published 67.4% without it being visible in
the logs, so the mapping is asserted here rather than trusted.

Test 1 parses the reference script and compares EVERY hydra override it sets
against the command exp_run.py builds, with an explicit exemption list.
Test 2 asserts --set still wins over the protocol, and that ALFWorld runs are
untouched by it.
"""
import importlib.util
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF = os.path.join(ROOT, "baselines/G2PO/examples/g2po_trainer/run_webshop.sh")

_spec = importlib.util.spec_from_file_location("exp_run", os.path.join(ROOT, "scripts/exp_run.py"))
er = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(er)

# Keys whose difference is hardware, bookkeeping or the estimator under test --
# every one of them is recorded in exp_run._REFERENCE_DELTA / _WEBSHOP_DELTA.
EXEMPT = {
    "algorithm.adv_estimator",              # ccpo is the arm
    "algorithm.g2po.step_advantage_w",      # verl-agent namespaces these as gigpo.*
    "algorithm.g2po.mode",
    "data.train_files", "data.val_files",   # local parquet
    "actor_rollout_ref.model.path",         # local snapshot
    "actor_rollout_ref.actor.optim.lr",     # 1e-6 vs python's 1e-06
    "actor_rollout_ref.rollout.tensor_model_parallel_size",   # 2 on 8 GPUs; 1 here
    "actor_rollout_ref.rollout.gpu_memory_utilization",       # 0.6 on 8 GPUs; 0.25 here
    "trainer.n_gpus_per_node", "trainer.logger", "trainer.project_name",
    "trainer.experiment_name", "trainer.save_freq",           # we keep checkpoints
    "trainer.val_before_train",             # trailing shell redirection in the script
    "env.resources_per_worker.num_cpus",    # shell variable, checked separately
}
SUBST = {"$train_data_size": "16", "$val_data_size": "128", "$group_size": "8",
         "$mode": "mean_norm", "$ENGINE": "vllm"}


def _reference_overrides():
    out = {}
    for ln in open(REF):
        ln = ln.strip().rstrip("\\").strip()
        if "=" not in ln or ln.startswith("#") or " " in ln.split("=")[0]:
            continue
        k, _, v = ln.partition("=")
        if k.startswith(("data.", "actor_rollout_ref.", "algorithm.", "env.", "trainer.")):
            out[k] = SUBST.get(v.strip().strip("'\""), v.strip().strip("'\""))
    return out


def _build(**overrides):
    cfg = dict(er.DEFAULTS)
    cfg.update(overrides)
    if "webshop" in str(cfg["env_name"]).lower():
        for k, v in er.WEBSHOP_PROTOCOL.items():
            if k not in overrides:
                cfg[k] = v
    cfg.update(exp_id="t", model_path="/m", data_dir="/d")
    argv = er.build_command(cfg, "/tmp/exp")
    return dict(a.split("=", 1) for a in argv[3:] if "=" in a)


def test_matches_published_webshop_script():
    ours = _build(env_name="Webshop", gpus="0,1", total_epochs=100)
    ref = _reference_overrides()
    checked, bad = 0, []
    for k, v in ref.items():
        if k in EXEMPT:
            continue
        checked += 1
        if str(ours.get(k)) != str(v):
            bad.append(f"{k}: reference {v!r}, ours {ours.get(k)!r}")
    assert not bad, "WebShop protocol drifted from run_webshop.sh:\n  " + "\n  ".join(bad)
    assert checked >= 30, f"only {checked} keys compared -- the reference parse broke"
    # the one exempted value that is still ours to get right
    assert ours["env.resources_per_worker.num_cpus"] == "0.05"
    # and the estimator-facing setting that is protocol, not budget
    assert ours["algorithm.gigpo.mode"] == "mean_norm"
    print(f"test 1 PASS: {checked} overrides match run_webshop.sh")


def test_command_line_wins_and_alfworld_untouched():
    # --set beats the protocol
    forced = _build(env_name="Webshop", max_steps=50, adv_mode="mean_std_norm")
    assert forced["env.max_steps"] == "50", forced["env.max_steps"]
    assert forced["algorithm.gigpo.mode"] == "mean_std_norm"
    # ALFWorld keeps its own protocol
    alf = _build(gpus="0,1")
    assert alf["data.max_prompt_length"] == "2048"
    assert alf["env.max_steps"] == "50"
    assert alf["algorithm.gigpo.mode"] == "mean_std_norm"
    assert alf["data.val_batch_size"] == "64"
    assert alf["env.resources_per_worker.num_cpus"] == "0.1"
    assert alf["actor_rollout_ref.rollout.tensor_model_parallel_size"] == "1"
    print("test 2 PASS: --set wins; ALFWorld defaults unchanged")


if __name__ == "__main__":
    test_matches_published_webshop_script()
    test_command_line_wins_and_alfworld_untouched()
    print("OK")
