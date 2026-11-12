#!/usr/bin/env python3
"""Verify env_name=search reproduces GiGPO's published Search-R1 protocol.

DEFAULTS in scripts/exp_run.py are the ALFWorld reference. The published script
(verl-agent examples/gigpo_trainer/run_search.sh, upstream 20bd331) differs in
batch geometry, horizon, history, prompt budget, KL, warmup, entropy, truncation
and validation decoding. REFERENCE below is that script's overrides with its shell
variables substituted; when the provisioned checkout is present the two are
compared, so a transcription error or an upstream change shows up here.

    python3 tests/test_search_protocol.py
"""
import importlib.util
import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF = os.path.join(ROOT, "verl-agent/examples/gigpo_trainer/run_search.sh")

_spec = importlib.util.spec_from_file_location("exp_run_search", os.path.join(ROOT, "scripts/exp_run.py"))
er = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(er)

SUBST = {"$train_data_size": "256", "$val_data_size": "512", "$group_size": "5",
         "$mode": "mean_std_norm", "$ENGINE": "vllm", "$enable_similarity": "True",
         "$similarity_thresh": "0.9"}
REFERENCE = {
    "algorithm.adv_estimator": "gigpo",
    "data.train_files": "$TRAIN_DATA",
    "data.val_files": "$VAL_DATA",
    "data.train_batch_size": "256",
    "data.val_batch_size": "512",
    "data.max_prompt_length": "4096",
    "data.max_response_length": "512",
    "data.filter_overlong_prompts": "True",
    "data.truncation": "left",
    "data.return_raw_chat": "True",
    "actor_rollout_ref.model.path": "Qwen/Qwen2.5-7B-Instruct",
    "actor_rollout_ref.actor.optim.lr": "1e-6",
    "actor_rollout_ref.actor.optim.lr_warmup_steps_ratio": "0.1",
    "actor_rollout_ref.model.use_remove_padding": "True",
    "actor_rollout_ref.actor.ppo_mini_batch_size": "512",
    "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu": "16",
    "actor_rollout_ref.actor.use_kl_loss": "True",
    "actor_rollout_ref.actor.kl_loss_coef": "0.001",
    "actor_rollout_ref.actor.kl_loss_type": "low_var_kl",
    "actor_rollout_ref.actor.entropy_coeff": "0",
    "actor_rollout_ref.model.enable_gradient_checkpointing": "True",
    "actor_rollout_ref.actor.fsdp_config.param_offload": "False",
    "actor_rollout_ref.actor.fsdp_config.optimizer_offload": "False",
    "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu": "32",
    "actor_rollout_ref.rollout.tensor_model_parallel_size": "1",
    "actor_rollout_ref.rollout.name": "vllm",
    "actor_rollout_ref.rollout.gpu_memory_utilization": "0.6",
    "actor_rollout_ref.rollout.enable_chunked_prefill": "False",
    "actor_rollout_ref.rollout.enforce_eager": "False",
    "actor_rollout_ref.rollout.free_cache_engine": "False",
    "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu": "32",
    "actor_rollout_ref.ref.fsdp_config.param_offload": "True",
    "actor_rollout_ref.actor.use_invalid_action_penalty": "True",
    "actor_rollout_ref.actor.invalid_action_penalty_coef": "0.01",
    "algorithm.use_kl_in_reward": "False",
    "algorithm.gamma": "0.95",
    "algorithm.gigpo.step_advantage_w": "1.0",
    "algorithm.gigpo.mode": "mean_std_norm",
    "algorithm.gigpo.enable_similarity": "True",
    "algorithm.gigpo.similarity_thresh": "0.9",
    "env.env_name": "search",
    "env.seed": "0",
    "env.max_steps": "4",
    "env.rollout.n": "5",
    "env.history_length": "4",
    "env.search.search_url": "http://127.0.0.1:8000/retrieve",
    "trainer.critic_warmup": "0",
    "trainer.logger": "['console','wandb']",
    "trainer.project_name": "verl_agent_search",
    "trainer.experiment_name": "gigpo_sim0.9_qwen2.5_7b_instruct",
    "trainer.n_gpus_per_node": "8",
    "trainer.nnodes": "1",
    "trainer.save_freq": "50",
    "trainer.test_freq": "50",
    "trainer.total_epochs": "1",
    "trainer.val_before_train": "False",
}
# Differences that are hardware, bookkeeping or documented in exp_run._SEARCH_DELTA.
# The last three are checked for equivalence below rather than for equality.
EXEMPT = {
    "data.train_files", "data.val_files",                    # local parquet
    "actor_rollout_ref.model.path",                          # local snapshot
    "actor_rollout_ref.actor.optim.lr",                      # 1e-6 vs python's 1e-06
    "actor_rollout_ref.rollout.gpu_memory_utilization",      # this tree keeps 0.25
    "trainer.logger", "trainer.project_name", "trainer.experiment_name",
    "actor_rollout_ref.actor.invalid_action_penalty_coef",   # 0.1 on a 10/0 reward
    "trainer.total_epochs",                                  # a step cap of one epoch
}
SEARCH_ONLY = ("actor_rollout_ref.actor.optim.lr_warmup_steps_ratio", "actor_rollout_ref.actor.entropy_coeff",
               "trainer.total_training_steps", "algorithm.gigpo.enable_similarity",
               "algorithm.gigpo.similarity_thresh", "env.search.search_url", "env.search.topk")
TRAIN_QUESTIONS = 169615    # PeterJinGo/nq_hotpotqa_train, train split


def _parse(path):
    out = {}
    for line in Path(path).read_text().splitlines():
        line = line.strip().rstrip("\\").strip()
        if "=" not in line or line.startswith("#") or " " in line.split("=")[0]:
            continue
        key, _, value = line.partition("=")
        if key.startswith(("data.", "actor_rollout_ref.", "algorithm.", "env.", "trainer.")):
            value = value.strip()
            if value.endswith(" $@"):
                value = value[:-3].strip()
            value = value.strip("'\"")
            out[key] = SUBST.get(value, value)
    return out


def _config(**overrides):
    cfg = dict(er.DEFAULTS)
    cfg.update(overrides)
    protocol = (er.SEARCH_PROTOCOL if er.is_search(cfg)
                else er.WEBSHOP_PROTOCOL if "webshop" in str(cfg["env_name"]).lower() else {})
    for key, value in protocol.items():
        if key not in overrides:
            cfg[key] = value
    cfg.update(exp_id="t", model_path="/m", data_dir="/d", alfworld_data="/a", hf_home="/h",
               venv_python="/nonexistent/python3")
    return cfg


def _build(**overrides):
    argv = er.build_command(_config(**overrides), "/tmp/exp")
    return dict(a.split("=", 1) for a in argv[3:] if "=" in a)


EIGHT = "0,1,2,3,4,5,6,7"


class PublishedProtocolTests(unittest.TestCase):
    def test_gigpo_arm_matches_the_published_script(self):
        ours = _build(arm="gigpo", env_name="search", gpus=EIGHT)
        checked, bad = 0, []
        for key, value in REFERENCE.items():
            if key in EXEMPT:
                continue
            checked += 1
            if str(ours.get(key)) != value:
                bad.append(f"{key}: reference {value!r}, ours {ours.get(key)!r}")
        self.assertFalse(bad, "Search-R1 protocol drifted from run_search.sh:\n  " + "\n  ".join(bad))
        self.assertEqual(checked, len(REFERENCE) - len(EXEMPT))
        self.assertGreaterEqual(checked, 45)

    def test_exempted_values_are_equivalent(self):
        ours = _build(arm="gigpo", env_name="search", gpus=EIGHT)
        scale = er.SEARCH_FIXED["reward_scale"]
        # Same penalty-to-reward ratio as the reference's 0.01 on a 1/0 reward.
        self.assertAlmostEqual(float(ours["actor_rollout_ref.actor.invalid_action_penalty_coef"]) / scale,
                               float(REFERENCE["actor_rollout_ref.actor.invalid_action_penalty_coef"]))
        # One epoch with drop_last at the reference batch size.
        steps = TRAIN_QUESTIONS // int(REFERENCE["data.train_batch_size"])
        self.assertEqual(ours["trainer.total_training_steps"], str(steps))
        self.assertGreaterEqual(int(ours["trainer.total_epochs"]), 1)
        self.assertEqual(float(ours["actor_rollout_ref.actor.optim.lr"]),
                         float(REFERENCE["actor_rollout_ref.actor.optim.lr"]))

    def test_validation_is_greedy_as_in_the_reference(self):
        # run_search.sh sets no val_kwargs; verl's defaults are temperature 0, do_sample False.
        ours = _build(arm="ccpo", env_name="search", gpus=EIGHT)
        self.assertEqual(ours["actor_rollout_ref.rollout.val_kwargs.temperature"], "0")
        self.assertEqual(ours["actor_rollout_ref.rollout.val_kwargs.do_sample"], "False")
        sampled = _build(arm="ccpo", env_name="search", gpus=EIGHT, val_temperature=0.4)
        self.assertEqual(sampled["actor_rollout_ref.rollout.val_kwargs.do_sample"], "True")

    @unittest.skipUnless(os.path.exists(REF), "needs the provisioned verl-agent checkout")
    def test_transcription_matches_the_checkout(self):
        self.assertEqual(_parse(REF), REFERENCE)


class ArmTests(unittest.TestCase):
    def test_only_gigpo_gets_similarity_grouping(self):
        for arm in ("ccpo", "grpo"):
            ours = _build(arm=arm, env_name="search", gpus=EIGHT)
            self.assertEqual(ours["algorithm.adv_estimator"], arm)
            self.assertNotIn("algorithm.gigpo.enable_similarity", ours)
            self.assertNotIn("algorithm.gigpo.similarity_thresh", ours)

    def test_reward_scale_reaches_every_search_arm_and_no_other_benchmark(self):
        for arm in ("ccpo", "grpo", "gigpo"):
            env = er.build_env(_config(arm=arm, env_name="search", gpus=EIGHT), "/tmp/exp")
            self.assertEqual(float(env["ACG_SEARCH_REWARD_SCALE"]), 10.0)
        for name in ("alfworld/AlfredTWEnv", "Webshop"):
            self.assertNotIn("ACG_SEARCH_REWARD_SCALE", er.build_env(_config(env_name=name), "/tmp/exp"))

    def test_command_line_wins(self):
        ours = _build(arm="ccpo", env_name="search", gpus=EIGHT, total_epochs=150, max_steps=6, test_freq=25)
        self.assertEqual(ours["trainer.total_training_steps"], "150")
        self.assertEqual(ours["trainer.total_epochs"], "150")
        self.assertEqual(ours["env.max_steps"], "6")
        self.assertEqual(ours["trainer.test_freq"], "25")

    def test_other_benchmarks_are_untouched(self):
        for overrides in ({}, {"env_name": "Webshop"}, {"arm": "gigpo"}, {"arm": "grpo", "env_name": "Webshop"}):
            ours = _build(gpus="0,1", **overrides)
            for key in SEARCH_ONLY:
                self.assertNotIn(key, ours, (overrides, key))
            self.assertEqual(ours["data.truncation"], "error")
            self.assertEqual(ours["actor_rollout_ref.rollout.val_kwargs.do_sample"], "True")
            self.assertEqual(ours["actor_rollout_ref.actor.invalid_action_penalty_coef"], "0.1")
        alfworld = _build(gpus="0,1")
        self.assertEqual((alfworld["env.max_steps"], alfworld["env.history_length"],
                          alfworld["env.rollout.n"], alfworld["data.train_batch_size"]), ("50", "2", "8", "16"))

    def test_no_launcher_key_was_added(self):
        # Prepare scripts require a prepared config to have exactly its control's keys.
        self.assertTrue(set(er.SEARCH_PROTOCOL) <= set(er.DEFAULTS))
        self.assertFalse(set(er.SEARCH_FIXED) & set(er.DEFAULTS))

    def test_rejected_configurations(self):
        for overrides, message in (({"ccpo_target": "score"}, "ccpo_target=return"),
                                   ({"arm": "hgpo"}, "HGPO"),
                                   ({"total_epochs": 0}, "step budget")):
            with self.assertRaisesRegex(ValueError, message):
                _build(env_name="search", gpus=EIGHT, **overrides)
        # The same keys remain legal elsewhere.
        _build(env_name="Webshop", gpus="0,1", ccpo_target="score")


class LauncherTests(unittest.TestCase):
    def _dry_run(self, *sets):
        with tempfile.TemporaryDirectory() as tmp:
            argv = ["exp_run.py", "--name", "search-check", "--date", "20260101", "--dry-run", *sets]
            with patch.object(er, "ROOT", tmp), patch.object(sys, "argv", argv), redirect_stdout(io.StringIO()):
                er.main()
            folder = Path(tmp, "experiments", "search-check-20260101")
            return json.loads((folder / "config.json").read_text()), (folder / "run.sh").read_text()

    def test_main_applies_the_protocol_where_the_command_line_is_silent(self):
        record, run = self._dry_run("--arm", "ccpo", "--set", "env_name=search", "--set", f"gpus={EIGHT}",
                                    "--set", "total_epochs=200")
        cfg = record["config"]
        for key, value in er.SEARCH_PROTOCOL.items():
            self.assertEqual(cfg[key], 200 if key == "total_epochs" else value, key)
        reference = record["reference_protocol"]
        self.assertEqual(reference["source"], "verl-agent/examples/gigpo_trainer/run_search.sh")
        self.assertEqual(reference["departures"], {"total_epochs": {"reference": 662, "launched": 200}})
        self.assertEqual(float(record["env"]["ACG_SEARCH_REWARD_SCALE"]), 10.0)
        self.assertIn("trainer.total_training_steps=200", record["hydra_overrides"])
        self.assertIn("ACG_SEARCH_REWARD_SCALE=10.0", run)
        self.assertEqual(set(cfg) - set(er.DEFAULTS),
                         {"exp_id", "data_dir", "alfworld_data", "hf_home", "venv_python", "model_path"})

    def test_main_leaves_alfworld_alone(self):
        record, run = self._dry_run("--arm", "ccpo", "--set", "gpus=0,1")
        cfg = record["config"]
        for key in er.SEARCH_PROTOCOL:
            self.assertEqual(cfg[key], er.DEFAULTS[key], key)
        self.assertNotIn("ACG_SEARCH_REWARD_SCALE", run)
        self.assertNotIn("departures", record["reference_protocol"])


if __name__ == "__main__":
    unittest.main()
