"""CPU checks for the HGPO launcher, actual recipe dispatch and budget contract."""
import ast
from contextlib import contextmanager, redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'verl-agent')]
import exp_run as er
import prepare_hgpo_baseline as prepare
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from recipe.hgpo import core_hgpo
from verl import DataProto
from verl.trainer.ppo.core_algos import compute_policy_loss


@contextmanager
def recipe_config_dir():
    previous = Path.cwd()
    try:
        os.chdir(ROOT / 'verl-agent')
        with initialize_config_dir(config_dir=str(ROOT / 'verl-agent/recipe/hgpo/config'), version_base=None):
            yield
    finally:
        os.chdir(previous)


def command(benchmark='webshop', backbone='1.5b', **options):
    cfg = dict(er.DEFAULTS)
    if benchmark == 'webshop':
        cfg.update(er.WEBSHOP_PROTOCOL)
    cfg.update(prepare.settings(benchmark, backbone, **options))
    cfg.update(exp_id='hgpo-test', model_path=cfg['model'], data_dir=str(ROOT / 'envdata/verl_data'))
    return cfg, er.build_command(cfg, '/tmp/hgpo-cpu-test')


def recipe_compute_advantage():
    path = ROOT / 'verl-agent/recipe/hgpo/hgpo_ray_trainer.py'
    tree = ast.parse(path.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'compute_advantage')
    names = ['GAE', 'GRPO', 'GRPO_PASSK', 'REINFORCE_PLUS_PLUS_BASELINE', 'REINFORCE_PLUS_PLUS',
             'REMAX', 'RLOO', 'GiGPO', 'HGPO']
    namespace = dict(DataProto=object, torch=torch, np=np, core_hgpo=core_hgpo,
                     AdvantageEstimator=SimpleNamespace(**{k:k for k in names}))
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['compute_advantage']


class HGPOConfigTests(unittest.TestCase):
    def test_all_model_benchmark_history_combinations_compose_with_real_hydra_schema(self):
        for benchmark in ('alfworld', 'webshop'):
            for backbone in ('1.5b', '7b'):
                for history in (2, 4):
                    with self.subTest(benchmark=benchmark, backbone=backbone, history=history):
                        cfg, argv = command(benchmark, backbone, history=history)
                        self.assertEqual(argv[2], 'recipe.hgpo.main_hgpo')
                        with recipe_config_dir():
                            resolved = compose(config_name='hgpo_trainer', overrides=argv[3:])
                            OmegaConf.resolve(resolved)
                        self.assertEqual(resolved.algorithm.adv_estimator, 'hgpo')
                        self.assertEqual(resolved.data.max_prompt_length, 2048)
                        self.assertEqual(resolved.data.max_response_length, 512)
                        self.assertEqual(resolved.actor_rollout_ref.rollout.prompt_length, 2048)
                        self.assertEqual(resolved.actor_rollout_ref.rollout.response_length, 512)
                        self.assertEqual(resolved.trainer.total_training_steps, 150)
                        self.assertEqual(resolved.trainer.total_epochs, 150)
                        self.assertEqual(resolved.env.max_steps, 15 if benchmark == 'webshop' else 50)
                        self.assertEqual(resolved.env.history_length, history)
                        self.assertEqual(resolved.algorithm.hgpo.mode, cfg['adv_mode'])
                        self.assertFalse(resolved.algorithm.hgpo.base_group)
                        self.assertEqual(resolved.algorithm.hgpo.length_weight_alpha, 1.)
                        self.assertEqual(resolved.data.truncation, 'left')
                        self.assertEqual(resolved.env.rollout.n * resolved.data.train_batch_size, 128)
                        self.assertEqual(resolved.trainer.resume_mode, 'disable')
                        self.assertFalse(any(arg.startswith('algorithm.gigpo.') for arg in argv))

    def test_prompt_and_generation_caps_are_independent(self):
        _, argv = command(prompt_tokens=4096, response_tokens=2048)
        with recipe_config_dir():
            resolved = compose(config_name='hgpo_trainer', overrides=argv[3:])
        self.assertEqual(resolved.actor_rollout_ref.rollout.prompt_length, 4096)
        self.assertEqual(resolved.actor_rollout_ref.rollout.response_length, 2048)

    def test_existing_estimator_routes_are_unchanged(self):
        cfg, _ = command()
        for arm in ('ccpo', 'grpo', 'gigpo'):
            cfg['arm'] = arm
            argv = er.build_command(cfg, '/tmp/existing-arm')
            self.assertEqual(argv[2], 'verl.trainer.main_ppo')
            self.assertIn('algorithm.adv_estimator=' + arm, argv)
            self.assertIn('data.truncation=error', argv)
            self.assertIn('algorithm.gigpo.mode=mean_norm', argv)
            self.assertFalse(any(s.startswith(('algorithm.hgpo.', 'trainer.total_training_steps=')) for s in argv))

    def test_bad_gpu_geometry_and_limits_fail_before_preparation(self):
        for kwargs in ({'gpus':'0,0'}, {'gpus':'0,1,2'}, {'gpus':'-1,0'},
                       {'gpus':'0', 'backbone':'7b'}, {'prompt_tokens':0}, {'response_tokens':0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                prepare.settings('webshop', **kwargs)

    def test_estimator_is_the_pinned_official_source(self):
        data = (ROOT / 'verl-agent/recipe/hgpo/core_hgpo.py').read_bytes()
        self.assertEqual(hashlib.sha256(data).hexdigest(), prepare.CORE_SHA256)

    def test_prepared_runs_use_actual_hgpo_entrypoint_and_explicit_step_cap(self):
        for benchmark in ('alfworld', 'webshop'):
            configs = {}
            for backbone in ('1.5b', '7b'):
                with self.subTest(benchmark=benchmark, backbone=backbone):
                    settings = prepare.settings(benchmark, backbone)
                    folder = ROOT / 'experiments' / prepare.name_for(benchmark, backbone, settings, '20260924')
                    record = json.loads((folder / 'config.json').read_text())
                    cfg = record['config']; configs[backbone] = cfg
                    self.assertEqual(record['hydra_overrides'], er.build_command(cfg, str(folder))[3:])
                    self.assertEqual(record['env']['ACG_ADV_ESTIMATOR'], 'hgpo')
                    self.assertFalse(record['preparation']['launched'])
                    self.assertFalse(record['preparation']['queued'])
                    self.assertIn('trainer.total_training_steps=150', record['hydra_overrides'])
                    self.assertEqual(cfg['max_prompt_length'], 2048)
                    self.assertEqual(cfg['max_response_length'], 512)
                    self.assertEqual(cfg['model'], settings['model'])
                    self.assertEqual(cfg['tp_size'], 1 if backbone == '1.5b' else 2)
                    self.assertEqual(len(cfg['gpus'].split(',')), 4 if backbone == '1.5b' else 8)
                    if backbone == '7b':
                        self.assertIn('7B', cfg['model_path'])
                        self.assertNotIn('1.5B', cfg['model_path'])
            before, after = configs['1.5b'], configs['7b']
            self.assertEqual(set(before), set(after))
            self.assertEqual({key for key in before if before[key] != after[key]},
                             {'model', 'model_path', 'gpus', 'tp_size', 'exp_id'})


class HGPOEstimatorTests(unittest.TestCase):
    def data(self):
        mask = torch.ones(9, 3); mask[::2, -1] = 0
        return SimpleNamespace(batch={
            'step_rewards': torch.tensor([0.,1.,2.,2.,3.,4.,4.,9.,10.], requires_grad=True),
            'token_level_rewards': torch.zeros_like(mask, requires_grad=True),
            'response_mask': mask}, non_tensor_batch={
            'anchor_obs': np.array(['a','x','z','b','x','z','a','x','z']),
            'uid': np.array(['q']*9), 'traj_uid': np.repeat(['a','b','c'], 3)})

    def test_actual_trainer_dispatch_matches_hand_calculated_hierarchy_and_updates_policy(self):
        data = self.data(); compute = recipe_compute_advantage()
        compute(data, 'HGPO', history_length=2, hgpo_mode='mean_norm', hgpo_base_group=False)
        # At a's z: common z and x,z groups give -10/3; a,x,z gives -4.
        # Upstream weights these sequence lengths with (k+1)^alpha = 2,3,4.
        expected = (2*(-10/3) + 3*(-10/3) + 4*(-4)) / (9 + 1e-6)
        self.assertAlmostEqual(data.batch['advantages'][2, 0].item(), expected, places=5)
        mask = data.batch['response_mask']
        advantage = data.batch['advantages']
        self.assertFalse(advantage.requires_grad)
        torch.testing.assert_close(advantage[mask==0], torch.zeros_like(advantage[mask==0]))
        torch.testing.assert_close(data.batch['returns'], advantage)
        logits = torch.linspace(-1, 1, mask.numel()*2).reshape(*mask.shape, 2).requires_grad_()
        logp = logits.log_softmax(-1)[..., 0]
        loss, *_ = compute_policy_loss(old_log_prob=logp.detach(), log_prob=logp,
            advantages=advantage, response_mask=mask, cliprange=.2, loss_agg_mode='token-mean')
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(float(logits.grad.abs().sum()), 0)
        self.assertIsNone(data.batch['step_rewards'].grad)
        self.assertIsNone(data.batch['token_level_rewards'].grad)

    def test_disabled_episode_group_and_benchmark_modes_reach_real_core(self):
        compute = recipe_compute_advantage()
        for mode in ('mean_norm', 'mean_std_norm'):
            a, b = self.data(), self.data()
            b.batch['token_level_rewards'] = torch.arange(27).reshape(9,3).float()
            for data in (a, b):
                compute(data, 'HGPO', history_length=2, hgpo_mode=mode, hgpo_base_group=False)
            torch.testing.assert_close(a.batch['advantages'], b.batch['advantages'], rtol=0, atol=0)
            self.assertTrue(torch.isfinite(a.batch['advantages']).all())

    def test_discounted_returns_and_current_invalid_action_penalty(self):
        path = ROOT / 'verl-agent/recipe/hgpo/hgpo_ray_trainer.py'
        function = next(n for n in ast.parse(path.read_text()).body
                        if isinstance(n, ast.FunctionDef) and n.name == 'apply_invalid_action_penalty')
        namespace = dict(DataProto=object, torch=torch, np=np)
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), namespace)
        data = DataProto.from_dict(tensors={'input_ids': torch.zeros(6,5),
            'prompts': torch.zeros(6,2), 'attention_mask': torch.ones(6,5, dtype=torch.long),
            'token_level_scores': torch.zeros(6,3), 'responses': torch.ones(6,3)},
            non_tensors={'rewards': np.array([0,0,10,0,0,0], dtype=object),
                'traj_uid': np.array(['a']*3+['b']*3), 'active_masks': np.ones(6, dtype=object),
                'is_action_valid': np.array([True,False,True,True,True,True])})
        with redirect_stdout(io.StringIO()):
            data.batch['step_rewards'] = core_hgpo.compute_step_discounted_returns(data, .95)
            namespace['apply_invalid_action_penalty'](data, .1)
        torch.testing.assert_close(data.batch['step_rewards'], torch.tensor([9.025,9.4,10.,0,0,0]))


if __name__ == '__main__':
    unittest.main(verbosity=2)
