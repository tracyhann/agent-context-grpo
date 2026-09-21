"""CPU checks for M11 shrinkage and one-history/one-future controls."""
import ast
import copy
import importlib.util
import json
import os
from pathlib import Path
import runpy
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple
import unittest

from test_future_progress_ablation import registry, fixture, core, np, torch, patch
from test_future_progress import ccpo_future_progress_advantage
from ccpo.future_progress import finalize_progress_logging

ROOT = Path(__file__).resolve().parents[1]
H1 = ROOT / 'experiments/m11-ccpo-attncred-ctxadv-future-progress-webshop-1.5b-2gpu-20260918'
H2 = ROOT / 'experiments/m11-h2-ccpo-attncred-ctxadv-future-progress-webshop-1.5b-2gpu-20260918'
CASES = {
    'future-progress-h2-no-credit-shrinkage': ('m11-h2-no-credit-shrinkage-webshop-1.5b-2gpu-20260919', H2, {'ccpo_lk_fix': 1.}),
    'future-progress-h2-kappa4': ('m11-h2-kappa4-webshop-1.5b-2gpu-20260919', H2, {'ccpo_prior_kappa': 4.}),
    'future-progress-history1-future1': ('m11-history1-future1-webshop-1.5b-2gpu-20260919', H1, {'history_length': 1}),
}


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class WebshopFutureProgressAblationTests(unittest.TestCase):
    def test_recorded_configs_and_runtime_flags_match_the_declared_deltas(self):
        er = load('m11_ablation_exp_run', ROOT / 'scripts/exp_run.py')
        for key, (name, parent, delta) in CASES.items():
            with self.subTest(ablation=key):
                folder = ROOT / 'experiments' / name
                record = json.loads((folder / 'config.json').read_text())
                cfg = record['config']
                old = json.loads((parent / 'config.json').read_text())['config']
                self.assertEqual(set(cfg), set(old))
                self.assertEqual({k:v for k,v in cfg.items() if old[k] != v}, dict(delta, exp_id=name))
                _, built = registry.build(key, 'webshop')
                for k, value in built.items():
                    self.assertEqual(cfg[k], value, k)
                er.validate_future_progress_config(cfg)
                self.assertEqual(record['hydra_overrides'], er.build_command(cfg, str(folder))[3:])
                self.assertIn(f"env.history_length={cfg['history_length']}", record['hydra_overrides'])
                self.assertIn('trainer.n_gpus_per_node=2', record['hydra_overrides'])
                for k, env_key in er.ENV_KEYS.items():
                    if k in ('ccpo_progress_history_weight', 'ccpo_loo') and k not in cfg:
                        self.assertEqual(er.DEFAULTS[k], 1.0); self.assertNotIn(env_key, record['env'])
                    else:
                        self.assertEqual(record['env'][env_key], str(cfg[k]), env_key)
                self.assertEqual(cfg['ccpo_target'], 'return')
                self.assertEqual(cfg['adv_mode'], 'mean_norm')
                self.assertEqual(cfg['ccpo_ep_w'], 0.)
                self.assertEqual(cfg['ccpo_edge_w'], 0.)
                self.assertEqual(cfg['ccpo_progress_weight'], 1.)
                self.assertEqual(cfg['ccpo_ctx_w'], 1.)
                self.assertEqual(cfg['ccpo_phi'], 'hidden+ctx')
                self.assertEqual(cfg['total_epochs'], 150)
                self.assertEqual(cfg['resume_from'], '')
                self.assertEqual(record['env']['ACG_EXP_DIR'], str(folder))
                self.assertNotEqual(record['env']['RAY_TMPDIR'], json.loads((parent/'config.json').read_text())['env']['RAY_TMPDIR'])
        with self.assertRaises(KeyError):
            registry.build('future-progress-history1-future1', 'alfworld')

    def test_kappa4_is_applied_to_all_three_readouts_under_webshop_mean_norm(self):
        kw = fixture()
        with patch.object(core, '_PRIOR_KAPPA', 4.), patch.object(core, '_LK_FIX', ''):
            advantage, diag = ccpo_future_progress_advantage(**kw, horizon=2)
        # WebShop mean_norm leaves the combined step channel in these units.
        with patch.dict(os.environ, {'ACG_EXP_DIR': ''}):
            finalize_progress_logging(diag, advantage, kw['index'], normalize=False,
                                      step_tag=1, episode_weight=0., step_weight=1.)
        a = diag['progress_payload']['arrays']
        np.testing.assert_allclose(a['combined_applied'], a['combined_pre'], rtol=1e-6, atol=1e-6)
        for prefix, baseline in [('history', 'history_baseline'), ('current', 'current_value'), ('future', 'future_value')]:
            exact = a[prefix + '_level'] == 0
            self.assertTrue(exact.any())
            weight = a[prefix + '_J'][exact] / (a[prefix + '_J'][exact] + 4.)
            np.testing.assert_allclose(a[prefix + '_lambda_k'][exact], weight)
            np.testing.assert_allclose(a[baseline][exact], weight * a[prefix + '_kernel'][exact] + (1 - weight) * a[prefix + '_task_prior'][exact])
        np.testing.assert_allclose(a['combined_pre'], a['history_adv'] + a['progress_normalized'])
        self.assertTrue(torch.isfinite(advantage).all())
        self.assertFalse(advantage.requires_grad)

    def test_no_shrink_uses_full_context_under_webshop_mean_norm(self):
        kw = fixture()
        with patch.object(core, '_PRIOR_KAPPA', 2.), patch.object(core, '_LK_FIX', '1.0'):
            advantage, diag = ccpo_future_progress_advantage(**kw, horizon=2)
        with patch.dict(os.environ, {'ACG_EXP_DIR': ''}):
            finalize_progress_logging(diag, advantage, kw['index'], normalize=False,
                                      step_tag=1, episode_weight=0., step_weight=1.)
        a = diag['progress_payload']['arrays']
        self.assertTrue(((a['history_J'] == 1) & (a['history_level'] == 0)).any())
        for prefix, baseline in [('history', 'history_baseline'), ('current', 'current_value'), ('future', 'future_value')]:
            usable = np.isfinite(a[prefix + '_kernel']) & (a[prefix + '_level'] <= 1)
            self.assertTrue(usable.any())
            np.testing.assert_array_equal(a[prefix + '_lambda_k'][usable], 1.)
            np.testing.assert_allclose(a[baseline][usable], a[prefix + '_kernel'][usable])
        np.testing.assert_allclose(a['combined_applied'], a['history_applied'] + a['future_applied'], rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(a['combined_applied'], a['combined_pre'], rtol=1e-6, atol=1e-6)
        with patch.object(core, '_PRIOR_KAPPA', 400.), patch.object(core, '_LK_FIX', '1.0'):
            same, _ = ccpo_future_progress_advantage(**kw, horizon=2)
        torch.testing.assert_close(advantage, same, rtol=0, atol=0)
        self.assertTrue(torch.isfinite(advantage).all())
        self.assertFalse(advantage.requires_grad)

    def test_real_webshop_prompt_drops_only_older_history_and_keeps_current_observation(self):
        # Compile the real memory class and pure WebShop prompt methods, avoiding
        # imports that would load environments, Ray, the search engine or a JVM.
        mem_path = ROOT / 'verl-agent/agent_system/memory/memory.py'
        mem_tree = ast.parse(mem_path.read_text())
        cls = next(n for n in mem_tree.body if isinstance(n, ast.ClassDef) and n.name == 'SimpleMemory')
        ns = dict(BaseMemory=object, List=List, Dict=Dict, Any=Any, Tuple=Tuple)
        exec(compile(ast.Module(body=[cls], type_ignores=[]), str(mem_path), 'exec'), ns)
        memory = ns['SimpleMemory']()
        memory.reset(1)
        for i in range(3):
            memory.store({'text_obs': [f'PAST_OBS_{i}'], 'action': [f'PAST_ACT_{i}']})
        before = copy.deepcopy(memory._data)
        env_path = ROOT / 'verl-agent/agent_system/environments/env_manager.py'
        tree = ast.parse(env_path.read_text())
        ws = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'WebshopEnvironmentManager')
        methods = [n for n in ws.body if isinstance(n, ast.FunctionDef) and n.name in ('build_text_obs', 'format_avail_actions')]
        ns.update(runpy.run_path(str(ROOT / 'verl-agent/agent_system/environments/prompts/webshop.py')))
        exec(compile(ast.Module(body=methods, type_ignores=[]), str(env_path), 'exec'), ns)
        env = SimpleNamespace(memory=memory, config=SimpleNamespace(env=SimpleNamespace(history_length=1)), tasks=['BUY_A_MUG'])
        env.format_avail_actions = lambda avail: ns['format_avail_actions'](env, avail)
        info = [{'available_actions': {'has_search_bar': False, 'clickables': ['Buy Now']}}]
        one = ns['build_text_obs'](env, ['CURRENT_OBS'], info)[0]
        for value in ['PAST_OBS_2', 'PAST_ACT_2', 'CURRENT_OBS', 'BUY_A_MUG', 'click[Buy Now]']:
            self.assertIn(value, one)
        for value in ['PAST_OBS_0', 'PAST_ACT_0', 'PAST_OBS_1', 'PAST_ACT_1']:
            self.assertNotIn(value, one)
        self.assertIn('most recent 1 observations', one)
        env.config.env.history_length = 2
        two = ns['build_text_obs'](env, ['CURRENT_OBS'], info)[0]
        self.assertIn('PAST_OBS_1', two)
        self.assertIn('PAST_ACT_1', two)
        self.assertEqual(memory._data, before)  # No truncation of accumulated memory.
        first = ns['build_text_obs'](env, ['CURRENT_OBS'], info, init=True)[0]
        self.assertNotIn('PAST_OBS_', first)
        self.assertIn('CURRENT_OBS', first)


if __name__ == '__main__':
    unittest.main()
