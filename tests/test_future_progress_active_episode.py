"""H2 full-strength baselines plus an active episode channel: configs and PPO gradients."""
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
from contextlib import redirect_stdout
import unittest
from unittest.mock import patch

from test_future_progress import (fixture, core, np, torch, rows, ccpo_future_progress_advantage,
                                  finalize_progress_logging, load_compute_advantage,
                                  make_data, policy_gradient, core_gigpo)
from test_future_progress_ablation import registry

ROOT = Path(__file__).resolve().parents[1]
KEY = 'future-progress-h2-no-credit-shrinkage-active-episode'


def training_fixture():
    kw = fixture()
    kw['anchor_obs'] = np.array(['turn-' + str(t) for t in kw['turn_index']])
    kw['anchor_obs'][0] = 'unique-anchor'
    kw['response_mask'][::2, -1] = 0
    return kw


def reward_data(kw):
    data = make_data(kw)
    last = data.batch['response_mask'].sum(-1).long() - 1
    score = torch.as_tensor(kw['episode_rewards'], dtype=torch.float32) - .1 * torch.as_tensor(~kw['is_action_valid'])
    data.batch['token_level_rewards'][torch.arange(len(last)), last] = score
    return data


class ActiveEpisodeFutureProgressTests(unittest.TestCase):
    def test_configs_preserve_h2_protocol_and_enable_exact_two_key_delta(self):
        spec = importlib.util.spec_from_file_location('active_episode_launcher', ROOT/'scripts/exp_run.py')
        er = importlib.util.module_from_spec(spec); spec.loader.exec_module(er)
        for benchmark, prefix in [('alfworld', 'm10'), ('webshop', 'm11')]:
            with self.subTest(benchmark=benchmark):
                folder = ROOT/f'experiments/{prefix}-h2-noshrink-active-episode-{benchmark}-1.5b-2gpu-20260920'
                old_path = ROOT/f'experiments/{prefix}-h2-ccpo-attncred-ctxadv-future-progress-{benchmark}-1.5b-2gpu-20260918/config.json'
                old = json.loads(old_path.read_text())['config']
                record = json.loads((folder/'config.json').read_text()); cfg = record['config']
                self.assertEqual(set(old), set(cfg))
                self.assertEqual({k:v for k,v in cfg.items() if old[k]!=v},
                                 dict(ccpo_lk_fix=1., ccpo_ep_w=1., exp_id=folder.name))
                _, built = registry.build(KEY, benchmark)
                for k,v in built.items(): self.assertEqual(cfg[k], v, k)
                self.assertEqual(cfg['ccpo_ctx_w'], 1.)
                self.assertEqual(cfg['ccpo_progress_horizon'], 2)
                self.assertEqual(cfg['ccpo_edge_w'], 0.)
                self.assertEqual(cfg['total_epochs'], 150)
                self.assertEqual(cfg['model'], 'Qwen/Qwen2.5-1.5B-Instruct')
                self.assertEqual(cfg['resume_from'], '')
                self.assertEqual(record['hydra_overrides'], er.build_command(cfg, str(folder))[3:])
                self.assertIn('trainer.n_gpus_per_node=2', record['hydra_overrides'])
                for k, env_name in er.ENV_KEYS.items():
                    if k == 'ccpo_progress_history_weight' and k not in cfg:
                        self.assertEqual(er.DEFAULTS[k], 1.0); self.assertNotIn(env_name, record['env'])
                    else:
                        self.assertEqual(record['env'][env_name], str(cfg[k]))
                prepared = json.loads((folder/'PREPARED.json').read_text())
                self.assertFalse(prepared['launched']); self.assertFalse(prepared['queued'])
                for bad in [-1, float('nan'), float('inf')]:
                    with self.assertRaises(ValueError):
                        er.validate_future_progress_config(dict(cfg, ccpo_ep_w=bad))

    def test_real_episode_normalization_is_fused_after_step_scaling_and_changes_ppo_gradient(self):
        compute = load_compute_advantage()
        kw = training_fixture()
        mask = kw['response_mask']
        for mode in ['mean_norm', 'mean_std_norm']:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temp:
                expected_data = reward_data(kw)
                row_scores = expected_data.batch['token_level_rewards'].sum(-1)
                ep = row_scores - row_scores.mean()
                if mode == 'mean_std_norm': ep = ep / (row_scores.std() + 1e-6)
                expected_episode = ep[:, None] * mask
                results = {}
                for weight in [0, 1]:
                    folder = Path(temp)/str(weight)
                    env = dict(ACG_CCPO_PROGRESS_HORIZON='2', ACG_CCPO_PROGRESS_WEIGHT='1',
                               ACG_CCPO_EP_W=str(weight), ACG_CCPO_FIXED_ANCHOR='0',
                               ACG_CCPO_STEP_NORM='mode', ACG_CCPO_OUTLOOK_HORIZON='0',
                               ACG_CCPO_OUTLOOK_BETA='0', ACG_EXP_DIR=str(folder), ACG_CCPO_PROGRESS_SNAPSHOT_EVERY='1')
                    data = reward_data(kw)
                    with patch.dict(os.environ, env), patch.object(core, '_LK_FIX', '1.0'), redirect_stdout(io.StringIO()):
                        compute(data, 'CCPO', gamma=.95, gigpo_mode=mode, ccpo_step_tag=1)
                    results[weight] = (data, *policy_gradient(data))
                    with np.load(folder/'outputs/future_progress/step-0001.npz', allow_pickle=False) as a:
                        for prefix, value in [('history','history_baseline'),('current','current_value'),('future','future_value')]:
                            usable = np.isfinite(a[prefix+'_kernel'])
                            np.testing.assert_array_equal(a[prefix+'_lambda_k'][usable], 1.)
                            np.testing.assert_array_equal(a[value][usable], a[prefix+'_kernel'][usable])
                        self.assertTrue(((a['history_J']==1)&(a['history_level']==0)).any())
                        np.testing.assert_allclose(a['episode_adv'], ep.numpy(), atol=1e-6)
                        np.testing.assert_allclose(a['episode_applied'], weight*ep.numpy(), atol=1e-6)
                        np.testing.assert_allclose(a['history_applied']+a['future_applied'], a['combined_applied'], atol=3e-5)
                        np.testing.assert_allclose(a['combined_applied']+a['episode_applied'], a['actor_applied'], atol=3e-5)
                        np.testing.assert_allclose(a['actor_applied'][:,None]*mask.numpy(), data.batch['advantages'].numpy(), atol=3e-5)
                    metrics = data.meta_info['ccpo_diag']
                    self.assertEqual(metrics['ccpo/progress_episode_weight'], weight)
                    self.assertLess(metrics['ccpo/progress_actor_identity_error'], 3e-5)
                    self.assertEqual(metrics['ccpo/progress_original_edge_weight'], 0.)
                    self.assertFalse(data.batch['advantages'].requires_grad)
                    self.assertTrue(torch.isfinite(data.batch['advantages']).all())
                    torch.testing.assert_close(data.batch['advantages'][mask==0], torch.zeros_like(data.batch['advantages'][mask==0]))
                off, on = results[0][0], results[1][0]
                torch.testing.assert_close(on.batch['advantages'] - off.batch['advantages'], expected_episode, atol=1e-6, rtol=1e-6)
                self.assertFalse(torch.allclose(results[0][2], results[1][2]))
                self.assertTrue(torch.isfinite(results[1][2]).all())

    def test_zero_weight_stays_gradient_isolated_under_no_shrink_h2(self):
        kw = training_fixture(); compute = load_compute_advantage()
        base = torch.linspace(-2,2,len(kw['index']))[:,None]*kw['response_mask']
        for mode in ['mean_norm','mean_std_norm']:
            results=[]
            for ep in [base, (base*13+17)*kw['response_mask']]:
                data=make_data(kw)
                env=dict(ACG_CCPO_PROGRESS_HORIZON='2', ACG_CCPO_PROGRESS_WEIGHT='1', ACG_CCPO_EP_W='0',
                         ACG_CCPO_FIXED_ANCHOR='0', ACG_CCPO_OUTLOOK_HORIZON='0', ACG_CCPO_OUTLOOK_BETA='0',
                         ACG_CCPO_STEP_NORM='mode', ACG_EXP_DIR='')
                with patch.dict(os.environ,env), patch.object(core,'_LK_FIX','1.0'), patch.object(core_gigpo,'episode_norm_reward',return_value=ep), redirect_stdout(io.StringIO()):
                    compute(data,'CCPO',gamma=.95,gigpo_mode=mode,ccpo_step_tag=1)
                results.append((data,*policy_gradient(data)))
            for i in [1,2]: torch.testing.assert_close(results[0][i],results[1][i],rtol=0,atol=0)
            self.assertNotEqual(results[0][0].meta_info['ccpo_diag']['ccpo/progress_episode_adv_absmean'],
                                results[1][0].meta_info['ccpo_diag']['ccpo/progress_episode_adv_absmean'])
            self.assertEqual(results[0][0].meta_info['ccpo_diag']['ccpo/progress_episode_applied_absmean'],0.)

    def test_active_episode_logging_requires_and_checks_actual_masked_actor_tensor(self):
        kw=training_fixture(); kw.pop('immediate_rewards')
        with patch.object(core,'_LK_FIX','1.0'):
            step,diag=ccpo_future_progress_advantage(**kw,horizon=2)
        mask=kw['response_mask']; ep=mask.clone(); actor=(step[:,None]+1)*mask
        with patch.dict(os.environ,{'ACG_EXP_DIR':''}):
            with self.assertRaisesRegex(ValueError,'requires episode_adv'):
                finalize_progress_logging(diag,step,kw['index'],False,1,1.)
            bad=actor.clone(); bad[0,-1]=1  # This token is masked out in the real response.
            with self.assertRaisesRegex(ValueError,'actual actor advantage'):
                finalize_progress_logging(diag,step,kw['index'],False,1,1.,episode_adv=ep,response_mask=mask,actor_adv=bad)
            finalize_progress_logging(diag,step,kw['index'],False,1,1.,episode_adv=ep,response_mask=mask,actor_adv=actor)
            self.assertEqual(diag['progress_episode_weight'],1.)


if __name__ == '__main__':
    unittest.main()
