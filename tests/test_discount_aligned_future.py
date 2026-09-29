"""Reviewer ablation: isolate discount-clock drift without altering historical credit."""
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from test_future_progress import (
    arguments, rows, core, np, torch, ccpo_future_progress_advantage,
    progress_from_values, standardize_progress, load_compute_advantage,
    policy_gradient, core_gigpo)
from test_future_progress_active_episode import training_fixture, reward_data

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import prepare_discount_aligned_ablations as prep

NO_SHRINK = dict(_LK_FIX='1.0', _LAM_FIX='1.0', _LOO='1', _GATE='hard',
                 _WMODE='soft', _PHI_MODE='hidden+ctx', _CTX_W=1., _WHITEN_K=3,
                 _JW_C=0., _STD_MODE='task', _BACKOFF_TASK=True, _SIM=0.,
                 _SIM_BACKOFF=0., _EDGE_W=0., _TAU_ENV='.15')


def payload(kw, **extra):
    return ccpo_future_progress_advantage(**dict(kw, horizon=2, progress_mode='discount_aligned', **extra))[1]['progress_payload']['arrays']


class DiscountAlignedMathTests(unittest.TestCase):
    def test_discount_clock_only_is_zero_even_when_old_task_centering_survives(self):
        gamma = .95
        groups = [np.arange(5), np.arange(5, 8)]
        values = np.r_[gamma**np.arange(5, 0, -1)*10, gamma**np.arange(3, 0, -1)*10]
        rewards = np.full(8, 10.)
        for horizon in range(1, 5):
            with self.subTest(horizon=horizon):
                old, _, _, _, eligible = progress_from_values(values, groups, rewards, horizon)
                td, _, _, window, _ = progress_from_values(values, groups, rewards, horizon, gamma=gamma, mode='discount_aligned')
                self.assertTrue((old > 0).all())
                self.assertGreater(abs(standardize_progress(old, ['task']*8, eligible)[0]).max(), .1)
                np.testing.assert_allclose(td, 0, atol=2e-15)
                np.testing.assert_allclose(standardize_progress(td, ['task']*8, eligible)[0], 0, atol=3e-9)
                self.assertEqual(window[4], 1); self.assertEqual(window[7], 1)

    def test_clipped_multistep_residual_telescopes_with_discounted_one_step_terms(self):
        values = np.array([1., 7., 4., 3., 6., 9., 2., 5.])
        rewards = np.r_[np.full(5, 10.), np.zeros(3)]
        groups = [np.arange(5), np.arange(5, 8)]
        for gamma in (0., .5, .95, 1.):
            one = progress_from_values(values, groups, rewards, 1, gamma=gamma, mode='discount_aligned')[0]
            for horizon in range(1, 5):
                expected = np.zeros(8)
                for group in groups:
                    for t, row in enumerate(group):
                        expected[row] = sum(gamma**j * one[k] for j,k in enumerate(group[t:t+horizon]))
                raw, future, endpoint, window, eligible = progress_from_values(values, groups, rewards, horizon, gamma=gamma, mode='discount_aligned')
                np.testing.assert_allclose(raw, expected, atol=3e-15)
                np.testing.assert_array_equal(eligible, True)
                np.testing.assert_array_equal(future[endpoint<0], rewards[endpoint<0])
                if gamma == 1:
                    np.testing.assert_array_equal(raw, progress_from_values(values, groups, rewards, horizon)[0])

    def test_unsupported_endpoints_history_only_and_invalid_settings(self):
        values = np.array([np.nan, 3., 4., 5.]); groups = [np.arange(4)]; rewards = np.full(4, 10.)
        raw, _, _, _, eligible = progress_from_values(values, groups, rewards, 2, mode='discount_aligned')
        self.assertFalse(eligible[0]); self.assertEqual(raw[0], 0)
        raw, _, _, window, eligible = progress_from_values(values, groups, rewards, 0, mode='discount_aligned')
        np.testing.assert_array_equal(raw, 0); np.testing.assert_array_equal(window, 0)
        self.assertFalse(eligible.any())
        for extra in ({'mode':'typo'}, {'gamma':float('nan')}, {'gamma':-1}, {'gamma':1.1}):
            with self.assertRaises(ValueError):progress_from_values(values, groups, rewards, **extra)
        with self.assertRaises(ValueError):ccpo_future_progress_advantage(**arguments(), progress_mode='typo')
        for extra in ({'ccpo_progress_mode':'typo'}, {'ccpo_progress_mode':'discount_aligned','ccpo_progress_horizon':0}):
            with self.assertRaises(ValueError):prep.er.validate_future_progress_config(extra)
        compute = load_compute_advantage()
        with patch.dict(os.environ, {'ACG_CCPO_PROGRESS_MODE':'typo'}), self.assertRaises(ValueError), redirect_stdout(io.StringIO()):
            compute(reward_data(training_fixture()), 'CCPO', gamma=.95)


class DiscountAlignedEstimatorTests(unittest.TestCase):
    def setUp(self):
        patches = patch.multiple(core, **NO_SHRINK); patches.start(); self.addCleanup(patches.stop)

    def test_same_historical_credit_values_support_and_peer_weights(self):
        kw = arguments()
        _, old = ccpo_future_progress_advantage(**kw, horizon=2)
        _, new = ccpo_future_progress_advantage(**kw, horizon=2, progress_mode='discount_aligned')
        a, b = old['progress_payload']['arrays'], new['progress_payload']['arrays']
        for key in ('history_adv','history_baseline','value_target','current_value','future_value',
                    'current_kernel','future_kernel','current_lambda_k','future_lambda_k',
                    'current_J','future_J','eligible','live','current_phi','future_phi'):
            np.testing.assert_array_equal(a[key], b[key], err_msg=key)
        np.testing.assert_allclose(b['endpoint_discount'], kw['gamma']**b['window_length'])
        np.testing.assert_allclose(b['raw_progress'], b['discount_aligned_raw'])
        np.testing.assert_allclose(a['raw_progress'], b['raw_progress']+b['time_passage_raw'], atol=2e-15)
        self.assertGreater(abs(a['raw_progress']-b['raw_progress']).max(), .1)
        for task in np.unique(b['uid']):
            ids = (b['uid']==task)&b['eligible']; raw = b['raw_progress'][ids]
            expected = (raw-raw.mean())/(raw.std(ddof=1)+1e-6)
            np.testing.assert_allclose(b['progress_normalized'][ids], expected)
        self.assertLess(new['progress_decomposition_max_error'], 2e-15)
        self.assertEqual(new['progress_discount_aligned'], 1)
        np.testing.assert_array_equal(b['future_credit_mode'], 'discount_aligned')

    def test_gamma_one_is_exact_full_estimator_parity(self):
        kw = dict(arguments(), gamma=1.)
        old, a = ccpo_future_progress_advantage(**kw, horizon=2)
        new, b = ccpo_future_progress_advantage(**kw, horizon=2, progress_mode='discount_aligned')
        torch.testing.assert_close(old, new, rtol=0, atol=0)
        np.testing.assert_array_equal(a['progress_payload']['arrays']['raw_progress'], b['progress_payload']['arrays']['raw_progress'])

    def test_padding_shuffle_loo_and_invalid_penalty_are_unchanged(self):
        kw = arguments(); before = payload(kw)
        expected, _ = ccpo_future_progress_advantage(**kw, horizon=2, progress_mode='discount_aligned')
        ids = np.random.default_rng(81).permutation(np.r_[np.arange(len(expected)), 0, 4, 9])
        got, _ = ccpo_future_progress_advantage(**rows(kw, ids), horizon=2, progress_mode='discount_aligned')
        torch.testing.assert_close(got, expected[ids], rtol=0, atol=0)
        own = kw['traj_index']=='traj0'; kw['episode_rewards'][own]=25.
        after = payload(kw)
        np.testing.assert_allclose(before['current_value'][own], after['current_value'][own], atol=1e-12)
        keep = own&~before['terminal']
        np.testing.assert_allclose(before['raw_progress'][keep], after['raw_progress'][keep], atol=1e-12)
        self.assertGreater(abs(before['current_value'][~own]-after['current_value'][~own]).max(), .1)
        kw = arguments(); kw['step_rewards'][2]-=.1; kw['is_action_valid'][2]=False
        after = payload(kw)
        for key in ('current_value','future_value','raw_progress','progress_normalized'):
            np.testing.assert_array_equal(before[key], after[key])
        self.assertGreater(abs(before['history_adv']-after['history_adv']).max(), .05)

    def test_actual_trainer_routes_td_and_keeps_ep0_ep1_fusion_and_masked_ppo_gradients(self):
        compute = load_compute_advantage(); kw = training_fixture(); mask = kw['response_mask']
        episode = torch.linspace(-2, 2, len(kw['index']))[:,None]*mask
        for benchmark in prep.CONTROLS:
            folder = ROOT/'experiments'/prep.run_name(benchmark, '20260929')
            saved = json.loads((folder/'config.json').read_text()); cfg = saved['config']; env = saved['env']
            results = {}; arrays = {}
            with self.subTest(benchmark=benchmark), tempfile.TemporaryDirectory() as temp:
                for credit_mode, perturb in [('difference',False), ('discount_aligned',False), ('discount_aligned',True)]:
                    dest = Path(temp)/f'{credit_mode}-{perturb}'
                    settings = dict(env, ACG_CCPO_PROGRESS_MODE=credit_mode, ACG_EXP_DIR=str(dest), ACG_CCPO_PROGRESS_SNAPSHOT_EVERY='1')
                    ep = (episode*11+19)*mask if perturb else episode
                    data = reward_data(kw)
                    with patch.dict(os.environ, settings), patch.object(core_gigpo, 'episode_norm_reward', return_value=ep), redirect_stdout(io.StringIO()):
                        compute(data, 'CCPO', gamma=cfg['gamma'], gigpo_mode=cfg['adv_mode'], ccpo_step_tag=1)
                    loss, grad = policy_gradient(data); results[credit_mode,perturb]=(data,loss,grad)
                    with np.load(dest/'outputs/future_progress/step-0001.npz', allow_pickle=False) as snap:
                        a = {k:snap[k].copy() for k in snap.files}
                    arrays[credit_mode,perturb]=a
                    selected = 'discount_aligned_raw' if credit_mode=='discount_aligned' else 'undiscounted_raw'
                    np.testing.assert_array_equal(a['raw_progress'], a[selected])
                    np.testing.assert_allclose(a['history_applied']+a['future_applied']+a['episode_applied'], a['actor_applied'], atol=3e-5)
                    np.testing.assert_allclose(a['actor_applied'][:,None]*mask.numpy(), data.batch['advantages'].numpy(), atol=3e-5)
                    np.testing.assert_allclose(a['episode_applied'], cfg['ccpo_ep_w']*ep[:,0].numpy(), atol=1e-6)
                    self.assertFalse(data.batch['advantages'].requires_grad)
                    self.assertTrue(torch.isfinite(grad).all()); self.assertGreater(float(grad.abs().sum()), 0)
                    torch.testing.assert_close(grad[mask==0], torch.zeros_like(grad[mask==0]))
                    self.assertEqual(data.meta_info['ccpo_diag']['ccpo/progress_discount_aligned'], int(credit_mode=='discount_aligned'))
                old, td, changed = results['difference',False], results['discount_aligned',False], results['discount_aligned',True]
                self.assertFalse(torch.allclose(old[2], td[2]))
                np.testing.assert_array_equal(arrays['difference',False]['history_adv'], arrays['discount_aligned',False]['history_adv'])
                if cfg['ccpo_ep_w']==0:
                    for i in (1,2):torch.testing.assert_close(td[i], changed[i], rtol=0, atol=0)
                else:
                    self.assertFalse(torch.allclose(td[2], changed[2]))
                    torch.testing.assert_close(changed[0].batch['advantages']-td[0].batch['advantages'], (episode*10+19)*mask, atol=3e-5, rtol=1e-6)


class PreparedDiscountAlignedTests(unittest.TestCase):
    def test_configs_change_only_future_residual_from_exact_paired_control(self):
        self.assertEqual(prep.er.DEFAULTS['ccpo_progress_mode'], 'difference')
        for benchmark, control in prep.CONTROLS.items():
            folder = ROOT/'experiments'/prep.run_name(benchmark, '20260929')
            old = json.loads((ROOT/'experiments'/control/'config.json').read_text())['config']
            for key, value in prep.IMPLIED.items():old.setdefault(key, value)
            saved = json.loads((folder/'config.json').read_text()); cfg = saved['config']
            self.assertEqual(set(old), set(cfg))
            self.assertEqual({k for k in old if old[k]!=cfg[k]}, {'exp_id','ccpo_progress_mode'})
            self.assertEqual(cfg['ccpo_progress_mode'], 'discount_aligned')
            for key,value in dict(ccpo_ep_w=int(benchmark=='webshop'), ccpo_lam_fix=1., ccpo_lk_fix=1.,
                                  ccpo_progress_horizon=2, history_length=2, ccpo_progress_weight=1.,
                                  ccpo_progress_history_weight=1., gamma=.95, ccpo_gate='hard',
                                  ccpo_wmode='soft', ccpo_loo=1, total_epochs=150,
                                  model='Qwen/Qwen2.5-1.5B-Instruct').items():self.assertEqual(cfg[key],value,key)
            for key, env in prep.er.ENV_KEYS.items():self.assertEqual(saved['env'][env], str(cfg[key]))
            self.assertIn('ACG_CCPO_PROGRESS_MODE=discount_aligned', (folder/'run.sh').read_text())
            self.assertEqual(saved['hydra_overrides'], prep.er.build_command(cfg,str(folder))[3:])
            self.assertEqual(saved['reference_protocol']['changes']['ccpo_progress_mode'], dict(control='difference',prepared='discount_aligned'))
            status=json.loads((folder/'PREPARED.json').read_text())
            self.assertFalse(status['launched']); self.assertFalse(status['queued'])
            self.assertIn('--dry-run', json.loads((folder/'prepare-command.json').read_text())['argv'])
            self.assertFalse((folder/'outputs/train.pid').exists()); self.assertFalse((folder/'outputs/metrics.jsonl').exists())
            with patch.object(prep.subprocess, 'run') as run:
                with self.assertRaises(FileExistsError):prep.prepare(benchmark, '20260929')
            run.assert_not_called()


if __name__=='__main__':unittest.main()
