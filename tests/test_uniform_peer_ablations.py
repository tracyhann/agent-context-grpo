"""Uniform peer readout: independent mean oracle, LOO, padding and actual PPO fusion."""
from contextlib import contextmanager, redirect_stdout
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from test_future_progress import (fixture, core, np, torch, rows,
    ccpo_future_progress_advantage, load_compute_advantage, policy_gradient, core_gigpo)
from test_future_progress_active_episode import reward_data

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import prepare_uniform_peer_ablations as prep
STAMP = '20260925'


def record(benchmark):
    folder = ROOT/'experiments'/prep.run_name(benchmark, STAMP)
    return folder, json.loads((folder/'config.json').read_text())


def batch():
    kw = fixture()
    kw['anchor_obs'] = kw['anchor_obs'].astype(object)
    kw['anchor_obs'][0] = 'unique-fallback-anchor'
    kw['response_mask'][::2, -1] = 0
    return kw


@contextmanager
def settings(weighting='uniform', **overrides):
    values = dict(_WMODE=weighting, _LAM_FIX='1.0', _LK_FIX='1.0', _LOO='1',
        _PHI_MODE='hidden+ctx', _CTX_W=1., _WHITEN_K=3, _GATE='hard', _TAU_ENV='.15',
        _JW_C=0., _STD_MODE='task', _BACKOFF_TASK=True, _SIM=0., _SIM_BACKOFF=0.,
        _EDGE_W=0., _TARGET='return')
    values.update(overrides)
    with patch.multiple(core, **values), redirect_stdout(io.StringIO()):
        yield


def estimate(kw, **extra):
    args = {k: v for k, v in kw.items() if k != 'immediate_rewards'}
    return ccpo_future_progress_advantage(**args, horizon=2, **extra)


class UniformPeerTests(unittest.TestCase):
    def test_configs_are_single_weighting_change_and_load_the_runtime_mode(self):
        for benchmark in prep.CONTROLS:
            folder, saved = record(benchmark); cfg = saved['config']
            control = json.loads((ROOT/'experiments'/prep.CONTROLS[benchmark]/'config.json').read_text())['config']
            for k, v in prep.IMPLIED.items():
                control.setdefault(k, v)
            self.assertEqual(set(cfg), set(control))
            self.assertEqual({k for k in cfg if cfg[k] != control[k]}, {'exp_id', 'ccpo_wmode'})
            self.assertEqual(cfg['ccpo_wmode'], 'uniform')
            self.assertEqual(cfg['ccpo_ep_w'], int(benchmark == 'webshop'))
            for k in ('ccpo_lam_fix', 'ccpo_lk_fix', 'ccpo_loo', 'ccpo_progress_history_weight', 'ccpo_progress_weight'):
                self.assertEqual(cfg[k], 1)
            self.assertEqual(cfg['history_length'], 2)
            self.assertEqual(cfg['ccpo_progress_horizon'], 2)
            self.assertEqual(cfg['total_epochs'], 150)
            self.assertEqual(cfg['model'], 'Qwen/Qwen2.5-1.5B-Instruct')
            self.assertEqual(saved['hydra_overrides'], prep.er.build_command(cfg, str(folder))[3:])
            for k, env in prep.er.ENV_KEYS.items():
                self.assertEqual(saved['env'][env], str(cfg[k]))
            self.assertIn('  ACG_CCPO_WMODE=uniform ', (folder/'run.sh').read_text())
            status = json.loads((folder/'PREPARED.json').read_text())
            self.assertFalse(status['launched']); self.assertFalse(status['queued'])
            self.assertFalse(any(p.is_file() for p in (folder/'outputs').rglob('*')))
            env = dict(os.environ, **saved['env']); env['CUDA_VISIBLE_DEVICES'] = ''
            result = subprocess.run([sys.executable, '-c',
                "from ccpo import core_ccpo as c; assert c._WMODE == 'uniform'; "
                "assert c._LOO == '1'; assert float(c._LK_FIX) == float(c._LAM_FIX) == 1"],
                cwd=ROOT, env=env, capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            with patch.object(prep.subprocess, 'run') as run:
                with self.assertRaises(FileExistsError):
                    prep.prepare(benchmark, STAMP)
                run.assert_not_called()

    def test_manual_occurrence_means_for_history_current_endpoint_and_fallback(self):
        with settings():
            _, diag = estimate(batch())
        a = diag['progress_payload']['arrays']
        fallback = 0; differs_from_trajectory_mean = 0
        # Potential label input is cast to float32 by the existing core readout.
        labels = a['value_target'].astype(np.float32).astype(float)
        for i in range(len(labels)):
            other = (a['uid'] == a['uid'][i]) & (a['traj_uid'] != a['traj_uid'][i])
            peers = np.flatnonzero(other & (a['anchor_obs'] == a['anchor_obs'][i]))
            level = 0
            if not len(peers):
                peers = np.flatnonzero(other); level = 1; fallback += 1
            trajectories, counts = np.unique(a['traj_uid'][peers], return_counts=True)
            for channel, target, value in [('history', a['target'], 'history_baseline'),
                                           ('current', labels, 'current_value')]:
                expected = target[peers].mean()
                self.assertAlmostEqual(a[value][i], expected, places=7)
                self.assertAlmostEqual(a[channel+'_kernel'][i], expected, places=7)
                self.assertAlmostEqual(a[channel+'_uniform'][i], expected, places=7)
                self.assertEqual(a[channel+'_lambda_k'][i], 1)
                self.assertEqual(a[channel+'_same_traj_mass'][i], 0)
                self.assertEqual(a[channel+'_peer_rows'][i], len(peers))
                self.assertEqual(a[channel+'_J'][i], len(trajectories))
                self.assertEqual(a[channel+'_level'][i], level)
                self.assertAlmostEqual(a[channel+'_n_eff'][i], len(peers)**2/(counts**2).sum())
            trajectory_mean = np.mean([labels[peers[a['traj_uid'][peers] == tr]].mean() for tr in trajectories])
            differs_from_trajectory_mean += abs(trajectory_mean-labels[peers].mean()) > 1e-4
        self.assertGreater(fallback, 0)
        self.assertGreater(differs_from_trajectory_mean, 0)
        endpoint = a['endpoint_index']; nonterminal = endpoint >= 0
        np.testing.assert_array_equal(a['future_value'][nonterminal], a['current_value'][endpoint[nonterminal]])
        np.testing.assert_array_equal(a['future_value'][~nonterminal], a['episode_rewards'][~nonterminal])
        raw = a['future_value']-a['current_value']
        np.testing.assert_allclose(a['progress_normalized'], (raw-raw.mean())/(raw.std(ddof=1)+1e-6))
        np.testing.assert_allclose(a['combined_pre'], a['history_adv']+a['progress_normalized'])
        self.assertEqual(diag['progress_uniform_weighting'], 1)

    def test_feature_tau_invariance_with_soft_positive_control_and_future_only_difference(self):
        kw = batch(); changed = batch()
        changed['phi_feats'] = torch.tensor(np.random.default_rng(55).normal(size=kw['phi_feats'].shape), dtype=torch.float32)
        with settings():
            uniform, _ = estimate(kw)
            future_only, d = estimate(kw, history_weight=0)
        with settings(_TAU_ENV='19', _CTX_W=0.):
            altered, _ = estimate(changed)
        torch.testing.assert_close(uniform, altered, rtol=0, atol=0)
        self.assertFalse(torch.allclose(uniform, future_only))
        np.testing.assert_array_equal(d['progress_payload']['arrays']['weighted_history'], 0.)
        with settings('soft'):
            soft, _ = estimate(kw)
        with settings('soft', _TAU_ENV='19', _CTX_W=0.):
            soft_changed, _ = estimate(changed)
        self.assertFalse(torch.allclose(soft, soft_changed))
        self.assertFalse(torch.allclose(uniform, soft))

    def test_excludes_entire_query_trajectory_deduplicates_padding_and_handles_no_peers(self):
        kw = batch(); changed = batch(); own = kw['traj_index'] == 'traj0'
        changed['step_rewards'][own] += 13
        changed['episode_rewards'][own] = 25
        with settings():
            adv, before = estimate(kw); _, after = estimate(changed)
            a, b = [d['progress_payload']['arrays'] for d in (before, after)]
            for key in ('history_baseline', 'current_value'):
                np.testing.assert_array_equal(a[key][own], b[key][own])
            keep = own & ~a['terminal']
            np.testing.assert_array_equal(a['future_value'][keep], b['future_value'][keep])
            self.assertGreater(np.max(abs(a['current_value'][~own]-b['current_value'][~own])), .1)
            order = np.random.default_rng(72).permutation(np.r_[np.arange(len(adv)), 0, 4, 9])
            padded, diag = estimate(rows(kw, order))
            torch.testing.assert_close(padded, adv[order], rtol=0, atol=0)
            self.assertEqual(diag['progress_unique_rows'], len(adv))
            self.assertGreater(diag['progress_padding_frac'], 0)
            unsupported, diag = estimate(rows(kw, np.flatnonzero(own)))
            np.testing.assert_array_equal(unsupported.numpy(), 0)
            self.assertFalse(diag['progress_payload']['arrays']['eligible'].any())

    def train(self, benchmark, episode_weight=None, perturbed_features=False, episode_shift=None):
        _, saved = record(benchmark); cfg = saved['config']; kw = batch()
        if perturbed_features:
            kw['phi_feats'] = torch.tensor(np.random.default_rng(55).normal(size=kw['phi_feats'].shape), dtype=torch.float32)
        data = reward_data(kw)
        with tempfile.TemporaryDirectory() as tmp:
            env = dict(saved['env'], CUDA_VISIBLE_DEVICES='', ACG_EXP_DIR=tmp, ACG_CCPO_DUMP='')
            if episode_weight is not None:
                env['ACG_CCPO_EP_W'] = str(episode_weight)
            original_ep = core_gigpo.episode_norm_reward
            def episode(*args, **kwargs):
                result = original_ep(*args, **kwargs)
                return result if episode_shift is None else result+episode_shift*kw['response_mask']
            with patch.dict(os.environ, env), settings(_TAU_ENV='19' if perturbed_features else '.15'), \
                    patch.object(core_gigpo, 'episode_norm_reward', side_effect=episode):
                load_compute_advantage()(data, 'CCPO', gamma=.95, gigpo_mode=cfg['adv_mode'], ccpo_step_tag=1)
            with np.load(Path(tmp)/'outputs/future_progress/step-0001.npz', allow_pickle=False) as file:
                arrays = {k: v.copy() for k, v in file.items()}
        return data, policy_gradient(data)[1], arrays, kw

    def test_real_ppo_fusion_masks_and_feature_invariance_in_both_benchmarks(self):
        for benchmark in prep.CONTROLS:
            data, grad, a, kw = self.train(benchmark)
            changed, changed_grad, _, _ = self.train(benchmark, perturbed_features=True)
            torch.testing.assert_close(data.batch['advantages'], changed.batch['advantages'], rtol=0, atol=0)
            torch.testing.assert_close(grad, changed_grad, rtol=0, atol=0)
            self.assertTrue(torch.isfinite(grad).all())
            self.assertGreater(float(grad.abs().sum()), 0)
            self.assertFalse(data.batch['advantages'].requires_grad)
            mask = kw['response_mask']
            torch.testing.assert_close(data.batch['advantages'][mask == 0], torch.zeros_like(data.batch['advantages'][mask == 0]))
            np.testing.assert_allclose(a['actor_applied'], a['history_applied']+a['future_applied']+a['episode_applied'], atol=3e-5)
            self.assertGreater(np.max(abs(a['history_applied'])), .1)
            self.assertGreater(np.max(abs(a['future_applied'])), .1)
            self.assertEqual(data.meta_info['ccpo_diag']['ccpo/progress_uniform_weighting'], 1.)
            self.assertEqual(data.meta_info['ccpo_diag']['ccpo/progress_episode_weight'], int(benchmark == 'webshop'))
            if benchmark == 'alfworld':
                np.testing.assert_array_equal(a['episode_applied'], 0.)
            else:
                off, off_grad, _, _ = self.train(benchmark, episode_weight=0)
                scores = reward_data(kw).batch['token_level_rewards'].sum(-1)
                expected = (scores-scores.mean())[:, None]*mask
                torch.testing.assert_close(data.batch['advantages']-off.batch['advantages'], expected, rtol=1e-6, atol=2e-6)
                self.assertFalse(torch.allclose(grad, off_grad))

    def test_episode_diagnostics_cannot_leak_into_alfworld_gradient(self):
        before, grad, _, _ = self.train('alfworld')
        after, changed_grad, _, _ = self.train('alfworld', episode_shift=99.)
        torch.testing.assert_close(before.batch['advantages'], after.batch['advantages'], rtol=0, atol=0)
        torch.testing.assert_close(grad, changed_grad, rtol=0, atol=0)
        self.assertNotEqual(before.meta_info['ccpo_diag']['ccpo/adv_ep_absmean'], after.meta_info['ccpo_diag']['ccpo/adv_ep_absmean'])


if __name__ == '__main__':
    unittest.main()
