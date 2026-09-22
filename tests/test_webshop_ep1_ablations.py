"""Matched WebShop EP1 peer/feature ablations exercise actual trainer and PPO."""
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from test_future_progress import core, np, torch, load_compute_advantage, policy_gradient
from test_future_progress_active_episode import training_fixture, reward_data

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import prepare_webshop_ep1_ablations as prep

STAMP = '20260922'


def record(variant):
    folder = ROOT/'experiments'/prep.run_name(variant, STAMP)
    return folder, json.loads((folder/'config.json').read_text())


class WebshopEp1Tests(unittest.TestCase):
    def test_full_configs_change_one_setting_from_the_active_episode_control(self):
        control = json.loads((ROOT/'experiments'/prep.CONTROL/'config.json').read_text())['config']
        for k, v in prep.IMPLIED.items():
            control.setdefault(k, v)
        for variant, spec in prep.CASES.items():
            folder, saved = record(variant); cfg = saved['config']
            self.assertEqual(set(cfg), set(control))
            self.assertEqual({k for k in cfg if cfg[k] != control[k]}, {'exp_id', spec['flag']})
            self.assertEqual(cfg[spec['flag']], spec['value'])
            for k in ['ccpo_ep_w', 'ccpo_lk_fix', 'ccpo_lam_fix',
                      'ccpo_progress_weight', 'ccpo_progress_history_weight']:
                self.assertEqual(cfg[k], 1.)
            self.assertEqual(cfg['ccpo_edge_w'], 0.)
            self.assertEqual(cfg['history_length'], 2)
            self.assertEqual(cfg['ccpo_progress_horizon'], 2)
            self.assertEqual(cfg['total_epochs'], 150)
            self.assertEqual(cfg['max_steps'], 15)
            self.assertEqual(cfg['adv_mode'], 'mean_norm')
            self.assertEqual(cfg['model'], 'Qwen/Qwen2.5-1.5B-Instruct')
            _, built = prep.ablations.build(spec['key'], 'webshop')
            for k, v in built.items():
                self.assertEqual(cfg[k], v, k)
            self.assertEqual(saved['hydra_overrides'], prep.er.build_command(cfg, str(folder))[3:])
            for k, env in prep.er.ENV_KEYS.items():
                self.assertEqual(saved['env'][env], str(cfg[k]))
            self.assertIn('trainer.n_gpus_per_node=2', saved['hydra_overrides'])
            self.assertIn('  ACG_CCPO_EP_W=1.0 ', (folder/'run.sh').read_text())
            status = json.loads((folder/'PREPARED.json').read_text())
            self.assertFalse(status['launched']); self.assertFalse(status['queued'])
            self.assertFalse(any(p.is_file() for p in (folder/'outputs').rglob('*')))
            self.assertIn('--dry-run', json.loads((folder/'prepare-command.json').read_text())['argv'])
        # The existing cosine implementation, not a second estimator/registry arm.
        old = json.loads((ROOT/'experiments/m11-h2-noshrink-active-episode-cosine-webshop-1.5b-2gpu-20260920/config.json').read_text())['config']
        for k, v in prep.IMPLIED.items():
            old.setdefault(k, v)
        new = record('cosine')[1]['config']
        self.assertEqual({k for k in old if old[k] != new[k]}, {'exp_id'})

    def test_preparation_refuses_overwrite_before_spawning_a_process(self):
        for variant in prep.CASES:
            with patch.object(prep.subprocess, 'run') as run:
                with self.assertRaises(FileExistsError):
                    prep.prepare(variant, STAMP)
                run.assert_not_called()

    def test_prepared_environment_selects_all_three_modes_in_a_fresh_process(self):
        code = '''
import os
from ccpo import core_ccpo as c
assert c._LOO == os.environ['ACG_CCPO_LOO']
assert c._CTX_W == float(os.environ['ACG_CCPO_CTX_W'])
assert c._WMODE == os.environ['ACG_CCPO_WMODE']
assert float(c._LK_FIX) == float(c._LAM_FIX) == 1
assert os.environ['ACG_CCPO_EP_W'] == '1.0'
assert os.environ['ACG_CCPO_PROGRESS_HORIZON'] == '2'
assert c._PHI_MODE == 'hidden+ctx'
print('EP1 runtime settings loaded')
'''
        for variant in prep.CASES:
            saved = record(variant)[1]
            env = dict(os.environ, **saved['env'])
            env.update(CUDA_VISIBLE_DEVICES='', ACG_EXP_DIR='', ACG_CCPO_DUMP='')
            result = subprocess.run([sys.executable, '-c', code], cwd=ROOT, env=env,
                                    capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def evaluate(self, variant, episode_weight=1, perturb=False):
        saved = record(variant)[1]; cfg = saved['config']
        kw = training_fixture()
        # Production 1.5B feature width; deterministic fixture with unequal lengths.
        kw['phi_feats'] = torch.tensor(np.random.default_rng(92).normal(
            size=(len(kw['index']), 1536)), dtype=torch.float32)
        data = reward_data(kw)
        scores = data.batch['token_level_rewards'].sum(-1)
        expected_episode = (scores - scores.mean())[:, None] * kw['response_mask']
        original_context = core.derive_context

        def context(*args, **kwargs):
            values = original_context(*args, **kwargs)
            if perturb and variant == 'no-context-vector':
                return [dict(v, t=99, n_unique=37, progress=.01, revisit=1.) for v in values]
            return values

        with tempfile.TemporaryDirectory() as tmp:
            env = dict(saved['env'], CUDA_VISIBLE_DEVICES='', ACG_EXP_DIR=tmp,
                       ACG_CCPO_DUMP='', ACG_CCPO_EP_W=str(episode_weight))
            with patch.dict(os.environ, env), patch.multiple(core,
                _LK_FIX='1.0', _LAM_FIX='1.0', _LOO=str(cfg['ccpo_loo']),
                _CTX_W=cfg['ccpo_ctx_w'], _PHI_MODE=cfg['ccpo_phi'],
                _WMODE=cfg['ccpo_wmode'], _WHITEN_K=3, _GATE='hard',
                _TAU_ENV='19' if perturb and variant == 'cosine' else '.15',
                _JW_C=0., _STD_MODE='task', _BACKOFF_TASK=True,
                _SIM=0., _SIM_BACKOFF=0., _EDGE_W=0.), \
                patch.object(core, 'derive_context', side_effect=context), redirect_stdout(io.StringIO()):
                load_compute_advantage()(data, 'CCPO', gamma=.95,
                                         gigpo_mode='mean_norm', ccpo_step_tag=1)
            with np.load(Path(tmp)/'outputs/future_progress/step-0001.npz', allow_pickle=False) as saved_arrays:
                arrays = {k: v.copy() for k, v in saved_arrays.items()}
        return data, policy_gradient(data), arrays, expected_episode, kw

    def test_real_actor_fusion_peer_membership_features_and_weighting(self):
        for variant in prep.CASES:
            on, (_, grad), a, ep, kw = self.evaluate(variant)
            off, (_, grad_off), b, _, _ = self.evaluate(variant, episode_weight=0)
            mask = kw['response_mask']
            torch.testing.assert_close(on.batch['advantages'] - off.batch['advantages'], ep,
                                       rtol=1e-6, atol=2e-6)
            self.assertFalse(torch.allclose(grad, grad_off))
            self.assertTrue(torch.isfinite(grad).all())
            self.assertGreater(float(grad.abs().sum()), 0.)
            self.assertFalse(on.batch['advantages'].requires_grad)
            torch.testing.assert_close(on.batch['advantages'][mask == 0], torch.zeros_like(on.batch['advantages'][mask == 0]))
            np.testing.assert_allclose(a['actor_applied'], a['history_applied'] +
                                       a['future_applied'] + a['episode_applied'], atol=3e-5)
            np.testing.assert_allclose(a['episode_applied'][:, None] * mask.numpy(), ep.numpy(), atol=1e-6)
            for channel in ['history', 'current', 'future']:
                finite = np.isfinite(a[channel + '_kernel'])
                np.testing.assert_array_equal(a[channel + '_lambda_k'][finite], 1.)
                value = {'history':'history_baseline', 'current':'current_value', 'future':'future_value'}[channel]
                np.testing.assert_array_equal(a[value][finite], a[channel + '_kernel'][finite])
                masses = a[channel + '_same_traj_mass']; finite = np.isfinite(masses)
                if variant == 'no-loo':
                    self.assertTrue((masses[finite] > 0).all())
                else:
                    np.testing.assert_array_equal(masses[finite], 0.)
            self.assertEqual(on.meta_info['ccpo_diag']['ccpo/progress_episode_weight'], 1.)
            self.assertEqual(a['current_phi'].shape[1], 1536 if variant == 'no-context-vector' else 1573)
            if variant == 'no-context-vector':
                np.testing.assert_array_equal(a['current_phi'], core.whiten_feats(kw['phi_feats'].numpy()))
            if variant == 'cosine':
                # Independent oracle on supported exact groups: same z, clipped
                # cosine weights, whole-trajectory exclusion, no shrinkage.
                z = a['current_phi']; checked = 0
                for i in range(len(z)):
                    peers = np.flatnonzero((a['uid'] == a['uid'][i]) &
                        (a['anchor_obs'] == a['anchor_obs'][i]) & (a['traj_uid'] != a['traj_uid'][i]))
                    if not len(peers):
                        continue
                    w = np.maximum(z[peers] @ z[i] /
                                   (np.linalg.norm(z[peers], axis=1) * np.linalg.norm(z[i])), 0)
                    if not w.sum():
                        w[np.argmin(np.linalg.norm(z[peers] - z[i], axis=1))] = 1
                    self.assertAlmostEqual(a['current_value'][i], float(w @ a['value_target'][peers] / w.sum()), places=6)
                    checked += 1
                self.assertGreater(checked, 0)
            if variant in ('no-context-vector', 'cosine'):
                changed, (_, changed_grad), _, _, _ = self.evaluate(variant, perturb=True)
                torch.testing.assert_close(changed.batch['advantages'], on.batch['advantages'], rtol=0, atol=0)
                torch.testing.assert_close(changed_grad, grad, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
