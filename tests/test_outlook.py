"""CPU tests for future outlook credit, temporal alignment and padded batches."""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ.update(ACG_CCPO_PHI='hidden+ctx', ACG_CCPO_LAM_FIX='1',
                  ACG_CCPO_PRIOR_KAPPA='2', ACG_CCPO_EDGE_W='0',
                  ACG_CCPO_TARGET='return', ACG_CCPO_STD='task',
                  ACG_CCPO_JWEIGHT_C='0', ACG_CCPO_BACKOFF_TASK='1',
                  ACG_CCPO_LK_FIX='', ACG_CCPO_GATE='hard')
import importlib
from pathlib import Path
import sys
import unittest
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ccpo import core_ccpo as core
from ccpo.outlook import canonical_trajectory_rows, mix_outlook_returns, ccpo_outlook_advantage


def fixture():
    lengths = [4, 4, 3, 3]
    rewards, obs, trajs, turns, lens, eps, tasks = [], [], [], [], [], [], []
    for j, length in enumerate(lengths):
        outcome = 10.0 if j % 2 == 0 else 0.0
        for t in range(length):
            rewards.append(outcome if t == length - 1 else 0.0)
            obs.append('room' + str(t % 2)); trajs.append('traj' + str(j))
            turns.append(t); lens.append(length); eps.append(outcome); tasks.append('task')
    rewards = np.array(rewards)
    returns = np.zeros(len(rewards))
    offset = 0
    for length in lengths:
        running = 0.0
        for i in reversed(range(offset, offset + length)):
            running = rewards[i] + 0.95 * running
            returns[i] = running
        offset += length
    valid = np.ones(len(rewards), dtype=bool); valid[1] = False
    returns[1] -= 0.1
    return dict(step_rewards=torch.tensor(returns, dtype=torch.float32),
                response_mask=torch.ones(len(rewards), 4), anchor_obs=np.array(obs),
                index=np.array(tasks), traj_index=np.array(trajs),
                turn_index=np.array(turns), episode_lengths=np.array(lens),
                immediate_rewards=rewards, episode_rewards=np.array(eps),
                is_action_valid=valid,
                phi_feats=torch.tensor(np.random.default_rng(12).normal(size=(len(rewards), 8)), dtype=torch.float32),
                phi=core.FrozenPhi(), gamma=0.95, return_diag=True)


class OutlookTests(unittest.TestCase):
    def test_two_step_discount_and_mixture(self):
        # Rewards 1,2,3; MC returns 2.75,3.5,3 at gamma=.5.
        result, hist, outlook, raw, used, terminal = mix_outlook_returns(
            [2.75, 3.5, 3], [1, 2, 3], [1, 1, 1], [8, 8, 8],
            [True]*3, [np.arange(3)], 2, .25, .5)
        np.testing.assert_allclose(outlook, [3, 2.5, 2])
        np.testing.assert_allclose(result, .75 * hist + .25 * outlook)
        np.testing.assert_allclose(raw, [2.75, 3.5, 3])
        self.assertEqual(terminal.tolist(), [False, True, True])
        self.assertTrue(used.all())

    def test_terminal_success_and_failure_no_bootstrap(self):
        result, hist, out, *_ = mix_outlook_returns(
            [10, 0], [10, 0], [4, 4], [999, 999], [True, True],
            [np.array([0]), np.array([1])], 2, 1, .95)
        np.testing.assert_array_equal(result, [6, -4])
        np.testing.assert_array_equal(result, hist)

    def test_local_invalid_penalty_once(self):
        # Penalty on t=0 should remain -.1, with no additional penalty from t=1.
        result, _, out, *_ = mix_outlook_returns(
            [7.19, 8.0, 9, 10], [0, 0, 0, 10], [1]*4, [5]*4,
            [True]*4, [np.arange(4)], 2, 1, .9)
        self.assertAlmostEqual(out[0], .9**2 * 5 - .1 - 1)
        self.assertAlmostEqual(result[1], .9**2 * 5 - .1 - 1)

    def test_no_supported_endpoint_preserves_history(self):
        mixed, hist, _, _, used, _ = mix_outlook_returns(
            [1, 1, 1], [0, 0, 1], [0, 0, np.nan], [0, 0, np.nan],
            [True, True, False], [np.arange(3)], 2, .25, 1)
        self.assertEqual(mixed[0], hist[0]); self.assertFalse(used[0])
        self.assertEqual(mixed[2], 0)

    def test_exact_future_values_recover_mc(self):
        mixed, hist, *_ = mix_outlook_returns(
            [2.75, 3.5, 3], [1, 2, 3], [2.75, 3.5, 3], [2.75, 3.5, 3],
            [True]*3, [np.arange(3)], 2, .25, .5)
        np.testing.assert_allclose(mixed, hist)
        np.testing.assert_allclose(mixed, 0)

    def test_shuffle_and_padding_invariance(self):
        kw = fixture(); expected, diag = ccpo_outlook_advantage(**kw)
        rng = np.random.default_rng(15)
        order = rng.permutation(np.r_[np.arange(len(expected)), [0, 3, 7, 12]])
        moved = {k: (v[order] if isinstance(v, (np.ndarray, torch.Tensor)) else v) for k, v in kw.items()}
        actual, changed = ccpo_outlook_advantage(**moved)
        torch.testing.assert_close(actual, expected[order])
        np.testing.assert_allclose(changed['baseline_values'], diag['baseline_values'][order])
        self.assertEqual(changed['outlook_unique_rows'], len(expected))
        self.assertGreater(changed['outlook_padding_frac'], 0)

    def test_beta_zero_matches_ordered_history(self):
        kw = fixture()
        actual, _ = ccpo_outlook_advantage(beta=0, **kw)
        direct = {k:v for k,v in kw.items() if k not in ('turn_index', 'episode_lengths', 'immediate_rewards')}
        expected, _ = core.ccpo_step_advantage(**direct)
        torch.testing.assert_close(actual, expected)

    def test_future_baseline_excludes_own_trajectory_targets(self):
        kw = fixture()
        direct = {k:v for k,v in kw.items() if k not in ('turn_index', 'episode_lengths', 'immediate_rewards')}
        target = np.arange(len(kw['index']), dtype=float)
        _, d1 = core.ccpo_step_advantage(**direct, value_targets=target)
        target[:4] += 1000
        _, d2 = core.ccpo_step_advantage(**direct, value_targets=target)
        np.testing.assert_allclose(d1['bootstrap_values'][:4], d2['bootstrap_values'][:4])
        self.assertGreater(np.max(np.abs(d1['bootstrap_values'][4:] - d2['bootstrap_values'][4:])), 0)

    def test_future_channel_matches_independent_unpenalised_readout(self):
        kw = fixture()
        direct = {k:v for k,v in kw.items() if k not in ('turn_index', 'episode_lengths', 'immediate_rewards')}
        raw = kw['step_rewards'].numpy().copy(); raw[1] += .1
        _, d1 = core.ccpo_step_advantage(**direct, value_targets=raw)
        direct['step_rewards'] = torch.tensor(raw)
        _, d2 = core.ccpo_step_advantage(**direct)
        np.testing.assert_allclose(d1['bootstrap_values'], d2['baseline_values'], atol=1e-6)

    def test_credibility_blends_node_and_task_prior(self):
        # Two sibling trajectories; each query has J=1 and lambda_k=1/3.
        kw = dict(step_rewards=torch.tensor([1., 2., 6., 12.]), response_mask=torch.ones(4, 1),
                  anchor_obs=np.array(['a', 'b', 'a', 'b']), index=np.array(['task']*4),
                  traj_index=np.array(['left', 'left', 'right', 'right']),
                  phi=core.FrozenPhi(), return_diag=True, value_targets=np.array([1., 2., 6., 12.]))
        _, diag = core.ccpo_step_advantage(**kw)
        self.assertAlmostEqual(diag['bootstrap_values'][0], (1/3)*6 + (2/3)*9)
        self.assertAlmostEqual(diag['lam_k_mean'], 1/3)

    def test_missing_turn_fails(self):
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            canonical_trajectory_rows(['task']*2, ['traj']*2, [0, 2], [3, 3])

    def test_bad_configuration_fails(self):
        for kwargs in [dict(beta=-1), dict(beta=1.1), dict(horizon=0), dict(horizon=1.5), dict(edge_w=1), dict(target='score')]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                ccpo_outlook_advantage(**fixture(), **kwargs)

    def test_missing_hidden_features_fails(self):
        kw=fixture();kw.pop('phi_feats')
        with self.assertRaisesRegex(ValueError, 'reference features'):
            ccpo_outlook_advantage(**kw)

    def test_trainer_integration_both_normalization_modes(self):
        # Execute the real trainer function with its CPU dependencies. This checks
        # dispatch, scalar-to-token assignment, normalization, and metric export.
        import ast
        from collections import defaultdict
        from types import SimpleNamespace
        from unittest.mock import patch
        path = Path(__file__).resolve().parents[1] / 'patches/verl-agent/verl/trainer/ppo/ray_trainer.py'
        tree = ast.parse(path.read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'compute_advantage')
        names = ['GAE','GRPO','GRPO_PASSK','REINFORCE_PLUS_PLUS_BASELINE','REINFORCE_PLUS_PLUS',
                 'REMAX','RLOO','CCPO','GiGPO']
        enum = SimpleNamespace(**{n:n for n in names})
        sys.path.insert(0, str(path.parents[5] / 'verl-agent'))
        from gigpo import core_gigpo
        namespace = dict(DataProto=object, torch=torch, np=np, os=os,
                         defaultdict=defaultdict, AdvantageEstimator=enum, core_gigpo=core_gigpo)
        exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), 'exec'), namespace)
        kw = fixture()
        direct, _ = ccpo_outlook_advantage(**kw)
        for mode in ['mean_norm', 'mean_std_norm']:
            data = SimpleNamespace(batch={'step_rewards':kw['step_rewards'],
                'response_mask':kw['response_mask'], 'ccpo_phi_feats':kw['phi_feats'],
                'token_level_rewards':torch.zeros(len(direct),4)},
                non_tensor_batch={'anchor_obs':kw['anchor_obs'], 'uid':kw['index'],
                'traj_uid':kw['traj_index'], 'episode_rewards':kw['episode_rewards'],
                'episode_lengths':kw['episode_lengths'], 'is_action_valid':kw['is_action_valid'],
                'ccpo_turn_index':kw['turn_index'], 'rewards':kw['immediate_rewards']},meta_info={})
            with patch.dict(os.environ, {'ACG_CCPO_OUTLOOK_HORIZON':'2',
                    'ACG_CCPO_OUTLOOK_BETA':'.25','ACG_CCPO_EP_W':'0','ACG_CCPO_STEP_NORM':'mode'}):
                namespace['compute_advantage'](data, 'CCPO', gamma=.95, gigpo_mode=mode)
            expected = direct if mode == 'mean_norm' else (direct-direct.mean())/(direct.std()+1e-6)
            torch.testing.assert_close(data.batch['advantages'], expected[:,None]*kw['response_mask'])
            self.assertEqual(data.meta_info['ccpo_diag']['ccpo/outlook_beta'], .25)
            self.assertGreater(data.meta_info['ccpo_diag']['ccpo/outlook_delta_absmean'], 0)

    def test_complete_estimator_has_nonzero_finite_effect(self):
        actual, diag = ccpo_outlook_advantage(**fixture())
        self.assertTrue(torch.isfinite(actual).all())
        self.assertGreater(diag['outlook_delta_absmean'], 0)
        self.assertEqual(diag['outlook_beta'], .25)
        self.assertEqual(diag['outlook_horizon'], 2)
        self.assertEqual(diag['edge_cov'], 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
