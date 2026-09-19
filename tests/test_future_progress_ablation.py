"""M10 H2: credibility shrinkage and hidden-only context readouts."""
import importlib.util
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from test_future_progress import arguments, rows, core, np, torch, ccpo_future_progress_advantage

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('h2_ablation_registry', ROOT / 'ablations/ablations.py')
registry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(registry)
CONTROL = ROOT / 'experiments/m10-h2-ccpo-attncred-ctxadv-future-progress-alfworld-1.5b-2gpu-20260918'
CASES = {
    'future-progress-h2-no-context-vector': ('m10-h2-noctx-alfworld-1.5b-2gpu-20260919', {'ccpo_ctx_w': 0.0}),
    'future-progress-h2-kappa4': ('m10-h2-kappa4-alfworld-1.5b-2gpu-20260919', {'ccpo_prior_kappa': 4.0}),
    'future-progress-h2-no-credit-shrinkage': ('m10-h2-no-credit-shrinkage-alfworld-1.5b-2gpu-20260919', {'ccpo_lk_fix': 1.0}),
}


def fixture():
    kw = arguments()
    # Turn 3 has J=1; shared earlier turns have larger J. One unique anchor
    # exercises task-bucket fallback with the same finite, detached features.
    kw['anchor_obs'] = np.array(['turn-' + str(t) for t in kw['turn_index']], dtype=object)
    kw['anchor_obs'][0] = 'unique-anchor'
    return kw


def estimate(kw, kappa=2., fixed=''):
    with patch.object(core, '_PRIOR_KAPPA', kappa), patch.object(core, '_LK_FIX', fixed):
        return ccpo_future_progress_advantage(**kw, horizon=2)


class FutureProgressAblationTests(unittest.TestCase):
    def test_registered_and_prepared_configs_change_only_the_declared_setting(self):
        control = json.loads((CONTROL / 'config.json').read_text())['config']
        _, base = registry.arms.build('attncred-context-future-progress-h2', 'alfworld', '1.5b')
        for name, (folder, delta) in CASES.items():
            with self.subTest(name=name):
                _, cfg = registry.build(name, 'alfworld')
                self.assertEqual({k: v for k, v in cfg.items() if base.get(k) != v}, delta)
                self.assertEqual(registry.control_name('alfworld', name),
                                 'ccpo-attncred-ctxadv-future-progress-h2-alfworld-1.5b')
                with self.assertRaises(KeyError):
                    registry.build(name, 'webshop')
                record = json.loads((ROOT / 'experiments' / folder / 'config.json').read_text())
                actual = record['config']
                moved = {k: v for k, v in actual.items() if control.get(k) != v}
                self.assertEqual(moved, dict(delta, exp_id=folder))
                self.assertEqual(record['env']['ACG_CCPO_PROGRESS_HORIZON'], '2')
                self.assertEqual(float(record['env']['ACG_CCPO_EP_W']), 0.)
                self.assertEqual(float(record['env']['ACG_CCPO_EDGE_W']), 0.)
                self.assertEqual(float(record['env']['ACG_CCPO_LAM_FIX']), 1.)
                for key in ('ccpo_prior_kappa', 'ccpo_lk_fix', 'ccpo_ctx_w'):
                    env_name = {'ccpo_prior_kappa': 'ACG_CCPO_PRIOR_KAPPA', 'ccpo_lk_fix': 'ACG_CCPO_LK_FIX', 'ccpo_ctx_w': 'ACG_CCPO_CTX_W'}[key]
                    self.assertEqual(record['env'][env_name], str(actual[key]))

    def test_h2_noctx_uses_hidden_only_and_is_invariant_to_context_statistics(self):
        kw = fixture()
        kw['phi_feats'] = torch.tensor(np.random.default_rng(71).normal(
            size=(len(kw['index']), 1536)), dtype=torch.float32)
        with patch.object(core, '_CTX_W', 1.):
            _, control = estimate(kw)
        with patch.object(core, '_CTX_W', 0.):
            advantage, diag = estimate(kw)
            changed_kw = dict(kw, ctx_override=[dict(t=30, n_unique=25, progress=1., revisit=1.)
                                              for _ in kw['index']])
            changed, changed_diag = estimate(changed_kw)
        a = diag['progress_payload']['arrays']
        b = changed_diag['progress_payload']['arrays']
        expected = core.whiten_feats(kw['phi_feats'].numpy())
        self.assertEqual(control['progress_payload']['arrays']['current_phi'].shape[1], 1573)
        self.assertEqual(a['current_phi'].shape[1], 1536)
        np.testing.assert_allclose(a['current_phi'], expected, rtol=0, atol=0)
        nonterminal = ~a['terminal']
        np.testing.assert_array_equal(a['future_phi'][nonterminal], expected[a['endpoint_index'][nonterminal]])
        np.testing.assert_array_equal(a['future_hidden'][nonterminal],
                                      a['history_hidden'][a['endpoint_index'][nonterminal]])
        torch.testing.assert_close(advantage, changed, rtol=0, atol=0)
        for key in ['history_baseline', 'history_adv', 'current_value', 'future_value',
                    'raw_progress', 'progress_normalized', 'current_phi', 'future_phi']:
            np.testing.assert_array_equal(a[key], b[key])
        for prefix in ['history', 'current', 'future']:
            exact = a[prefix + '_level'] == 0
            J = a[prefix + '_J'][exact]
            np.testing.assert_allclose(a[prefix + '_lambda_k'][exact], J / (J + 2), rtol=0, atol=0)
        self.assertEqual(diag['progress_horizon'], 2.)
        self.assertFalse(advantage.requires_grad)

    def test_kappa_four_changes_history_and_both_potential_readouts(self):
        kw = fixture()
        before, old_diag = estimate(kw)
        after, diag = estimate(kw, kappa=4.)
        old, new = old_diag['progress_payload']['arrays'], diag['progress_payload']['arrays']
        for prefix, baseline in [('history', 'history_baseline'), ('current', 'current_value'), ('future', 'future_value')]:
            exact = (new[prefix + '_level'] == 0) & np.isfinite(new[prefix + '_kernel'])
            self.assertTrue(exact.any())
            J = new[prefix + '_J'][exact]
            weight = J / (J + 4.)
            np.testing.assert_allclose(new[prefix + '_lambda_k'][exact], weight, rtol=0, atol=0)
            np.testing.assert_allclose(new[baseline][exact],
                weight * new[prefix + '_kernel'][exact] + (1 - weight) * new[prefix + '_task_prior'][exact])
            self.assertTrue(np.all(new[prefix + '_lambda_k'][exact] < old[prefix + '_lambda_k'][exact]))
            np.testing.assert_array_equal(new[prefix + '_kernel'], old[prefix + '_kernel'])
        self.assertGreater(float((before - after).abs().max()), 1e-3)
        self.assertFalse(after.requires_grad)

    def test_no_shrink_is_full_context_including_one_peer_and_ignores_kappa(self):
        kw = fixture()
        advantage, diag = estimate(kw, fixed='1.0')
        a = diag['progress_payload']['arrays']
        self.assertTrue(((a['history_J'] == 1) & (a['history_level'] == 0)).any())
        for prefix, baseline in [('history', 'history_baseline'), ('current', 'current_value'), ('future', 'future_value')]:
            usable = np.isfinite(a[prefix + '_kernel'])
            np.testing.assert_array_equal(a[prefix + '_lambda_k'][usable], 1.)
            np.testing.assert_array_equal(a[baseline][usable], a[prefix + '_kernel'][usable])
        # It is the explicit full-strength pin, not a second kappa experiment.
        changed, changed_diag = estimate(kw, kappa=400., fixed='1.0')
        torch.testing.assert_close(advantage, changed, rtol=0, atol=0)
        np.testing.assert_array_equal(a['current_value'], changed_diag['progress_payload']['arrays']['current_value'])
        self.assertFalse(advantage.requires_grad)

    def test_fallback_and_terminal_conventions_do_not_change(self):
        kw = fixture()
        _, control = estimate(kw)
        before = control['progress_payload']['arrays']
        for kappa, fixed in [(4., ''), (2., '1.0')]:
            _, diag = estimate(kw, kappa, fixed)
            a = diag['progress_payload']['arrays']
            fallback = a['history_level'] == 1
            self.assertTrue(fallback.any())
            np.testing.assert_array_equal(a['history_baseline'][fallback], before['history_baseline'][fallback])
            np.testing.assert_array_equal(a['current_value'][fallback], before['current_value'][fallback])
            np.testing.assert_array_equal(a['history_lambda_k'][fallback], 1.)
            np.testing.assert_array_equal(a['future_value'][a['terminal']], a['episode_rewards'][a['terminal']])
            self.assertTrue(np.isnan(a['future_lambda_k'][a['terminal']]).all())
            np.testing.assert_array_equal(a['endpoint_index'], before['endpoint_index'])
            np.testing.assert_allclose(a['combined_pre'], a['history_adv'] + a['progress_normalized'])
            self.assertEqual(diag['progress_horizon'], 2.)

    def test_no_peer_anywhere_does_not_invent_a_context_baseline(self):
        kw = fixture()
        kw = rows(kw, np.flatnonzero(kw['traj_index'] == 'traj0'))
        for kappa, fixed in [(4., ''), (2., '1.0')]:
            with np.errstate(all='ignore'):
                advantage, diag = estimate(kw, kappa, fixed)
            a = diag['progress_payload']['arrays']
            self.assertFalse(a['history_live'].any())
            self.assertFalse(a['eligible'].any())
            np.testing.assert_array_equal(advantage.numpy(), 0.)


if __name__ == '__main__':
    unittest.main()
