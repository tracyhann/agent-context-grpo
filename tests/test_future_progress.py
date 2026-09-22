"""CPU tests for future-state progress, M5 equivalence and actual trainer gradients."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                  ACG_CCPO_TAU='.15', ACG_CCPO_WHITEN='3', ACG_CCPO_FIXED_ANCHOR='0',
                  ACG_CCPO_PROGRESS_HORIZON='0', ACG_CCPO_OUTLOOK_HORIZON='0', ACG_CCPO_OUTLOOK_BETA='0')
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'verl-agent'), str(ROOT/'tests')]
from test_outlook import fixture, core, np, torch
from test_episode_adv_isolation import load_compute_advantage, make_data, policy_gradient, core_gigpo
from ccpo.future_progress import (ccpo_future_progress_advantage, progress_from_values,
                                  standardize_progress, finalize_progress_logging)
from ccpo.outlook import canonical_trajectory_rows
import unittest
from unittest.mock import patch
from contextlib import redirect_stdout
import io, tempfile, json, csv


def arguments():
    kw = fixture(); kw.pop('immediate_rewards')
    return kw


def core_args(kw):
    return {k:v for k,v in kw.items() if k not in ('turn_index', 'episode_lengths')}


def rows(kw, ids):
    return {k:(v[ids] if isinstance(v, (np.ndarray, torch.Tensor)) else v) for k,v in kw.items()}


class FutureProgressTests(unittest.TestCase):
    def test_one_step_m5_reference_recovers_original_edge_and_combined_credit(self):
        kw = arguments()
        got, diag = ccpo_future_progress_advantage(**kw, readout='m5')
        expected, _ = core.ccpo_step_advantage(**dict(core_args(kw), edge_w=1.))
        torch.testing.assert_close(got, expected, rtol=1e-6, atol=1e-6)
        a = diag['progress_payload']['arrays']
        np.testing.assert_allclose(a['progress_normalized'], a['m5_edge_normalized'], atol=1e-12)
        self.assertAlmostEqual(diag['progress_future_edge_corr'], 1.)

    def test_m5_targets_discount_and_terminal_convention(self):
        kw = arguments(); _, d = ccpo_future_progress_advantage(**kw)
        a = d['progress_payload']['arrays']
        np.testing.assert_allclose(a['value_target'], .95**(a['episode_lengths']-a['turn_index'])*a['episode_rewards'])
        np.testing.assert_array_equal(a['future_value'][a['terminal']], a['episode_rewards'][a['terminal']])
        np.testing.assert_allclose(a['raw_progress'], a['future_value']-a['current_value'])
        self.assertNotEqual(a['value_target'][1], float(kw['step_rewards'][1]))

    def test_two_step_telescopes_and_clips_at_terminal(self):
        kw = arguments(); _, d = ccpo_future_progress_advantage(**kw, horizon=2)
        a = d['progress_payload']['arrays']; _, _, groups = canonical_trajectory_rows(kw['index'],kw['traj_index'],kw['turn_index'],kw['episode_lengths'])
        one, _, endpoint, _, _ = progress_from_values(a['current_value'], groups, a['episode_rewards'], 1)
        total = one.copy(); m = endpoint >= 0; total[m] += one[endpoint[m]]
        np.testing.assert_allclose(a['raw_progress'], total)
        self.assertTrue(np.all(a['window_length'][a['turn_index']==a['episode_lengths']-1] == 1))

    def test_weight_zero_retains_history_exactly(self):
        kw = arguments(); got, _ = ccpo_future_progress_advantage(**kw, progress_weight=0.)
        expected, _ = core.ccpo_step_advantage(**core_args(kw))
        torch.testing.assert_close(got, expected, rtol=0, atol=0)

    def test_future_uses_destination_group_from_other_origins(self):
        kw = arguments(); kw['anchor_obs'] = kw['anchor_obs'].astype(object)
        for j in range(4):
            ids = np.flatnonzero(kw['traj_index']=='traj'+str(j)); kw['anchor_obs'][ids[0]] = 'unique-start-'+str(j)
        _, d = ccpo_future_progress_advantage(**kw); a = d['progress_payload']['arrays']; starts = a['turn_index']==0
        self.assertTrue(np.all(a['current_level'][starts] > 0))
        self.assertTrue(np.all(a['future_level'][starts] == 0))
        self.assertTrue(np.all(a['future_J'][starts] == 3))
        np.testing.assert_array_equal(a['future_value'][starts], a['current_value'][a['endpoint_index'][starts]])

    def test_values_exclude_whole_query_trajectory(self):
        kw = arguments(); _, before = ccpo_future_progress_advantage(**kw)
        own = kw['traj_index']=='traj0'; kw['episode_rewards'][own] = 25.
        _, after = ccpo_future_progress_advantage(**kw)
        a,b = before['progress_payload']['arrays'],after['progress_payload']['arrays']
        np.testing.assert_allclose(a['current_value'][own],b['current_value'][own],atol=1e-12)
        keep = own & ~a['terminal']; np.testing.assert_allclose(a['future_value'][keep],b['future_value'][keep],atol=1e-12)
        self.assertGreater(np.max(abs(a['current_value'][~own]-b['current_value'][~own])), .1)

    def test_penalty_changes_history_but_not_progress(self):
        kw = arguments(); _, before = ccpo_future_progress_advantage(**kw)
        kw['step_rewards'][2] -= .1; kw['is_action_valid'][2] = False
        _, after = ccpo_future_progress_advantage(**kw)
        a,b = before['progress_payload']['arrays'],after['progress_payload']['arrays']
        for key in ['current_value','future_value','raw_progress','progress_normalized']:
            np.testing.assert_allclose(a[key],b[key],atol=0,rtol=0)
        self.assertGreater(np.max(abs(a['history_adv']-b['history_adv'])), .05)

    def test_shuffle_and_duplicate_padding_do_not_change_estimator(self):
        kw = arguments(); expected, _ = ccpo_future_progress_advantage(**kw)
        ids=np.random.default_rng(72).permutation(np.r_[np.arange(len(expected)),0,4,9])
        got, _ = ccpo_future_progress_advantage(**rows(kw,ids))
        torch.testing.assert_close(got,expected[ids],rtol=0,atol=0)
        padded = rows(kw, np.r_[np.arange(len(expected)),0]); padded['phi_feats'][-1,0] += 1
        with self.assertRaisesRegex(ValueError,'Inconsistent padded'):
            ccpo_future_progress_advantage(**padded)

    def test_no_other_trajectory_produces_no_supported_progress(self):
        kw = arguments(); kw=rows(kw,np.flatnonzero(kw['traj_index']=='traj0'))
        with np.errstate(all='ignore'):
            adv, d = ccpo_future_progress_advantage(**kw)
        self.assertFalse(d['progress_payload']['arrays']['eligible'].any())
        np.testing.assert_array_equal(adv.numpy(),0.)

    def test_invalid_settings_and_ambiguous_trajectory_fail(self):
        for extra in [{'horizon':0},{'horizon':5},{'progress_weight':float('nan')},{'edge_w':1.},{'target':'score'}]:
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                ccpo_future_progress_advantage(**dict(arguments(),**extra))
        kw=arguments();kw['episode_rewards'][1]=0
        with self.assertRaisesRegex(ValueError,'constant'):
            ccpo_future_progress_advantage(**kw)

    def test_trainer_components_logging_normalization_and_gradient_isolation(self):
        compute=load_compute_advantage()
        for mode in ['mean_norm','mean_std_norm']:
            for horizon in [1,2]:
                with self.subTest(mode=mode,horizon=horizon), tempfile.TemporaryDirectory() as tmp:
                    kw=fixture(); ep=torch.linspace(-2,2,len(kw['index']))[:,None]*kw['response_mask']; results=[]
                    env={'ACG_CCPO_PROGRESS_HORIZON':str(horizon),'ACG_CCPO_PROGRESS_WEIGHT':'1',
                         'ACG_CCPO_EP_W':'0','ACG_CCPO_FIXED_ANCHOR':'0','ACG_CCPO_STEP_NORM':'mode',
                         'ACG_CCPO_OUTLOOK_HORIZON':'0','ACG_CCPO_OUTLOOK_BETA':'0',
                         'ACG_EXP_DIR':tmp,'ACG_CCPO_PROGRESS_SNAPSHOT_EVERY':'1'}
                    for episode in [ep,ep*100+29]:
                        data=make_data(kw)
                        with patch.dict(os.environ,env), patch.object(core_gigpo,'episode_norm_reward',return_value=episode), redirect_stdout(io.StringIO()):
                            compute(data,'CCPO',gamma=.95,gigpo_mode=mode,ccpo_step_tag=1)
                        results.append((data,*policy_gradient(data)))
                    self.assertNotEqual(results[0][0].meta_info['ccpo_diag']['ccpo/adv_ep_absmean'],results[1][0].meta_info['ccpo_diag']['ccpo/adv_ep_absmean'])
                    for i in [1,2]:torch.testing.assert_close(results[0][i],results[1][i],rtol=0,atol=0)
                    out=Path(tmp)/'outputs/future_progress';metrics=json.loads((out/'step-0001.metrics.json').read_text())
                    with np.load(out/'step-0001.npz',allow_pickle=False) as a:
                        np.testing.assert_allclose(a['history_applied']+a['future_applied'],a['combined_applied'],atol=3e-5)
                        np.testing.assert_allclose(a['combined_applied'],results[0][0].batch['advantages'][:,0].numpy(),atol=1e-6)
                        for key in ['history_hidden','future_hidden','current_phi','future_phi','current_context','future_context','current_lambda_k','future_lambda_k','m5_edge_normalized','progress_norm_std']:
                            self.assertIn(key,a.files)
                        if mode=='mean_std_norm':self.assertAlmostEqual(float(torch.tensor(a['combined_applied']).std()),1.,places=5)
                        else:np.testing.assert_allclose(a['combined_pre'],a['combined_applied'],atol=1e-6)
                    self.assertLess(metrics['progress_applied_identity_error'],3e-5)
                    self.assertEqual(metrics['progress_episode_weight'],0.)
                    self.assertEqual(metrics['progress_original_edge_weight'],0.)
                    self.assertEqual(metrics['progress_weight'],1.)
                    with (out/'step-0001.csv').open() as fh:
                        data=list(csv.DictReader(fh));self.assertEqual(len(data),len(kw['index']))
                    for key in metrics:self.assertIn('ccpo/'+key,results[0][0].meta_info['ccpo_diag'])

    def test_trainer_rejects_duplicate_future_and_invalid_episode_weight(self):
        compute=load_compute_advantage()
        for extra in [{'ACG_CCPO_FIXED_ANCHOR':'1'},{'ACG_CCPO_OUTLOOK_HORIZON':'2','ACG_CCPO_OUTLOOK_BETA':'.25'},{'ACG_CCPO_EP_W':'-1'},{'ACG_CCPO_EP_W':'nan'}]:
            with self.subTest(extra=extra),patch.dict(os.environ,dict({'ACG_CCPO_PROGRESS_HORIZON':'1','ACG_CCPO_EP_W':'0'},**extra)),self.assertRaises(ValueError),redirect_stdout(io.StringIO()):
                compute(make_data(fixture()),'CCPO',gamma=.95,ccpo_step_tag=1)


class PaddedDuplicateFeatureTests(unittest.TestCase):
    """Finite roundoff is allowed; nonfinite rows fail in every padding order."""

    def padded(self, index=0):
        kw = arguments()
        return rows(kw, np.r_[np.arange(len(kw['index'])), index])

    def orders(self, n):
        return [np.arange(n), np.r_[n-1, np.arange(n-1)],
                np.random.default_rng(72).permutation(n)]

    def test_unpadded_features_report_zero_duplicates(self):
        _, diag = ccpo_future_progress_advantage(**arguments())
        for key in ('rows', 'max_diff', 'nonfinite'):
            self.assertEqual(diag['progress_phi_duplicate_' + key], 0.)

    def test_exact_duplicate_is_accepted_and_counted(self):
        _, diag = ccpo_future_progress_advantage(**self.padded(), horizon=2)
        self.assertEqual(diag['progress_phi_duplicate_rows'], 1.)
        self.assertEqual(diag['progress_phi_duplicate_max_diff'], 0.)
        self.assertEqual(diag['progress_phi_duplicate_nonfinite'], 0.)
        self.assertAlmostEqual(diag['progress_padding_frac'], 1/15)

    def test_bf16_duplicate_does_not_change_credit_or_gradients(self):
        for ctx_weight in (0., 1.):
            for horizon in (1, 2):
                with self.subTest(ctx_weight=ctx_weight, horizon=horizon), patch.object(core, '_CTX_W', ctx_weight):
                    kw = self.padded()
                    expected, before = ccpo_future_progress_advantage(**kw, horizon=horizon)
                    kw['phi_feats'][-1] = kw['phi_feats'][-1].to(torch.bfloat16).float()
                    kw['phi_feats'].requires_grad_()
                    actual, after = ccpo_future_progress_advantage(**kw, horizon=horizon)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    self.assertFalse(actual.requires_grad)
                    self.assertGreater(after['progress_phi_duplicate_max_diff'], 0.)
                    for key in ('history_adv', 'current_value', 'future_value', 'raw_progress',
                                'progress_normalized', 'current_phi', 'combined_pre'):
                        np.testing.assert_array_equal(before['progress_payload']['arrays'][key],
                                                      after['progress_payload']['arrays'][key])
                    # Same padded layout, changed only discarded frozen features.
                    # The actual trainer must retain identical PPO loss/gradients.
                    for mode in ('mean_norm', 'mean_std_norm'):
                        compute = load_compute_advantage()
                        reference = fixture(); padded_ids = np.r_[np.arange(len(reference['index'])), 0]
                        reference = rows(reference, padded_ids)
                        changed = rows(reference, np.arange(len(reference['index'])))
                        changed['phi_feats'][-1] = changed['phi_feats'][-1].to(torch.bfloat16).float()
                        settings = {'ACG_CCPO_PROGRESS_HORIZON': str(horizon), 'ACG_CCPO_PROGRESS_WEIGHT': '1',
                                    'ACG_CCPO_EP_W': '0', 'ACG_CCPO_STEP_NORM': 'mode', 'ACG_EXP_DIR': ''}
                        results = []
                        for inputs in (reference, changed):
                            data = make_data(inputs)
                            with patch.dict(os.environ, settings), redirect_stdout(io.StringIO()):
                                compute(data, 'CCPO', gamma=.95, gigpo_mode=mode, ccpo_step_tag=1)
                            results.append(policy_gradient(data))
                        for left, right in zip(*results):
                            torch.testing.assert_close(left, right, rtol=0, atol=0)

    def test_finite_rounding_is_accepted_after_reordering(self):
        kw = self.padded()
        kw['phi_feats'][-1] = kw['phi_feats'][-1].to(torch.bfloat16).float()
        for order in self.orders(len(kw['index'])):
            with self.subTest(order=order.tolist()):
                actual, diag = ccpo_future_progress_advantage(**rows(kw, order))
                self.assertTrue(torch.isfinite(actual).all())
                self.assertEqual(diag['progress_phi_duplicate_rows'], 1.)
                self.assertGreater(diag['progress_phi_duplicate_max_diff'], 0.)

    def test_nonfinite_features_fail_without_padding(self):
        for value in (float('nan'), float('inf'), -float('inf')):
            with self.subTest(value=value):
                kw = arguments(); kw['phi_feats'][0, 0] = value
                with self.assertRaisesRegex(ValueError, 'Non-finite frozen features'):
                    ccpo_future_progress_advantage(**kw)

    def test_nonfinite_copies_fail_in_every_order_before_estimation(self):
        for value in (float('nan'), float('inf'), -float('inf')):
            for bad_row in (0, -1):
                for partial in (False, True):
                    kw = self.padded()
                    if partial:
                        kw['phi_feats'][bad_row, 0] = value
                    else:
                        kw['phi_feats'][bad_row] = value
                    for order in self.orders(len(kw['index'])):
                        with self.subTest(value=value, bad_row=bad_row, partial=partial,
                                          order=order.tolist()), patch.object(core, 'ccpo_step_advantage') as estimator:
                            with self.assertRaisesRegex(ValueError, 'Non-finite frozen features'):
                                ccpo_future_progress_advantage(**rows(kw, order))
                            estimator.assert_not_called()

    def test_different_or_zero_filled_duplicate_still_fails_after_reordering(self):
        for fill_zero in (False, True):
            kw = self.padded()
            kw['phi_feats'][-1] = 0. if fill_zero else kw['phi_feats'][-1] + 1.
            for order in self.orders(len(kw['index'])):
                with self.subTest(fill_zero=fill_zero, order=order.tolist()):
                    with self.assertRaisesRegex(ValueError, 'beyond bf16 tolerance'):
                        ccpo_future_progress_advantage(**rows(kw, order))

    def test_metadata_mismatch_is_still_exact(self):
        for key in ('step_rewards', 'response_mask', 'episode_rewards', 'is_action_valid',
                    'anchor_obs', 'index', 'traj_index'):
            with self.subTest(key=key):
                kw = self.padded()
                if key in ('anchor_obs', 'index', 'traj_index'):
                    kw[key] = kw[key].astype(object); kw[key][-1] = 'different-metadata'
                elif key == 'is_action_valid':
                    kw[key][-1] = not kw[key][-1]
                else:
                    kw[key][-1] += 1
                with self.assertRaises(ValueError):
                    ccpo_future_progress_advantage(**kw)


if __name__=='__main__':
    unittest.main(verbosity=2)
