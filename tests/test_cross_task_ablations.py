"""Cross-task kernels: independent oracle, peer eligibility, LOO and PPO gradients."""
from contextlib import redirect_stdout
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
from test_uniform_peer_ablations import settings, estimate
from ccpo.cross_task import peer_records

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import prepare_cross_task_ablations as prep
# WebShop was re-prepared against its EP1 control; ALFWorld keeps its EP0 arm.
STAMPS = {'alfworld': '20260928', 'webshop': '20261007'}


def batch():
    kw = fixture()
    kw['index'] = np.array(['task-a' if int(t[-1]) < 2 else 'task-b' for t in kw['traj_index']])
    # Same observation strings exist across tasks, while each trajectory also has
    # a unique starting observation. Neither condition should gate cross-task peers.
    kw['anchor_obs'] = kw['anchor_obs'].astype(object)
    kw['anchor_obs'][kw['turn_index'] == 0] = kw['traj_index'][kw['turn_index'] == 0]
    kw['response_mask'][::2, -1] = 0
    return kw


def cross_settings(**extra):
    return settings(weighting='soft', _GATE='cross_task', **extra)


def record(benchmark):
    folder = ROOT/'experiments'/prep.run_name(benchmark, STAMPS[benchmark])
    return folder, json.loads((folder/'config.json').read_text())


class CrossTaskTests(unittest.TestCase):
    def test_manual_kernel_history_current_future_support_and_cross_task_mass(self):
        with cross_settings():
            _, diag = estimate(batch())
        a = diag['progress_payload']['arrays']; f = a['current_phi']
        distances = np.linalg.norm(f[:, None] - f[None, :], axis=-1)
        tau = .15 * np.median(distances[np.triu_indices(len(f), 1)])
        values = a['value_target'].astype(np.float32).astype(float)
        for i in range(len(f)):
            peers = np.flatnonzero(a['traj_uid'] != a['traj_uid'][i])
            weights = np.exp(-distances[i, peers]/tau); weights /= weights.sum()
            tr = a['traj_uid'][peers]
            masses = np.array([weights[tr == t].sum() for t in np.unique(tr)])
            for channel, labels, baseline in [('history', a['target'], 'history_baseline'),
                                               ('current', values, 'current_value')]:
                self.assertAlmostEqual(a[baseline][i], weights @ labels[peers], places=7)
                self.assertAlmostEqual(a[channel+'_kernel'][i], a[baseline][i], places=10)
                self.assertAlmostEqual(a[channel+'_cross_task_mass'][i], weights[a['uid'][peers] != a['uid'][i]].sum())
                self.assertAlmostEqual(a[channel+'_n_eff'][i], 1/(masses @ masses))
                self.assertAlmostEqual(a[channel+'_kernel_tau'][i], tau)
                self.assertEqual(a[channel+'_J'][i], 3)
                self.assertEqual(a[channel+'_peer_tasks'][i], 2)
                self.assertEqual(a[channel+'_peer_rows'][i], len(peers))
                self.assertEqual(a[channel+'_same_traj_mass'][i], 0)
                self.assertEqual(a[channel+'_lambda_k'][i], 1)
                self.assertTrue(np.isnan(a[channel+'_task_prior'][i]))
        terminal = a['terminal']; endpoint = a['endpoint_index']
        np.testing.assert_array_equal(a['future_value'][~terminal], a['current_value'][endpoint[~terminal]])
        np.testing.assert_array_equal(a['future_value'][terminal], a['episode_rewards'][terminal])
        np.testing.assert_allclose(a['raw_progress'], a['future_value']-a['current_value'])
        for task in np.unique(a['uid']):
            ids = a['uid'] == task; p = a['raw_progress'][ids]
            np.testing.assert_allclose(a['progress_normalized'][ids], (p-p.mean())/(p.std(ddof=1)+1e-6))
        self.assertEqual(diag['progress_cross_task_grouping'], 1)
        self.assertEqual(diag['progress_current_exact_frac'], 0)
        self.assertEqual(diag['progress_current_backoff_frac'], 0)

    def test_other_task_rewards_change_both_readouts_but_hard_and_global_remain_task_local(self):
        kw = batch(); changed = batch(); remote = changed['index'] == 'task-b'
        changed['episode_rewards'][remote] += 4
        changed['step_rewards'][remote] += 4
        query = kw['index'] == 'task-a'
        with cross_settings():
            _, d = estimate(kw); _, e = estimate(changed)
        a, b = d['progress_payload']['arrays'], e['progress_payload']['arrays']
        for key in ['history_baseline', 'current_value']:
            self.assertGreater(np.max(abs(a[key][query]-b[key][query])), .1)
        for gate in ['hard', 'global']:
            args = {k:v for k,v in kw.items() if k not in ('turn_index','episode_lengths','immediate_rewards')}
            edited = {k:v for k,v in changed.items() if k in args}
            with settings(weighting='soft', _GATE=gate):
                _, a = core.ccpo_step_advantage(**args); _, b = core.ccpo_step_advantage(**edited)
            np.testing.assert_array_equal(a['baseline_values'][query], b['baseline_values'][query])

    def test_task_observation_labels_do_not_gate_and_single_trajectory_tasks_still_have_peers(self):
        kw = batch()
        kw['ctx_override'] = core.derive_context(kw['anchor_obs'], kw['index'], kw['traj_index'])
        changed = dict(kw, index=np.array(['only-'+t for t in kw['traj_index']]),
                       anchor_obs=np.array(['unique-'+str(i) for i in range(len(kw['index']))]))
        with cross_settings():
            _, d = estimate(kw); _, e = estimate(changed)
        a, b = d['progress_payload']['arrays'], e['progress_payload']['arrays']
        np.testing.assert_array_equal(a['traj_uid'], b['traj_uid'])
        for key in ['history_baseline','current_value','future_value','raw_progress']:
            np.testing.assert_allclose(a[key], b[key], atol=1e-7)
        self.assertTrue(b['live'].all())
        np.testing.assert_allclose(b['history_cross_task_mass'], 1.)
        np.testing.assert_array_equal(b['history_peer_tasks'], 3.)
        # Observation may still affect derived context/prompt embeddings; this
        # test freezes those features to isolate the hard matching decision.

    def test_whole_trajectory_loo_and_penalty_isolation(self):
        kw = batch(); own = kw['traj_index'] == 'traj0'
        with cross_settings():
            _, d = estimate(kw)
            kw['step_rewards'][own] += 100; kw['episode_rewards'][own] += 100
            _, e = estimate(kw)
            penalty = batch(); penalty['step_rewards'][2] -= .1
            penalty['is_action_valid'][2] = False
            _, p = estimate(penalty)
        a,b,c = [x['progress_payload']['arrays'] for x in (d,e,p)]
        for key in ['history_baseline','current_value']:
            np.testing.assert_array_equal(a[key][own], b[key][own])
        nonterminal = own & ~a['terminal']
        np.testing.assert_array_equal(a['future_value'][nonterminal], b['future_value'][nonterminal])
        for key in ['current_value','future_value','raw_progress','progress_normalized']:
            np.testing.assert_array_equal(a[key], c[key])
        self.assertGreater(np.max(abs(a['history_adv']-c['history_adv'])), .05)

    def test_padding_order_unsupported_batches_and_inactive_prior(self):
        kw = batch()
        with cross_settings():
            expected, _ = estimate(kw)
            ids = np.random.default_rng(77).permutation(np.r_[np.arange(len(expected)),0,4,9])
            actual, d = estimate(rows(kw, ids))
            torch.testing.assert_close(actual, expected[ids], rtol=0, atol=0)
            self.assertGreater(d['progress_padding_frac'], 0)
            single = rows(kw, np.flatnonzero(kw['traj_index'] == 'traj0'))
            unsupported, d = estimate(single)
            np.testing.assert_array_equal(unsupported.numpy(), 0)
            self.assertFalse(d['progress_payload']['arrays']['eligible'].any())
        with cross_settings(_BACKOFF_TASK=False, _PRIOR_KAPPA=4000.):
            same, _ = estimate(kw)
        torch.testing.assert_close(same, expected, rtol=0, atol=0)

    def test_similarity_affects_credit_and_identical_features_reduce_to_occurrence_mean(self):
        with cross_settings():
            expected, _ = estimate(batch())
            kw = batch(); kw['phi_feats'] = torch.tensor(np.random.default_rng(55).normal(size=kw['phi_feats'].shape), dtype=torch.float32)
            changed, _ = estimate(kw)
        self.assertFalse(torch.allclose(expected, changed))
        records, diag = peer_records(np.ones((5,3)), [1,3,5,7,9], ['a','a','b','b','b'], ['x','x','y','y','z'], .15)
        for rec in records:
            peers = np.array(['x','x','y','y','z']) != ['x','x','y','y','z'][rec['i']]
            self.assertAlmostEqual(rec['b_loo'], np.array([1,3,5,7,9])[peers].mean())
        self.assertEqual(diag['tau'], .15)
        # A very narrow kernel must stay finite, including for distant peers.
        records, _ = peer_records([[0.],[100.],[101.]], [1.,4.,9.], ['a','b','c'], ['x','y','z'], 1e-12)
        self.assertEqual(records[0]['b_loo'], 4.)
        self.assertTrue(all(np.isfinite(r['b_loo']) for r in records))

    def test_config_guards(self):
        cfg = dict(prep.er.DEFAULTS, **prep.ablations.build(prep.key_for('alfworld'), 'alfworld')[1])
        prep.er.validate_future_progress_config(cfg)
        for key, value in [('ccpo_loo',0),('ccpo_wmode','uniform'),('ccpo_lk_fix',.5),
                           ('ccpo_lam_fix',.5),('ccpo_sim',.9),('ccpo_sim_backoff',.9),
                           ('ccpo_tau',0),('ccpo_tau',float('nan'))]:
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'Cross-task'):
                prep.er.validate_future_progress_config(dict(cfg, **{key:value}))
        for key,value in [('_LOO','0'),('_WMODE','uniform'),('_LK_FIX','.5'),('_SIM',.9)]:
            with self.subTest(key=key), cross_settings(**{key:value}), self.assertRaises(ValueError):
                estimate(batch())

    def test_prepared_configs_change_only_gate_and_runtime_reads_it(self):
        for benchmark in prep.CONTROLS:
            folder, saved = record(benchmark); cfg = saved['config']
            control = json.loads((ROOT/'experiments'/prep.CONTROLS[benchmark]/'config.json').read_text())['config']
            for key, value in prep.IMPLIED.items(): control.setdefault(key,value)
            self.assertEqual(set(cfg),set(control))
            self.assertEqual({k for k in cfg if cfg[k] != control[k]}, {'exp_id','ccpo_gate'})
            self.assertEqual(cfg['ccpo_gate'],'cross_task'); self.assertEqual(cfg['ccpo_ep_w'],int(benchmark=='webshop'))
            self.assertEqual(cfg['model'],'Qwen/Qwen2.5-1.5B-Instruct')
            self.assertEqual(saved['hydra_overrides'],prep.er.build_command(cfg,str(folder))[3:])
            self.assertIn('ACG_CCPO_GATE=cross_task', (folder/'run.sh').read_text())
            for key,env in ((k, v) for k, v in prep.er.ENV_KEYS.items() if k != 'ccpo_progress_mode' or k in cfg): self.assertEqual(saved['env'][env],str(cfg[key]))
            self.assertFalse(saved['preparation']['launched']); self.assertFalse(saved['preparation']['queued'])
            result=subprocess.run([sys.executable,'-c',"from ccpo import core_ccpo as c; assert c._GATE == 'cross_task'; assert c._WMODE == 'soft'; assert float(c._LK_FIX) == 1"],
                cwd=ROOT,env=dict(os.environ,**saved['env']),capture_output=True,text=True,timeout=60)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            with patch.object(prep.subprocess,'run') as run:
                with self.assertRaises(FileExistsError): prep.prepare(benchmark,STAMPS[benchmark])
                run.assert_not_called()

    def train(self, benchmark, episode_shift=0., changed_features=False):
        _, saved = record(benchmark); kw=batch()
        if changed_features:
            kw['phi_feats']=torch.tensor(np.random.default_rng(55).normal(size=kw['phi_feats'].shape),dtype=torch.float32)
        data=reward_data(kw); original=core_gigpo.episode_norm_reward
        def episode(*args,**kwargs): return original(*args,**kwargs)+episode_shift*kw['response_mask']
        with tempfile.TemporaryDirectory() as tmp:
            env=dict(saved['env'],CUDA_VISIBLE_DEVICES='',ACG_EXP_DIR=tmp,ACG_CCPO_DUMP='')
            with patch.dict(os.environ,env),cross_settings(),patch.object(core_gigpo,'episode_norm_reward',side_effect=episode):
                load_compute_advantage()(data,'CCPO',gamma=.95,gigpo_mode=saved['config']['adv_mode'],ccpo_step_tag=1)
            with np.load(Path(tmp)/'outputs/future_progress/step-0001.npz',allow_pickle=False) as file:
                a={k:v.copy() for k,v in file.items()}
        return data,policy_gradient(data)[1],a,kw

    def test_actual_trainer_ppo_gradients_masks_normalization_and_episode_weight_in_both_benchmarks(self):
        for benchmark in prep.CONTROLS:
            with self.subTest(benchmark=benchmark):
                ep=int(benchmark=='webshop')
                data,grad,a,kw=self.train(benchmark)
                shifted,shift_grad,_,_=self.train(benchmark,episode_shift=99)
                changed,changed_grad,_,_=self.train(benchmark,changed_features=True)
                mask=kw['response_mask']
                if ep:
                    torch.testing.assert_close(shifted.batch['advantages']-data.batch['advantages'],99.*mask,rtol=1e-6,atol=3e-5)
                    self.assertFalse(torch.allclose(grad,shift_grad))
                else:
                    torch.testing.assert_close(data.batch['advantages'],shifted.batch['advantages'],rtol=0,atol=0)
                    torch.testing.assert_close(grad,shift_grad,rtol=0,atol=0)
                    np.testing.assert_array_equal(a['episode_applied'],0)
                self.assertFalse(torch.allclose(grad,changed_grad))
                self.assertTrue(torch.isfinite(grad).all()); self.assertGreater(grad.abs().sum(),0)
                self.assertFalse(data.batch['advantages'].requires_grad)
                np.testing.assert_array_equal(data.batch['advantages'][mask==0].numpy(),0)
                np.testing.assert_allclose(a['actor_applied'],a['history_applied']+a['future_applied']+a['episode_applied'],atol=3e-5)
                np.testing.assert_allclose(a['actor_applied'],data.batch['advantages'][:,0].numpy(),atol=1e-6)
                self.assertEqual(data.meta_info['ccpo_diag']['ccpo/progress_cross_task_grouping'],1)
                self.assertEqual(data.meta_info['ccpo_diag']['ccpo/progress_episode_weight'],ep)
                if benchmark=='alfworld':
                    for task in np.unique(a['uid']):
                        values=a['actor_applied'][a['uid']==task]
                        self.assertAlmostEqual(values.mean(),0.,places=5)
                        self.assertAlmostEqual(values.std(ddof=1),1.,places=5)
                else:
                    np.testing.assert_allclose(a['actor_applied']-a['episode_applied'],a['combined_pre'],atol=3e-5)


if __name__=='__main__':
    unittest.main()
