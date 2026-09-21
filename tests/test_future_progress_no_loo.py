"""No-LOO peer membership, target alignment, padding safety and actual PPO gradients."""
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from test_future_progress import (arguments, core_args, rows, core, np, torch,
    ccpo_future_progress_advantage, load_compute_advantage, policy_gradient, core_gigpo)
from test_future_progress_active_episode import training_fixture, reward_data

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import prepare_no_loo_ablations as prep


class NoLooTests(unittest.TestCase):
    def setUp(self):
        self.settings = patch.multiple(core, _LK_FIX='1.0', _LAM_FIX='1.0', _GATE='hard',
            _BACKOFF_TASK=True, _SIM=0., _SIM_BACKOFF=0., _JW_C=0., _STD_MODE='task',
            _WMODE='soft', _TAU_ENV='.15', _EDGE_W=0., _PRIOR_KAPPA=2., _LOO='1')
        self.settings.start(); self.addCleanup(self.settings.stop)

    def small(self):
        return dict(step_rewards=torch.tensor([2.,6.,10.,100.,50.],requires_grad=True),
            response_mask=torch.ones(5,2), anchor_obs=np.array(['match','match','match','other','match']),
            index=np.array(['a','a','a','a','b']),traj_index=np.array(['own','own','peer','own','foreign']),
            phi=core.FrozenPhi(), prepared_phi=np.array([[0.],[1.],[2.],[0.],[0.]]),
            return_diag=True, value_targets=np.array([3.,7.,11.,101.,51.]))

    def test_hand_computed_self_revisit_weights_and_distinct_trajectory_support(self):
        with patch.object(core,'_TAU_ENV','1'):
            got,d=core.ccpo_step_advantage(**self.small(),loo=0)
        w=np.exp(-np.array([0.,1.,2.]));w/=w.sum()
        self.assertAlmostEqual(d['baseline_values'][0],float(w@np.array([2.,6.,10.])))
        self.assertAlmostEqual(d['bootstrap_values'][0],float(w@np.array([3.,7.,11.])))
        self.assertAlmostEqual(d['self_mass_values'][0],w[0])
        self.assertAlmostEqual(d['same_traj_mass_values'][0],w[:2].sum())
        self.assertEqual(d['support_values'][0],2)
        self.assertEqual(d['peer_rows_values'][0],3)
        self.assertAlmostEqual(d['effective_support_values'][0],1/(w[:2].sum()**2+w[2]**2))
        self.assertFalse(got.requires_grad)
        self.assertEqual(d['loo'],0)
        self.assertEqual(d['baseline_values'][4],50)  # Same observation, different task.
        np.testing.assert_array_equal(d['credibility_values'],1.)

    def test_default_parity_and_own_return_dependence_only_without_loo(self):
        kw=arguments()
        default,d=ccpo_future_progress_advantage(**kw,horizon=2)
        explicit,e=ccpo_future_progress_advantage(**kw,horizon=2,loo=1)
        torch.testing.assert_close(default,explicit,rtol=0,atol=0)
        own=kw['traj_index']=='traj0'
        for loo in [0,1]:
            aargs=arguments();_,before=ccpo_future_progress_advantage(**aargs,horizon=2,loo=loo)
            aargs['episode_rewards'][own]=25.;aargs['step_rewards'][own]+=3.
            _,after=ccpo_future_progress_advantage(**aargs,horizon=2,loo=loo)
            a,b=before['progress_payload']['arrays'],after['progress_payload']['arrays']
            for key in ['history_baseline','current_value','future_value']:
                keep=own & ~a['terminal']
                if loo:np.testing.assert_allclose(a[key][keep],b[key][keep],atol=0,rtol=0)
                else:self.assertGreater(np.max(abs(a[key][keep]-b[key][keep])),.01)
            for source in ['history','current','future']:
                mass=a[source+'_same_traj_mass'];finite=np.isfinite(mass)
                if loo:np.testing.assert_array_equal(mass[finite],0)
                else:self.assertTrue(np.all(mass[finite]>0))
        self.assertEqual(d['progress_loo'],1)

    def test_singleton_is_exact_supported_and_does_not_force_task_backoff(self):
        kw=arguments();kw['anchor_obs']=np.array(['unique-'+str(i) for i in range(len(kw['index']))])
        _,d=ccpo_future_progress_advantage(**kw,horizon=2,loo=0)
        a=d['progress_payload']['arrays']
        np.testing.assert_array_equal(a['history_adv'],0.)
        np.testing.assert_allclose(a['current_value'],a['value_target'],rtol=1e-7,atol=0)
        np.testing.assert_array_equal(a['history_J'],1.)
        np.testing.assert_array_equal(a['current_level'],0)
        np.testing.assert_array_equal(a['current_self_mass'],1.)
        self.assertTrue(a['eligible'].all())
        self.assertGreater(np.max(abs(a['raw_progress'])),0.)
        self.assertTrue(np.isnan(a['future_self_mass'][a['terminal']]).all())
        _,old=ccpo_future_progress_advantage(**kw,horizon=2,loo=1)
        self.assertTrue(np.all(old['progress_payload']['arrays']['current_level']>0))

    def test_task_and_auxiliary_priors_follow_same_exclusion_policy(self):
        with patch.object(core,'_LK_FIX','0'):
            for loo,target,value in [(0,29.5,30.5),(1,10.,11.)]:
                _,d=core.ccpo_step_advantage(**self.small(),loo=loo)
                self.assertEqual(d['task_prior_values'][0],target)
                self.assertEqual(d['baseline_values'][0],target)
                self.assertEqual(d['bootstrap_values'][0],value)

    def test_padding_duplicates_do_not_become_self_peers(self):
        kw=arguments();expected,d=ccpo_future_progress_advantage(**kw,horizon=2,loo=0)
        ids=np.random.default_rng(72).permutation(np.r_[np.arange(len(expected)),0,0,4,9])
        got,e=ccpo_future_progress_advantage(**rows(kw,ids),horizon=2,loo=0)
        torch.testing.assert_close(got,expected[ids],rtol=0,atol=0)
        for key in ['history_self_mass','current_same_traj_mass','future_J','current_peer_rows','current_value']:
            np.testing.assert_array_equal(d['progress_payload']['arrays'][key],e['progress_payload']['arrays'][key])
        padded=rows(kw,np.r_[np.arange(len(expected)),0]);padded['phi_feats'][-1,0]+=1.
        with self.assertRaisesRegex(ValueError,'Inconsistent padded'):
            ccpo_future_progress_advantage(**padded,horizon=2,loo=0)

    def test_invalid_flag_fails_before_core_and_launch(self):
        for value in [-1,2,'false','none',float('nan')]:
            with self.assertRaisesRegex(ValueError,'LOO'):core.ccpo_step_advantage(**self.small(),loo=value)
            with self.assertRaisesRegex(ValueError,'ccpo_loo'):prep.er.validate_future_progress_config(dict(ccpo_loo=value))

    def test_actual_trainer_gradients_change_with_no_loo_but_episode_stays_off(self):
        kw=training_fixture();compute=load_compute_advantage()
        for mode in ['mean_std_norm','mean_norm']:
            results={}
            for loo,perturb in [(1,False),(0,False),(0,True)]:
                with tempfile.TemporaryDirectory() as tmp:
                    env=dict(ACG_CCPO_PROGRESS_HORIZON='2',ACG_CCPO_PROGRESS_HISTORY_WEIGHT='1',
                        ACG_CCPO_PROGRESS_WEIGHT='1',ACG_CCPO_EP_W='0',ACG_CCPO_STEP_NORM='mode',
                        ACG_CCPO_FIXED_ANCHOR='0',ACG_CCPO_OUTLOOK_HORIZON='0',ACG_CCPO_OUTLOOK_BETA='0',
                        ACG_EXP_DIR=tmp,ACG_CCPO_PROGRESS_SNAPSHOT_EVERY='1')
                    data=reward_data(kw)
                    raw=torch.arange(len(kw['index']),dtype=torch.float32)[:,None].expand_as(kw['response_mask'])
                    ep=(raw*31+100 if perturb else raw)*kw['response_mask']
                    with patch.dict(os.environ,env),patch.object(core,'_LOO',str(loo)),\
                         patch.object(core_gigpo,'episode_norm_reward',return_value=ep),redirect_stdout(io.StringIO()):
                        compute(data,'CCPO',gamma=.95,gigpo_mode=mode,ccpo_step_tag=1)
                    results[loo,perturb]=(data,*policy_gradient(data))
                    with np.load(Path(tmp)/'outputs/future_progress/step-0001.npz',allow_pickle=False) as a:
                        np.testing.assert_array_equal(a['episode_applied'],0.)
                        np.testing.assert_allclose(a['history_applied']+a['future_applied'],a['actor_applied'],atol=3e-5)
                        self.assertEqual(data.meta_info['ccpo_diag']['ccpo/progress_loo'],loo)
                        self.assertIn('history_self_mass',a.files)
                    self.assertFalse(data.batch['advantages'].requires_grad)
                    self.assertTrue(torch.isfinite(results[loo,perturb][2]).all())
                    self.assertGreater(float(results[loo,perturb][2].abs().sum()),0)
            self.assertFalse(torch.allclose(results[0,False][2],results[1,False][2]))
            torch.testing.assert_close(results[0,False][0].batch['advantages'],results[0,True][0].batch['advantages'],atol=0,rtol=0)
            for i in [1,2]:torch.testing.assert_close(results[0,False][i],results[0,True][i],atol=0,rtol=0)


class PreparedNoLooTests(unittest.TestCase):
    def test_two_configs_change_only_exclusion_from_ep0_main_controls(self):
        for benchmark in prep.CONTROLS:
            folder=ROOT/'experiments'/prep.run_name(benchmark,'20260921')
            record=json.loads((folder/'config.json').read_text());cfg=record['config']
            old=json.loads((ROOT/'experiments'/prep.CONTROLS[benchmark]/'config.json').read_text())['config']
            for k,v in prep.IMPLIED.items():old.setdefault(k,v)
            self.assertEqual(set(old),set(cfg))
            self.assertEqual({k for k in cfg if cfg[k]!=old[k]},{'exp_id','ccpo_loo'})
            for k,v in dict(ccpo_loo=0,ccpo_ep_w=0.,ccpo_edge_w=0.,ccpo_lk_fix=1.,ccpo_lam_fix=1.,
                ccpo_progress_horizon=2,history_length=2,ccpo_progress_history_weight=1.,ccpo_progress_weight=1.,
                ccpo_phi='hidden+ctx',ccpo_ctx_w=1.,total_epochs=150,model='Qwen/Qwen2.5-1.5B-Instruct').items():
                self.assertEqual(cfg[k],v,k)
            self.assertEqual(record['hydra_overrides'],prep.er.build_command(cfg,str(folder))[3:])
            for k,e in prep.er.ENV_KEYS.items():self.assertEqual(record['env'][e],str(cfg[k]))
            self.assertIn('  ACG_CCPO_LOO=0 ',(folder/'run.sh').read_text())
            status=json.loads((folder/'PREPARED.json').read_text())
            self.assertFalse(status['launched']);self.assertFalse(status['queued'])
            self.assertFalse((folder/'outputs/train.pid').exists())
            self.assertIn('--dry-run',json.loads((folder/'prepare-command.json').read_text())['argv'])

    def test_prepared_environment_selects_self_inclusion_in_fresh_process(self):
        code = """
import numpy as np
import torch
from ccpo import core_ccpo as core
from ccpo.future_progress import ccpo_future_progress_advantage
_,d=ccpo_future_progress_advantage(
    step_rewards=torch.tensor([9.025,9.5,10.]),response_mask=torch.ones(3,2),
    anchor_obs=np.array(['a','b','c']),index=np.array(['task']*3),
    traj_index=np.array(['own']*3),phi=core.FrozenPhi(),
    phi_feats=torch.eye(3),episode_rewards=np.full(3,10.),is_action_valid=np.ones(3),
    turn_index=np.arange(3),episode_lengths=np.full(3,3),horizon=2,return_diag=True)
assert d['progress_loo']==0
assert (d['progress_payload']['arrays']['current_self_mass']==1).all()
assert (d['progress_payload']['arrays']['history_adv']==0).all()
print('Fresh process used self-inclusive H/current/future readouts')
"""
        for benchmark in prep.CONTROLS:
            folder=ROOT/'experiments'/prep.run_name(benchmark,'20260921')
            record=json.loads((folder/'config.json').read_text())
            env=dict(os.environ,**record['env'])
            env.update(CUDA_VISIBLE_DEVICES='',ACG_EXP_DIR='',ACG_CCPO_DUMP='')
            result=prep.subprocess.run([str(ROOT/'.venv/bin/python'),'-c',code],cwd=ROOT,
                env=env,capture_output=True,text=True,timeout=60)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)

    def test_prepare_refuses_to_overwrite(self):
        with patch.object(prep.subprocess,'run') as run:
            with self.assertRaises(FileExistsError):prep.prepare('alfworld','20260921')
        run.assert_not_called()


if __name__ == '__main__':unittest.main()
