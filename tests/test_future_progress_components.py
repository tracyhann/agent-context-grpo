"""Component ablations must isolate actual PPO gradients and normalization masks."""
import io
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from test_future_progress import (arguments, core_args, core, np, torch,
    ccpo_future_progress_advantage, load_compute_advantage, policy_gradient)
from test_future_progress_active_episode import training_fixture, reward_data
import ccpo.future_progress as fp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import prepare_component_ablations as prep


class ComponentCreditTests(unittest.TestCase):
    def test_default_is_unchanged_and_single_channels_match_raw_estimators(self):
        kw=arguments()
        with patch.object(core,'_LK_FIX','1.0'):
            default,d=ccpo_future_progress_advantage(**kw,horizon=2)
            explicit,_=ccpo_future_progress_advantage(**kw,horizon=2,history_weight=1,progress_weight=1)
            history,h=ccpo_future_progress_advantage(**kw,horizon=2,progress_weight=0)
            future,f=ccpo_future_progress_advantage(**kw,horizon=2,history_weight=0)
            original,_=core.ccpo_step_advantage(**core_args(kw))
        torch.testing.assert_close(default,explicit,rtol=0,atol=0)
        torch.testing.assert_close(history,original,rtol=0,atol=0)
        np.testing.assert_allclose(future.numpy(),f['progress_payload']['arrays']['progress_normalized'],atol=1e-6)
        torch.testing.assert_close(history+future,default,atol=1e-6,rtol=1e-6)
        np.testing.assert_array_equal(h['progress_components_future'],0.)
        np.testing.assert_array_equal(f['progress_components_history'],0.)

    def test_disabled_support_cannot_enter_live_normalization_population(self):
        kw=arguments(); n=len(kw['index']); hist_live=np.arange(n)%2==0; eligible=np.arange(n)%3==0
        original_core=core.ccpo_step_advantage; original_progress=fp.progress_from_values
        def history_mask(**args):
            adv,diag=original_core(**args)
            if 'prepared_phi' not in args: diag['live_mask']=hist_live.copy()
            return adv,diag
        def future_mask(*args,**kwargs):
            raw,future,endpoint,window,_=original_progress(*args,**kwargs)
            return raw,future,endpoint,window,eligible.copy()
        for hw,fw in [(1,0),(0,1),(1,1),(0,0)]:
            with patch.object(core,'ccpo_step_advantage',side_effect=history_mask),patch.object(fp,'progress_from_values',side_effect=future_mask):
                _,d=ccpo_future_progress_advantage(**kw,horizon=2,history_weight=hw,progress_weight=fw)
            np.testing.assert_array_equal(d['live_mask'],((hw>0)&hist_live)|((fw>0)&eligible))

    def test_penalty_changes_history_but_not_future_only_actor_credit(self):
        kw=arguments()
        a,da=ccpo_future_progress_advantage(**kw,horizon=2,history_weight=0)
        kw['step_rewards'][2]-=5.;kw['is_action_valid'][2]=False
        b,db=ccpo_future_progress_advantage(**kw,horizon=2,history_weight=0)
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        self.assertGreater(np.max(abs(da['progress_payload']['arrays']['history_adv']-db['progress_payload']['arrays']['history_adv'])),1.)

    def test_all_trainer_paths_log_zero_disabled_channel_and_isolate_ppo_gradient(self):
        compute=load_compute_advantage();kw=training_fixture()
        original_core=core.ccpo_step_advantage;original_progress=fp.progress_from_values
        def altered_history(**args):
            adv,diag=original_core(**args)
            if 'prepared_phi' not in args:
                adv=adv*37+torch.arange(len(adv),dtype=adv.dtype)*11-50
                diag['live_mask']=~np.asarray(diag['live_mask'])
            return adv,diag
        def altered_future(*args,**kwargs):
            raw,future,endpoint,window,eligible=original_progress(*args,**kwargs)
            raw=np.arange(len(raw),dtype=float)**2
            eligible=np.arange(len(raw))%2==0
            return raw,future,endpoint,window,eligible
        for benchmark,mode,ep in [('alfworld','mean_std_norm',0),('webshop','mean_norm',1),('webshop','mean_norm',0)]:
            for component,hw,fw in [('history',1,0),('future',0,1)]:
                with self.subTest(benchmark=benchmark,component=component),tempfile.TemporaryDirectory() as temp:
                    outputs=[];snapshots=[]
                    for perturb in [False,True]:
                        folder=Path(temp)/str(perturb)
                        env=dict(ACG_CCPO_PROGRESS_HORIZON='2',ACG_CCPO_PROGRESS_HISTORY_WEIGHT=str(hw),
                            ACG_CCPO_PROGRESS_WEIGHT=str(fw),ACG_CCPO_EP_W=str(ep),ACG_CCPO_STEP_NORM='mode',
                            ACG_CCPO_FIXED_ANCHOR='0',ACG_CCPO_OUTLOOK_HORIZON='0',ACG_CCPO_OUTLOOK_BETA='0',
                            ACG_EXP_DIR=str(folder),ACG_CCPO_PROGRESS_SNAPSHOT_EVERY='1')
                        data=reward_data(kw)
                        with patch.dict(os.environ,env),patch.object(core,'_LK_FIX','1.0'),redirect_stdout(io.StringIO()),\
                            patch.object(core,'ccpo_step_advantage',side_effect=altered_history if perturb and not hw else original_core),\
                            patch.object(fp,'progress_from_values',side_effect=altered_future if perturb and not fw else original_progress):
                            compute(data,'CCPO',gamma=.95,gigpo_mode=mode,ccpo_step_tag=1)
                        outputs.append((data,*policy_gradient(data)))
                        with np.load(folder/'outputs/future_progress/step-0001.npz',allow_pickle=False) as a:
                            snapshots.append({k:a[k].copy() for k in a.files})
                        a=snapshots[-1]; disabled='future' if hw else 'history'
                        np.testing.assert_array_equal(a[disabled+'_applied'],0.)
                        np.testing.assert_array_equal(a['weighted_progress' if hw else 'weighted_history'],0.)
                        np.testing.assert_allclose(a['history_applied']+a['future_applied']+a['episode_applied'],a['actor_applied'],atol=3e-5)
                        np.testing.assert_allclose(a['actor_applied'][:,None]*kw['response_mask'].numpy(),data.batch['advantages'].numpy(),atol=3e-5)
                        np.testing.assert_array_equal(a['episode_applied'],ep*a['episode_adv'])
                        self.assertFalse(data.batch['advantages'].requires_grad)
                        self.assertTrue(torch.isfinite(outputs[-1][2]).all())
                        self.assertGreater(float(outputs[-1][2].abs().sum()),0)
                        diag=data.meta_info['ccpo_diag']
                        self.assertEqual(diag['ccpo/progress_history_weight'],hw)
                        self.assertEqual(diag['ccpo/progress_weight'],fw)
                        self.assertEqual(diag['ccpo/progress_episode_weight'],ep)
                        self.assertLess(diag['ccpo/progress_actor_identity_error'],3e-5)
                    torch.testing.assert_close(outputs[0][0].batch['advantages'],outputs[1][0].batch['advantages'],rtol=0,atol=0)
                    for i in [1,2]:torch.testing.assert_close(outputs[0][i],outputs[1][i],rtol=0,atol=0)
                    raw_key='raw_progress' if hw else 'history_adv'
                    self.assertFalse(np.allclose(snapshots[0][raw_key],snapshots[1][raw_key]))

    def test_invalid_weights_fail_in_estimator_and_launcher(self):
        for weight in [-1,float('nan'),float('inf')]:
            for key in ['history_weight','progress_weight']:
                with self.assertRaises(ValueError):ccpo_future_progress_advantage(**arguments(),**{key:weight})
            for key in ['ccpo_progress_history_weight','ccpo_progress_weight']:
                with self.assertRaises(ValueError):prep.er.validate_future_progress_config(dict(ccpo_progress_horizon=2,**{key:weight}))
        with self.assertRaises(ValueError):prep.er.validate_future_progress_config(dict(ccpo_progress_horizon=0,ccpo_progress_history_weight=0))


class PreparedComponentTests(unittest.TestCase):
    def test_four_configs_change_only_requested_coefficient_from_matched_control(self):
        for benchmark in prep.CONTROLS:
            for component in ['history','future']:
                with self.subTest(benchmark=benchmark,component=component):
                    folder=ROOT/'experiments'/prep.run_name(benchmark,component,'20260920')
                    record=json.loads((folder/'config.json').read_text());cfg=record['config']
                    old=json.loads((ROOT/'experiments'/prep.CONTROLS[benchmark]/'config.json').read_text())['config']
                    old.setdefault('ccpo_progress_history_weight',1.)
                    changed='ccpo_progress_weight' if component=='history' else 'ccpo_progress_history_weight'
                    self.assertEqual(set(old),set(cfg))
                    self.assertEqual({k for k in cfg if cfg[k]!=old[k]},{'exp_id',changed})
                    _,method=prep.ablations.build(prep.key_for(benchmark,component),benchmark)
                    for k,v in method.items():self.assertEqual(cfg[k],v,k)
                    self.assertEqual(cfg['ccpo_progress_history_weight'],int(component=='history'))
                    self.assertEqual(cfg['ccpo_progress_weight'],int(component=='future'))
                    self.assertEqual(cfg['ccpo_ep_w'],int(benchmark=='webshop'))
                    self.assertEqual(cfg['ccpo_lk_fix'],1.)
                    self.assertEqual(cfg['ccpo_progress_horizon'],2);self.assertEqual(cfg['history_length'],2)
                    self.assertEqual(cfg['ccpo_ctx_w'],1.);self.assertEqual(cfg['total_epochs'],150)
                    self.assertEqual(cfg['model'],'Qwen/Qwen2.5-1.5B-Instruct')
                    self.assertEqual(record['hydra_overrides'],prep.er.build_command(cfg,str(folder))[3:])
                    for k,e in prep.er.ENV_KEYS.items():
                        if k == 'ccpo_loo' and k not in cfg:
                            self.assertEqual(prep.er.DEFAULTS[k],1); self.assertNotIn(e,record['env'])
                        else:
                            self.assertEqual(record['env'][e],str(cfg[k]))
                    status=json.loads((folder/'PREPARED.json').read_text())
                    self.assertFalse(status['launched']);self.assertFalse(status['queued'])
                    self.assertIn('--dry-run',json.loads((folder/'prepare-command.json').read_text())['argv'])
                    self.assertFalse((folder/'outputs/train.pid').exists());self.assertFalse((folder/'outputs/metrics.jsonl').exists())

    def test_preparation_cannot_overwrite_existing_experiment(self):
        with patch.object(prep.subprocess,'run') as run:
            with self.assertRaises(FileExistsError):prep.prepare('alfworld','history','20260920')
        run.assert_not_called()


if __name__=='__main__':unittest.main()
