"""ALFWorld window arms: endpoint math, zero-side gradients, prompts and configs."""
import ast
import copy
import io
import json
import os
from pathlib import Path
import runpy
import sys
import tempfile
from types import SimpleNamespace
from typing import Any,Dict,List,Tuple
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from test_future_progress import (core,np,torch,rows,core_args,ccpo_future_progress_advantage,
    load_compute_advantage,make_data,policy_gradient,core_gigpo)
from ccpo.future_progress import endpoint_map,progress_from_values

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import prepare_alfworld_window_ablations as prep


def fixture():
    lengths=[7,6,5,4]
    turns=np.concatenate([np.arange(n) for n in lengths]);lens=np.repeat(lengths,lengths)
    traj=np.repeat(['a','b','c','d'],lengths);ep=np.repeat([10.,0.,10.,0.],lengths)
    immediate=np.where(turns==lens-1,ep,0.)
    valid=np.ones(len(turns),dtype=bool);valid[1]=False
    y=.95**(lens-1-turns)*ep-.1*(~valid)
    return dict(step_rewards=torch.tensor(y,dtype=torch.float32),response_mask=torch.ones(len(turns),4),
        anchor_obs=np.array(['room-'+str(t%3) for t in turns]),index=np.array(['task']*len(turns)),
        traj_index=traj,turn_index=turns,episode_lengths=lens,episode_rewards=ep,
        is_action_valid=valid,immediate_rewards=immediate,phi=core.FrozenPhi(),
        phi_feats=torch.tensor(np.random.default_rng(5).normal(size=(len(turns),12)),dtype=torch.float32),
        gamma=.95,return_diag=True)


def arguments():
    kw=fixture();kw.pop('immediate_rewards');return kw


class WindowRuntimeTests(unittest.TestCase):
    def setUp(self):
        p=patch.multiple(core,_LK_FIX='1.0',_LAM_FIX='1.0',_LOO='1',_GATE='hard',_CTX_W=1.,
            _PHI_MODE='hidden+ctx',_EDGE_W=0.,_SIM=0.,_SIM_BACKOFF=0.,_JW_C=0.,_STD_MODE='task',
            _BACKOFF_TASK=True,_WMODE='soft',_TAU_ENV='.15')
        p.start();self.addCleanup(p.stop)

    def test_endpoints_and_three_four_step_telescoping_clip_at_terminal(self):
        groups=[np.arange(7),np.arange(7,12)]
        v=np.array([1.,2.,3.,4.,5.,6.,7.,8.,6.,4.,2.,1.]);ep=np.r_[np.full(7,10.),np.zeros(5)]
        for horizon in range(5):
            e,w=endpoint_map(groups,len(v),horizon)
            expected=np.full(len(v),-1)
            for ids in groups:
                for t,i in enumerate(ids):
                    self.assertEqual(w[i],min(horizon,len(ids)-t))
                    if t+horizon<len(ids):expected[i]=ids[t+horizon]
            np.testing.assert_array_equal(e,expected)
            raw,future,_,_,eligible=progress_from_values(v,groups,ep,horizon)
            if horizon==0:
                np.testing.assert_array_equal(raw,0.);self.assertFalse(eligible.any())
                np.testing.assert_array_equal(future,v)
            else:
                np.testing.assert_allclose(raw,future-v)
                one,_,next_idx,_,_=progress_from_values(v,groups,ep,1)
                total=np.zeros(len(v));position=np.arange(len(v))
                for _ in range(horizon):
                    active=position>=0;total[active]+=one[position[active]];position[active]=next_idx[position[active]]
                np.testing.assert_allclose(raw,total)
                np.testing.assert_array_equal(future[e<0],ep[e<0])

    def test_zero_future_is_history_only_with_zero_future_support(self):
        kw=arguments();zero,d=ccpo_future_progress_advantage(**kw,horizon=0,progress_weight=0.)
        old,_=ccpo_future_progress_advantage(**kw,horizon=2,progress_weight=0.)
        pure,_=core.ccpo_step_advantage(**core_args(kw))
        torch.testing.assert_close(zero,old,rtol=0,atol=0);torch.testing.assert_close(zero,pure,rtol=0,atol=0)
        a=d['progress_payload']['arrays']
        for key in ['raw_progress','progress_normalized','weighted_progress','window_length']:np.testing.assert_array_equal(a[key],0.)
        self.assertFalse(a['eligible'].any());np.testing.assert_array_equal(a['live'],a['history_live'])
        self.assertEqual(d['progress_future_active'],0.)
        self.assertEqual(d['progress_horizon'],0.)

    def test_new_horizons_keep_padding_and_finiteness_guards(self):
        kw=arguments();n=len(kw['index']);ids=np.random.default_rng(44).permutation(np.r_[np.arange(n),0,6,10])
        for f in [0,3,4]:
            opts=dict(horizon=f,progress_weight=float(f>0))
            base,d=ccpo_future_progress_advantage(**kw,**opts)
            padded,e=ccpo_future_progress_advantage(**rows(kw,ids),**opts)
            torch.testing.assert_close(padded,base[ids],rtol=0,atol=0)
            for key in ['current_value','future_value','window_length','endpoint_index']:
                np.testing.assert_array_equal(d['progress_payload']['arrays'][key],e['progress_payload']['arrays'][key])
            bad=rows(kw,np.arange(n));bad['phi_feats'][0,0]=float('nan')
            with self.assertRaisesRegex(ValueError,'Non-finite'):ccpo_future_progress_advantage(**bad,**opts)

    def test_invalid_horizons_and_zero_future_weight_contract(self):
        for h in [-1,5,1.5,3.0,True,'3']:
            with self.assertRaises(ValueError):endpoint_map([np.arange(2)],2,h)
            with self.assertRaises(ValueError):ccpo_future_progress_advantage(**arguments(),horizon=h)
            with self.assertRaises(ValueError):prep.er.validate_future_progress_config(dict(ccpo_progress_horizon=h))
        with self.assertRaisesRegex(ValueError,'progress_weight=0'):ccpo_future_progress_advantage(**arguments(),horizon=0)
        prep.er.validate_future_progress_config(dict(ccpo_progress_horizon=0,ccpo_progress_weight=0))

    def test_all_five_real_trainer_paths_and_episode_gradient_isolation(self):
        compute=load_compute_advantage();kw=fixture()
        for window,(h,f) in prep.WINDOWS.items():
            record=json.loads((ROOT/'experiments'/prep.run_name(window,'20260922')/'config.json').read_text())
            results=[]
            for perturb in [False,True]:
                with tempfile.TemporaryDirectory() as tmp:
                    env=dict(record['env'],CUDA_VISIBLE_DEVICES='',ACG_EXP_DIR=tmp,ACG_CCPO_DUMP='')
                    episode=torch.linspace(-2,2,len(kw['index']))[:,None]*kw['response_mask']
                    if perturb:episode=(episode*31+100)*kw['response_mask']
                    data=make_data(kw)
                    with patch.dict(os.environ,env),patch.object(core_gigpo,'episode_norm_reward',return_value=episode),redirect_stdout(io.StringIO()):
                        compute(data,'CCPO',gamma=.95,gigpo_mode='mean_std_norm',ccpo_step_tag=1)
                    results.append((data,*policy_gradient(data)))
                    with np.load(Path(tmp)/'outputs/future_progress/step-0001.npz',allow_pickle=False) as a:
                        np.testing.assert_array_equal(a['episode_applied'],0.)
                        np.testing.assert_allclose(a['actor_applied'],a['history_applied']+a['future_applied'],atol=3e-5)
                        np.testing.assert_allclose(a['actor_applied'][:,None]*kw['response_mask'].numpy(),data.batch['advantages'].numpy(),atol=3e-5)
                        if h==0:
                            np.testing.assert_array_equal(a['history_applied'],0.);np.testing.assert_array_equal(a['live'],a['eligible'])
                        if f==0:
                            np.testing.assert_array_equal(a['future_applied'],0.);np.testing.assert_array_equal(a['live'],a['history_live'])
                        for prefix in ['history','current','future']:
                            finite=np.isfinite(a[prefix+'_lambda_k']);np.testing.assert_array_equal(a[prefix+'_lambda_k'][finite],1.)
                    self.assertEqual(data.meta_info['ccpo_diag']['ccpo/progress_horizon'],f)
                    self.assertEqual(data.meta_info['ccpo_diag']['ccpo/progress_episode_weight'],0.)
                    self.assertTrue(torch.isfinite(results[-1][2]).all());self.assertGreater(float(results[-1][2].abs().sum()),0.)
                    self.assertFalse(data.batch['advantages'].requires_grad)
            for i in [1,2]:torch.testing.assert_close(results[0][i],results[1][i],rtol=0,atol=0)

    def test_history_only_still_uses_verified_reference_capture_before_padding(self):
        tree=ast.parse((ROOT/'verl-agent/verl/trainer/ppo/ray_trainer.py').read_text())
        expr=next(n.value for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_verified_phi' for t in n.targets))
        ns=dict(os=os,self=SimpleNamespace(config=SimpleNamespace(algorithm=SimpleNamespace(adv_estimator='ccpo'))),AdvantageEstimator=SimpleNamespace(CCPO='ccpo'))
        for horizon,weight,expected in [(0,1,False),(0,0,True),(1,1,True),(2,1,True),(3,1,True),(4,1,True)]:
            with patch.dict(os.environ,ACG_CCPO_PROGRESS_HORIZON=str(horizon),ACG_CCPO_PROGRESS_WEIGHT=str(weight)):
                self.assertEqual(eval(compile(ast.Expression(expr),'<verified_phi>','eval'),ns),expected)


class PromptAndConfigTests(unittest.TestCase):
    def test_actual_alfworld_prompt_windows_and_zero_history_goal(self):
        path=ROOT/'verl-agent/agent_system/memory/memory.py'
        cls=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='SimpleMemory')
        ns=dict(BaseMemory=object,List=List,Dict=Dict,Any=Any,Tuple=Tuple,os=os)
        exec(compile(ast.Module(body=[cls],type_ignores=[]),str(path),'exec'),ns)
        memory=ns['SimpleMemory']();memory.reset(1)
        for i in range(6):memory.store({'text_obs':[f'PAST_OBS_{i}'],'action':[f'PAST_ACT_{i}']})
        before=copy.deepcopy(memory._data)
        manager=ROOT/'verl-agent/agent_system/environments/env_manager.py'
        cls=next(n for n in ast.parse(manager.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='AlfWorldEnvironmentManager')
        fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='build_text_obs')
        ns.update(runpy.run_path(str(ROOT/'verl-agent/agent_system/environments/prompts/alfworld.py')))
        exec(compile(ast.Module(body=[fn],type_ignores=[]),str(manager),'exec'),ns)
        for history in [0,1,2,3,4]:
            env=SimpleNamespace(memory=memory,config=SimpleNamespace(env=SimpleNamespace(history_length=history)),tasks=['TASK_GOAL'])
            with patch.dict(os.environ,ACG_COMPACT_BUDGET='0'):
                text=ns['build_text_obs'](env,['CURRENT_OBS'],[['look','help']])[0]
                initial=ns['build_text_obs'](env,['INITIAL_WITH_GOAL'],[['look','help']],init=True)[0]
            self.assertIn('TASK_GOAL',text);self.assertIn('CURRENT_OBS',text)
            for i in range(6):
                for key in ['PAST_OBS_','PAST_ACT_']:self.assertEqual(key+str(i) in text,i>=6-history)
            self.assertEqual(memory._data,before)
            self.assertEqual(initial,ns['ALFWORLD_TEMPLATE_NO_HIS'].format(current_observation='INITIAL_WITH_GOAL',admissible_actions="'look'"))
        for suffix in ['env_manager.py','prompts/alfworld.py']:
            self.assertEqual((ROOT/'patches/verl-agent/agent_system/environments'/suffix).read_bytes(),(ROOT/'verl-agent/agent_system/environments'/suffix).read_bytes())

    def test_prepared_configs_match_main_and_only_change_window_controls(self):
        control=json.loads((ROOT/'experiments'/prep.CONTROL/'config.json').read_text())['config']
        for k,v in prep.IMPLIED.items():control.setdefault(k,v)
        for window,(h,f) in prep.WINDOWS.items():
            folder=ROOT/'experiments'/prep.run_name(window,'20260922');record=json.loads((folder/'config.json').read_text());cfg=record['config']
            expected={'exp_id','history_length','ccpo_progress_horizon'}
            if not h:expected.add('ccpo_progress_history_weight')
            if not f:expected.add('ccpo_progress_weight')
            self.assertEqual(set(control),set(cfg));self.assertEqual({k for k in cfg if cfg[k]!=control[k]},expected)
            self.assertEqual((cfg['history_length'],cfg['ccpo_progress_horizon']),(h,f))
            self.assertEqual((cfg['ccpo_progress_history_weight'],cfg['ccpo_progress_weight']),(float(h>0),float(f>0)))
            self.assertEqual(record['hydra_overrides'],prep.er.build_command(cfg,str(folder))[3:])
            for k,e in prep.er.ENV_KEYS.items():self.assertEqual(record['env'][e],str(cfg[k]))
            self.assertEqual(cfg['ccpo_ep_w'],0);self.assertEqual(cfg['ccpo_lk_fix'],1);self.assertEqual(cfg['ccpo_loo'],1)
            status=json.loads((folder/'PREPARED.json').read_text());self.assertFalse(status['launched']);self.assertFalse(status['queued'])
            self.assertFalse(any(f.is_file() for f in (folder/'outputs').rglob('*')))

    def test_preparer_refuses_to_overwrite(self):
        with patch.object(prep.subprocess,'run') as run:
            with self.assertRaises(FileExistsError):prep.prepare('h4f0','20260922')
        run.assert_not_called()


if __name__=='__main__':unittest.main()
