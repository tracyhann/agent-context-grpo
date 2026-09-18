"""CPU verification of the production fixed-anchor arm and complete logging."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
 ACG_CCPO_PHI='hidden+ctx',ACG_CCPO_LAM_FIX='1',ACG_CCPO_PRIOR_KAPPA='2',
 ACG_CCPO_EDGE_W='0',ACG_CCPO_TARGET='return',ACG_CCPO_STD='task',
 ACG_CCPO_JWEIGHT_C='0',ACG_CCPO_BACKOFF_TASK='1',ACG_CCPO_LK_FIX='',
 ACG_CCPO_GATE='hard',ACG_CCPO_TAU='.15',ACG_CCPO_WHITEN='3')
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'verl-agent'))
import importlib.util
spec=importlib.util.spec_from_file_location('outlook_fixture',ROOT/'tests/test_outlook.py')
fixture_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture_module)
import unittest
from unittest.mock import patch
import tempfile,json,csv,gzip
from types import SimpleNamespace
import numpy as np
import torch
from ccpo import core_ccpo as core
from ccpo.fixed_anchor import ccpo_fixed_anchor_advantage, future_records, tokenize_future, finalize_fixed_logging, capture_future_features


def fixture():
    kw=fixture_module.fixture();n=len(kw['index'])
    non=dict(uid=kw['index'],traj_uid=kw['traj_index'],ccpo_turn_index=kw['turn_index'],
        episode_lengths=kw['episode_lengths'],anchor_obs=kw['anchor_obs'],
        ccpo_prompt_text=np.array([f'Historical context for row {i}. Current observation: {kw["anchor_obs"][i]}' for i in range(n)]),
        ccpo_action=np.array([f'action_{i}' for i in range(n)]))
    nextobs=[]
    for i in range(n):
        nextobs.append(kw['anchor_obs'][i+1] if kw['turn_index'][i]+1<kw['episode_lengths'][i] else 'TERMINAL_OBSERVATION')
    non['ccpo_next_obs']=np.array(nextobs)
    take,restore,_,records=future_records(non)
    for r in records:r.update(input_ids=[1,2,3],original_tokens=3,truncated=False)
    contexts=np.array([[r['future_context'][k] for k in ('t','n_unique','revisit','progress')] for r in records])[restore]
    kw.update(future_phi_feats=torch.tensor(np.random.default_rng(4).normal(size=(n,8)),dtype=torch.float32),
        future_context=contexts,prompt_metadata=records)
    return kw,non


class TinyTokenizer:
    pad_token_id=0;eos_token_id=1
    def encode(self,text,add_special_tokens=False):return [ord(c)+2 for c in text]
    def decode(self,ids,skip_special_tokens=False):return ''.join(chr(int(i)-2) for i in ids if i>1)
    def apply_chat_template(self,chat,**kwargs):return self.encode(chat[0]['content'])+[2]


class FixedAnchorTests(unittest.TestCase):
    def test_gain_identity_and_anchor_support(self):
        kw,_=fixture();a,d=ccpo_fixed_anchor_advantage(**kw);v=d['fixed_log_payload']['arrays']
        np.testing.assert_allclose(v['future_gain'],v['future_baseline']-v['history_baseline'])
        np.testing.assert_allclose(v['combined_pre'],v['history_adv']+v['future_gain'])
        np.testing.assert_allclose(v['history_adv'],v['future_gain']+v['future_residual'],atol=1e-6)
        self.assertGreater(d['fixed_future_gain_absmean'],0)
        np.testing.assert_array_equal(v['history_J'],v['future_J'])
        np.testing.assert_allclose(v['history_lambda_k'],v['future_lambda_k'])
        self.assertFalse(a.requires_grad)

    def test_gain_zero_preserves_history(self):
        kw,_=fixture();a,d=ccpo_fixed_anchor_advantage(gain_weight=0,**kw)
        np.testing.assert_allclose(a.numpy(),d['components_history'])

    def test_future_targets_exclude_entire_own_trajectory(self):
        kw,_=fixture();_,a=ccpo_fixed_anchor_advantage(**kw)
        kw['step_rewards']=kw['step_rewards'].clone();kw['step_rewards'][:4]+=100
        _,b=ccpo_fixed_anchor_advantage(**kw)
        for k in ['history_baseline','future_baseline','future_gain']:
            np.testing.assert_allclose(a['fixed_log_payload']['arrays'][k][:4],b['fixed_log_payload']['arrays'][k][:4],atol=1e-6)

    def test_shuffle_padding_invariance(self):
        kw,_=fixture();a,d=ccpo_fixed_anchor_advantage(**kw)
        ix=np.random.default_rng(2).permutation(np.r_[np.arange(len(a)),[0,3,7]])
        moved={k:(v[ix] if isinstance(v,(np.ndarray,torch.Tensor)) else v) for k,v in kw.items()}
        b,e=ccpo_fixed_anchor_advantage(**moved)
        torch.testing.assert_close(b,a[ix])
        self.assertEqual(d['fixed_unique_rows'],e['fixed_unique_rows'])

    def test_no_peer_gets_zero_gain_and_retains_history(self):
        kw,_=fixture();kw['anchor_obs']=kw['anchor_obs'].astype(object);kw['anchor_obs'][0]='unique'
        kw['prompt_metadata']=None
        a,d=ccpo_fixed_anchor_advantage(**kw);v=d['fixed_log_payload']['arrays']
        self.assertEqual(v['future_gain'][0],0)
        self.assertFalse(v['eligible'][0]);self.assertTrue(v['live'][0])
        self.assertAlmostEqual(float(a[0]),v['history_adv'][0])

    def test_terminal_future_includes_last_action_and_observation(self):
        _,non=fixture();_,_,_,rs=future_records(non)
        last=next(r for r in rs if r['traj_uid']=='traj0' and r['turn']==3)
        self.assertEqual(last['window_length'],1);self.assertTrue(last['terminal'])
        self.assertEqual(last['transitions'][0]['action'],'action_3')
        self.assertEqual(last['transitions'][0]['next_observation'],'TERMINAL_OBSERVATION')
        self.assertEqual(last['future_context']['t'],4)
        ids,_,_=tokenize_future(last,TinyTokenizer(),1000)
        text=TinyTokenizer().decode(ids)
        self.assertIn('Historical context for row 3',text)
        self.assertIn('action_3',text);self.assertIn('TERMINAL_OBSERVATION',text)

    def test_capture_is_separate_and_aligned(self):
        from verl import DataProto
        _,non=fixture();n=len(non['uid']);perm=np.random.default_rng(3).permutation(n)
        batch=DataProto.from_dict(tensors={'original':torch.arange(n)[:,None]},
            non_tensors={k:v[perm] for k,v in non.items()})
        class Reference:
            world_size=2
            def compute_ref_log_prob(self,data):
                # Hidden feature ties directly to the actual future prompt tokens.
                x=data.batch['input_ids'].float().sum(-1)[:,None]
                return DataProto.from_dict(tensors={'ccpo_phi_feats':x.expand(-1,8).clone()})
        before=batch.batch['original'].clone()
        capture_future_features(batch,TinyTokenizer(),Reference(),max_tokens=1000)
        torch.testing.assert_close(before,batch.batch['original'])
        _,restore,_,rs=future_records(batch.non_tensor_batch)
        expected=[]
        for r in rs:
            ids,_,_=tokenize_future(r,TinyTokenizer(),1000);expected.append(sum(ids)+1)
        np.testing.assert_allclose(batch.batch['ccpo_future_phi_feats'][:,0].numpy(),np.array(expected)[restore])

    def test_all_terms_saved_and_applied_components_sum(self):
        kw,_=fixture();a,d=ccpo_fixed_anchor_advantage(**kw)
        for normalized in (False,True):
            expected=(a-a.mean())/(a.std()+1e-6) if normalized else a
            with tempfile.TemporaryDirectory() as temp,patch.dict(os.environ,{'ACG_EXP_DIR':temp,'ACG_CCPO_FIXED_SNAPSHOT_EVERY':'1'}):
                finalize_fixed_logging(d,expected,kw['index'],normalized,1,0)
                out=Path(temp)/'outputs/fixed_anchor'
                with np.load(out/'step-0001.npz',allow_pickle=False) as z:
                    for k in ['history_baseline','future_baseline','history_adv','future_gain','future_residual','history_kernel','future_kernel','history_lambda_k','future_lambda_k','history_hidden','future_hidden','history_phi','future_phi','history_applied','gain_applied','combined_applied']:
                        self.assertIn(k,z)
                    np.testing.assert_allclose(z['history_applied']+z['gain_applied'],z['combined_applied'],atol=3e-5)
                with gzip.open(out/'step-0001.prompts.jsonl.gz','rt') as f:self.assertEqual(sum(1 for _ in f),len(a))
                self.assertTrue((out/'step-0001.csv').exists())

    def test_episode_advantage_cannot_enter_arm(self):
        kw,_=fixture();a,d=ccpo_fixed_anchor_advantage(**kw)
        with self.assertRaisesRegex(ValueError,'episode advantage'):
            finalize_fixed_logging(d,a,kw['index'],False,1,1)

    def test_episode_diagnostic_perturbation_cannot_change_policy_gradient(self):
        from test_episode_adv_isolation import load_compute_advantage, make_data, policy_gradient
        from gigpo import core_gigpo
        compute=load_compute_advantage();kw,non=fixture()
        results=[]
        for scale in (1.,100.):
            data=make_data(kw)
            data.batch['ccpo_future_phi_feats']=kw['future_phi_feats']
            data.non_tensor_batch['ccpo_future_context']=kw['future_context']
            data.meta_info['ccpo_future_metadata']=kw['prompt_metadata']
            episode=torch.arange(len(kw['index']),dtype=torch.float32)[:,None].expand_as(kw['response_mask'])*scale
            with tempfile.TemporaryDirectory() as temp,patch.dict(os.environ,{'ACG_CCPO_FIXED_ANCHOR':'1',
                'ACG_CCPO_OUTLOOK_HORIZON':'0','ACG_CCPO_OUTLOOK_BETA':'0','ACG_CCPO_EP_W':'0',
                'ACG_EXP_DIR':temp}),patch.object(core_gigpo,'episode_norm_reward',return_value=episode):
                compute(data,'CCPO',gamma=.95,gigpo_mode='mean_std_norm',ccpo_step_tag=1)
            results.append((data,*policy_gradient(data)))
        a,la,ga=results[0];b,lb,gb=results[1]
        torch.testing.assert_close(a.batch['advantages'],b.batch['advantages'],rtol=0,atol=0)
        torch.testing.assert_close(la,lb,rtol=0,atol=0)
        torch.testing.assert_close(ga,gb,rtol=0,atol=0)
        self.assertNotEqual(a.meta_info['ccpo_diag']['ccpo/adv_ep_absmean'],b.meta_info['ccpo_diag']['ccpo/adv_ep_absmean'])

    def test_trainer_dispatch_metrics_and_normalization(self):
        import ast
        from collections import defaultdict
        from gigpo import core_gigpo
        path=ROOT/'patches/verl-agent/verl/trainer/ppo/ray_trainer.py'
        tree=ast.parse(path.read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='compute_advantage')
        names=['GAE','GRPO','GRPO_PASSK','REINFORCE_PLUS_PLUS_BASELINE','REINFORCE_PLUS_PLUS','REMAX','RLOO','CCPO','GiGPO']
        ns=dict(DataProto=object,torch=torch,np=np,os=os,defaultdict=defaultdict,
                AdvantageEstimator=SimpleNamespace(**{n:n for n in names}),core_gigpo=core_gigpo)
        exec(compile(ast.Module(body=[fn],type_ignores=[]),str(path),'exec'),ns)
        kw,non=fixture();raw,_=ccpo_fixed_anchor_advantage(**kw)
        for mode in ['mean_norm','mean_std_norm']:
            data=SimpleNamespace(batch=dict(step_rewards=kw['step_rewards'],response_mask=kw['response_mask'],
                ccpo_phi_feats=kw['phi_feats'],ccpo_future_phi_feats=kw['future_phi_feats'],
                token_level_rewards=torch.zeros(len(raw),4)),non_tensor_batch=dict(non,
                episode_rewards=kw['episode_rewards'],is_action_valid=kw['is_action_valid'],
                rewards=kw['immediate_rewards'],ccpo_future_context=kw['future_context']),
                meta_info={'ccpo_future_metadata':kw['prompt_metadata']})
            with tempfile.TemporaryDirectory() as temp,patch.dict(os.environ,{'ACG_CCPO_FIXED_ANCHOR':'1',
                'ACG_CCPO_OUTLOOK_HORIZON':'0','ACG_CCPO_OUTLOOK_BETA':'0','ACG_CCPO_EP_W':'0',
                'ACG_CCPO_STEP_NORM':'mode','ACG_EXP_DIR':temp}):
                ns['compute_advantage'](data,'CCPO',gamma=.95,gigpo_mode=mode,ccpo_step_tag=1)
            expect=(raw-raw.mean())/(raw.std()+1e-6) if mode=='mean_std_norm' else raw
            torch.testing.assert_close(data.batch['advantages'],expect[:,None]*kw['response_mask'])
            metrics=data.meta_info['ccpo_diag']
            for k in ['fixed_future_gain_absmean','fixed_history_baseline_mean','fixed_future_baseline_mean',
                'fixed_history_lambda_k_mean','fixed_future_lambda_k_mean','fixed_history_applied_absmean',
                'fixed_gain_applied_absmean','fixed_combined_applied_absmean']:
                self.assertIn('ccpo/'+k,metrics)
            self.assertEqual(metrics['ccpo/fixed_episode_weight'],0)

if __name__=='__main__':unittest.main(verbosity=2)
