"""WebShop EP0 ablations: exact prepared protocols and actual PPO behavior."""
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from test_future_progress import core,np,torch,load_compute_advantage,policy_gradient,core_gigpo
from test_future_progress_active_episode import training_fixture,reward_data

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import prepare_webshop_ep0_ablations as prep


class WebshopEp0Tests(unittest.TestCase):
    def test_three_arms_match_ep0_control_and_resolve_to_expected_runtime(self):
        control=json.loads((ROOT/'experiments'/prep.CONTROL/'config.json').read_text())['config']
        for k,v in prep.IMPLIED.items():control.setdefault(k,v)
        for variant,spec in prep.CASES.items():
            folder=ROOT/'experiments'/prep.run_name(variant,'20260922')
            record=json.loads((folder/'config.json').read_text());cfg=record['config']
            self.assertEqual(set(cfg),set(control))
            self.assertEqual({k for k in cfg if cfg[k]!=control[k]},{'exp_id',spec['flag']})
            self.assertEqual(cfg[spec['flag']],0.)
            for key in ('ccpo_ep_w','ccpo_edge_w'):self.assertEqual(cfg[key],0.)
            for key in ('ccpo_lk_fix','ccpo_lam_fix','ccpo_loo'):self.assertEqual(cfg[key],1.)
            self.assertEqual(cfg['ccpo_progress_horizon'],2);self.assertEqual(cfg['history_length'],2)
            self.assertEqual(cfg['total_epochs'],150);self.assertEqual(cfg['max_steps'],15)
            self.assertEqual(cfg['model'],'Qwen/Qwen2.5-1.5B-Instruct')
            self.assertEqual(record['hydra_overrides'],prep.er.build_command(cfg,str(folder))[3:])
            for k,e in prep.er.ENV_KEYS.items():self.assertEqual(record['env'][e],str(cfg[k]))
            self.assertIn('trainer.n_gpus_per_node=2',record['hydra_overrides'])
            self.assertIn('env.history_length=2',record['hydra_overrides'])
            status=json.loads((folder/'PREPARED.json').read_text())
            self.assertFalse(status['launched']);self.assertFalse(status['queued'])
            self.assertFalse(any(f.is_file() for f in (folder/'outputs').rglob('*')))
            self.assertIn('--dry-run',json.loads((folder/'prepare-command.json').read_text())['argv'])

    def test_actual_prepared_trainer_paths_have_no_episode_credit_and_correct_components(self):
        compute=load_compute_advantage();kw=training_fixture()
        # Actual 1.5B hidden-state width, for the no-context dimensionality assertion.
        kw['phi_feats']=torch.tensor(np.random.default_rng(90).normal(size=(len(kw['index']),1536)),dtype=torch.float32)
        for variant in prep.CASES:
            folder=ROOT/'experiments'/prep.run_name(variant,'20260922')
            record=json.loads((folder/'config.json').read_text());cfg=record['config']
            results=[]
            for perturb in [False,True]:
                with tempfile.TemporaryDirectory() as tmp:
                    env=dict(record['env'],CUDA_VISIBLE_DEVICES='',ACG_EXP_DIR=tmp,ACG_CCPO_DUMP='')
                    raw=torch.arange(len(kw['index']),dtype=torch.float32)[:,None]*kw['response_mask']
                    episode=(raw*31+100)*kw['response_mask'] if perturb else raw
                    data=reward_data(kw)
                    original_context=core.derive_context
                    def context(*args,**kwargs):
                        values=original_context(*args,**kwargs)
                        return [dict(v,t=99,n_unique=37,progress=.01,revisit=1.) for v in values] if perturb and variant=='no-context-vector' else values
                    with patch.dict(os.environ,env),patch.multiple(core,_LK_FIX='1.0',_LAM_FIX='1.0',
                        _CTX_W=cfg['ccpo_ctx_w'],_PHI_MODE=cfg['ccpo_phi'],_LOO='1',_GATE='hard',
                        _WMODE='soft',_TAU_ENV='.15',_JW_C=0.,_STD_MODE='task',_BACKOFF_TASK=True,
                        _SIM=0.,_SIM_BACKOFF=0.,_EDGE_W=0.),\
                        patch.object(core_gigpo,'episode_norm_reward',return_value=episode),\
                        patch.object(core,'derive_context',side_effect=context),redirect_stdout(io.StringIO()):
                        compute(data,'CCPO',gamma=cfg['gamma'],gigpo_mode=cfg['adv_mode'],ccpo_step_tag=1)
                    results.append((data,*policy_gradient(data)))
                    a=dict(np.load(Path(tmp)/'outputs/future_progress/step-0001.npz',allow_pickle=False))
                    np.testing.assert_array_equal(a['episode_applied'],0.)
                    np.testing.assert_allclose(a['actor_applied'],a['history_applied']+a['future_applied'],atol=3e-5)
                    np.testing.assert_allclose(a['actor_applied'][:,None]*kw['response_mask'].numpy(),data.batch['advantages'].numpy(),atol=3e-5)
                    if variant=='no-future':
                        np.testing.assert_array_equal(a['future_applied'],0.)
                        np.testing.assert_array_equal(a['live'],a['history_live'])
                    elif variant=='no-history':
                        np.testing.assert_array_equal(a['history_applied'],0.)
                        np.testing.assert_array_equal(a['live'],a['eligible'])
                    else:
                        self.assertEqual(a['current_phi'].shape[1],1536)
                        np.testing.assert_array_equal(a['current_phi'],core.whiten_feats(kw['phi_feats'].numpy()))
                    self.assertEqual(data.meta_info['ccpo_diag']['ccpo/progress_episode_weight'],0.)
                    self.assertEqual(data.meta_info['ccpo_diag']['ccpo/progress_loo'],1.)
                    for prefix in ['history','current','future']:
                        mask=np.isfinite(a[prefix+'_lambda_k'])
                        np.testing.assert_array_equal(a[prefix+'_lambda_k'][mask],1.)
                    self.assertFalse(data.batch['advantages'].requires_grad)
                    self.assertTrue(torch.isfinite(results[-1][2]).all())
                    self.assertGreater(float(results[-1][2].abs().sum()),0.)
            torch.testing.assert_close(results[0][0].batch['advantages'],results[1][0].batch['advantages'],rtol=0,atol=0)
            for i in [1,2]:torch.testing.assert_close(results[0][i],results[1][i],rtol=0,atol=0)

    def test_preparation_refuses_overwrite(self):
        with patch.object(prep.subprocess,'run') as run:
            with self.assertRaises(FileExistsError):prep.prepare('no-future','20260922')
        run.assert_not_called()


if __name__=='__main__':unittest.main()
