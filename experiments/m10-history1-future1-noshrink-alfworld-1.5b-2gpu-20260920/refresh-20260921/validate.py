"""CPU-only audit of the existing H1/F1 ALFWorld preparation; never launches."""
import ast
import copy
import importlib.util
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[3]
EXP=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'tests'),str(ROOT/'scripts'),str(ROOT)]
os.environ.update(CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',
                  ACG_CCPO_DUMP='',ACG_EXP_DIR='',ACG_CCPO_LOO='1')
import exp_run as er
from test_future_progress import arguments,ccpo_future_progress_advantage,core,np,FutureProgressTests
record=json.loads((EXP/'config.json').read_text());cfg=record['config']
expected=dict(history_length=1,ccpo_progress_horizon=1,ccpo_progress_weight=1.,
 ccpo_progress_history_weight=1.,ccpo_lk_fix=1.,ccpo_lam_fix=1.,ccpo_ep_w=0.,
 ccpo_edge_w=0.,ccpo_loo=1,ccpo_phi='hidden+ctx',ccpo_ctx_w=1.,ccpo_wmode='soft',
 total_epochs=150,seed=0,model='Qwen/Qwen2.5-1.5B-Instruct',max_steps=50)
for k,v in expected.items():assert cfg[k]==v,(k,cfg[k],v)
control=json.loads((ROOT/'experiments/m10-h2-no-credit-shrinkage-alfworld-1.5b-2gpu-20260919/config.json').read_text())['config']
control.update(ccpo_loo=1,ccpo_progress_history_weight=1.)
assert set(control)==set(cfg)
assert {k for k in cfg if cfg[k]!=control[k]}=={'exp_id','history_length','ccpo_progress_horizon'}
assert record['hydra_overrides']==er.build_command(cfg,str(EXP))[3:]
for k,env in er.ENV_KEYS.items():assert record['env'][env]==str(cfg[k]),env
assert 'env.history_length=1' in record['hydra_overrides']
subprocess.run(['bash','-n',str(EXP/'run.sh')],check=True)
print('PASS: complete config, resolved environment, H2 matched delta and shell syntax')

with patch.object(core,'_LK_FIX','1.0'),patch.object(core,'_LOO','1'):
 _,d=ccpo_future_progress_advantage(**arguments(),horizon=1)
 a=d['progress_payload']['arrays'];m=~a['terminal'];endpoint=a['endpoint_index'][m]
 np.testing.assert_array_equal(a['window_length'],1)
 np.testing.assert_array_equal(a['turn_index'][endpoint],a['turn_index'][m]+1)
 np.testing.assert_array_equal(a['traj_uid'][endpoint],a['traj_uid'][m])
 for prefix,keep in [('history',a['history_live']),('current',np.isfinite(a['current_value'])),('future',m&a['eligible'])]:
  np.testing.assert_array_equal(a[prefix+'_lambda_k'][keep],1.)
  np.testing.assert_array_equal(a[prefix+'_same_traj_mass'][keep],0.)
  key='history_baseline' if prefix=='history' else prefix+'_value'
  np.testing.assert_allclose(a[key][keep],a[prefix+'_kernel'][keep],atol=1e-12)
 np.testing.assert_array_equal(a['future_value'][a['terminal']],a['episode_rewards'][a['terminal']])
 suite=unittest.TestSuite([FutureProgressTests('test_trainer_components_logging_normalization_and_gradient_isolation')])
 result=unittest.TextTestRunner(verbosity=2).run(suite)
 assert result.wasSuccessful()
print('PASS: H1 endpoints, full-strength readouts, LOO, terminal sentinels and actual trainer/PPO gradient isolation')

# Execute the actual pure ALFWorld prompt builder and memory implementation,
# avoiding environment creation, Ray actors and game/model loading.
mem_path=ROOT/'verl-agent/agent_system/memory/memory.py'
cls=next(n for n in ast.parse(mem_path.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='SimpleMemory')
ns=dict(BaseMemory=object,List=List,Dict=Dict,Any=Any,Tuple=Tuple,os=os)
exec(compile(ast.Module(body=[cls],type_ignores=[]),str(mem_path),'exec'),ns)
memory=ns['SimpleMemory']();memory.reset(1)
for i in range(3):memory.store({'text_obs':[f'PAST_OBS_{i}'],'action':[f'PAST_ACT_{i}']})
before=copy.deepcopy(memory._data)
env_path=ROOT/'verl-agent/agent_system/environments/env_manager.py'
cls=next(n for n in ast.parse(env_path.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='AlfWorldEnvironmentManager')
fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='build_text_obs')
ns.update(runpy.run_path(str(ROOT/'verl-agent/agent_system/environments/prompts/alfworld.py')))
exec(compile(ast.Module(body=[fn],type_ignores=[]),str(env_path),'exec'),ns)
env=SimpleNamespace(memory=memory,config=SimpleNamespace(env=SimpleNamespace(history_length=cfg['history_length'])),tasks=['TASK_MARKER'])
with patch.dict(os.environ,ACG_COMPACT_BUDGET='0'):
 prompt=ns['build_text_obs'](env,['CURRENT_OBS'],[['look','help']])[0]
for x in ['PAST_OBS_2','PAST_ACT_2','CURRENT_OBS','TASK_MARKER']:assert x in prompt,x
for x in ['PAST_OBS_0','PAST_ACT_0','PAST_OBS_1','PAST_ACT_1']:assert x not in prompt,x
assert memory._data==before
print('PASS: actual ALFWorld prompt uses only the previous observation/action pair and retains accumulated memory')
print('ALL TARGETED CPU CHECKS PASSED; NO TRAINING LAUNCHED')
