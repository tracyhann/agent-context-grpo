#!/usr/bin/env python3
"""Prepare five M10 1.5B no-shrink EP0 history/future windows; never launch."""
import argparse
from datetime import date,datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'ablations'),str(ROOT/'scripts')]
import ablations
import exp_run as er

CONTROL='m10-h2-no-credit-shrinkage-alfworld-1.5b-2gpu-20260919'
WINDOWS={'h4f0':(4,0),'h0f4':(0,4),'h3f1':(3,1),'h1f3':(1,3),'h3f3':(3,3)}
IMPLIED={'ccpo_loo':1,'ccpo_progress_history_weight':1.0}
SOURCES=['ablations/ablations.py','ccpo/arms.py','ccpo/core_ccpo.py','ccpo/future_progress.py',
 'ccpo/phi_capture.py','scripts/exp_run.py','scripts/prepare_alfworld_window_ablations.py',
 'patches/verl-agent/verl/trainer/ppo/ray_trainer.py','verl-agent/verl/trainer/ppo/ray_trainer.py',
 'patches/verl-agent/agent_system/environments/env_manager.py','verl-agent/agent_system/environments/env_manager.py',
 'patches/verl-agent/agent_system/environments/prompts/alfworld.py','verl-agent/agent_system/environments/prompts/alfworld.py',
 'tests/test_alfworld_windows.py']


def dump(path,obj):path.write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n')

def key_for(window):
    h,f=WINDOWS[window]
    return f'future-progress-noshrink-history{h}-future{f}'

def run_name(window,stamp):
    h,f=WINDOWS[window]
    return f'm10-hist{h}-fut{f}-noshrink-alfworld-1.5b-2gpu-{stamp}'


def prepare(window,stamp):
    h,f=WINDOWS[window];key=key_for(window);canonical,method=ablations.build(key,'alfworld')
    control=ROOT/'experiments'/CONTROL
    old=json.loads((control/'config.json').read_text())['config']
    missing={k:v for k,v in IMPLIED.items() if k not in old};old.update(missing)
    folder=ROOT/'experiments'/run_name(window,stamp)
    if folder.exists():raise FileExistsError(f'Refusing to overwrite existing experiment: {folder}')
    cfg={k:v for k,v in old.items() if k in er.DEFAULTS};cfg.update(method)
    argv=[sys.executable,str(ROOT/'scripts/exp_run.py'),'--name',folder.name.rsplit('-',1)[0],
          '--date',stamp,'--dry-run',*ablations.arms.flags(cfg)]
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
      HF_HOME=str(ROOT/'hf'),ALFWORLD_DATA=str(ROOT/'alfworld_data'),ACG_DATA_DIR=str(ROOT/'envdata/verl_data'))
    result=subprocess.run(argv,cwd=ROOT,env=env,capture_output=True,text=True)
    if result.returncode:raise RuntimeError(result.stdout+result.stderr)
    (folder/'prepare.log').write_text(result.stdout+result.stderr)
    record=json.loads((folder/'config.json').read_text());new=record['config']
    if set(new)!=set(old):raise ValueError(f'Unexpected config keys: {set(new)^set(old)}')
    changes={k:dict(control=old[k],prepared=new[k]) for k in old if new[k]!=old[k]}
    expected={'exp_id','history_length','ccpo_progress_horizon'}
    if not h:expected.add('ccpo_progress_history_weight')
    if not f:expected.add('ccpo_progress_weight')
    if set(changes)!=expected:raise ValueError(f'Unexpected main-control delta: {changes}')
    for k,v in method.items():
        if new[k]!=v:raise ValueError(f'Method mismatch: {k}')
    assert new['history_length']==h and new['ccpo_progress_horizon']==f
    assert new['ccpo_progress_history_weight']==int(h>0) and new['ccpo_progress_weight']==int(f>0)
    assert new['ccpo_ep_w']==new['ccpo_edge_w']==0
    assert new['ccpo_lk_fix']==new['ccpo_lam_fix']==new['ccpo_loo']==1
    assert new['ccpo_ctx_w']==1 and new['ccpo_phi']=='hidden+ctx'
    assert new['total_epochs']==150 and new['max_steps']==50 and new['adv_mode']=='mean_std_norm'
    assert record['hydra_overrides']==er.build_command(new,str(folder))[3:]
    subprocess.run(['bash','-n',str(folder/'run.sh')],check=True)
    p=dict(status='prepared_not_launched',experiment=folder.name,window=window,ablation=key,
       canonical_id=canonical,benchmark='alfworld',backbone='1.5b',steps=150,seed=new['seed'],
       history_length=h,future_horizon=f,history_weight=float(h>0),future_weight=float(f>0),
       episode_weight=0.,original_edge_weight=0.,context_stat_weight=1.,lambda_k_fixed=1.,loo=True,
       max_steps=50,gpus=new['gpus'],gpu_count=2,control=str(control.relative_to(ROOT)),
       launched=False,queued=False,gpu_preflight_performed=False,
       prepared_at=datetime.now(timezone.utc).isoformat(),source_git=er.git_state())
    record['preparation']=p
    record['reference_protocol']=dict(source_config=str(control.relative_to(ROOT)/'config.json'),
        changes=changes,missing_control_defaults=missing)
    dump(folder/'config.json',record);dump(folder/'PREPARED.json',p)
    dump(folder/'config-diff-from-control.json',dict(control=p['control'],changes=changes,missing_control_defaults=missing))
    dump(folder/'prepare-command.json',dict(argv=argv,cwd=str(ROOT),gpu_visibility_for_preparation=''))
    dump(folder/'prepared-source-sha256.json',{s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in SOURCES})
    (folder/'NOTES.md').write_text(notes(p))
    return folder


def notes(p):
    h,f=p['history_length'],p['future_horizon']
    formula='H' if not f else ('F' if not h else 'H + F')
    return f'''# M10 ALFWorld no shrinkage, EP0: history {h} / future {f}

**Prepared only; not launched or queued.** Qwen2.5-1.5B-Instruct, 150 steps,
seed {p['seed']}, two GPUs (`{p['gpus']}` configured, not reserved), 50-turn cap.
Registry key `{p['ablation']}`; canonical ID `{p['canonical_id']}`.

## Window and method

Prompt-history length: **{h} previous observation/action pairs**.
Future-progress horizon: **{f} steps**. These are window lengths, not numerical
advantage multipliers. History/future/episode actor weights are
**{p['history_weight']:g} / {p['future_weight']:g} / 0**.

At history=0, both actor and frozen reference see the task goal, current
observation and action instructions, with no previous observation/action pairs;
the historical residual has zero actor weight. At future=0, the future horizon
and actor weight are both zero; current endpoints produce zero raw/normalized
future credit and no future support. Canonicalization, verified feature capture,
history/value diagnostics and component snapshots remain enabled in that mode.

The accumulated context-statistics block remains enabled for every arm, even at
history=0. Prompt windows do not truncate rollout memory or return labels.
Processed frozen hidden + context features, soft exponential kernel (tau 0.15),
three-direction whitening and whole-query-trajectory LOO are retained. Usable
readouts are full strength, lambda_u=lambda_k=1, with existing task fallback.

C_t[q] is the contextual readout over matching peers in the same task/observation
group, excluding the query trajectory. H_t=Y_t-C_t[Y],
Z_t=gamma^(T-t)*R_episode, V_t=C_t[Z], gamma=0.95.
For f>0, F_t=z_task(V_min(t+f,T)-V_t); for f=0, F_t=0.
Terminal potential is success 10 / failure 0. Each nonterminal future endpoint
uses its own observation group. Y retains the invalid-action penalty, Z does not.
The ALFWorld actor advantage is **A=mask*N_CC({formula})**, where N_CC is the
existing per-task mean/sample-std normalization over enabled contextual support.

Episode rewards still supply return labels. The separate episode-advantage and
original-edge coefficients are 0. Raw H, episode and original-edge diagnostics
remain available when their actor contribution is disabled. No extra rollout
steps are generated to fill a window; future endpoints clip at termination.

## Control and reproduction

[Exact config diff](config-diff-from-control.json) compares with
`{p['control']}`. The H2 main control has two prompt-history turns and two future
steps. Only the two window lengths, zero-side coefficients where needed, and
experiment identity change. Learning/evaluation settings stay matched: 16 tasks
x 8 rollouts, 150 steps, evaluation/checkpoints every 5, 2048/512 prompt/response
limits, 50 train/eval turns. Four-turn prompts keep the same token limits.

[Shared plan and equations](../ALFWORLD_WINDOW_ABLATIONS.md).
Prepare for a fresh date without training:

```bash
python scripts/prepare_alfworld_window_ablations.py --window {p['window']} --date YYYYMMDD
```

Prepared launch command, for separate scheduling:

```bash
bash experiments/{p['experiment']}/run.sh
```

[config.json](config.json), [prepare-command.json](prepare-command.json),
[prepared-source-sha256.json](prepared-source-sha256.json), and
[VALIDATION.json](VALIDATION.json) record reproduction and CPU verification.
No GPU run or benchmark result is implied by preparation.
'''


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--date',default=date.today().strftime('%Y%m%d'))
    parser.add_argument('--window',choices=tuple(WINDOWS),action='append',help='Repeat to select a subset; default: all five')
    args=parser.parse_args()
    try:datetime.strptime(args.date,'%Y%m%d')
    except ValueError:parser.error('Use a YYYYMMDD date')
    windows=list(dict.fromkeys(args.window or WINDOWS))
    for w in windows:
        if (ROOT/'experiments'/run_name(w,args.date)).exists():parser.error(f'Experiment exists: {run_name(w,args.date)}')
    folders=[prepare(w,args.date) for w in windows]
    print(json.dumps(dict(status='prepared_not_launched',experiments=[str(p.relative_to(ROOT)) for p in folders])))


if __name__=='__main__':main()
