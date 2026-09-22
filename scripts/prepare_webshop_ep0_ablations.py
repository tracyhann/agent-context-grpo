#!/usr/bin/env python3
"""Prepare M11 1.5B H2 no-shrink EP0 component ablations; never launch."""
import argparse
from datetime import date, datetime, timezone
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

CONTROL='m11-h2-no-credit-shrinkage-webshop-1.5b-2gpu-20260919'
CASES={
    'no-future': dict(suffix='history-only', key='future-progress-h2-no-credit-shrinkage-history-only',
                      flag='ccpo_progress_weight'),
    'no-history': dict(suffix='future-only', key='future-progress-h2-no-credit-shrinkage-future-only',
                       flag='ccpo_progress_history_weight'),
    'no-context-vector': dict(suffix='noctx', key='future-progress-h2-no-credit-shrinkage-no-context-vector',
                              flag='ccpo_ctx_w'),
}
IMPLIED={'ccpo_loo':1,'ccpo_progress_history_weight':1.0}
SOURCES=['ablations/ablations.py','scripts/prepare_webshop_ep0_ablations.py','scripts/exp_run.py',
         'ccpo/arms.py','ccpo/core_ccpo.py','ccpo/future_progress.py','scripts/plot_metrics.py',
         'patches/verl-agent/verl/trainer/ppo/ray_trainer.py',
         'verl-agent/verl/trainer/ppo/ray_trainer.py','tests/test_webshop_ep0_ablations.py']


def dump(path,obj):
    path.write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n')


def run_name(variant,stamp):
    return f'm11-h2-noshrink-{CASES[variant]["suffix"]}-webshop-1.5b-2gpu-{stamp}'


def prepare(variant,stamp):
    spec=CASES[variant];canonical,method=ablations.build(spec['key'],'webshop')
    control=ROOT/'experiments'/CONTROL
    old=json.loads((control/'config.json').read_text())['config']
    missing={k:v for k,v in IMPLIED.items() if k not in old};old.update(missing)
    folder=ROOT/'experiments'/run_name(variant,stamp)
    if folder.exists():raise FileExistsError(f'Refusing to overwrite existing experiment: {folder}')
    cfg={k:v for k,v in old.items() if k in er.DEFAULTS};cfg.update(method)
    argv=[sys.executable,str(ROOT/'scripts/exp_run.py'),'--name',folder.name.rsplit('-',1)[0],
          '--date',stamp,'--dry-run',*ablations.arms.flags(cfg)]
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',
             MKL_NUM_THREADS='1',HF_HOME=str(ROOT/'hf'),ALFWORLD_DATA=str(ROOT/'alfworld_data'),
             ACG_DATA_DIR=str(ROOT/'envdata/verl_data'))
    result=subprocess.run(argv,cwd=ROOT,env=env,capture_output=True,text=True)
    if result.returncode:raise RuntimeError(result.stdout+result.stderr)
    (folder/'prepare.log').write_text(result.stdout+result.stderr)
    record=json.loads((folder/'config.json').read_text());new=record['config']
    if set(new)!=set(old):raise ValueError(f'Unexpected config keys: {set(new)^set(old)}')
    changes={k:dict(control=old[k],prepared=new[k]) for k in old if new[k]!=old[k]}
    if set(changes)!={'exp_id',spec['flag']}:raise ValueError(f'Unexpected control delta: {changes}')
    for k,v in method.items():
        if new[k]!=v:raise ValueError(f'Method mismatch: {k}')
    assert new['ccpo_ep_w']==new['ccpo_edge_w']==0
    assert new['ccpo_lk_fix']==new['ccpo_lam_fix']==new['ccpo_loo']==1
    assert new['history_length']==new['ccpo_progress_horizon']==2
    assert new['total_epochs']==150 and new['max_steps']==15
    assert new['adv_mode']=='mean_norm' and new['model']=='Qwen/Qwen2.5-1.5B-Instruct'
    assert record['hydra_overrides']==er.build_command(new,str(folder))[3:]
    subprocess.run(['bash','-n',str(folder/'run.sh')],check=True)
    status=dict(status='prepared_not_launched',experiment=folder.name,variant=variant,
        ablation=spec['key'],canonical_id=canonical,benchmark='webshop',backbone='1.5b',steps=150,
        seed=new['seed'],history_length=2,future_horizon=2,history_weight=new['ccpo_progress_history_weight'],
        future_weight=new['ccpo_progress_weight'],episode_weight=0.,original_edge_weight=0.,
        context_stat_weight=new['ccpo_ctx_w'],lambda_k_fixed=1.,loo=True,
        gpu_count=2,gpus=new['gpus'],max_steps=15,control=str(control.relative_to(ROOT)),
        launched=False,queued=False,gpu_preflight_performed=False,
        prepared_at=datetime.now(timezone.utc).isoformat(),source_git=er.git_state())
    record['preparation']=status
    record['reference_protocol']=dict(source_config=str(control.relative_to(ROOT)/'config.json'),
        changes=changes,missing_control_defaults=missing)
    dump(folder/'config.json',record);dump(folder/'PREPARED.json',status)
    dump(folder/'config-diff-from-control.json',dict(control=status['control'],changes=changes,
                                                    missing_control_defaults=missing))
    dump(folder/'prepare-command.json',dict(argv=argv,cwd=str(ROOT),gpu_visibility_for_preparation=''))
    dump(folder/'prepared-source-sha256.json',{s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in SOURCES})
    (folder/'NOTES.md').write_text(notes(status))
    return folder


def notes(p):
    meaning={
        'no-future': 'Only historical residual H contributes to the actor. Future progress is calculated and logged at weight 0; it cannot affect credit or its support mask.',
        'no-history': 'Only future progress F contributes to the actor. Historical residual H is diagnostic at weight 0. History-2 prompts still supply the representations used by the potential estimator.',
        'no-context-vector': 'Both H and F contribute. ccpo_ctx_w=0 removes the accumulated statistics block from history and both potential representations. The frozen hidden state is still whitened and normalized, with the original soft exponential kernel.',
    }[p['variant']]
    formula={'no-future':'H','no-history':'F','no-context-vector':'H_hidden + F_hidden'}[p['variant']]
    return f'''# M11 WebShop H2 no shrinkage, EP0: {p['variant']}

**Prepared only; not launched or queued.** Qwen2.5-1.5B-Instruct, 150 steps,
seed {p['seed']}, two GPUs (`{p['gpus']}` configured, not reserved), 15 turns.
Registry key `{p['ablation']}`; canonical ID `{p['canonical_id']}`.

## Method

History/future/episode coefficients: **{p['history_weight']:g} / {p['future_weight']:g} / 0**.
Context-statistics weight: **{p['context_stat_weight']:g}**. Original-edge weight 0.
Prompt history and diagnostic future horizon remain 2/2; LOO excludes the whole
query trajectory. All usable contextual readouts have lambda_u=lambda_k=1.

{meaning}

Let C_t[q] be the kernel-weighted contextual readout of peers in the same task
and observation group, excluding the query trajectory. Retain existing task
fallback, unsupported-row handling and terminal success/failure potential 10/0.
H_t=Y_t-C_t[Y], Z_t=gamma^(T-t)*R_episode, V_t=C_t[Z],
F_t=z_task(V_min(t+2,T)-V_t), gamma=0.95. Y retains the historical invalid-action
penalty; Z is the unpenalized potential label. The WebShop actor advantage is
**A=mask*({formula})**. F retains per-task standardization; the WebShop mean_norm
path does not add a combined standard-deviation normalization.

The policy and frozen reference keep two prompt-history turns. No-context means
no accumulated statistics block in similarity features; it does not erase prompt
history or bypass hidden-state processing, and it does not switch to cosine.
Episode rewards still label the returns. Raw episode advantage remains logged
but its applied contribution and original-edge contribution are exactly zero.

## Matched control and diagnostics

[Config diff](config-diff-from-control.json) changes one scientific setting from
`{p['control']}`, plus experiment identity. Both control and variant use EP0;
the earlier WebShop EP1 component arms are separate comparisons. Implicit legacy
history-weight/LOO defaults of 1 are made explicit in this preparation.

16 tasks x 8 rollouts; 1K WebShop catalog/scorer; evaluation and checkpoints every
5 steps. Frozen features, penalties, terminal handling and the standard main
protocol are retained. History/current/future raw readouts, J/n_eff/lambda_k,
actual coefficients, raw and applied H/F/episode terms, actor identity checks and
per-step snapshots remain enabled. Disabled channels have exactly zero applied
credit. [Shared equations and comparison table](../WEBSHOP_EP0_ABLATIONS.md).

## Reproduction

Prepare for a fresh date without training:

```bash
python scripts/prepare_webshop_ep0_ablations.py --variant {p['variant']} --date YYYYMMDD
```

Prepared launch command, to be scheduled separately:

```bash
bash experiments/{p['experiment']}/run.sh
```

[config.json](config.json), [prepare-command.json](prepare-command.json) and
[prepared-source-sha256.json](prepared-source-sha256.json) record the resolved
recipe and sources. [VALIDATION.json](VALIDATION.json) records CPU-only checks;
no GPU execution or benchmark training result is implied.
'''


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--date',default=date.today().strftime('%Y%m%d'))
    parser.add_argument('--variant',choices=tuple(CASES),action='append',help='Repeat to select a subset; default: all three')
    args=parser.parse_args()
    try:datetime.strptime(args.date,'%Y%m%d')
    except ValueError:parser.error('Use a YYYYMMDD date')
    variants=list(dict.fromkeys(args.variant or CASES))
    for v in variants:
        if (ROOT/'experiments'/run_name(v,args.date)).exists():parser.error(f'Experiment exists: {run_name(v,args.date)}')
    folders=[prepare(v,args.date) for v in variants]
    print(json.dumps(dict(status='prepared_not_launched',experiments=[str(p.relative_to(ROOT)) for p in folders])))


if __name__=='__main__':main()
