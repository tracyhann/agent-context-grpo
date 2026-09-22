#!/usr/bin/env python3
"""Prepare M11 1.5B H2 no-shrink EP1 peer/representation ablations; never launch."""
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

CONTROL='m11-h2-noshrink-active-episode-webshop-1.5b-2gpu-20260920'
CASES={
    'no-loo': dict(suffix='noloo', key='future-progress-h2-no-credit-shrinkage-active-episode-no-loo',
                   flag='ccpo_loo', value=0),
    'no-context-vector': dict(suffix='noctx', key='future-progress-h2-no-credit-shrinkage-active-episode-no-context-vector',
                              flag='ccpo_ctx_w', value=0.0),
    'cosine': dict(suffix='cosine', key='future-progress-h2-no-credit-shrinkage-active-episode-cosine',
                   flag='ccpo_wmode', value='cos'),
}
IMPLIED={'ccpo_loo':1,'ccpo_progress_history_weight':1.0}
SOURCES=['ablations/ablations.py','scripts/prepare_webshop_ep1_ablations.py','scripts/exp_run.py',
         'ccpo/arms.py','ccpo/core_ccpo.py','ccpo/future_progress.py','scripts/plot_metrics.py',
         'patches/verl-agent/verl/trainer/ppo/ray_trainer.py',
         'verl-agent/verl/trainer/ppo/ray_trainer.py','tests/test_webshop_ep1_ablations.py']


def dump(path,obj):
    path.write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n')


def run_name(variant,stamp):
    return f'm11-h2-noshrink-active-episode-{CASES[variant]["suffix"]}-webshop-1.5b-2gpu-{stamp}'


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
    assert new['ccpo_ep_w']==1 and new['ccpo_edge_w']==0
    assert new['ccpo_lk_fix']==new['ccpo_lam_fix']==1
    assert new['ccpo_progress_weight']==new['ccpo_progress_history_weight']==1
    assert new[spec['flag']]==spec['value']
    assert new['history_length']==new['ccpo_progress_horizon']==2
    assert new['total_epochs']==150 and new['max_steps']==15
    assert new['adv_mode']=='mean_norm' and new['model']=='Qwen/Qwen2.5-1.5B-Instruct'
    assert record['hydra_overrides']==er.build_command(new,str(folder))[3:]
    subprocess.run(['bash','-n',str(folder/'run.sh')],check=True)
    status=dict(status='prepared_not_launched',experiment=folder.name,variant=variant,
        ablation=spec['key'],canonical_id=canonical,benchmark='webshop',backbone='1.5b',steps=150,
        seed=new['seed'],history_length=2,future_horizon=2,history_weight=new['ccpo_progress_history_weight'],
        future_weight=new['ccpo_progress_weight'],episode_weight=1.,original_edge_weight=0.,
        context_stat_weight=new['ccpo_ctx_w'],lambda_k_fixed=1.,loo=bool(new['ccpo_loo']),
        weighting=new['ccpo_wmode'],
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
        'no-loo': 'Allow the query occurrence and all matching turns of its own trajectory into history/current/future readouts. Exact task/observation grouping remains; transport padding copies are still deduplicated. Distinct-trajectory support now includes self. Singleton exact groups have H=0 and V=their own label. This removes trajectory-exclusion protection from the contextual estimator.',
        'no-context-vector': 'Set ccpo_ctx_w=0: remove the 37-dimensional accumulated-statistics summary from history and both potential representations. The frozen hidden representation remains centered, stripped of its top three principal directions and L2-normalized (1536 dimensions). Keep the two-turn prompt history, LOO and soft exponential weights.',
        'cosine': 'Reuse the existing cosine EP1 implementation: w_ij=max(cos(z_i,z_j),0), normalized over eligible peers. If all weights vanish, retain the nearest eligible peer. Tau is inert. Keep the normalized hidden-plus-statistics representation (1573 dimensions), LOO and both prompt/future windows.',
    }[p['variant']]
    return f'''# M11 WebShop H2 no shrinkage, active episode: {p['variant']}

**Prepared only; not launched or queued.** Qwen2.5-1.5B-Instruct, 150 iterations,
seed {p['seed']}, 16 tasks x 8 rollouts, two GPUs (`{p['gpus']}` configured),
15-turn train/eval ceiling. Registry key `{p['ablation']}`; ID `{p['canonical_id']}`.

## One-setting comparison

Control: `{p['control']}`. [Full config diff](config-diff-from-control.json)
changes only the requested setting and experiment identity. Legacy implicit LOO
and history-weight defaults are explicitly resolved to 1. These are three
independent ablations, not one arm combining all three changes.

{meaning}

H/F/episode weights are **1/1/1**; original-edge weight is 0. Both history and
future windows are 2. Every usable baseline uses full strength, lambda_u=lambda_k=1;
kappa stays recorded but has no shrinkage effect. Existing fallback/unsupported
handling remains, except self can supply an exact baseline in the no-LOO arm.

## Credit and episode fusion

For target q in {{Y,Z}}, C_t[q] is the weighted contextual readout under this arm's
peer and feature rules. H_t=Y_t-C_t[Y], Z_t=gamma^(T-t)*R_episode, V_t=C_t[Z],
F_t=z_task(V_min(t+2,T)-V_t), gamma=0.95. Each endpoint uses its own observation
group. Terminal V is 10 for success and 0 otherwise. Y includes the existing
per-turn -0.1 invalid-action penalty; Z excludes it.

WebShop retains mean_norm: **A=mask*(H+F+E)**, where E=S_t-mean_task(S),
S_t=R_episode-0.1*invalid_t. Episode moments use turn rows, including their existing
length weighting, not one equally weighted record per trajectory. Future progress
is standardized per task; there is no final H+F normalization or renormalization
after adding E. Episode credit has unit weight and affects the actual PPO gradient.
Frozen features/credit targets remain detached.

## Protocol and logging

The 1K WebShop catalog, original item-option scorer, binary 10/0 rewards,
4096/512 prompt/response token limits, learning rate 1e-6, temperature 1.0 and
separate KL coefficient 0.01 remain matched. Save/evaluate every 5 steps.
History/current/future features, baselines, J/n_eff/lambda_k, self/own-trajectory
mass, raw/applied H/F/E and the complete actor identity remain logged. Removing
context summary may leave raw summary statistics in diagnostics; they do not
enter the similarity representation. No original edge credit is added.

## Reproduction and evidence

Prepare a fresh dated set (or select one with --variant):

```bash
python scripts/prepare_webshop_ep1_ablations.py --date YYYYMMDD
```

For a later authorized launch:

```bash
bash experiments/{p['experiment']}/run.sh
```

[config.json](config.json), [PREPARED.json](PREPARED.json),
[prepare-command.json](prepare-command.json), [source hashes](prepared-source-sha256.json)
and [VALIDATION.json](VALIDATION.json) record preparation and CPU checks.
[Shared comparison](../WEBSHOP_EP1_ABLATIONS.md) describes all three variants.
The cosine registry/estimator is the same as the September 20 cosine EP1 arm;
this dated set records current source/config provenance. Earlier artifacts are
preserved. No GPU smoke test or training result is claimed.
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
