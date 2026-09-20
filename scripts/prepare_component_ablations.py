#!/usr/bin/env python3
"""Prepare four matched 1.5B H2 history/future component ablations; never launch."""
import argparse
from datetime import date, datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'ablations'))
import ablations
sys.path.insert(0, str(ROOT/'scripts'))
import exp_run as er

CONTROLS = {
    'alfworld': 'm10-h2-no-credit-shrinkage-alfworld-1.5b-2gpu-20260919',
    'webshop': 'm11-h2-noshrink-active-episode-webshop-1.5b-2gpu-20260920',
}
SOURCES = ['ccpo/future_progress.py', 'scripts/exp_run.py', 'ablations/ablations.py',
           'scripts/prepare_component_ablations.py', 'ccpo/core_ccpo.py',
           'patches/verl-agent/verl/trainer/ppo/ray_trainer.py',
           'verl-agent/verl/trainer/ppo/ray_trainer.py', 'scripts/plot_metrics.py']


def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2, sort_keys=True)+'\n')


def key_for(benchmark, component):
    suffix = '-active-episode' if benchmark == 'webshop' else ''
    return f'future-progress-h2-no-credit-shrinkage-{component}-only{suffix}'


def run_name(benchmark, component, stamp):
    prefix = 'm10' if benchmark == 'alfworld' else 'm11'
    suffix = '-active-episode' if benchmark == 'webshop' else ''
    return f'{prefix}-h2-noshrink-{component}-only{suffix}-{benchmark}-1.5b-2gpu-{stamp}'


def prepare(benchmark, component, stamp):
    key = key_for(benchmark, component)
    canonical, method = ablations.build(key, benchmark)
    control = ROOT/'experiments'/CONTROLS[benchmark]
    old = json.loads((control/'config.json').read_text())['config']
    old.setdefault('ccpo_progress_history_weight', 1.0)  # Implicit legacy coefficient.
    name = run_name(benchmark, component, stamp)
    folder = ROOT/'experiments'/name
    if folder.exists():
        raise FileExistsError(f'Refusing to overwrite existing experiment: {folder}')
    cfg = {k: v for k, v in old.items() if k in er.DEFAULTS}
    cfg.update(method)
    command = [sys.executable, str(ROOT/'scripts/exp_run.py'), '--name', name.rsplit('-',1)[0],
               '--date', stamp, '--dry-run', *ablations.arms.flags(cfg)]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
               MKL_NUM_THREADS='1', HF_HOME=str(ROOT/'hf'), ALFWORLD_DATA=str(ROOT/'alfworld_data'),
               ACG_DATA_DIR=str(ROOT/'envdata/verl_data'))
    completed = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True)
    if completed.returncode:
        raise RuntimeError(completed.stdout+completed.stderr)
    (folder/'prepare.log').write_text(completed.stdout+completed.stderr)
    record = json.loads((folder/'config.json').read_text()); new = record['config']
    changes = {k: dict(control=old[k], prepared=new[k]) for k in old if old[k] != new[k]}
    flag = 'ccpo_progress_weight' if component == 'history' else 'ccpo_progress_history_weight'
    if set(old) != set(new) or set(changes) != {'exp_id', flag}:
        raise ValueError(f'Unexpected change from matched control: {changes}')
    for k,v in method.items():
        if new[k] != v: raise ValueError(f'Method mismatch: {k}')
    assert new['ccpo_lk_fix'] == 1 and new['ccpo_edge_w'] == 0
    assert new['ccpo_ep_w'] == int(benchmark == 'webshop')
    assert new['total_epochs'] == 150 and new['history_length'] == new['ccpo_progress_horizon'] == 2
    assert record['hydra_overrides'] == er.build_command(new, str(folder))[3:]
    subprocess.run(['bash','-n',str(folder/'run.sh')],check=True)
    p = dict(status='prepared_not_launched', experiment=name, ablation=key, canonical_id=canonical,
             benchmark=benchmark, component=component, backbone='1.5b', steps=150, seed=new['seed'],
             history_weight=new['ccpo_progress_history_weight'], future_weight=new['ccpo_progress_weight'],
             episode_weight=new['ccpo_ep_w'], history_length=2, future_horizon=2,
             lambda_k_fixed=1., original_edge_weight=0., gpus=new['gpus'], gpu_count=2,
             max_steps=new['max_steps'], control=str(control.relative_to(ROOT)),
             launched=False, queued=False, gpu_preflight_performed=False,
             prepared_at=datetime.now(timezone.utc).isoformat(), source_git=er.git_state())
    record['preparation'] = p
    record['reference_protocol'] = dict(source_config=str(control.relative_to(ROOT)/'config.json'),
                                      changes=changes, missing_control_defaults={'ccpo_progress_history_weight':1.0})
    dump(folder/'config.json',record)
    dump(folder/'PREPARED.json',p)
    dump(folder/'config-diff-from-control.json',dict(control=p['control'],changes=changes,
         missing_control_defaults={'ccpo_progress_history_weight':1.0}))
    dump(folder/'prepare-command.json',dict(argv=command,gpu_visibility_for_preparation=''))
    dump(folder/'prepared-source-sha256.json',{s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in SOURCES})
    (folder/'NOTES.md').write_text(notes(p))
    return folder


def notes(p):
    channel = 'H' if p['component']=='history' else 'F'
    formula = f'N_CC({channel})' if p['benchmark']=='alfworld' else f'E + {channel}'
    return f'''# {'M10' if p['benchmark']=='alfworld' else 'M11'} H2 no shrinkage: {p['component']}-only

**Prepared only; not launched or queued.** Qwen2.5-1.5B-Instruct, 150 steps, seed {p['seed']},
two GPUs (`{p['gpus']}` configured, not reserved). Registry key `{p['ablation']}`;
canonical ID `{p['canonical_id']}`.

## Method

History / future / episode actor weights: **{p['history_weight']:g} / {p['future_weight']:g} / {p['episode_weight']:g}**.
Original edge weight 0. All usable context baselines have lambda_k=1; task fallback
and unsupported-row behavior remain unchanged. Context-statistics vectors,
frozen reference features, soft similarity and history-2 prompts are retained.
H2 means the future endpoint is min(t+2,T); history length remains 2 independently.

Let H_t=Y_t-C_t[Y], V_t=C_t[Z], and F_t=z_task(V_min(t+2,T)-V_t).
Y includes the existing invalid-action penalty; Z is the unpenalized discounted
return target. Current/future potentials use their own observation groups with
whole-trajectory exclusion. Terminal potentials are success 10 / failure 0.
The masked policy advantage is **A = mask * ({formula})**. N_CC is the existing
ALFWorld per-task mean/sample-std normalization; WebShop has no additional
contextual normalization. Future F is task-standardized on both benchmarks.
E is the existing trainer episode channel, added after contextual normalization.
Episode rewards still provide return labels even when episode actor weight is 0.

Only enabled contextual channels contribute their support mask to normalization.
The disabled channel remains calculated and logged as a diagnostic; its weighted
and applied advantages are identically zero and cannot affect the PPO gradient.
Future-only removes H from credit, not the historical information used by V.
History-only keeps H2 future diagnostics at weight 0, with no future credit.

Full equations and interpretation: [HISTORY_FUTURE_ABLATIONS.md](../HISTORY_FUTURE_ABLATIONS.md).
Control: `{p['control']}`. [Config diff](config-diff-from-control.json) changes only
one component coefficient and experiment identity (the old implicit history
weight 1 is made explicit). History/current/future raw and applied diagnostics,
actual weights, actor-sum identity and per-step snapshots remain enabled.

## Protocol and reproduction

16 tasks x 8 rollouts, 150 steps, turn ceiling {p['max_steps']}, checkpoint/eval every
5 steps. Other learning, rollout, data and evaluation settings match the control.
WebShop retains its standard 15-turn protocol and 1K catalog.

Prepare these four arms for a new date without starting training:

```bash
python scripts/prepare_component_ablations.py --date YYYYMMDD
```

For a later authorized launch from the project root:

```bash
bash experiments/{p['experiment']}/run.sh
```

[config.json](config.json) records the full config, environment and Hydra command;
[prepare-command.json](prepare-command.json) and [prepared-source-sha256.json](prepared-source-sha256.json)
record generation and source provenance. CPU validation is recorded separately
in [VALIDATION.json](VALIDATION.json); no GPU run is implied.
'''


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--date',default=date.today().strftime('%Y%m%d'))
    args=parser.parse_args()
    try: datetime.strptime(args.date,'%Y%m%d')
    except ValueError: parser.error('Use a YYYYMMDD date')
    cases=[(b,c) for b in CONTROLS for c in ('history','future')]
    for b,c in cases:
        if (ROOT/'experiments'/run_name(b,c,args.date)).exists():
            parser.error(f'Experiment already exists: {run_name(b,c,args.date)}')
    folders=[prepare(b,c,args.date) for b,c in cases]
    print(json.dumps(dict(status='prepared_not_launched',experiments=[str(p.relative_to(ROOT)) for p in folders])))


if __name__=='__main__':main()
