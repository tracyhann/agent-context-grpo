#!/usr/bin/env python3
"""Prepare the rebuttal future-channel ablations (E2, E5) against the H2 no-shrink controls; never launch.

E5a terminal-difference   ALFWorld EP0   discount-aligned, but V_u - V_t on terminal windows
E5b terminal-aligned      ALFWorld EP0   V_u - V_t, but discount-aligned on terminal windows
E2  remaining-turns (P1)  ALFWorld EP0, WebShop EP1   F = z_task(-(T - t))
E2  terminal-proximity (P2) ALFWorld EP0, WebShop EP1 F = z_task(gamma^(T-t) * R)

Each prepared config differs from its control in `exp_id` and `ccpo_progress_mode` only.
Run `bash scripts/sync_patches.sh` first: the source hashes include the live trainer.
"""
import argparse
from datetime import date, datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'ablations'), str(ROOT/'scripts')]
import ablations
import exp_run as er

CONTROLS = {
    'alfworld': 'm10-h2-no-credit-shrinkage-alfworld-1.5b-2gpu-20260919',
    'webshop': 'm11-h2-noshrink-active-episode-webshop-1.5b-2gpu-20260920',
}
ARMS = {
    'terminal-difference': dict(mode='terminal_difference', benchmarks=('alfworld',), plan='E5a',
        formula='F_t = z_task(d_t * V_u - V_t),  d_t = 1 if u = T else gamma^h'),
    'terminal-aligned': dict(mode='terminal_aligned', benchmarks=('alfworld',), plan='E5b',
        formula='F_t = z_task(d_t * V_u - V_t),  d_t = gamma^h if u = T else 1'),
    'remaining-turns': dict(mode='remaining_turns', benchmarks=('alfworld', 'webshop'), plan='E2 P1',
        formula='F_t = z_task(-(T - t))'),
    'terminal-proximity': dict(mode='terminal_proximity', benchmarks=('alfworld', 'webshop'), plan='E2 P2',
        formula='F_t = z_task(gamma^(T-t) * R),  R = 10 on success, else 0'),
}
IMPLIED = {'ccpo_loo': 1, 'ccpo_progress_history_weight': 1.0,
           'ccpo_progress_mode': 'difference'}
SOURCES = ['ccpo/future_progress.py', 'ccpo/core_ccpo.py', 'ccpo/arms.py',
           'ablations/ablations.py', 'scripts/exp_run.py',
           'scripts/prepare_rebuttal_progress_ablations_ccpo.py',
           'tests/test_rebuttal_progress_modes_ccpo.py',
           'patches/verl-agent/verl/trainer/ppo/ray_trainer.py',
           'verl-agent/verl/trainer/ppo/ray_trainer.py']


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True)+'\n')


def key_for(arm, benchmark):
    suffix = '-active-episode' if benchmark == 'webshop' else ''
    return 'future-progress-h2-no-credit-shrinkage-'+arm+suffix


def run_name(arm, benchmark, stamp):
    prefix = 'm10' if benchmark == 'alfworld' else 'm11'
    ep = int(benchmark == 'webshop')
    return f'{prefix}-h2-noshrink-{arm}-ep{ep}-{benchmark}-1.5b-2gpu-{stamp}'


def prepare(arm, benchmark, stamp):
    spec = ARMS[arm]
    if benchmark not in spec['benchmarks']:
        raise ValueError(f'{arm} is defined for {", ".join(spec["benchmarks"])} only')
    folder = ROOT/'experiments'/run_name(arm, benchmark, stamp)
    if folder.exists():
        raise FileExistsError(f'Refusing to overwrite existing experiment: {folder}')
    canonical, method = ablations.build(key_for(arm, benchmark), benchmark)
    control = ROOT/'experiments'/CONTROLS[benchmark]
    old = json.loads((control/'config.json').read_text())['config']
    missing = {k:v for k,v in IMPLIED.items() if k not in old}; old.update(missing)
    cfg = {k:v for k,v in old.items() if k in er.DEFAULTS}; cfg.update(method)
    argv = [sys.executable, str(ROOT/'scripts/exp_run.py'), '--name', folder.name.rsplit('-',1)[0],
            '--date', stamp, '--dry-run', *ablations.arms.flags(cfg)]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
               MKL_NUM_THREADS='1', HF_HOME=str(ROOT/'hf'), ALFWORLD_DATA=str(ROOT/'alfworld_data'),
               ACG_DATA_DIR=str(ROOT/'envdata/verl_data'))
    result = subprocess.run(argv, cwd=ROOT, env=env, capture_output=True, text=True)
    if result.returncode:raise RuntimeError(result.stdout+result.stderr)
    (folder/'prepare.log').write_text(result.stdout+result.stderr)
    record = json.loads((folder/'config.json').read_text()); new = record['config']
    if set(new) != set(old):raise ValueError(f'Unexpected config keys: {set(new)^set(old)}')
    changes = {k:dict(control=old[k],prepared=new[k]) for k in old if old[k]!=new[k]}
    if set(changes) != {'exp_id','ccpo_progress_mode'}:raise ValueError(f'Unexpected control delta: {changes}')
    for k,v in method.items():
        if new[k]!=v:raise ValueError(f'Method mismatch: {k}')
    assert new['ccpo_progress_mode']==spec['mode']
    assert new['ccpo_lk_fix']==new['ccpo_lam_fix']==1
    assert new['ccpo_progress_weight']==new['ccpo_progress_history_weight']==1
    assert new['ccpo_ep_w']==int(benchmark=='webshop') and new['ccpo_edge_w']==0
    assert new['history_length']==new['ccpo_progress_horizon']==2
    assert new['ccpo_gate']=='hard' and new['ccpo_wmode']=='soft' and new['ccpo_loo']==1
    assert new['model']=='Qwen/Qwen2.5-1.5B-Instruct' and new['total_epochs']==150
    assert record['hydra_overrides']==er.build_command(new,str(folder))[3:]
    subprocess.run(['bash','-n',str(folder/'run.sh')],check=True)
    status = dict(status='prepared_not_launched', experiment=folder.name, benchmark=benchmark,
        ablation=key_for(arm, benchmark), canonical_id=canonical, rebuttal_plan=spec['plan'],
        backbone='1.5b', steps=150, seed=new['seed'], future_mode=spec['mode'], formula=spec['formula'],
        history_length=2, future_horizon=2, gamma=new['gamma'],
        history_weight=1., future_weight=1., episode_weight=new['ccpo_ep_w'], original_edge_weight=0.,
        lambda_u_fixed=1., lambda_k_fixed=1., loo=True, grouping='task_observation_matched',
        gpu_count=2, gpus=new['gpus'], max_steps=new['max_steps'],
        control=str(control.relative_to(ROOT)), launched=False, queued=False, gpu_preflight_performed=False,
        prepared_at=datetime.now(timezone.utc).isoformat(), source_git=er.git_state())
    record['preparation']=status
    record['reference_protocol']=dict(source_config=str(control.relative_to(ROOT)/'config.json'),
                                     changes=changes, missing_control_defaults=missing)
    dump(folder/'config.json',record); dump(folder/'PREPARED.json',status)
    dump(folder/'config-diff-from-control.json',dict(control=status['control'],changes=changes,missing_control_defaults=missing))
    dump(folder/'prepare-command.json',dict(argv=argv,cwd=str(ROOT),gpu_visibility_for_preparation=''))
    dump(folder/'prepared-source-sha256.json',{s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in SOURCES})
    (folder/'NOTES.md').write_text(notes(status))
    return folder


def notes(p):
    actor = 'mask * N_task(H + F)' if p['benchmark']=='alfworld' else 'mask * (E + H + F)'
    return f'''# Rebuttal {p['rebuttal_plan']}: future channel `{p['future_mode']}`, {p['benchmark']}, EP{int(p['episode_weight'])}

**Prepared only; not launched or queued.** Qwen2.5-1.5B-Instruct, 150 steps,
seed {p['seed']}, two GPUs (`{p['gpus']}` configured, not reserved).
Registry: `{p['ablation']}`. Matched control: `{p['control']}`.

Only `ccpo_progress_mode: difference -> {p['future_mode']}` changes the method;
[the full diff](config-diff-from-control.json) also records experiment identity
and previously implicit defaults.

```text
Z_s = gamma^(T-s) * R
V_s = C_s[Z]                    # same contextual estimator and peer weights
H_t = Y_t - C_t[Y]              # unchanged historical residual
u = min(t+2, T); h = u-t        # u = T marks a terminal window (last two turns)
{p['formula']}
A_actor = {actor}
```

Gamma is {p['gamma']}. Terminal potentials stay success 10 / other endings 0.
Eligible rows, task-wise standardization (sample std, epsilon 1e-6), history
credit, peer groups and the training protocol match the control. The per-turn
dumps still log `undiscounted_raw`, `discount_aligned_raw` and `time_passage_raw`
next to the selected `raw_progress`, so both reference forms can be recomputed.

Launch from the repository root, after selecting available GPUs:
`bash experiments/{p['experiment']}/run.sh`.
'''


def main():
    p=argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--arm',choices=[*ARMS,'all'],default='all')
    p.add_argument('--benchmark',choices=['alfworld','webshop','both'],default='both')
    p.add_argument('--date',default=date.today().strftime('%Y%m%d'));a=p.parse_args()
    arms=list(ARMS) if a.arm=='all' else [a.arm]
    for arm in arms:
        for benchmark in ARMS[arm]['benchmarks']:
            if a.benchmark in ('both', benchmark):print(prepare(arm,benchmark,a.date))

if __name__=='__main__':main()
