#!/usr/bin/env python3
"""Prepare matched M10/M11 1.5B H2 no-shrink EP0 no-LOO arms; never launch."""
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

KEY = 'future-progress-h2-no-credit-shrinkage-no-loo'
CONTROLS = {
    'alfworld': 'm10-h2-no-credit-shrinkage-alfworld-1.5b-2gpu-20260919',
    'webshop': 'm11-h2-no-credit-shrinkage-webshop-1.5b-2gpu-20260919',
}
IMPLIED = {'ccpo_loo': 1, 'ccpo_progress_history_weight': 1.0}
SOURCES = ['ccpo/core_ccpo.py', 'ccpo/future_progress.py', 'ccpo/outlook.py',
           'scripts/exp_run.py', 'ablations/ablations.py', 'ccpo/arms.py',
           'scripts/prepare_no_loo_ablations.py', 'scripts/plot_metrics.py',
           'patches/verl-agent/verl/trainer/ppo/ray_trainer.py',
           'verl-agent/verl/trainer/ppo/ray_trainer.py', 'tests/test_future_progress_no_loo.py']


def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2, sort_keys=True)+'\n')


def run_name(benchmark, stamp):
    prefix = 'm10' if benchmark == 'alfworld' else 'm11'
    return f'{prefix}-h2-noshrink-noloo-{benchmark}-1.5b-2gpu-{stamp}'


def prepare(benchmark, stamp):
    canonical, method = ablations.build(KEY, benchmark)
    control = ROOT/'experiments'/CONTROLS[benchmark]
    old = json.loads((control/'config.json').read_text())['config']
    missing = {k: v for k, v in IMPLIED.items() if k not in old}
    old.update(missing)
    name = run_name(benchmark, stamp)
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
    if set(old) != set(new):
        raise ValueError(f'Unexpected config keys: {set(old) ^ set(new)}')
    changes = {k: dict(control=old[k], prepared=new[k]) for k in old if old[k] != new[k]}
    if set(changes) != {'exp_id', 'ccpo_loo'}:
        raise ValueError(f'Unexpected change from matched EP0 control: {changes}')
    for k, v in method.items():
        if new[k] != v: raise ValueError(f'Method mismatch: {k}')
    assert new['ccpo_loo'] == new['ccpo_ep_w'] == new['ccpo_edge_w'] == 0
    assert new['ccpo_lk_fix'] == new['ccpo_lam_fix'] == 1
    assert new['ccpo_progress_weight'] == new['ccpo_progress_history_weight'] == 1
    assert new['total_epochs'] == 150 and new['history_length'] == new['ccpo_progress_horizon'] == 2
    assert record['env']['ACG_CCPO_LOO'] == '0'
    assert record['hydra_overrides'] == er.build_command(new, str(folder))[3:]
    subprocess.run(['bash', '-n', str(folder/'run.sh')], check=True)
    p = dict(status='prepared_not_launched', experiment=name, ablation=KEY, canonical_id=canonical,
             benchmark=benchmark, backbone='1.5b', steps=150, seed=new['seed'],
             history_length=2, future_horizon=2, history_weight=1., future_weight=1.,
             episode_weight=0., lambda_k_fixed=1., original_edge_weight=0.,
             loo=False, query_included=True, same_trajectory_matches=True,
             gpus=new['gpus'], gpu_count=2, max_steps=new['max_steps'],
             control=str(control.relative_to(ROOT)), launched=False, queued=False,
             gpu_preflight_performed=False, prepared_at=datetime.now(timezone.utc).isoformat(),
             source_git=er.git_state())
    record['preparation'] = p
    record['reference_protocol'] = dict(source_config=str(control.relative_to(ROOT)/'config.json'),
                                      changes=changes, missing_control_defaults=missing)
    dump(folder/'config.json', record)
    dump(folder/'PREPARED.json', p)
    dump(folder/'config-diff-from-control.json', dict(control=p['control'], changes=changes,
                                                    missing_control_defaults=missing))
    dump(folder/'prepare-command.json', dict(argv=command, gpu_visibility_for_preparation=''))
    dump(folder/'prepared-source-sha256.json', {s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in SOURCES})
    (folder/'NOTES.md').write_text(notes(p))
    return folder


def notes(p):
    return f'''# {'M10' if p['benchmark']=='alfworld' else 'M11'} H2 no shrinkage, no episode: no LOO

**Prepared only; not launched or queued.** Qwen2.5-1.5B-Instruct, 150 training
steps, seed {p['seed']}, 16 tasks x 8 rollouts, two GPUs (`{p['gpus']}` configured,
not reserved). Turn ceiling {p['max_steps']}; evaluation/checkpoints every 5 steps.

Registry key `{p['ablation']}`; canonical ID `{p['canonical_id']}`.
Control: `{p['control']}`. [Full config diff](config-diff-from-control.json) changes
only experiment identity and `ccpo_loo: 1 -> 0`; the previously implicit history
weight 1 and LOO default 1 are resolved explicitly for comparison.

## Exact ablation

**Full self-inclusion:** matching same-trajectory turns AND the query occurrence
itself are allowed in the exact (task, observation) peer group. This applies to
the history baseline, current potential and future endpoint potential. Different
observations do not become exact peers; DP padding copies are still deduplicated.
Current and future potentials retain their respective observation groups.

All usable readouts use full kernel strength: lambda_u=lambda_k=1. History/future
weights are 1/1; episode/original-edge weights are 0/0. History length 2 and future
horizon 2, hidden+context-statistics features, soft kernel tau scale 0.15, and
three-direction whitening are retained. Episode rewards still label the returns;
the separate episode advantage has zero applied PPO weight on BOTH benchmarks.

For q in {{Y,Z}}, let C_t^all[q] be the context-weighted mean over all exact
matches, including self. Then H_t=Y_t-C_t^all[Y], V_t=C_t^all[Z],
F_t=z_task(V_min(t+2,T)-V_t), and A=mask*N_CC(H+F).
Y is the historical penalized return; Z=gamma^(T-t)*R_episode is the unpenalized
potential label, gamma=0.95. N_CC is task mean/std normalization on ALFWorld and
identity on WebShop. Terminal V is success 10 / failure 0.

A single matching trajectory is sufficient, including a singleton query. A
singleton has H=0 and V=its own Z; F may remain nonzero from discounting or a
change at the endpoint. J counts distinct represented trajectories, now including
the query trajectory, while occurrence weights retain the existing aggregation.
The estimator stays detached, but allowing its own outcome into the baseline
removes the old trajectory-exclusion protection; this is a statistical ablation.

Equations, exact support semantics and diagnostics: [NO_LOO_ABLATIONS.md](../NO_LOO_ABLATIONS.md).
Self-occurrence and own-trajectory kernel mass are logged for history/current/
future, alongside peer row counts, J, n_eff, actual shrinkage coefficients,
component advantages and actor identity checks. Terminal future peer metrics are
undefined and excluded from means. Future-progress plots include self-mass panels.

## Reproduction

Prepare both benchmark arms for a fresh date:

```bash
python scripts/prepare_no_loo_ablations.py --date YYYYMMDD
```

For a later authorized launch from the project root:

```bash
bash experiments/{p['experiment']}/run.sh
```

[config.json](config.json), [run.sh](run.sh), [prepare-command.json](prepare-command.json),
and [prepared-source-sha256.json](prepared-source-sha256.json) capture the resolved
launch recipe and sources. [VALIDATION.json](VALIDATION.json) records CPU checks.
WebShop retains the control's 15-turn, 1K-catalog benchmark protocol and scorer.
No training result or GPU validation is claimed by preparation.
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--date', default=date.today().strftime('%Y%m%d'))
    args = parser.parse_args()
    try: datetime.strptime(args.date, '%Y%m%d')
    except ValueError: parser.error('Use a YYYYMMDD date')
    for benchmark in CONTROLS:
        if (ROOT/'experiments'/run_name(benchmark,args.date)).exists():
            parser.error(f'Experiment already exists: {run_name(benchmark,args.date)}')
    folders = [prepare(b,args.date) for b in CONTROLS]
    print(json.dumps(dict(status='prepared_not_launched', experiments=[str(p.relative_to(ROOT)) for p in folders])))


if __name__ == '__main__': main()
