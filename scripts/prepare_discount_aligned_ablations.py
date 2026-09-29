#!/usr/bin/env python3
"""Prepare matched 1.5B H2 discount-aligned future residual ablations; never launch."""
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
IMPLIED = {'ccpo_loo': 1, 'ccpo_progress_history_weight': 1.0,
           'ccpo_progress_mode': 'difference'}
SOURCES = ['ccpo/future_progress.py', 'ccpo/core_ccpo.py', 'ccpo/arms.py',
           'ablations/ablations.py', 'scripts/exp_run.py',
           'scripts/prepare_discount_aligned_ablations.py', 'tests/test_discount_aligned_future.py',
           'patches/verl-agent/verl/trainer/ppo/ray_trainer.py',
           'verl-agent/verl/trainer/ppo/ray_trainer.py']


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True)+'\n')


def key_for(benchmark):
    suffix = '-active-episode' if benchmark == 'webshop' else ''
    return 'future-progress-h2-no-credit-shrinkage-discount-aligned'+suffix


def run_name(benchmark, stamp):
    prefix = 'm10' if benchmark == 'alfworld' else 'm11'
    ep = int(benchmark == 'webshop')
    return f'{prefix}-h2-noshrink-discount-aligned-ep{ep}-{benchmark}-1.5b-2gpu-{stamp}'


def prepare(benchmark, stamp):
    folder = ROOT/'experiments'/run_name(benchmark, stamp)
    if folder.exists():
        raise FileExistsError(f'Refusing to overwrite existing experiment: {folder}')
    canonical, method = ablations.build(key_for(benchmark), benchmark)
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
    assert new['ccpo_lk_fix']==new['ccpo_lam_fix']==1
    assert new['ccpo_progress_weight']==new['ccpo_progress_history_weight']==1
    assert new['ccpo_ep_w']==int(benchmark=='webshop') and new['ccpo_edge_w']==0
    assert new['history_length']==new['ccpo_progress_horizon']==2
    assert new['ccpo_gate']=='hard' and new['ccpo_wmode']=='soft' and new['ccpo_loo']==1
    assert new['model']=='Qwen/Qwen2.5-1.5B-Instruct' and new['total_epochs']==150
    assert record['hydra_overrides']==er.build_command(new,str(folder))[3:]
    subprocess.run(['bash','-n',str(folder/'run.sh')],check=True)
    status = dict(status='prepared_not_launched', experiment=folder.name, benchmark=benchmark,
        ablation=key_for(benchmark), canonical_id=canonical, backbone='1.5b', steps=150, seed=new['seed'],
        future_mode='discount_aligned', history_length=2, future_horizon=2,
        discount_exponent='actual_clipped_window=t_plus-t', gamma=new['gamma'],
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
    actor = 'mask * N_task(H + F_TD)' if p['benchmark']=='alfworld' else 'mask * (E + H + F_TD)'
    return f'''# Historical + discount-aligned future residual: {p['benchmark']}, EP{int(p['episode_weight'])}

**Prepared only; not launched or queued.** Qwen2.5-1.5B-Instruct, 150 steps,
seed {p['seed']}, two GPUs (`{p['gpus']}` configured, not reserved).
Registry: `{p['ablation']}`. Matched control: `{p['control']}`.

Only `ccpo_progress_mode: difference -> discount_aligned` changes the method;
[the full diff](config-diff-from-control.json) also records experiment identity
and previously implicit defaults. This arm retains task-observation matching.

```text
Z_s = gamma^(T-s) * R
V_s = C_s[Z]                    # same contextual estimator and peer weights
H_t = Y_t - C_t[Y]              # unchanged historical residual
u = min(t+2, T); h = u-t        # actual elapsed turns, not always the configured 2
F_TD_t = z_task(gamma^h * V_u - V_t)
A_actor = {actor}
```

Gamma is {p['gamma']}. Near the terminal, h can be 1; terminal potentials remain
success 10 / other endings 0. There is no new immediate-reward term. Historical
invalid-action penalties stay in Y, outside the unpenalized value labels Z.
History/future/episode weights are 1/1/{int(p['episode_weight'])}; lambda_u=lambda_k=1.
The frozen reference, hidden+context features, whole-trajectory LOO, soft kernel,
tau, horizon, target, training protocol and task-wise future standardization
remain matched. ALFWorld normalizes H+F together; WebShop retains mean_norm and
adds its episode channel after the step channel, exactly as the EP1 control.

## Reviewer question

The original raw difference decomposes as
`V_u - V_t = (gamma^h*V_u - V_t) + (1-gamma^h)*V_u`.
This experiment removes the second term BEFORE task-wise z-scoring. Subtracting
separately standardized components would define a different experiment.
For discount-consistent potentials on an otherwise redundant trajectory, the
new raw residual is zero (up to floating-point roundoff), whereas the original
difference can remain nonzero after task-wise centering. This does not establish
that any remaining contextual residual uniquely measures semantic or causal
progress; it isolates this particular time-discount contribution.

## Audit outputs and acceptance tests

`outputs/future_progress/step-*.npz` and CSVs include `window_length`,
`endpoint_discount`, `future_credit_mode`, `undiscounted_raw`,
`discount_aligned_raw`, `time_passage_raw`, selected `raw_progress`,
`progress_normalized`, historical/future/episode applied credits and actor credit.
Metrics include `progress_discount_aligned`, `progress_gamma` and
`progress_decomposition_max_error`. The M5 edge correlation remains the original
one-step, undiscounted diagnostic reference.

CPU tests cover cancellation of pure discount-clock changes, gamma=1 parity,
clipped terminal windows, discounted telescoping, padding/LOO/penalty isolation,
actual trainer routing, masked PPO gradients and EP0/EP1 channel composition.
No training results are claimed by this preparation.

Launch from the repository root, after selecting available GPUs:
`bash experiments/{p['experiment']}/run.sh`.
'''


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--benchmark',choices=['alfworld','webshop','both'],default='both')
    p.add_argument('--date',default=date.today().strftime('%Y%m%d'));a=p.parse_args()
    benches=CONTROLS if a.benchmark=='both' else [a.benchmark]
    for benchmark in benches:print(prepare(benchmark,a.date))

if __name__=='__main__':main()
