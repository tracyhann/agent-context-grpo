#!/usr/bin/env python3
"""Prepare the rebuttal task-gate ablation against the H2 no-shrink controls; never launch.

The ablation removes exact-observation matching and keeps task identity:
`ccpo_gate: hard -> global`. Every turn then reads the task bucket that the full
method uses only as its fallback, so the result measures how much the method
depends on observations recurring across rollouts of a task. (The cross-task arms
drop the task gate as well and cannot separate the two.)

    ALFWorld EP0   control m10-h2-no-credit-shrinkage-alfworld-1.5b-2gpu-20260919
    WebShop  EP1   control m11-h2-noshrink-active-episode-webshop-1.5b-2gpu-20260920

Each prepared config differs from its control in `exp_id` and `ccpo_gate` only.
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
KEYS = {
    'alfworld': 'future-progress-h2-no-credit-shrinkage-task-gate',
    'webshop': 'future-progress-h2-no-credit-shrinkage-active-episode-task-gate',
}
# Keys the controls predate; they ran with these runtime defaults.
IMPLIED = {'ccpo_loo': 1, 'ccpo_progress_history_weight': 1.0, 'ccpo_progress_mode': 'difference'}
SOURCES = ['ccpo/future_progress.py', 'ccpo/core_ccpo.py', 'ccpo/arms.py', 'ablations/ablations.py',
           'scripts/exp_run.py', 'scripts/prepare_rebuttal_task_gate_ccpo.py',
           'tests/test_rebuttal_task_gate_ccpo.py',
           'patches/verl-agent/verl/trainer/ppo/ray_trainer.py']
LIVE_TRAINER = 'verl-agent/verl/trainer/ppo/ray_trainer.py'


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def run_name(benchmark, stamp):
    prefix = 'm10' if benchmark == 'alfworld' else 'm11'
    return f'{prefix}-h2-noshrink-task-gate-ep{int(benchmark == "webshop")}-{benchmark}-1.5b-2gpu-{stamp}'


def trainer_synced():
    live, patch = ROOT/LIVE_TRAINER, ROOT/'patches'/LIVE_TRAINER
    return live.is_file() and live.read_bytes() == patch.read_bytes()


def prepare(benchmark, stamp, skip_runtime_check=False):
    synced = trainer_synced()
    if not synced and not skip_runtime_check:
        raise RuntimeError(f'{LIVE_TRAINER} is missing or differs from patches/. '
                           'Run `bash scripts/sync_patches.sh`.')
    folder = ROOT/'experiments'/run_name(benchmark, stamp)
    if folder.exists():
        raise FileExistsError(f'Refusing to overwrite existing experiment: {folder}')
    canonical, method = ablations.build(KEYS[benchmark], benchmark)
    control = ROOT/'experiments'/CONTROLS[benchmark]
    old = json.loads((control/'config.json').read_text())['config']
    missing = {k: v for k, v in IMPLIED.items() if k not in old}; old.update(missing)
    cfg = {k: v for k, v in old.items() if k in er.DEFAULTS}; cfg.update(method)
    argv = [sys.executable, str(ROOT/'scripts/exp_run.py'), '--name', folder.name.rsplit('-', 1)[0],
            '--date', stamp, '--dry-run', *ablations.arms.flags(cfg)]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
               MKL_NUM_THREADS='1', HF_HOME=str(ROOT/'hf'), ALFWORLD_DATA=str(ROOT/'alfworld_data'),
               ACG_DATA_DIR=str(ROOT/'envdata/verl_data'))
    result = subprocess.run(argv, cwd=ROOT, env=env, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)
    (folder/'prepare.log').write_text(result.stdout + result.stderr)
    record = json.loads((folder/'config.json').read_text()); new = record['config']
    if set(new) != set(old):
        raise ValueError(f'Unexpected config keys: {set(new) ^ set(old)}')
    changes = {k: dict(control=old[k], prepared=new[k]) for k in old if old[k] != new[k]}
    if set(changes) != {'exp_id', 'ccpo_gate'}:
        raise ValueError(f'Unexpected control delta: {changes}')
    for k, v in method.items():
        if new[k] != v:
            raise ValueError(f'Method mismatch: {k}')
    assert new['ccpo_gate'] == 'global' and new['ccpo_wmode'] == 'soft' and new['ccpo_loo'] == 1
    assert new['ccpo_lk_fix'] == new['ccpo_lam_fix'] == 1 and new['ccpo_sim'] == new['ccpo_sim_backoff'] == 0
    assert new['ccpo_progress_weight'] == new['ccpo_progress_history_weight'] == 1
    assert new['ccpo_progress_mode'] == 'difference'
    assert new['ccpo_ep_w'] == int(benchmark == 'webshop') and new['ccpo_edge_w'] == 0
    assert new['history_length'] == new['ccpo_progress_horizon'] == 2
    assert new['model'] == 'Qwen/Qwen2.5-1.5B-Instruct' and new['total_epochs'] == 150
    assert record['env']['ACG_CCPO_GATE'] == 'global'
    assert record['hydra_overrides'] == er.build_command(new, str(folder))[3:]
    subprocess.run(['bash', '-n', str(folder/'run.sh')], check=True)
    status = dict(status='prepared_not_launched', experiment=folder.name, benchmark=benchmark,
        ablation=KEYS[benchmark], canonical_id=canonical, backbone='1.5b', steps=150, seed=new['seed'],
        grouping='task', observation_matching=False, task_matching=True, task_fallback_level=False,
        weighting='soft', peer_pool='all_other_trajectory_rows_of_the_same_task',
        history_length=2, future_horizon=2, gamma=new['gamma'], history_weight=1., future_weight=1.,
        episode_weight=new['ccpo_ep_w'], original_edge_weight=0., lambda_u_fixed=1., lambda_k_fixed=1.,
        loo=True, gpu_count=2, gpus=new['gpus'], max_steps=new['max_steps'],
        control=str(control.relative_to(ROOT)), runtime_trainer_verified=synced,
        launched=False, queued=False, gpu_preflight_performed=False,
        prepared_at=datetime.now(timezone.utc).isoformat(), source_git=er.git_state())
    record['preparation'] = status
    record['reference_protocol'] = dict(source_config=str(control.relative_to(ROOT)/'config.json'),
                                       changes=changes, missing_control_defaults=missing)
    dump(folder/'config.json', record); dump(folder/'PREPARED.json', status)
    dump(folder/'config-diff-from-control.json',
         dict(control=status['control'], changes=changes, missing_control_defaults=missing))
    dump(folder/'prepare-command.json', dict(argv=argv, cwd=str(ROOT), gpu_visibility_for_preparation=''))
    sources = SOURCES + ([LIVE_TRAINER] if (ROOT/LIVE_TRAINER).is_file() else [])
    dump(folder/'prepared-source-sha256.json',
         {s: hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in sources})
    (folder/'NOTES.md').write_text(notes(status))
    return folder


def notes(p):
    ep = int(p['episode_weight'])
    actor = 'mask * N_task(H + F)' if p['benchmark'] == 'alfworld' else 'mask * (E + H + F)'
    overlay = '' if p['runtime_trainer_verified'] else (
        '\n\n**Overlay.** The live trainer was missing or stale when this was prepared. '
        'Run `bash scripts/sync_patches.sh` before launching.')
    return f'''# Rebuttal task-gate ablation, {p['benchmark']}, EP{ep}

**Prepared only; not launched or queued.** Qwen2.5-1.5B-Instruct, 150 steps,
seed {p['seed']}, two GPUs (`{p['gpus']}` configured, not reserved).
Registry: `{p['ablation']}`. Matched control: `{p['control']}`.

**Question.** How much of the method depends on observations recurring across
rollouts of a task?

Only `ccpo_gate: hard -> global` changes the method; [the full diff](config-diff-from-control.json)
also records experiment identity and previously implicit defaults.

```text
P_i = {{j: task(j) = task(i), trajectory(j) != trajectory(i)}}    # no observation condition
w_ij = exp(-||phi_i - phi_j|| / tau),  tau = 0.15 * median distance in the task bucket
C_i[q] = sum_j w_ij q_j / sum_j w_ij
H_t = Y_t - C_t[Y];  Z_s = gamma^(T-s) * R;  V_s = C_s[Z]
F_t = z_task(V_min(t+2,T) - V_t)
A_actor = {actor}
```

This is the readout the control applies to a turn with no exact-observation peer,
now applied to every turn. Features, kernel, bandwidth rule, whole-trajectory
exclusion, targets, normalization and the training protocol match the control.
The cross-task arm also drops the task condition; this one does not.

**Watch.** `ccpo/progress_task_grouping=1` identifies the variant. The exact and
backoff fractions read zero, since there is a single level. Compare success with
the control and with GRPO and GiGPO at the same step.{overlay}

Launch from the repository root, after selecting available GPUs:
`bash experiments/{p['experiment']}/run.sh`.
'''


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--benchmark', choices=['alfworld', 'webshop', 'both'], default='both')
    p.add_argument('--skip-runtime-check', action='store_true',
                   help='prepare without a synced verl-agent/ checkout; recorded in PREPARED.json')
    p.add_argument('--date', default=date.today().strftime('%Y%m%d'))
    a = p.parse_args()
    for benchmark in CONTROLS:
        if a.benchmark in ('both', benchmark):
            print(prepare(benchmark, a.date, a.skip_runtime_check))


if __name__ == '__main__':
    main()
