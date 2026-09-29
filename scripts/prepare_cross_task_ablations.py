#!/usr/bin/env python3
"""Prepare ALFWorld and WebShop 1.5B H2 no-shrink EP0 cross-task arms; never launch."""
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
    'webshop': 'm11-h2-no-credit-shrinkage-webshop-1.5b-2gpu-20260919',
}
IMPLIED = {'ccpo_loo': 1, 'ccpo_progress_history_weight': 1.0}
SOURCES = ['ccpo/cross_task.py', 'ccpo/core_ccpo.py', 'ccpo/future_progress.py', 'ccpo/arms.py',
           'ablations/ablations.py', 'scripts/exp_run.py',
           'scripts/prepare_cross_task_ablations.py', 'tests/test_cross_task_ablations.py',
           'patches/verl-agent/verl/trainer/ppo/ray_trainer.py',
           'verl-agent/verl/trainer/ppo/ray_trainer.py', 'scripts/plot_metrics.py']


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True)+'\n')


def key_for(benchmark):
    return 'future-progress-h2-no-credit-shrinkage-cross-task'


def run_name(benchmark, stamp):
    prefix = 'm10' if benchmark == 'alfworld' else 'm11'
    return f'{prefix}-h2-noshrink-cross-task-{benchmark}-1.5b-2gpu-{stamp}'


def prepare(benchmark, stamp):
    canonical, method = ablations.build(key_for(benchmark), benchmark)
    control = ROOT/'experiments'/CONTROLS[benchmark]
    old = json.loads((control/'config.json').read_text())['config']
    missing = {k: v for k, v in {**IMPLIED, 'ccpo_progress_mode': 'difference'}.items() if k not in old}
    old.update(missing)
    folder = ROOT/'experiments'/run_name(benchmark, stamp)
    if folder.exists():
        raise FileExistsError(f'Refusing to overwrite existing experiment: {folder}')
    cfg = {k: v for k, v in old.items() if k in er.DEFAULTS}
    cfg.update(method)
    argv = [sys.executable, str(ROOT/'scripts/exp_run.py'), '--name', folder.name.rsplit('-', 1)[0],
            '--date', stamp, '--dry-run', *ablations.arms.flags(cfg)]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
               MKL_NUM_THREADS='1', HF_HOME=str(ROOT/'hf'), ALFWORLD_DATA=str(ROOT/'alfworld_data'),
               ACG_DATA_DIR=str(ROOT/'envdata/verl_data'))
    result = subprocess.run(argv, cwd=ROOT, env=env, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stdout+result.stderr)
    (folder/'prepare.log').write_text(result.stdout+result.stderr)
    record = json.loads((folder/'config.json').read_text()); new = record['config']
    if set(new) != set(old):
        raise ValueError(f'Unexpected config keys: {set(new)^set(old)}')
    changes = {k: dict(control=old[k], prepared=new[k]) for k in old if new[k] != old[k]}
    if set(changes) != {'exp_id', 'ccpo_gate'}:
        raise ValueError(f'Unexpected control delta: {changes}')
    for k, v in method.items():
        if new[k] != v:
            raise ValueError(f'Method mismatch: {k}')
    assert new['ccpo_gate'] == 'cross_task' and new['ccpo_wmode'] == 'soft' and new['ccpo_loo'] == 1
    assert new['ccpo_ep_w'] == 0 and new['ccpo_edge_w'] == 0
    assert new['ccpo_lk_fix'] == new['ccpo_lam_fix'] == 1
    assert new['ccpo_progress_weight'] == new['ccpo_progress_history_weight'] == 1
    assert new['history_length'] == new['ccpo_progress_horizon'] == 2
    assert new['total_epochs'] == 150 and new['model'] == 'Qwen/Qwen2.5-1.5B-Instruct'
    assert record['hydra_overrides'] == er.build_command(new, str(folder))[3:]
    subprocess.run(['bash', '-n', str(folder/'run.sh')], check=True)
    status = dict(status='prepared_not_launched', experiment=folder.name, benchmark=benchmark,
        ablation=key_for(benchmark), canonical_id=canonical, backbone='1.5b', steps=150,
        seed=new['seed'], grouping='cross_task', weighting='soft', weighting_unit='eligible_peer_occurrence',
        peer_pool='all_other_trajectory_rows_in_canonical_rollout_batch', task_prior=False, task_backoff=False,
        history_length=2, future_horizon=2, history_weight=1., future_weight=1.,
        episode_weight=new['ccpo_ep_w'], original_edge_weight=0., lambda_u_fixed=1., lambda_k_fixed=1.,
        loo=True, gpu_count=2, gpus=new['gpus'], max_steps=new['max_steps'],
        control=str(control.relative_to(ROOT)), launched=False, queued=False,
        gpu_preflight_performed=False, prepared_at=datetime.now(timezone.utc).isoformat(),
        source_git=er.git_state())
    record['preparation'] = status
    record['reference_protocol'] = dict(source_config=str(control.relative_to(ROOT)/'config.json'),
                                       changes=changes, missing_control_defaults=missing)
    dump(folder/'config.json', record)
    dump(folder/'PREPARED.json', status)
    dump(folder/'config-diff-from-control.json', dict(control=status['control'], changes=changes,
                                                     missing_control_defaults=missing))
    dump(folder/'prepare-command.json', dict(argv=argv, cwd=str(ROOT), gpu_visibility_for_preparation=''))
    dump(folder/'prepared-source-sha256.json', {s: hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in SOURCES})
    (folder/'NOTES.md').write_text(notes(status))
    return folder


def notes(p):
    formula = 'A = mask * N_task(H + F)' if p['benchmark'] == 'alfworld' else 'A = mask * (H + F)'
    return f'''# {p['benchmark'].upper()} H2 no shrinkage, EP0: cross-task soft grouping

**Prepared only; not launched or queued.** Qwen2.5-1.5B-Instruct, 150 training
steps, seed {p['seed']}, two GPUs (`{p['gpus']}` configured, not reserved).
Registry key `{p['ablation']}`; canonical ID `{p['canonical_id']}`.

## Change from control

Change only `ccpo_gate: hard -> cross_task` from `{p['control']}`.
[Full config diff](config-diff-from-control.json) also records experiment identity;
legacy implicit history weight and whole-trajectory LOO are made explicit.
Both benchmarks retain EP=0, lambda_u=lambda_k=1, history=2, future horizon=2,
binary-return targets, and the original soft kernel with tau scale 0.15.

For query i, P_i contains every occurrence in this canonical rollout batch
except occurrences belonging to i's own trajectory. Neither task UID nor
observation equality restricts P_i. ALFWorld and WebShop have separate runs and
separate batches; cross-task does not mean pooling the two environments.
Padding copies are deduplicated first; real repeated visits retain their mass.

```text
phi = existing batch-whitened frozen hidden state + trajectory-context features
w_ij = exp(-||phi_i - phi_j|| / tau), j in P_i
C_i[q] = sum(w_ij * q_j) / sum(w_ij)
tau = 0.15 * median distance over all distinct row pairs in this batch
H_i = Y_i - C_i[Y]
Z_i = gamma^(T_i-t_i) * R_i; V_i = C_i[Z]
F_i = z_task(V_min(t_i+2,T_i) - V_i)
```

The median includes same-trajectory pairs, following the existing bandwidth
rule; the readout always excludes them. A degenerate median uses 1 before the
0.15 multiplier. The normalized kernel uses a numerically stable common shift.
No hard task/observation gates, top-k cutoff, task prior, or task fallback apply.
`ccpo_prior_kappa` and `ccpo_backoff_task` remain recorded control values but are
inactive under `cross_task`. With no other trajectory, contextual credit is
unsupported and zero. Lambda_u=lambda_k=1 wherever supported.

History Y retains the existing invalid-action penalty; potential labels Z do
not. Terminal potentials remain success 10 / failure 0, gamma=0.95. The future
endpoint uses its own context to weight the same batch-wide candidate pool.
Task IDs still define task-wise advantage normalization, not peer eligibility.

**{formula}**. H/F/EP weights are **1 / 1 / 0** on BOTH benchmarks, original
edge weight zero. ALFWorld retains its combined per-task mean/sample-std
normalization. WebShop retains mean_norm, without final contextual normalization.
The actor/reference prompts retain history length 2 and the task goal.

## Protocol and diagnostics

16 tasks x 8 rollouts, 150 steps, {p['max_steps']}-turn cap, save/evaluate every
5 steps. Prompt/response budgets, optimizer, reward and environment settings
match the control: ALFWorld 2048/512; WebShop 4096/512 with its 1K catalog.
The exact distance matrix uses about 313 MiB at the 6,400-row ALFWorld ceiling;
median selection temporarily adds about 156 MiB. No additional model forward.

`ccpo/progress_cross_task_grouping=1` identifies the variant. History/current/
future snapshots include cross-task normalized weight mass, peer task count,
peer rows, effective trajectory support and kernel bandwidth. Same-trajectory
mass is zero; cross-task mass can be zero for a one-task batch. Terminal future
rows have no neighborhood statistics. Existing H/F/EP actor-sum logging remains.

```bash
python scripts/prepare_cross_task_ablations.py --date YYYYMMDD
# Launch from the project root when ready:
bash experiments/{p['experiment']}/run.sh
```

[Shared definition](../CROSS_TASK_SOFT_GROUPING.md).
[config.json](config.json), [PREPARED.json](PREPARED.json),
[prepare-command.json](prepare-command.json), [source hashes](prepared-source-sha256.json)
and [VALIDATION.json](VALIDATION.json) record preparation and CPU validation.
No training result or GPU smoke test is claimed.
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--date', default=date.today().strftime('%Y%m%d'))
    args = parser.parse_args()
    try:
        datetime.strptime(args.date, '%Y%m%d')
    except ValueError:
        parser.error('Use a YYYYMMDD date')
    for benchmark in CONTROLS:
        if (ROOT/'experiments'/run_name(benchmark, args.date)).exists():
            parser.error(f'Experiment exists: {run_name(benchmark, args.date)}')
    folders = [prepare(b, args.date) for b in CONTROLS]
    print(json.dumps(dict(status='prepared_not_launched', experiments=[str(p.relative_to(ROOT)) for p in folders])))


if __name__ == '__main__':
    main()
