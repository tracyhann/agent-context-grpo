#!/usr/bin/env python3
"""Prepare matched ALFWorld EP0 and WebShop EP1 uniform-peer arms; never launch."""
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
IMPLIED = {'ccpo_loo': 1, 'ccpo_progress_history_weight': 1.0}
SOURCES = ['ccpo/core_ccpo.py', 'ccpo/future_progress.py', 'ccpo/arms.py',
           'ablations/ablations.py', 'scripts/exp_run.py',
           'scripts/prepare_uniform_peer_ablations.py', 'tests/test_uniform_peer_ablations.py',
           'patches/verl-agent/verl/trainer/ppo/ray_trainer.py',
           'verl-agent/verl/trainer/ppo/ray_trainer.py', 'scripts/plot_metrics.py']


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True)+'\n')


def key_for(benchmark):
    episode = '-active-episode' if benchmark == 'webshop' else ''
    return f'future-progress-h2-no-credit-shrinkage{episode}-uniform-peers'


def run_name(benchmark, stamp):
    prefix = 'm11-h2-noshrink-active-episode' if benchmark == 'webshop' else 'm10-h2-noshrink'
    return f'{prefix}-uniform-peers-{benchmark}-1.5b-2gpu-{stamp}'


def prepare(benchmark, stamp):
    canonical, method = ablations.build(key_for(benchmark), benchmark)
    control = ROOT/'experiments'/CONTROLS[benchmark]
    old = json.loads((control/'config.json').read_text())['config']
    missing = {k: v for k, v in IMPLIED.items() if k not in old}
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
    if set(changes) != {'exp_id', 'ccpo_wmode'}:
        raise ValueError(f'Unexpected control delta: {changes}')
    for k, v in method.items():
        if new[k] != v:
            raise ValueError(f'Method mismatch: {k}')
    assert new['ccpo_wmode'] == 'uniform' and new['ccpo_loo'] == 1
    assert new['ccpo_ep_w'] == int(benchmark == 'webshop') and new['ccpo_edge_w'] == 0
    assert new['ccpo_lk_fix'] == new['ccpo_lam_fix'] == 1
    assert new['ccpo_progress_weight'] == new['ccpo_progress_history_weight'] == 1
    assert new['history_length'] == new['ccpo_progress_horizon'] == 2
    assert new['total_epochs'] == 150 and new['model'] == 'Qwen/Qwen2.5-1.5B-Instruct'
    assert record['hydra_overrides'] == er.build_command(new, str(folder))[3:]
    subprocess.run(['bash', '-n', str(folder/'run.sh')], check=True)
    status = dict(status='prepared_not_launched', experiment=folder.name, benchmark=benchmark,
        ablation=key_for(benchmark), canonical_id=canonical, backbone='1.5b', steps=150,
        seed=new['seed'], weighting='uniform', weighting_unit='eligible_peer_occurrence',
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
    formula = 'A = mask * N_task(H + F)' if p['benchmark'] == 'alfworld' else 'A = mask * (H + F + A_EP)'
    return f'''# {p['benchmark'].upper()} H2 no shrinkage: uniform peer weighting

**Prepared only; not launched or queued.** Qwen2.5-1.5B-Instruct, 150 training
steps, seed {p['seed']}, two GPUs (`{p['gpus']}` configured, not reserved).
Registry key `{p['ablation']}`; canonical ID `{p['canonical_id']}`.

## Change and estimator

Change only `ccpo_wmode: soft -> uniform` from `{p['control']}`.
[Full config diff](config-diff-from-control.json) also records experiment identity;
legacy implicit history weight and whole-trajectory LOO are made explicit.

For each query t, retain all other-trajectory occurrences in its exact task and
observation group. If none exist, use other-trajectory occurrences from the same
task. Every eligible occurrence has weight one: C_t[q] = mean(q_j, j in P_t).
Repeated real visits contribute separately; this is not a mean of trajectory
means. Padding copies are deduplicated before estimation. Whole-query-trajectory
exclusion remains on. No peers means the existing unsupported-credit behavior.
The rule applies to history, current and future potential, including fallback.
Usable baselines have full strength, lambda_u=lambda_k=1.

H_t = Y_t - C_t[Y], Z_t = gamma^(T-t) * R_episode, V_t = C_t[Z],
F_t = z_task(V_min(t+2,T) - V_t), gamma=0.95. Each endpoint uses its own
observation group. Terminal potential is success 10 / failure 0. Y retains the
existing local invalid-action penalty; Z excludes it. H/F/EP actor weights:
**1 / 1 / {p['episode_weight']:g}**. Original edge weight is zero.

**{formula}**. ALFWorld uses the existing per-task contextual mean/sample-std
normalization. WebShop uses no final contextual normalization and adds the
mean-centered episode advantage at unit weight. Episode moments follow the
existing turn-row weighting, including the local invalid-action penalty.

This is distinct from future-only: both H and F still affect the PPO gradient.
Future-only sets H's actor weight to zero and keeps similarity-weighted potential
readouts. Here frozen hidden/context features remain computed for diagnostics,
but under exact grouping they cannot influence uniform weights or credit.
The actor still receives history-2 prompts. This also differs from no-context
vector, which keeps hidden-state similarity weighting.

## Protocol, logs and reproduction

16 tasks x 8 rollouts, 150 steps, {p['max_steps']}-turn cap, save/evaluate every
5 steps. Prompt/response limits and all other optimization/environment settings
match the control: ALFWorld 2048/512; WebShop 4096/512 with its 1K catalog.
History/current/future baselines, support, lambda, H/F/EP raw/applied components,
feature diagnostics and actor-sum identity remain logged. The explicit metric
`ccpo/progress_uniform_weighting=1` identifies this weighting rule.

```bash
python scripts/prepare_uniform_peer_ablations.py --date YYYYMMDD
# Later launch from the project root:
bash experiments/{p['experiment']}/run.sh
```

[Shared equations and comparison](../UNIFORM_PEER_ABLATIONS.md).
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
