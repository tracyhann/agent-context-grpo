#!/usr/bin/env python3
"""Prepare the Search-R1 runs for the rebuttal (a third benchmark); never launch.

All arms run verl-agent's `search` env under GiGPO's published protocol
(examples/gigpo_trainer/run_search.sh), which scripts/exp_run.py applies for
env_name=search:

  ccpo      the main method unchanged: H2, no shrinkage, EP0 (A_CC only), exact
            task/observation peers with the task-bucket fallback
  ccpo-ep1  the same plus the episode advantage (A_CC + A_EP)
  grpo      baseline
  gigpo     baseline with its published similarity grouping at 0.9

The CCPO arms take every estimator setting from the registry entry that defines the
ALFWorld main method and are checked against that run's recorded config, so the
only differences are the benchmark protocol, the backbone and the resources.
Baselines share the same resolved config and differ in `arm` alone.

Run `bash scripts/sync_patches.sh` first: the search reward scale lives in the
env-manager overlay, and a stale overlay would train on the wrong reward.
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

MAIN = 'future-progress-h2-no-credit-shrinkage'
ARMS = {
    'ccpo': dict(arm='ccpo', registry=MAIN, tag='ccpo-h2-noshrink-ep0', episode=0,
                 control='m10-h2-no-credit-shrinkage-alfworld-1.5b-2gpu-20260919'),
    'ccpo-ep1': dict(arm='ccpo', registry=MAIN + '-active-episode', tag='ccpo-h2-noshrink-ep1', episode=1,
                     control='m10-h2-noshrink-active-episode-alfworld-1.5b-2gpu-20260920'),
    'grpo': dict(arm='grpo', registry=MAIN, tag='grpo', episode=None, control=None),
    'gigpo': dict(arm='gigpo', registry=MAIN, tag='gigpo-sim0.9', episode=None, control=None),
}
# GiGPO publishes Search-R1 on 3B and 7B; the paper's other benchmarks use 1.5B and 7B.
MODELS = {'1.5b': 'Qwen/Qwen2.5-1.5B-Instruct', '3b': 'Qwen/Qwen2.5-3B-Instruct',
          '7b': 'Qwen/Qwen2.5-7B-Instruct'}
# Benchmark, budget and logging keys this script may set on top of the registry method.
BENCHMARK_KEYS = {'env_name', 'max_steps', 'pin_steps', 'ccpo_progress_snapshot_every',
                  'model', 'gpus', 'tp_size', 'seed', *er.SEARCH_PROTOCOL}
# Estimator and optimisation settings that must equal the recorded ALFWorld control.
SHARED_WITH_CONTROL = ('gamma', 'adv_mode', 'step_advantage_w', 'lr', 'kl_loss_type',
                       'max_response_length', 'train_temperature', 'train_top_p', 'train_top_k',
                       'remove_padding')
# Keys the controls predate; they ran with these runtime defaults.
IMPLIED = {'ccpo_loo': 1, 'ccpo_progress_history_weight': 1.0, 'ccpo_progress_mode': 'difference'}
OVERLAY = ROOT/'patches/verl-agent'
RUNTIME = ROOT/'verl-agent'
SOURCES = ['ccpo/future_progress.py', 'ccpo/core_ccpo.py', 'ccpo/arms.py', 'ablations/ablations.py',
           'scripts/exp_run.py', 'scripts/prepare_search_r1_ccpo.py', 'tests/test_search_protocol.py',
           'tests/test_search_r1_ccpo.py',
           'patches/verl-agent/agent_system/environments/env_manager.py',
           'patches/verl-agent/verl/trainer/ppo/ray_trainer.py']


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def run_name(arm, backbone, gpus, stamp, seed=0):
    suffix = f'-seed{seed}' if seed else ''
    return f'search-{ARMS[arm]["tag"]}-{backbone}-{len(gpus.split(","))}gpu{suffix}-{stamp}'


def stale_overlay_files():
    """Overlay files that are missing from, or differ in, the runnable checkout."""
    stale = []
    for patch in sorted(p for p in OVERLAY.rglob('*') if p.is_file() and '__pycache__' not in p.parts):
        live = RUNTIME/patch.relative_to(OVERLAY)
        if not live.is_file() or live.read_bytes() != patch.read_bytes():
            stale.append(str(patch.relative_to(OVERLAY)))
    return stale


def method_config(arm):
    """Registry method for an arm, with ALFWorld's benchmark keys removed."""
    _, cfg = ablations.build(ARMS[arm]['registry'], 'alfworld')
    return {k: v for k, v in cfg.items() if k not in BENCHMARK_KEYS}


def resolve(arm, backbone, gpus, steps, overrides, seed=0):
    cfg = method_config(arm)
    cfg.update(er.SEARCH_PROTOCOL)
    cfg.update(env_name='search', total_epochs=steps, model=MODELS[backbone], gpus=gpus, seed=seed,
               # Eight-GPU 7B runs in this tree pair GPUs per vLLM replica (prepare_main_7b.py).
               tp_size=2 if backbone == '7b' else 1,
               # The 150-step protocol pins step 100; this benchmark keeps best and last.
               pin_steps='',
               # Per-step .npz snapshots hold every hidden state: ~5,000 rows a step here.
               ccpo_progress_snapshot_every=er.SEARCH_PROTOCOL['test_freq'])
    unknown = sorted(set(overrides) - set(er.DEFAULTS))
    if unknown:
        raise ValueError('Unknown config key(s): ' + ', '.join(unknown))
    locked = sorted(k for k in overrides if k.startswith('ccpo_') and k != 'ccpo_progress_snapshot_every')
    if locked:
        raise ValueError('Estimator settings come from the registry and cannot be overridden here: '
                         + ', '.join(locked))
    cfg.update(overrides)
    return cfg


def check_method(arm, new, overrides=()):
    """The CCPO estimator must be the ALFWorld main method, key for key.

    Keys named in `overrides` were set on the command line; resolve() already
    refuses estimator keys there, so only budget and resource keys can be skipped.
    """
    spec = ARMS[arm]
    registry = method_config(arm)
    for key, value in registry.items():
        if key not in overrides and new[key] != value:
            raise ValueError(f'Method differs from the registry: {key}={new[key]!r}, expected {value!r}')
    if spec['control'] is None:
        return {}
    control = ROOT/'experiments'/spec['control']/'config.json'
    old = json.loads(control.read_text())['config']
    for key, value in IMPLIED.items():
        old.setdefault(key, value)
    keys = sorted(k for k in old if k.startswith('ccpo_') and k != 'ccpo_progress_snapshot_every')
    for key in [*keys, *SHARED_WITH_CONTROL]:
        if key not in overrides and new[key] != old[key]:
            raise ValueError(f'Differs from the ALFWorld control {spec["control"]}: '
                             f'{key}={new[key]!r}, control {old[key]!r}')
    return {k: dict(control=old[k], prepared=new[k]) for k in sorted(old) if k in new and old[k] != new[k]}


def prepare(arm, backbone, gpus, steps, stamp, overrides, skip_runtime_check=False, seed=0):
    spec = ARMS[arm]
    stale = stale_overlay_files()
    if stale and not skip_runtime_check:
        raise RuntimeError('verl-agent/ does not match patches/verl-agent/ (' + ', '.join(stale[:3])
                           + ('...' if len(stale) > 3 else '') + '). Run `bash scripts/sync_patches.sh`.')
    n_gpus = len(gpus.split(','))
    name = run_name(arm, backbone, gpus, stamp, seed)
    folder = ROOT/'experiments'/name
    if folder.exists():
        raise FileExistsError(f'Refusing to overwrite existing experiment: {folder}')
    cfg = resolve(arm, backbone, gpus, steps, overrides, seed)
    if cfg['train_batch_size'] * cfg['group_size'] % n_gpus:
        raise ValueError('train_batch_size * group_size must be divisible by the GPU count')
    argv = [sys.executable, str(ROOT/'scripts/exp_run.py'), '--name', name.rsplit('-', 1)[0],
            '--date', stamp, '--arm', spec['arm'], '--dry-run', *ablations.arms.flags(cfg)]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
               MKL_NUM_THREADS='1', HF_HOME=str(ROOT/'hf'))
    result = subprocess.run(argv, cwd=ROOT, env=env, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)
    (folder/'prepare.log').write_text(result.stdout + result.stderr)
    record = json.loads((folder/'config.json').read_text()); new = record['config']
    for key, value in cfg.items():
        if new[key] != value:
            raise ValueError(f'Resolved config mismatch: {key}={new[key]!r}, requested {value!r}')
    changes = check_method(arm, new, overrides)
    overrides_cli = record['hydra_overrides']
    assert overrides_cli == er.build_command(new, str(folder))[3:]
    assert f'algorithm.adv_estimator={spec["arm"]}' in overrides_cli
    assert 'env.env_name=search' in overrides_cli and 'data.truncation=left' in overrides_cli
    assert f'trainer.total_training_steps={steps}' in overrides_cli
    assert ('algorithm.gigpo.enable_similarity=True' in overrides_cli) == (spec['arm'] == 'gigpo')
    assert record['env']['ACG_SEARCH_REWARD_SCALE'] == str(er.SEARCH_FIXED['reward_scale'])
    if spec['arm'] == 'ccpo':
        assert new['ccpo_ep_w'] == spec['episode'] and new['ccpo_edge_w'] == 0
        assert new['ccpo_lk_fix'] == new['ccpo_lam_fix'] == 1 and new['ccpo_progress_horizon'] == 2
        assert new['ccpo_gate'] == 'hard' and new['ccpo_sim'] == 0 and new['ccpo_backoff_task'] == 1
        assert new['ccpo_target'] == 'return' and new['ccpo_phi'] == 'hidden+ctx'
    subprocess.run(['bash', '-n', str(folder/'run.sh')], check=True)
    departures = record['reference_protocol']['departures']
    status = dict(status='prepared_not_launched', experiment=folder.name, benchmark='search-r1',
        arm=arm, estimator=spec['arm'], registry=spec['registry'] if spec['arm'] == 'ccpo' else None,
        alfworld_control=spec['control'], episode_weight=new['ccpo_ep_w'] if spec['arm'] == 'ccpo' else None,
        backbone=backbone, model=new['model'], steps=steps, seed=new['seed'],
        questions_per_step=new['train_batch_size'], group_size=new['group_size'],
        max_turns=new['max_steps'], history_length=new['history_length'],
        future_horizon=new['ccpo_progress_horizon'] if spec['arm'] == 'ccpo' else None,
        reward_scale=er.SEARCH_FIXED['reward_scale'], gpu_count=n_gpus, gpus=new['gpus'],
        train_files=new['search_train_files'], val_files=new['search_val_files'],
        train_files_present=Path(new['search_train_files']).is_file(),
        val_files_present=Path(new['search_val_files']).is_file(),
        protocol_departures=departures, extra_overrides=overrides,
        runtime_overlay_verified=not stale, stale_overlay_files=stale,
        launched=False, queued=False, gpu_preflight_performed=False,
        prepared_at=datetime.now(timezone.utc).isoformat(), source_git=er.git_state())
    record['preparation'] = status
    dump(folder/'config.json', record); dump(folder/'PREPARED.json', status)
    if spec['control']:
        dump(folder/'config-diff-from-alfworld-control.json',
             dict(control='experiments/' + spec['control'], changes=changes, implied_control_defaults=IMPLIED))
    dump(folder/'prepare-command.json', dict(argv=argv, cwd=str(ROOT), gpu_visibility_for_preparation=''))
    dump(folder/'prepared-source-sha256.json',
         {s: hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in SOURCES})
    (folder/'NOTES.md').write_text(notes(status))
    return folder


def notes(p):
    if p['estimator'] == 'ccpo':
        actor = 'mask * N_task(H + F)' + (' + E' if p['episode_weight'] else '')
        method = f'''**Arm.** The ALFWorld main method (`{p['registry']}`) with no estimator change;
[the diff](config-diff-from-alfworld-control.json) against `{p['alfworld_control']}`
lists benchmark, backbone and resource keys only.

```text
Z_s = gamma^(T-s) * R,  R = 10 on exact match, else 0
V_s = C_s[Z];  H_t = Y_t - C_t[Y];  u = min(t+{p['future_horizon']}, T)
F_t = z_task(V_u - V_t)
A_actor = {actor}
```

Peers share the task and the exact observation; a turn without such a peer uses
the task bucket. Step 0 always matches (the question). A later turn matches only
when two rollouts issued the same query, since the observation is the retrieved
passages.

**Watch.** `outputs/future_progress/step-*.metrics.json`:
`progress_current_exact_frac` and `progress_current_backoff_frac` are the share of
turns with an exact peer and on the task fallback. Report them next to the score;
they are the recurrence statistic for this benchmark.'''
    else:
        extra = ' with similarity grouping at 0.9, as published' if p['estimator'] == 'gigpo' else ''
        method = f'''**Arm.** `{p['estimator']}` baseline{extra}. Same resolved config as the CCPO
arms prepared by this script; only `arm` differs.'''
    departures = ''.join(f'\n- `{k}`: {v["launched"]!r} instead of the reference {v["reference"]!r}'
                         for k, v in sorted(p['protocol_departures'].items())) or '\n- none'
    missing = [label for label, ok in (('training', p['train_files_present']),
                                       ('validation', p['val_files_present'])) if not ok]
    data = (f'\n\n**Data.** Not found when this was prepared: {" and ".join(missing)} parquet. '
            'See experiments/SEARCH_R1.md.') if missing else ''
    overlay = '' if p['runtime_overlay_verified'] else (
        '\n\n**Overlay.** `verl-agent/` did not match `patches/verl-agent/` when this was prepared. '
        'Run `bash scripts/sync_patches.sh` before launching.')
    return f'''# Search-R1, {p['arm']}, {p['model']}

**Prepared only; not launched or queued.** {p['steps']} steps of {p['questions_per_step']} questions x
{p['group_size']} rollouts, at most {p['max_turns']} turns, seed {p['seed']}, {p['gpu_count']} GPUs
(`{p['gpus']}` configured, not reserved).

**Question.** Does the method hold on a benchmark with free-form actions and
retrieved-passage observations, where observations recur far less than on
ALFWorld and WebShop?

{method}

**Protocol.** GiGPO's published `run_search.sh`, applied by `scripts/exp_run.py`.
The exact-match reward is scaled to 10/0 (`ACG_SEARCH_REWARD_SCALE`) with the
invalid-action penalty at 0.1, the reference's 1/0 and 0.01 times ten, so
`val/test_score` is ten times exact match and `success_rate` is exact match.
Launcher keys that depart from the reference:{departures}{data}{overlay}

Start the retrieval server first (experiments/SEARCH_R1.md), then launch from the
repository root: `bash experiments/{p['experiment']}/run.sh`.
'''


def parse_overrides(pairs):
    out = {}
    for pair in pairs:
        key, sep, value = pair.partition('=')
        if not sep:
            raise ValueError(f'--set expects KEY=VALUE, got {pair!r}')
        out[key] = er._coerce(value)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--arm', choices=[*ARMS, 'all'], default='all')
    p.add_argument('--backbone', choices=sorted(MODELS), required=True)
    p.add_argument('--gpus', required=True, help='CUDA device ids recorded for the run, e.g. 0,1,2,3,4,5,6,7')
    p.add_argument('--steps', type=int, default=er.SEARCH_PROTOCOL['total_epochs'],
                   help='training steps; the default is the reference single epoch')
    p.add_argument('--seed', type=int, default=0, help='env seed; nonzero seeds are named in the folder')
    p.add_argument('--val-files', default=None,
                   help='validation parquet, e.g. a subset from scripts/subset_search_val.py')
    p.add_argument('--set', action='append', default=[], metavar='KEY=VALUE',
                   help='further launcher overrides (budget or resources), recorded as departures')
    p.add_argument('--skip-runtime-check', action='store_true',
                   help='prepare without a synced verl-agent/ checkout; recorded in PREPARED.json')
    p.add_argument('--date', default=date.today().strftime('%Y%m%d'))
    a = p.parse_args()
    overrides = parse_overrides(a.set)
    if a.val_files:
        overrides['search_val_files'] = str(Path(a.val_files).resolve())
    for arm in (list(ARMS) if a.arm == 'all' else [a.arm]):
        print(prepare(arm, a.backbone, a.gpus, a.steps, a.date, overrides, a.skip_runtime_check, a.seed))


if __name__ == '__main__':
    main()
