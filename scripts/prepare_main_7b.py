#!/usr/bin/env python3
"""Prepare the four 7B/eight-GPU H2 main-method experiments. Never starts training."""
import argparse
from datetime import date, datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'ablations'))
import ablations
spec = importlib.util.spec_from_file_location('main_7b_exp_run', ROOT/'scripts/exp_run.py')
er = importlib.util.module_from_spec(spec); spec.loader.exec_module(er)

MAIN = 'future-progress-h2-no-credit-shrinkage'
ACTIVE = MAIN + '-active-episode'
CONTROLS = {
    ('alfworld', 0): 'm10-h2-no-credit-shrinkage-alfworld-1.5b-2gpu-20260919',
    ('alfworld', 1): 'm10-h2-noshrink-active-episode-alfworld-1.5b-2gpu-20260920',
    ('webshop', 0): 'm11-h2-no-credit-shrinkage-webshop-1.5b-2gpu-20260919',
    ('webshop', 1): 'm11-h2-noshrink-active-episode-webshop-1.5b-2gpu-20260920',
}
SCALE_KEYS = {'exp_id', 'model', 'model_path', 'gpus', 'tp_size'}
SOURCES = ['ablations/ablations.py', 'ablations/run.py', 'ccpo/arms.py',
    'ccpo/core_ccpo.py', 'ccpo/future_progress.py', 'scripts/exp_run.py',
    'scripts/prepare_main_7b.py', 'scripts/plot_metrics.py',
    'patches/verl-agent/verl/trainer/ppo/ray_trainer.py',
    'verl-agent/verl/trainer/ppo/ray_trainer.py']


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True)+'\n')


def run_name(benchmark, episode, stamp):
    prefix = 'm10' if benchmark=='alfworld' else 'm11'
    suffix = '-active-episode' if episode else ''
    return f'{prefix}-h2-noshrink{suffix}-{benchmark}-7b-8gpu-{stamp}'


def prepare(benchmark, episode, stamp, gpus):
    key = ACTIVE if episode else MAIN
    control = ROOT/'experiments'/CONTROLS[benchmark, episode]
    old = json.loads((control/'config.json').read_text())['config']
    old.setdefault('ccpo_progress_history_weight', 1.0)  # Legacy runtime default.
    old.setdefault('ccpo_loo', 1)  # Implicit legacy trajectory exclusion.
    name = run_name(benchmark, episode, stamp)
    destination = ROOT/'experiments'/name
    if destination.exists():
        raise FileExistsError(f'Refusing to overwrite existing experiment: {destination}')
    canonical, method = ablations.build(key, benchmark, backbone='7b')
    old_name, old_method = ablations.build(key, benchmark)
    for k, v in old_method.items():
        if old[k] != v: raise ValueError(f'Recorded 1.5B control differs from registry: {k}')
    cfg = {k: v for k, v in old.items() if k in er.DEFAULTS}
    cfg.update(method, gpus=gpus, tp_size=2)
    command = [sys.executable, str(ROOT/'scripts/exp_run.py'), '--name', name.rsplit('-', 1)[0],
               '--date', stamp, '--dry-run', *ablations.arms.flags(cfg)]
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1',
                       OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                       HF_HOME=str(ROOT/'hf'), ALFWORLD_DATA=str(ROOT/'alfworld_data'),
                       ACG_DATA_DIR=str(ROOT/'envdata/verl_data'))
    completed = subprocess.run(command, cwd=ROOT, env=environment, capture_output=True, text=True)
    if completed.returncode:
        raise RuntimeError(completed.stdout + completed.stderr)
    (destination/'prepare.log').write_text(completed.stdout + completed.stderr)
    record = json.loads((destination/'config.json').read_text()); new = record['config']
    changes = {k: dict(control=old[k], prepared=new[k]) for k in old if old[k] != new[k]}
    if set(old) != set(new) or set(changes) != SCALE_KEYS:
        raise ValueError(f'Unexpected changes from the matched 1.5B control: {changes}')
    if new['model_path'] == old['model_path']:
        raise ValueError('7B run must never retain the 1.5B resolved checkpoint')
    for k, v in method.items():
        if new[k] != v: raise ValueError(f'Resolved method mismatch: {k}')
    assert new['train_batch_size'] * new['group_size'] % 8 == 0
    assert 8 % new['tp_size'] == 0 and new['resume_from'] == ''
    assert new['total_epochs'] == 150 and new['seed'] == 0
    assert new['ccpo_lk_fix'] == 1 and new['ccpo_ep_w'] == episode
    assert new['ccpo_progress_horizon'] == new['history_length'] == 2
    assert record['env']['VERL_ATTN_IMPL'] == 'flash_attention_2'
    assert record['hydra_overrides'] == er.build_command(new, str(destination))[3:]
    subprocess.run(['bash', '-n', str(destination/'run.sh')], check=True)
    prepared = dict(status='prepared_not_launched', experiment=name, ablation=key, canonical_id=canonical,
        benchmark=benchmark, model=new['model'], backbone='7b', steps=150, seed=0,
        gpus=gpus, gpu_count=8, rollout_tensor_parallel_size=2, history_length=2,
        future_horizon=2, lambda_k_fixed=1., episode_weight=float(episode),
        context_stat_weight=1., original_edge_weight=0., max_steps=new['max_steps'],
        control=str(control.relative_to(ROOT)), canonical_1_5b_control=old_name,
        paired_experiment=run_name(benchmark, 1-episode, stamp),
        launched=False, queued=False, gpu_preflight_performed=False,
        gpu_memory_calibrated=False, model_weights_present=Path(new['model_path']).is_dir(),
        model_path_resolution='Existing cache snapshot if available; otherwise Hugging Face model ID',
        prepared_at=datetime.now(timezone.utc).isoformat(), source_git=er.git_state())
    record['preparation'] = prepared
    # The generic launcher's static reference deltas describe TP=1; record the
    # actual matched scaling comparison instead of retaining that stale label.
    record['reference_protocol'] = dict(source_config=str(control.relative_to(ROOT)/'config.json'),
        benchmark_and_learning_settings_unchanged=True, changes=changes,
        rollout_parallelism='TP=2, four rollout replicas on eight GPUs; FSDP world size eight')
    dump(destination/'config.json', record)
    dump(destination/'PREPARED.json', prepared)
    dump(destination/'config-diff-from-1.5b.json', dict(control=str(control.relative_to(ROOT)), changes=changes))
    dump(destination/'prepare-command.json', dict(argv=command, gpu_visibility_for_preparation=''))
    dump(destination/'prepared-source-sha256.json', {
        p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in SOURCES})
    (destination/'NOTES.md').write_text(notes(prepared))
    return destination


def notes(p):
    name = p['paired_experiment']; active = bool(p['episode_weight'])
    fusion = 'E + N_CC(H + F)' if active else 'N_CC(H + F)'
    return f'''# {'M10' if p['benchmark']=='alfworld' else 'M11'} H2 no-shrink {'+ active episode' if active else 'main method'}: 7B / eight GPUs

**Prepared only; not launched or queued.** Fresh `Qwen/Qwen2.5-7B-Instruct`, seed 0,
150 training steps, eight GPUs (`{p['gpus']}`). GPU IDs are configuration, not a reservation.
Registry key: `{p['ablation']}`. Canonical ID: `{p['canonical_id']}`.

## Method and comparison

History length 2, future horizon 2, context statistics enabled, soft exponential
similarity (`ccpo_tau=0.15`, whitening removes 3 directions). Whole query trajectories
are excluded from the contextual readout. All usable history/current/future
baselines use their context estimate at full strength (`ccpo_lk_fix=1`,
`ccpo_lam_fix=1`); kappa 2 is retained but cannot shrink the baseline. Existing
fallback, unsupported-row handling, terminal 10/0 potentials and detached features
are unchanged. Original edge weight 0; history/future weights 1/1.

Let `H = Y - C[Y]`, `V_s = C_s[Z]`, and
`F = z_task(V_min(t+2,T) - V_t)`. Current/future readouts use their respective
observation groups. Y includes the existing invalid-action penalty; Z is the
unpenalized discounted potential target. The masked actor advantage is
`A = mask * ({fusion})`. N_CC is task mean/std normalization on ALFWorld and
identity on WebShop. Future F is task-standardized on both benchmarks.
Episode coefficient is **{p['episode_weight']:.0f}**. {'E is the existing trainer episode channel, added after contextual normalization without normalizing the final sum.' if active else 'Episode advantage remains diagnostic-only and has zero actor coefficient; episode reward still defines return/potential targets.'}
E uses the existing task-level turn-row mean/std on ALFWorld and mean subtraction
on WebShop, including the -0.1 invalid-action penalty. No new episode normalization
or reward definition is introduced. KL remains a separate loss.

Full math: [main-method definition](../MAIN_METHOD.md) and
[active-episode implementation](../{'m10' if p['benchmark']=='alfworld' else 'm11'}-h2-noshrink-active-episode-{p['benchmark']}-1.5b-2gpu-20260920/NOTES.md).

Paired 7B arm: [notes](../{name}/NOTES.md). Only `ccpo_ep_w` differs between the
pair's learning settings. [config-diff-from-pair.json](config-diff-from-pair.json)
records that change and the separate experiment ID.
The matched 1.5B control is `{p['control']}`. Scale changes are only model,
resolved model path, GPU allocation, rollout TP1→2, and experiment identity;
see [config-diff-from-1.5b.json](config-diff-from-1.5b.json).

## Protocol and resource preparation

16 tasks × 8 rollouts = 128 episodes per training step; eight GPUs do **not**
change the number of rollouts per task. Rollout TP 2 gives four replicas; actor/ref
FSDP use eight ranks. Gradient checkpointing and dynamic token batching stay on.
Existing per-GPU token budgets remain update 12288 / log-prob 24576 and require an
8-GPU memory smoke test before claiming runtime readiness. There is no GPU test
or memory-fit claim from this preparation.

Training and validation turn ceiling: **{p['max_steps']}**. This is the standard
{'ALFWorld 50-turn' if p['benchmark']=='alfworld' else 'WebShop 15-turn'} protocol;
WebShop retains its 1K catalog and original scorer. Prompt/response limits,
optimizer, minibatches, validation sampling and all other benchmark settings
match the recorded 1.5B control. Learning rate 1e-6, discount 0.95, separate KL 0.01;
validation/checkpoint every 5 steps, early stopping off; best, step 100 and last
checkpoints retained. History/current/future and applied episode diagnostics
continue through the existing estimator and plot code.

## Reproduce / launch later

The preparation command is recorded in [prepare-command.json](prepare-command.json).
To prepare all four arms in a new dated directory, run:

```bash
python scripts/prepare_main_7b.py --date YYYYMMDD --gpus 0,1,2,3,4,5,6,7
```

The generator refuses existing experiment directories, runs the launcher only
with `--dry-run`, and preserves the full paired 1.5B protocol. The public registry
also accepts `--backbone 7b`; use the prepared full config when reproducing the
local benchmark-specific settings.

When these eight GPUs are assigned and launch is requested:

```bash
bash experiments/{p['experiment']}/run.sh
```

The full command
and exported method flags are in [config.json](config.json). The model field is
7B and the resolved checkpoint cannot point at the old 1.5B weights. If 7B weights
are not cached, `model_path` is the public Hugging Face model ID; a future launch
needs access to those weights. No weights are downloaded by this preparation.
Environment/setup details and source hashes are retained. [VALIDATION.json](VALIDATION.json)
records CPU checks separately from the pending GPU smoke test.
'''


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--date', default=date.today().strftime('%Y%m%d'))
    p.add_argument('--gpus', default='0,1,2,3,4,5,6,7')
    a = p.parse_args()
    try:
        datetime.strptime(a.date, '%Y%m%d')
        ids = [int(v) for v in a.gpus.split(',')]
        if len(ids)!=8 or len(set(ids))!=8 or min(ids)<0: raise ValueError()
    except ValueError:
        p.error('Use a YYYYMMDD date and exactly eight distinct nonnegative GPU IDs')
    # Refuse the whole batch before writing anything if one destination exists.
    for b, e in CONTROLS:
        if (ROOT/'experiments'/run_name(b,e,a.date)).exists():
            p.error(f'Experiment already exists: {run_name(b,e,a.date)}')
    folders = {(b,e): prepare(b,e,a.date,a.gpus) for b,e in CONTROLS}
    for (b,e), destination in folders.items():
        other = folders[b,1-e]
        cfg = json.loads((destination/'config.json').read_text())['config']
        parent = json.loads((other/'config.json').read_text())['config']
        changes = {k:dict(control=parent[k],prepared=cfg[k]) for k in cfg if cfg[k]!=parent[k]}
        if set(changes)!={'exp_id','ccpo_ep_w'}: raise ValueError(changes)
        dump(destination/'config-diff-from-pair.json', dict(control=other.name,changes=changes))
    print(json.dumps({'status':'prepared_not_launched','experiments':[str(p.relative_to(ROOT)) for p in folders.values()]}))


if __name__ == '__main__': main()
