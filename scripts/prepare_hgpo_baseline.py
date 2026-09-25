#!/usr/bin/env python3
"""Prepare the HGPO budget-matched baseline; never launch or reserve GPUs."""
import argparse
from datetime import date, datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import exp_run as er

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = '20bd331bdbc9026a5668e11362178e10ab7400c8'
# The estimator is used verbatim, not translated into CCPO's grouping/fusion.
CORE_SHA256 = 'cc8bdb17d04c47b8514f9d2539f49f00074df729ceeea72eb45bde775c7c0200'


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def settings(benchmark, backbone='1.5b', gpus=None, history=2, prompt_tokens=2048, response_tokens=512):
    if benchmark not in ('alfworld', 'webshop') or backbone not in ('1.5b', '7b'):
        raise ValueError('Unsupported benchmark or backbone')
    if history not in (2, 4) or min(prompt_tokens, response_tokens) < 1:
        raise ValueError('Use history 2 or 4 and positive token limits')
    gpus = gpus or ('0,1,2,3' if backbone == '1.5b' else '0,1,2,3,4,5,6,7')
    ids = [int(value) for value in gpus.split(',')]
    tp = 1 if backbone == '1.5b' else 2
    if not ids or min(ids) < 0 or len(set(ids)) != len(ids) or 128 % len(ids) or len(ids) % tp:
        raise ValueError('GPU IDs must be distinct; count must divide 128 and support rollout TP')
    return dict(arm='hgpo', model=f'Qwen/Qwen2.5-{ "1.5B" if backbone == "1.5b" else "7B" }-Instruct',
        env_name='Webshop' if benchmark == 'webshop' else 'alfworld/AlfredTWEnv',
        gpus=','.join(map(str, ids)), tp_size=tp, seed=0, train_batch_size=16, group_size=8,
        max_prompt_length=prompt_tokens, max_response_length=response_tokens,
        total_epochs=150, max_steps=15 if benchmark == 'webshop' else 50,
        history_length=history, val_batch_size=64, test_freq=5, save_freq=5,
        early_stop_patience=0, resume_from='', compact_mode='off', obs_repair=0,
        anchor_aff=0, force_budget=0, vllm_attn_backend='FLASH_ATTN',
        adv_mode='mean_norm' if benchmark == 'webshop' else 'mean_std_norm')


def name_for(benchmark, backbone, cfg, stamp):
    return (f'hgpo-k{cfg["history_length"]}-prompt{cfg["max_prompt_length"]}'
            f'-response{cfg["max_response_length"]}-{benchmark}-{backbone}'
            f'-{len(cfg["gpus"].split(","))}gpu-{stamp}')


def prepare(benchmark, backbone, cfg, stamp):
    folder = ROOT / 'experiments' / name_for(benchmark, backbone, cfg, stamp)
    if folder.exists():
        raise FileExistsError(f'Refusing to overwrite {folder}')
    recipe = ROOT / 'verl-agent/recipe/hgpo'
    if hashlib.sha256((recipe / 'core_hgpo.py').read_bytes()).hexdigest() != CORE_SHA256:
        raise ValueError('HGPO core differs from the pinned upstream recipe; inspect before preparing')
    command = [sys.executable, str(ROOT / 'scripts/exp_run.py'), '--arm', 'hgpo',
               '--name', folder.name.rsplit('-', 1)[0], '--date', stamp, '--dry-run']
    for key, value in cfg.items():
        if key != 'arm':
            command += ['--set', f'{key}={value}']
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
        OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
        HF_HOME=str(ROOT / 'hf'), ALFWORLD_DATA=str(ROOT / 'alfworld_data'),
        ACG_DATA_DIR=str(ROOT / 'envdata/verl_data'))
    completed = subprocess.run(command, cwd=ROOT, env=environment, text=True, capture_output=True)
    if completed.returncode:
        raise RuntimeError(completed.stdout + completed.stderr)
    record = json.loads((folder / 'config.json').read_text())
    resolved = record['config']
    for key, value in cfg.items():
        if resolved[key] != value:
            raise ValueError(f'Prepared setting drifted: {key}')
    expected = er.build_command(resolved, str(folder))
    assert expected[2] == 'recipe.hgpo.main_hgpo'
    assert record['hydra_overrides'] == expected[3:]
    assert 'trainer.total_training_steps=150' in expected
    assert record['env']['ACG_ADV_ESTIMATOR'] == 'hgpo'
    subprocess.run(['bash', '-n', str(folder / 'run.sh')], check=True)
    record['preparation'] = dict(status='prepared_not_launched', launched=False, queued=False,
        gpu_preflight_performed=False, benchmark=benchmark, backbone=backbone,
        prompt_tokens=cfg['max_prompt_length'], response_tokens=cfg['max_response_length'],
        steps=150, max_turns=cfg['max_steps'], history_length=cfg['history_length'],
        upstream_commit=UPSTREAM, prepared_at=datetime.now(timezone.utc).isoformat())
    record['reference_protocol']['budget_changes_from_official'] = {
        'prompt_tokens': {'official': 4096, 'prepared': cfg['max_prompt_length']},
        'response_tokens': {'official': 512, 'prepared': cfg['max_response_length']},
        'training': {'official': '160 epochs', 'prepared': '150 training iterations; 150 epoch ceiling'},
        'max_turns': {'official': 30 if benchmark == 'webshop' else 50, 'prepared': cfg['max_steps']},
    }
    record['reference_protocol']['runtime_changes'] = [
        'Shared benchmark overlay, dynamic token batching and local model/data paths',
        'Validation chunk size 64; entire existing validation parquet is evaluated',
        f'GPU count {len(cfg["gpus"].split(","))}; rollout TP {cfg["tp_size"]}; vLLM memory fraction 0.25',
        'Evaluate/save every 5 steps; retain the two latest actor checkpoints',
        'No memory compaction, observation repair, extra episode group, or CCPO feature extraction',
    ]
    source_files = [ROOT / 'scripts/exp_run.py', Path(__file__).resolve(),
                    *sorted(recipe.rglob('*.py')), *sorted(recipe.rglob('*.yaml')),
                    *sorted((ROOT / 'patches/verl-agent').rglob('*.py'))]
    hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in source_files if '__pycache__' not in path.parts}
    dump(folder / 'config.json', record)
    dump(folder / 'PREPARED.json', record['preparation'])
    dump(folder / 'prepare-command.json', dict(argv=command, cwd=str(ROOT), gpu_visibility=''))
    dump(folder / 'prepared-source-sha256.json', hashes)
    (folder / 'NOTES.md').write_text(notes(folder.name, benchmark, backbone, cfg))
    return folder


def notes(name, benchmark, backbone, cfg):
    return f'''# HGPO K={cfg['history_length']}: {benchmark}, {backbone}

**Prepared only; not launched or queued.** Seed 0, 16 tasks x 8 trajectories,
150 training iterations. Training and validation use at most {cfg['max_steps']} turns.
Prompt cap **{cfg['max_prompt_length']} tokens**, response cap **{cfg['max_response_length']} tokens per turn**.
These are separate limits; neither is the per-GPU dynamic minibatch token budget.

The official HGPO recipe at `{UPSTREAM}` supplies the estimator, trainer and
environment manager. Entry point: `recipe.hgpo.main_hgpo`; estimator:
`recipe.hgpo.core_hgpo.compute_hgpo_outcome_advantage`. The core is unchanged.
Exact observation-history groups, length weighting with alpha=1, and
`base_group=False` retain the official method. No separate episode advantage
is added. Grouping includes self and repeated visits, as upstream specifies.
Advantage mode is `{cfg['adv_mode']}`. The shared collector and PPO workers use
this repository's benchmark/runtime patches; source hashes are recorded.

`trainer.total_training_steps=150` explicitly caps iterations independently of
parquet length. `trainer.total_epochs=150` supplies enough epochs. Fresh start,
no early stopping, evaluate/save every 5 steps, keep two actor checkpoints.
ALFWorld uses the seen validation split; WebShop uses the 1K catalog and existing
scorer. Validation processes the whole configured parquet in chunks of 64.
Learning rate 1e-6, gamma=.95, terminal reward 10/0, local invalid penalty .1,
training temperature 1.0, validation temperature .4, separate KL loss .01.

Original HGPO budgets were prompt 4096, response 512, 160 epochs, and
WebShop 30 turns. The prepared delta and actual runtime settings are in
[config.json](config.json). Historical WebShop CCPO/GiGPO configurations used
4096 input tokens, so this 2048-input variant needs a matching control for a
strict token-budget comparison. HGPO retains upstream left truncation.

[Method, comparisons and preparation commands](../HGPO_BASELINE.md).
For a later launch on allocated GPUs:

```bash
bash experiments/{name}/run.sh
```

`PREPARED.json` records preparation, not runtime success. `VALIDATION.json`
records CPU checks separately; no GPU memory-fit or training result is claimed.
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark', choices=('alfworld', 'webshop', 'both'), default='both')
    parser.add_argument('--backbone', choices=('1.5b', '7b'), default='1.5b')
    parser.add_argument('--gpus', help='Allocation to record; defaults to four GPUs for 1.5B, eight for 7B')
    parser.add_argument('--history-length', type=int, choices=(2, 4), default=2)
    parser.add_argument('--prompt-tokens', type=int, default=2048)
    parser.add_argument('--response-tokens', type=int, default=512)
    parser.add_argument('--date', default=date.today().strftime('%Y%m%d'))
    args = parser.parse_args()
    benchmarks = ('alfworld', 'webshop') if args.benchmark == 'both' else (args.benchmark,)
    try:
        datetime.strptime(args.date, '%Y%m%d')
        configs = {b: settings(b, args.backbone, args.gpus, args.history_length,
                               args.prompt_tokens, args.response_tokens) for b in benchmarks}
        for b, cfg in configs.items():
            if (ROOT / 'experiments' / name_for(b, args.backbone, cfg, args.date)).exists():
                raise FileExistsError(f'Experiment already exists: {name_for(b, args.backbone, cfg, args.date)}')
    except (ValueError, FileExistsError) as error:
        parser.error(str(error))
    folders = [prepare(b, args.backbone, cfg, args.date) for b, cfg in configs.items()]
    print(json.dumps({'status': 'prepared_not_launched',
                      'experiments': [str(p.relative_to(ROOT)) for p in folders]}, indent=2))


if __name__ == '__main__':
    main()
