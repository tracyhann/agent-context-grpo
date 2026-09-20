#!/usr/bin/env python3
"""Evaluate the pinned Qwen model via a local vLLM /v1/completions server."""
import argparse
import concurrent.futures
import datetime
import importlib.metadata
import json
import multiprocessing
import math
from pathlib import Path
import re
import sys
import urllib.request

# Also works through the baselines/qwen compatibility link.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qwen_baseline.common import MODEL, PROFILES, SERVED_MODEL, aggregate, atomic_json, complete, episode_plan, fingerprint, snapshot_path, source_hashes
from qwen_baseline.environments import build_environment, runtime_setup

_TOKENIZER = None


def plain(value):
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [plain(v) for v in value]
    if hasattr(value, 'tolist'):
        return plain(value.tolist())
    if isinstance(value, Path):
        return str(value)
    return value


def tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        from transformers import AutoTokenizer
        _TOKENIZER = AutoTokenizer.from_pretrained(str(snapshot_path()), local_files_only=True)
    return _TOKENIZER


def run_episode(item, settings):
    runtime_setup()
    manager = raw = None
    record = dict(item, status='error', benchmark=settings['benchmark'], turns=[])
    try:
        manager, raw = build_environment(settings['benchmark'], item, settings['split'], settings['history_length'])
        observations, infos = manager.reset({})
        record['initial_observation'] = observations['anchor'][0]
        record['initial_info'] = plain(infos[0])
        if settings['benchmark'] == 'webshop':
            record['initial_session'] = plain(raw.details())
        rewards, scores = [], []
        for turn in range(settings['max_steps']):
            prompt = observations['text'][0]
            ids = tokenizer().apply_chat_template([dict(role='user', content=prompt)],
                add_generation_prompt=True, tokenize=True, enable_thinking=settings['thinking'],
                preserve_thinking=False, reasoning_effort=settings['reasoning_effort'])
            if len(ids) > settings['max_prompt_tokens']:
                raise ValueError(f'Prompt has {len(ids)} tokens; configured limit is {settings["max_prompt_tokens"]}. No silent truncation.')
            result = complete(ids, settings, settings['generation_seed'] + item['episode_id'] * 1000 + turn)
            before = observations['anchor'][0]
            observations, reward, done, infos = manager.step([result['text']])
            info = infos[0]
            rewards.append(float(reward[0]))
            scores.append(float(info.get('task_score', bool(info.get('won', False)))))
            record['turns'].append(dict(turn=turn, prompt=prompt, prompt_tokens=len(ids),
                prompt_ids_sha256=fingerprint(ids), observation=before,
                next_observation=observations['anchor'][0], response=result['text'],
                action=raw.last_action, reward=float(reward[0]), score=scores[-1], done=bool(done[0]),
                legacy_format_valid=bool(info['is_action_valid']),
                action_tag_valid=bool(re.search(r'<action>\s*[^<>]+\s*</action>', result['text'], re.I)),
                finish_reason=result['finish_reason'], usage=result['usage'], request_seed=result['request_seed']))
            if bool(done[0]):
                break
        record.update(status='completed', reward=sum(rewards), success=any(r == 10.0 for r in rewards),
                      score=max(scores, default=0.), length=len(rewards),
                      truncated=not bool(done[0]), ended=datetime.datetime.now(datetime.timezone.utc).isoformat())
        if settings['benchmark'] == 'webshop':
            record['final_session'] = plain(raw.details())
    except Exception as error:
        record.update(error=f'{type(error).__name__}: {error}', length=len(record['turns']))
    finally:
        if raw is not None:
            try:
                raw.close()
            except Exception as error:
                record['cleanup_error'] = f'{type(error).__name__}: {error}'
    return plain(record)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark', required=True, choices=PROFILES)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--base-url', default='http://127.0.0.1:8018/v1')
    parser.add_argument('--episodes', type=int)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--env-seed', type=int, default=1000)
    parser.add_argument('--generation-seed', type=int, default=0)
    parser.add_argument('--eval-round', type=int, default=0)
    parser.add_argument('--history-length', type=int, default=2)
    parser.add_argument('--max-steps', type=int)
    parser.add_argument('--max-prompt-tokens', type=int)
    parser.add_argument('--max-tokens', type=int, default=512)
    parser.add_argument('--temperature', type=float, default=.4)
    parser.add_argument('--thinking', action='store_true')
    parser.add_argument('--reasoning-effort', choices=['low','medium','xhigh'], default='low')
    parser.add_argument('--alfworld-split', choices=['eval_in_distribution','eval_out_of_distribution'], default='eval_in_distribution')
    parser.add_argument('--timeout', type=float, default=180)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    if args.workers < 1 or args.history_length < 0 or args.max_tokens < 1:
        parser.error('workers/max-tokens must be positive and history-length nonnegative')
    if any(n is not None and n < 1 for n in [args.max_steps, args.max_prompt_tokens]):
        parser.error('max-steps/max-prompt-tokens must be positive')
    if not math.isfinite(args.temperature) or args.temperature < 0 or not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error('temperature must be finite and nonnegative; timeout must be finite and positive')
    profile = PROFILES[args.benchmark]
    settings = dict(benchmark=args.benchmark, base_url=args.base_url.rstrip('/'),
        max_steps=args.max_steps or profile['max_steps'], max_prompt_tokens=args.max_prompt_tokens or profile['max_prompt_tokens'],
        max_tokens=args.max_tokens, temperature=args.temperature, thinking=args.thinking,
        reasoning_effort=args.reasoning_effort, history_length=args.history_length,
        generation_seed=args.generation_seed, timeout=args.timeout,
        split=args.alfworld_split if args.benchmark == 'alfworld' else profile['split'])
    try:
        plan = episode_plan(args.benchmark, args.episodes, args.env_seed, args.eval_round)
    except ValueError as error:
        parser.error(str(error))
    config = dict(model=MODEL, served_model=SERVED_MODEL, settings=settings, episode_plan=plan,
                  source_sha256=source_hashes(args.benchmark),
                  scorer='original benchmark scorer; no item-option correction',
                  prompt_protocol='existing environment manager; single user message; native model chat template',
                  evaluation_only=True, eval_round=args.eval_round)
    config['fingerprint'] = fingerprint(config)
    output = args.output.resolve()
    config_path = output / 'config.json'
    if config_path.exists():
        old = json.loads(config_path.read_text())
        if not args.resume or old != config:
            parser.error('Output exists; use --resume only with the identical model/protocol/episode plan')
    elif args.resume:
        parser.error('Cannot resume an output without config.json')
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(config_path, config)
    if args.prepare_only:
        print(f'Prepared {len(plan)} {args.benchmark} episodes in {output}')
        return 0
    with urllib.request.urlopen(settings['base_url'] + '/models', timeout=10) as response:
        models = json.load(response)
    if SERVED_MODEL not in [m['id'] for m in models.get('data', [])]:
        raise RuntimeError('Serving endpoint does not advertise the pinned model/revision ID; use serve.py')
    snapshot_path()
    versions = {}
    for package in ['transformers', 'tokenizers', 'numpy', 'gym', 'alfworld', 'textworld', 'ray']:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    atomic_json(output/'runtime.json', dict(python=sys.version, packages=versions, server_models=models,
                started=datetime.datetime.now(datetime.timezone.utc).isoformat(), workers=args.workers))
    records = []
    for item in plan:
        path = output / 'episodes' / f'{item["episode_id"]:04d}.json'
        if path.exists():
            record = json.loads(path.read_text())
            if any(record.get(k) != value for k, value in item.items()):
                raise ValueError(f'Saved episode identity differs from the plan: {path}')
            if record.get('status') == 'completed':
                records.append(record)
    done_ids = {r['episode_id'] for r in records}
    context = multiprocessing.get_context('spawn')
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as pool:
        jobs = {pool.submit(run_episode, item, settings):item for item in plan if item['episode_id'] not in done_ids}
        for future in concurrent.futures.as_completed(jobs):
            item = jobs[future]
            try:
                record = future.result()
            except Exception as error:
                record = dict(item, status='error', error=f'{type(error).__name__}: {error}', turns=[])
            atomic_json(output/'episodes'/f'{item["episode_id"]:04d}.json', record)
            records.append(record)
            metrics = aggregate(records, len(plan))
            atomic_json(output/'metrics.json', metrics)
            print(json.dumps({k:metrics[k] for k in ['status','completed_episodes','error_episodes','expected_episodes']}), flush=True)
    metrics = aggregate(records, len(plan))
    atomic_json(output/'metrics.json', metrics)
    with (output/'episodes.jsonl').open('w') as stream:
        for record in sorted(records,key=lambda r:r['episode_id']):
            stream.write(json.dumps(record, ensure_ascii=False) + '\n')
    print(json.dumps(metrics, indent=2))
    return 0 if metrics['status'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
