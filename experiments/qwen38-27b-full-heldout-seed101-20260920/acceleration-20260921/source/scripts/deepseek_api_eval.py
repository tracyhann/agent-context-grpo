#!/usr/bin/env python3
"""CPU-only DeepSeek benchmark pilot with exact task IDs and API usage accounting."""
import argparse
import concurrent.futures
import copy
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import random
import re
import sys
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qwen_baseline.common import atomic_json, fingerprint
from qwen_baseline.environments import runtime_setup, build_environment, SingleAlfworld

API_URL = 'https://api.deepseek.com/chat/completions'
RATES = {'cache_hit': .003, 'cache_miss': .15, 'output': .6}  # USD per million, off-peak Flash
_BASE = None


def plain(value):
    if isinstance(value, dict): return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)): return [plain(v) for v in value]
    if hasattr(value, 'tolist'): return plain(value.tolist())
    if isinstance(value, Path): return str(value)
    return value


def plan(benchmark, count, seed):
    rng = random.Random(seed)
    if benchmark == 'webshop':
        # Full local test split is each of the 500 goals once, not 256 sampled resets.
        ids = list(range(500)); rng.shuffle(ids)
        return [dict(episode_id=i, goal_index=g, worker_seed=seed) for i, g in enumerate(ids[:count])], {'test': 500}
    groups = {}
    for game in sorted((ROOT/'alfworld_data/json_2.1.1/valid_seen').rglob('game.tw-pddl')):
        if 'movable' in str(game) or 'Sliced' in str(game): continue
        if not json.loads(game.read_text()).get('solvable'): continue
        task = json.loads((game.parent/'traj_data.json').read_text())['task_type']
        groups.setdefault(task, []).append(str(game.relative_to(ROOT)))
    counts = {k: len(v) for k, v in groups.items()}
    if sum(counts.values()) != 140: raise ValueError(f'Expected 140 valid_seen games; got {counts}')
    # Six-task pilot: one randomly selected game of each task type, then round-robin.
    for values in groups.values(): rng.shuffle(values)
    chosen = []
    while len(chosen) < count:
        for task in sorted(groups):
            if groups[task] and len(chosen) < count: chosen.append((task, groups[task].pop()))
        if not any(groups.values()): break
    return [dict(episode_id=i, task_type=t, gamefile=g, worker_seed=seed)
            for i, (t, g) in enumerate(chosen)], counts


class ExactAlfworld(SingleAlfworld):
    def __init__(self, item):
        global _BASE
        from agent_system.environments.env_package.alfworld.envs import AlfworldWorker, load_config_file, get_environment
        if _BASE is None:
            cfg = load_config_file(str(ROOT/'verl-agent/agent_system/environments/env_package/alfworld/configs/config_tw.yaml'))
            cfg['general']['use_cuda'] = False
            _BASE = cfg, get_environment(cfg['env']['type'])(cfg, train_eval='eval_in_distribution')
        cfg, base = _BASE
        game = str(ROOT/item['gamefile'])
        if game not in base.game_files: raise ValueError('Selected game is absent from actual environment split')
        selected = copy.copy(base); selected.game_files = [game]; selected.num_games = 1
        self.worker = AlfworldWorker(cfg, item['worker_seed'], selected)
        self.reset_index = 0
        self.get_admissible_commands = []
        self.last_action = None


def environment(benchmark, item, history):
    if benchmark == 'webshop':
        manager, raw = build_environment(benchmark, item, history_length=history)
        # Lazy imports/initialization can consume Python RNG before price generation.
        # Pin the benchmark data explicitly after all imports, independent of worker reuse.
        import numpy as np
        from web_agent_site.engine.engine import generate_product_prices
        from web_agent_site.engine.goal import get_goals
        server = raw.worker.env.unwrapped.server
        random.seed(item['worker_seed'])
        server.product_prices = generate_product_prices(server.all_products)
        server.goals = get_goals(server.all_products, server.product_prices, human_goals=False)
        random.seed(item['worker_seed'])
        random.shuffle(server.goals)
        server.weights = [goal['weight'] for goal in server.goals]
        server.cum_weights = np.cumsum(server.weights)
        server.user_sessions.clear()
        return manager, raw
    from types import SimpleNamespace
    from agent_system.environments.env_manager import AlfWorldEnvironmentManager
    from agent_system.environments.env_package.alfworld import alfworld_projection
    raw = ExactAlfworld(item)
    return AlfWorldEnvironmentManager(raw, alfworld_projection,
        SimpleNamespace(env=SimpleNamespace(history_length=history))), raw


def usage_cost(usage):
    prompt = int(usage['prompt_tokens']); output = int(usage['completion_tokens'])
    hit = int(usage.get('prompt_cache_hit_tokens', usage.get('prompt_tokens_details', {}).get('cached_tokens', 0)))
    miss = int(usage.get('prompt_cache_miss_tokens', prompt-hit))
    if min(prompt, output, hit, miss) < 0 or hit+miss != prompt: raise ValueError('Invalid API token accounting')
    return dict(prompt_tokens=prompt, completion_tokens=output, cache_hit_tokens=hit, cache_miss_tokens=miss,
                reasoning_tokens=int(usage.get('completion_tokens_details', {}).get('reasoning_tokens', 0)),
                off_peak_usd=(hit*RATES['cache_hit']+miss*RATES['cache_miss']+output*RATES['output'])/1e6,
                no_cache_off_peak_usd=(prompt*RATES['cache_miss']+output*RATES['output'])/1e6)


def parser_response(content):
    # Native reasoning is a separate API field, never a source of executable actions.
    return content if '<think>' in content else '<think></think>\n' + content


def complete(prompt, settings, ledger):
    # Read the explicitly supplied credential only inside the request process. Never log it.
    key = Path(settings['key_file']).read_text().strip()
    if not key.startswith('sk-') or '\n' in key: raise ValueError('Credential file must contain one API key')
    payload = dict(model=settings['model'], messages=[dict(role='user', content=prompt)],
                   max_tokens=settings['max_tokens'], stream=False,
                   thinking={'type': 'enabled' if settings['thinking'] else 'disabled'})
    if settings['thinking']: payload['reasoning_effort'] = settings['reasoning_effort']
    else: payload['temperature'] = settings['temperature']
    failed = []
    ledger.parent.mkdir(parents=True, exist_ok=True)
    def log_failure(attempt, started):
        with ledger.with_suffix('.errors.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(attempt=attempt+1, requested_at=datetime.fromtimestamp(started,timezone.utc).isoformat(),
                prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(), error=failed[-1], usage_available=False))+'\n')
    for attempt in range(settings.get('max_attempts', 5)):
        started = time.time()
        request = urllib.request.Request(API_URL, data=json.dumps(payload).encode(),
                  headers={'Content-Type': 'application/json', 'Authorization': 'Bearer '+key})
        try:
            with urllib.request.urlopen(request, timeout=settings['timeout']) as response:
                result = json.load(response)
            account = usage_cost(result['usage'])
            choice = result['choices'][0]; message = choice['message']
            content = message.get('content') or ''; reasoning = message.get('reasoning_content') or ''
            parser_text = parser_response(content)
            record = dict(request_id=result.get('id'), model=result.get('model'),
                system_fingerprint=result.get('system_fingerprint'), api_created=result.get('created'),
                requested_at=datetime.fromtimestamp(started,timezone.utc).isoformat(),
                elapsed_s=time.time()-started, finish_reason=choice['finish_reason'],
                content=content, reasoning_content=reasoning, parser_text=parser_text,
                prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                usage=result['usage'], accounting=account, prior_attempt_errors=failed)
            ledger.parent.mkdir(parents=True,exist_ok=True)
            with ledger.open('a') as stream: stream.write(json.dumps(record,ensure_ascii=False)+'\n')
            return record
        except urllib.error.HTTPError as error:
            failed.append(dict(http_status=error.code, elapsed_s=time.time()-started))
            log_failure(attempt, started)
            if error.code not in [429,500,502,503,504] or attempt + 1 >= settings.get('max_attempts', 5):
                raise RuntimeError(f'API HTTP {error.code}; response body omitted') from None
        except (urllib.error.URLError,TimeoutError) as error:
            failed.append(dict(error_type=type(error).__name__,elapsed_s=time.time()-started))
            log_failure(attempt, started)
            if attempt + 1 >= settings.get('max_attempts', 5):
                raise RuntimeError('API network request failed after configured attempts') from None
        time.sleep(min(30, 2 ** (attempt + 1)))
    raise RuntimeError('No API response')


def run_episode(item, settings):
    runtime_setup()
    output = Path(settings['output']); path = output/'episodes'/f'{item["episode_id"]:04d}.json'
    record = dict(item, status='running', benchmark=settings['benchmark'], turns=[],
                  started=datetime.now(timezone.utc).isoformat())
    manager = raw = None
    try:
        manager, raw = environment(settings['benchmark'], item, settings['history_length'])
        observations, infos = manager.reset({})
        record['initial_observation'] = observations['anchor'][0]
        record['initial_info'] = plain(infos[0])
        if settings['benchmark'] == 'alfworld':
            got = infos[0].get('extra.gamefile')
            if got is None or Path(got).resolve() != (ROOT/item['gamefile']).resolve():
                raise ValueError('Actual ALFWorld task differs from selected task')
        else:
            record['initial_session'] = plain(raw.details())
            base = raw.worker.env.unwrapped
            record['test_goals_sha256'] = fingerprint(plain(base.server.goals[:500]))
            expected = settings.get('webshop_test_goals_sha256')
            if expected and record['test_goals_sha256'] != expected:
                raise ValueError('WebShop test goal manifest differs from preflight')
            if record['initial_session']['goal'] != plain(base.server.goals[item['goal_index']]):
                raise ValueError('Actual WebShop goal differs from selected task')
        atomic_json(path,record)
        for turn in range(settings['max_steps']):
            prompt = observations['text'][0]
            result = complete(prompt, settings, output/'api_calls'/f'{item["episode_id"]:04d}.jsonl')
            before = observations['anchor'][0]
            before_session = plain(raw.details()) if settings['benchmark'] == 'webshop' else None
            observations,reward,done,infos = manager.step([result['parser_text']])
            info=infos[0]
            record['turns'].append(dict(turn=turn,prompt=prompt,observation=before,
                next_observation=observations['anchor'][0],action=raw.last_action,
                reward=float(reward[0]),score=float(info.get('task_score',bool(info.get('won',False)))),
                done=bool(done[0]), env_info=plain(info), state_before_action=before_session, legacy_format_valid=bool(info['is_action_valid']),
                action_tag_valid=bool(re.search(r'<action>\s*[^<>]+\s*</action>',result['content'],re.I)),
                **result))
            atomic_json(path,plain(record))
            print(json.dumps(dict(benchmark=settings['benchmark'],episode=item['episode_id'],turn=turn+1,
                action=raw.last_action,done=bool(done[0]),cost=result['accounting']['off_peak_usd'])),flush=True)
            if done[0]: break
        turns=record['turns']
        record.update(status='completed',success=any(t['reward']==10. for t in turns),
                      score=max((t['score'] for t in turns),default=0.),length=len(turns),
                      horizon_truncated=not turns[-1]['done'],ended=datetime.now(timezone.utc).isoformat())
        record['termination'] = ('success' if record['success'] else
                                 'action_limit' if len(turns) >= settings['max_steps'] else 'environment_done')
        if settings['benchmark']=='webshop':
            record['final_session']=plain(raw.details())
            # WebShop resets itself before returning a terminal purchase: retain the pre-action state.
            record['purchase_state'] = next((t['state_before_action'] for t in turns if t['done']), None)
    except Exception as error:
        record.update(status='error',error=f'{type(error).__name__}: {error}',length=len(record['turns']))
    finally:
        if raw is not None:
            try: raw.close()
            except Exception as error: record['cleanup_error']=type(error).__name__
    atomic_json(path,plain(record))
    return plain(record)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--benchmark',required=True,choices=['alfworld','webshop'])
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--key-file',type=Path,default=ROOT/'baselines/deepseek/DEEPSEEK-API-KEY.txt')
    p.add_argument('--model',default='deepseek-flash',choices=['deepseek-flash'])
    p.add_argument('--episodes',type=int,default=6);p.add_argument('--workers',type=int,default=2)
    p.add_argument('--seed',type=int,default=0);p.add_argument('--history-length',type=int,default=2)
    p.add_argument('--max-steps',type=int);p.add_argument('--max-tokens',type=int,default=4096)
    p.add_argument('--reasoning-effort',default='high',choices=['low','high','max'])
    p.add_argument('--no-thinking',action='store_true');p.add_argument('--temperature',type=float,default=.4)
    p.add_argument('--timeout',type=float,default=900);p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--max-attempts',type=int,default=5)
    p.add_argument('--webshop-goal-manifest',type=Path)
    args=p.parse_args();runtime_setup()
    full=140 if args.benchmark=='alfworld' else 500
    if not 1<=args.episodes<=full or args.workers<1 or args.max_tokens<1 or args.max_attempts<1:
        p.error('Invalid episode/worker/token/attempt count')
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    if (output/'config.json').exists():p.error('Use a new output directory; automatic paid retries/resume are disabled')
    items,counts=plan(args.benchmark,args.episodes,args.seed)
    settings=dict(benchmark=args.benchmark,output=str(output),key_file=str(args.key_file.resolve()),
        model=args.model,history_length=args.history_length,max_steps=args.max_steps or (50 if args.benchmark=='alfworld' else 15),
        max_tokens=args.max_tokens,thinking=not args.no_thinking,reasoning_effort=args.reasoning_effort,
        temperature=None if not args.no_thinking else args.temperature,timeout=args.timeout,seed=args.seed,
        max_attempts=args.max_attempts)
    if args.webshop_goal_manifest:
        manifest=json.loads(args.webshop_goal_manifest.read_text())
        if manifest['seed'] != args.seed: p.error('Goal manifest seed mismatch')
        settings['webshop_test_goals_sha256']=manifest['test_goals_sha256']
    # Record only the credential source path, never the secret value or its hash.
    record=dict(settings=settings,episodes=items,full_population=counts,workers=args.workers,
        requested_full_count=full,created=datetime.now(timezone.utc).isoformat(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        rates_usd_per_million_off_peak=RATES,price_source='https://api-docs.deepseek.com/quick_start/pricing/',
        sampling=f'full census; order seed {args.seed}' if args.episodes == full else f'stratified/random subset; seed {args.seed}',
        prompt_policy=f'same benchmark prompt, history {args.history_length}; no tokenizer truncation; only content enters action parser',
        parser_native_reasoning_excluded=True,
        official_defaults_pinned=(not args.no_thinking and args.reasoning_effort=='high' and args.max_tokens==65536),
        generation_seed_supported=False)
    atomic_json(output/'config.json',record)
    if args.prepare_only:return 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        jobs=[pool.submit(run_episode,item,settings) for item in items]
        results=[job.result() for job in concurrent.futures.as_completed(jobs)]
    completed=[r for r in results if r['status']=='completed']
    metrics=dict(status='completed' if len(completed)==len(items) else 'incomplete',
        requested=len(items),completed=len(completed),errors=len(results)-len(completed),
        observed_success_rate=sum(r['success'] for r in completed)/len(completed) if completed else None,
        mean_length=sum(r['length'] for r in completed)/len(completed) if completed else None)
    atomic_json(output/'metrics.json',metrics);print(json.dumps(metrics),flush=True)
    return 0 if metrics['status']=='completed' else 1


if __name__=='__main__':raise SystemExit(main())
