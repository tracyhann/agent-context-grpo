#!/usr/bin/env python3
"""Evaluate pinned local Qwen on a complete held-out split, with audited task traces."""
import argparse
import concurrent.futures
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing
from pathlib import Path
import time
import urllib.request
import deepseek_api_eval as evaluator
import benchmark_census as census
from qwen_baseline.common import MODEL, SERVED_MODEL, post_json, source_hashes


def split_native_thinking(text, enabled):
    if not enabled:
        return text, ''
    # The native template already prefills <think>. An unfinished reasoning stream
    # must never be interpreted as an environment action, even if it quotes a tag.
    if '</think>' not in text:
        return '', text
    reasoning, content = text.split('</think>', 1)
    return content.lstrip(), reasoning.removeprefix('<think>').lstrip()


def complete(prompt, settings, ledger):
    started = time.time()
    kwargs = dict(enable_thinking=settings['thinking'], preserve_thinking=True,
                  reasoning_effort=settings['reasoning_effort'])
    tokenized = post_json(settings['base_url'].removesuffix('/v1'), 'tokenize',
        dict(model=SERVED_MODEL, messages=[dict(role='user', content=prompt)],
             add_generation_prompt=True, chat_template_kwargs=kwargs), timeout=120)
    ids = tokenized['tokens']
    if len(ids) + settings['max_tokens'] > settings['max_model_len']:
        raise ValueError('Prompt plus output budget exceeds context; refusing silent truncation')
    turn = settings['_turn']; settings['_turn'] += 1
    seed = settings['seed'] + settings['suite_seed_offset'] + settings['_episode_id'] * 1000 + turn
    payload = dict(model=SERVED_MODEL, prompt=ids, max_tokens=settings['max_tokens'],
        temperature=settings['temperature'], top_p=settings['top_p'], top_k=20, min_p=0.0,
        presence_penalty=settings['presence_penalty'], frequency_penalty=0.0, repetition_penalty=1.0,
        seed=seed, stream=False, skip_special_tokens=False)
    result = post_json(settings['base_url'], 'completions', payload, timeout=settings['timeout'])
    if len(result['choices']) != 1:
        raise ValueError('Expected exactly one generated continuation')
    choice = result['choices'][0]; text = choice['text']
    content, reasoning = split_native_thinking(text, settings['thinking'])
    usage = result['usage']
    prompt_tokens = int(usage['prompt_tokens']); output_tokens = int(usage['completion_tokens'])
    if prompt_tokens != len(ids):
        raise ValueError('Server token count differs from exact template token IDs')
    record = dict(request_id=result.get('id'), model=result.get('model'),
        system_fingerprint=result.get('system_fingerprint'), api_created=result.get('created'),
        requested_at=datetime.fromtimestamp(started, timezone.utc).isoformat(),
        elapsed_s=time.time()-started, finish_reason=choice['finish_reason'],
        content=content, reasoning_content=reasoning, raw_completion=text,
        native_reasoning_finished=(not settings['thinking'] or '</think>' in text),
        parser_text=evaluator.parser_response(content), request_seed=seed,
        prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
        template_token_ids_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
        usage=usage, accounting=dict(prompt_tokens=prompt_tokens, completion_tokens=output_tokens,
            cache_hit_tokens=0, cache_miss_tokens=prompt_tokens, reasoning_tokens=None,
            off_peak_usd=0.0, local_api_cost_usd=0.0), prior_attempt_errors=[])
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open('a') as stream:
        stream.write(json.dumps(record, ensure_ascii=False)+'\n')
    return record


def worker(item, settings):
    evaluator.environment = census.environment
    evaluator.complete = complete
    settings = dict(settings, _turn=0, _episode_id=item['episode_id'])
    return evaluator.run_episode(item, settings)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite', choices=['alfworld-seen', 'alfworld-unseen', 'webshop'], required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed', type=int, default=101)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--base-url', default='http://127.0.0.1:8018/v1')
    p.add_argument('--max-tokens', type=int, default=65536)
    p.add_argument('--max-model-len', type=int, default=131072)
    p.add_argument('--no-thinking', action='store_true')
    p.add_argument('--webshop-goal-manifest', type=Path)
    p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--managed-owner', action='store_true', help='Own an explicitly coordinated parallel census')
    a = p.parse_args()
    from qwen_eval_coordination import claim_owner, finish_owner, join_if_managed
    owner_lock = None  # Keep the exclusive lock alive through all worker processes.
    if a.managed_owner:
        if a.prepare_only: p.error('Managed ownership requires actual evaluation')
        owner_lock = claim_owner(a)
    else:
        joined = join_if_managed(a)
        if joined is not None: return joined
    evaluator.runtime_setup()
    benchmark = 'webshop' if a.suite == 'webshop' else 'alfworld'
    split = 'unseen' if a.suite.endswith('-unseen') else 'seen'
    items, counts = census.plan(benchmark, a.seed, split)
    output = a.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    if (output/'config.json').exists(): p.error('Use a new output folder; preserve previous records')
    settings = dict(benchmark=benchmark, output=str(output), model=SERVED_MODEL, seed=a.seed,
        suite_seed_offset={'alfworld-seen':0, 'alfworld-unseen':1000000, 'webshop':2000000}[a.suite],
        alfworld_split=split if benchmark=='alfworld' else None,
        history_length=2, max_steps=50 if benchmark=='alfworld' else 15,
        thinking=not a.no_thinking, reasoning_effort='xhigh' if not a.no_thinking else None,
        max_tokens=a.max_tokens, max_model_len=a.max_model_len, base_url=a.base_url,
        temperature=1.0 if not a.no_thinking else .7, top_p=.95 if not a.no_thinking else .8,
        presence_penalty=0.0 if not a.no_thinking else 1.5, timeout=7200)
    if benchmark == 'webshop':
        if not a.webshop_goal_manifest: p.error('WebShop requires the frozen goal/price manifest')
        manifest = json.loads(a.webshop_goal_manifest.read_text())
        if manifest['seed'] != a.seed: p.error('WebShop manifest seed mismatch')
        settings['webshop_test_goals_sha256'] = manifest['test_goals_sha256']
    hashes = source_hashes(benchmark)
    for name in ['qwen_census_eval.py','qwen_eval_coordination.py','benchmark_census.py','deepseek_api_eval.py']:
        hashes['scripts/'+name] = hashlib.sha256((evaluator.ROOT/'scripts'/name).read_bytes()).hexdigest()
    config = dict(provider='local-qwen', benchmark=benchmark, split=split if benchmark=='alfworld' else 'test',
        settings=settings, episodes=items, full_population=counts, requested_full_count=len(items),
        model=MODEL, workers=a.workers, created=datetime.now(timezone.utc).isoformat(), source_sha256=hashes,
        generation_seed_formula='101 + suite_offset + episode_id * 1000 + turn_index',
        reasoning_policy='native xhigh; exact checkpoint template; retain final content only for actions',
        output_budget_policy='Explicit experimental 65536-token cap, not a claimed universal Qwen default',
        scoring='Unmodified environment scorer; failures and action-limit truncations remain in denominator')
    evaluator.atomic_json(output/'config.json', config)
    if a.prepare_only: return 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        results = list(pool.map(worker, items, [settings]*len(items)))
    done = [r for r in results if r['status']=='completed']
    metrics = dict(status='completed' if len(done)==len(items) else 'incomplete', requested=len(items),
        completed=len(done), errors=len(results)-len(done),
        observed_success_rate=sum(r['success'] for r in done)/len(done) if done else None,
        observed_score=sum(r['score'] for r in done)/len(done) if done else None)
    evaluator.atomic_json(output/'metrics.json', metrics); print(json.dumps(metrics), flush=True)
    if a.managed_owner: finish_owner(a, metrics)
    return 0 if metrics['status']=='completed' else 1


if __name__ == '__main__': raise SystemExit(main())
