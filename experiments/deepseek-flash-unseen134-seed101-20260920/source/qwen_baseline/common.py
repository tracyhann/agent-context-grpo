"""Shared protocol, HTTP inference and result accounting (no model imports)."""
import hashlib
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL = json.loads((Path(__file__).parent / 'model.json').read_text())
SERVED_MODEL = f"{MODEL['repo_id']}@{MODEL['revision']}"
PROFILES = {
    'alfworld': dict(episodes=128, logical_batch_size=32, max_steps=50, max_prompt_tokens=2048,
                    split='eval_in_distribution'),
    'webshop': dict(episodes=256, logical_batch_size=128, max_steps=15, max_prompt_tokens=4096,
                   split='test-goals-0-499'),
}


def snapshot_path():
    path = ROOT / 'hf/hub' / ('models--' + MODEL['repo_id'].replace('/', '--')) / 'snapshots' / MODEL['revision']
    if not (path / 'model.safetensors.index.json').is_file():
        raise FileNotFoundError('Run download_model.py first; the pinned checkpoint is missing')
    return path


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def source_hashes(benchmark):
    """Pin the adapter, prompt, parser and scorer actually used by this checkout."""
    paths = list((ROOT / 'qwen_baseline').glob('*.py'))
    envs = ROOT / 'verl-agent/agent_system/environments'
    paths += [envs / 'base.py', envs / 'env_manager.py', envs / 'prompts' / f'{benchmark}.py']
    paths += list((envs.parent / 'memory').glob('*.py'))
    package = envs / 'env_package' / benchmark
    paths += list(package.glob('*.py'))
    if benchmark == 'webshop':
        site = package / 'webshop/web_agent_site'
        paths += list(site.rglob('*.py'))
        paths += [package / 'webshop/data' / name for name in
                  ['items_shuffle_1000.json', 'items_ins_v2_1000.json']]
    else:
        paths += [package / 'configs/config_tw.yaml']
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(set(paths))}


def episode_plan(benchmark, count=None, env_seed=1000, eval_round=0):
    """Replay logical validation batches independently of execution concurrency."""
    import numpy as np
    profile = PROFILES[benchmark]
    size = profile['logical_batch_size']
    count = profile['episodes'] if count is None else count
    if not 1 <= count <= profile['episodes'] or eval_round < 0:
        raise ValueError('Episode count must be within the reference eval size; eval_round >= 0')
    batches = (profile['episodes'] + size - 1) // size
    rng = np.random.RandomState(env_seed)
    goals = []
    if benchmark == 'webshop':
        for _ in range(eval_round * batches):
            rng.choice(range(500), size=size, replace=False)
        goals = [rng.choice(range(500), size=size, replace=False).tolist() for _ in range(batches)]
    return [dict(episode_id=i, slot=i % size, batch=i // size,
                 worker_seed=env_seed + i % size,
                 reset_index=eval_round * batches + i // size,
                 goal_index=goals[i // size][i % size] if goals else None)
            for i in range(count)]


def post_json(base_url, endpoint, payload, timeout=180, attempts=3):
    url = base_url.rstrip('/') + '/' + endpoint.lstrip('/')
    body = json.dumps(payload).encode()
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(url, data=body, headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            # Malformed requests and context overflow are never repaired silently.
            if error.code not in (429, 500, 502, 503, 504) or attempt + 1 == attempts:
                detail = error.read().decode(errors='replace')[:1000]
                raise RuntimeError(f'Inference HTTP {error.code}: {detail}') from error
        except (urllib.error.URLError, TimeoutError):
            if attempt + 1 == attempts:
                raise
        time.sleep(2 ** attempt)


def complete(prompt_ids, settings, seed):
    payload = dict(model=SERVED_MODEL, prompt=prompt_ids, max_tokens=settings['max_tokens'],
                   temperature=settings['temperature'], top_p=1.0, top_k=-1,
                   presence_penalty=0.0, frequency_penalty=0.0, repetition_penalty=1.0,
                   seed=seed, stream=False)
    result = post_json(settings['base_url'], 'completions', payload, settings['timeout'])
    choices = result.get('choices', [])
    if len(choices) != 1 or not isinstance(choices[0].get('text'), str):
        raise ValueError('Inference server did not return one text completion')
    return dict(text=choices[0]['text'], finish_reason=choices[0].get('finish_reason'),
                usage=result.get('usage', {}), request_seed=seed)


def aggregate(records, expected):
    completed = [r for r in records if r['status'] == 'completed']
    errors = [r for r in records if r['status'] == 'error']
    n = len(completed)
    result = dict(expected_episodes=expected, completed_episodes=n, error_episodes=len(errors),
                  status='completed' if n == expected and not errors else 'incomplete',
                  success_rate=None, score=None)
    if n:
        result.update(observed_success_rate=sum(r['success'] for r in completed) / n,
                      observed_score=sum(r['score'] for r in completed) / n,
                      mean_length=sum(r['length'] for r in completed) / n)
        turns = [t for r in completed for t in r['turns']]
        result['legacy_valid_action_ratio'] = sum(t['legacy_format_valid'] for t in turns) / len(turns) if turns else None
        result['action_tag_valid_ratio'] = sum(t['action_tag_valid'] for t in turns) / len(turns) if turns else None
        result['response_truncation_ratio'] = sum(t['finish_reason'] == 'length' for t in turns) / len(turns) if turns else None
        if result['status'] == 'completed':
            result['success_rate'] = result['observed_success_rate']
            result['score'] = result['observed_score']
    return result
