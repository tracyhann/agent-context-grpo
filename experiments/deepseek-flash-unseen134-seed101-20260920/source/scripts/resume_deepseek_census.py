#!/usr/bin/env python3
"""Resume only errored census episodes by replaying saved responses, after credit is restored."""
import argparse
import concurrent.futures
import json
import multiprocessing
from pathlib import Path
import urllib.request
import benchmark_census as census
import recover_deepseek_eval as recover


def worker(path):
    recover.evaluator.environment=census.environment
    return recover.recover_one(path)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('output',type=Path);p.add_argument('--workers',type=int,default=2);a=p.parse_args()
    settings=json.loads((a.output/'config.json').read_text())['settings']
    key=Path(settings['key_file']).read_text().strip()
    request=urllib.request.Request('https://api.deepseek.com/user/balance',headers={'Authorization':'Bearer '+key})
    with urllib.request.urlopen(request,timeout=30) as response:balance=json.load(response)
    if not balance['is_available']:
        print(json.dumps({'status':'blocked_insufficient_balance','new_generation_requests':0}));return 2
    errors=[p for p in sorted((a.output/'episodes').glob('*.json')) if json.loads(p.read_text())['status']=='error']
    print(json.dumps({'resuming_episode_ids':[int(p.stem) for p in errors]}),flush=True)
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        results=list(pool.map(worker,errors))
    print(json.dumps({'recovery_results':results}),flush=True)
    if all(r['status'] in ['completed','skipped'] for r in results):
        rows=[json.loads(p.read_text()) for p in (a.output/'episodes').glob('*.json')]
        config=json.loads((a.output/'config.json').read_text());done=[r for r in rows if r['status']=='completed']
        recover.evaluator.atomic_json(a.output/'metrics.json',dict(status='completed' if len(done)==len(config['episodes']) else 'incomplete',
            requested=len(config['episodes']),completed=len(done),errors=len(rows)-len(done),observed_success_rate=sum(r['success'] for r in done)/len(done)))
        return 0
    return 1

if __name__=='__main__':raise SystemExit(main())
