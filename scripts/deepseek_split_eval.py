#!/usr/bin/env python3
"""Full ALFWorld split evaluation with the previously audited DeepSeek protocol."""
import argparse
import concurrent.futures
from datetime import datetime,timezone
import hashlib
import http.client
import json
import multiprocessing
from pathlib import Path
import urllib.error
import deepseek_api_eval as evaluator
import benchmark_census as census


def worker(item,settings):
    evaluator.environment=census.environment
    original=evaluator.urllib.request.urlopen
    class Response:
        def __init__(self,response):self.response=response
        def __enter__(self):return self
        def __exit__(self,*args):self.response.close()
        def read(self):
            try:
                data=self.response.read();json.loads(data);return data
            except (http.client.HTTPException,OSError,json.JSONDecodeError) as error:
                raise urllib.error.URLError(type(error).__name__) from None
    def safe_urlopen(*args,**kwargs):
        try:return Response(original(*args,**kwargs))
        except http.client.HTTPException as error:raise urllib.error.URLError(type(error).__name__) from None
    evaluator.urllib.request.urlopen=safe_urlopen
    try:return evaluator.run_episode(item,settings)
    finally:evaluator.urllib.request.urlopen=original


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--split',choices=census.SPLITS,default='unseen');p.add_argument('--seed',type=int,default=101)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=32);p.add_argument('--prepare-only',action='store_true')
    a=p.parse_args();evaluator.runtime_setup();items,counts=census.plan('alfworld',a.seed,a.split)
    output=a.output.resolve();output.mkdir(parents=True,exist_ok=True)
    if (output/'config.json').exists():p.error('Use a new output folder; preserve prior records')
    settings={'benchmark':'alfworld','output':str(output),'key_file':str(evaluator.ROOT/'baselines/deepseek/DEEPSEEK-API-KEY.txt'),
        'model':'deepseek-flash','history_length':2,'max_steps':50,'max_tokens':65536,'thinking':True,'reasoning_effort':'high',
        'temperature':None,'timeout':900,'seed':a.seed,'max_attempts':5,'alfworld_split':a.split}
    config={'provider':'deepseek','benchmark':'alfworld','split':a.split,'settings':settings,'episodes':items,'full_population':counts,'requested_full_count':len(items),
        'created':datetime.now(timezone.utc).isoformat(),'workers':a.workers,'generation_seed_supported':False,'official_defaults_pinned':True,
        'source_sha256':{name:hashlib.sha256((evaluator.ROOT/'scripts'/name).read_bytes()).hexdigest() for name in ['deepseek_split_eval.py','benchmark_census.py','deepseek_api_eval.py']}}
    evaluator.atomic_json(output/'config.json',config)
    if a.prepare_only:return 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        results=list(pool.map(worker,items,[settings]*len(items)))
    done=[r for r in results if r['status']=='completed']
    metrics={'status':'completed' if len(done)==len(items) else 'incomplete','requested':len(items),'completed':len(done),'errors':len(results)-len(done),
        'observed_success_rate':sum(r['success'] for r in done)/len(done) if done else None}
    evaluator.atomic_json(output/'metrics.json',metrics);print(json.dumps(metrics),flush=True)
    return 0 if metrics['status']=='completed' else 1

if __name__=='__main__':raise SystemExit(main())
