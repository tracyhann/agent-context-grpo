#!/usr/bin/env python3
"""Resume errored episodes by replaying saved API responses; never regenerate a saved turn."""
import argparse
import concurrent.futures
from datetime import datetime,timezone
import hashlib
import http.client
import json
import multiprocessing
from pathlib import Path
import shutil
import urllib.error
import deepseek_api_eval as evaluator


def recover_one(episode_path):
    episode_path=Path(episode_path);out=episode_path.parent.parent
    old=json.loads(episode_path.read_text());config=json.loads((out/'config.json').read_text())
    if old['status']!='error':return {'episode_id':old['episode_id'],'status':'skipped'}
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')
    archive=out/'recovery'/stamp;archive.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(episode_path,archive/episode_path.name)
    ledger=out/'api_calls'/f"{old['episode_id']:04d}.jsonl"
    cached=[json.loads(line) for line in ledger.read_text().splitlines()] if ledger.exists() else []
    assert len(cached)>=len(old['turns']), 'Missing saved API responses'
    original_complete=evaluator.complete;original_urlopen=evaluator.urllib.request.urlopen
    cursor=0;recorded_failure=False
    class SafeResponse:
        def __init__(self,response):self.response=response
        def __enter__(self):return self
        def __exit__(self,*args):self.response.close()
        def read(self):
            try:
                data=self.response.read()
                json.loads(data)  # Incomplete JSON is also a retryable transport failure.
                return data
            except (http.client.HTTPException,OSError,json.JSONDecodeError) as error:
                raise urllib.error.URLError(type(error).__name__) from None
    def safe_urlopen(*args,**kwargs):
        try:return SafeResponse(original_urlopen(*args,**kwargs))
        except http.client.HTTPException as error:
            raise urllib.error.URLError(type(error).__name__) from None
    def resumed_complete(prompt,settings,path):
        nonlocal cursor,recorded_failure
        digest=hashlib.sha256(prompt.encode()).hexdigest()
        if cursor<len(cached):
            r=cached[cursor]
            assert r['prompt_sha256']==digest,f'Replayed prompt differs at turn {cursor}'
            cursor+=1
            return r
        if not recorded_failure:
            # The original runner did not catch these transport exceptions; record their unpriced attempt.
            if old.get('error','').startswith(('IncompleteRead:','RemoteDisconnected:','JSONDecodeError:','ConnectionResetError:')):
                event={'attempt':1,'requested_at':datetime.now(timezone.utc).isoformat(),
                    'timestamp_is_recovery_time':True,'prompt_sha256':digest,'usage_available':False,
                    'error':{'error_type':old['error'].split(':',1)[0]},'source':'original_unhandled_transport_error',
                    'original_episode_archive':str(archive/episode_path.name)}
                with path.with_suffix('.errors.jsonl').open('a') as stream:stream.write(json.dumps(event)+'\n')
            recorded_failure=True
        return original_complete(prompt,settings,path)
    evaluator.complete=resumed_complete;evaluator.urllib.request.urlopen=safe_urlopen
    settings=dict(config['settings']);settings['timeout']=900;settings['max_attempts']=5
    item=next(x for x in config['episodes'] if x['episode_id']==old['episode_id'])
    try:
        result=evaluator.run_episode(item,settings)
        assert cursor==len(cached),'Some paid responses were not replayed'
        for a,b in zip(old['turns'],result['turns']):
            for key in ['prompt','observation','next_observation','action','reward','score','done','request_id']:
                assert a[key]==b[key],f'Replay mismatch in {key}'
        metadata={'episode_id':old['episode_id'],'benchmark':old['benchmark'],
            'status':result['status'],'replayed_requests':len(cached),'new_requests':len(result['turns'])-len(cached),
            'original_error':old.get('error'),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        evaluator.atomic_json(archive/'recovery.json',metadata)
        return metadata
    finally:
        evaluator.complete=original_complete;evaluator.urllib.request.urlopen=original_urlopen


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('output',type=Path);p.add_argument('--workers',type=int,default=8);a=p.parse_args()
    errors=[path for path in sorted((a.output/'episodes').glob('*.json')) if json.loads(path.read_text())['status']=='error']
    if not errors:print(json.dumps({'status':'nothing_to_recover'}));return 0
    print(json.dumps({'recovering_episode_ids':[int(x.stem) for x in errors]}),flush=True)
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        results=list(pool.map(recover_one,errors))
    print(json.dumps({'recovery_results':results}),flush=True)
    return 0 if all(x['status'] in ['completed','skipped'] for x in results) else 1

if __name__=='__main__':raise SystemExit(main())
