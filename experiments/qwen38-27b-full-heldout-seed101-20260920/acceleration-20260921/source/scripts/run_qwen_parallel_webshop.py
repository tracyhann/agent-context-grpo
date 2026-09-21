#!/usr/bin/env python3
"""Run the queued WebShop census on an optimized replica; let the old chain join it."""
import argparse
from datetime import datetime,timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from qwen_baseline.common import MODEL,SERVED_MODEL,atomic_json
from qwen_eval_coordination import identity,protocol,MARKER


def stamp():return datetime.now(timezone.utc).isoformat()


def accelerated_command(original,base_url,workers):
    command=list(original)
    command[command.index('--workers')+1]=str(workers)
    if '--base-url' in command:command[command.index('--base-url')+1]=base_url
    else:command+=['--base-url',base_url]
    return command+['--managed-owner']


def stop_owned(pid,start_ticks,timeout=45):
    if identity(pid)!=start_ticks:return
    os.killpg(pid,signal.SIGTERM)
    deadline=time.monotonic()+timeout
    while identity(pid)==start_ticks and time.monotonic()<deadline:time.sleep(1)
    if identity(pid)==start_ticks:os.killpg(pid,signal.SIGKILL)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('experiment',type=Path)
    parser.add_argument('--workers',type=int,default=16)
    args=parser.parse_args();exp=args.experiment.resolve();work=exp/'acceleration-20260921'
    if args.workers<1:parser.error('workers must be positive')
    lock=(work/'supervisor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (work/'STATE.json').exists():raise RuntimeError('Existing acceleration state; inspect before relaunch')
    plan=json.loads((work/'PLAN.json').read_text());server_pid=int((work/'server.pid').read_text());server_start=identity(server_pid)
    if not server_start:raise RuntimeError('Optimized server is not alive')
    cmdline=Path(f'/proc/{server_pid}/cmdline').read_bytes().decode().split('\0')
    if '--port' not in cmdline or cmdline[cmdline.index('--port')+1]!=str(plan['new_server_port']):
        raise RuntimeError('Optimized server identity/port mismatch')
    base_url=f"http://127.0.0.1:{plan['new_server_port']}/v1"
    with urllib.request.urlopen(base_url+'/models',timeout=5) as response:
        if SERVED_MODEL not in [m['id'] for m in json.load(response)['data']]:raise RuntimeError('Wrong model on optimized endpoint')
    manifest=json.loads((exp/'chain.json').read_text())
    suite=next(s for s in manifest['evaluations'] if s['name']=='webshop')
    output=Path(suite['command'][suite['command'].index('--output')+1]);output.mkdir(parents=True,exist_ok=True)
    if (output/'config.json').exists() or (output/MARKER).exists():raise RuntimeError('WebShop already started or managed; refusing duplicate')
    # Verify all originally pinned inputs, except the explicitly revised scheduler entrypoint.
    allowed={'scripts/qwen_census_eval.py'};expected=json.loads((exp/'input-sha256.json').read_text());changes={}
    for name,digest in expected.items():
        current=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
        if current!=digest:
            if name not in allowed:raise RuntimeError(f'Unexpected prepared input change: {name}')
            changes[name]=dict(original=digest,accelerated=current)
    command=accelerated_command(suite['command'],base_url,args.workers)
    protocol_args=argparse.Namespace(suite='webshop',seed=101,max_tokens=65536,max_model_len=131072,
                                   no_thinking=False,webshop_goal_manifest=exp/'webshop-goals-seed101.json')
    marker=dict(protocol=protocol(protocol_args),output=str(output),expected_count=500,status='prepared',
                coordinator_pid=os.getpid(),coordinator_start_ticks=identity(os.getpid()),
                base_url=base_url,workers=args.workers,command=command,original_command=suite['command'],
                authorized_change='Parallel scheduling plus compiled CUDA-graph serving; model and sampling protocol retained',created=stamp())
    sources={}
    for name in ['scripts/qwen_census_eval.py','scripts/qwen_eval_coordination.py','scripts/run_qwen_parallel_webshop.py',
                 'scripts/report_census_eval.py','scripts/benchmark_census.py','scripts/deepseek_api_eval.py','baselines/qwen/serve.py']:
        source=ROOT/name;target=work/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(source.read_bytes())
        sources[name]=hashlib.sha256(source.read_bytes()).hexdigest()
    atomic_json(work/'source-sha256.json',sources)
    atomic_json(work/'input-verification.json',dict(checked=stamp(),original_inputs=len(expected),allowed_changes=changes))
    atomic_json(output/MARKER,marker)
    state=dict(status='starting_webshop',started=stamp(),coordinator_pid=os.getpid(),server_pid=server_pid,
               server_start_ticks=server_start,gpus=plan['gpus'],command=command,output=str(output),workers=args.workers)
    def save(**values):state.update(values,updated=stamp());atomic_json(work/'STATE.json',state)
    def terminate(signum,frame):raise SystemExit(f'Received signal {signum}')
    signal.signal(signal.SIGTERM,terminate);signal.signal(signal.SIGINT,terminate)
    job=None
    try:
        save()
        with (work/'webshop.log').open('ab') as log:
            job=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                                 env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1'),start_new_session=True)
        job_start=identity(job.pid);save(status='evaluating_webshop',evaluator_pid=job.pid,evaluator_start_ticks=job_start)
        while job.poll() is None:
            if identity(server_pid)!=server_start:raise RuntimeError('Optimized server exited during WebShop evaluation')
            save();time.sleep(20)
        metrics=json.loads((output/'metrics.json').read_text()) if (output/'metrics.json').exists() else None
        save(status='completed' if job.returncode==0 else 'incomplete',evaluator_returncode=job.returncode,metrics=metrics,ended=stamp())
        return job.returncode
    except BaseException as error:
        save(status='failed',error=f'{type(error).__name__}: {error}',ended=stamp())
        raise
    finally:
        if job is not None and job.poll() is None:
            stop_owned(job.pid,job_start);job.wait(timeout=15)
        stop_owned(server_pid,server_start)
        save(server_stopped=stamp())


if __name__=='__main__':raise SystemExit(main())
