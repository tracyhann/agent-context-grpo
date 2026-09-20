#!/usr/bin/env python3
"""Durably wait for successful training, then run the three Qwen held-out censuses."""
import argparse
import csv
from datetime import datetime,timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.request
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from qwen_baseline.common import atomic_json
from report_census_eval import report

ROOT=Path(__file__).resolve().parents[1]


def stamp():return datetime.now(timezone.utc).isoformat()


def alive(pid,start=None):
    try:
        fields=Path(f'/proc/{int(pid)}/stat').read_text().rsplit(')',1)[1].split()
        return fields[0] not in {'Z','X'} and (start is None or fields[19]==str(start))
    except (OSError,ValueError,TypeError):return False


def verify_training(directory,total=150):
    directory=Path(directory)
    rows=[json.loads(x) for x in (directory/'outputs/metrics.jsonl').read_text().splitlines() if x]
    if rows[-1]['step']!=total or not math.isfinite(rows[-1].get('val/success_rate',float('nan'))):
        raise RuntimeError('Missing final training validation')
    base=directory/'outputs/checkpoints';best=json.loads((base/'best.json').read_text())
    for name in [f'global_step_{total}',f'step{total}-last','step100-pin',f"step{best['step']}-best"]:
        for rank in range(2):
            for kind in ['model','optim','extra_state']:
                p=base/name/'actor'/f'{kind}_world_size_2_rank_{rank}.pt'
                if not p.is_file() or not p.stat().st_size:raise RuntimeError(f'Missing checkpoint shard {p}')
    return dict(step=total,final_success_rate=rows[-1]['val/success_rate'],best=best)


def capacity(manifest):
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=20)
    selected=[]
    for row in csv.reader(raw.splitlines()):
        idx,uuid,memory,util=[s.strip() for s in row]
        if int(idx) not in manifest['gpus']:continue
        if uuid!=manifest['gpu_uuids'][idx]:raise RuntimeError('GPU UUID changed')
        selected.append(dict(index=int(idx),uuid=uuid,memory_mib=int(memory),utilization=int(util)))
    if len(selected)!=len(manifest['gpus']):raise RuntimeError('GPU inventory incomplete')
    return all(g['memory_mib']<=2048 and g['utilization']<=10 for g in selected),selected


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('experiment',type=Path);a=p.parse_args();exp=a.experiment.resolve()
    manifest=json.loads((exp/'chain.json').read_text())
    lock=(exp/'chain.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (exp/'RUN_STATE.json').exists():raise RuntimeError('Existing chain state; refusing duplicate launch')
    state=dict(status='waiting_for_training',controller_pid=os.getpid(),started=stamp(),gpus=manifest['gpus'],suites={})
    def save(**values):state.update(values,updated=stamp());atomic_json(exp/'RUN_STATE.json',state)
    save();server=None
    try:
        dep=manifest['dependency']
        while True:
            current=json.loads(Path(dep['controller_state']).read_text())
            run=next(r for r in current['runs'] if r['id']==dep['run_id'])
            save(training_status=current['status'],training_step=run.get('step'),training_eval_step=run.get('eval_step'),training_eval_success=run.get('eval_success_rate'))
            if current['status']=='failed' or run['status']=='failed':raise RuntimeError('Predecessor training failed; evaluation was not launched')
            if current['status']=='complete' and run['status']=='complete' and not alive(dep['pid'],dep['start_ticks']):
                save(training_verified=verify_training(dep['directory']));break
            time.sleep(15)
        expected=json.loads((exp/'input-sha256.json').read_text())
        for name,digest in expected.items():
            if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:raise RuntimeError(f'Prepared input changed: {name}')
        idle=0
        while idle<2:
            ready,gpus=capacity(manifest);idle=idle+1 if ready else 0
            save(status='waiting_for_gpu_idle',gpu_capacity=gpus,idle_samples=idle)
            if idle<2:time.sleep(15)
        with socket.socket() as sock:sock.bind(('127.0.0.1',manifest['port']))
        with (exp/'server.log').open('ab') as log:
            server=subprocess.Popen(manifest['server_command'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        save(status='starting_server',server_pid=server.pid,server_started=stamp())
        deadline=time.monotonic()+1200
        while True:
            if server.poll() is not None:raise RuntimeError(f'Qwen server exited {server.returncode}')
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{manifest['port']}/health",timeout=5) as response:
                    if response.status==200:break
            except OSError:pass
            if time.monotonic()>deadline:raise RuntimeError('Qwen server startup timeout')
            save();time.sleep(10)
        save(status='evaluating',server_ready=stamp())
        for suite in manifest['evaluations']:
            name=suite['name']
            with (exp/(name+'.log')).open('ab') as log:
                job=subprocess.Popen(suite['command'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
            state['suites'][name]=dict(status='running',pid=job.pid,started=stamp());save(current_suite=name)
            while job.poll() is None:
                result=report(exp);save(progress={n:f"{s['completed']}/{s['expected']}" for n,s in result['suites'].items()})
                time.sleep(20)
            state['suites'][name].update(status='completed' if job.returncode==0 else 'incomplete',returncode=job.returncode,ended=stamp());save()
            if server.poll() is not None:raise RuntimeError('Qwen server stopped during evaluation')
        result=report(exp);save(status=result['status'],ended=stamp(),current_suite=None)
        atomic_json(exp/'FINAL_AUDIT.json',dict(status='passed' if result['status']=='completed' else 'incomplete',
            checked=stamp(),suite_counts={n:s['completed'] for n,s in result['suites'].items()},
            checks=['Full unique task populations','Exact planned/actual task identities','Complete API request/trajectory reconciliation','Token accounting','Final tables require all episodes'],
            source_inputs_verified_before_launch=True))
    except Exception as error:
        save(status='failed',error=f'{type(error).__name__}: {error}',ended=stamp())
        raise
    finally:
        if server is not None:
            if server.poll() is None:
                os.killpg(server.pid,signal.SIGTERM)
                try:server.wait(timeout=45)
                except subprocess.TimeoutExpired:os.killpg(server.pid,signal.SIGKILL);server.wait(timeout=15)
            save(server_returncode=server.returncode,server_stopped=stamp())
        report(exp)


if __name__=='__main__':main()
