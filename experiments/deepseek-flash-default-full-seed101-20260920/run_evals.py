#!/usr/bin/env python3
"""Launch the authorized two-benchmark seed-101 census and refresh its result table."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
EXP=Path(__file__).resolve().parent


def save(path,data):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(path)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output-root',type=Path,default=EXP);p.add_argument('--workers',type=int,default=32);a=p.parse_args()
    dest=a.output_root.resolve();(dest/'outputs').mkdir(parents=True,exist_ok=True)
    if any((dest/'outputs'/b/'config.json').exists() for b in ['alfworld','webshop']):
        raise SystemExit('Output already contains an evaluation; use a new --output-root to reproduce.')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONHASHSEED='101',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
    children={};handles=[];state={'status':'running','supervisor_pid':os.getpid(),'started':datetime.now(timezone.utc).isoformat(),'processes':{}}
    for bench,n,venv in [('alfworld',140,'.venv'),('webshop',500,'.venv-webshop')]:
        cmd=[str(ROOT/venv/'bin/python'),str(ROOT/'scripts/deepseek_api_eval.py'),'--benchmark',bench,'--episodes',str(n),'--workers',str(a.workers),'--seed','101','--history-length','2','--max-steps','50' if bench=='alfworld' else '15','--max-tokens','65536','--reasoning-effort','high','--timeout','900','--max-attempts','5','--output',str(dest/'outputs'/bench)]
        if bench=='webshop':cmd+=['--webshop-goal-manifest',str(EXP/'webshop-goals-seed101.json')]
        handle=(dest/'outputs'/f'{bench}.log').open('a');handles.append(handle)
        child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT)
        children[bench]=child;state['processes'][bench]={'pid':child.pid,'command':cmd,'returncode':None}
        save(dest/'RUN_STATE.json',state)
    while True:
        for bench,child in children.items():state['processes'][bench]['returncode']=child.poll()
        finished=all(c.poll() is not None for c in children.values())
        state['updated_at']=datetime.now(timezone.utc).isoformat()
        if finished:
            state['status']='completed' if all(c.returncode==0 for c in children.values()) else 'incomplete'
            state['ended']=state['updated_at']
        save(dest/'RUN_STATE.json',state)
        report=subprocess.run([sys.executable,str(ROOT/'scripts/report_deepseek_full_eval.py'),str(dest)],cwd=ROOT,env=env,capture_output=True,text=True)
        print(report.stdout.strip() or report.stderr[-500:],flush=True)
        if finished:break
        time.sleep(30)
    for handle in handles:handle.close()
    return 0 if state['status']=='completed' else 1

if __name__=='__main__':raise SystemExit(main())
