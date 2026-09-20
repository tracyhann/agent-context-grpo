#!/usr/bin/env python3
"""Keep API evaluation progress and final results documented after terminal disconnects."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import time
from chain_qwen_census import alive
from report_census_eval import report
from qwen_baseline.common import atomic_json

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('experiment',type=Path);a=p.parse_args();exp=a.experiment.resolve()
    state=json.loads((exp/'RUN_STATE.json').read_text());pid=state['pid']
    fields=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split();start=fields[19]
    while True:
        result=report(exp);running=alive(pid,start)
        state.update(updated=datetime.now(timezone.utc).isoformat(),progress={n:f"{s['completed']}/{s['expected']}" for n,s in result['suites'].items()},
            cost_usd=result['cost_usd'],status='running' if running else result['status'])
        if not running:
            state['ended']=datetime.now(timezone.utc).isoformat()
            atomic_json(exp/'RUN_STATE.json',state)
            atomic_json(exp/'FINAL_AUDIT.json',dict(status='passed' if result['status']=='completed' else 'incomplete',checked=state['ended'],
                suite_counts={n:s['completed'] for n,s in result['suites'].items()},
                checks=['Complete unique planned task census','Actual environment task identity','Received request/trajectory reconciliation','Usage and cost accounting']))
            return 0 if result['status']=='completed' else 1
        atomic_json(exp/'RUN_STATE.json',state);time.sleep(30)
if __name__=='__main__':raise SystemExit(main())
