#!/usr/bin/env python3
"""Validate and report complete benchmark censuses from raw trajectory/usage ledgers."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qwen_baseline.common import atomic_json

TASKS = [('Pick','pick_and_place_simple'),('Look','look_at_obj_in_light'),
    ('Clean','pick_clean_then_place_in_recep'),('Heat','pick_heat_then_place_in_recep'),
    ('Cool','pick_cool_then_place_in_recep'),('Pick2','pick_two_obj_and_place')]


def pricing_factor(stamp):
    d = datetime.fromtimestamp(stamp,timezone.utc) if isinstance(stamp,(int,float)) else datetime.fromisoformat(stamp)
    return 2 if d.weekday()<5 and (1<=d.hour<4 or 6<=d.hour<10) else 1


def jsonlines(path):
    for line in path.read_text().splitlines():
        try: yield json.loads(line)
        except json.JSONDecodeError: continue  # concurrent final append


def report(root):
    manifest=json.loads((root/'suites.json').read_text()); local=manifest['provider']=='local-qwen'
    result=dict(title=manifest['title'],provider=manifest['provider'],updated=datetime.now(timezone.utc).isoformat(),suites={})
    all_rows=[]
    for suite in manifest['suites']:
        name=suite['name'];out=root/'outputs'/name;config_path=out/'config.json'
        config=json.loads(config_path.read_text()) if config_path.exists() else None
        planned={x['episode_id']:x for x in config['episodes']} if config else {}
        episodes={}
        for p in sorted((out/'episodes').glob('*.json')):
            e=json.loads(p.read_text());eid=e['episode_id']
            if eid not in planned or any(e.get(k)!=v for k,v in planned[eid].items()):
                raise ValueError(f'{name}: task identity differs from plan ({eid})')
            if eid in episodes:raise ValueError('Duplicate episode')
            episodes[eid]=e
            row={k:e[k] for k in ['episode_id','status','task_type','gamefile','goal_index','success','score','length','termination','error'] if k in e}
            row.update(suite=name,trajectory_sha256=hashlib.sha256(p.read_bytes()).hexdigest())
            all_rows.append(row)
        completed=[e for e in episodes.values() if e['status']=='completed']
        errors=[e for e in episodes.values() if e['status']=='error']
        requests={};usage=Counter();finishes=Counter();models=Counter();unknown=[];rejected=[];cost=0.;hash_tokens={}
        for p in sorted((out/'api_calls').glob('*.jsonl')):
            if p.name.endswith('.errors.jsonl'):
                for error in jsonlines(p):
                    (rejected if error.get('error',{}).get('http_status') in {400,401,402,403,404,422,429} else unknown).append(error)
                continue
            for r in jsonlines(p):
                rid=r['request_id']
                if not rid or rid in requests:raise ValueError(f'{name}: missing/duplicate request ID')
                requests[rid]=r;a=r['accounting'];u=r['usage']
                if u['total_tokens']!=u['prompt_tokens']+u['completion_tokens']:raise ValueError('Invalid total usage')
                if a['cache_hit_tokens']+a['cache_miss_tokens']!=a['prompt_tokens']:raise ValueError('Invalid cached usage')
                for k in ['prompt_tokens','completion_tokens','cache_hit_tokens','cache_miss_tokens','reasoning_tokens']:
                    if a.get(k) is not None:usage[k]+=a[k]
                hash_tokens[r['prompt_sha256']]=u['prompt_tokens']
                finishes[r['finish_reason']]+=1;models[r['model']]+=1
                if not local:
                    amount=(a['cache_hit_tokens']*.003+a['cache_miss_tokens']*.15+a['completion_tokens']*.6)/1e6
                    if abs(amount-a['off_peak_usd'])>1e-9:raise ValueError('Cost mismatch')
                    cost+=amount*pricing_factor(r['api_created'])
        unpriced_ceiling=0.
        for e in unknown:
            n=hash_tokens.get(e.get('prompt_sha256'))
            if n is None:unpriced_ceiling=None;break
            unpriced_ceiling+=(n*.15+config['settings']['max_tokens']*.6)/1e6*pricing_factor(e['requested_at'])
        final=len(completed)==suite['count'] and not errors
        if final:
            ids=[t['request_id'] for e in completed for t in e['turns']]
            if len(set(ids))!=len(ids) or set(ids)!=set(requests):raise ValueError('Unreconciled received requests')
            if len(planned)!=suite['count'] or set(episodes)!=set(planned):raise ValueError('Incomplete population')
            identity='gamefile' if suite['benchmark']=='alfworld' else 'goal_index'
            if len({e[identity] for e in completed})!=suite['count']:raise ValueError('Repeated held-out task')
        stats=dict(benchmark=suite['benchmark'],split=suite.get('split'),expected=suite['count'],completed=len(completed),
            running=sum(e['status']=='running' for e in episodes.values()),errors=len(errors),final=final,
            successes=sum(e['success'] for e in completed),
            success_pct=100*sum(e['success'] for e in completed)/len(completed) if completed else None,
            score_pct=100*sum(e['score'] for e in completed)/len(completed) if completed else None,
            api_requests=len(requests),usage=dict(usage),cost_usd=cost,unpriced_attempts=len(unknown),
            rejected_api_attempts=len(rejected),unpriced_ceiling_estimate_usd=unpriced_ceiling,finish_reasons=dict(finishes),models=dict(models),
            mean_episode_length=sum(len(e['turns']) for e in completed)/len(completed) if completed else None)
        if suite['benchmark']=='alfworld':
            stats['task_types']={label:dict(count=sum(e['task_type']==task for e in completed),
                successes=sum(e['success'] for e in completed if e['task_type']==task),
                expected=config['full_population'][task] if config else None) for label,task in TASKS}
            for t in stats['task_types'].values():t['success_pct']=100*t['successes']/t['count'] if t['count'] else None
        result['suites'][name]=stats
    result['status']='completed' if all(s['final'] for s in result['suites'].values()) else ('incomplete' if any(s['errors'] for s in result['suites'].values()) else 'in_progress')
    result['cost_usd']=sum(s['cost_usd'] for s in result['suites'].values())
    result['unpriced_attempts']=sum(s['unpriced_attempts'] for s in result['suites'].values())
    result['rejected_api_attempts']=sum(s['rejected_api_attempts'] for s in result['suites'].values())
    bounds=[s['unpriced_ceiling_estimate_usd'] for s in result['suites'].values()]
    result['unpriced_ceiling_estimate_usd']=sum(bounds) if all(v is not None for v in bounds) else None
    lines=[f"# {manifest['title']}",'',f"Status: **{result['status']}**. Updated {result['updated']}.",'',
        'One round, seed 101. Percent units. Final cells remain pending until the entire split finishes.','',
        '| ALFWorld split | Pick | Look | Clean | Heat | Cool | Pick2 | All |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name,s in result['suites'].items():
        if s['benchmark']!='alfworld':continue
        vals=[s['task_types'][label]['success_pct'] for label,_ in TASKS]+[s['success_pct']]
        lines.append('| '+name+' | '+' | '.join(f'{v:.2f}' if s['final'] else 'pending' for v in vals)+' |')
    lines+=['','| WebShop | Score | Succ. |','|---|---:|---:|']
    for name,s in result['suites'].items():
        if s['benchmark']=='webshop':lines.append('| '+name+' | '+' | '.join(f'{v:.2f}' if s['final'] else 'pending' for v in [s['score_pct'],s['success_pct']])+' |')
    lines+=['','All is successes / tasks (micro-average). WebShop uses the original dense scorer and original option matching. All horizon failures remain in the denominator.','',
        '| Suite | Finished | Successful finished tasks | Calls | Output-capped calls | API cost USD |',
        '|---|---:|---:|---:|---:|---:|']
    for name,s in result['suites'].items():
        lines.append(f"| {name} | {s['completed']}/{s['expected']} | {s['successes']}/{s['completed']} | {s['api_requests']} | {s['finish_reasons'].get('length',0)} | ${s['cost_usd']:.6f} |")
    lines+=['',f"**API usage-derived charge: ${result['cost_usd']:.6f}.**"]
    if local:lines+=['Local Qwen has no API fee. GPU rental/electricity cost is not priced here; server timestamps are retained.']
    else:
        lines+=[f"Interrupted/unknown attempts without returned usage: {result['unpriced_attempts']}; their potential charges are excluded.",
            f"Rejected requests (including HTTP402 balance errors): {result['rejected_api_attempts']}; no generated response or token usage was returned.",
            f"Conservative additional allowance USD: {result['unpriced_ceiling_estimate_usd']} (full output ceiling per interrupted request, not measured usage).",
            'DeepSeek Flash off-peak USD per million: cache hit 0.003, cache miss 0.15, output 0.6; weekday UTC 01–04/06–10 peak rates are double. Reasoning tokens are included in output.',
            '[Official pricing](https://api-docs.deepseek.com/quick_start/pricing/).']
    previous=manifest.get('previous_experiment')
    if previous:
        old=json.loads((root.parent/previous/'RESULTS.json').read_text())
        result['previous_completed_api_cost_usd']=old['total_cost_usd']
        result['combined_api_cost_usd']=old['total_cost_usd']+result['cost_usd']
        lines+=['',f"Previously completed seen140 + WebShop500: ${old['total_cost_usd']:.6f}. Combined with this unseen run: **${result['combined_api_cost_usd']:.6f}** (usage-derived charges).",f"See [previous results](../{previous}/RESULTS.md) for the separate original table and its unpriced-attempt allowance."]
    lines+=['','Exact task manifests, configuration, immutable source snapshots, dependency/data hashes, prompts, native reasoning, final responses, actions and token usage are retained in this folder. See NOTES.md for reproduction.','']
    atomic_json(root/'RESULTS.json',result);atomic_json(root/'per-task-results.json',all_rows)
    tmp=root/'RESULTS.md.tmp';tmp.write_text('\n'.join(lines));tmp.replace(root/'RESULTS.md')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('experiment',type=Path);a=p.parse_args()
    r=report(a.experiment);print(json.dumps(dict(status=r['status'],cost_usd=r['cost_usd'],
        progress={k:f"{v['completed']}/{v['expected']}" for k,v in r['suites'].items()})))


if __name__=='__main__':main()
