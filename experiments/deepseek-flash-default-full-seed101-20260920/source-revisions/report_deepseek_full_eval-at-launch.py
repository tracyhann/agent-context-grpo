#!/usr/bin/env python3
"""Read-only aggregation of the full DeepSeek eval; raw requests are the cost ledger."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

TASKS = [('Pick', 'pick_and_place_simple'), ('Look', 'look_at_obj_in_light'),
         ('Clean', 'pick_clean_then_place_in_recep'), ('Heat', 'pick_heat_then_place_in_recep'),
         ('Cool', 'pick_cool_then_place_in_recep'), ('Pick2', 'pick_two_obj_and_place')]


def write_json(path, value):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)


def aggregate(root):
    result={'updated_at':datetime.now(timezone.utc).isoformat(),'benchmarks':{},'currency':'USD'}
    tasks=[]
    for bench,expected in [('alfworld',140),('webshop',500)]:
        out=root/'outputs'/bench
        config=json.loads((out/'config.json').read_text()) if (out/'config.json').exists() else None
        rows=[]; running=[]; errors=[]
        planned={r['episode_id']:r for r in config['episodes']} if config else {}
        for path in sorted((out/'episodes').glob('*.json')):
            episode=json.loads(path.read_text())
            identity=planned.get(episode['episode_id'])
            if identity:
                for key,value in identity.items():
                    if episode.get(key)!=value:raise ValueError(f'{bench}: task identity mismatch')
            n=len(episode['turns'])
            row={k:episode[k] for k in ['episode_id','status','task_type','gamefile','goal_index','success','score','length','termination','started','ended','error'] if k in episode}
            row['turns']=n
            row['capped_requests']=sum(t['finish_reason']=='length' for t in episode['turns'])
            row['invalid_action_tags']=sum(not t['action_tag_valid'] for t in episode['turns'])
            row['cost_off_peak_usd']=sum(t['accounting']['off_peak_usd'] for t in episode['turns'])
            row['trajectory_sha256']=hashlib.sha256(path.read_bytes()).hexdigest() if row['status']=='completed' else None
            if row['status']=='completed':rows.append(row)
            elif row['status']=='error':errors.append(row)
            else:running.append(row)
            tasks.append(dict(benchmark=bench,**row))
        usage=Counter();finishes=Counter();models=Counter();fingerprints=Counter();periods=Counter();request_ids=set()
        cost=0.;ledger_turns=0;network_failures=[]
        for path in sorted((out/'api_calls').glob('*.jsonl')):
            if path.name.endswith('.errors.jsonl'):
                for line in path.read_text().splitlines():
                    try:network_failures.append(json.loads(line))
                    except json.JSONDecodeError:pass
                continue
            with path.open() as stream:
                for line in stream:
                    try:r=json.loads(line)
                    except json.JSONDecodeError:continue  # a concurrent append can expose its final partial line
                    rid=r['request_id']
                    if rid in request_ids:raise ValueError(f'{bench}: duplicate request ID')
                    request_ids.add(rid);ledger_turns+=1
                    u=r['usage'];a=r['accounting']
                    if u['prompt_tokens']+u['completion_tokens']!=u['total_tokens']:raise ValueError('Token total mismatch')
                    for key in ['prompt_tokens','completion_tokens','cache_hit_tokens','cache_miss_tokens','reasoning_tokens']:
                        usage[key]+=a[key]
                    if a['cache_hit_tokens']+a['cache_miss_tokens']!=a['prompt_tokens']:raise ValueError('Prompt accounting mismatch')
                    if not 0<=a['reasoning_tokens']<=a['completion_tokens']:raise ValueError('Reasoning token mismatch')
                    t=datetime.fromtimestamp(r['api_created'],timezone.utc)
                    peak=t.weekday()<5 and (1<=t.hour<4 or 6<=t.hour<10)
                    factor=2 if peak else 1
                    amount=(a['cache_hit_tokens']*.003+a['cache_miss_tokens']*.15+a['completion_tokens']*.6)/1e6
                    if abs(amount-a['off_peak_usd'])>1e-10:raise ValueError('Cost mismatch')
                    cost+=factor*amount;periods['peak' if peak else 'off_peak']+=1
                    finishes[r['finish_reason']]+=1;models[r['model']]+=1;fingerprints[r['system_fingerprint']]+=1
        complete=len(rows)==expected and not errors and not running
        if complete and ledger_turns!=sum(r['turns'] for r in rows):raise ValueError('Unreconciled API requests')
        stats={'status':'completed' if complete else 'running' if running or rows else 'not_started',
               'expected':expected,'completed':len(rows),'running':len(running),'errors':len(errors),
               'observed_successes':sum(r['success'] for r in rows),'observed_score_sum':sum(r['score'] for r in rows),
               'observed_success_pct':100*sum(r['success'] for r in rows)/len(rows) if rows else None,
               'observed_score_pct':100*sum(r['score'] for r in rows)/len(rows) if rows else None,
               'finished_action_turns':sum(r['turns'] for r in rows),'api_requests':ledger_turns,
               'usage':dict(usage),'cost_usd':cost,'pricing_period_requests':dict(periods),
               'finish_reasons':dict(finishes),'models':dict(models),'system_fingerprints':dict(fingerprints),
               'api_attempt_errors':len(network_failures),'unpriced_attempts':network_failures,
               'result_is_final':complete,'settings':config['settings'] if config else None}
        if bench=='alfworld':
            stats['task_types']={label:{'count':sum(r['task_type']==task for r in rows),
                'successes':sum(r['success'] for r in rows if r['task_type']==task),
                'expected':config['full_population'][task] if config else None} for label,task in TASKS}
            for group in stats['task_types'].values():
                group['success_pct']=100*group['successes']/group['count'] if group['count'] else None
        if errors:stats['status']='incomplete_with_errors'
        result['benchmarks'][bench]=stats
    result['status']='completed' if all(v['result_is_final'] for v in result['benchmarks'].values()) else 'in_progress'
    result['total_cost_usd']=sum(v['cost_usd'] for v in result['benchmarks'].values())
    result['total_api_requests']=sum(v['api_requests'] for v in result['benchmarks'].values())
    result['unpriced_api_attempts']=sum(v['api_attempt_errors'] for v in result['benchmarks'].values())
    return result,tasks


def render(result):
    a=result['benchmarks']['alfworld'];w=result['benchmarks']['webshop']
    lines=['# DeepSeek Flash official-default full evaluation — seed 101','',
        f"Status: **{result['status']}**. Updated {result['updated_at']}.",'',
        'Model `deepseek-flash`; native thinking/high; 65,536 max output tokens per action. ',
        'History 2; ALFWorld 50 actions; WebShop 15 actions. Environment/task seed 101. ',
        'API generation is not deterministically seeded. Raw prompts, responses, task IDs and usage are retained.','',
        '## Requested results','',
        'Percent units (0–100). Final scores appear only after every requested episode completes.','',
        '| Model | Pick | Look | Clean | Heat | Cool | Pick2 | All | WebShop Score | WebShop Succ |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    fields=[]
    for label,_ in TASKS:
        v=a['task_types'][label]
        fields.append(f"{v['success_pct']:.2f}" if a['result_is_final'] else 'pending')
    fields += [f"{a['observed_success_pct']:.2f}" if a['result_is_final'] else 'pending',
        f"{w['observed_score_pct']:.2f}" if w['result_is_final'] else 'pending',
        f"{w['observed_success_pct']:.2f}" if w['result_is_final'] else 'pending']
    lines+=['| DeepSeek Flash | '+' | '.join(fields)+' |','',
        'ALFWorld All is total successes / 140, not the unweighted mean of task-type percentages. ',
        'WebShop Score is 100 × mean original environment dense reward; Succ is 100 × fraction with reward exactly 1. ',
        'Every action-limit failure remains in the denominator; unpurchased WebShop tasks receive zero.','',
        '## Progress and measured cost','',
        '| Benchmark | Finished | Successes among finished | API calls | Truncated calls | Cost USD |',
        '|---|---:|---:|---:|---:|---:|']
    for name,b in result['benchmarks'].items():
        lines.append(f"| {name} | {b['completed']}/{b['expected']} | {b['observed_successes']}/{b['completed']} | {b['api_requests']} | {b['finish_reasons'].get('length',0)} | ${b['cost_usd']:.6f} |")
    lines+=['',f"**Total API usage-derived charge: ${result['total_cost_usd']:.6f}.**",'',
        'Only this full seed-101 evaluation is included; earlier pilots and diagnostic replays are excluded. ',
        'Costs use cache-hit input, cache-miss input and completion tokens; reasoning is already included in completion. ',
        'Rates per million off-peak USD: 0.003 / 0.15 / 0.6. Peak doubles each component. ',
        'All timestamps, pricing-window counts and backend fingerprints are in RESULTS.json. ',
        'This is calculated from returned usage, rather than an account invoice.','',
        f"API attempts without returned usage: {result['unpriced_api_attempts']}. If nonzero, their possible charges are not included.",'',
        '## Split and reproducibility','',
        '- ALFWorld: all 140 solvable `valid_seen` tasks, held out from training in seen environments. The distinct `valid_unseen` split has 134 tasks and is not evaluated here.',
        '- WebShop: all 500 test goal positions (0–499) under seed-101 goal construction, on the existing 1,000-product catalog. Goal metadata and prices are saved; positions 500 onward are excluded.',
        '- Existing original WebShop option matching/scoring is used; no post-hoc item-option score correction is applied to the headline table.',
        '- Source snapshots, exact task plans, dependency versions, dataset hashes, and commands are in this experiment folder. Model alias/backend updates can prevent bit-identical API reproduction.',
        '- The native reasoning field is logged separately and cannot supply executable actions. Existing benchmark prompts and action parsers otherwise apply.',
        '- WebShop pre-action purchase state is retained because the environment auto-resets on terminal actions.','',
        '[Official API defaults](https://api-docs.deepseek.com/api/create-chat-completion/), ',
        '[official pricing](https://api-docs.deepseek.com/quick_start/pricing/).','']
    return '\n'.join(lines)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('experiment',type=Path);a=p.parse_args()
    result,tasks=aggregate(a.experiment)
    write_json(a.experiment/'RESULTS.json',result);write_json(a.experiment/'per-task-results.json',tasks)
    temp=a.experiment/'RESULTS.md.tmp';temp.write_text(render(result));temp.replace(a.experiment/'RESULTS.md')
    print(json.dumps({'status':result['status'],'cost_usd':result['total_cost_usd'],
        'progress':{k:f"{v['completed']}/{v['expected']}" for k,v in result['benchmarks'].items()}}))

if __name__=='__main__':main()
