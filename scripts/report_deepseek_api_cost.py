#!/usr/bin/env python3
"""Summarize completed DeepSeek pilots and extrapolate measured token charges."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics


def summarize(folder):
    config=json.loads((folder/'config.json').read_text())
    records=[json.loads(p.read_text()) for p in sorted((folder/'episodes').glob('*.json'))]
    if len(records)!=len(config['episodes']) or any(r['status']!='completed' for r in records):
        raise ValueError(f'{folder}: pilot is incomplete; no full-eval estimate emitted')
    sample=[];calls=[]
    for record in records:
        ledger=folder/'api_calls'/f'{record["episode_id"]:04d}.jsonl'
        requests=[json.loads(s) for s in ledger.read_text().splitlines() if s.strip()]
        if len(requests)!=len(record['turns']):raise ValueError('Billed request/turn mismatch requires manual audit')
        calls.extend(requests)
        account={key:sum(r['accounting'][key] for r in requests) for key in requests[0]['accounting']}
        sample.append(dict(episode_id=record['episode_id'],task_type=record.get('task_type','test'),
            task_id=record.get('gamefile',record.get('goal_index')),success=record['success'],
            score=record['score'],turns=record['length'],horizon_truncated=record['horizon_truncated'],
            response_truncations=sum(r['finish_reason']=='length' for r in requests),
            actions=[t['action'] for t in record['turns']],**account))
    grouped=defaultdict(list)
    for item in sample:grouped[item['task_type']].append(item)
    if set(grouped)!=set(config['full_population']):raise ValueError('Some population strata have no samples')
    estimate={key:sum(n*statistics.mean(x[key] for x in grouped[group]) for group,n in config['full_population'].items())
              for key in ['off_peak_usd','no_cache_off_peak_usd','prompt_tokens','completion_tokens','reasoning_tokens','turns']}
    estimate['peak_usd']=2*estimate['off_peak_usd']
    estimate['no_cache_peak_usd']=2*estimate['no_cache_off_peak_usd']
    # Budget allowance, not a confidence interval: deliberately separate from measured extrapolation.
    estimate['planning_budget_off_peak_usd']=2*estimate['no_cache_off_peak_usd']
    estimate['planning_budget_peak_usd']=2*estimate['planning_budget_off_peak_usd']
    return dict(benchmark=config['settings']['benchmark'],model=config['settings']['model'],
        settings=config['settings'],source_sha256=config['source_sha256'],
        population=config['full_population'],full_episodes=sum(config['full_population'].values()),
        sample_episodes=len(sample),sample_successes=sum(x['success'] for x in sample),
        sample_mean_score=statistics.mean(x['score'] for x in sample),
        sample_total_cost_off_peak_usd=sum(x['off_peak_usd'] for x in sample),
        sample_mean_turns=statistics.mean(x['turns'] for x in sample),
        sample_mean_prompt_tokens=statistics.mean(x['prompt_tokens'] for x in sample),
        sample_mean_completion_tokens=statistics.mean(x['completion_tokens'] for x in sample),
        sample_total_reasoning_tokens=sum(x['reasoning_tokens'] for x in sample),
        response_truncations=sum(x['response_truncations'] for x in sample),
        requests=len(calls),retried_requests=sum(bool(c['prior_attempt_errors']) for c in calls),
        request_latency_mean_s=statistics.mean(c['elapsed_s'] for c in calls),
        backend_fingerprints=sorted({str(c.get('system_fingerprint')) for c in calls}),
        episode_cost_min_usd=min(x['off_peak_usd'] for x in sample),
        episode_cost_max_usd=max(x['off_peak_usd'] for x in sample),
        estimation_method='task-type population weighted' if config['settings']['benchmark']=='alfworld' else 'sample mean times 500',
        estimate=estimate,examples=sample)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('experiment',type=Path);args=p.parse_args()
    results={b:summarize(args.experiment/'outputs'/b) for b in ['alfworld','webshop']}
    total={k:sum(r['estimate'][k] for r in results.values()) for k in results['alfworld']['estimate']}
    probe_path=args.experiment/'outputs/budget-probe-summary.json'
    probe=json.loads(probe_path.read_text()) if probe_path.exists() else None
    report=dict(benchmarks=results,total_full_evaluation=total,
        pilot_spend_off_peak_usd=sum(r['sample_total_cost_off_peak_usd'] for r in results.values()),
        price_source='https://api-docs.deepseek.com/quick_start/pricing/',currency='USD',
        limits=['Six episodes per benchmark: cost extrapolation, not a reliable performance estimate.',
                'ALFWorld valid_seen 140, WebShop test goals 0..499 on the existing 1000-product catalog.',
                'One seed fixes task/environment sampling. The API does not expose a deterministic generation seed.',
                'Completion tokens already include reasoning tokens; reasoning is not billed twice.',
                'Planning budget is 2x the no-cache estimate, not a statistical confidence bound.',
                'Future model versions, prompt/turn budgets and failures can change the cost.'])
    report['budget_probe']=probe
    report['diagnostic_spend_off_peak_usd']=probe['accounting']['off_peak_usd'] if probe else 0.
    report['total_api_spend_off_peak_usd']=report['pilot_spend_off_peak_usd']+report['diagnostic_spend_off_peak_usd']
    (args.experiment/'COST_ESTIMATE.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    lines=['# DeepSeek Flash API: measured pilot and full-evaluation cost estimate','',
        'Actual pilot: six complete tasks per benchmark, seed 0, history 2, native thinking/high,',
        '4,096 maximum output tokens per action. ALFWorld has a 50-action ceiling; WebShop has 15.',
        'No local model or GPU is used. Prices below are USD; off-peak is half peak.','',
        '| Benchmark | Full tasks | Sample success | Mean turns | Mean input tokens/task | Mean output tokens/task | Pilot cost | Full off-peak | Full peak |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for bench,r in results.items():
        lines.append(f"| {bench} | {r['full_episodes']} | {r['sample_successes']}/{r['sample_episodes']} | {r['sample_mean_turns']:.2f} | {r['sample_mean_prompt_tokens']:.0f} | {r['sample_mean_completion_tokens']:.0f} | ${r['sample_total_cost_off_peak_usd']:.4f} | ${r['estimate']['off_peak_usd']:.2f} | ${r['estimate']['peak_usd']:.2f} |")
    lines += ['',f"**Combined full-evaluation estimate: ${total['off_peak_usd']:.2f} off-peak / ${total['peak_usd']:.2f} peak.**",
        f"Without cache savings: ${total['no_cache_off_peak_usd']:.2f} / ${total['no_cache_peak_usd']:.2f}.",
        f"Planning allowance (2x no-cache estimate): ${total['planning_budget_off_peak_usd']:.2f} off-peak / ${total['planning_budget_peak_usd']:.2f} peak.",
        f"Actual pilot token charge at off-peak rates: ${report['pilot_spend_off_peak_usd']:.4f}.", '',
        '## Accounting', '',
        'Per request: cost = (cache-hit input * 0.003 + cache-miss input * 0.15 + completion * 0.6) / 1,000,000.',
        'Completion includes native reasoning. Peak rates double each component. ALFWorld is weighted by',
        'the six task-type populations, since the pilot takes one task of each type; WebShop uses a random',
        'six-goal sample and extrapolates its mean to all 500 goals.', '',
        '[Official rate card](https://api-docs.deepseek.com/quick_start/pricing/).', '', '## Completed examples','',
        '| Benchmark | Task / goal | Success | Score | Turns | Input tokens | Output tokens | Off-peak cost |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for bench,r in results.items():
        for item in r['examples']:
            label=item['task_type'] if bench=='alfworld' else str(item['task_id'])
            lines.append(f"| {bench} | {label} | {int(item['success'])} | {item['score']:.3f} | {item['turns']} | {item['prompt_tokens']} | {item['completion_tokens']} | ${item['off_peak_usd']:.4f} |")
    if probe:
        lines += ['', '## Output-budget diagnostic', '',
            f"One truncated ALFWorld prompt was replayed at 8,192 tokens without stepping the environment: finish={probe['finish_reason']}, valid action tag={probe['action_tag_valid']}, completion tokens={probe['accounting']['completion_tokens']}.",
            f"This additional request cost ${report['diagnostic_spend_off_peak_usd']:.5f}; all pilot plus diagnostic calls cost ${report['total_api_spend_off_peak_usd']:.4f}.",
            'The probe shows that 4,096 tokens can cut off useful reasoning. It does not provide a full-benchmark cost estimate at 8,192 tokens; the estimates above remain conditional on the measured 4,096-token protocol.']
    lines += ['', '## Generation truncation', '',
        'Response token ceilings were reached on '+', '.join(f"{b}: {r['response_truncations']}/{r['requests']} requests" for b,r in results.items())+'.',
        'All of these requests and subsequent failed/horizon-limited trajectories are included in cost. The 4,096-token cap can prevent an action from being emitted; this pilot is not evidence of uncapped model accuracy. A full evaluation at 8,192 or more output tokens is a different cost condition.']
    lines += ['', '## Limits', '']+['- '+s for s in report['limits']]
    lines += ['', 'Task identities, action traces, original item-option scorer state and API usage are saved under',
              '`outputs/<benchmark>/episodes/` and `outputs/<benchmark>/api_calls/`. No API secret is written to these files.',
              'The WebShop setup-only serialization failure was preserved separately and made no API calls.', '']
    (args.experiment/'COST_ESTIMATE.md').write_text('\n'.join(lines))
    print(json.dumps({'total':total,'pilot_spend':report['pilot_spend_off_peak_usd']},indent=2))


if __name__=='__main__':main()
