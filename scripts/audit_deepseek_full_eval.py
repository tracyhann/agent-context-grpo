#!/usr/bin/env python3
"""Validate full task coverage, raw API accounting, task identity and provenance offline."""
import argparse
from collections import Counter
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import statistics
import deepseek_api_eval as evaluator
from report_deepseek_full_eval import aggregate


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('experiment',type=Path);a=p.parse_args();root=a.experiment
    result,_=aggregate(root)
    assert result['status']=='completed','Full evaluation is not complete'
    manifest=json.loads((root/'webshop-goals-seed101.json').read_text())
    all_ids=set();summary={};types=Counter();successes=Counter()
    for bench,expected in [('alfworld',140),('webshop',500)]:
        out=root/'outputs'/bench;config=json.loads((out/'config.json').read_text());settings=config['settings']
        assert settings['seed']==101 and settings['max_tokens']==65536 and settings['reasoning_effort']=='high' and settings['thinking']
        assert hashlib.sha256((evaluator.ROOT/'scripts/deepseek_api_eval.py').read_bytes()).hexdigest()==config['source_sha256']
        records=[json.loads(p.read_text()) for p in sorted((out/'episodes').glob('*.json'))]
        assert len(records)==expected and all(e['status']=='completed' for e in records)
        ledger={}
        for f in (out/'api_calls').glob('*.jsonl'):
            if f.name.endswith('.errors.jsonl'):continue
            for line in f.read_text().splitlines():
                r=json.loads(line);rid=r['request_id']
                assert rid not in all_ids,'Duplicate API request ID'
                all_ids.add(rid);ledger[rid]=r
        consumed=set();lengths=[];success=0;score=0.
        for e in records:
            assert e['length']==len(e['turns'])
            assert e['success']==any(t['reward']==10.0 for t in e['turns'])
            assert abs(e['score']-max(t['score'] for t in e['turns']))<1e-9
            success+=e['success'];score+=e['score']
            if bench=='alfworld':types[e['task_type']]+=1;successes[e['task_type']]+=e['success']
            else:
                assert e['initial_session']['goal']==manifest['goals'][e['goal_index']]
                assert e['test_goals_sha256']==manifest['test_goals_sha256']
                if e['purchase_state'] is not None:
                    assert e['turns'][-1]['done'] and e['turns'][-1]['action']=='click[buy now]'
                    assert e['purchase_state']==e['turns'][-1]['state_before_action']
            for t in e['turns']:
                r=ledger[t['request_id']];assert t['request_id'] not in consumed;consumed.add(t['request_id'])
                assert hashlib.sha256(t['prompt'].encode()).hexdigest()==r['prompt_sha256']
                assert r['usage']==t['usage'] and r['accounting']==t['accounting']
                assert t['parser_text']==evaluator.parser_response(t['content'])
                lengths.append(r['usage']['completion_tokens'])
                if bench=='webshop':
                    s=t['state_before_action']
                    if s.get('product_price') is not None:assert s['product_price']==manifest['product_prices'][s['asin']]
        assert consumed==set(ledger),'Unaccounted paid response or trajectory request'
        if bench=='alfworld':assert len({e['gamefile'] for e in records})==140
        else:assert {e['goal_index'] for e in records}==set(range(500))
        summary[bench]={'episodes':expected,'successes':success,'success_pct':100*success/expected,
            'score_pct':100*score/expected,'api_requests':len(ledger),
            'output_tokens_max':max(lengths),'output_tokens_median':statistics.median(lengths),
            'responses_over_4096_tokens':sum(x>4096 for x in lengths),
            'responses_over_16384_tokens':sum(x>16384 for x in lengths)}
    files_checked=0
    for name in ['source-sha256.json','dataset-sha256.json']:
        for filename,digest in json.loads((root/name).read_text()).items():
            assert hashlib.sha256((evaluator.ROOT/filename).read_bytes()).hexdigest()==digest,f'Provenance mismatch: {filename}'
            files_checked+=1
    key=(evaluator.ROOT/'baselines/deepseek/DEEPSEEK-API-KEY.txt').read_bytes().strip()
    for f in root.rglob('*'):
        if f.is_file():assert key not in f.read_bytes(),f'Credential found in {f}'
    validation={'status':'passed','completed_at':datetime.now(timezone.utc).isoformat(),
        'full_unique_task_coverage':True,'all_received_api_usage_reconciled_to_turns':True,
        'task_goals_prices_and_scores_verified':True,'no_credential_in_artifacts':True,
        'source_and_data_files_verified':files_checked,'no_api_calls_by_audit':True,
        'benchmark_summary':summary,'alfworld_task_counts':dict(types),'alfworld_task_successes':dict(successes),
        'usage_derived_cost_usd':result['total_cost_usd'],'unpriced_interrupted_attempts':result['unpriced_api_attempts'],
        'unpriced_charge_ceiling_allowance_usd':result['unpriced_charge_ceiling_estimate_usd']}
    evaluator.atomic_json(root/'FINAL_AUDIT.json',validation)
    print(json.dumps(validation,indent=2))

if __name__=='__main__':main()
