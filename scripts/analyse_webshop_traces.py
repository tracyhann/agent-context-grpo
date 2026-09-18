#!/usr/bin/env python3
"""Read-only comparison of instrumented WebShop evaluation episodes."""
import argparse,json,collections,statistics
from pathlib import Path


def purchase(e):return next((s['purchase'] for s in e['steps'] if 'purchase' in s),None)
def key(e):return e['batch'],e['slot']
def summarize(episodes,oracles):
    failures=[e for e in episodes if not e['success']]
    all_steps=[s for e in episodes for s in e['steps']]
    categories=collections.Counter();issues=collections.Counter()
    for e in failures:
        p=purchase(e)
        if p is None:
            categories['no_purchase']+=1
        else:
            c=p['components'];flags=[name for name,field in [('type','r_type'),('attributes','r_att'),('options','r_option'),('price','r_price')] if float(c.get(field,1))<1-1e-9]
            for flag in flags:issues[flag]+=1
            categories['+'.join(flags) or 'unclassified']+=1
    scores=[e['task_score'] for e in episodes]
    def avg(xs):return statistics.mean(xs) if xs else None
    result={'episodes':len(episodes),'successes':sum(e['success'] for e in episodes),'failures':len(failures),
        'success_rate':avg([e['success'] for e in episodes]),'task_score':avg(scores),'mean_length':avg([e['length'] for e in episodes]),
        'failure_categories_disjoint':dict(categories),'failure_components_overlapping':dict(issues),
        'format_valid_fraction':avg([s['format_valid'] for s in all_steps]),'admissible_fraction':avg([s['admissible'] for s in all_steps]),
        'episodes_with_invalid_click':sum(any(not s['admissible'] for s in e['steps']) for e in episodes),
        'failures_with_invalid_click':sum(any(not s['admissible'] for s in e['steps']) for e in failures),
        'failure_mean_score':avg([e['task_score'] for e in failures]),
        'failure_scores_at_least_0_8':sum(e['task_score']>=.8 for e in failures),
        'failures_on_oracle_bad_tasks':sum(oracles[key(e)]['oracle_score']<1 for e in failures),
        'exact_target_purchase_failures':sum(purchase(e) is not None and purchase(e)['asin']==e['goal']['asin'] and purchase(e)['options']==e['goal']['goal_options'] for e in failures),
        'failures_buy_first_product':sum(purchase(e) is not None and len(purchase(e)['visited_asins'])==1 for e in failures),
        'failures_read_description_or_features':sum(any(s['action'] in ['click[description]','click[features]','click[attributes]'] for s in e['steps']) for e in failures),
        'failure_examples':[]}
    for e in failures:
        p=purchase(e)
        result['failure_examples'].append({'batch':e['batch'],'slot':e['slot'],'goal':e['goal'],'score':e['task_score'],
          'oracle_score':oracles[key(e)]['oracle_score'],'actions':[s['action'] for s in e['steps']],
          'purchase':p,'invalid_turns':[s['turn'] for s in e['steps'] if not s['admissible']]})
    return result

def main():
    parser=argparse.ArgumentParser();parser.add_argument('directory',type=Path);args=parser.parse_args()
    root=args.directory;oracles={key(x):x for x in json.loads((root/'oracle_targets.json').read_text())}
    all_episodes={};results={}
    for label in ['m11','m5']:
        path=root/label/'episodes.jsonl'
        if not path.exists():continue
        episodes=[json.loads(s) for s in path.read_text().splitlines()]
        for e in episodes:
            assert all(e['goal'][k]==oracles[key(e)]['goal'][k] for k in ['asin','goal_options','attributes']), ('Oracle draw mismatch',label,key(e))
            p=purchase(e)
            if p:
                c=p['components'];calculated=c['r_type']*sum(c.get('r_'+part,0)*c.get('w_'+part,0) for part in ['att','option','price'])
                assert abs(calculated-e['task_score'])<1e-8,(label,key(e),calculated,e['task_score'])
            assert e['success']==(e['task_score']==1)
        all_episodes[label]=episodes;results[label]=summarize(episodes,oracles)
    if all(len(all_episodes.get(label,[]))==256 for label in ['m11','m5']):
        paired=[];m5={key(e):e for e in all_episodes['m5']}
        counts=collections.Counter()
        for e in all_episodes['m11']:
            other=m5[key(e)];assert e['goal']==other['goal']
            state=('both_success' if e['success'] and other['success'] else 'm11_only' if e['success'] else 'm5_only' if other['success'] else 'both_fail')
            counts[state]+=1
            paired.append({'batch':e['batch'],'slot':e['slot'],'outcome':state,'m11_score':e['task_score'],'m5_score':other['task_score']})
        results['paired']={'counts':dict(counts),'episodes':paired}
    (root/'trace_analysis.json').write_text(json.dumps(results,indent=2))
    print(json.dumps({k:({j:v for j,v in d.items() if j not in ['failure_examples','episodes']}) for k,d in results.items()},indent=2))
if __name__=='__main__':main()
