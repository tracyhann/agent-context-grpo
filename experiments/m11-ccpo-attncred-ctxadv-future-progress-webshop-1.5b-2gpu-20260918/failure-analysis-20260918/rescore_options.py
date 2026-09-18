"""Offline counterfactual only: pass goal option values, not dictionary items."""
import os,sys,json,copy,random
from pathlib import Path
root=Path(__file__).resolve().parents[3];out=Path(__file__).resolve().parent
source=out.parent
env=json.loads((source/'config.json').read_text())['env'];os.environ.update({k:str(v) for k,v in env.items()});os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path.insert(0,str(root/'verl-agent/agent_system/environments/env_package/webshop/webshop'))
from web_agent_site.engine.goal import get_reward,get_option_reward,get_goals
from web_agent_site.engine.engine import load_products
base=root/'verl-agent/agent_system/environments/env_package/webshop/webshop/data'
random.seed(1000)
products,byasin,prices,_=load_products(str(base/'items_shuffle_1000.json'),str(base/'items_ins_v2_1000.json'),human_goals=False)
goals=get_goals(products,prices,False)
rows=[]
for g in goals:
    old,_=get_option_reward(list(g['goal_options'].values()),g['goal_options'].items())
    new,_=get_option_reward(list(g['goal_options'].values()),g['goal_options'].values())
    if old is not None and old<1:rows.append({'asin':g['asin'],'options':g['goal_options'],'original_option_score':old,'fixed_option_score':new})
result={'all_synthetic_goals':len(goals),'exact_options_not_fully_credited':len(rows),'exact_options_all_fixed':all(r['fixed_option_score']==1 for r in rows),'bad_goal_options':rows,'models':{}}
for label in ['m11','m5']:
    path=out/label/'episodes.jsonl'
    if not path.exists():continue
    episodes=[json.loads(s) for s in path.read_text().splitlines()]; rescored=[]
    for e in episodes:
        p=next((s['purchase'] for s in e['steps'] if 'purchase'in s),None)
        if not p:score=0;components={}
        else:
            old_score,_=get_reward(p['product'],e['goal'],p['price'],p['options'],verbose=True)
            assert abs(old_score-e['task_score'])<1e-9
            goal=copy.deepcopy(e['goal']);goal['goal_options']=list(goal['goal_options'].values())
            score,components=get_reward(p['product'],goal,p['price'],p['options'],verbose=True)
        rescored.append({'batch':e['batch'],'slot':e['slot'],'original_score':e['task_score'],'fixed_score':score,'fixed_components':components})
    result['models'][label]={'episodes':len(episodes),'original_successes':sum(e['success'] for e in episodes),'fixed_successes':sum(r['fixed_score']==1 for r in rescored),'fixed_mean_score':sum(r['fixed_score'] for r in rescored)/len(rescored),'rescored':rescored}
(out/'option_rescoring.json').write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k not in ['bad_goal_options','models']},indent=2))
print(json.dumps({k:{j:v for j,v in d.items() if j!='rescored'} for k,d in result['models'].items()},indent=2))
