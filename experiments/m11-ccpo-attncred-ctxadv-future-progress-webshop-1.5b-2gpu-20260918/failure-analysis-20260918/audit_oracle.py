"""CPU check of whether exact target purchases receive full benchmark score."""
import os,json,random,sys
from pathlib import Path
root=Path(__file__).resolve().parents[3];out=Path(__file__).resolve().parent
source=root/'experiments/m11-ccpo-attncred-ctxadv-future-progress-webshop-1.5b-2gpu-20260918'
env=json.loads((source/'config.json').read_text())['env'];os.environ.update({k:str(v) for k,v in env.items()});os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path.insert(0,str(root/'verl-agent/agent_system/environments/env_package/webshop/webshop'))
import numpy as np
from web_agent_site.engine.engine import load_products,generate_product_prices
from web_agent_site.engine.goal import get_goals,get_reward,get_option_reward
base=root/'verl-agent/agent_system/environments/env_package/webshop/webshop/data'
random.seed(1000)
products,byasin,_,_=load_products(str(base/'items_shuffle_1000.json'),str(base/'items_ins_v2_1000.json'),human_goals=False)
rng=np.random.RandomState(1000);draws=[rng.choice(range(500),128,replace=False) for _ in range(2)]
records=[]
for slot in range(128):
    seed=1000+slot;random.seed(seed)
    prices=generate_product_prices(products)
    goals=get_goals(products,prices,False)
    random.seed(seed);random.shuffle(goals)
    for batch in range(2):
        idx=int(draws[batch][slot]);goal=goals[idx]
        score,components=get_reward(byasin[goal['asin']],goal,prices[goal['asin']],goal['goal_options'],verbose=True)
        records.append({'batch':batch,'slot':slot,'worker_seed':seed,'goal_index':idx,'goal':goal,'oracle_score':score,'oracle_components':components})
    if slot%16==0:print('oracle slots',slot,flush=True)
(out/'oracle_targets.json').write_text(json.dumps(records,indent=2))
print('EXACT TARGET FULL SCORE',sum(r['oracle_score']==1 for r in records),'/',len(records),flush=True)
for r in records:
    if r['oracle_score']<1:print(json.dumps(r),flush=True)
