import sys,json
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/'scripts'))
import benchmark_census as c
import qwen_census_eval as q
q.evaluator.runtime_setup();benchmark=sys.argv[1];item=c.plan(benchmark,101,'seen')[0][0]
manager,raw=c.environment(benchmark,item,2)
try:
 obs,infos=manager.reset({})
 settings=dict(base_url='http://127.0.0.1:8019/v1',thinking=True,reasoning_effort='xhigh',max_tokens=65536,max_model_len=131072,
  _turn=0,seed=101,suite_seed_offset=0,_episode_id=item['episode_id'],temperature=1.0,top_p=.95,presence_penalty=0.,timeout=7200)
 r=q.complete(obs['text'][0],settings,Path('.local/qwen38-census-smoke')/(benchmark+'.jsonl'))
 nxt,reward,done,info=manager.step([r['parser_text']])
 result=dict(benchmark=benchmark,native_reasoning_finished=r['native_reasoning_finished'],finish_reason=r['finish_reason'],usage=r['usage'],
  action=raw.last_action,format_valid=bool(info[0]['is_action_valid']),generation_seed=r['request_seed'],content=r['content'])
 Path('.local/qwen38-census-smoke/'+benchmark+'-result.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(result),flush=True)
 assert result['native_reasoning_finished'] and result['format_valid']
finally:raw.close()
