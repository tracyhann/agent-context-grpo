"""Verify both blocked trajectories and their next request entirely offline."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'scripts'))
import benchmark_census as census
import deepseek_api_eval as evaluator
exp=Path(__file__).resolve().parent;out=exp/'outputs/alfworld-unseen';settings=json.loads((out/'config.json').read_text())['settings'];evaluator.runtime_setup();results=[]
for eid in [100,123]:
 e=json.loads((out/'episodes'/f'{eid:04d}.json').read_text());assert e['status']=='error' and 'HTTP 402' in e['error']
 manager,raw=census.environment('alfworld',e,settings['history_length'])
 try:
  obs,info=manager.reset({})
  for index,t in enumerate(e['turns']):
   assert obs['text'][0]==t['prompt'] and obs['anchor'][0]==t['observation'],index
   obs,reward,done,info=manager.step([t['parser_text']])
   assert raw.last_action==t['action'] and obs['anchor'][0]==t['next_observation'],index
   assert float(reward[0])==t['reward'] and bool(done[0])==t['done'],index
   assert float(info[0].get('task_score',bool(info[0].get('won',False))))==t['score'],index
  errors=[json.loads(x) for x in (out/'api_calls'/f'{eid:04d}.errors.jsonl').read_text().splitlines()]
  assert hashlib.sha256(obs['text'][0].encode()).hexdigest()==errors[-1]['prompt_sha256']
  results.append(dict(episode_id=eid,saved_actions_replayed=len(e['turns']),next_prompt_matches_rejected_request=True))
 finally:raw.close()
result=dict(status='passed',api_calls=0,replays=results)
(exp/'RESUME_VALIDATION.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
