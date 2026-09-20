#!/usr/bin/env python3
"""Replay saved benchmark actions locally, verifying prompts and scores without API calls."""
import argparse
import json
from pathlib import Path
import deepseek_api_eval as evaluator


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('episode',type=Path);a=p.parse_args()
    record=json.loads(a.episode.read_text())
    config=json.loads((a.episode.parent.parent/'config.json').read_text())
    settings=config['settings'];evaluator.runtime_setup()
    manager,raw=evaluator.environment(record['benchmark'],record,settings['history_length'])
    try:
        observations,infos=manager.reset({})
        assert observations['anchor'][0]==record['initial_observation'],'Initial observation differs'
        for index,turn in enumerate(record['turns']):
            assert observations['text'][0]==turn['prompt'],f'Prompt differs at turn {index}'
            assert observations['anchor'][0]==turn['observation'],f'Observation differs at turn {index}'
            if record['benchmark']=='webshop':
                current=evaluator.plain(raw.details());saved=turn['state_before_action']
                for key in ['goal','asin','options','product_price']:
                    assert current.get(key)==saved.get(key),f'WebShop {key} differs at turn {index}'
            observations,rewards,dones,infos=manager.step([turn['parser_text']])
            assert raw.last_action==turn['action'],f'Parsed action differs at turn {index}'
            assert observations['anchor'][0]==turn['next_observation'],f'Next observation differs at turn {index}'
            assert float(rewards[0])==turn['reward'],f'Reward differs at turn {index}'
            assert bool(dones[0])==turn['done'],f'Done flag differs at turn {index}'
            score=float(infos[0].get('task_score',bool(infos[0].get('won',False))))
            assert abs(score-turn['score'])<1e-9,f'Score differs at turn {index}'
        print(json.dumps({'status':'passed','benchmark':record['benchmark'],'episode_id':record['episode_id'],
            'turns_replayed':len(record['turns']),'success':record['success'],'score':record['score'],'api_calls':0}))
    finally:raw.close()

if __name__=='__main__':main()
