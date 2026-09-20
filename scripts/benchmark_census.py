"""Exact held-out task populations shared by API and local-model evaluations."""
import copy
import json
from pathlib import Path
import random
from types import SimpleNamespace
import deepseek_api_eval as legacy

ROOT=legacy.ROOT
SPLITS={'seen':('valid_seen','eval_in_distribution',140),'unseen':('valid_unseen','eval_out_of_distribution',134)}
_BASES={}


def plan(benchmark,seed=101,split='seen'):
    if benchmark=='webshop':return legacy.plan('webshop',500,seed)
    directory,mode,expected=SPLITS[split];groups={};rng=random.Random(seed)
    for p in sorted((ROOT/'alfworld_data/json_2.1.1'/directory).rglob('game.tw-pddl')):
        if 'movable' in str(p) or 'Sliced' in str(p) or not json.loads(p.read_text()).get('solvable'):continue
        kind=json.loads((p.parent/'traj_data.json').read_text())['task_type']
        groups.setdefault(kind,[]).append(str(p.relative_to(ROOT)))
    counts={k:len(v) for k,v in groups.items()};assert sum(counts.values())==expected,counts
    for values in groups.values():rng.shuffle(values)
    items=[]
    while any(groups.values()):
        for kind in sorted(groups):
            if groups[kind]:items.append({'episode_id':len(items),'task_type':kind,'gamefile':groups[kind].pop(),'worker_seed':seed,'alfworld_split':split})
    assert len({x['gamefile'] for x in items})==expected
    return items,counts


class ExactSplitAlfworld(legacy.SingleAlfworld):
    def __init__(self,item):
        from agent_system.environments.env_package.alfworld.envs import AlfworldWorker,load_config_file,get_environment
        split=item.get('alfworld_split','seen');directory,mode,expected=SPLITS[split]
        if split not in _BASES:
            cfg=load_config_file(str(ROOT/'verl-agent/agent_system/environments/env_package/alfworld/configs/config_tw.yaml'))
            cfg['general']['use_cuda']=False
            _BASES[split]=(cfg,get_environment(cfg['env']['type'])(cfg,train_eval=mode))
        cfg,base=_BASES[split];game=str(ROOT/item['gamefile'])
        assert game in base.game_files,'Exact task not in selected environment split'
        selected=copy.copy(base);selected.game_files=[game];selected.num_games=1
        self.worker=AlfworldWorker(cfg,item['worker_seed'],selected)
        self.reset_index=0;self.get_admissible_commands=[];self.last_action=None


_LEGACY_ENVIRONMENT=legacy.environment

def environment(benchmark,item,history):
    if benchmark=='webshop':return _LEGACY_ENVIRONMENT(benchmark,item,history)
    from agent_system.environments.env_manager import AlfWorldEnvironmentManager
    from agent_system.environments.env_package.alfworld import alfworld_projection
    raw=ExactSplitAlfworld(item)
    return AlfWorldEnvironmentManager(raw,alfworld_projection,SimpleNamespace(env=SimpleNamespace(history_length=history))),raw
