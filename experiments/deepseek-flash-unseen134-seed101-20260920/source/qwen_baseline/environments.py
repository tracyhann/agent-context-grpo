"""Serial adapters around the SAME workers/managers used by local training.

Run with the benchmark's existing virtual environment. No Ray cluster is started.
"""
import copy
import os
import sys
from pathlib import Path
from types import SimpleNamespace

from .common import ROOT


def runtime_setup():
    os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Environments run on CPU; model is an HTTP service.
    for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
        os.environ[key] = '1'
    os.environ['ALFWORLD_DATA'] = str(ROOT / 'alfworld_data')
    for key in ['ACG_COMPACT_BUDGET', 'ACG_COMPACT_STALL', 'ACG_OBS_REPAIR', 'ACG_ANCHOR_AFF']:
        os.environ[key] = '0'
    java = Path(os.environ.get('ACG_JAVA_HOME', ROOT / 'jdk/jdk-11.0.32.1+1'))
    os.environ['JAVA_HOME'] = str(java)
    os.environ['PATH'] = str(java / 'bin') + os.pathsep + os.environ.get('PATH', '')
    os.environ['JAVA_TOOL_OPTIONS'] = '-XX:ActiveProcessorCount=1 -XX:+UseSerialGC -Xss512k -Xms32m -Xmx512m'
    sys.path.insert(0, str(ROOT / 'verl-agent'))


_BASES = {}


class SingleAlfworld:
    def __init__(self, item, split):
        from agent_system.environments.env_package.alfworld.envs import AlfworldWorker, load_config_file, get_environment
        config_path = ROOT / 'verl-agent/agent_system/environments/env_package/alfworld/configs/config_tw.yaml'
        if split not in _BASES:
            cfg = load_config_file(str(config_path))
            _BASES[split] = (cfg, get_environment(cfg['env']['type'])(cfg, train_eval=split))
        cfg, base = _BASES[split]
        self.worker = AlfworldWorker(cfg, item['worker_seed'], base)
        self.reset_index = item['reset_index']
        self.get_admissible_commands = []
        self.last_action = None

    def unpack(self, obs, infos):
        info = {k: v[0] for k, v in infos.items()}
        self.get_admissible_commands = [info['admissible_commands']]
        return obs, None, [info]

    def reset(self):
        for _ in range(self.reset_index + 1):
            obs, infos = self.worker.reset()
        return self.unpack(obs, infos)

    def step(self, actions):
        self.last_action = actions[0]
        obs, _, done, infos = self.worker.step(actions[0])
        obs, image, info = self.unpack(obs, infos)
        return obs, image, [10.0 * float(info[0]['won'])], done, info

    def close(self):
        self.worker.env.close()


class SingleWebshop:
    def __init__(self, item):
        from agent_system.environments.env_package.webshop.envs import WebshopWorker
        data = ROOT / 'verl-agent/agent_system/environments/env_package/webshop/webshop/data'
        self.worker = WebshopWorker(item['worker_seed'], dict(observation_mode='text', num_products=None,
            human_goals=False, file_path=str(data / 'items_shuffle_1000.json'), attr_path=str(data / 'items_ins_v2_1000.json')))
        self.goal_index = item['goal_index']
        self.last_action = None

    def reset(self):
        obs, info = self.worker.reset(self.goal_index)
        return [obs], [info]

    def step(self, actions):
        self.last_action = actions[0]
        obs, reward, done, info = self.worker.step(actions[0])
        return [obs], [reward], [done], [info]

    def details(self):
        base = self.worker.env.unwrapped
        session = copy.deepcopy(base.server.user_sessions[base.session])
        # Preserve goal and selected options, with product data for post-hoc scoring.
        asin = session.get('asin')
        if asin in base.server.product_item_dict:
            session['product'] = copy.deepcopy(base.server.product_item_dict[asin])
            session['product_price'] = base.server.product_prices[asin]
        return session

    def close(self):
        self.worker.close()


def build_environment(benchmark, item, split='eval_in_distribution', history_length=2):
    runtime_setup()
    from agent_system.environments.env_manager import AlfWorldEnvironmentManager, WebshopEnvironmentManager
    config = SimpleNamespace(env=SimpleNamespace(history_length=history_length))
    if benchmark == 'alfworld':
        from agent_system.environments.env_package.alfworld import alfworld_projection
        raw = SingleAlfworld(item, split)
        manager = AlfWorldEnvironmentManager(raw, alfworld_projection, config)
    elif benchmark == 'webshop':
        from agent_system.environments.env_package.webshop import webshop_projection
        raw = SingleWebshop(item)
        manager = WebshopEnvironmentManager(raw, webshop_projection, config)
    else:
        raise ValueError(benchmark)
    return manager, raw
