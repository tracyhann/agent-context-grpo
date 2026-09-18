#!/usr/bin/env python3
"""Evaluation-only WebShop replay, with isolated worker/collector trace hooks.
Uses the existing trainer and frozen FSDP checkpoint. Never updates or saves weights.
"""
import argparse
import copy
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def plain(value):
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [plain(v) for v in value]
    if hasattr(value, 'tolist'):
        return value.tolist()
    return value


def install_hooks(output):
    from agent_system.environments.env_package.webshop import envs as module
    from agent_system.multi_turn_rollout import TrajectoryCollector
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer

    class TraceWorker(module.WebshopWorker):
        def __init__(self, seed, env_kwargs):
            super().__init__(seed, env_kwargs)
            self.trace_seed = seed

        def reset(self, idx):
            obs, info = super().reset(idx)
            base = self.env.unwrapped
            info['trace_goal'] = plain(copy.deepcopy(base.server.user_sessions[base.session]['goal']))
            info['trace_goal_index'] = idx
            info['trace_worker_seed'] = self.trace_seed
            return obs, info

        def step(self, action):
            import re
            base = self.env.unwrapped
            old_session = base.session
            available = copy.deepcopy(base.get_available_actions())
            before = copy.deepcopy(base.server.user_sessions[old_session])
            match = re.fullmatch(r'(search|click)\[(.*)\]', action, flags=re.DOTALL)
            legal = bool(match and ((match[1] == 'search' and bool(match[2])) or
                         (match[1] == 'click' and match[2].lower() in available['clickables'] and match[2].lower() != 'search')))
            trace = {'action': action, 'available_before': available,
                     'admissible': legal, 'url_before': base.state['url'],
                     'asin_before': before.get('asin'), 'options_before': before.get('options')}
            obs, reward, done, info = super().step(action)
            after = base.server.user_sessions[old_session]
            trace['asin_after'] = after.get('asin')
            trace['options_after'] = copy.deepcopy(after.get('options'))
            if done:
                asin = after['asin']
                trace['purchase'] = plain({'asin': asin, 'product': base.server.product_item_dict[asin],
                    'price': base.server.product_prices[asin], 'options': after['options'],
                    'score': after['reward'], 'components': after['verbose_info'],
                    'visited_asins': after['asins'], 'action_counts': after['actions']})
            info['trace'] = plain(trace)
            return obs, reward, done, info

    module.WebshopWorker = TraceWorker
    # Evaluation never uses the training environments. Avoid creating their
    # 128 Ray actors; retain the original validation batch size and worker seeds.
    import agent_system.environments as environments
    original_make_envs = environments.make_envs
    def validation_envs_only(config):
        from omegaconf import OmegaConf
        small = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
        small.data.train_batch_size = 1
        small.env.rollout.n = 1
        unused_train, validation = original_make_envs(small)
        import ray
        for worker in unused_train.envs._workers:
            ray.kill(worker)
        validation.config = config
        return None, validation
    environments.make_envs = validation_envs_only
    original = TrajectoryCollector.vanilla_multi_turn_loop
    batch_counter = [0]

    def traced_loop(self, gen_batch, actor_rollout_wg, envs):
        original_reset, original_step = envs.reset, envs.step
        capture = {'steps': []}
        def reset(*args, **kwargs):
            obs, infos = original_reset(*args, **kwargs)
            capture['initial_obs'] = copy.deepcopy(obs)
            capture['initial_infos'] = copy.deepcopy(infos)
            capture['previous_obs'] = copy.deepcopy(obs)
            return obs, infos
        def step(actions):
            # The projection mutates the list, so preserve model responses first.
            raw = list(actions)
            obs, rewards, dones, infos = original_step(actions)
            capture['steps'].append({'raw': raw, 'obs': capture['previous_obs'],
                'next_obs': copy.deepcopy(obs), 'rewards': plain(rewards), 'dones': plain(dones),
                'infos': copy.deepcopy(infos)})
            capture['previous_obs'] = copy.deepcopy(obs)
            return obs, rewards, dones, infos
        envs.reset, envs.step = reset, step
        try:
            result = original(self, gen_batch, actor_rollout_wg, envs)
        finally:
            envs.reset, envs.step = original_reset, original_step
        rows, rewards, lengths, success, _, _ = result
        episodes = []
        for slot, length in enumerate(lengths):
            initial = capture['initial_infos'][slot]
            steps = []
            for turn, frame in enumerate(capture['steps'][:int(length)]):
                info = frame['infos'][slot]
                steps.append({'turn': turn, 'response': frame['raw'][slot],
                    'observation': frame['obs']['anchor'][slot],
                    'next_observation': frame['next_obs']['anchor'][slot],
                    'prompt': self.tokenizer.decode(rows[slot][turn]['prompts'], skip_special_tokens=True),
                    'reward': frame['rewards'][slot], 'done': frame['dones'][slot],
                    'task_score': info['task_score'], 'format_valid': plain(info['is_action_valid']),
                    **info['trace']})
            episodes.append({'batch': batch_counter[0], 'slot': slot,
                'goal_index': initial['trace_goal_index'], 'worker_seed': initial['trace_worker_seed'],
                'goal': initial['trace_goal'], 'length': int(length), 'reward': float(rewards[slot]),
                'success': bool(rewards[slot] == 10), 'task_score': max(s['task_score'] for s in steps),
                'steps': steps})
        with (output / 'episodes.jsonl').open('a') as stream:
            for episode in episodes:
                stream.write(json.dumps(plain(episode), ensure_ascii=False) + '\n')
        print(f"[trace] batch={batch_counter[0]} episodes={len(episodes)} successes={sum(e['success'] for e in episodes)}", flush=True)
        batch_counter[0] += 1
        return result
    TrajectoryCollector.vanilla_multi_turn_loop = traced_loop

    def validate_only(self):
        assert self.config.trainer.val_only and self.config.trainer.val_before_train
        self.global_steps = 0
        self._load_checkpoint()
        metrics = self._validate()
        (output / 'validation_metrics.json').write_text(json.dumps(plain(metrics), indent=2) + '\n')
        print('[trace] EVALUATION COMPLETE ' + json.dumps(plain(metrics)), flush=True)
    RayPPOTrainer.fit = validate_only


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    current = int(Path('/sys/fs/cgroup/pids.current').read_text())
    limit = int(Path('/sys/fs/cgroup/pids.max').read_text())
    if limit - current < 14000:
        raise RuntimeError('Insufficient thread headroom for diagnostic evaluation')
    assert not (output / 'episodes.jsonl').exists(), 'Refusing to mix replay traces'
    env = json.loads((source / 'config.json').read_text())['env']
    os.environ.update({k: str(v) for k,v in env.items()})
    os.environ.update(CUDA_VISIBLE_DEVICES='2,3', ACG_EXP_DIR=str(output),
        ACG_METRICS_JSONL=str(output/'metrics.jsonl'), ACG_CCPO_DUMP='',
        ACG_ALIGN_VAL_ON_RESUME='0', ACG_ALIGN_TRAIN_ON_RESUME='0',
        RAY_TMPDIR='/tmp/ray_ws_trace_' + output.name, RAY_ADDRESS='local',
        TORCHINDUCTOR_CACHE_DIR=str(ROOT/'.cache/inductor/webshop-failure-replay'),
        VLLM_CACHE_ROOT=str(ROOT/'.cache/vllm/webshop-failure-replay'))
    # Runtime environment setup precedes importing torch, Ray, or verl.
    import sys
    sys.path.insert(0, str(ROOT/'verl-agent'))
    sys.path.insert(0, str(ROOT))
    import ray
    from omegaconf import OmegaConf
    from verl.trainer import main_ppo
    cfg = OmegaConf.create(json.loads((source/'outputs/resolved_config.json').read_text()))
    cfg.trainer.val_only = True
    cfg.trainer.val_before_train = True
    cfg.trainer.resume_mode = 'resume_path'
    cfg.trainer.resume_from_path = str(source/'outputs/checkpoints/global_step_150')
    cfg.trainer.default_local_dir = str(output/'checkpoints-unused')
    cfg.trainer.experiment_name = output.name
    cfg.trainer.save_freq = -1
    assert cfg.trainer.n_gpus_per_node == 2
    (output/'eval_config.json').write_text(OmegaConf.to_json(cfg) if hasattr(OmegaConf,'to_json') else json.dumps(OmegaConf.to_container(cfg),indent=2))
    (output/'provenance.json').write_text(json.dumps({'source': str(source),
        'checkpoint': cfg.trainer.resume_from_path, 'evaluation_only': True,
        'draw': 'first two validation batches, seed 1000; worker seed 1000+slot',
        'training_code_modified': False, 'gpus': [2,3]}, indent=2))
    @ray.remote(num_cpus=1)
    class EvaluationRunner:
        def run(self, config):
            install_hooks(output)
            return main_ppo.TaskRunner.__ray_metadata__.modified_class.run(self, config)
    # Keep the original TaskRunner class accessible inside EvaluationRunner.
    from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
    ray.init(num_cpus=64, runtime_env=get_ppo_ray_runtime_env())
    try:
        runner = EvaluationRunner.remote()
        ray.get(runner.run.remote(cfg))
    finally:
        ray.shutdown()


if __name__ == '__main__':
    main()
