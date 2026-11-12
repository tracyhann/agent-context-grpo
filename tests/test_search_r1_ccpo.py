#!/usr/bin/env python3
"""CPU checks for the Search-R1 port of the main method.

1  SearchEnvironmentManager scales the exact-match reward only when asked to
2  on a search-shaped batch the unchanged estimator recognises success at 10/0,
   reads every terminal potential as failure at 1/0, and stops when the launcher's
   reward scale did not reach the environment
3  the prepare script builds all four arms, keeps the CCPO estimator identical to
   the ALFWorld control, and never launches
4  the validation subset keeps the first N rows of every dataset
5  the real environment, driven against a stub retriever, returns the same
   observation for the same query and rewards on the 10/0 scale

    .venv/bin/python -m unittest discover -s tests -p test_search_r1_ccpo.py

Sections 2 and 4 need torch and pyarrow; section 5 needs the synced verl-agent
overlay with gym, omegaconf and requests. Each is skipped when unavailable.
"""
import ast
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Dict, List, Tuple
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'tests'), str(ROOT/'scripts')]
import exp_run as er
import prepare_search_r1_ccpo as prep
import subset_search_val

try:
    import torch
    from ccpo import core_ccpo as core
    from ccpo.future_progress import (ccpo_future_progress_advantage, check_search_reward_scale,
                                      progress_from_values)
except ImportError:
    torch = None
try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:
    pa = None

SCALE_KEY = 'ACG_SEARCH_REWARD_SCALE'


# ---------------------------------------------------------------------------
# 1. the environment hook
# ---------------------------------------------------------------------------
def search_manager_class():
    """Compile the real class from the overlay without importing the runtime."""
    source = ROOT/'patches/verl-agent/agent_system/environments/env_manager.py'
    tree = ast.parse(source.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'SearchEnvironmentManager')

    class Base:
        def __init__(self, envs, projection_f, config):
            self.envs, self.projection_f, self.config = envs, projection_f, config

    class Memory:
        def reset(self, batch_size):
            self.rows = [[] for _ in range(batch_size)]

        def store(self, record):
            for i, row in enumerate(self.rows):
                row.append({k: v[i] for k, v in record.items()})

        def __getitem__(self, i):
            return self.rows[i]

        def fetch(self, history_length, obs_key, action_key):
            return ['history'] * len(self.rows), [len(r) for r in self.rows]

    namespace = dict(os=os, EnvironmentManagerBase=Base, SearchMemory=Memory, to_numpy=np.array,
                     Tuple=Tuple, Dict=Dict, List=List, Any=Any,
                     SEARCH_TEMPLATE='{task_description}{memory_context}{step_count}',
                     SEARCH_TEMPLATE_NO_HIS='{task_description}')
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), namespace)
    return namespace['SearchEnvironmentManager']


class FakeSearchEnvs:
    """Three environments: still searching, exact match, wrong answer."""
    rewards = [0, 1.0, 0.0]

    def reset(self, kwargs):
        return [k['question'] for k in kwargs], [{} for _ in kwargs]

    def step(self, actions):
        return (['<information>passage</information>', '', ''], list(self.rewards), [False, True, True],
                [{'won': False}, {'won': True}, {'won': False}])


def step_once(environment):
    with patch.dict(os.environ, environment):     # restores the caller's environment on exit
        if SCALE_KEY not in environment:
            os.environ.pop(SCALE_KEY, None)
        config = type('C', (), {'env': type('E', (), {'history_length': 4})()})()
        manager = search_manager_class()(FakeSearchEnvs(), lambda a: (a, [1] * len(a)), config)
    manager.reset([{'question': f'q{i}'} for i in range(3)])
    return manager.step(['<search>x</search>', '<answer>y</answer>', '<answer>z</answer>'])


class RewardScaleHookTests(unittest.TestCase):
    def test_scale_is_applied_when_the_launcher_sets_it(self):
        _, rewards, dones, infos = step_once({SCALE_KEY: '10.0'})
        np.testing.assert_array_equal(rewards, [0., 10., 0.])
        self.assertEqual(dones.tolist(), [False, True, True])
        # Success is decided upstream on the unscaled reward and is not rescaled.
        self.assertEqual([i['won'] for i in infos], [False, True, False])
        self.assertTrue(all('is_action_valid' in i for i in infos))

    def test_unset_keeps_the_upstream_reward(self):
        _, rewards, _, _ = step_once({})
        expected = np.array(FakeSearchEnvs.rewards)
        np.testing.assert_array_equal(rewards, expected)
        self.assertEqual(rewards.dtype, expected.dtype)

    def test_launcher_scale_is_the_estimators_success_potential(self):
        import inspect
        if torch is None:
            self.skipTest('needs torch to import the estimator')
        default = inspect.signature(progress_from_values).parameters['success_reward'].default
        self.assertEqual(er.SEARCH_FIXED['reward_scale'], default)


# ---------------------------------------------------------------------------
# 2. the unchanged estimator on a search-shaped batch
# ---------------------------------------------------------------------------
MAIN_METHOD = dict(_LK_FIX='1.0', _LAM_FIX='1.0', _LOO='1', _GATE='hard', _WMODE='soft',
                   _PHI_MODE='hidden+ctx', _CTX_W=1., _WHITEN_K=3, _JW_C=0., _STD_MODE='task',
                   _BACKOFF_TASK=True, _SIM=0., _SIM_BACKOFF=0., _EDGE_W=0., _TAU_ENV='.15',
                   _PRIOR_KAPPA=2., _TARGET='return')
LENGTHS = [2, 3, 3, 4, 2]       # turns per rollout; the protocol's group of five, at most four turns


def search_batch(success_reward):
    """Three questions x five rollouts. Turn 0 sees the question; later turns see
    retrieved passages. Rollouts 0 and 1 issue the same first query, so their turn-1
    observations match exactly; every other retrieval is unique to its rollout."""
    obs, tasks, trajs, turns, lens, episodes, returns = [], [], [], [], [], [], []
    for q in range(3):
        for r, length in enumerate(LENGTHS):
            outcome = success_reward if (q + r) % 2 == 0 else 0.
            for t in range(length):
                query = 'shared' if (t == 1 and r in (0, 1)) else f'r{r}t{t}'
                obs.append(f'question {q}' if t == 0 else f'<information>q{q} {query}</information>')
                tasks.append(f'task{q}'); trajs.append(f'task{q}-traj{r}')
                turns.append(t); lens.append(length); episodes.append(outcome)
                returns.append(.95 ** (length - 1 - t) * outcome)
    n = len(obs)
    return dict(step_rewards=torch.tensor(returns, dtype=torch.float32), response_mask=torch.ones(n, 4),
                anchor_obs=np.array(obs), index=np.array(tasks), traj_index=np.array(trajs),
                turn_index=np.array(turns), episode_lengths=np.array(lens),
                episode_rewards=np.array(episodes), is_action_valid=np.ones(n, dtype=bool),
                phi_feats=torch.tensor(np.random.default_rng(7).normal(size=(n, 16)), dtype=torch.float32),
                phi=core.FrozenPhi(), gamma=.95, return_diag=True)


@unittest.skipIf(torch is None, 'needs torch')
class SearchShapedEstimatorTests(unittest.TestCase):
    def setUp(self):
        patches = patch.multiple(core, **MAIN_METHOD); patches.start(); self.addCleanup(patches.stop)
        saved = os.environ.pop(SCALE_KEY, None)
        self.addCleanup(lambda: os.environ.__setitem__(SCALE_KEY, saved) if saved is not None
                        else os.environ.pop(SCALE_KEY, None))

    def test_partial_recurrence_uses_exact_peers_where_they_exist_and_the_task_bucket_elsewhere(self):
        kw = search_batch(10.)
        advantage, diag = ccpo_future_progress_advantage(**kw, horizon=2)
        a = diag['progress_payload']['arrays']
        self.assertTrue(torch.isfinite(advantage).all())
        self.assertEqual(len(advantage), len(kw['anchor_obs']))
        shared = np.char.find(a['anchor_obs'], 'shared') >= 0
        exact = (a['turn_index'] == 0) | shared
        np.testing.assert_array_equal(a['current_level'][exact], 0)
        self.assertTrue((a['current_level'][~exact] > 0).all())
        self.assertTrue(np.isfinite(a['current_value']).all() and np.isfinite(a['history_baseline']).all())
        self.assertAlmostEqual(diag['progress_current_exact_frac'], exact.mean())
        self.assertAlmostEqual(diag['progress_current_exact_frac'] + diag['progress_current_backoff_frac'], 1.)
        self.assertTrue(0 < diag['progress_current_exact_frac'] < 1)
        self.assertEqual(diag['progress_live_frac'], 1.)

    def test_success_potential_requires_the_ten_zero_scale(self):
        for reward, potential in ((10., 10.), (1., 0.)):
            with self.subTest(reward=reward):
                kw = search_batch(reward)
                _, diag = ccpo_future_progress_advantage(**kw, horizon=2)
                a = diag['progress_payload']['arrays']
                won = a['terminal'] & (a['episode_rewards'] > 0)
                self.assertTrue(won.any())
                # At 1/0 an exact-match success is indistinguishable from a failure.
                np.testing.assert_array_equal(a['future_value'][won], potential)
                np.testing.assert_array_equal(a['future_value'][a['terminal'] & ~won], 0.)

    def test_stale_overlay_is_stopped_instead_of_training_on_zero_potentials(self):
        with patch.dict(os.environ, {SCALE_KEY: '10.0'}):
            with self.assertRaisesRegex(ValueError, 'sync_patches'):
                ccpo_future_progress_advantage(**search_batch(1.), horizon=2)
            scaled, _ = ccpo_future_progress_advantage(**search_batch(10.), horizon=2)
        plain, _ = ccpo_future_progress_advantage(**search_batch(10.), horizon=2)
        torch.testing.assert_close(scaled, plain, rtol=0, atol=0)

    def test_guard_is_inactive_outside_search(self):
        check_search_reward_scale(np.array([0., 1., 7.5]))          # variable unset: any scale passes
        with patch.dict(os.environ, {SCALE_KEY: '10.0'}):
            check_search_reward_scale(np.array([0., 10., 0.]))
            check_search_reward_scale(np.zeros(4))                  # an all-failure batch cannot be judged
            with self.assertRaises(ValueError):
                check_search_reward_scale(np.array([0., 10., 1.]))


# ---------------------------------------------------------------------------
# 3. the prepare script
# ---------------------------------------------------------------------------
STAMP = '19990101'      # a date no real preparation uses; folders are removed after each test


class PrepareScriptTests(unittest.TestCase):
    def prepare(self, arm, **kwargs):
        options = dict(backbone='7b', gpus='0,1,2,3,4,5,6,7', steps=er.SEARCH_PROTOCOL['total_epochs'],
                       stamp=STAMP, overrides={}, skip_runtime_check=True)
        options.update(kwargs)
        name = prep.run_name(arm, options['backbone'], options['gpus'], STAMP, options.get('seed', 0))
        folder = ROOT/'experiments'/name
        if folder.exists():
            self.skipTest(f'{folder} already exists')
        self.addCleanup(shutil.rmtree, folder, True)
        return prep.prepare(arm, **options)

    def read(self, folder, name='config.json'):
        return json.loads((folder/name).read_text())

    def test_all_arms_share_one_config_and_differ_in_the_estimator_alone(self):
        records = {arm: self.read(self.prepare(arm)) for arm in prep.ARMS}
        base = records['ccpo']['config']
        for arm, expected in (('grpo', {'arm', 'exp_id'}), ('gigpo', {'arm', 'exp_id'}),
                              ('ccpo-ep1', {'ccpo_ep_w', 'exp_id'})):
            other = records[arm]['config']
            self.assertEqual(set(other), set(base))
            self.assertEqual({k for k in base if base[k] != other[k]}, expected, arm)
        for arm, record in records.items():
            flags = record['hydra_overrides']
            self.assertIn(f'algorithm.adv_estimator={prep.ARMS[arm]["arm"]}', flags)
            self.assertEqual('algorithm.gigpo.enable_similarity=True' in flags, arm == 'gigpo')
            self.assertIn('trainer.total_training_steps=662', flags)
            self.assertIn('actor_rollout_ref.rollout.tensor_model_parallel_size=2', flags)
            self.assertEqual(float(record['env'][SCALE_KEY]), 10.)
            self.assertEqual(record['reference_protocol']['departures'], {})
            self.assertEqual(record['preparation']['status'], 'prepared_not_launched')
            self.assertFalse(record['preparation']['launched'])
        self.assertEqual(base['ccpo_ep_w'], 0)
        self.assertEqual(records['ccpo-ep1']['config']['ccpo_ep_w'], 1)

    def test_ccpo_estimator_equals_the_alfworld_control(self):
        folder = self.prepare('ccpo')
        new = self.read(folder)['config']
        control = self.read(ROOT/'experiments'/prep.ARMS['ccpo']['control'])['config']
        for key, value in prep.IMPLIED.items():
            control.setdefault(key, value)
        for key in control:
            if key.startswith('ccpo_') and key != 'ccpo_progress_snapshot_every':
                self.assertEqual(new[key], control[key], key)
        changed = set(self.read(folder, 'config-diff-from-alfworld-control.json')['changes'])
        self.assertFalse({k for k in changed if k.startswith('ccpo_')} - {'ccpo_progress_snapshot_every'})
        self.assertTrue({'env_name', 'max_steps', 'history_length', 'group_size', 'train_batch_size',
                         'model', 'kl_loss_coef', 'max_prompt_length'} <= changed)
        self.assertEqual((new['env_name'], new['max_steps'], new['history_length'], new['group_size']),
                         ('search', 4, 4, 5))
        self.assertEqual((new['ccpo_gate'], new['ccpo_sim'], new['ccpo_backoff_task'], new['ccpo_target']),
                         ('hard', 0, 1, 'return'))
        self.assertEqual((new['adv_mode'], new['gamma'], new['pin_steps']), ('mean_std_norm', .95, ''))
        notes = (folder/'NOTES.md').read_text()
        for text in ('Prepared only', 'progress_current_exact_frac', 'ACG_SEARCH_REWARD_SCALE', 'run.sh'):
            self.assertIn(text, notes)
        self.assertTrue((folder/'run.sh').exists() and (folder/'PREPARED.json').exists())
        self.assertFalse((folder/'outputs/train.pid').exists())

    def test_budget_overrides_are_recorded_as_departures(self):
        folder = self.prepare('ccpo', backbone='3b', gpus='0,1,2,3', steps=200, seed=1,
                              overrides={'test_freq': 25, 'search_val_files': '/data/subset.parquet'})
        self.assertIn('-3b-4gpu-seed1-', folder.name)
        record = self.read(folder)
        self.assertEqual(record['reference_protocol']['departures'],
                         {'total_epochs': {'reference': 662, 'launched': 200},
                          'test_freq': {'reference': 50, 'launched': 25}})
        self.assertEqual(record['config']['model'], 'Qwen/Qwen2.5-3B-Instruct')
        self.assertEqual((record['config']['seed'], record['config']['tp_size']), (1, 1))
        self.assertIn('data.val_files=/data/subset.parquet', record['hydra_overrides'])
        self.assertIn('total_epochs', (folder/'NOTES.md').read_text())

    def test_estimator_keys_cannot_be_overridden(self):
        for overrides, message in (({'ccpo_sim': .9}, 'cannot be overridden'), ({'no_such_key': 1}, 'Unknown')):
            with self.assertRaisesRegex(ValueError, message):
                prep.resolve('ccpo', '7b', '0,1,2,3,4,5,6,7', 662, overrides)
        self.assertEqual(prep.resolve('ccpo', '7b', '0,1', 10, {'ccpo_progress_snapshot_every': 5})
                         ['ccpo_progress_snapshot_every'], 5)

    def test_unsynced_overlay_blocks_preparation(self):
        if not prep.stale_overlay_files():
            self.skipTest('verl-agent/ is in sync with patches/verl-agent/')
        with self.assertRaisesRegex(RuntimeError, 'sync_patches'):
            self.prepare('ccpo', skip_runtime_check=False)


# ---------------------------------------------------------------------------
# 4. the validation subset
# ---------------------------------------------------------------------------
@unittest.skipIf(pa is None, 'needs pyarrow')
class ValidationSubsetTests(unittest.TestCase):
    def test_first_rows_of_every_dataset_with_the_schema_intact(self):
        sources = ['nq'] * 5 + ['hotpotqa'] * 2 + ['nq'] * 3 + ['bamboogle'] * 4
        rows = [dict(data_source=s, prompt=[dict(role='user', content=f'q{i}')],
                     env_kwargs=dict(question=f'q{i}', data_source=s, ground_truth=dict(target=[f'a{i}'])))
                for i, s in enumerate(sources)]
        with tempfile.TemporaryDirectory() as tmp:
            source, out = Path(tmp)/'test.parquet', Path(tmp)/'subset.parquet'
            pq.write_table(pa.Table.from_pylist(rows), source)
            counts, total = subset_search_val.subset(source, out, 3)
            self.assertEqual((counts, total), ({'nq': 3, 'hotpotqa': 2, 'bamboogle': 3}, 14))
            kept = pq.read_table(out)
            self.assertEqual(kept.schema, pq.read_table(source).schema)
            self.assertEqual([r['env_kwargs']['question'] for r in kept.to_pylist()],
                             ['q0', 'q1', 'q2', 'q5', 'q6', 'q10', 'q11', 'q12'])
            with self.assertRaises(FileExistsError):
                subset_search_val.subset(source, out, 3)


# ---------------------------------------------------------------------------
# 5. the real environment against a stub retriever
# ---------------------------------------------------------------------------
def overlay_ready():
    try:
        import gym, omegaconf, requests  # noqa: F401
    except ImportError:
        return False
    return torch is not None and not prep.stale_overlay_files()


def stub_retriever():
    """A local /retrieve endpoint: three passages that depend on the query alone."""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    import threading

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            query = json.loads(self.rfile.read(int(self.headers['Content-Length'])))['query']
            body = json.dumps({'result': [[{'document': {'contents': f'"{query}"\npassage {i} about {query}'},
                                            'score': 1. / (i + 1)} for i in range(3)]]}).encode()
            self.send_response(200); self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body))); self.end_headers(); self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


@unittest.skipUnless(overlay_ready(), 'needs the synced verl-agent overlay with gym, omegaconf and requests')
class RealEnvironmentTests(unittest.TestCase):
    def test_two_turn_episode_through_the_real_manager(self):
        from omegaconf import OmegaConf
        if str(ROOT/'verl-agent') not in sys.path:
            sys.path.insert(0, str(ROOT/'verl-agent'))
        from agent_system.environments.env_manager import make_envs
        server = stub_retriever(); self.addCleanup(server.server_close); self.addCleanup(server.shutdown)
        config = OmegaConf.create(dict(
            data=dict(train_batch_size=2, val_batch_size=2),
            env=dict(env_name='search', seed=0, max_steps=4, history_length=4, rollout=dict(n=5),
                     resources_per_worker=dict(num_cpus=.1, num_gpus=0),
                     search=dict(log_requests=False, topk=3, timeout=10,
                                 search_url=f'http://127.0.0.1:{server.server_address[1]}/retrieve'))))
        with patch.dict(os.environ, {SCALE_KEY: str(er.SEARCH_FIXED['reward_scale'])}):
            envs, val_envs = make_envs(config)
        self.addCleanup(envs.close); self.addCleanup(val_envs.close)
        questions = ['who wrote the book', 'capital of the country']
        obs, _ = envs.reset([dict(question=q, ground_truth={'target': ['Right Answer']}, data_source='nq')
                             for q in questions for _ in range(5)])
        self.assertEqual(list(obs['anchor']), [q for q in questions for _ in range(5)])

        # Turn 1. In each group rollouts 0 and 1 issue the same query; the others differ.
        queries = ['same query', 'same query', 'second', 'third', 'fourth'] * 2
        obs, rewards, dones, infos = envs.step([f'<think>t</think><search>{q}</search>' for q in queries])
        anchors = list(obs['anchor'])
        self.assertEqual(anchors[0], anchors[1])            # same query -> byte-identical observation
        self.assertEqual(len(set(anchors[:5])), 4)
        self.assertTrue(all(a.startswith('<information>') and a.endswith('</information>') for a in anchors))
        np.testing.assert_array_equal(rewards, 0); self.assertFalse(dones.any())
        self.assertIn('same query', obs['text'][0])         # the retrieved passages reach the next prompt

        # Turn 2: correct, wrong, correct, another search, and a reply with no action tag.
        replies = ['<answer>right answer</answer>', '<answer>wrong</answer>', '<answer>The Right Answer</answer>',
                   '<search>more</search>', 'no tags at all'] * 2
        _, rewards, dones, infos = envs.step(replies)
        np.testing.assert_array_equal(rewards, [10., 0., 10., 0., 0.] * 2)
        self.assertEqual(dones.tolist(), [True, True, True, False, False] * 2)
        self.assertEqual([bool(i['won']) for i in infos], [True, False, True, False, False] * 2)
        self.assertEqual([bool(i['is_action_valid']) for i in infos], [True, True, True, True, False] * 2)


if __name__ == '__main__':
    unittest.main()
