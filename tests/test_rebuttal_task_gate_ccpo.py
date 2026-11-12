#!/usr/bin/env python3
"""Rebuttal task-gate ablation: observation matching removed, task identity kept.

1  the launcher and the registry accept `ccpo_gate=global` under future progress,
   as a one-key change from the no-shrink H2 controls
2  the estimate equals the hard gate's task fallback applied to every turn, matches
   an independent kernel computation, ignores observation labels, and stays inside
   the task and outside the query trajectory
3  the prepare script changes `ccpo_gate` alone and never launches
4  the real trainer routes the gate (needs the synced verl-agent overlay)

    .venv/bin/python -m unittest discover -s tests -p test_rebuttal_task_gate_ccpo.py

Sections 2 and 4 need torch; section 4 is skipped without the overlay.
"""
from contextlib import contextmanager, redirect_stdout
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'tests'), str(ROOT/'scripts')]
import exp_run as er
import prepare_rebuttal_task_gate_ccpo as prep

try:
    import torch
    from ccpo import core_ccpo as core
    from ccpo import future_progress as fp
except ImportError:
    torch = None

MAIN = 'future-progress-h2-no-credit-shrinkage'
PARENTS = {'alfworld': MAIN, 'webshop': MAIN + '-active-episode'}


# ---------------------------------------------------------------------------
# 1. launcher and registry
# ---------------------------------------------------------------------------
class ConfigurationTests(unittest.TestCase):
    def config(self, benchmark='alfworld', **overrides):
        cfg = dict(er.DEFAULTS)
        cfg.update(prep.ablations.build(prep.KEYS[benchmark], benchmark)[1])
        cfg.update(overrides)
        return cfg

    def test_registry_changes_the_gate_alone(self):
        for benchmark, parent in PARENTS.items():
            _, arm = prep.ablations.build(prep.KEYS[benchmark], benchmark)
            _, control = prep.ablations.build(parent, benchmark)
            self.assertEqual(set(arm), set(control))
            self.assertEqual({k: (control[k], arm[k]) for k in arm if arm[k] != control[k]},
                             {'ccpo_gate': ('hard', 'global')}, benchmark)
            self.assertEqual(arm['ccpo_ep_w'], int(benchmark == 'webshop'))
        with self.assertRaises(KeyError):
            prep.ablations.build(prep.KEYS['alfworld'], 'webshop')

    def test_launcher_accepts_the_gate_and_exports_it(self):
        cfg = self.config()
        er.validate_future_progress_config(cfg)
        env = er.build_env(dict(cfg, gpus='0,1', alfworld_data='/a', hf_home='/h',
                                venv_python='/nonexistent/python3'), '/tmp/exp')
        self.assertEqual(env['ACG_CCPO_GATE'], 'global')

    def test_unsupported_gates_stop_at_launch_for_future_progress_only(self):
        for gate in ('sig', 'task', 'soft'):
            with self.subTest(gate=gate), self.assertRaisesRegex(ValueError, 'ccpo_gate'):
                er.validate_future_progress_config(self.config(ccpo_gate=gate))
        for gate in er.FUTURE_PROGRESS_GATES:
            if gate != 'cross_task':        # cross_task has further requirements of its own
                er.validate_future_progress_config(self.config(ccpo_gate=gate))
        # Arms without future progress keep whatever gate they declare.
        er.validate_future_progress_config(dict(er.DEFAULTS, ccpo_gate='sig'))

    def test_gate_lists_agree(self):
        if torch is None:
            self.skipTest('needs torch to import the estimator')
        self.assertEqual(er.FUTURE_PROGRESS_GATES, fp.SUPPORTED_GATES)


# ---------------------------------------------------------------------------
# 2. the estimator
# ---------------------------------------------------------------------------
NO_SHRINK = dict(_LK_FIX='1.0', _LAM_FIX='1.0', _LOO='1', _WMODE='soft', _PHI_MODE='hidden+ctx',
                 _CTX_W=1., _WHITEN_K=3, _JW_C=0., _STD_MODE='task', _BACKOFF_TASK=True, _SIM=0.,
                 _SIM_BACKOFF=0., _EDGE_W=0., _TAU_ENV='.15', _PRIOR_KAPPA=2., _TARGET='return')
LENGTHS = [5, 4, 3, 5]


@contextmanager
def gate(name, **overrides):
    with patch.multiple(core, **dict(NO_SHRINK, _GATE=name, **overrides)), redirect_stdout(io.StringIO()):
        yield


def batch():
    """Two tasks x four rollouts. Observations recur across rollouts ('room0',
    'room1', 'room2') and inside a rollout (a revisit of 'room0')."""
    obs, tasks, trajs, turns, lens, episodes, returns = [], [], [], [], [], [], []
    for q in range(2):
        for r, length in enumerate(LENGTHS):
            outcome = 10. if (q + r) % 2 == 0 else 0.
            for t in range(length):
                obs.append('room0' if t in (0, 3) else f'room{1 + (t + r) % 2}')
                tasks.append(f'task{q}'); trajs.append(f'task{q}-traj{r}')
                turns.append(t); lens.append(length); episodes.append(outcome)
                returns.append(.95 ** (length - 1 - t) * outcome)
    n = len(obs)
    valid = np.ones(n, dtype=bool); valid[2] = False
    returns = np.array(returns); returns[2] -= .1
    return dict(step_rewards=torch.tensor(returns, dtype=torch.float32), response_mask=torch.ones(n, 4),
                anchor_obs=np.array(obs, dtype=object), index=np.array(tasks), traj_index=np.array(trajs),
                turn_index=np.array(turns), episode_lengths=np.array(lens),
                episode_rewards=np.array(episodes), is_action_valid=valid,
                phi_feats=torch.tensor(np.random.default_rng(21).normal(size=(n, 16)), dtype=torch.float32),
                phi=core.FrozenPhi(), gamma=.95, return_diag=True)


def without_cross_rollout_matches(kw):
    """Tag every observation with its rollout: no observation is shared between
    rollouts any more, while repeats inside a rollout, and with them the context
    statistics, are untouched."""
    return dict(kw, anchor_obs=np.array([f'{t}|{o}' for t, o in zip(kw['traj_index'], kw['anchor_obs'])],
                                        dtype=object))


def estimate(kw):
    advantage, diag = fp.ccpo_future_progress_advantage(**kw, horizon=2)
    return advantage, diag, diag['progress_payload']['arrays']


READOUTS = ('history_baseline', 'history_adv', 'current_value', 'future_value', 'raw_progress',
            'progress_normalized', 'combined_pre')


@unittest.skipIf(torch is None, 'needs torch')
class TaskGateEstimatorTests(unittest.TestCase):
    def test_equals_the_hard_gate_fallback_applied_to_every_turn(self):
        kw = batch()
        with gate('hard'):
            fallback, fallback_diag, f = estimate(without_cross_rollout_matches(kw))
        with gate('global'):
            task, diag, a = estimate(kw)
            relabelled, _, b = estimate(without_cross_rollout_matches(kw))
        # With no shared observation the hard gate serves every turn from its task fallback...
        self.assertTrue((f['current_level'] == 1).all() and (f['history_level'] == 1).all())
        self.assertEqual(fallback_diag['progress_current_backoff_frac'], 1.)
        # ...and the task gate gives that same estimate without needing the observations to differ.
        torch.testing.assert_close(task, fallback, rtol=0, atol=0)
        for key in READOUTS:
            np.testing.assert_array_equal(a[key], f[key], err_msg=key)
            np.testing.assert_array_equal(a[key], b[key], err_msg=key)       # observation labels do not gate
        torch.testing.assert_close(task, relabelled, rtol=0, atol=0)
        self.assertTrue(a['live'].all())

    def test_differs_from_the_hard_gate_when_observations_recur(self):
        kw = batch()
        with gate('hard'):
            hard, hard_diag, h = estimate(kw)
        with gate('global'):
            task, diag, a = estimate(kw)
        self.assertGreater(hard_diag['progress_current_exact_frac'], .5)       # the batch does recur
        exact = h['current_level'] == 0
        self.assertGreater(np.max(abs(a['history_baseline'][exact] - h['history_baseline'][exact])), .1)
        self.assertFalse(torch.allclose(task, hard))
        # One level, which is neither an exact match nor a fallback.
        np.testing.assert_array_equal(a['current_level'], 0)
        self.assertEqual((diag['progress_current_exact_frac'], diag['progress_current_backoff_frac']), (0., 0.))
        self.assertEqual((diag['progress_task_grouping'], hard_diag['progress_task_grouping']), (1., 0.))
        self.assertEqual(diag['progress_cross_task_grouping'], 0.)
        for i in range(len(a['uid'])):
            peers = (a['uid'] == a['uid'][i]) & (a['traj_uid'] != a['traj_uid'][i])
            self.assertEqual(a['current_peer_rows'][i], peers.sum())
            self.assertEqual(a['history_J'][i], len(LENGTHS) - 1)
            self.assertEqual(a['history_same_traj_mass'][i], 0)

    def test_independent_kernel_computation(self):
        with gate('global'):
            _, _, a = estimate(batch())
        features = a['current_phi']
        distances = np.linalg.norm(features[:, None] - features[None, :], axis=-1)
        values = a['value_target'].astype(np.float32).astype(float)     # the estimator reads float32 targets
        for i in range(len(features)):
            bucket = np.flatnonzero(a['uid'] == a['uid'][i])
            pairs = distances[np.ix_(bucket, bucket)][np.triu_indices(len(bucket), 1)]
            tau = .15 * np.median(pairs)
            peers = bucket[a['traj_uid'][bucket] != a['traj_uid'][i]]
            weights = np.exp(-distances[i, peers] / tau); weights /= weights.sum()
            self.assertAlmostEqual(a['history_baseline'][i], weights @ a['target'][peers], places=7)
            self.assertAlmostEqual(a['current_value'][i], weights @ values[peers], places=7)
            self.assertEqual(a['history_lambda_k'][i], 1)
            self.assertTrue(np.isnan(a['history_task_prior'][i]))
        terminal, endpoint = a['terminal'], a['endpoint_index']
        np.testing.assert_array_equal(a['future_value'][~terminal], a['current_value'][endpoint[~terminal]])
        np.testing.assert_array_equal(a['future_value'][terminal], a['episode_rewards'][terminal])
        np.testing.assert_allclose(a['raw_progress'], a['future_value'] - a['current_value'])
        np.testing.assert_allclose(a['history_adv'], a['target'] - a['history_baseline'], atol=1e-12)

    def test_task_identity_is_kept_unlike_cross_task(self):
        kw, changed = batch(), batch()
        remote = changed['index'] == 'task1'
        changed['episode_rewards'][remote] += 4; changed['step_rewards'][remote] += 4
        with gate('global'):
            _, _, a = estimate(kw); _, _, b = estimate(changed)
        with gate('cross_task'):
            _, _, c = estimate(kw); _, _, d = estimate(changed)
        rows = a['uid'] == 'task0'
        for key in ('history_baseline', 'current_value', 'future_value'):
            np.testing.assert_array_equal(a[key][rows], b[key][rows], err_msg=key)
        self.assertGreater(np.max(abs(c['history_baseline'][c['uid'] == 'task0']
                                      - d['history_baseline'][d['uid'] == 'task0'])), .1)

    def test_whole_query_trajectory_is_excluded(self):
        kw, changed = batch(), batch()
        own = changed['traj_index'] == 'task0-traj0'
        changed['episode_rewards'][own] += 100; changed['step_rewards'][own] += 100
        with gate('global'):
            _, _, a = estimate(kw); _, _, b = estimate(changed)
        mine = a['traj_uid'] == 'task0-traj0'
        for key in ('history_baseline', 'current_value'):
            np.testing.assert_array_equal(a[key][mine], b[key][mine], err_msg=key)
        others = (a['uid'] == 'task0') & ~mine
        self.assertGreater(np.max(abs(a['history_baseline'][others] - b['history_baseline'][others])), 1.)

    def test_single_rollout_task_has_no_credit(self):
        kw = batch()
        keep = np.flatnonzero((kw['index'] == 'task0') | (kw['traj_index'] == 'task1-traj0'))
        subset = {k: (v[keep] if isinstance(v, (np.ndarray, torch.Tensor)) else v) for k, v in kw.items()}
        with gate('global'):
            advantage, _, a = estimate(subset)
        alone = a['uid'] == 'task1'
        self.assertFalse(a['live'][alone].any())
        np.testing.assert_array_equal(advantage.numpy()[np.asarray(subset['index']) == 'task1'], 0)
        self.assertTrue(a['live'][~alone].all())

    def test_other_gates_and_similarity_clustering_stay_rejected(self):
        for name, overrides in (('sig', {}), ('task', {}), ('global', {'_SIM': .9}), ('global', {'_SIM_BACKOFF': .9})):
            with self.subTest(gate=name, **overrides), gate(name, **overrides), self.assertRaises(ValueError):
                estimate(batch())


# ---------------------------------------------------------------------------
# 3. the prepare script
# ---------------------------------------------------------------------------
STAMP = '19990103'      # a date no real preparation uses; folders are removed after each test


def control_recorded_here(benchmark):
    """Preparation compares every key with the control, including its absolute paths,
    so it only succeeds in the checkout location the control was recorded in."""
    control = json.loads((ROOT/'experiments'/prep.CONTROLS[benchmark]/'config.json').read_text())['config']
    return str(control['venv_python']).startswith(str(ROOT) + os.sep)


class PrepareScriptTests(unittest.TestCase):
    def prepare(self, benchmark, **kwargs):
        if not control_recorded_here(benchmark):
            self.skipTest('the control was recorded under another checkout path; run on the training host')
        folder = ROOT/'experiments'/prep.run_name(benchmark, STAMP)
        if folder.exists():
            self.skipTest(f'{folder} already exists')
        self.addCleanup(shutil.rmtree, folder, True)
        return prep.prepare(benchmark, STAMP, **dict(dict(skip_runtime_check=True), **kwargs))

    def test_prepared_configs_change_only_the_gate_and_the_runtime_reads_it(self):
        for benchmark in prep.CONTROLS:
            with self.subTest(benchmark=benchmark):
                folder = self.prepare(benchmark)
                saved = json.loads((folder/'config.json').read_text()); cfg = saved['config']
                control = json.loads((ROOT/'experiments'/prep.CONTROLS[benchmark]/'config.json').read_text())['config']
                for key, value in prep.IMPLIED.items():
                    control.setdefault(key, value)
                self.assertEqual(set(cfg), set(control))
                self.assertEqual({k: (control[k], cfg[k]) for k in cfg if cfg[k] != control[k] and k != 'exp_id'},
                                 {'ccpo_gate': ('hard', 'global')})
                self.assertEqual((cfg['ccpo_ep_w'], cfg['model'], cfg['total_epochs']),
                                 (int(benchmark == 'webshop'), 'Qwen/Qwen2.5-1.5B-Instruct', 150))
                self.assertEqual(saved['hydra_overrides'], er.build_command(cfg, str(folder))[3:])
                self.assertIn('ACG_CCPO_GATE=global', (folder/'run.sh').read_text())
                self.assertEqual(saved['env']['ACG_CCPO_GATE'], 'global')
                status = saved['preparation']
                self.assertEqual((status['status'], status['launched'], status['queued']),
                                 ('prepared_not_launched', False, False))
                self.assertEqual((status['observation_matching'], status['task_matching']), (False, True))
                self.assertFalse((folder/'outputs/train.pid').exists())
                notes = (folder/'NOTES.md').read_text()
                for text in ('Prepared only', 'ccpo_gate: hard -> global', 'progress_task_grouping', 'run.sh'):
                    self.assertIn(text, notes)
                if torch is not None:
                    check = ("from ccpo import core_ccpo as c; assert c._GATE == 'global'; "
                             "assert c._WMODE == 'soft'; assert float(c._LK_FIX) == 1")
                    result = subprocess.run([sys.executable, '-c', check], cwd=ROOT, capture_output=True,
                                            text=True, timeout=120, env=dict(os.environ, **saved['env']))
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                with patch.object(prep.subprocess, 'run') as run:
                    with self.assertRaises(FileExistsError):
                        prep.prepare(benchmark, STAMP, skip_runtime_check=True)
                    run.assert_not_called()

    def test_unsynced_trainer_blocks_preparation(self):
        if prep.trainer_synced():
            self.skipTest('the live trainer matches patches/')
        # Raised before anything is read or written, so no folder needs cleaning up.
        with self.assertRaisesRegex(RuntimeError, 'sync_patches'):
            prep.prepare('alfworld', STAMP)
        self.assertFalse((ROOT/'experiments'/prep.run_name('alfworld', STAMP)).exists())


# ---------------------------------------------------------------------------
# 4. the real trainer
# ---------------------------------------------------------------------------
def overlay_ready():
    return torch is not None and prep.trainer_synced()


@unittest.skipUnless(overlay_ready(), 'needs torch and the synced verl-agent overlay')
class TrainerTests(unittest.TestCase):
    def test_actual_trainer_routes_the_gate_in_both_benchmarks(self):
        sys.path.insert(0, str(ROOT/'verl-agent'))
        from test_future_progress import load_compute_advantage, policy_gradient
        from test_future_progress_active_episode import reward_data
        for benchmark in prep.CONTROLS:
            with self.subTest(benchmark=benchmark), tempfile.TemporaryDirectory() as tmp:
                saved = json.loads((ROOT/'experiments'/prep.CONTROLS[benchmark]/'config.json').read_text())
                kw = batch(); kw['response_mask'][::2, -1] = 0
                kw['immediate_rewards'] = np.zeros(len(kw['index']))
                data = reward_data(kw)
                env = dict(saved['env'], ACG_CCPO_GATE='global', CUDA_VISIBLE_DEVICES='',
                           ACG_EXP_DIR=tmp, ACG_CCPO_DUMP='', ACG_CCPO_PROGRESS_SNAPSHOT_EVERY='1')
                with patch.dict(os.environ, env), gate('global'):
                    load_compute_advantage()(data, 'CCPO', gamma=.95, gigpo_mode=saved['config']['adv_mode'],
                                             ccpo_step_tag=1)
                with np.load(Path(tmp)/'outputs/future_progress/step-0001.npz', allow_pickle=False) as file:
                    a = {k: v.copy() for k, v in file.items()}
                mask, advantages = kw['response_mask'], data.batch['advantages']
                _, grad = policy_gradient(data)
                self.assertEqual(data.meta_info['ccpo_diag']['ccpo/progress_task_grouping'], 1)
                self.assertEqual(data.meta_info['ccpo_diag']['ccpo/progress_episode_weight'], int(benchmark == 'webshop'))
                self.assertTrue(torch.isfinite(grad).all()); self.assertGreater(float(grad.abs().sum()), 0)
                self.assertFalse(advantages.requires_grad)
                np.testing.assert_array_equal(advantages[mask == 0].numpy(), 0)
                np.testing.assert_allclose(a['actor_applied'],
                                           a['history_applied'] + a['future_applied'] + a['episode_applied'], atol=3e-5)
                if benchmark == 'alfworld':
                    np.testing.assert_array_equal(a['episode_applied'], 0)
                    for task in np.unique(a['uid']):
                        values = a['actor_applied'][a['uid'] == task]
                        self.assertAlmostEqual(values.mean(), 0., places=5)
                        self.assertAlmostEqual(values.std(ddof=1), 1., places=5)


if __name__ == '__main__':
    unittest.main()
