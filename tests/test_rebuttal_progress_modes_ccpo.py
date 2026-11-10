"""Rebuttal future-channel ablations (E2, E5): terminal-window splits and clock-only controls.

Math and config tests need numpy only (plus importable ccpo modules). The estimator and
trainer tests need the training environment: torch and the synced verl-agent overlay.

    .venv/bin/python -m unittest discover -s tests -p test_rebuttal_progress_modes_ccpo.py
"""
import ast
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'verl-agent'), str(ROOT/'tests'), str(ROOT/'scripts')]
from ccpo.future_progress import PROGRESS_MODES, progress_from_values, standardize_progress
import exp_run as er
import prepare_rebuttal_progress_ablations_ccpo as prep

NEW = ('terminal_difference', 'terminal_aligned', 'remaining_turns', 'terminal_proximity')
VALUE_MODES = ('difference', 'discount_aligned', 'terminal_difference', 'terminal_aligned')
GAMMA = .95


def toy():
    """Two successes (lengths 5, 3), one failure (length 4), one partial WebShop score (length 2)."""
    groups = [np.arange(5), np.arange(5, 8), np.arange(8, 12), np.arange(12, 14)]
    values = np.array([1., 7., 4., 3., 6., 9., 2., 5., 3., 1., 4., 2., 6., 8.])
    rewards = np.r_[np.full(8, 10.), np.zeros(4), np.full(2, 7.5)]
    return values, groups, rewards


def turns_left(groups, n):
    out = np.zeros(n)
    for ids in groups:
        out[ids] = len(ids) - np.arange(len(ids))
    return out


class ModeMathTests(unittest.TestCase):
    def test_mode_lists_agree_across_estimator_launcher_and_trainer(self):
        self.assertEqual(PROGRESS_MODES[:2], ('difference', 'discount_aligned'))
        self.assertEqual(set(PROGRESS_MODES[2:]), set(NEW))
        self.assertEqual(er.PROGRESS_MODES, PROGRESS_MODES)
        source = (ROOT/'patches/verl-agent/verl/trainer/ppo/ray_trainer.py').read_text()
        tuples = [node for node in ast.walk(ast.parse(source)) if isinstance(node, ast.Compare)
                  and isinstance(node.left, ast.Name) and node.left.id == '_progress_mode'
                  and isinstance(node.ops[0], ast.NotIn)]
        self.assertEqual(len(tuples), 1)
        self.assertEqual(tuple(ast.literal_eval(tuples[0].comparators[0])), PROGRESS_MODES)

    def test_terminal_splits_take_each_form_on_the_right_windows(self):
        values, groups, rewards = toy()
        for gamma in (0., .5, GAMMA, 1.):
            for horizon in range(1, 5):
                with self.subTest(gamma=gamma, horizon=horizon):
                    out = {m: progress_from_values(values, groups, rewards, horizon, gamma=gamma, mode=m)
                           for m in VALUE_MODES}
                    _, future, endpoint, window, eligible = out['difference']
                    terminal = endpoint < 0
                    for m in VALUE_MODES:
                        np.testing.assert_array_equal(out[m][1], future)
                        np.testing.assert_array_equal(out[m][4], eligible)
                    d, a = out['difference'][0], out['discount_aligned'][0]
                    np.testing.assert_array_equal(out['terminal_difference'][0], np.where(terminal, d, a))
                    np.testing.assert_array_equal(out['terminal_aligned'][0], np.where(terminal, a, d))
                    np.testing.assert_allclose(out['terminal_difference'][0] + out['terminal_aligned'][0], d + a,
                                               atol=1e-14)
                    # A terminal window is exactly the last `horizon` turns of each trajectory.
                    np.testing.assert_array_equal(terminal, turns_left(groups, len(values)) <= horizon)
                    if gamma == 1:
                        for m in VALUE_MODES:
                            np.testing.assert_array_equal(out[m][0], d)

    def test_solved_equal_length_task_isolates_the_terminal_time_term(self):
        # Exact discount-consistent potentials V_t = gamma^(T-t) * R: the discount-aligned
        # residual is zero, so each split keeps the time term only where it uses Eq. 7.
        groups = [np.arange(6), np.arange(6, 12)]
        values = np.tile(GAMMA**np.arange(6, 0, -1)*10, 2)
        rewards = np.full(12, 10.)
        td = progress_from_values(values, groups, rewards, 2, gamma=GAMMA, mode='terminal_difference')[0]
        ta = progress_from_values(values, groups, rewards, 2, gamma=GAMMA, mode='terminal_aligned')[0]
        left = turns_left(groups, 12)
        np.testing.assert_allclose(td[left > 2], 0, atol=2e-15)
        np.testing.assert_allclose(td[left == 2], (1-GAMMA**2)*10, atol=2e-15)   # 0.975
        np.testing.assert_allclose(td[left == 1], (1-GAMMA)*10, atol=2e-15)      # 0.5
        np.testing.assert_allclose(ta[left <= 2], 0, atol=2e-15)
        # Non-terminal windows keep Eq. 7: V_{t+2} - V_t = V_t / gamma^2 - V_t (the 0.0975 * V_u ramp).
        np.testing.assert_allclose(ta[left > 2], values[left > 2]/GAMMA**2 - values[left > 2], atol=1e-12)
        # The split is no longer degenerate in such a task, unlike discount_aligned.
        self.assertGreater(np.std(td, ddof=1), .1); self.assertGreater(np.std(ta, ddof=1), .1)

    def test_remaining_turns_ignores_values_and_outcome(self):
        values, groups, rewards = toy()
        ref = progress_from_values(values, groups, rewards, 2)
        raw, future, endpoint, window, eligible = progress_from_values(
            values, groups, rewards, 2, gamma=GAMMA, mode='remaining_turns')
        np.testing.assert_array_equal(raw, -turns_left(groups, len(values)))
        for got, want in zip((future, endpoint, window, eligible), ref[1:]):
            np.testing.assert_array_equal(got, want)
        other = progress_from_values(values[::-1].copy()*3, groups, rewards[::-1].copy(), 2, gamma=.5,
                                     mode='remaining_turns')[0]
        np.testing.assert_array_equal(other, raw)

    def test_terminal_proximity_is_the_discounted_success_label(self):
        values, groups, rewards = toy()
        raw, _, _, _, eligible = progress_from_values(values, groups, rewards, 2, gamma=GAMMA,
                                                      mode='terminal_proximity')
        left = turns_left(groups, len(values))
        expected = np.where(rewards == 10., GAMMA**left*10, 0.)
        np.testing.assert_allclose(raw, expected, rtol=0, atol=0)
        self.assertEqual(raw[13], 0.)               # a partial score is not a success
        np.testing.assert_allclose(raw[4], GAMMA*10)  # last turn of a success: T - t = 1
        np.testing.assert_array_equal(
            progress_from_values(values*0+123., groups, rewards, 2, gamma=GAMMA, mode='terminal_proximity')[0], raw)
        # Failure-only tasks standardize to exact zeros, not to amplified roundoff.
        z = standardize_progress(raw, np.r_[['a']*8, ['b']*4, ['c']*2], eligible)[0]
        np.testing.assert_array_equal(z[8:], 0.)

    def test_eligibility_and_history_only_are_unchanged(self):
        values = np.array([np.nan, 3., 4., 5.]); groups = [np.arange(4)]; rewards = np.full(4, 10.)
        base = progress_from_values(values, groups, rewards, 2)[4]
        for m in NEW:
            with self.subTest(mode=m):
                raw, _, _, window, eligible = progress_from_values(values, groups, rewards, 2, mode=m)
                np.testing.assert_array_equal(eligible, base); self.assertEqual(raw[0], 0)
                raw, _, _, window, eligible = progress_from_values(values, groups, rewards, 0, mode=m)
                np.testing.assert_array_equal(raw, 0); self.assertFalse(eligible.any())

    def test_launcher_validation(self):
        for m in NEW:
            er.validate_future_progress_config({'ccpo_progress_mode': m, 'ccpo_progress_horizon': 2})
            with self.assertRaises(ValueError):
                er.validate_future_progress_config({'ccpo_progress_mode': m, 'ccpo_progress_horizon': 0})
        with self.assertRaises(ValueError):
            er.validate_future_progress_config({'ccpo_progress_mode': 'typo', 'ccpo_progress_horizon': 2})
        with self.assertRaises(ValueError):
            progress_from_values(np.ones(2), [np.arange(2)], np.ones(2), 2, mode='typo')

    def test_registry_entries_match_prepare_script(self):
        for arm, spec in prep.ARMS.items():
            for benchmark in spec['benchmarks']:
                _, cfg = prep.ablations.build(prep.key_for(arm, benchmark), benchmark)
                self.assertEqual(cfg['ccpo_progress_mode'], spec['mode'])
                self.assertEqual(cfg['ccpo_ep_w'], float(benchmark == 'webshop'))
                er.validate_future_progress_config(cfg)
        self.assertEqual({s['mode'] for s in prep.ARMS.values()}, set(NEW))


def _training_env():
    try:
        import torch  # noqa: F401
        return (ROOT/'verl-agent/verl/trainer/ppo/ray_trainer.py').exists()
    except ImportError:
        return False


@unittest.skipUnless(_training_env(), 'needs torch and the synced verl-agent overlay')
class ModeEstimatorTests(unittest.TestCase):
    NO_SHRINK = dict(_LK_FIX='1.0', _LAM_FIX='1.0', _LOO='1', _GATE='hard',
                     _WMODE='soft', _PHI_MODE='hidden+ctx', _CTX_W=1., _WHITEN_K=3,
                     _JW_C=0., _STD_MODE='task', _BACKOFF_TASK=True, _SIM=0.,
                     _SIM_BACKOFF=0., _EDGE_W=0., _TAU_ENV='.15')

    def setUp(self):
        from test_future_progress import core
        patches = patch.multiple(core, **self.NO_SHRINK); patches.start(); self.addCleanup(patches.stop)

    def test_only_the_future_raw_signal_changes(self):
        from test_future_progress import arguments, ccpo_future_progress_advantage
        kw = arguments()
        _, old = ccpo_future_progress_advantage(**kw, horizon=2)
        a = old['progress_payload']['arrays']
        for m in NEW:
            with self.subTest(mode=m):
                _, new = ccpo_future_progress_advantage(**kw, horizon=2, progress_mode=m)
                b = new['progress_payload']['arrays']
                for key in ('history_adv', 'history_baseline', 'value_target', 'current_value', 'future_value',
                            'eligible', 'live', 'undiscounted_raw', 'discount_aligned_raw', 'time_passage_raw'):
                    np.testing.assert_array_equal(a[key], b[key], err_msg=key)
                groups = [np.flatnonzero(b['traj_uid'] == u)[np.argsort(b['turn_index'][b['traj_uid'] == u])]
                          for u in dict.fromkeys(b['traj_uid'])]
                expected = progress_from_values(b['current_value'], groups, b['episode_rewards'], 2,
                                                gamma=kw['gamma'], mode=m)[0]
                np.testing.assert_allclose(b['raw_progress'], expected, atol=1e-12)
                if m in ('terminal_difference', 'terminal_aligned'):
                    pick = b['terminal'] if m == 'terminal_difference' else ~b['terminal']
                    np.testing.assert_allclose(b['raw_progress'][pick], b['undiscounted_raw'][pick], atol=1e-12)
                    np.testing.assert_allclose(b['raw_progress'][~pick], b['discount_aligned_raw'][~pick], atol=1e-12)
                for task in np.unique(b['uid']):
                    ids = (b['uid'] == task) & b['eligible']; raw = b['raw_progress'][ids]
                    np.testing.assert_allclose(b['progress_normalized'][ids], (raw-raw.mean())/(raw.std(ddof=1)+1e-6))
                np.testing.assert_array_equal(b['future_credit_mode'], m)
                self.assertEqual(new['progress_' + m], 1.); self.assertEqual(new['progress_discount_aligned'], 0.)

    def test_actual_trainer_routes_each_mode_with_ep0_and_ep1_controls(self):
        import torch
        from test_future_progress import load_compute_advantage, policy_gradient, core_gigpo
        from test_future_progress_active_episode import training_fixture, reward_data
        compute = load_compute_advantage(); kw = training_fixture(); mask = kw['response_mask']
        episode = torch.linspace(-2, 2, len(kw['index']))[:, None]*mask
        for benchmark, control in prep.CONTROLS.items():
            saved = json.loads((ROOT/'experiments'/control/'config.json').read_text())
            cfg, env = saved['config'], saved['env']
            for m in NEW:
                with self.subTest(benchmark=benchmark, mode=m), tempfile.TemporaryDirectory() as temp:
                    settings = dict(env, ACG_CCPO_PROGRESS_MODE=m, ACG_EXP_DIR=temp, ACG_CCPO_PROGRESS_SNAPSHOT_EVERY='1')
                    data = reward_data(kw)
                    with patch.dict(os.environ, settings), patch.object(core_gigpo, 'episode_norm_reward', return_value=episode), \
                            redirect_stdout(io.StringIO()):
                        compute(data, 'CCPO', gamma=cfg['gamma'], gigpo_mode=cfg['adv_mode'], ccpo_step_tag=1)
                    _, grad = policy_gradient(data)
                    with np.load(Path(temp)/'outputs/future_progress/step-0001.npz', allow_pickle=False) as snap:
                        a = {k: snap[k].copy() for k in snap.files}
                    np.testing.assert_array_equal(a['future_credit_mode'], m)
                    np.testing.assert_allclose(a['history_applied']+a['future_applied']+a['episode_applied'], a['actor_applied'], atol=3e-5)
                    np.testing.assert_allclose(a['actor_applied'][:, None]*mask.numpy(), data.batch['advantages'].numpy(), atol=3e-5)
                    self.assertTrue(torch.isfinite(grad).all()); self.assertGreater(float(grad.abs().sum()), 0)
                    torch.testing.assert_close(grad[mask == 0], torch.zeros_like(grad[mask == 0]))
                    self.assertEqual(data.meta_info['ccpo_diag']['ccpo/progress_' + m], 1)


class PreparedRebuttalTests(unittest.TestCase):
    """Checks prepared folders when they exist (they are machine-specific, so not committed)."""

    def test_prepared_configs_change_only_the_future_mode(self):
        found = 0
        for arm, spec in prep.ARMS.items():
            for benchmark in spec['benchmarks']:
                stem = prep.run_name(arm, benchmark, 'X').rsplit('-', 1)[0]
                for folder in sorted((ROOT/'experiments').glob(stem + '-*')):
                    found += 1
                    with self.subTest(folder=folder.name):
                        diff = json.loads((folder/'config-diff-from-control.json').read_text())
                        self.assertEqual(set(diff['changes']), {'exp_id', 'ccpo_progress_mode'})
                        self.assertEqual(diff['changes']['ccpo_progress_mode'],
                                         dict(control='difference', prepared=spec['mode']))
                        status = json.loads((folder/'PREPARED.json').read_text())
                        self.assertFalse(status['launched']); self.assertFalse(status['queued'])
        if not found:
            self.skipTest('no prepared rebuttal folders yet; run scripts/prepare_rebuttal_progress_ablations_ccpo.py')


if __name__ == '__main__':
    unittest.main()
