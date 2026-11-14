#!/usr/bin/env python3
"""Offline credit-channel analysis (scripts/analyse_credit_channels.py).

1  the statistics against hand-computed values and an independent ANOVA
2  the alternative future signals against hand-computed values, and the check that
   stops the analysis when the logged credit does not follow from the logged values
3  logs written by the real estimator and log writer: the reader returns what was
   logged, the statistics match independent computations, and the command line
   writes its report

    .venv/bin/python tests/test_credit_channel_analysis.py

Needs torch (the analysis imports the training code's future-credit functions).
"""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'tests'), str(ROOT/'scripts')]

try:
    import torch
    from ccpo import core_ccpo as core
    from ccpo import future_progress as fp
    import analyse_credit_channels as acc
    from test_rebuttal_task_gate_ccpo import gate
except ImportError:
    torch = None


def hand_log(gamma=.9):
    """One task, a successful 3-turn rollout and a failed 2-turn one, with chosen values.
    Returns the log of a horizon-2 `difference` run and the trajectory groups."""
    values = np.array([2., 4., 6., 2., 1.])
    rewards = np.array([10., 10., 10., 0., 0.])
    log = dict(uid=np.array(['q'] * 5), traj_uid=np.array(['a', 'a', 'a', 'b', 'b']),
               turn_index=np.array([0., 1., 2., 0., 1.]), episode_lengths=np.array([3., 3., 3., 2., 2.]),
               episode_rewards=rewards, current_value=values)
    groups = acc.trajectory_groups(log)
    raw, _, endpoint, window, eligible = fp.progress_from_values(values, groups, rewards, 2, 10., gamma=gamma)
    log.update(raw_progress=raw, window_length=window.astype(float), eligible=eligible, terminal=endpoint < 0,
               progress_normalized=fp.standardize_progress(raw, log['uid'], eligible)[0],
               history_live=np.ones(5, dtype=bool), history_adv=np.array([1., 2., 3., -1., -2.]),
               history_applied=np.array([1., 2., 3., -1., -2.]), future_applied=raw.copy())
    return log, groups


@unittest.skipIf(torch is None, 'needs torch')
class StatisticTests(unittest.TestCase):
    def test_ranks_average_ties(self):
        np.testing.assert_allclose(acc.ranks([3., 1., 3., 2.]), [3.5, 1., 3.5, 2.])

    def test_spearman_with_ties(self):
        # ranks x = [1, 2.5, 2.5, 4], y = [1, 3, 2, 4]: 4.5 / sqrt(4.5 * 5)
        self.assertAlmostEqual(acc.spearman([1., 2., 2., 3.], [1., 3., 2., 4.]), 4.5 / np.sqrt(22.5), places=12)

    def test_correlation_is_undefined_without_spread_or_rows(self):
        self.assertTrue(np.isnan(acc.pearson([1., 1., 1.], [1., 2., 3.])))
        self.assertTrue(np.isnan(acc.pearson([1., 2.], [2., 1.])))
        self.assertTrue(np.isnan(acc.spearman([], [])))

    def test_sign_disagreement_ignores_zeros(self):
        self.assertAlmostEqual(acc.sign_disagreement([1., -1., 0., 2.], [1., 1., 5., -3.]), 2 / 3)
        self.assertTrue(np.isnan(acc.sign_disagreement([0., 0.], [1., 1.])))

    def test_variance_share_extremes_and_chance(self):
        tasks = np.array(['q'] * 4); labels = np.array(['a', 'a', 'b', 'b']); mask = np.ones(4, dtype=bool)
        share, chance, used = acc.variance_share(np.array([0., 0., 1., 1.]), tasks, labels, mask)
        self.assertEqual((share, used), (1., 1)); self.assertAlmostEqual(chance, 1 / 3)
        self.assertEqual(acc.variance_share(np.array([0., 1., 0., 1.]), tasks, labels, mask)[0], 0.)

    def test_variance_share_matches_one_way_anova(self):
        rng = np.random.default_rng(3)
        tasks = np.repeat(['p', 'q', 'r'], [9, 12, 7])
        labels = np.array([f'{t}{g}' for t, g in zip(tasks, rng.integers(3, size=len(tasks)))])
        x = rng.normal(size=len(tasks)); mask = rng.random(len(tasks)) > .15
        expected, chances = [], []
        for task in 'pqr':
            rows = [i for i in range(len(tasks)) if mask[i] and tasks[i] == task]
            grand = np.mean(x[rows])
            between = sum(sum(labels[i] == g for i in rows) * (np.mean([x[i] for i in rows if labels[i] == g]) - grand) ** 2
                          for g in sorted({labels[i] for i in rows}))
            expected.append(between / sum((x[i] - grand) ** 2 for i in rows))
            chances.append((len({labels[i] for i in rows}) - 1) / (len(rows) - 1))
        share, chance, used = acc.variance_share(x, tasks, labels, mask)
        self.assertEqual(used, 3)
        self.assertAlmostEqual(share, np.mean(expected), places=12)
        self.assertAlmostEqual(chance, np.mean(chances), places=12)

    def test_variance_share_skips_uninformative_tasks(self):
        mask = np.ones(3, dtype=bool); tasks = np.array(['q'] * 3)
        for labels, x in [(['a', 'a', 'a'], [1., 2., 3.]),      # one group
                          (['a', 'b', 'c'], [1., 2., 3.]),      # every group a single row
                          (['a', 'a', 'b'], [1., 1., 1.])]:     # no spread
            share, chance, used = acc.variance_share(np.array(x), tasks, np.array(labels), mask)
            self.assertTrue(np.isnan(share) and np.isnan(chance)); self.assertEqual(used, 0)

    def test_outcome_signs_use_tasks_with_both_outcomes(self):
        tasks = np.array(['p'] * 4 + ['q'] * 2)
        success = np.array([True, True, False, False, True, True])      # q has one outcome only
        x = np.array([1., -1., 2., -3., -5., -5.])
        positive_in_failed, negative_in_successful = acc.outcome_signs(x, tasks, success, np.ones(6, dtype=bool))
        self.assertEqual((positive_in_failed, negative_in_successful), (.5, .5))
        self.assertTrue(all(np.isnan(v) for v in acc.outcome_signs(x, tasks, success, tasks == 'q')))

    def test_outcome_driven_and_local_channels_are_told_apart(self):
        # H is the trajectory's outcome and nothing else; F is noise unrelated to it.
        rng = np.random.default_rng(0)
        tasks = np.repeat([f'q{i}' for i in range(4)], 240)
        trajectories = np.array([f'{t}-{i // 30 % 8}' for i, t in enumerate(tasks)])
        success = np.array([int(t.split('-')[1]) % 2 == 0 for t in trajectories])
        h = np.where(success, 1., -1.); f = rng.normal(size=len(tasks))
        everything = np.ones(len(tasks), dtype=bool)
        s = acc.share_statistics(h, f, tasks, trajectories, success, everything)
        self.assertAlmostEqual(s['trajectory_h'], 1.); self.assertAlmostEqual(s['outcome_h'], 1.)
        self.assertAlmostEqual(s['trajectory_chance_h'], 7 / 239); self.assertAlmostEqual(s['outcome_chance_h'], 1 / 239)
        self.assertEqual((s['trajectory_chance_f'], s['outcome_chance_f']), (s['trajectory_chance_h'], s['outcome_chance_h']))
        self.assertLess(s['trajectory_f'], .1); self.assertLess(s['outcome_f'], .03)
        # Shares ignore task-wise centering and scaling, so raw and applied credit agree.
        scale = np.array([{'q0': 1., 'q1': 3., 'q2': .5, 'q3': 7.}[t] for t in tasks])
        moved = acc.share_statistics(scale * h + scale, scale * f - 2 * scale, tasks, trajectories, success, everything)
        for key in s:
            self.assertAlmostEqual(moved[key], s[key], places=10, msg=key)
        r = acc.relation_statistics(h, f, tasks, success, everything)
        self.assertEqual((r['positive_in_failed_h'], r['negative_in_successful_h']), (0., 0.))
        self.assertAlmostEqual(r['positive_in_failed_f'], .5, delta=.08)
        self.assertAlmostEqual(r['sign_disagreement'], .5, delta=.08)
        self.assertLess(abs(r['pearson']), .1); self.assertEqual(r['rows'], 960.)


@unittest.skipIf(torch is None, 'needs torch')
class SignalTests(unittest.TestCase):
    def test_alternative_signals_by_hand(self):
        log, groups = hand_log()
        signals = acc.alternative_signals(log, groups, .9, 2, 10.)
        expected = {
            'one_step_difference': [4 - 2, 6 - 4, 10 - 6, 1 - 2, 0 - 1],
            'one_step_td_residual': [.9 * 4 - 2, .9 * 6 - 4, .9 * 10 - 6, .9 * 1 - 2, 0 - 1],
            # windows are 2, 2, 1 and 2, 1: the endpoint is the terminal value where t + 2 >= T
            'k_step_td_residual': [.81 * 6 - 2, .81 * 10 - 4, .9 * 10 - 6, 0 - 2, 0 - 1],
            'discounted_final_reward': [.9 ** 3 * 10, .9 ** 2 * 10, .9 * 10, 0, 0],
            'episode_outcome': [10, 10, 10, 0, 0],
        }
        self.assertEqual(set(signals), set(expected))
        for name, values in expected.items():
            raw, eligible = signals[name]
            self.assertTrue(eligible.all(), name)
            np.testing.assert_allclose(raw, values, atol=1e-12, err_msg=name)
        np.testing.assert_allclose(log['raw_progress'], [6 - 2, 10 - 4, 10 - 6, 0 - 2, 0 - 1])

    def test_logged_credit_must_follow_from_logged_values(self):
        log, groups = hand_log()
        acc.verify_reconstruction(log, groups, .9, 2, 'difference', 10.)
        acc.verify_reconstruction(log, groups, .5, 2, 'difference', 10.)     # this mode has no discount
        for kwargs, message in [(dict(horizon=1), 'window lengths'), (dict(mode='discount_aligned'), 'raw future credit'),
                                (dict(success_reward=5.), 'raw future credit')]:
            call = dict(gamma=.9, horizon=2, mode='difference', success_reward=10.); call.update(kwargs)
            with self.subTest(**kwargs), self.assertRaisesRegex(ValueError, message):
                acc.verify_reconstruction(log, groups, **call)
        tampered = dict(log, current_value=log['current_value'] + np.array([0, .01, 0, 0, 0]))
        with self.assertRaisesRegex(ValueError, 'raw future credit'):
            acc.verify_reconstruction(tampered, groups, .9, 2, 'difference', 10.)
        with self.assertRaisesRegex(ValueError, 'eligible rows'):
            acc.verify_reconstruction(dict(log, eligible=~log['eligible']), groups, .9, 2, 'difference', 10.)

    def test_incomplete_trajectories_are_rejected(self):
        log, _ = hand_log()
        with self.assertRaisesRegex(ValueError, 'Incomplete trajectory'):
            acc.trajectory_groups({k: v[[0, 2, 3, 4]] for k, v in log.items()})
        with self.assertRaisesRegex(ValueError, 'Incomplete trajectory'):
            acc.trajectory_groups(dict(log, episode_lengths=np.array([4., 4., 4., 2., 2.])))

    def test_step_record_on_the_hand_log(self):
        log, _ = hand_log()
        record = acc.analyse_step(log, .9, 2, 'difference', 10.)
        self.assertEqual((record['rows_both_channels'], record['rows_without_future_signal']), (5., 0.))
        self.assertAlmostEqual(record['terminal_window_frac'], 4 / 5)
        self.assertEqual(record['raw/all/rows'], 5.); self.assertEqual(record['raw/nonterminal/rows'], 1.)
        # H = [1, 2, 3, -1, -2], F = [4, 6, 4, -2, -1]: same signs everywhere
        self.assertEqual(record['raw/all/sign_disagreement'], 0.)
        self.assertAlmostEqual(record['raw/all/pearson'],
                               np.corrcoef([1, 2, 3, -1, -2], [4, 6, 4, -2, -1])[0, 1], places=12)
        self.assertEqual((record['raw/all/positive_in_failed_f'], record['raw/all/negative_in_successful_f']), (0., 0.))
        # With gamma = 1 the TD residual is the difference itself.
        log1, _ = hand_log(gamma=1.)
        same = acc.analyse_step(log1, 1., 2, 'difference', 10.)
        self.assertAlmostEqual(same['alternative/k_step_td_residual/all/pearson'], 1., places=12)
        self.assertEqual(same['alternative/k_step_td_residual/all/sign_agreement'], 1.)
        self.assertLess(record['alternative/k_step_td_residual/all/pearson'], 1.)

    def test_tasks_without_future_signal_are_left_out(self):
        log, _ = hand_log()
        flat = {k: np.concatenate([v, v]) for k, v in log.items()}
        flat['uid'] = np.array(['q'] * 5 + ['dead'] * 5)
        dead = flat['uid'] == 'dead'
        # Every rollout of the second task failed: values, and with them future credit, are zero.
        for key in ('episode_rewards', 'current_value', 'raw_progress', 'progress_normalized', 'future_applied'):
            flat[key] = np.where(dead, 0., flat[key])
        record = acc.analyse_step(flat, .9, 2, 'difference', 10.)
        self.assertEqual((record['rows_both_channels'], record['rows_without_future_signal']), (10., 5.))
        alone = acc.analyse_step(log, .9, 2, 'difference', 10.)
        for key in alone:
            if key.startswith(('raw/', 'applied/', 'shares/', 'alternative/')):
                np.testing.assert_equal(record[key], alone[key], err_msg=key)


def rollouts(seed, tasks=3, per_task=4, gamma=.95):
    """Estimator inputs: `tasks` x `per_task` rollouts of 3-6 turns, both outcomes in every task."""
    rng = np.random.default_rng(seed)
    obs, uid, traj, turn, lens, episodes, returns = [], [], [], [], [], [], []
    for q in range(tasks):
        for r in range(per_task):
            length = int(rng.integers(3, 7)); outcome = 10. if (q + r + seed) % 2 == 0 else 0.
            for t in range(length):
                obs.append(f'room{int(rng.integers(3))}' if t else 'start')
                uid.append(f'task{q}'); traj.append(f'task{q}-traj{r}'); turn.append(t); lens.append(length)
                episodes.append(outcome); returns.append(gamma ** (length - 1 - t) * outcome)
    n = len(obs)
    return dict(step_rewards=torch.tensor(returns, dtype=torch.float32), response_mask=torch.ones(n, 4),
                anchor_obs=np.array(obs, dtype=object), index=np.array(uid), traj_index=np.array(traj),
                turn_index=np.array(turn), episode_lengths=np.array(lens),
                episode_rewards=np.array(episodes), is_action_valid=np.ones(n, dtype=bool),
                phi_feats=torch.tensor(rng.normal(size=(n, 16)), dtype=torch.float32),
                phi=core.FrozenPhi(), gamma=gamma, return_diag=True)


def write_step(run, step, kw, horizon=2, mode='difference'):
    """Run the estimator and the trainer's task-wise normalization, and let the real
    writer log the step under `run`. Returns the arrays it logged."""
    with gate('hard'), patch.dict(os.environ, {'ACG_EXP_DIR': str(run)}):
        advantage, diag = fp.ccpo_future_progress_advantage(**kw, horizon=horizon, progress_mode=mode)
        uid = np.asarray(kw['index']).astype(str); live = np.asarray(diag['live_mask'])
        for task in np.unique(uid):
            ids = np.flatnonzero((uid == task) & live)
            if len(ids) > 1:
                ix = torch.as_tensor(ids); v = advantage[ix]
                advantage[ix] = (v - v.mean()) / (v.std() + 1e-6)
        fp.finalize_progress_logging(diag, advantage, uid, True, step, 0., 1.)
    return diag['progress_payload']['arrays']


def new_run(directory, horizon=2, mode=None, gamma=.95):
    run = Path(directory) / 'run'
    (run / 'outputs').mkdir(parents=True)
    (run / 'config.json').write_text(json.dumps(
        {'config': {'gamma': gamma, 'ccpo_progress_horizon': horizon, 'ccpo_progress_mode': mode}}))
    return run


@unittest.skipIf(torch is None, 'needs torch')
class RealLogTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)

    def test_reader_returns_what_the_estimator_logged(self):
        run = new_run(self.tmp.name)
        arrays = write_step(run, 7, rollouts(1))
        log = acc.read_step(run / 'outputs/future_progress/step-0007.csv')
        self.assertEqual(len(log['uid']), len(arrays['uid']))
        for key in acc.TEXT + acc.BOOLEAN:
            np.testing.assert_array_equal(log[key], arrays[key], err_msg=key)
        for key in acc.NUMERIC:
            np.testing.assert_allclose(log[key], np.asarray(arrays[key], dtype=float), rtol=1e-6, atol=1e-6, err_msg=key)
        self.assertTrue(log['eligible'].any() and log['terminal'].any() and (~log['terminal']).any())

    def test_statistics_match_independent_computations(self):
        run = new_run(self.tmp.name)
        arrays = write_step(run, 1, rollouts(2))
        log = acc.read_step(run / 'outputs/future_progress/step-0001.csv')
        record = acc.analyse_step(log, .95, 2, 'difference', 10.)
        self.assertEqual(record['rows_without_future_signal'], 0.)
        both = arrays['eligible'] & arrays['history_live']
        self.assertEqual(record['applied/all/rows'], float(both.sum()))
        self.assertEqual(record['applied/nonterminal/rows'], float((both & ~arrays['terminal']).sum()))
        for pair, h, f in (('applied', 'history_applied', 'future_applied'), ('raw', 'history_adv', 'raw_progress')):
            for population, mask in (('all', both), ('nonterminal', both & ~arrays['terminal'])):
                x, y = np.asarray(arrays[h], dtype=float)[mask], np.asarray(arrays[f], dtype=float)[mask]
                self.assertAlmostEqual(record[f'{pair}/{population}/pearson'], np.corrcoef(x, y)[0, 1], places=5)
                # Opposite signs among turns where neither channel is exactly zero.
                self.assertAlmostEqual(record[f'{pair}/{population}/sign_disagreement'],
                                       np.sum(x * y < 0) / np.sum(x * y != 0), places=12)
        # The logged applied components add up to the logged step advantage.
        np.testing.assert_allclose(log['history_applied'] + log['future_applied'],
                                   np.asarray(arrays['combined_applied'], dtype=float), atol=1e-5)
        # A failed rollout's raw H is never positive when its return is zero and peers' are not negative.
        failed = both & (arrays['episode_rewards'] == 0)
        self.assertEqual(record['raw/all/positive_in_failed_h'], float(np.mean(arrays['history_adv'][failed] > 1e-12)))

    def test_reference_reconstruction_on_both_discount_modes(self):
        for mode in ('difference', 'discount_aligned'):
            with self.subTest(mode=mode):
                run = new_run(self.tmp.name + '/' + mode, mode=mode)
                write_step(run, 1, rollouts(3), mode=mode)
                log = acc.read_step(run / 'outputs/future_progress/step-0001.csv')
                groups = acc.trajectory_groups(log)
                acc.verify_reconstruction(log, groups, .95, 2, mode, 10.)
                raw, _, _, _, eligible = fp.progress_from_values(
                    log['current_value'], groups, log['episode_rewards'], 2, 10., gamma=.95, mode=mode)
                np.testing.assert_allclose(fp.standardize_progress(raw, log['uid'], eligible)[0],
                                           log['progress_normalized'], atol=1e-4)
                other = 'discount_aligned' if mode == 'difference' else 'difference'
                with self.assertRaisesRegex(ValueError, 'raw future credit'):
                    acc.verify_reconstruction(log, groups, .95, 2, other, 10.)
                with self.assertRaisesRegex(ValueError, 'window lengths'):
                    acc.verify_reconstruction(log, groups, .95, 1, mode, 10.)
                record = acc.analyse_step(log, .95, 2, mode, 10.)
                own = record['alternative/k_step_td_residual/all/pearson']
                if mode == 'discount_aligned':      # the run's own signal
                    self.assertAlmostEqual(own, 1., places=6)
                    self.assertEqual(record['alternative/k_step_td_residual/all/sign_agreement'], 1.)
                else:
                    self.assertLess(own, 1. - 1e-6)

    def test_command_line_report(self):
        run = new_run(self.tmp.name)
        for step in (1, 2, 40):
            write_step(run, step, rollouts(10 + step))
        with patch('builtins.print'):
            acc.main([str(run)])
        out = run / 'outputs/credit_channels'
        result = json.loads((out / 'credit_channels.json').read_text())
        self.assertEqual(result['meta']['steps_analysed'], 3)
        self.assertEqual((result['meta']['gamma'], result['meta']['horizon'], result['meta']['mode']), (.95, 2, 'difference'))
        self.assertEqual(sorted(result['per_step']), ['1', '2', '40'])
        self.assertEqual(sorted(result['summary']), ['1-30', '1-40', '31-120'])     # no logs in 121-150
        early = result['summary']['1-30']['applied/all/pearson']
        self.assertEqual(early['steps'], 2)
        self.assertAlmostEqual(early['mean'], np.mean([result['per_step'][s]['applied/all/pearson'] for s in '12']))
        self.assertEqual(result['summary']['1-40']['applied/all/pearson']['steps'], 3)
        report = (out / 'credit_channels.md').read_text()
        for text in ('## A. Historical credit H versus future credit F', '## B.', '`one_step_td_residual`',
                     '| 1-30 (n=2) | applied |', '| 31-120 (n=1) | raw |', '| 1-40 (n=3) | `episode_outcome` |',
                     'trajectory share, H', 'not which one trains a better policy'):
            self.assertIn(text, report)
        self.assertNotIn('121-150', report)
        # The shares are reported once per population and match the raw-credit computation.
        log = acc.read_step(run / 'outputs/future_progress/step-0040.csv')
        mask = log['eligible'] & log['history_live']
        share, chance, _ = acc.variance_share(log['history_adv'], log['uid'], log['traj_uid'], mask)
        late = result['summary']['31-120']
        self.assertAlmostEqual(late['shares/all/trajectory_h']['mean'], share, places=12)
        self.assertAlmostEqual(late['shares/all/trajectory_chance_h']['mean'], chance, places=12)
        applied, _, _ = acc.variance_share(log['history_applied'], log['uid'], log['traj_uid'], mask)
        self.assertAlmostEqual(applied, share, places=5)
        rows = (out / 'credit_channels_per_step.csv').read_text().splitlines()
        self.assertEqual(len(rows), 4); self.assertTrue(rows[0].startswith('step,'))
        # A restricted run and an explicit output directory.
        with patch('builtins.print'):
            acc.main([str(run), '--steps', '1-2', '--phases', '1-2', '--output', str(Path(self.tmp.name) / 'o')])
        restricted = json.loads((Path(self.tmp.name) / 'o/credit_channels.json').read_text())
        self.assertEqual(sorted(restricted['per_step']), ['1', '2']); self.assertEqual(sorted(restricted['summary']), ['1-2'])

    def test_wrong_settings_and_missing_logs_stop_the_analysis(self):
        run = new_run(self.tmp.name)
        write_step(run, 1, rollouts(4))
        with self.assertRaisesRegex(ValueError, 'window lengths'):
            acc.analyse_run(run, horizon=1)
        with self.assertRaisesRegex(ValueError, 'raw future credit'):
            acc.analyse_run(run, mode='discount_aligned')
        with self.assertRaisesRegex(FileNotFoundError, 'No step-NNNN.csv'):
            acc.analyse_run(run, steps='50-60')
        with self.assertRaisesRegex(ValueError, 'Invalid step range'):
            acc.analyse_run(run, steps='5')
        # A bare log directory has no config: the settings must be given.
        logs = run / 'outputs/future_progress'
        with self.assertRaisesRegex(ValueError, 'pass --gamma and --horizon'):
            acc.analyse_run(logs)
        meta, per_step, _ = acc.analyse_run(logs, gamma=.95, horizon=2)
        self.assertEqual((meta['mode'], sorted(per_step)), ('difference', [1]))
        with self.assertRaisesRegex(ValueError, 'no future credit'):
            acc.analyse_run(logs, gamma=.95, horizon=0)
        header = (logs / 'step-0001.csv').read_text().splitlines()
        (logs / 'step-0002.csv').write_text('\n'.join(line.split(',', 1)[1] for line in header) + '\n')
        with self.assertRaisesRegex(ValueError, "missing columns \\['uid'\\]"):
            acc.analyse_run(run)


if __name__ == '__main__':
    unittest.main()
